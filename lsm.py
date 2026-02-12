"""
lsm.py - liquid state machine (reservoir computing)
throw inputs in like stones, read the ripples. only readout layer is trained.
"""

# ---- backend config ----
import numpy as xp  # CPU
# import cupy as xp  # GPU (uncomment for RTX)

from typing import Optional, Tuple, List
import warnings
import math

# detect cupy
IS_CUPY_MODE = 'cupy' in str(type(xp.array([1])))
if IS_CUPY_MODE:
    print("gpu mode - numba jit disabled")

try:
    from numba import jit, prange
    HAS_NUMBA = True and not IS_CUPY_MODE
    if HAS_NUMBA:
        print("numba jit available")
except ImportError:
    HAS_NUMBA = False
    # dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    print("no numba - pip install numba for 10-100x speedup")


# ---- jit compiled core ----

@jit(nopython=True, cache=True, fastmath=True)
def _jit_step_core(u, state, W_in, W_res, leak_rate, pain_level):
    """native speed LSM step, bypasses GIL"""
    # pain noise
    if pain_level > 30:
        ni = (pain_level - 30) / 70
        for i in range(len(u)):
            u[i] += (2 * (hash((i, state[0])) % 1000) / 1000 - 1) * ni * 0.5

    pre = W_in @ u + W_res @ state
    for i in range(len(pre)):
        pre[i] = math.tanh(pre[i])

    return (1 - leak_rate) * state + leak_rate * pre


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def _jit_batch_matmul(inputs, W_in, W_res, states, leak_rate):
    """parallel batch processing across cpu cores"""
    n = inputs.shape[0]
    rsz = W_res.shape[0]
    result = states.copy()

    for t in prange(n):
        u = inputs[t]
        pre = W_in @ u + W_res @ result[t]
        for i in range(rsz):
            pre[i] = math.tanh(pre[i])
        result[t] = (1 - leak_rate) * result[t] + leak_rate * pre

    return result


# ---- main class ----

class LiquidStateMachine:
    """
    reservoir computing neural net.
    input -> reservoir (fixed sparse recurrent) -> readout (trained via ridge regression)
    """

    # vision class labels
    CLASSES = [
        "Login Screen",      # 0
        "Desktop (Idle)",    # 1
        "Browser",           # 2
        "Video/Media",       # 3
        "Empty/Black",       # 4
        "Coding IDE",        # 5
        "Terminal",          # 6
        "Split Screen",      # 7
        "Error/Popup",       # 8
        "Chat App",          # 9
    ]

    def __init__(self, input_size=256, reservoir_size=2000, output_size=10,
                 sparsity=0.1, spectral_radius=0.9, input_scaling=0.5,
                 leak_rate=0.3, seed=42):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate

        if seed is not None:
            xp.random.seed(seed)

        self._init_input_weights()
        self._init_reservoir_weights()
        self._init_readout_weights()

        self.state = xp.zeros(reservoir_size)
        self._is_trained = False

        # plasticity flags
        self.plasticity_enabled = False
        self.hebbian_rate = 0.0001
        self.stdp_rate = 0.00005

        print(f"-- lsm loaded ({input_size}->{reservoir_size}->{output_size}) --")
        print(f"   sparsity:{sparsity*100:.0f}% sr:{spectral_radius} plasticity:{'Y' if self.plasticity_enabled else 'N'}")

    def _init_input_weights(self):
        """W_in: input -> reservoir (sparse, random, FIXED)"""
        W = xp.random.randn(self.reservoir_size, self.input_size)
        mask = xp.random.rand(self.reservoir_size, self.input_size) < self.sparsity
        self.W_in = W * mask * self.input_scaling
        print(f"   W_in: {self.reservoir_size}x{self.input_size}, {int(xp.sum(mask))} conns")

    def _init_reservoir_weights(self):
        """W_res: reservoir internal (sparse, random, FIXED). must normalize spectral radius < 1"""
        W = xp.random.randn(self.reservoir_size, self.reservoir_size)
        mask = xp.random.rand(self.reservoir_size, self.reservoir_size) < self.sparsity
        self.W_res = self._normalize_spectral_radius(W * mask, self.spectral_radius)
        actual = self._compute_spectral_radius(self.W_res)
        print(f"   W_res: {self.reservoir_size}x{self.reservoir_size}, sr={actual:.4f}")

    def _init_readout_weights(self):
        """W_out: reservoir -> output (TRAINABLE, starts at zero)"""
        self.W_out = xp.zeros((self.output_size, self.reservoir_size))
        print(f"   W_out: {self.output_size}x{self.reservoir_size} (trainable)")

    def _normalize_spectral_radius(self, W, target):
        cur = self._compute_spectral_radius(W)
        if cur < 1e-10:
            warnings.warn("spectral radius near zero, matrix too sparse?")
            return W
        return W * (target / cur)

    def _compute_spectral_radius(self, W):
        if self.reservoir_size <= 3000:
            try:
                if hasattr(xp, 'asnumpy'):  # cupy
                    import numpy as np
                    eigs = np.linalg.eigvals(xp.asnumpy(W))
                else:
                    eigs = xp.linalg.eigvals(W)
                return float(xp.max(xp.abs(eigs)))
            except:
                pass
        return self._power_iteration(W)

    def _power_iteration(self, W, n_iter=100):
        v = xp.random.randn(W.shape[0])
        v = v / xp.linalg.norm(v)
        for _ in range(n_iter):
            Wv = W @ v
            norm = xp.linalg.norm(Wv)
            if norm < 1e-10:
                return 0.0
            v = Wv / norm
        return float(xp.abs(v @ W @ v))

    def reset_state(self):
        self.state = xp.zeros(self.reservoir_size)

    def step(self, input_signal):
        """single timestep: x(t+1) = (1-a)*x(t) + a*tanh(W_in*u + W_res*x)"""
        u = xp.asarray(input_signal).flatten()
        if len(u) != self.input_size:
            raise ValueError(f"expected input {self.input_size}, got {len(u)}")

        # pain disrupts processing
        if hasattr(self, 'pain_level') and self.pain_level > 30:
            ni = (self.pain_level - 30) / 70
            u = u + xp.random.randn(self.input_size) * ni * 0.5
            if self.pain_level > 70:
                self.state += xp.random.randn(self.reservoir_size) * (ni * 0.3)

        pre = self.W_in @ u + self.W_res @ self.state
        act = xp.tanh(pre)

        prev = self.state.copy()
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * act

        if self.plasticity_enabled:
            self._apply_plasticity(prev, self.state)

        return self.state.copy()

    def step_fast(self, input_signal):
        """jit-accelerated step, falls back to regular step if no numba"""
        if not HAS_NUMBA:
            return self.step(input_signal)

        u = xp.asarray(input_signal).flatten().astype(xp.float64)
        if len(u) != self.input_size:
            raise ValueError(f"expected input {self.input_size}, got {len(u)}")

        pain = getattr(self, 'pain_level', 0.0)
        self.state = _jit_step_core(
            u.copy(), self.state.astype(xp.float64),
            self.W_in.astype(xp.float64), self.W_res.astype(xp.float64),
            self.leak_rate, pain)

        if self.plasticity_enabled:
            pass  # accumulate during sleep instead

        return self.state.copy()

    def set_pain_level(self, pain):
        self.pain_level = max(0, min(100, pain))

    # ---- plasticity ----

    def enable_plasticity(self, hebbian_rate=0.0001, stdp_rate=0.00005):
        """brain learns its own structure! batch mode: accumulate during day, apply during sleep"""
        self.plasticity_enabled = True
        self.hebbian_rate = hebbian_rate
        self.stdp_rate = stdp_rate
        self._plasticity_delta = xp.zeros_like(self.W_res)
        self._plasticity_samples = 0
        print(f"plasticity ON (hebb:{hebbian_rate} stdp:{stdp_rate})")

    def disable_plasticity(self):
        self.plasticity_enabled = False
        print("plasticity OFF")

    def apply_batch_plasticity(self):
        """apply accumulated plasticity changes - call during sleep!"""
        if not hasattr(self, '_plasticity_delta') or self._plasticity_samples == 0:
            print("no plasticity data accumulated")
            return

        print(f"applying batch plasticity ({self._plasticity_samples} samples)...")
        avg = self._plasticity_delta / self._plasticity_samples

        mask = self.W_res != 0
        self.W_res += avg * mask
        self.W_res = self._normalize_spectral_radius(self.W_res, self.spectral_radius)

        chg = float(xp.abs(avg).mean())
        print(f"   change: {chg:.6f}, sr: {self._compute_spectral_radius(self.W_res):.4f}")

        self._plasticity_delta = xp.zeros_like(self.W_res)
        self._plasticity_samples = 0

    def _apply_plasticity(self, prev_state, new_state):
        """accumulate hebbian + stdp deltas (applied in batch during sleep)"""
        if not hasattr(self, '_plasticity_delta'):
            self._plasticity_delta = xp.zeros_like(self.W_res)
            self._plasticity_samples = 0

        # hebbian: fire together wire together
        hebb = xp.outer(new_state, prev_state) * self.hebbian_rate

        # stdp: timing matters
        pre_up = prev_state > 0.5
        post_up = new_state > 0.5
        ltp = xp.outer(post_up, pre_up).astype(float)
        ltd = xp.outer(pre_up, post_up).astype(float) * 0.5
        stdp = (ltp - ltd) * self.stdp_rate

        self._plasticity_delta += hebb + stdp
        self._plasticity_samples += 1

    # ---- training / prediction ----

    def harvest_states(self, input_sequence, reset_before=True, warmup_steps=0):
        """run sequence through reservoir, collect all states for training"""
        input_sequence = xp.asarray(input_sequence)
        if input_sequence.ndim != 2:
            raise ValueError(f"expected 2D input, got {input_sequence.ndim}D")

        ts = input_sequence.shape[0]
        if reset_before:
            self.reset_state()

        states = []
        for t in range(ts):
            st = self.step(input_sequence[t])
            if t >= warmup_steps:
                states.append(st)
        return xp.array(states)

    def train_readout(self, states, targets, regularization=1e-6):
        """ridge regression on readout layer. ONLY part we train."""
        states = xp.asarray(states)
        targets = xp.asarray(targets)

        # one-hot if needed
        if targets.ndim == 1:
            n = len(targets)
            oh = xp.zeros((n, self.output_size))
            for i, t in enumerate(targets):
                oh[i, int(t)] = 1.0
            targets = oh

        n = states.shape[0]
        StS = states.T @ states + regularization * xp.eye(self.reservoir_size)
        StY = states.T @ targets

        try:
            self.W_out = xp.linalg.solve(StS, StY).T
        except:
            self.W_out = (xp.linalg.pinv(states) @ targets).T

        self._is_trained = True
        print(f"readout trained on {n} samples")

    def predict(self, input_signal):
        """single prediction -> (class_idx, probabilities)"""
        if not self._is_trained:
            raise RuntimeError("readout not trained!")

        state = self.step(input_signal)
        out = self.W_out @ state

        # softmax
        exp_out = xp.exp(out - xp.max(out))
        probs = exp_out / xp.sum(exp_out)
        return int(xp.argmax(probs)), probs

    def predict_sequence(self, input_sequence, reset_before=True):
        """predict for each timestep in sequence"""
        if not self._is_trained:
            raise RuntimeError("readout not trained!")

        input_sequence = xp.asarray(input_sequence)
        if reset_before:
            self.reset_state()

        preds, all_probs = [], []
        for t in range(input_sequence.shape[0]):
            cls, probs = self.predict(input_sequence[t])
            preds.append(cls)
            all_probs.append(probs)
        return xp.array(preds), xp.array(all_probs)

    def get_class_name(self, idx):
        return self.CLASSES[idx] if 0 <= idx < len(self.CLASSES) else f"Unknown ({idx})"

    # ---- neurogenesis / atrophy ----

    def resize(self, new_size):
        """grow brain by adding neurons. new conns are weak & random."""
        if new_size <= self.reservoir_size:
            print(f"new size must be > {self.reservoir_size}")
            return False

        old = self.reservoir_size
        growth = new_size - old
        print(f"\nneurogenesis: {old} -> {new_size} (+{growth})")

        # expand W_res
        nW = xp.zeros((new_size, new_size))
        nW[:old, :old] = self.W_res

        # new <-> old connections (weak)
        n2o = xp.random.randn(growth, old) * (xp.random.rand(growth, old) < self.sparsity) * 0.1
        o2n = xp.random.randn(old, growth) * (xp.random.rand(old, growth) < self.sparsity) * 0.1
        n2n = xp.random.randn(growth, growth) * (xp.random.rand(growth, growth) < self.sparsity) * 0.1
        nW[old:, :old] = n2o
        nW[:old, old:] = o2n
        nW[old:, old:] = n2n

        self.W_res = self._normalize_spectral_radius(nW, self.spectral_radius)

        # expand W_in
        nWin = xp.zeros((new_size, self.input_size))
        nWin[:old, :] = self.W_in
        new_in = xp.random.randn(growth, self.input_size)
        mask_in = xp.random.rand(growth, self.input_size) < self.sparsity
        nWin[old:, :] = new_in * mask_in * self.input_scaling
        self.W_in = nWin

        # expand W_out
        nWout = xp.zeros((self.output_size, new_size))
        nWout[:, :old] = self.W_out
        self.W_out = nWout

        # expand state
        ns = xp.zeros(new_size)
        ns[:old] = self.state
        self.state = ns

        self.reservoir_size = new_size
        sr = self._compute_spectral_radius(self.W_res)
        print(f"   {new_size}x{new_size} sr={sr:.4f} done")
        return True

    def atrophy(self, target_size, min_size=1000):
        """shrink brain by removing least active neurons"""
        if target_size >= self.reservoir_size:
            print(f"target must be < {self.reservoir_size}")
            return False
        if target_size < min_size:
            target_size = min_size
            if target_size >= self.reservoir_size:
                return False

        old = self.reservoir_size
        cut = old - target_size
        print(f"\natrophy: {old} -> {target_size} (-{cut})")

        # activity score = sum of abs weights
        incoming = xp.abs(self.W_res).sum(axis=0)
        outgoing = xp.abs(self.W_res).sum(axis=1)
        score = incoming + outgoing

        keep = xp.sort(xp.argsort(score)[-target_size:])
        print(f"   keeping {len(keep)} most active neurons")

        self.W_res = self._normalize_spectral_radius(
            self.W_res[xp.ix_(keep, keep)], self.spectral_radius)
        self.W_in = self.W_in[keep, :]
        self.W_out = self.W_out[:, keep]
        self.state = self.state[keep]
        self.reservoir_size = target_size

        sr = self._compute_spectral_radius(self.W_res)
        print(f"   {target_size}x{target_size} sr={sr:.4f} done")
        return True

    def get_stats(self):
        return {
            "input_size": self.input_size,
            "reservoir_size": self.reservoir_size,
            "output_size": self.output_size,
            "spectral_radius": self.spectral_radius,
            "W_in_density": float(xp.mean(self.W_in != 0)),
            "W_res_density": float(xp.mean(self.W_res != 0)),
            "leak_rate": self.leak_rate,
            "trained": self._is_trained,
            "state_norm": float(xp.linalg.norm(self.state)),
        }


# ---- test ----
if __name__ == "__main__":
    print("=" * 60)
    print("lsm test")
    print("=" * 60)

    lsm = LiquidStateMachine()

    print("\nstats:")
    for k, v in lsm.get_stats().items():
        print(f"   {k}: {v}")

    print("\nsingle step...")
    inp = xp.random.rand(256)
    state = lsm.step(inp)
    print(f"   in:{inp.shape} out:{state.shape} norm:{xp.linalg.norm(state):.4f}")

    print("\nfading memory (zero input)...")
    norms = []
    for i in range(10):
        state = lsm.step(xp.zeros(256))
        norms.append(float(xp.linalg.norm(state)))
    print(f"   norms: {[f'{n:.4f}' for n in norms]}")
    print(f"   fading: {'Y' if norms[-1] < norms[0] else 'N'}")

    print("\ndone")
