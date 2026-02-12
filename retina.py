"""
retina.py - event-based vision system
differential optic nerve: only CHANGES get detected, static = invisible.
"""

import numpy as xp
# import cupy as xp

from typing import Optional, Tuple, Generator, Callable
import time
from dataclasses import dataclass


@dataclass
class SpikeEvent:
    timestamp: float
    energy: float
    spikes: xp.ndarray
    hotspots: int


class BioMimeticRetina:
    """event-based vision. captures CHANGES not frames.
    static -> invisible, motion -> spikes. feeds into LSM."""

    def __init__(self, grid_size=(16, 16), threshold=0.02,
                 decay_rate=0.8, spike_sensitivity=1.0, monitor=1):
        self.grid_size = grid_size
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.spike_sensitivity = spike_sensitivity
        self.monitor = monitor

        self._prev_frame = None
        self._activity_buffer = None
        self._frame_count = 0
        self._start_time = None

        # screen capture
        try:
            import mss
            self._sct = mss.mss()
            self._has_mss = True
        except ImportError:
            self._has_mss = False

        mode = 'mss' if self._has_mss else 'fallback'
        print(f"-- retina loaded ({grid_size[0]}x{grid_size[1]}={grid_size[0]*grid_size[1]} neurons, {mode}) --")

    def _capture_raw(self):
        """grab screen pixels -> grayscale float array"""
        if self._has_mss:
            try:
                import mss
                with mss.mss() as sct:
                    mon = sct.monitors[self.monitor]
                    raw = sct.grab(mon)
                    img = xp.frombuffer(raw.rgb, dtype=xp.uint8)
                    img = img.reshape((raw.height, raw.width, 3))
                    # green channel only (fastest, closest to human brightness)
                    return img[:, :, 1].astype(xp.float32) / 255.0
            except Exception as e:
                if not hasattr(self, '_warned_capture'):
                    print(f"capture error: {e}, using simulated data")
                    self._warned_capture = True
                return xp.random.rand(1080, 1920).astype(xp.float32) * 0.1
        else:
            return xp.random.rand(1080, 1920).astype(xp.float32) * 0.1

    def _downsample(self, frame):
        """block average to grid size (vectorized, no loops)"""
        h, w = frame.shape
        gh, gw = self.grid_size
        hc = (h // gh) * gh
        wc = (w // gw) * gw
        frame = frame[:hc, :wc]
        bh, bw = hc // gh, wc // gw
        return frame.reshape(gh, bh, gw, bw).mean(axis=(1, 3)).astype(xp.float32)

    def _compute_diff(self, current, previous):
        """| screen(t) - screen(t-1) |"""
        return xp.abs(current - previous)

    def _encode_spikes(self, diff):
        """threshold + amplify + sqrt nonlinearity"""
        spikes = xp.where(diff > self.threshold, diff, 0.0)
        spikes = xp.sqrt(spikes * self.spike_sensitivity)
        return xp.clip(spikes, 0.0, 1.0)

    def _temporal_filter(self, spikes):
        """leaky integration for temporal smoothing"""
        if self._activity_buffer is None:
            self._activity_buffer = xp.zeros(self.grid_size, dtype=xp.float32)
        self._activity_buffer = (
            self.decay_rate * self._activity_buffer +
            (1 - self.decay_rate) * spikes)
        return self._activity_buffer.copy()

    def process_frame(self):
        """capture + diff + encode -> SpikeEvent"""
        ts = time.perf_counter()
        if self._start_time is None:
            self._start_time = ts

        raw = self._capture_raw()
        current = self._downsample(raw)

        # first frame: no diff
        if self._prev_frame is None:
            self._prev_frame = current
            return SpikeEvent(timestamp=ts - self._start_time, energy=0.0,
                spikes=xp.zeros(self.grid_size).flatten(), hotspots=0)

        diff = self._compute_diff(current, self._prev_frame)
        spikes = self._encode_spikes(diff)
        filtered = self._temporal_filter(spikes)

        self._prev_frame = current
        self._frame_count += 1

        energy = float(xp.sum(filtered))
        hotspots = int(xp.sum(filtered > 0.3))

        return SpikeEvent(timestamp=ts - self._start_time, energy=energy,
            spikes=filtered.flatten(), hotspots=hotspots)

    def stream(self, target_fps=30.0, max_frames=None, callback=None):
        """continuous spike stream generator"""
        dt = 1.0 / target_fps
        count = 0
        print(f"retina stream at {target_fps} fps...")

        while max_frames is None or count < max_frames:
            t0 = time.perf_counter()
            spike = self.process_frame()
            if callback:
                callback(spike)
            yield spike
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            count += 1

    def get_stats(self):
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        return {
            "frame_count": self._frame_count,
            "elapsed_time": elapsed,
            "actual_fps": fps,
            "grid_size": self.grid_size,
            "threshold": self.threshold,
        }

    def reset(self):
        self._prev_frame = None
        self._activity_buffer = None
        self._frame_count = 0
        self._start_time = None


class LiveVisualizer:
    """realtime retina activity viewer (matplotlib)"""

    def __init__(self, grid_size=(16, 16)):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self.grid_size = grid_size
        self.plt = plt
        self.energy_history = []
        self.max_history = 200

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))

        self.im = self.ax1.imshow(xp.zeros(grid_size), cmap='hot',
            vmin=0, vmax=1, interpolation='nearest')
        self.ax1.set_title('retina activity')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.cbar = self.fig.colorbar(self.im, ax=self.ax1)
        self.cbar.set_label('spike energy')

        self.line, = self.ax2.plot([], [], 'g-', linewidth=1.5)
        self.ax2.set_xlim(0, self.max_history)
        self.ax2.set_ylim(0, 50)
        self.ax2.set_title('energy over time')
        self.ax2.set_xlabel('frame')
        self.ax2.set_ylabel('energy')
        self.ax2.grid(True, alpha=0.3)

        self.stats_text = self.ax2.text(0.02, 0.98, '',
            transform=self.ax2.transAxes, verticalalignment='top',
            fontfamily='monospace', fontsize=10)
        self.plt.tight_layout()

    def update(self, spike):
        spikes_2d = spike.spikes.reshape(self.grid_size)
        if hasattr(spikes_2d, 'get'):
            spikes_2d = spikes_2d.get()
        self.im.set_array(spikes_2d)

        self.energy_history.append(spike.energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)

        self.line.set_data(range(len(self.energy_history)), self.energy_history)
        if self.energy_history:
            mx = max(self.energy_history) * 1.2
            if mx > 0:
                self.ax2.set_ylim(0, max(50, mx))

        self.stats_text.set_text(
            f"t: {spike.timestamp:.1f}s\nenergy: {spike.energy:.2f}\nhotspots: {spike.hotspots}")
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def show(self):
        self.plt.show(block=False)

    def close(self):
        self.plt.close(self.fig)


# ---- test ----
if __name__ == "__main__":
    print("=" * 60)
    print("retina test")
    print("=" * 60)

    retina = BioMimeticRetina(grid_size=(16, 16), threshold=0.02, decay_rate=0.7)

    if not retina._has_mss:
        print("\nmss not installed, running with simulated data...")

    print("\nprocessing 10 frames...")
    for i in range(10):
        spike = retina.process_frame()
        print(f"   frame {i+1}: energy={spike.energy:.4f} hotspots={spike.hotspots}")
        time.sleep(0.05)

    print("\nstats:")
    for k, v in retina.get_stats().items():
        print(f"   {k}: {v}")

    print("\ndone")
