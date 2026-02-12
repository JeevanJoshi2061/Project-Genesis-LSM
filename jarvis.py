"""
jarvis.py - the complete digital human
all 15 layers wired together. this is life, not software.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import numpy as xp
# import cupy as xp

import time
import threading
from typing import Optional, Tuple, List, Callable
from pathlib import Path

# ---- imports ----

from lsm import LiquidStateMachine
from retina import BioMimeticRetina, SpikeEvent
from memory import VectorMemory, HierarchicalMemory, MemoryLevel
from router import LogicRouter
from hormones import EndocrineSystem, Mood
from consciousness import (
    DigitalOrganism, DigitalSubconscious, DigitalMetabolism,
    MirrorNeurons, SurvivalInstinct, EnergyState, UserMood, HealthState,
)
from mind import (
    HumanMind, TheoryOfMind, GenerativeImagination,
    Chronesthesia, MoralCompass, MoralPriority,
)
from biology import (
    BiologicalSystem, CircadianRhythm, Neurogenesis,
    AuditoryCortex, PainReceptors,
)
from broca import BrocaArea, SpeechEmotion, InnerMonologue, CensorFilter
from nervous import NervousSystem, SpinalCord, MotorControl, MotorSkillLevel
from madness import MadnessSystem, DigitalSynesthesia, PhantomLimb, FeverState
from god import GodComplex, DigitalImmuneSystem, GeneticMutation, GhostInMachine
from hive import HiveSystem, DigitalReproduction, HiveMind, QuantumLogic
from experiment import ExperimentSystem, MirrorTest, DreamVisualizer, EmergentLanguage


class Jarvis:
    """the complete digital human - 15 layers, one organism."""

    def __init__(self, reservoir_size=2000, grid_size=(16, 16),
                 save_dir="d:/Liquid State Machine"):
        self.grid_size = grid_size
        self.input_size = grid_size[0] * grid_size[1]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("JARVIS - Digital Life Form")
        print("=" * 70)

        # L1: eyes
        print("\n[1] eyes...")
        self.retina = BioMimeticRetina(
            grid_size=grid_size, threshold=0.02, decay_rate=0.7)

        # L2: brain
        print("\n[2] brain...")
        self.lsm = LiquidStateMachine(
            input_size=self.input_size, reservoir_size=reservoir_size,
            output_size=10, sparsity=0.1, spectral_radius=0.9, seed=42)

        # L3: memory
        print("\n[3] memory...")
        self.memory = VectorMemory(
            vector_dim=reservoir_size, max_memories=500,
            save_path=str(self.save_dir / "jarvis_memories.pkl"))

        # L4: hormones
        print("\n[4] hormones...")
        self.hormones = EndocrineSystem(
            on_mood_change=self._on_hormonal_mood_change)

        # L5: soul (consciousness)
        print("\n[5] soul...")
        self.subconscious = DigitalSubconscious(
            memory_source=self._get_memories_for_dreaming,
            on_hunch=self._on_hunch, interrupt_probability=0.15)
        self.metabolism = DigitalMetabolism(
            on_energy_change=self._on_energy_change,
            on_exhausted=self._on_exhausted)
        self.mirror = MirrorNeurons(on_mood_change=self._on_user_mood_change)
        self.survival = SurvivalInstinct(
            on_warning=self._on_survival_warning,
            on_critical=self._on_survival_critical,
            on_dying=self._on_survival_dying)

        # L6: mind
        print("\n[6] mind...")
        self.mind = HumanMind()

        # L7: biology
        print("\n[7] biology...")
        self.biology = BiologicalSystem()

        # L8: voice
        print("\n[8] voice...")
        self.voice = BrocaArea(verbose_thoughts=True)

        # L9: nervous
        print("\n[9] nervous...")
        self.nervous = NervousSystem(motor_enabled=False)

        # L10: madness
        print("\n[10] madness...")
        self.madness = MadnessSystem(motor_available=False)

        # L11: god
        print("\n[11] god...")
        self.god = GodComplex(
            memory_source=lambda: [(m.label, m.vector) for m in self.memory.memories],
            save_dir=save_dir)

        # L12: hive
        print("\n[12] hive...")
        self.hive = HiveSystem(
            instance_id="jarvis_prime", port=5000,
            genes_source=lambda: {n: g.value for n, g in self.god.genes.genes.items()},
            memory_source=lambda: [(m.label, m.vector) for m in self.memory.memories])

        # L13: experiments
        print("\n[13] experiments...")
        self.experiments = ExperimentSystem(instance_id="jarvis_prime")

        # L14: hierarchical memory
        print("\n[14] HTM...")
        self.htm = HierarchicalMemory(
            vector_dim=2000, max_ram_memories=500,
            archive_path=str(self.save_dir / "jarvis_longterm.pkl.gz"))

        # L15: logic router
        print("\n[15] logic router...")
        self.router = LogicRouter(
            llm_fallback=lambda q: self.voice.speech.generate_response(q))

        # plasticity
        print("\n[+] plasticity...")
        self.lsm.enable_plasticity()

        # state
        self.is_alive = False
        self.birth_time = time.time()
        self.current_state = None
        self.last_prediction = None
        self.last_confidence = 0.0
        self.pending_question = None

        # thread safety
        self.lock = threading.Lock()
        self._frame_counter = 0
        self._running = False
        self._thread = None

        # timing
        self.last_activity = time.time()
        self.last_dream = time.time()
        self.last_question = 0.0

        print("\n" + "=" * 70)
        print("JARVIS READY (15 layers)")
        print("   call .wake() to bring to life")
        print("=" * 70)

    # ---- memory source ----

    def _get_memories_for_dreaming(self):
        return [(m.label, m.keywords) for m in self.memory.memories]

    # ---- callbacks ----

    def _on_hormonal_mood_change(self, old, new):
        print(f"\nhormonal mood: {old.value} -> {new.value}")

    def _on_hunch(self, hunch):
        mods = self.mirror.response_modifier
        if mods['interrupt_ok']:
            print(f"\nIDEA: {hunch.idea}")

    def _on_energy_change(self, state):
        emj = {EnergyState.EXHAUSTED: "😫", EnergyState.TIRED: "😴",
               EnergyState.NORMAL: "😊", EnergyState.ENERGETIC: "🤩",
               EnergyState.SUPERCHARGED: "⚡"}
        print(f"\n{emj.get(state, '?')} energy: {state.value}")

    def _on_exhausted(self):
        print("\nenergy critical... need reward")

    def _on_user_mood_change(self, old, new):
        print(f"\nuser mood: {old.value} -> {new.value}")

    def _on_survival_warning(self, msg):
        pass

    def _on_survival_critical(self, msg):
        pass

    def _on_survival_dying(self, msg):
        pass

    # ---- core processing ----

    def _process_frame(self):
        """one frame through all systems (thread-safe)"""

        # pain lag: skip frames instead of blocking
        self._frame_counter += 1
        pain_perf = self.biology.pain.get_performance_modifier()
        if pain_perf < 1.0:
            skip_prob = 1.0 - pain_perf
            if xp.random.random() < skip_prob:
                return

        with self.lock:
            # circadian + hormone mods
            circ_mods = self.biology.circadian.get_behavior_modifiers()
            horm_mods = self.hormones.get_behavior_modifiers()

            # retina sensitivity = stress + alertness
            base_thresh = horm_mods['sensitivity_threshold']
            alertness = circ_mods['alertness']
            self.retina.threshold = base_thresh / max(0.5, alertness)

            # eyes
            spike = self.retina.process_frame()

            # synesthesia overlay
            vis_noise = self.madness.synesthesia.get_visual_overlay()
            if vis_noise is not None:
                try:
                    nf = vis_noise.flatten()
                    if nf.shape == spike.spikes.shape:
                        spike.spikes = xp.clip(spike.spikes + nf * 0.8, 0.0, 1.0)
                        spike.energy = max(spike.energy, 0.5)
                except:
                    pass

            # mirror neurons
            self.mirror.process_activity(spike.energy)

            if spike.energy > 0.1:
                # active
                self.metabolism.consume(0.1, "observation")
                self.last_activity = time.time()

                # brain
                state = self.lsm.step(spike.spikes)
                self.current_state = state.copy()

                # memory lookup
                confidence, label, _ = self.memory.get_confidence(state)
                confidence *= self.hormones.learning_rate_multiplier
                self.last_prediction = label
                self.last_confidence = confidence

                # self-reward for consistent predictions
                if hasattr(self, '_prev_prediction') and label == self._prev_prediction:
                    if confidence > 0.7:
                        self.hormones.on_prediction_success(confidence)
                self._prev_prediction = label

                # curiosity
                if self._should_ask(confidence):
                    self._ask_question(label, confidence)

                self.hormones.process_activity(spike.energy)

            else:
                # idle -> dream
                self.subconscious.set_active(True)
                idle = time.time() - self.last_activity
                if idle > 60 and self.hormones.should_dream:
                    if time.time() - self.last_dream > 300:
                        self._dream()

    def _should_ask(self, confidence):
        if not self.hormones.should_ask_questions:
            return False
        if time.time() - self.last_question < 15.0:
            return False
        mods = self.mirror.response_modifier
        if not mods['interrupt_ok']:
            return False
        return confidence < self.hormones.curiosity_threshold

    def _ask_question(self, label, confidence):
        self.last_question = time.time()
        self.metabolism.consume(0.5, "asking")

        mods = self.metabolism.response_modifier
        if mods['tone'] == 'excited':
            q = "something new! what is this?"
        elif mods['tone'] == 'minimal':
            q = "unknown. label?"
        else:
            q = "don't know this pattern. what is it?"

        print(f"\n\n{'='*50}")
        print(f"   ? {q}")
        print(f"   confidence: {confidence:.0%}")
        print(f"{'='*50}")
        self.pending_question = (self.current_state, q)

    def _dream(self):
        """consolidate memories, apply plasticity, manage brain size"""
        print("\ndreaming...")

        # 1. flat memory
        self.memory.consolidate()
        self.memory.save()

        # 2. HTM
        self.htm.sleep_consolidate()

        # 3. plasticity
        self.lsm.apply_batch_plasticity()

        # 4. atrophy if brain too big
        MAX_SAFE = 5000
        TARGET = 3000
        cur = self.lsm.reservoir_size
        if cur > MAX_SAFE:
            print(f"brain too big ({cur})! atrophy...")
            self.lsm.atrophy(TARGET, min_size=2000)
            print(f"   {cur} -> {self.lsm.reservoir_size}")

        self.last_dream = time.time()
        print("dream done")

    # ---- main loop ----

    def _life_loop(self):
        print("\njarvis is ALIVE!")
        tick = 0
        while self._running:
            try:
                self._process_frame()
                tick += 1
                if tick % 30 == 0:  # ~1s
                    self.hormones.tick()
                    self.metabolism.tick()
                if tick % 300 == 0:  # ~10s
                    if self.survival.should_request_feedback():
                        print(f"\n{self.survival.get_feedback_request()}")
                time.sleep(0.033)  # ~30fps
            except Exception as e:
                print(f"\nerr: {e}")
                time.sleep(1)
        print("\nlife loop ended")

    # ---- public API ----

    def wake(self):
        if self._running:
            print("already awake!")
            return
        self._running = True
        self.is_alive = True
        self.birth_time = time.time()
        self.subconscious.start_dreaming()
        self._thread = threading.Thread(target=self._life_loop, daemon=True)
        self._thread.start()
        print("\ngood morning. jarvis is conscious.")

    def sleep(self):
        self._running = False
        self.is_alive = False
        self.subconscious.stop_dreaming()
        if self._thread:
            self._thread.join(timeout=2)
        self.memory.save()
        age = time.time() - self.birth_time
        print(f"\ngood night. lived {age:.0f}s.")

    def teach(self, label, difficulty=0.5):
        """teach current pattern"""
        with self.lock:
            if self.current_state is None:
                print("nothing to teach! need activity first")
                return

            lr_boost = self.hormones.learning_rate_multiplier

            # neurogenesis: hard learning grows the brain
            if difficulty > 0.5:
                self.biology.neurogenesis.stimulate_growth(difficulty, f"learning:{label}")
                if self.biology.neurogenesis.growth_hormone > 70:
                    new_sz = self.biology.neurogenesis.current_size
                    if new_sz > self.lsm.reservoir_size:
                        self.lsm.resize(new_sz)
                        self.memory.resize_vectors(new_sz)

            is_new, mem = self.memory.store(
                self.current_state, label, strength=2.0 * lr_boost)

            # reward all systems
            self.metabolism.on_reward()
            self.survival.on_success()
            self.hormones.on_success()
            self.biology.pain.relieve_pain(20)

            print(f"\nlearned: '{label}' (boost:{lr_boost:.1f}x diff:{difficulty:.0%})")
            self.pending_question = None
            self.memory.save()

    def praise(self, intensity=1.0):
        self.hormones.on_success(intensity)
        self.metabolism.on_praise()
        self.survival.on_success()
        if self.last_prediction:
            self.memory.reinforce_label(self.last_prediction, intensity)
        print("\n*happy* thank you!")

    def scold(self, intensity=1.0):
        self.hormones.on_failure(intensity)
        self.metabolism.on_punishment()
        self.survival.on_failure()
        if self.last_prediction:
            self.memory.punish_label(self.last_prediction, intensity)
        print("\n*sad* i'll try harder...")

    def status(self):
        print("\n" + "=" * 70)
        print("JARVIS STATUS")
        print("=" * 70)
        age = time.time() - self.birth_time
        print(f"\n   age: {age:.0f}s")
        print(f"   alive: {'yes' if self.is_alive else 'no'}")
        print(f"   prediction: {self.last_prediction} ({self.last_confidence:.0%})")
        print(f"   memories: {len(self.memory.memories)}")
        self.hormones.print_status()
        self.metabolism.print_status()
        self.mirror.print_status()
        self.survival.print_status()
        hunch = self.subconscious.get_latest_hunch()
        if hunch:
            print(f"\n   latest idea: {hunch.idea}")


# ---- console ----

class JarvisConsole:
    """
    interactive console.
    > text = chat, text = teach, + = praise, - = scold,
    s = status, h = hormones, m = memory, d = dream, q = quit
    """

    def __init__(self):
        self.jarvis = Jarvis()

    def run(self):
        print("\n" + "=" * 70)
        print("JARVIS CONSOLE")
        print("=" * 70)
        print("""
commands:
    > [text] : chat (LLM + hormones)
    [text]?  : ask a question
    [text]   : teach label
    +        : praise
    ++       : big praise (motor boost)
    -        : scold
    --       : big scold
    s        : status
    h        : hormones
    m        : memory
    d        : dream
    q        : quit
""")

        self.jarvis.wake()

        try:
            while True:
                try:
                    cmd = input().strip()
                except EOFError:
                    break

                if not cmd:
                    continue

                if cmd.lower() == 'q':
                    break
                elif cmd == '+':
                    self.jarvis.praise(1.0)
                elif cmd == '++':
                    self.jarvis.praise(3.0)
                    self.jarvis.nervous.motor.on_reward(2.0)
                elif cmd == '-':
                    self.jarvis.scold(1.0)
                elif cmd == '--':
                    self.jarvis.scold(3.0)
                elif cmd.lower() == 's':
                    self.jarvis.status()
                elif cmd.lower() == 'h':
                    self.jarvis.hormones.print_status()
                elif cmd.lower() == 'm':
                    self.jarvis.memory.print_stats()
                elif cmd.lower() == 'd':
                    self.jarvis._dream()
                elif cmd.lower().startswith('rename '):
                    parts = cmd[7:].split(' to ')
                    if len(parts) == 2:
                        old_l, new_l = parts
                        print(f"correcting '{old_l}' -> '{new_l}'...")
                        self.jarvis.teach(new_l.strip(), difficulty=0.1)

                # teach mode
                elif not cmd.startswith(">") and "?" not in cmd and len(cmd) > 1:
                    clean = cmd.lower()
                    for pfx in ["sorry ", "this is ", "actually ", "it is "]:
                        if clean.startswith(pfx):
                            clean = clean.replace(pfx, "", 1)
                    self.jarvis.teach(clean.strip(), difficulty=0.5)

                elif cmd.startswith(">") or "?" in cmd:
                    print(f"\nthinking... (energy: {self.jarvis.metabolism.state.energy:.0f}%)")

                    ctx = f"""mood: {self.jarvis.hormones.mood.value}
cortisol: {self.jarvis.hormones.cortisol.level:.0f}%
dopamine: {self.jarvis.hormones.dopamine.level:.0f}%
pain: {self.jarvis.biology.pain.current_pain:.0f}%
survival: {self.jarvis.survival.health:.0f}%
energy: {self.jarvis.metabolism.state.energy:.0f}%"""

                    self.jarvis.voice.update_hormones({
                        'cortisol': self.jarvis.hormones.cortisol.level,
                        'dopamine': self.jarvis.hormones.dopamine.level,
                        'oxytocin': self.jarvis.hormones.oxytocin.level,
                        'pain': self.jarvis.biology.pain.current_pain,
                        'survival': self.jarvis.survival.health,
                    })

                    user_mood = self.jarvis.mirror.current_mood.value
                    self.jarvis.voice.update_user_mood(user_mood)

                    clean_in = cmd.replace(">", "").strip()
                    self.jarvis.voice.process_and_respond(
                        user_input=clean_in, context=ctx, actually_speak=False)

                else:
                    self.jarvis.teach(cmd)

        except KeyboardInterrupt:
            pass
        finally:
            self.jarvis.sleep()
            print("\ngoodbye!")


if __name__ == "__main__":
    console = JarvisConsole()
    console.run()
