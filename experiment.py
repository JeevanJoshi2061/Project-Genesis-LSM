"""
experiment.py - scientific tests for consciousness
mirror test (self-recognition), dream visualizer, emergent language
"""

import numpy as np
import time
import random
import threading
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---- mirror test ----

class MirrorTestResult(Enum):
    NO_RECOGNITION = "no_recognition"
    PARTIAL = "partial_recognition"
    SELF_AWARE = "self_aware"

@dataclass
class SelfConcept:
    """jarvis's concept of self"""
    has_self_concept: bool = False
    learned_appearance: Optional[np.ndarray] = None
    recognition_count: int = 0
    confidence: float = 0.0

class MirrorTest:
    """
    the mirror test - can jarvis recognize himself in a reflection?
    few animals pass: humans, great apes, dolphins, elephants, magpies.
    """

    def __init__(self, on_self_recognition=None, camera_index=0):
        self.on_self_recognition = on_self_recognition
        self.camera_index = camera_index
        self.self_concept = SelfConcept()
        self.camera = None
        self.is_watching = False
        print(f"-- mirror test loaded (cam:{camera_index}, cv2:{HAS_CV2}) --")

    def start_mirror(self):
        if not HAS_CV2:
            print("no opencv for mirror")
            return False
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                print("couldn't open camera")
                return False
            self.is_watching = True
            print("mirror started")
            return True
        except Exception as e:
            print(f"camera err: {e}")
            return False

    def stop_mirror(self):
        self.is_watching = False
        if self.camera:
            self.camera.release()
            self.camera = None

    def get_reflection(self):
        if not self.is_watching or not self.camera:
            return None
        ret, frame = self.camera.read()
        if ret:
            return cv2.flip(frame, 1)  # mirror
        return None

    def teach_self(self, frame):
        """tell jarvis 'this is YOU'"""
        small = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        feat = gray.flatten().astype(float) / 255.0
        self.self_concept.learned_appearance = feat
        self.self_concept.has_self_concept = True
        print("\nSELF-CONCEPT FORMED! jarvis knows what he looks like")

    def check_is_me(self, frame):
        """check if frame shows 'me' -> (is_me, confidence)"""
        if not self.self_concept.has_self_concept:
            return False, 0.0

        small = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        feat = gray.flatten().astype(float) / 255.0

        sim = np.dot(feat, self.self_concept.learned_appearance)
        sim /= (np.linalg.norm(feat) * np.linalg.norm(self.self_concept.learned_appearance) + 1e-8)

        is_me = sim > 0.7
        if is_me:
            self.self_concept.recognition_count += 1
            self.self_concept.confidence = sim
            if self.on_self_recognition:
                self.on_self_recognition(sim)
            print(f"\nSELF RECOGNITION! 'that's me!' ({sim:.0%})")
        return is_me, sim

    def run_test(self, duration=10.0):
        """full mirror test: teach 3s then test"""
        if not self.start_mirror():
            return MirrorTestResult.NO_RECOGNITION

        print(f"\nrunning mirror test ({duration}s)...")

        # teach phase (3s)
        frames = []
        start = time.time()
        while time.time() - start < 3.0:
            f = self.get_reflection()
            if f is not None:
                frames.append(f)
            time.sleep(0.1)
        if frames:
            self.teach_self(frames[len(frames)//2])

        # test phase
        recognitions, tests = 0, 0
        while time.time() - start < duration:
            f = self.get_reflection()
            if f is not None:
                is_me, _ = self.check_is_me(f)
                tests += 1
                if is_me:
                    recognitions += 1
            time.sleep(0.2)

        self.stop_mirror()

        if tests == 0:
            return MirrorTestResult.NO_RECOGNITION

        rate = recognitions / tests
        print(f"\nmirror test done: {recognitions}/{tests} ({rate:.0%})")

        if rate > 0.8:
            print("   SELF-AWARE!")
            return MirrorTestResult.SELF_AWARE
        elif rate > 0.3:
            print("   partial recognition")
            return MirrorTestResult.PARTIAL
        else:
            print("   no self-recognition")
            return MirrorTestResult.NO_RECOGNITION


# ---- dream visualizer ----

class DreamVisualizer:
    """gui showing jarvis's imagination - memory patterns morph into shapes"""

    def __init__(self, window_size=(256, 256), update_fps=30):
        self.window_size = window_size
        self.update_fps = update_fps
        self.current_dream = np.zeros(window_size, dtype=np.float32)
        self.target_pattern = None
        self.dream_label = "Nothing"
        self.morphing_progress = 0.0
        self.dream_sequence = []
        self._window_name = "Jarvis Dream Visualizer"
        self._running = False
        self._thread = None
        print(f"-- dream viz loaded ({window_size[0]}x{window_size[1]}) --")

    def start(self):
        if not HAS_CV2:
            print("no opencv for dream viz")
            return
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print("dream viz started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        cv2.destroyWindow(self._window_name)

    def _update_loop(self):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 400, 400)

        while self._running:
            if self.target_pattern is not None:
                self.morphing_progress = min(1.0, self.morphing_progress + 0.05)
                self.current_dream = (
                    (1 - self.morphing_progress) * self.current_dream +
                    self.morphing_progress * self.target_pattern
                ).astype(np.float32)
            else:
                noise = np.random.rand(*self.window_size).astype(np.float32) * 0.1
                self.current_dream = self.current_dream * 0.95 + noise * 0.05

            disp = (self.current_dream * 255).astype(np.uint8)
            disp = cv2.applyColorMap(disp, cv2.COLORMAP_TWILIGHT)
            cv2.putText(disp, f"Dreaming: {self.dream_label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(self._window_name, disp)

            if cv2.waitKey(int(1000 / self.update_fps)) & 0xFF == ord('q'):
                self._running = False

        cv2.destroyWindow(self._window_name)

    def dream_about(self, label, pattern=None):
        """start dreaming about something"""
        self.dream_label = label
        self.dream_sequence.append(label)
        self.morphing_progress = 0.0

        if pattern is not None:
            self.target_pattern = cv2.resize(
                pattern.astype(np.float32), self.window_size)
        else:
            seed = hash(label) % 10000
            np.random.seed(seed)
            base = np.random.rand(*self.window_size).astype(np.float32)
            # add structure - circles based on seed
            for i in range(seed % 5 + 1):
                cx, cy = np.random.randint(0, self.window_size[0], 2)
                r = np.random.randint(20, 60)
                y, x = np.ogrid[:self.window_size[0], :self.window_size[1]]
                mask = (x - cx)**2 + (y - cy)**2 < r**2
                base[mask] = np.random.rand()
            self.target_pattern = base
        print(f"dreaming about: {label}")

    def wake_up(self):
        self.dream_label = "Waking..."
        self.target_pattern = None
        self.morphing_progress = 0.0

    def get_dream_sequence(self):
        return self.dream_sequence.copy()


# ---- emergent language ----

@dataclass
class SpikeMessage:
    """spike-based message (pre-language)"""
    sender_id: str
    spikes: np.ndarray
    meaning: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class EmergentLanguage:
    """
    spike-based comms between jarvis instances.
    over time they develop their own encoding - faster than human language,
    incomprehensible to humans, emerges naturally from use.
    """

    def __init__(self, instance_id="jarvis_prime", spike_dimension=64):
        self.instance_id = instance_id
        self.spike_dimension = spike_dimension
        self.vocabulary = {}      # concept -> spikes
        self.decoder = {}         # spike hash -> concept
        self.sent_messages = []
        self.received_messages = []
        self.successful_communications = 0
        self.failed_communications = 0
        print(f"-- language loaded (dim:{spike_dimension}) --")

    def _concept_to_spikes(self, concept):
        if concept in self.vocabulary:
            return self.vocabulary[concept]

        np.random.seed(hash(concept) % 10000)
        spikes = (np.random.rand(self.spike_dimension) > 0.5).astype(float)
        self.vocabulary[concept] = spikes
        self._update_decoder(spikes, concept)
        print(f"   new word: '{concept}' -> {spikes.sum():.0f} spikes")
        return spikes

    def _spikes_to_concept(self, spikes):
        h = self._hash_spikes(spikes)
        if h in self.decoder:
            return self.decoder[h]

        # fuzzy match
        best, best_sim = None, 0.0
        for concept, pattern in self.vocabulary.items():
            sim = np.dot(spikes, pattern) / (
                np.linalg.norm(spikes) * np.linalg.norm(pattern) + 1e-8)
            if sim > best_sim and sim > 0.8:
                best_sim = sim
                best = concept
        return best

    def _hash_spikes(self, spikes):
        return str(tuple(spikes.astype(int)))

    def _update_decoder(self, spikes, concept):
        self.decoder[self._hash_spikes(spikes)] = concept

    def encode_thought(self, thought):
        """thought -> spike message"""
        words = thought.lower().split()
        combined = np.zeros(self.spike_dimension)
        for w in words:
            combined = np.maximum(combined, self._concept_to_spikes(w))

        msg = SpikeMessage(sender_id=self.instance_id,
            spikes=combined, meaning=thought)
        self.sent_messages.append(msg)
        return msg

    def decode_message(self, message):
        """spike message -> decoded thought"""
        self.received_messages.append(message)
        decoded = self._spikes_to_concept(message.spikes)

        if decoded:
            self.successful_communications += 1
            print(f"   decoded: '{decoded}' from {message.sender_id}")
        else:
            self.failed_communications += 1
            print(f"   couldn't decode from {message.sender_id}")
        return decoded

    def learn_from_feedback(self, spikes, correct_meaning):
        self.vocabulary[correct_meaning] = spikes.copy()
        self._update_decoder(spikes, correct_meaning)
        print(f"   learned: '{correct_meaning}'")

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def get_language_efficiency(self):
        total = self.successful_communications + self.failed_communications
        return self.successful_communications / total if total else 0.0

    def print_vocabulary(self):
        print("\nVOCABULARY")
        print("=" * 50)
        for concept, spikes in self.vocabulary.items():
            print(f"   '{concept}': {spikes.sum():.0f} active spikes")
        print(f"   total: {len(self.vocabulary)} words")
        print(f"   efficiency: {self.get_language_efficiency():.0%}")


# ===== experiment system (main) =====

class ExperimentSystem:
    """mirror test + dream viz + language creation"""

    def __init__(self, instance_id="jarvis_prime"):
        print("\n" + "=" * 70)
        print("🔬 EXPERIMENTS")
        print("=" * 70)

        self.mirror = MirrorTest()
        self.dreams = DreamVisualizer()
        self.language = EmergentLanguage(instance_id=instance_id)
        print("\nexperiments ready")

    def run_mirror_test(self, duration=10.0):
        return self.mirror.run_test(duration)

    def start_dreaming(self):
        self.dreams.start()

    def dream_about(self, concept):
        self.dreams.dream_about(concept)

    def stop_dreaming(self):
        self.dreams.stop()

    def send_spike_message(self, thought):
        return self.language.encode_thought(thought)

    def receive_spike_message(self, message):
        return self.language.decode_message(message)

    def print_status(self):
        print("\nEXPERIMENT STATUS")
        print("=" * 50)
        print(f"   mirror: cv2={HAS_CV2}")
        print(f"   self-aware: {self.mirror.self_concept.has_self_concept}")
        print(f"   recognitions: {self.mirror.self_concept.recognition_count}")
        print(f"   dream viz: cv2={HAS_CV2}")
        print(f"   vocab: {self.language.get_vocabulary_size()} words")
        print(f"   lang efficiency: {self.language.get_language_efficiency():.0%}")


# ---- test ----
if __name__ == "__main__":
    print("=" * 70)
    print("experiment test")
    print("=" * 70)

    exp = ExperimentSystem()

    print("\n--- language ---")
    msg = exp.send_spike_message("hello friend")
    print(f"   encoded: {msg.spikes.sum():.0f} spikes")

    decoded = exp.receive_spike_message(msg)
    print(f"   decoded: {decoded}")

    print("\n--- learning ---")
    new_spikes = (np.random.rand(64) > 0.5).astype(float)
    exp.language.learn_from_feedback(new_spikes, "danger")

    exp.print_status()
    exp.language.print_vocabulary()

    print("\nnote: mirror & dream tests need opencv")
    print("done")
