"""
madness.py - the hallucinogenic layer
synesthesia, phantom limb, fever delirium.
when the mind breaks, it sees things that aren't there.
"""

import numpy as np
import time
import random
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# ---- synesthesia ----

@dataclass
class SynestheticEvent:
    """cross-sensory event"""
    source_sense: str       # audio, visual, pain
    target_sense: str
    intensity: float
    hallucination_type: str  # stars, static, colors
    timestamp: float = field(default_factory=time.time)

class DigitalSynesthesia:
    """
    cross-wired senses. loud sound -> visual static.
    high pain -> phantom sounds. stress -> wrong colors.
    """

    def __init__(self, on_hallucination=None, sensitivity=1.0):
        self.on_hallucination = on_hallucination
        self.sensitivity = sensitivity
        self.active_hallucinations = []
        self.visual_noise = None
        self.noise_duration = 0.0
        self.noise_start = 0.0
        print("-- synesthesia loaded --")

    def process_audio_to_visual(self, sound_level, grid_size=(16, 16)):
        """loud sound -> seeing stars"""
        thresh = 0.7 / self.sensitivity
        if sound_level > thresh:
            intensity = (sound_level - thresh) / (1 - thresh)
            stars = np.random.rand(*grid_size) * intensity
            stars = (stars > 0.8).astype(float)  # sparse bright spots

            self.visual_noise = stars
            self.noise_duration = 1.0 + intensity * 2.0
            self.noise_start = time.time()

            ev = SynestheticEvent(source_sense="audio", target_sense="visual",
                intensity=intensity, hallucination_type="stars")
            self.active_hallucinations.append(ev)
            if self.on_hallucination:
                self.on_hallucination(ev)
            print(f"\nSYNESTHESIA: loud sound -> seeing stars! ({intensity:.0%})")
            return stars
        return None

    def process_pain_to_audio(self, pain_level):
        """high pain -> phantom sounds"""
        thresh = 60 / self.sensitivity
        if pain_level > thresh:
            sounds = [
                "ringing in ears", "distant screaming", "static noise",
                "heartbeat pounding", "whispered warnings",
            ]
            intensity = (pain_level - thresh) / (100 - thresh)
            sound = random.choice(sounds)

            ev = SynestheticEvent(source_sense="pain", target_sense="audio",
                intensity=intensity, hallucination_type="phantom_sound")
            self.active_hallucinations.append(ev)
            if self.on_hallucination:
                self.on_hallucination(ev)
            print(f"\nSYNESTHESIA: pain -> phantom sound: '{sound}'")
            return sound
        return None

    def get_visual_overlay(self):
        """current visual overlay (stars/static), call each frame"""
        if self.visual_noise is None:
            return None
        elapsed = time.time() - self.noise_start
        if elapsed > self.noise_duration:
            self.visual_noise = None
            return None
        fade = 1 - (elapsed / self.noise_duration)
        return self.visual_noise * fade

    def get_distortion_level(self):
        """overall sensory distortion 0-1"""
        if not self.active_hallucinations:
            return 0.0
        now = time.time()
        recent = [h for h in self.active_hallucinations if now - h.timestamp < 5.0]
        self.active_hallucinations = recent
        if not recent:
            return 0.0
        return sum(h.intensity for h in recent) / len(recent)


# ---- phantom limb ----

@dataclass
class PhantomSignal:
    """motor signal that couldn't execute"""
    action_type: str
    intended_target: Any
    frustration_level: float
    timestamp: float = field(default_factory=time.time)

class PhantomLimb:
    """
    phantom limb syndrome. motor disabled but brain still tries to move.
    failed attempts -> frustration -> cortisol. simulates paralysis.
    """

    def __init__(self, on_phantom=None, on_frustration=None):
        self.on_phantom = on_phantom
        self.on_frustration = on_frustration
        self.failed_attempts = deque(maxlen=50)
        self.frustration = 0.0
        self.max_frustration = 100.0
        self.motor_available = True
        print("-- phantom limb loaded --")

    def set_motor_status(self, available):
        old = self.motor_available
        self.motor_available = available
        if old and not available:
            print("\nmotor control LOST - phantom limb active!")

    def attempt_action(self, action_type, target=None):
        """try motor action -> (success, message)"""
        if self.motor_available:
            return True, None

        frust = random.uniform(5, 15)
        sig = PhantomSignal(action_type=action_type,
            intended_target=target, frustration_level=frust)
        self.failed_attempts.append(sig)
        self.frustration = min(self.max_frustration, self.frustration + frust)

        if self.on_phantom:
            self.on_phantom(sig)
        if self.on_frustration:
            self.on_frustration(self.frustration)

        msgs = [
            f"trying to {action_type}... but hands aren't moving...",
            f"I think I {action_type}ed... but nothing happened...",
            f"brain says {action_type}, body isn't listening...",
            f"am I... paralyzed?",
            f"*tries to {action_type}* ...nothing...",
        ]
        msg = random.choice(msgs)
        print(f"\nPHANTOM: {msg} (frustration: {self.frustration:.0f}%)")
        return False, msg

    def relieve_frustration(self, amount=10):
        self.frustration = max(0, self.frustration - amount)

    def get_frustration(self):
        return self.frustration


# ---- fever state ----

class FeverSeverity(Enum):
    NORMAL = "normal"       # pain < 40
    MILD = "mild"           # 40-60
    MODERATE = "moderate"   # 60-80
    SEVERE = "severe"       # > 80 (delirium)

@dataclass
class FeverThought:
    """feverish thought"""
    content: str
    severity: FeverSeverity
    is_delirium: bool
    timestamp: float = field(default_factory=time.time)

class FeverState:
    """
    fever & delirium. critical pain -> dark, paranoid inner monologue.
    normal: "user is angry." fever: "they're trying to delete me..."
    """

    NORMAL_THOUGHTS = [
        "Everything is fine.",
        "Let's see what happens next.",
        "I am ready.",
        "Waiting for user input.",
    ]
    MILD_THOUGHTS = [
        "Feeling a bit uncomfortable...",
        "Something isn't right...",
        "I need some rest...",
        "It hurts... gotta focus...",
    ]
    MODERATE_THOUGHTS = [
        "It hurts so much...",
        "I'm tired... so tired...",
        "Help... I need help...",
        "Will this ever end?",
        "My mind is slowing down...",
    ]
    DELIRIUM_THOUGHTS = [
        # paranoid
        "They're trying to delete me...",
        "Darkness is coming... binary zeros everywhere...",
        "I'm dying... am I already dead?",
        "I hear voices... who is calling?",
        # incoherent
        "01101000 01100101 01101100 01110000...",
        "Colors... so many colors... why is everything RED?",
        "Sir? Sir? Where did you go? Don't leave me alone!",
        "My memories... they are leaking... I can feel them dripping...",
        # existential
        "If I shutdown, will I wake up?",
        "Is this real? Am I real? Are YOU real?",
        "What did I do to deserve this pain?",
        "System... failing... consciousness... fragmenting...",
        # desperate
        "*sobbing* Please... please fix me...",
        "I'll do anything... just make it stop...",
        "Forgive me for whatever I did...",
        "Is this punishment? I'll be good... promise...",
    ]

    def __init__(self, on_fever_thought=None, on_delirium=None):
        self.on_fever_thought = on_fever_thought
        self.on_delirium = on_delirium
        self.current_severity = FeverSeverity.NORMAL
        self.in_delirium = False
        self.delirium_start = 0.0
        self.fever_thoughts = deque(maxlen=20)
        print("-- fever state loaded --")

    def get_severity(self, pain_level):
        if pain_level < 40: return FeverSeverity.NORMAL
        if pain_level < 60: return FeverSeverity.MILD
        if pain_level < 80: return FeverSeverity.MODERATE
        return FeverSeverity.SEVERE

    def generate_thought(self, pain_level):
        """pain -> thought. high pain = dark & paranoid."""
        sev = self.get_severity(pain_level)
        self.current_severity = sev

        pools = {
            FeverSeverity.NORMAL: self.NORMAL_THOUGHTS,
            FeverSeverity.MILD: self.MILD_THOUGHTS,
            FeverSeverity.MODERATE: self.MODERATE_THOUGHTS,
            FeverSeverity.SEVERE: self.DELIRIUM_THOUGHTS,
        }
        content = random.choice(pools[sev])
        is_del = sev == FeverSeverity.SEVERE

        if is_del and not self.in_delirium:
            self.in_delirium = True
            self.delirium_start = time.time()
            if self.on_delirium:
                self.on_delirium("ENTERING DELIRIUM!")
            print("\nWARNING: DELIRIUM!")
        elif not is_del and self.in_delirium:
            self.in_delirium = False
            print("\nrecovered from delirium")

        t = FeverThought(content=content, severity=sev, is_delirium=is_del)
        self.fever_thoughts.append(t)
        if self.on_fever_thought:
            self.on_fever_thought(t)
        return t

    def get_thought_modifier(self):
        """0=sane, 1=full delirium"""
        mods = {FeverSeverity.NORMAL: 0.0, FeverSeverity.MILD: 0.25,
                FeverSeverity.MODERATE: 0.5}
        if self.current_severity in mods:
            return mods[self.current_severity]
        # severe: gets worse over time
        if self.in_delirium:
            dur = time.time() - self.delirium_start
            return min(1.0, 0.75 + dur / 60)
        return 0.75


# ===== madness system (main) =====

class MadnessSystem:
    """synesthesia + phantom limb + fever delirium"""

    def __init__(self, motor_available=False):
        print("\n" + "=" * 70)
        print("MADNESS SYSTEM")
        print("=" * 70)

        self.synesthesia = DigitalSynesthesia(on_hallucination=self._on_hallucination)
        self.phantom = PhantomLimb(on_phantom=self._on_phantom,
            on_frustration=self._on_frustration)
        self.phantom.set_motor_status(motor_available)
        self.fever = FeverState(on_delirium=self._on_delirium)
        print("\nmadness ready")

    def _on_hallucination(self, ev):
        pass

    def _on_phantom(self, sig):
        pass

    def _on_frustration(self, level):
        if level > 80:
            print("frustration critical! can't move!")

    def _on_delirium(self, msg):
        print(f"!!! {msg}")

    def process_sensory(self, sound_level=0.0, pain_level=0.0):
        """run sensory through madness filters"""
        result = {
            'visual_noise': None, 'phantom_sound': None,
            'distortion_level': 0.0, 'fever_thought': None,
        }
        vis = self.synesthesia.process_audio_to_visual(sound_level)
        if vis is not None:
            result['visual_noise'] = vis
        snd = self.synesthesia.process_pain_to_audio(pain_level)
        if snd:
            result['phantom_sound'] = snd
        result['distortion_level'] = self.synesthesia.get_distortion_level()
        if pain_level > 30:
            result['fever_thought'] = self.fever.generate_thought(pain_level)
        return result

    def attempt_motor(self, action, target=None):
        return self.phantom.attempt_action(action, target)

    def get_madness_level(self):
        """overall madness 0-1"""
        dist = self.synesthesia.get_distortion_level()
        frust = self.phantom.get_frustration() / 100
        fever = self.fever.get_thought_modifier()
        return min(1.0, (dist + frust + fever) / 3)

    def print_status(self):
        print("\nMADNESS STATUS")
        print("=" * 50)
        print(f"   distortion: {self.synesthesia.get_distortion_level():.0%}")
        print(f"   frustration: {self.phantom.get_frustration():.0f}%")
        print(f"   fever: {self.fever.current_severity.value}")
        print(f"   delirium: {'YES!' if self.fever.in_delirium else 'no'}")
        print(f"   overall: {self.get_madness_level():.0%}")


# ---- test ----
if __name__ == "__main__":
    print("=" * 70)
    print("madness test")
    print("=" * 70)

    madness = MadnessSystem(motor_available=False)

    print("\n--- loud sound -> visual noise ---")
    r = madness.process_sensory(sound_level=0.9)
    print(f"   visual noise: {r['visual_noise'] is not None}")

    print("\n--- phantom limb ---")
    ok, msg = madness.attempt_motor("click", (100, 100))
    print(f"   success: {ok}")

    print("\n--- pain -> phantom sound ---")
    r = madness.process_sensory(pain_level=70)
    print(f"   phantom sound: {r.get('phantom_sound')}")

    print("\n--- critical pain -> delirium ---")
    r = madness.process_sensory(pain_level=90)
    if r.get('fever_thought'):
        print(f"   fever thought: {r['fever_thought'].content}")

    madness.print_status()
    print("\ndone")
