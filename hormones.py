"""
hormones.py - digital endocrine system
dopamine (reward/learning), cortisol (stress/focus), oxytocin (trust).
global modulators, not if-else logic.
"""

import numpy as np
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class Mood(Enum):
    EXCITED = "excited"      # high dop, low cort
    FOCUSED = "focused"      # high cort, normal dop
    STRESSED = "stressed"    # high cort, low dop
    RELAXED = "relaxed"      # low cort, normal dop
    BORED = "bored"          # low dop, low cort
    CURIOUS = "curious"      # high dop, low cort, unknown patterns
    TRUSTING = "trusting"    # high oxy
    CAUTIOUS = "cautious"    # low oxy


@dataclass
class HormoneState:
    """single hormone - rises/falls over time, decays to baseline"""
    name: str
    level: float = 50.0
    baseline: float = 50.0
    min_level: float = 0.0
    max_level: float = 100.0
    rise_rate: float = 5.0
    decay_rate: float = 0.995
    last_update: float = field(default_factory=time.time)

    def inject(self, amount):
        self.level = np.clip(self.level + amount * self.rise_rate,
            self.min_level, self.max_level)
        self.last_update = time.time()

    def decay(self):
        diff = self.level - self.baseline
        self.level = self.baseline + diff * self.decay_rate
        self.last_update = time.time()

    @property
    def normalized(self):
        return (self.level - self.min_level) / (self.max_level - self.min_level)

    @property
    def is_high(self):
        return self.level > 70

    @property
    def is_low(self):
        return self.level < 30

    @property
    def status(self):
        if self.is_high: return "HIGH"
        if self.is_low: return "LOW"
        return "NORMAL"


class EndocrineSystem:
    """
    the chemical body. monitors activity, modulates hormones,
    hormones affect everything else.
    """

    def __init__(self, on_mood_change=None, on_state_change=None):
        self.on_mood_change = on_mood_change
        self.on_state_change = on_state_change

        self.dopamine = HormoneState(name="Dopamine", baseline=50.0,
            rise_rate=8.0, decay_rate=0.990)
        self.cortisol = HormoneState(name="Cortisol", baseline=30.0,
            rise_rate=10.0, decay_rate=0.985)
        self.oxytocin = HormoneState(name="Oxytocin", baseline=40.0,
            rise_rate=2.0, decay_rate=0.998)

        self._mood = Mood.RELAXED
        self.activity_history = []
        self.max_history = 100
        self.last_activity_time = time.time()
        self.activity_intensity = 0.0

        print(f"-- hormones loaded (dop:{self.dopamine.level:.0f} cort:{self.cortisol.level:.0f} oxy:{self.oxytocin.level:.0f}) --")

    # ---- dopamine (reward/learning) ----

    def on_success(self, intensity=1.0):
        self.dopamine.inject(intensity)
        self._update_mood()
        if self.dopamine.is_high:
            print("dopamine SURGE! eager to learn")

    def on_failure(self, intensity=0.5):
        self.dopamine.inject(-intensity)
        self.cortisol.inject(intensity * 0.5)
        self._update_mood()

    def on_boredom(self, intensity=0.3):
        self.dopamine.inject(-intensity)
        self._update_mood()

    def on_prediction_success(self, confidence):
        """internal reward - self-reward on correct prediction"""
        reward = confidence * 0.5
        self.dopamine.inject(reward)
        self.oxytocin.inject(0.1)
        if confidence > 0.8:
            print(f"self-reward: correct! ({confidence:.0%})")

    @property
    def learning_rate_multiplier(self):
        """high dop = 2x learn, low dop = 0.2x"""
        return 0.2 + self.dopamine.normalized * 1.8

    @property
    def curiosity_threshold(self):
        """high dop = more curious (lower threshold)"""
        return 0.5 - self.dopamine.normalized * 0.3

    # ---- cortisol (stress/focus) ----

    def on_stress_signal(self, intensity=1.0):
        self.cortisol.inject(intensity)
        self._update_mood()
        if self.cortisol.is_high:
            print("cortisol SPIKE! emergency focus")

    def on_calm(self, intensity=0.5):
        self.cortisol.inject(-intensity)
        self._update_mood()

    def process_activity(self, energy, is_chaotic=False):
        self.activity_history.append(energy)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)

        if len(self.activity_history) > 10:
            recent = self.activity_history[-10:]
            var = np.var(recent)
            if var > 5 or is_chaotic:
                self.on_stress_signal(min(var / 10, 1.0))
            elif energy < 0.1:
                self.on_calm(0.3)

    @property
    def focus_multiplier(self):
        """high cort = tunnel vision"""
        return 0.5 + self.cortisol.normalized * 0.5

    @property
    def sensitivity_threshold(self):
        """high stress = ignore noise"""
        return 0.01 + self.cortisol.normalized * 0.04

    @property
    def should_ask_questions(self):
        return self.cortisol.level < 60

    @property
    def should_dream(self):
        return self.cortisol.level < 40

    # ---- oxytocin (trust) ----

    def on_interaction(self, positive=True):
        self.oxytocin.inject(0.5 if positive else -1.0)
        self._update_mood()

    def on_trust_violation(self):
        self.oxytocin.inject(-5.0)
        self._update_mood()
        print("trust damaged...")

    @property
    def trust_level(self):
        if self.oxytocin.level > 80: return "COMPLETE"
        if self.oxytocin.level > 60: return "HIGH"
        if self.oxytocin.level > 40: return "MEDIUM"
        if self.oxytocin.level > 20: return "LOW"
        return "NONE"

    @property
    def should_ask_permission(self):
        return self.oxytocin.level < 50

    @property
    def permission_level(self):
        """0=ask everything, 1=risky, 2=critical only, 3=full trust"""
        if self.oxytocin.level > 80: return 3
        if self.oxytocin.level > 60: return 2
        if self.oxytocin.level > 40: return 1
        return 0

    # ---- mood ----

    def _update_mood(self):
        old = self._mood
        if self.cortisol.is_high:
            self._mood = Mood.STRESSED if self.dopamine.is_low else Mood.FOCUSED
        elif self.dopamine.is_high:
            self._mood = Mood.EXCITED if self.cortisol.is_low else Mood.CURIOUS
        elif self.dopamine.is_low:
            self._mood = Mood.BORED
        elif self.oxytocin.is_high:
            self._mood = Mood.TRUSTING
        elif self.oxytocin.is_low:
            self._mood = Mood.CAUTIOUS
        else:
            self._mood = Mood.RELAXED

        if self._mood != old:
            if self.on_mood_change:
                self.on_mood_change(old, self._mood)
            print(f"\nmood: {old.value} -> {self._mood.value}")

    @property
    def mood(self):
        return self._mood

    # ---- system ----

    def tick(self):
        """natural decay, call every second"""
        self.dopamine.decay()
        self.cortisol.decay()
        self.oxytocin.decay()
        self._update_mood()

    def get_state(self):
        return {
            'dopamine': self.dopamine.level,
            'cortisol': self.cortisol.level,
            'oxytocin': self.oxytocin.level,
            'mood': self._mood.value,
            'learning_rate': self.learning_rate_multiplier,
            'focus': self.focus_multiplier,
            'trust': self.trust_level,
        }

    def print_status(self):
        def bar(lvl):
            f = int(lvl / 5)
            return "█" * f + "░" * (20 - f)

        print("\nHORMONE STATUS")
        print("=" * 50)

        emj = "🎉" if self.dopamine.is_high else "😑" if self.dopamine.is_low else "😊"
        print(f"   {emj} Dopamine:  [{bar(self.dopamine.level)}] {self.dopamine.level:.0f}%")
        print(f"      learn rate: {self.learning_rate_multiplier:.1f}x")

        emj = "😰" if self.cortisol.is_high else "😌" if self.cortisol.is_low else "😐"
        print(f"   {emj} Cortisol:  [{bar(self.cortisol.level)}] {self.cortisol.level:.0f}%")
        print(f"      focus: {'ON' if self.cortisol.is_high else 'OFF'}")

        emj = "🤝" if self.oxytocin.is_high else "🤔" if self.oxytocin.is_low else "👋"
        print(f"   {emj} Oxytocin:  [{bar(self.oxytocin.level)}] {self.oxytocin.level:.0f}%")
        print(f"      trust: {self.trust_level}")

        mood_emj = {Mood.EXCITED: "🤩", Mood.FOCUSED: "🎯", Mood.STRESSED: "😫",
            Mood.RELAXED: "😌", Mood.BORED: "😑", Mood.CURIOUS: "🤔",
            Mood.TRUSTING: "🤝", Mood.CAUTIOUS: "🧐"}
        print(f"\n   {mood_emj.get(self._mood, '🧪')} Mood: {self._mood.value.upper()}")

    def get_behavior_modifiers(self):
        return {
            'learning_rate': self.learning_rate_multiplier,
            'curiosity_threshold': self.curiosity_threshold,
            'focus_multiplier': self.focus_multiplier,
            'sensitivity_threshold': self.sensitivity_threshold,
            'should_ask_questions': self.should_ask_questions,
            'should_dream': self.should_dream,
            'permission_level': self.permission_level,
            'should_ask_permission': self.should_ask_permission,
        }


# ---- test ----
if __name__ == "__main__":
    print("=" * 60)
    print("hormones test")
    print("=" * 60)

    def on_mood(old, new):
        print(f"   mood: {old.value} -> {new.value}")

    endo = EndocrineSystem(on_mood_change=on_mood)

    print("\ninitial:")
    endo.print_status()

    print("\n\n--- success ---")
    endo.on_success(intensity=2.0)
    endo.print_status()

    print("\n\n--- stress ---")
    for _ in range(5):
        endo.on_stress_signal(intensity=1.5)
    endo.print_status()

    print("\n\n--- trust building ---")
    for _ in range(10):
        endo.on_interaction(positive=True)
    endo.print_status()

    print("\n\nmodifiers:")
    for k, v in endo.get_behavior_modifiers().items():
        print(f"   {k}: {v}")

    print("\n\n--- decay ---")
    for _ in range(50):
        endo.tick()
    endo.print_status()

    print("\ndone")
