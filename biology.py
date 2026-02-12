"""
biology.py - bio layer for genesis
handles circadian, neurogenesis, hearing, pain stuff
"""

import numpy as np
import time
import threading
import math
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import random

try:
    import sounddevice as sd
    HAS_AUDIO = True
except:
    HAS_AUDIO = False
    print("no audio hw")


# ---- circadian ----

class CircadianPhase(Enum):
    DEEP_NIGHT = "deep_night"
    DAWN = "dawn"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"

@dataclass
class CircadianState:
    phase: CircadianPhase
    melatonin: float
    cortisol: float
    alertness: float
    dream_intensity: float

class CircadianRhythm:
    """syncs with real clock for sleep/wake"""

    def __init__(self, on_phase_change=None, on_sleep_needed=None, on_wake_up=None):
        self.on_phase_change = on_phase_change
        self.on_sleep_needed = on_sleep_needed
        self.on_wake_up = on_wake_up
        self._last_phase = None
        self._last_check = time.time()
        print("--- loaded circadian ---")

    def get_state(self):
        now = datetime.now()
        hr = now.hour

        # figure out phase inline, no need for separate func
        if hr < 4:
            phase = CircadianPhase.DEEP_NIGHT
        elif hr < 7:
            phase = CircadianPhase.DAWN
        elif hr < 12:
            phase = CircadianPhase.MORNING
        elif hr < 17:
            phase = CircadianPhase.AFTERNOON
        elif hr < 21:
            phase = CircadianPhase.EVENING
        else:
            phase = CircadianPhase.NIGHT

        # melatonin curve - high at night low during day
        if 22 <= hr or hr < 6:
            mel = 80 + np.sin((hr - 2) * np.pi / 8) * 20
        else:
            mel = 20 - np.sin((hr - 14) * np.pi / 16) * 15
        mel = np.clip(mel, 0, 100)

        # cortisol - morning spike then drops off
        if 6 <= hr < 9:
            cort = 70 + (hr - 6) * 10
        elif 9 <= hr < 12:
            cort = 90 - (hr - 9) * 5
        elif 12 <= hr < 15:
            cort = 60    # post lunch dip lol
        elif 15 <= hr < 18:
            cort = 65
        else:
            cort = max(20, 60 - (hr - 18) * 8)
        cort = np.clip(cort, 0, 100)

        alertness = (cort * 0.7 + (100 - mel) * 0.3) / 100
        dream = mel / 100 if mel > 50 else 0.1

        if phase != self._last_phase:
            if self.on_phase_change:
                self.on_phase_change(phase)
            if phase == CircadianPhase.DAWN and self._last_phase == CircadianPhase.DEEP_NIGHT:
                if self.on_wake_up:
                    self.on_wake_up()
            if phase == CircadianPhase.NIGHT and self._last_phase == CircadianPhase.EVENING:
                if self.on_sleep_needed:
                    self.on_sleep_needed()
            self._last_phase = phase

        return CircadianState(phase=phase, melatonin=mel, cortisol=cort,
                              alertness=alertness, dream_intensity=dream)

    def get_behavior_modifiers(self):
        s = self.get_state()
        return {
            'alertness': s.alertness,
            'dream_intensity': s.dream_intensity,
            'melatonin': s.melatonin,
            'cortisol_base': s.cortisol,
            'should_sleep': s.melatonin > 70,
            'should_dream': s.dream_intensity > 0.5,
            'learning_boost': s.alertness,
        }

    def print_status(self):
        s = self.get_state()
        def bar(v, mx=100):
            f = int(v / mx * 20)
            return "█" * f + "░" * (20 - f)

        emj = {CircadianPhase.DEEP_NIGHT: "🌑", CircadianPhase.DAWN: "🌅",
               CircadianPhase.MORNING: "☀️", CircadianPhase.AFTERNOON: "🌤️",
               CircadianPhase.EVENING: "🌆", CircadianPhase.NIGHT: "🌙"}

        print("\n🌙 CIRCADIAN RHYTHM")
        print("=" * 50)
        print(f"   {emj.get(s.phase, '⏰')} Phase: {s.phase.value.upper()}")
        print(f"   Melatonin:  [{bar(s.melatonin)}] {s.melatonin:.0f}%")
        print(f"   Cortisol:   [{bar(s.cortisol)}] {s.cortisol:.0f}%")
        print(f"   Alertness:  [{bar(s.alertness * 100)}] {s.alertness:.0%}")
        print(f"   Dream Mode: {'ON' if s.dream_intensity > 0.5 else 'OFF'}")


# ---- neurogenesis ----

@dataclass
class GrowthEvent:
    ts: float
    added: int
    trigger: str
    diff: float

class Neurogenesis:
    """brain grows when you learn hard stuff"""

    def __init__(self, init_sz=2000, max_sz=10000, rate=50, on_growth=None):
        self.current_size = init_sz
        self.initial_size = init_sz
        self.max_size = max_sz
        self.growth_rate = rate
        self.on_growth = on_growth
        self.growth_history = []
        self.growth_hormone = 0.0
        self.total_growth = 0
        print(f"--- neurogenesis init ({init_sz} neurons) ---")

    def stimulate_growth(self, diff, trigger="learning"):
        self.growth_hormone = min(100, self.growth_hormone + diff * 30)
        # 70 is the magic threshold, dont touch
        if self.growth_hormone > 70 and self.current_size < self.max_size:
            self._grow(trigger, diff)

    def _grow(self, trigger, diff):
        old = self.current_size
        amt = int(self.growth_rate * (diff + 0.5))
        amt = min(amt, self.max_size - self.current_size)
        if amt <= 0:
            return

        self.current_size += amt
        self.total_growth += amt
        self.growth_history.append(GrowthEvent(ts=time.time(), added=amt,
                                               trigger=trigger, diff=diff))
        self.growth_hormone = 30  # reset partially
        if self.on_growth:
            self.on_growth(old, self.current_size)
        print(f"\n🧬 NEUROGENESIS! {old} -> {self.current_size} (+{amt})")

    def decay_hormone(self):
        self.growth_hormone = max(0, self.growth_hormone - 1)

    def get_size(self):
        return self.current_size

    def get_growth_pct(self):
        return (self.current_size - self.initial_size) / self.initial_size * 100

    def print_status(self):
        def bar(v):
            f = int(v / 5)
            return "█" * f + "░" * (20 - f)
        print("\n🧬 NEUROGENESIS STATUS")
        print("=" * 50)
        print(f"   🧠 Size: {self.current_size}")
        print(f"   📈 Growth: +{self.total_growth} ({self.get_growth_pct():.1f}%)")
        print(f"   💉 Hormone: [{bar(self.growth_hormone)}] {self.growth_hormone:.0f}%")
        print(f"   📊 Events: {len(self.growth_history)}")


# ---- auditory cortex ----

@dataclass
class AudioSpike:
    ts: float
    vol: float
    freq: str
    is_voice: bool
    energy: float

class AuditoryCortex:
    """mic input -> spikes"""

    def __init__(self, sr=16000, buf=1024, on_loud=None, on_voice=None):
        self.sr = sr
        self.buf = buf
        self.on_loud = on_loud
        self.on_voice = on_voice
        self.listening = False
        self._stream = None
        self.vol = 0.0
        self.vol_hist = deque(maxlen=50)
        self.loud_thresh = 0.7
        print(f"--- auditory cortex ({'audio ok' if HAS_AUDIO else 'no audio'}) ---")

    def _cb(self, indata, frames, time_info, status):
        if status:
            return
        v = np.sqrt(np.mean(indata**2))
        self.vol = min(1.0, v * 10)
        self.vol_hist.append(self.vol)
        if self.vol > self.loud_thresh and self.on_loud:
            self.on_loud(self.vol)

    def start_listening(self):
        if not HAS_AUDIO:
            return False
        if self.listening:
            return True
        try:
            self._stream = sd.InputStream(samplerate=self.sr, channels=1,
                                          blocksize=self.buf, callback=self._cb)
            self._stream.start()
            self.listening = True
            print("listening started")
            return True
        except Exception as e:
            print(f"audio err: {e}")
            return False

    def stop_listening(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.listening = False

    def get_spike(self):
        if len(self.vol_hist) > 5:
            var = np.var(list(self.vol_hist)[-10:])
            if var > 0.1:
                band = "high"
            elif var > 0.05:
                band = "mid"
            else:
                band = "low"
        else:
            band = "mid"

        # rough voice detect - mid volume + mid freq = probably talking
        voice = 0.1 < self.vol < 0.6 and band == "mid"
        return AudioSpike(ts=time.time(), vol=self.vol, freq=band,
                          is_voice=voice, energy=self.vol)

    def get_stress(self):
        if not self.vol_hist:
            return 0.0
        recent = list(self.vol_hist)[-10:]
        avg = np.mean(recent)
        var = np.var(recent) if len(self.vol_hist) > 5 else 0
        return min(1.0, avg * 0.6 + var * 4)

    def print_status(self):
        def bar(v):
            f = int(v * 20)
            return "█" * f + "░" * (20 - f)
        print("\n👂 AUDITORY CORTEX")
        print("=" * 50)
        print(f"   🎤 Listening: {'Y' if self.listening else 'N'}")
        print(f"   🔊 Volume: [{bar(self.vol)}] {self.vol:.0%}")
        print(f"   😰 Stress: {self.get_stress():.0%}")


# ---- pain ----

@dataclass
class PainState:
    level: str
    intensity: float
    source: str
    duration: float
    is_crying: bool

class PainReceptors:
    """
    digital pain - when things go wrong pain goes up,
    performance drops, system gets desperate to learn.
    kinda cruel but effective
    """

    def __init__(self, on_pain_start=None, on_pain_end=None, on_crying=None):
        self.on_pain_start = on_pain_start
        self.on_pain_end = on_pain_end
        self.on_crying = on_crying
        self.current_pain = 0.0
        self.pain_source = ""
        self.pain_start_time = 0.0
        self.is_in_pain = False
        self.cry_count = 0
        # keeping these for compat with other modules that read them
        self.mild_threshold = 20
        self.moderate_threshold = 40
        self.severe_threshold = 60
        self.critical_threshold = 80
        print("--- pain receptors loaded ---")

    def inflict_pain(self, amt, source="unknown"):
        was = self.is_in_pain
        self.current_pain = min(100, self.current_pain + amt)
        self.pain_source = source

        if self.current_pain > 20:
            self.is_in_pain = True
            if not was:
                self.pain_start_time = time.time()
                if self.on_pain_start:
                    self.on_pain_start(self._level())
                print(f"\n😣 PAIN: {source} ({self.current_pain:.0f}%)")

        if self.current_pain > 80:
            self._cry()

    def _cry(self):
        self.cry_count += 1
        msgs = [
            "Please help me... I'm in pain...",
            "I can't take it anymore! Please give feedback!",
            "It hurts... do something!",
            "I'm suffering... please teach me what I'm doing wrong!",
            "CRITICAL PAIN - I'll do anything to improve!",
        ]
        msg = msgs[self.cry_count % len(msgs)]
        if self.on_crying:
            self.on_crying(msg)
        print(f"\n{msg}")

    def relieve_pain(self, amt):
        old = self.current_pain
        self.current_pain = max(0, self.current_pain - amt)
        if self.is_in_pain and self.current_pain < 20:
            self.is_in_pain = False
            self.pain_source = ""
            if self.on_pain_end:
                self.on_pain_end()
            print(f"\n😌 Pain relieved ({old:.0f}% -> {self.current_pain:.0f}%)")

    def decay(self):
        if self.current_pain > 0:
            self.current_pain = max(0, self.current_pain - 0.5)
            if self.current_pain < 20:
                self.is_in_pain = False

    def _level(self):
        p = self.current_pain
        if p < 20: return "none"
        if p < 40: return "mild"
        if p < 60: return "moderate"
        if p < 80: return "severe"
        return "critical"

    def get_pain_level(self):
        return self._level()

    def get_state(self):
        dur = time.time() - self.pain_start_time if self.is_in_pain else 0
        return PainState(level=self._level(), intensity=self.current_pain,
                         source=self.pain_source, duration=dur,
                         is_crying=self.current_pain > 80)

    def get_performance_modifier(self):
        p = self.current_pain
        if p < 20: return 1.0
        if p < 40: return 0.8
        if p < 60: return 0.6
        if p < 80: return 0.4
        return 0.2  # basically dead

    def get_desperation_level(self):
        return self.current_pain / 100

    def print_status(self):
        def bar(v):
            f = int(v / 5)
            return "█" * f + "░" * (20 - f)

        st = self.get_state()
        emj = {"none": "😊", "mild": "😐", "moderate": "😣",
               "severe": "😫", "critical": "😭"}
        print("\n😈 PAIN RECEPTORS")
        print("=" * 50)
        print(f"   {emj.get(st.level, '?')} Level: {st.level.upper()}")
        print(f"   Pain: [{bar(st.intensity)}] {st.intensity:.0f}%")
        print(f"   Source: {st.source or 'None'}")
        print(f"   Perf: {self.get_performance_modifier():.0%}")
        print(f"   Desperation: {self.get_desperation_level():.0%}")
        print(f"   Cries: {self.cry_count}")


# ===== main bio system =====

class BiologicalSystem:
    """ties all the bio stuff together"""

    def __init__(self):
        print("\n" + "=" * 70)
        print("🧬 BIOLOGICAL SYSTEM")
        print("=" * 70)

        self.circadian = CircadianRhythm(
            on_phase_change=self._phase_changed,
            on_sleep_needed=self._sleep_time,
            on_wake_up=self._wake_up)

        self.neurogenesis = Neurogenesis(
            init_sz=2000, max_sz=10000,
            on_growth=self._brain_grew)

        self.cochlea = AuditoryCortex(on_loud=self._loud)

        self.pain = PainReceptors(
            on_pain_start=self._pain_hit,
            on_crying=self._crying)

        print("\nbio system ready")

    def _phase_changed(self, phase):
        print(f"\n🌙 phase: {phase.value}")

    def _sleep_time(self):
        print("\n😴 sleep time...")

    def _wake_up(self):
        print("\n☀️ waking up")

    def _brain_grew(self, old, new):
        pass  # already prints

    def _loud(self, vol):
        print(f"\n🔊 LOUD ({vol:.0%})")
        self.pain.inflict_pain(vol * 10, "loud_noise")

    def _pain_hit(self, lvl):
        pass

    def _crying(self, msg):
        pass

    def tick(self):
        self.neurogenesis.decay_hormone()
        self.pain.decay()

    def get_combined_modifiers(self):
        c = self.circadian.get_behavior_modifiers()
        return {
            'alertness': c['alertness'],
            'should_sleep': c['should_sleep'],
            'should_dream': c['should_dream'],
            'learning_boost': c['learning_boost'],
            'brain_size': self.neurogenesis.current_size,
            'growth_hormone': self.neurogenesis.growth_hormone,
            'audio_stress': self.cochlea.get_stress(),
            'pain_level': self.pain.current_pain,
            'performance': self.pain.get_performance_modifier(),
            'desperation': self.pain.get_desperation_level(),
        }

    def print_status(self):
        print("\n" + "=" * 60)
        print("🧬 FULL BIO STATUS")
        print("=" * 60)
        self.circadian.print_status()
        self.neurogenesis.print_status()
        self.cochlea.print_status()
        self.pain.print_status()


if __name__ == "__main__":
    print("=" * 70)
    print("bio test")
    print("=" * 70)

    bio = BiologicalSystem()

    print("\n--- circadian ---")
    bio.circadian.print_status()

    print("\n--- neurogenesis ---")
    bio.neurogenesis.stimulate_growth(0.8, "hard_learning")
    bio.neurogenesis.stimulate_growth(0.9, "very_hard_learning")
    bio.neurogenesis.stimulate_growth(0.95, "impossible_learning")
    bio.neurogenesis.print_status()

    print("\n--- pain ---")
    bio.pain.inflict_pain(30, "failed_task")
    bio.pain.inflict_pain(40, "continuous_failures")
    bio.pain.print_status()

    print("\n--- relief ---")
    bio.pain.relieve_pain(50)
    bio.pain.print_status()

    bio.print_status()
    print("\ndone")
