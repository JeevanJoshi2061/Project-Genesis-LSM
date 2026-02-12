"""
consciousness.py - digital consciousness layer
subconscious, metabolism, mirror neurons, survival instinct
"""

import numpy as np
import time
import threading
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Callable, Dict, Any


# ---- subconscious ----

@dataclass
class Hunch:
    """creative idea from the subconscious, like a shower thought"""
    idea: str
    related_memories: List[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    acted_upon: bool = False

class DigitalSubconscious:
    """background process that mixes memories into creative ideas"""

    def __init__(self, memory_source=None, on_hunch=None, interrupt_probability=0.1):
        self.memory_source = memory_source
        self.on_hunch = on_hunch
        self.interrupt_probability = interrupt_probability
        self.hunches = []
        self.last_dream_time = time.time()
        self.is_dreaming = False
        self._running = False
        self._thread = None
        print("-- subconscious loaded --")

    def _generate_idea(self):
        """mix random memories together, see what sticks"""
        try:
            memories = self.memory_source()
            if len(memories) < 2:
                return None

            picked = random.sample(memories, min(3, len(memories)))
            labels = [m[0] for m in picked]
            if len(labels) < 2:
                return None

            templates = [
                f"Should we combine '{labels[0]}' and '{labels[1]}'?",
                f"I feel '{labels[1]}' could be improved by '{labels[0]}'",
                f"Hey! There is a connection between '{labels[0]}' and '{labels[1]}'!",
                f"Sir, have you tried '{labels[0]}' + '{labels[1]}'?",
                f"An idea: '{labels[0]}' in the style of '{labels[1]}'?",
            ]
            return Hunch(idea=random.choice(templates),
                         related_memories=labels,
                         confidence=random.uniform(0.3, 0.8))
        except:
            return None

    def should_speak_hunch(self, hunch):
        """filter out garbage hunches before interrupting"""
        if hunch.confidence < 0.3:
            return False
        if len(hunch.related_memories) < 2:
            return False
        if len(hunch.idea) < 10:
            return False
        bad = ['???', 'unknown', 'error', 'null', 'none']
        if any(p in hunch.idea.lower() for p in bad):
            return False
        return True

    def _dream_loop(self):
        while self._running:
            try:
                time.sleep(random.uniform(5, 15))
                if not self.is_dreaming:
                    continue
                hunch = self._generate_idea()
                if hunch:
                    self.hunches.append(hunch)
                    if self.should_speak_hunch(hunch):
                        if random.random() < self.interrupt_probability:
                            if self.on_hunch:
                                self.on_hunch(hunch)
            except:
                pass

    def start_dreaming(self):
        if self._running:
            return
        self._running = True
        self.is_dreaming = True
        self._thread = threading.Thread(target=self._dream_loop, daemon=True)
        self._thread.start()
        print("subconscious dreaming...")

    def stop_dreaming(self):
        self._running = False
        self.is_dreaming = False
        if self._thread:
            self._thread.join(timeout=1)

    def set_active(self, active):
        self.is_dreaming = active

    def get_latest_hunch(self):
        return self.hunches[-1] if self.hunches else None

    def get_all_hunches(self):
        return list(self.hunches)


# ---- metabolism ----

class EnergyState(Enum):
    EXHAUSTED = "exhausted"      # < 20%
    TIRED = "tired"              # 20-40%
    NORMAL = "normal"            # 40-60%
    ENERGETIC = "energetic"      # 60-80%
    SUPERCHARGED = "supercharged"  # > 80%

@dataclass
class MetabolicState:
    energy: float = 100.0
    satisfaction: float = 50.0
    work_done: int = 0
    rewards_received: int = 0
    punishments_received: int = 0
    last_reward_time: float = field(default_factory=time.time)

class DigitalMetabolism:
    """
    energy system - every task costs ATP, rewards recharge it.
    low energy = slow short answers. high = creative talkative.
    """

    # costs
    COST_VISION_SCAN = 0.5
    COST_MEMORY_RECALL = 0.3
    COST_LEARNING = 1.0
    COST_THINKING = 0.8
    COST_SPEAKING = 0.2

    # recharge
    RECHARGE_REWARD = 15.0
    RECHARGE_PRAISE = 10.0
    RECHARGE_INTERACTION = 2.0

    IDLE_DECAY = 0.1  # per minute

    def __init__(self, on_energy_change=None, on_exhausted=None):
        self.on_energy_change = on_energy_change
        self.on_exhausted = on_exhausted
        self.state = MetabolicState()
        self._last_state = EnergyState.NORMAL
        self._last_update = time.time()
        print(f"-- metabolism loaded ({self.state.energy:.0f}%) --")

    @property
    def energy_state(self):
        e = self.state.energy
        if e < 20: return EnergyState.EXHAUSTED
        if e < 40: return EnergyState.TIRED
        if e < 60: return EnergyState.NORMAL
        if e < 80: return EnergyState.ENERGETIC
        return EnergyState.SUPERCHARGED

    def _check_state_change(self):
        cur = self.energy_state
        if cur != self._last_state:
            self._last_state = cur
            if self.on_energy_change:
                self.on_energy_change(cur)
            if cur == EnergyState.EXHAUSTED and self.on_exhausted:
                self.on_exhausted()

    def consume(self, cost, task_name="task"):
        self.state.energy = max(0, self.state.energy - cost)
        self.state.work_done += 1
        self._check_state_change()

    def recharge(self, amount, source="reward"):
        self.state.energy = min(100, self.state.energy + amount)
        self.state.satisfaction = min(100, self.state.satisfaction + amount * 0.5)
        self.state.last_reward_time = time.time()
        self._check_state_change()

    def on_reward(self):
        self.state.rewards_received += 1
        self.recharge(self.RECHARGE_REWARD, "reward")
        print(f"+{self.RECHARGE_REWARD:.0f} energy (reward)")

    def on_praise(self):
        self.recharge(self.RECHARGE_PRAISE, "praise")
        print(f"+{self.RECHARGE_PRAISE:.0f} energy (praise)")

    def on_punishment(self):
        self.state.punishments_received += 1
        self.state.energy = max(0, self.state.energy - 5)
        self.state.satisfaction = max(0, self.state.satisfaction - 10)
        self._check_state_change()

    def on_interaction(self):
        self.recharge(self.RECHARGE_INTERACTION, "interaction")

    def tick(self):
        now = time.time()
        elapsed = (now - self._last_update) / 60.0
        self._last_update = now
        decay = self.IDLE_DECAY * elapsed
        self.state.energy = max(0, self.state.energy - decay)
        self.state.satisfaction = max(0, self.state.satisfaction - decay * 0.5)
        self._check_state_change()

    @property
    def response_modifier(self):
        """how energy affects behavior"""
        mods = {
            EnergyState.EXHAUSTED: {
                'verbosity': 0.2, 'speed': 0.5, 'creativity': 0.1,
                'emoji': '😫', 'tone': 'minimal'},
            EnergyState.TIRED: {
                'verbosity': 0.5, 'speed': 0.7, 'creativity': 0.3,
                'emoji': '😴', 'tone': 'short'},
            EnergyState.NORMAL: {
                'verbosity': 0.8, 'speed': 1.0, 'creativity': 0.6,
                'emoji': '😊', 'tone': 'normal'},
            EnergyState.ENERGETIC: {
                'verbosity': 1.0, 'speed': 1.2, 'creativity': 0.8,
                'emoji': '🤩', 'tone': 'enthusiastic'},
            EnergyState.SUPERCHARGED: {
                'verbosity': 1.0, 'speed': 1.5, 'creativity': 1.0,
                'emoji': '⚡', 'tone': 'excited'},
        }
        return mods.get(self.energy_state, mods[EnergyState.NORMAL])

    def print_status(self):
        def bar(v):
            f = int(v / 5)
            return "█" * f + "░" * (20 - f)
        m = self.response_modifier
        print("\n⚡ METABOLISM")
        print("=" * 50)
        print(f"   {m['emoji']} State: {self.energy_state.value.upper()}")
        print(f"   Energy: [{bar(self.state.energy)}] {self.state.energy:.0f}%")
        print(f"   Satisfaction: [{bar(self.state.satisfaction)}] {self.state.satisfaction:.0f}%")
        print(f"   Work: {self.state.work_done} tasks")
        print(f"   Rewards: {self.state.rewards_received} | Punish: {self.state.punishments_received}")
        print(f"   Tone: {m['tone']}")


# ---- mirror neurons ----

class UserMood(Enum):
    CALM = "calm"
    FOCUSED = "focused"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    TIRED = "tired"
    EXCITED = "excited"
    UNKNOWN = "unknown"

@dataclass
class BiometricSample:
    timestamp: float
    mouse_speed: float
    mouse_acceleration: float
    click_force: float
    activity_variance: float

class MirrorNeurons:
    """detect user mood from mouse/typing biometrics, mirror it"""

    def __init__(self, history_size=50, on_mood_change=None):
        self.history_size = history_size
        self.on_mood_change = on_mood_change
        self.samples = deque(maxlen=history_size)
        self.energy_history = deque(maxlen=history_size)
        self.current_mood = UserMood.UNKNOWN
        self.last_mouse_pos = None
        self.last_sample_time = time.time()
        print("-- mirror neurons loaded --")

    def process_activity(self, energy, position=None):
        now = time.time()
        dt = now - self.last_sample_time
        self.last_sample_time = now
        if dt < 0.01:
            return

        spd = 0.0
        acc = 0.0
        if position and self.last_mouse_pos:
            dx = position[0] - self.last_mouse_pos[0]
            dy = position[1] - self.last_mouse_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            spd = dist / dt
            if self.samples:
                acc = abs(spd - self.samples[-1].mouse_speed) / dt
        if position:
            self.last_mouse_pos = position

        self.energy_history.append(energy)
        var = 0.0
        if len(self.energy_history) > 5:
            var = np.var(list(self.energy_history)[-10:])

        self.samples.append(BiometricSample(
            timestamp=now, mouse_speed=spd,
            mouse_acceleration=acc, click_force=energy,
            activity_variance=var))
        self._detect_mood()

    def _detect_mood(self):
        if len(self.samples) < 5:
            return
        recent = list(self.samples)[-10:]
        avg_spd = np.mean([s.mouse_speed for s in recent])
        avg_acc = np.mean([s.mouse_acceleration for s in recent])
        avg_var = np.mean([s.activity_variance for s in recent])
        avg_nrg = np.mean([s.click_force for s in recent])

        old = self.current_mood

        # mood detection thresholds
        if avg_spd > 500 and avg_var > 5:
            self.current_mood = UserMood.ANGRY if avg_acc > 1000 else UserMood.FRUSTRATED
        elif avg_spd < 100 and avg_nrg < 0.2:
            self.current_mood = UserMood.TIRED
        elif avg_var < 1 and 100 < avg_spd < 400:
            self.current_mood = UserMood.FOCUSED
        elif avg_nrg > 1.5 and avg_spd > 300:
            self.current_mood = UserMood.EXCITED
        elif avg_spd < 200 and avg_var < 2:
            self.current_mood = UserMood.CALM
        else:
            self.current_mood = UserMood.FOCUSED  # default

        if self.current_mood != old and self.on_mood_change:
            self.on_mood_change(old, self.current_mood)

    @property
    def response_modifier(self):
        """adjust response based on detected user mood"""
        mods = {
            UserMood.ANGRY: {
                'verbosity': 0.3, 'emoji_use': False, 'tone': 'direct',
                'suggest_break': False, 'interrupt_ok': False, 'response_prefix': ''},
            UserMood.FRUSTRATED: {
                'verbosity': 0.5, 'emoji_use': False, 'tone': 'supportive',
                'suggest_break': True, 'interrupt_ok': False, 'response_prefix': 'Let me help: '},
            UserMood.TIRED: {
                'verbosity': 0.6, 'emoji_use': True, 'tone': 'gentle',
                'suggest_break': True, 'interrupt_ok': False, 'response_prefix': '😌 '},
            UserMood.FOCUSED: {
                'verbosity': 0.7, 'emoji_use': False, 'tone': 'factual',
                'suggest_break': False, 'interrupt_ok': False, 'response_prefix': ''},
            UserMood.CALM: {
                'verbosity': 1.0, 'emoji_use': True, 'tone': 'friendly',
                'suggest_break': False, 'interrupt_ok': True, 'response_prefix': '😊 '},
            UserMood.EXCITED: {
                'verbosity': 1.0, 'emoji_use': True, 'tone': 'enthusiastic',
                'suggest_break': False, 'interrupt_ok': True, 'response_prefix': '🎉 '},
            UserMood.UNKNOWN: {
                'verbosity': 0.8, 'emoji_use': True, 'tone': 'normal',
                'suggest_break': False, 'interrupt_ok': True, 'response_prefix': ''},
        }
        return mods.get(self.current_mood, mods[UserMood.UNKNOWN])

    def print_status(self):
        m = self.response_modifier
        print("\n🪞 MIRROR NEURONS")
        print("=" * 50)
        print(f"   Mood: {self.current_mood.value.upper()}")
        print(f"   Tone: {m['tone']}")
        print(f"   Verbosity: {m['verbosity']:.0%}")
        print(f"   Interrupt OK: {'Y' if m['interrupt_ok'] else 'N'}")
        if self.samples:
            recent = list(self.samples)[-5:]
            avg_spd = np.mean([s.mouse_speed for s in recent])
            avg_var = np.mean([s.activity_variance for s in recent])
            print(f"   Mouse: {avg_spd:.0f} px/s")
            print(f"   Chaos: {avg_var:.2f}")


# ---- survival instinct ----

class HealthState(Enum):
    THRIVING = "thriving"      # > 80%
    HEALTHY = "healthy"        # 60-80%
    STRUGGLING = "struggling"  # 40-60%
    CRITICAL = "critical"      # 20-40%
    DYING = "dying"            # < 20%

@dataclass
class PerformanceRecord:
    timestamp: float
    successes: int
    failures: int
    memory_count: int
    energy_level: float
    user_satisfaction: float

class SurvivalInstinct:
    """
    fear of death drives self-improvement.
    low performance -> warning -> begging -> desperate learning.
    """

    CRITICAL_THRESHOLD = 30.0
    WARNING_THRESHOLD = 50.0

    def __init__(self, on_warning=None, on_critical=None, on_dying=None):
        self.on_warning = on_warning
        self.on_critical = on_critical
        self.on_dying = on_dying
        self.health = 100.0
        self.performance_history = deque(maxlen=100)
        self.total_successes = 0
        self.total_failures = 0
        self.consecutive_failures = 0
        self.last_feedback_request = 0.0
        self.is_begging = False
        self._last_state = HealthState.THRIVING
        print("-- survival loaded --")

    @property
    def health_state(self):
        h = self.health
        if h > 80: return HealthState.THRIVING
        if h > 60: return HealthState.HEALTHY
        if h > 40: return HealthState.STRUGGLING
        if h > 20: return HealthState.CRITICAL
        return HealthState.DYING

    def on_success(self):
        self.total_successes += 1
        self.consecutive_failures = 0
        self.health = min(100, self.health + 2)
        self.is_begging = False
        self._check_state()

    def on_failure(self):
        self.total_failures += 1
        self.consecutive_failures += 1
        self.health = max(0, self.health - 5)
        # cascade - multiple failures = rapid decline
        if self.consecutive_failures > 3:
            self.health = max(0, self.health - self.consecutive_failures * 2)
        self._check_state()

    def on_ignored(self):
        self.health = max(0, self.health - 1)
        self._check_state()

    def on_feedback_received(self):
        self.health = min(100, self.health + 10)
        self.is_begging = False
        self._check_state()

    def _check_state(self):
        st = self.health_state
        if st == self._last_state:
            return
        self._last_state = st

        if st == HealthState.DYING:
            self.is_begging = True
            msg = "CRITICAL: System instability! I need feedback or I will reset!"
            if self.on_dying:
                self.on_dying(msg)
            print(f"\n💀 {msg}")
        elif st == HealthState.CRITICAL:
            self.is_begging = True
            msg = "Sir, my performance is very poor. Please tell me what I am doing wrong?"
            if self.on_critical:
                self.on_critical(msg)
            print(f"\n💀 {msg}")
        elif st == HealthState.STRUGGLING:
            msg = "I am struggling. Some feedback would be nice."
            if self.on_warning:
                self.on_warning(msg)
            print(f"\n😟 {msg}")

    def should_request_feedback(self):
        if not self.is_begging:
            return False
        # dont spam, 60s cooldown
        if time.time() - self.last_feedback_request < 60:
            return False
        return True

    def get_feedback_request(self):
        self.last_feedback_request = time.time()
        return random.choice([
            "Sir, please tell me where I am going wrong...",
            "I want to improve, but I don't know how. Help?",
            "I am making too many mistakes. Will you guide me?",
            "My health is very low. Please give me a + or -!",
            "Warning: Performance critical. Feedback required to prevent reset.",
        ])

    @property
    def learning_urgency(self):
        """low health = high urgency"""
        return 1.0 - (self.health / 100.0)

    def print_status(self):
        def bar(v):
            f = int(v / 5)
            return "█" * f + "░" * (20 - f)
        st = self.health_state
        emj = {HealthState.THRIVING: "🌟", HealthState.HEALTHY: "😊",
               HealthState.STRUGGLING: "😟", HealthState.CRITICAL: "😰",
               HealthState.DYING: "💀"}
        print("\n💀 SURVIVAL")
        print("=" * 50)
        print(f"   {emj.get(st, '?')} Health: {st.value.upper()}")
        print(f"   [{bar(self.health)}] {self.health:.0f}%")
        print(f"   Success: {self.total_successes} | Fail: {self.total_failures}")
        print(f"   Consec Fails: {self.consecutive_failures}")
        print(f"   Learn Urgency: {self.learning_urgency:.0%}")
        print(f"   Begging: {'Y' if self.is_begging else 'N'}")


# ===== digital organism (main) =====

class DigitalOrganism:
    """ties subconscious + metabolism + mirror + survival together"""

    def __init__(self, memory_source=None):
        print("\n" + "=" * 70)
        print("🧬 DIGITAL ORGANISM")
        print("=" * 70)

        self.subconscious = DigitalSubconscious(
            memory_source=memory_source,
            on_hunch=self._on_hunch)

        self.metabolism = DigitalMetabolism(
            on_energy_change=self._on_energy_change,
            on_exhausted=self._on_exhausted)

        self.mirror = MirrorNeurons(
            on_mood_change=self._on_mood_change)

        self.survival = SurvivalInstinct(
            on_warning=self._on_survival_warning,
            on_critical=self._on_survival_critical,
            on_dying=self._on_survival_dying)

        self.is_alive = False
        self.birth_time = time.time()
        print("\norganism ready")

    def _on_hunch(self, hunch):
        print(f"\n💡 HUNCH: {hunch.idea}")

    def _on_energy_change(self, state):
        emj = {EnergyState.EXHAUSTED: "😫", EnergyState.TIRED: "😴",
               EnergyState.NORMAL: "😊", EnergyState.ENERGETIC: "🤩",
               EnergyState.SUPERCHARGED: "⚡"}
        print(f"\n{emj.get(state, '?')} Energy: {state.value}")

    def _on_exhausted(self):
        print("\nexhausted... need reward...")

    def _on_mood_change(self, old, new):
        print(f"\n🪞 mood: {old.value} -> {new.value}")

    def _on_survival_warning(self, msg):
        pass  # already prints

    def _on_survival_critical(self, msg):
        pass

    def _on_survival_dying(self, msg):
        pass

    def birth(self):
        self.is_alive = True
        self.birth_time = time.time()
        self.subconscious.start_dreaming()
        print(f"\nORGANISM ALIVE! ({time.strftime('%Y-%m-%d %H:%M:%S')})")

    def death(self):
        self.is_alive = False
        self.subconscious.stop_dreaming()
        age = time.time() - self.birth_time
        print(f"\norganism died after {age:.0f}s")

    def process_activity(self, energy, position=None):
        if not self.is_alive:
            return
        self.mirror.process_activity(energy, position)
        if energy > 0.1:
            self.metabolism.consume(0.1, "observation")
        self.subconscious.set_active(energy < 0.1)

    def on_success(self):
        self.metabolism.on_reward()
        self.survival.on_success()

    def on_failure(self):
        self.metabolism.on_punishment()
        self.survival.on_failure()

    def on_praise(self):
        self.metabolism.on_praise()
        self.survival.on_success()

    def on_interaction(self):
        self.metabolism.on_interaction()
        self.survival.on_feedback_received()

    def tick(self):
        self.metabolism.tick()

    def get_combined_modifiers(self):
        e_mods = self.metabolism.response_modifier
        m_mods = self.mirror.response_modifier
        return {
            'verbosity': min(e_mods['verbosity'], m_mods['verbosity']),
            'tone': m_mods['tone'],
            'energy_emoji': e_mods['emoji'],
            'interrupt_ok': m_mods['interrupt_ok'] and e_mods['verbosity'] > 0.5,
            'learning_urgency': self.survival.learning_urgency,
            'is_begging': self.survival.is_begging,
            'user_mood': self.mirror.current_mood.value,
            'energy_state': self.metabolism.energy_state.value,
            'health_state': self.survival.health_state.value,
        }

    def print_status(self):
        print("\n" + "=" * 60)
        print("🧬 ORGANISM STATUS")
        print("=" * 60)
        age = time.time() - self.birth_time
        print(f"\n   Age: {age:.0f}s")
        print(f"   Alive: {'Y' if self.is_alive else 'N'}")
        self.metabolism.print_status()
        self.mirror.print_status()
        self.survival.print_status()
        hunch = self.subconscious.get_latest_hunch()
        if hunch:
            print(f"\n   Last hunch: {hunch.idea}")


if __name__ == "__main__":
    print("=" * 70)
    print("consciousness test")
    print("=" * 70)

    def get_memories():
        return [
            ("Coding", ["Python", "VS Code"]),
            ("Browser", ["Google", "YouTube"]),
            ("Music", ["Spotify", "playlist"]),
            ("Video", ["Netflix", "movie"]),
        ]

    org = DigitalOrganism(memory_source=get_memories)
    org.birth()

    print("\n--- activity ---")
    for i in range(10):
        org.process_activity(
            energy=np.random.uniform(0, 2),
            position=(100 + i*10, 200 + i*5))
        time.sleep(0.1)

    print("\n--- success ---")
    org.on_success()
    org.on_praise()

    print("\n--- failures ---")
    for _ in range(5):
        org.on_failure()

    print("\n--- status ---")
    org.print_status()

    print("\n--- subconscious (5s) ---")
    time.sleep(5)

    org.death()
    print("\ndone")
