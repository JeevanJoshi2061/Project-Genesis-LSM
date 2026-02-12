"""
nervous.py - reflex arc + motor control
instant reactions (spinal cord) and hands with motor learning (baby mode).
"""

import numpy as np
import time
import threading
from typing import Optional, Callable, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# motor control
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    HAS_MOTOR = True
except ImportError:
    HAS_MOTOR = False

# screen capture
try:
    import mss
    HAS_SCREEN = True
except ImportError:
    HAS_SCREEN = False


# ---- reflex arc ----

class ReflexType(Enum):
    FREEZE = "freeze"
    ALERT = "alert"
    FLINCH = "flinch"
    SHIELD = "shield"

@dataclass
class ReflexTrigger:
    trigger_type: str  # loud_sound, sudden_movement, touch
    intensity: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class ReflexResponse:
    response_type: ReflexType
    duration: float
    override_brain: bool  # bypass conscious control?

class SpinalCord:
    """instant reflexes without thinking. runs BEFORE the brain."""

    LOUD_SOUND_THRESHOLD = 0.8
    SUDDEN_MOVEMENT_THRESHOLD = 0.7
    DANGER_THRESHOLD = 0.6

    def __init__(self, on_reflex=None, on_freeze=None, on_unfreeze=None):
        self.on_reflex = on_reflex
        self.on_freeze = on_freeze
        self.on_unfreeze = on_unfreeze
        self.is_frozen = False
        self.freeze_start = 0.0
        self.freeze_duration = 0.0
        self.recent_triggers = deque(maxlen=20)
        self.sensitivity = 1.0  # modulated by hormones
        print("-- spinal cord loaded --")

    def process_stimulus(self, sound_level=0.0, movement_level=0.0, danger_signal=0.0):
        """process sensory input, trigger reflexes if needed"""
        sound_level *= self.sensitivity
        movement_level *= self.sensitivity
        danger_signal *= self.sensitivity

        if sound_level > self.LOUD_SOUND_THRESHOLD:
            return self._trigger_reflex(ReflexType.FREEZE, "loud_sound",
                sound_level, 2.0)
        if movement_level > self.SUDDEN_MOVEMENT_THRESHOLD:
            return self._trigger_reflex(ReflexType.ALERT, "sudden_movement",
                movement_level, 1.0)
        if danger_signal > self.DANGER_THRESHOLD:
            return self._trigger_reflex(ReflexType.SHIELD, "danger",
                danger_signal, 3.0)
        return None

    def _trigger_reflex(self, rtype, source, intensity, duration):
        trigger = ReflexTrigger(trigger_type=source, intensity=intensity)
        self.recent_triggers.append(trigger)

        resp = ReflexResponse(response_type=rtype, duration=duration,
            override_brain=True)

        if rtype == ReflexType.FREEZE:
            self._activate_freeze(duration)

        if self.on_reflex:
            self.on_reflex(rtype, source)

        print(f"\nREFLEX: {rtype.value.upper()} (trigger:{source} intensity:{intensity:.0%})")
        return resp

    def _activate_freeze(self, duration):
        self.is_frozen = True
        self.freeze_start = time.time()
        self.freeze_duration = duration
        if self.on_freeze:
            self.on_freeze()
        print("FROZEN - all activity stopped")

    def check_freeze(self):
        """check if still frozen, unfreeze if duration passed"""
        if self.is_frozen:
            if time.time() - self.freeze_start > self.freeze_duration:
                self.is_frozen = False
                if self.on_unfreeze:
                    self.on_unfreeze()
                print("UNFROZEN - resuming")
                return False
            return True
        return False

    def set_sensitivity(self, sensitivity):
        self.sensitivity = max(0.5, min(2.0, sensitivity))

    def print_status(self):
        print("\nSPINAL CORD")
        print("=" * 50)
        print(f"   frozen: {'yes' if self.is_frozen else 'no'}")
        print(f"   sensitivity: {self.sensitivity:.1f}x")
        print(f"   recent triggers: {len(self.recent_triggers)}")


# ---- motor control ----

class MotorSkillLevel(Enum):
    NEWBORN = "newborn"   # very shaky
    BABY = "baby"         # less shaky
    TODDLER = "toddler"  # occasional mistakes
    CHILD = "child"       # good control
    ADULT = "adult"       # full precision

@dataclass
class MotorAction:
    action_type: str       # click, move, type, scroll
    target: Tuple[int, int]
    actual: Tuple[int, int]  # with tremor
    success: bool
    timestamp: float = field(default_factory=time.time)

class MotorControl:
    """hands with baby mode: starts shaky, learns through dopamine."""

    def __init__(self, initial_skill=MotorSkillLevel.NEWBORN,
                 on_action=None, enabled=True):
        self.skill_level = initial_skill
        self.on_action = on_action
        self.enabled = enabled and HAS_MOTOR
        self.experience = 0

        self.experience_thresholds = {
            MotorSkillLevel.NEWBORN: 0,
            MotorSkillLevel.BABY: 50,
            MotorSkillLevel.TODDLER: 150,
            MotorSkillLevel.CHILD: 400,
            MotorSkillLevel.ADULT: 1000,
        }
        self.tremor_levels = {
            MotorSkillLevel.NEWBORN: 50,
            MotorSkillLevel.BABY: 30,
            MotorSkillLevel.TODDLER: 15,
            MotorSkillLevel.CHILD: 5,
            MotorSkillLevel.ADULT: 0,
        }

        self.action_history = []
        self.safety_enabled = True
        self.dangerous_zones = []  # (x, y, w, h) tuples

        tremor = self.tremor_levels.get(initial_skill, 30)
        print(f"-- motor loaded (skill:{initial_skill.value} motor:{HAS_MOTOR} tremor:{tremor}px) --")

    def _get_tremor(self):
        t = self.tremor_levels.get(self.skill_level, 30)
        if t == 0:
            return (0, 0)
        return (np.random.randint(-t, t + 1), np.random.randint(-t, t + 1))

    def _is_safe_position(self, x, y):
        if not self.safety_enabled:
            return True
        for (zx, zy, zw, zh) in self.dangerous_zones:
            if zx <= x <= zx + zw and zy <= y <= zy + zh:
                print(f"dangerous zone at ({x}, {y})!")
                return False
        return True

    def move_to(self, x, y, duration=0.5, add_tremor=True):
        """move mouse with tremor"""
        if not self.enabled:
            return False
        tx, ty = self._get_tremor() if add_tremor else (0, 0)
        ax, ay = max(0, x + tx), max(0, y + ty)
        if not self._is_safe_position(ax, ay):
            return False
        try:
            pyautogui.moveTo(ax, ay, duration=duration)
            act = MotorAction("move", (x, y), (ax, ay), True)
            self.action_history.append(act)
            if self.on_action:
                self.on_action(act)
            return True
        except Exception as e:
            print(f"motor error: {e}")
            return False

    def move_to_async(self, x, y, duration=0.5, add_tremor=True, callback=None):
        """background thread mouse move (non-blocking)"""
        def _bg():
            if not self.enabled:
                if callback: callback(False)
                return
            tx, ty = self._get_tremor() if add_tremor else (0, 0)
            tgt_x, tgt_y = max(0, x + tx), max(0, y + ty)
            if not self._is_safe_position(tgt_x, tgt_y):
                if callback: callback(False)
                return
            try:
                cx, cy = pyautogui.position()
                steps = max(10, int(duration * 30))
                delay = duration / steps
                for i in range(steps + 1):
                    t = i / steps
                    st = t * t * (3 - 2 * t)  # ease-in-out
                    mx = int(cx + (tgt_x - cx) * st)
                    my = int(cy + (tgt_y - cy) * st)
                    pyautogui.moveTo(mx, my, duration=0)
                    time.sleep(delay)
                self.action_history.append(
                    MotorAction("move_async", (x, y), (tgt_x, tgt_y), True))
                if callback: callback(True)
            except Exception as e:
                print(f"async motor error: {e}")
                if callback: callback(False)

        thread = threading.Thread(target=_bg, daemon=True)
        thread.start()
        return thread

    def click(self, x=None, y=None, button="left"):
        """click with tremor"""
        if not self.enabled:
            return False
        if x is None or y is None:
            x, y = pyautogui.position()
        if not self.move_to(x, y, duration=0.3):
            return False
        try:
            pos = pyautogui.position()
            pyautogui.click(button=button)
            self.action_history.append(
                MotorAction(f"click_{button}", (x, y), pos, True))
            self._gain_experience(5)
            return True
        except Exception as e:
            print(f"click error: {e}")
            return False

    def type_text(self, text, interval=0.05):
        """type with occasional typos based on skill"""
        if not self.enabled:
            return False
        typo_rates = {
            MotorSkillLevel.NEWBORN: 0.3, MotorSkillLevel.BABY: 0.2,
            MotorSkillLevel.TODDLER: 0.1, MotorSkillLevel.CHILD: 0.02,
            MotorSkillLevel.ADULT: 0.0,
        }
        rate = typo_rates.get(self.skill_level, 0.1)
        try:
            for ch in text:
                if np.random.random() < rate:
                    wrong = chr(ord(ch) + np.random.randint(-1, 2))
                    pyautogui.typewrite(wrong, interval=interval)
                    time.sleep(0.1)
                    pyautogui.press("backspace")
                    pyautogui.typewrite(ch, interval=interval)
                else:
                    pyautogui.typewrite(ch, interval=interval)
            self.action_history.append(MotorAction("type", (0, 0), (0, 0), True))
            self._gain_experience(len(text))
            return True
        except Exception as e:
            print(f"type error: {e}")
            return False

    def _gain_experience(self, amount):
        self.experience += amount
        for skill, thresh in sorted(self.experience_thresholds.items(),
                                     key=lambda x: x[1], reverse=True):
            if self.experience >= thresh:
                if skill != self.skill_level:
                    old = self.skill_level
                    self.skill_level = skill
                    print(f"\nMOTOR SKILL UP! {old.value} -> {skill.value}")
                    print(f"   tremor: {self.tremor_levels.get(skill, 0)}px")
                break

    def on_reward(self, intensity=1.0):
        """dopamine reward -> faster motor learning"""
        bonus = int(20 * intensity)
        self._gain_experience(bonus)
        print(f"motor learning boost: +{bonus} XP")

    def add_dangerous_zone(self, x, y, w, h):
        self.dangerous_zones.append((x, y, w, h))

    def print_status(self):
        print("\nMOTOR CONTROL")
        print("=" * 50)
        print(f"   enabled: {'yes' if self.enabled else 'no'}")
        print(f"   skill: {self.skill_level.value}")
        print(f"   xp: {self.experience}")
        print(f"   tremor: {self.tremor_levels.get(self.skill_level, 0)}px")
        print(f"   actions: {len(self.action_history)}")

        cur_thresh = self.experience_thresholds.get(self.skill_level, 0)
        nxt_skill = None
        nxt_thresh = float('inf')
        for sk, th in self.experience_thresholds.items():
            if th > cur_thresh and th < nxt_thresh:
                nxt_skill = sk
                nxt_thresh = th
        if nxt_skill:
            prog = (self.experience - cur_thresh) / (nxt_thresh - cur_thresh) * 100
            print(f"   progress to {nxt_skill.value}: {prog:.0f}%")


# ===== nervous system (main) =====

class NervousSystem:
    """reflex arc + motor control"""

    def __init__(self, motor_enabled=True):
        print("\n" + "=" * 70)
        print("NERVOUS SYSTEM")
        print("=" * 70)
        self.spine = SpinalCord(on_freeze=self._on_freeze, on_unfreeze=self._on_unfreeze)
        self.motor = MotorControl(initial_skill=MotorSkillLevel.NEWBORN, enabled=motor_enabled)
        print("\nnervous system ready")

    def _on_freeze(self):
        self.motor.enabled = False

    def _on_unfreeze(self):
        self.motor.enabled = HAS_MOTOR

    def process_sensory(self, sound=0.0, movement=0.0, danger=0.0):
        """process sensory input -> True if reflex triggered"""
        self.spine.check_freeze()
        return self.spine.process_stimulus(sound, movement, danger) is not None

    def set_sensitivity_from_hormones(self, cortisol, dopamine):
        """cortisol=jumpy, dopamine=calm"""
        sens = 1.0 + (cortisol - 50) / 100 - (dopamine - 50) / 200
        self.spine.set_sensitivity(sens)

    def print_status(self):
        print("\n" + "=" * 60)
        print("NERVOUS SYSTEM STATUS")
        print("=" * 60)
        self.spine.print_status()
        self.motor.print_status()


# ---- test ----
if __name__ == "__main__":
    print("=" * 70)
    print("nervous system test")
    print("=" * 70)

    nervous = NervousSystem(motor_enabled=False)

    print("\n--- normal input ---")
    triggered = nervous.process_sensory(sound=0.3, movement=0.4)
    print(f"   reflex: {triggered}")

    print("\n--- loud sound ---")
    triggered = nervous.process_sensory(sound=0.9)
    print(f"   reflex: {triggered}")
    print(f"   frozen: {nervous.spine.is_frozen}")

    print("\n--- waiting for unfreeze... ---")
    time.sleep(2.5)
    nervous.spine.check_freeze()
    print(f"   frozen: {nervous.spine.is_frozen}")

    print("\n--- motor learning ---")
    motor = MotorControl(enabled=False)
    print(f"   initial: {motor.skill_level.value}")
    motor._gain_experience(60)
    print(f"   after 60 XP: {motor.skill_level.value}")
    motor._gain_experience(100)
    print(f"   after 160 XP: {motor.skill_level.value}")

    nervous.print_status()
    print("\ndone")
