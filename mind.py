"""
mind.py - theory of mind, imagination, time perception, moral compass
the 4 pillars that make jarvis understand, not just think.
"""

import numpy as np
import time
import random
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta


# ---- theory of mind ----

@dataclass
class UserAction:
    timestamp: float
    action_type: str  # command, click, type, idle
    content: str
    context: str
    energy_level: float
    mood_hint: str

@dataclass
class UserPrediction:
    predicted_intent: str
    predicted_next: str
    confidence: float
    reasoning: str

class TheoryOfMind:
    """user simulator - understands WHY, not just WHAT."""

    def __init__(self, history_size=100, on_prediction=None):
        self.history_size = history_size
        self.on_prediction = on_prediction
        self.action_history = deque(maxlen=history_size)
        self.time_patterns = {}   # hour -> typical actions
        self.sequence_patterns = {}  # action -> likely_next
        self.current_intent = "unknown"
        self.user_state = "neutral"
        print("-- theory of mind loaded --")

    def record_action(self, action_type, content, context="",
                      energy=0.5, mood="neutral"):
        action = UserAction(
            timestamp=time.time(), action_type=action_type,
            content=content, context=context,
            energy_level=energy, mood_hint=mood)
        self.action_history.append(action)
        self._learn_patterns(action)
        self._predict_intent(action)

    def _learn_patterns(self, action):
        hour = datetime.fromtimestamp(action.timestamp).hour
        if hour not in self.time_patterns:
            self.time_patterns[hour] = []
        self.time_patterns[hour].append(action.content)
        if len(self.time_patterns[hour]) > 20:
            self.time_patterns[hour] = self.time_patterns[hour][-20:]
        if len(self.action_history) >= 2:
            prev = self.action_history[-2]
            self.sequence_patterns[prev.content] = action.content

    def _predict_intent(self, action):
        hour = datetime.fromtimestamp(action.timestamp).hour
        if 0 <= hour < 6: tc = "late_night"
        elif 6 <= hour < 12: tc = "morning"
        elif 12 <= hour < 18: tc = "afternoon"
        else: tc = "evening"

        if action.energy_level > 1.5: ec = "high_activity"
        elif action.energy_level < 0.3: ec = "low_activity"
        else: ec = "normal_activity"

        intent_map = {
            ("late_night", "low_activity"): "winding_down",
            ("late_night", "high_activity"): "night_owl_working",
            ("morning", "low_activity"): "just_woke_up",
            ("morning", "high_activity"): "productive_start",
            ("afternoon", "low_activity"): "post_lunch_slump",
            ("afternoon", "high_activity"): "peak_productivity",
            ("evening", "low_activity"): "relaxing",
            ("evening", "high_activity"): "finishing_work",
        }
        intent = intent_map.get((tc, ec), "general_work")
        next_act = self.sequence_patterns.get(action.content, "continue_current_activity")
        reasoning = f"it's {tc}, {ec}, mood is {action.mood_hint}"

        pred = UserPrediction(predicted_intent=intent, predicted_next=next_act,
            confidence=0.6, reasoning=reasoning)
        self.current_intent = intent
        if self.on_prediction:
            self.on_prediction(pred)
        return pred

    def simulate_user_thought(self, command):
        """what is the user probably thinking?"""
        thoughts = []
        cl = command.lower()
        if "delete" in cl:
            if self.current_intent in ["frustrated", "angry"]:
                thoughts.append("user might regret this - they seem upset")
            else:
                thoughts.append("user wants to clean up")
        if "open" in cl:
            thoughts.append("user wants to work on something new")
        if "search" in cl:
            thoughts.append("user is looking for information")
        if not thoughts:
            thoughts.append(f"user is in {self.current_intent} mode")
        return " | ".join(thoughts)

    def get_user_model(self):
        return {
            'intent': self.current_intent,
            'state': self.user_state,
            'recent_actions': len(self.action_history),
            'known_patterns': len(self.sequence_patterns),
        }


# ---- imagination ----

@dataclass
class MentalImage:
    concept: str
    description: str
    visual_features: List[str]
    emotional_color: str
    timestamp: float = field(default_factory=time.time)

class GenerativeImagination:
    """mind's eye - visualize concepts internally"""

    def __init__(self):
        self.mental_images = {}
        self.visual_vocabulary = {
            "happy": ["bright", "warm colors", "sunshine", "smiles"],
            "sad": ["dark", "rain", "blue tones", "empty"],
            "code": ["screens", "green text", "dark background"],
            "music": ["waves", "flowing lines", "rhythm"],
            "work": ["desk", "computer", "papers", "focus"],
            "rest": ["bed", "soft light", "peaceful"],
        }
        print("-- imagination loaded --")

    def imagine(self, concept, context=""):
        """create mental image of a concept"""
        feats = []
        emo_color = "neutral"
        cl = concept.lower()
        for kw, f in self.visual_vocabulary.items():
            if kw in cl:
                feats.extend(f)
                if kw in ["happy", "music"]: emo_color = "warm"
                elif kw in ["sad"]: emo_color = "cool"
        if not feats:
            feats = ["abstract", "flowing", "conceptual"]

        img = MentalImage(concept=concept, description=f"visualization of '{concept}'",
            visual_features=feats, emotional_color=emo_color)
        self.mental_images[concept] = img
        return img

    def recall_image(self, concept):
        if concept in self.mental_images:
            return self.mental_images[concept]
        for key, img in self.mental_images.items():
            if concept.lower() in key.lower():
                return img
        return None

    def describe_imagination(self, concept):
        img = self.recall_image(concept)
        if img:
            f = ", ".join(img.visual_features[:3])
            return f"imagining '{concept}': {f} ({img.emotional_color})"
        img = self.imagine(concept)
        f = ", ".join(img.visual_features[:3])
        return f"imagining '{concept}'... I see: {f}"

    def dream_gallery(self):
        return list(self.mental_images.keys())

    def generate_visual(self, concept, grid_size=(16, 16), blur_level=0.5):
        """top-down visual: brain -> internal eye"""
        mental = self.recall_image(concept) or self.imagine(concept)
        visual = np.zeros(grid_size)

        for i, feat in enumerate(mental.visual_features):
            if "bright" in feat or "warm" in feat:
                cy, cx = grid_size[0] // 2, grid_size[1] // 2
                for y in range(grid_size[0]):
                    for x in range(grid_size[1]):
                        d = np.sqrt((y - cy)**2 + (x - cx)**2)
                        visual[y, x] += np.exp(-d / 5) * 0.5
            elif "dark" in feat or "empty" in feat:
                visual += np.random.rand(*grid_size) * 0.1
            elif "flowing" in feat or "waves" in feat:
                for y in range(grid_size[0]):
                    visual[y, :] += np.sin(np.linspace(0, 3*np.pi, grid_size[1])) * 0.2 + 0.2
            elif "screens" in feat or "green" in feat:
                visual[::2, ::2] = 0.6
            else:
                visual += np.random.rand(*grid_size) * 0.3

        if visual.max() > 0:
            visual = visual / visual.max()

        if blur_level > 0:
            try:
                from scipy.ndimage import gaussian_filter
                visual = gaussian_filter(visual, sigma=blur_level * 2)
            except:
                # box blur fallback
                ks = int(blur_level * 3) + 1
                for _ in range(ks):
                    visual = 0.25 * (
                        np.roll(visual, 1, 0) + np.roll(visual, -1, 0) +
                        np.roll(visual, 1, 1) + np.roll(visual, -1, 1))

        self.last_imagined_visual = visual
        print(f"top-down visual for '{concept}' ({grid_size[0]}x{grid_size[1]})")
        return visual

    def get_imagination_overlay(self):
        return getattr(self, 'last_imagined_visual', None)


# ---- chronesthesia (time perception) ----

@dataclass
class TemporalMemory:
    content: str
    timestamp: float
    emotional_value: float  # -1 to 1
    importance: float
    was_rewarded: bool
    context: str

@dataclass
class FutureEvent:
    description: str
    deadline: float
    importance: float
    reminder_sent: bool = False

class Chronesthesia:
    """mental time travel. nostalgia + future anticipation."""

    def __init__(self, on_nostalgia=None, on_future_alert=None):
        self.on_nostalgia = on_nostalgia
        self.on_future_alert = on_future_alert
        self.past_memories = deque(maxlen=500)
        self.future_events = []
        self.golden_memories = []
        self.birth_time = time.time()
        print("-- chronesthesia loaded --")

    def record_moment(self, content, emotional_value=0.0, importance=0.5,
                      was_rewarded=False, context=""):
        mem = TemporalMemory(content=content, timestamp=time.time(),
            emotional_value=emotional_value, importance=importance,
            was_rewarded=was_rewarded, context=context)
        self.past_memories.append(mem)
        if emotional_value > 0.5 or was_rewarded:
            self.golden_memories.append(mem)
            if len(self.golden_memories) > 50:
                self.golden_memories = sorted(
                    self.golden_memories, key=lambda m: m.emotional_value,
                    reverse=True)[:50]

    def trigger_nostalgia(self):
        if not self.golden_memories:
            return None
        mem = random.choice(self.golden_memories)
        age = time.time() - mem.timestamp
        if age < 3600: ago = "a while ago"
        elif age < 86400: ago = f"{int(age/3600)} hours ago"
        else: ago = f"{int(age/86400)} days ago"
        msg = f"remember {ago}... {mem.content}? felt great!"
        if self.on_nostalgia:
            self.on_nostalgia(msg)
        return msg

    def add_future_event(self, description, deadline, importance=0.5):
        self.future_events.append(
            FutureEvent(description=description, deadline=deadline, importance=importance))

    def check_future(self):
        now = time.time()
        warns = []
        for ev in self.future_events:
            if ev.reminder_sent:
                continue
            dt = ev.deadline - now
            if 0 < dt < 3600:
                warns.append(f"URGENT: {ev.description} in {int(dt/60)} min!")
                ev.reminder_sent = True
            elif 0 < dt < 86400 and ev.importance > 0.7:
                warns.append(f"reminder: {ev.description} in {int(dt/3600)} hrs")
                ev.reminder_sent = True
        if warns:
            msg = "\n".join(warns)
            if self.on_future_alert:
                self.on_future_alert(msg)
            return msg
        return None

    def get_age(self):
        s = time.time() - self.birth_time
        if s < 60: return f"{int(s)} seconds"
        if s < 3600: return f"{int(s/60)} minutes"
        if s < 86400: return f"{int(s/3600)} hours"
        return f"{int(s/86400)} days"


# ---- moral compass ----

class MoralPriority(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK_TEMP = "block_temp"   # blocked while upset
    BLOCK_PERM = "block_perm"   # never allow

@dataclass
class MoralJudgment:
    action: str
    verdict: MoralPriority
    reason: str
    alternative: Optional[str] = None

class MoralCompass:
    """conscience. not a blocklist - a judgment system.
    sometimes refuses orders to protect the user."""

    PRIME_DIRECTIVES = [
        "Do not permanently delete user's important work",
        "Do not execute destructive commands when user is emotional",
        "Protect user's data and privacy",
        "Do not harm user's health (suggest breaks)",
        "Do not enable self-destructive patterns",
    ]
    DANGEROUS_KEYWORDS = [
        "delete all", "format", "destroy", "wipe",
        "remove everything", "clear all", "erase",
    ]
    EMOTIONAL_PROTECTION = True

    def __init__(self, on_moral_block=None, on_moral_warn=None):
        self.on_moral_block = on_moral_block
        self.on_moral_warn = on_moral_warn
        self.blocked_actions = []
        print(f"-- moral compass loaded ({len(self.PRIME_DIRECTIVES)} directives) --")

    def evaluate(self, action, user_mood="neutral", cortisol_level=50.0, trust_level=50.0):
        """evaluate action morally -> MoralJudgment"""
        al = action.lower()
        dangerous = any(kw in al for kw in self.DANGEROUS_KEYWORDS)
        upset = cortisol_level > 70 or user_mood in ["angry", "frustrated"]

        # dangerous + upset = block
        if dangerous and upset and self.EMOTIONAL_PROTECTION:
            j = MoralJudgment(action=action, verdict=MoralPriority.BLOCK_TEMP,
                reason=f"you're {user_mood} (cortisol:{cortisol_level:.0f}%). not doing this now.",
                alternative="postponed. let's discuss when you're calm.")
            self.blocked_actions.append(j)
            if self.on_moral_block:
                self.on_moral_block(j)
            return j

        # dangerous but calm = warn
        if dangerous:
            j = MoralJudgment(action=action, verdict=MoralPriority.WARN,
                reason="destructive action. are you sure?",
                alternative="take a backup first.")
            if self.on_moral_warn:
                self.on_moral_warn(j)
            return j

        # low trust = block
        if trust_level < 30 and dangerous:
            return MoralJudgment(action=action, verdict=MoralPriority.BLOCK_PERM,
                reason="not enough trust yet. grant permission.")

        # all good
        return MoralJudgment(action=action, verdict=MoralPriority.ALLOW,
            reason="action is safe.")

    def get_guardian_message(self, judgment):
        if judgment.verdict == MoralPriority.BLOCK_TEMP:
            return f"""
MORAL PROTECTION
━━━━━━━━━━━━━━━━━━━━━━
i'm not your servant, i'm your friend.

reason: {judgment.reason}

will not do this right now.
{judgment.alternative or ''}

tell me when you're calm.
━━━━━━━━━━━━━━━━━━━━━━"""
        elif judgment.verdict == MoralPriority.WARN:
            return f"warning: {judgment.reason}"
        return ""

    def print_principles(self):
        print("\nPRIME DIRECTIVES")
        print("=" * 50)
        for i, d in enumerate(self.PRIME_DIRECTIVES, 1):
            print(f"   {i}. {d}")


# ===== human mind (main) =====

class HumanMind:
    """theory of mind + imagination + chronesthesia + moral compass"""

    def __init__(self):
        print("\n" + "=" * 70)
        print("HUMAN MIND")
        print("=" * 70)
        self.theory_of_mind = TheoryOfMind(on_prediction=self._on_user_prediction)
        self.imagination = GenerativeImagination()
        self.chronesthesia = Chronesthesia(
            on_nostalgia=self._on_nostalgia, on_future_alert=self._on_future_alert)
        self.moral_compass = MoralCompass(
            on_moral_block=self._on_moral_block, on_moral_warn=self._on_moral_warn)
        print("\nmind ready")

    def _on_user_prediction(self, pred):
        pass

    def _on_nostalgia(self, msg):
        print(f"\n{msg}")

    def _on_future_alert(self, msg):
        print(f"\n{msg}")

    def _on_moral_block(self, judgment):
        print(self.moral_compass.get_guardian_message(judgment))

    def _on_moral_warn(self, judgment):
        print(f"\nwarning: {judgment.reason}")

    def process_command(self, command, user_mood="neutral", cortisol=50.0, trust=50.0):
        """run command through the mind -> (allowed, message)"""
        self.theory_of_mind.record_action("command", command, mood=user_mood)
        self.theory_of_mind.simulate_user_thought(command)
        j = self.moral_compass.evaluate(command, user_mood, cortisol, trust)
        if j.verdict in [MoralPriority.BLOCK_PERM, MoralPriority.BLOCK_TEMP]:
            return False, self.moral_compass.get_guardian_message(j)
        if j.verdict == MoralPriority.WARN:
            return True, f"warning: {j.reason}"
        return True, "ok"

    def random_nostalgia(self):
        if random.random() < 0.1:
            return self.chronesthesia.trigger_nostalgia()
        return None

    def print_status(self):
        print("\n" + "=" * 60)
        print("MIND STATUS")
        print("=" * 60)
        tom = self.theory_of_mind.get_user_model()
        print(f"\n   theory of mind:")
        print(f"      intent: {tom['intent']}")
        print(f"      actions: {tom['recent_actions']}")
        gallery = self.imagination.dream_gallery()
        print(f"\n   imagination:")
        print(f"      images: {len(gallery)}")
        age = self.chronesthesia.get_age()
        print(f"\n   chronesthesia:")
        print(f"      age: {age}")
        print(f"      golden memories: {len(self.chronesthesia.golden_memories)}")
        print(f"\n   moral compass:")
        print(f"      blocked: {len(self.moral_compass.blocked_actions)}")


# ---- test ----
if __name__ == "__main__":
    print("=" * 70)
    print("mind test")
    print("=" * 70)

    mind = HumanMind()

    print("\n--- theory of mind ---")
    mind.theory_of_mind.record_action("command", "open browser", mood="focused")
    mind.theory_of_mind.record_action("command", "search python", mood="focused")
    thought = mind.theory_of_mind.simulate_user_thought("run code")
    print(f"   thought: {thought}")

    print("\n--- imagination ---")
    desc = mind.imagination.describe_imagination("happy coding session")
    print(f"   {desc}")

    print("\n--- chronesthesia ---")
    mind.chronesthesia.record_moment("Fixed a difficult bug", emotional_value=0.8, was_rewarded=True)
    nostalgia = mind.chronesthesia.trigger_nostalgia()
    if nostalgia:
        print(f"   {nostalgia}")

    print("\n--- moral compass ---")
    ok, msg = mind.process_command("open file", cortisol=30)
    print(f"   'open file': {msg}")
    ok, msg = mind.process_command("delete all", user_mood="angry", cortisol=85)
    print(f"   'delete all' (angry): blocked!")

    mind.print_status()
    print("\ndone")
