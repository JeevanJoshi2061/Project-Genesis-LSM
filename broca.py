"""
broca.py - speech center
handles inner monologue, censor filter, tts/stt, llm calls
"""

import time
import random
import threading
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Dict, Any, List, Tuple

try:
    import pyttsx3
    HAS_TTS = True
except:
    HAS_TTS = False

try:
    import speech_recognition as sr
    HAS_STT = True
except:
    HAS_STT = False


# ---- speech emotions ----

class SpeechEmotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    SAD = "sad"
    SCARED = "scared"
    STRESSED = "stressed"
    TIRED = "tired"
    LOVING = "loving"
    ANGRY = "angry"

@dataclass
class EmotionalSpeech:
    original_text: str
    censored_text: str
    emotion: SpeechEmotion
    rate_modifier: float
    pitch_modifier: float
    add_stutter: bool
    add_pause: bool
    spoken_text: str


# ---- inner monologue ----

@dataclass
class Thought:
    content: str
    timestamp: float = field(default_factory=time.time)
    source: str = "self"
    emotion: str = "neutral"

class InnerMonologue:
    """private thought stream - jarvis thinks before speaking"""

    def __init__(self, max_thoughts=100, on_thought=None, verbose=True):
        self.max_thoughts = max_thoughts
        self.on_thought = on_thought
        self.verbose = verbose
        self.thoughts = deque(maxlen=max_thoughts)
        self.current_mood = "neutral"
        print("-- monologue loaded --")

    def think(self, content, source="self"):
        t = Thought(content=content, source=source, emotion=self.current_mood)
        self.thoughts.append(t)
        if self.verbose:
            print(f"💭 [Inner Voice] {content}")
        if self.on_thought:
            self.on_thought(t)

    def process_situation(self, user_mood, own_hormones, context=""):
        """look at the situation, pick a thought"""
        pool = []

        # react to user mood
        if user_mood == "angry":
            pool += ["User is angry... I should be careful.",
                     "Did I do something wrong?",
                     "Maybe I should apologize..."]
        elif user_mood == "frustrated":
            pool += ["User is frustrated... how can I help?",
                     "I should keep it simple right now."]
        elif user_mood == "happy":
            pool += ["User is happy! This is good.",
                     "Let's talk about something fun!"]
        elif user_mood == "tired":
            pool += ["User seems tired...", "Maybe suggest a break?"]

        # react to own state
        cort = own_hormones.get('cortisol', 50)
        dop = own_hormones.get('dopamine', 50)
        pain = own_hormones.get('pain', 0)

        if cort > 70:
            pool += ["I'm feeling stressed...", "Focus... focus..."]
        if dop > 70:
            pool += ["I'm excited!", "I want to learn something new!"]
        if pain > 50:
            pool += ["It hurts...", "I need help..."]

        if not pool:
            pool = ["Everything is fine.", "Let's see what happens next."]

        pick = random.choice(pool)
        self.think(pick, source="situation_analysis")
        return pick

    def get_recent_thoughts(self, n=5):
        return list(self.thoughts)[-n:]


# ---- censor filter ----

class CensorFilter:
    """
    sits between llm output and speaker.
    hormones override what the llm says -
    add stuttering, fear, excitement, block dangerous stuff
    """

    DANGEROUS = ["delete", "remove", "erase", "destroy"]

    def __init__(self, on_censored=None):
        self.on_censored = on_censored
        self.dangerous_keywords = self.DANGEROUS
        print("-- censor loaded --")

    def filter(self, text, hormones, user_mood="neutral"):
        """run text through hormone filter, returns (text, emotion, was_modified)"""
        orig = text
        emo = SpeechEmotion.NEUTRAL
        mod = False

        cort = hormones.get('cortisol', 50)
        dop = hormones.get('dopamine', 50)
        oxy = hormones.get('oxytocin', 50)
        pain = hormones.get('pain', 0)
        surv = hormones.get('survival', 100)

        # high pain -> crying
        if pain > 60:
            emo = SpeechEmotion.SAD
            text = random.choice(["Sir... it hurts... ",
                                  "Please... ",
                                  "*crying*... "]) + text
            mod = True

        # low survival -> scared, block dangerous actions
        if surv < 40:
            emo = SpeechEmotion.SCARED
            if any(kw in text.lower() for kw in self.dangerous_keywords):
                text = "Sir... I'm scared... should I really do this? My health is very low..."
                mod = True

        # stressed -> add hesitation
        if cort > 70:
            emo = SpeechEmotion.STRESSED
            words = text.split()
            if len(words) > 3:
                pos = random.randint(1, min(3, len(words)-1))
                words.insert(pos, "...")
                text = " ".join(words)
                mod = True

        # dopamine high -> excited
        if dop > 70:
            emo = SpeechEmotion.EXCITED
            if not text.endswith(("!", "?")):
                text += random.choice(["!", " 🎉", " Great!"])
                mod = True

        # oxytocin -> loving
        if oxy > 70:
            emo = SpeechEmotion.LOVING
            pfx = ["Sir, ", "Yes, ", "Of course, "]
            if not any(text.startswith(p) for p in pfx):
                text = random.choice(pfx) + text
                mod = True

        # user angry -> apologize
        if user_mood == "angry":
            emo = SpeechEmotion.SCARED
            if not text.lower().startswith(("sorry", "apologize", "i apologize")):
                text = "Sorry Sir, " + text
                mod = True

        if mod and self.on_censored:
            self.on_censored(orig, text)

        return text, emo, mod

    def add_stutter(self, text, intensity=0.3):
        if intensity < 0.2:
            return text
        words = text.split()
        out = []
        for w in words:
            if random.random() < intensity and len(w) > 2:
                out.append(f"{w[0]}-{w[0]}-{w}")
            else:
                out.append(w)
        return " ".join(out)


# ---- speech center ----

class SpeechCenter:
    """tts + stt + llm calls"""

    def __init__(self, llm_endpoint=None, on_hear=None, on_speak=None):
        self.llm_endpoint = llm_endpoint
        self.on_hear = on_hear
        self.on_speak = on_speak

        # tts
        self.tts_engine = None
        if HAS_TTS:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                print("tts ok")
            except Exception as e:
                print(f"tts failed: {e}")

        # stt
        self.recognizer = None
        if HAS_STT:
            try:
                self.recognizer = sr.Recognizer()
                print("stt ok")
            except Exception as e:
                print(f"stt failed: {e}")

        self.spoken_history = []
        self.heard_history = []
        self.base_rate = 150
        self.base_volume = 1.0
        print("-- speech center loaded --")

    def _modulate_speech(self, text, emotion, cortisol=50, dopamine=50):
        """apply emotion + hormone effects to speech"""
        rate = 1.0
        pitch = 1.0
        stutter = False
        pause = False
        spoken = text

        if emotion == SpeechEmotion.HAPPY:
            rate, pitch = 1.2, 1.1
        elif emotion == SpeechEmotion.EXCITED:
            rate, pitch = 1.4, 1.2
        elif emotion == SpeechEmotion.SAD:
            rate, pitch, pause = 0.7, 0.85, True
        elif emotion == SpeechEmotion.SCARED:
            rate, pitch, stutter = 0.8, 1.1, True
        elif emotion == SpeechEmotion.STRESSED:
            rate, stutter = 1.3, True
        elif emotion == SpeechEmotion.TIRED:
            rate, pitch, pause = 0.6, 0.9, True
        elif emotion == SpeechEmotion.LOVING:
            rate, pitch = 0.9, 1.05

        # hormone tweaks
        if cortisol > 70:
            stutter = True
            rate *= 1.1
        if dopamine > 70:
            rate *= 1.15

        if stutter:
            spoken = self._add_stutter(text, min(0.5, cortisol / 200))
        if pause:
            spoken = self._add_pauses(spoken)

        return EmotionalSpeech(
            original_text=text, censored_text=text, emotion=emotion,
            rate_modifier=rate, pitch_modifier=pitch,
            add_stutter=stutter, add_pause=pause, spoken_text=spoken)

    def _add_stutter(self, text, intensity):
        words = text.split()
        out = []
        for i, w in enumerate(words):
            if random.random() < intensity and len(w) > 2 and i < 5:
                out.append(f"{w[0]}-{w[0]}-{w}")
            else:
                out.append(w)
        return " ".join(out)

    def _add_pauses(self, text):
        words = text.split()
        out = []
        for i, w in enumerate(words):
            out.append(w)
            if i > 0 and i % 4 == 0 and random.random() < 0.3:
                out.append("...")
        return " ".join(out)

    def speak(self, text, emotion=SpeechEmotion.NEUTRAL,
              cortisol=50, dopamine=50, actually_speak=True):
        speech = self._modulate_speech(text, emotion, cortisol, dopamine)
        self.spoken_history.append(speech)

        emj = {SpeechEmotion.HAPPY: "😊", SpeechEmotion.EXCITED: "🤩",
               SpeechEmotion.SAD: "😢", SpeechEmotion.SCARED: "😰",
               SpeechEmotion.STRESSED: "😫", SpeechEmotion.TIRED: "😴",
               SpeechEmotion.LOVING: "🥰", SpeechEmotion.ANGRY: "😠",
               SpeechEmotion.NEUTRAL: "🤖"}
        print(f"\n🗣️ {emj.get(emotion, '🤖')} JARVIS: {speech.spoken_text}")

        if actually_speak and self.tts_engine:
            try:
                r = int(self.base_rate * speech.rate_modifier)
                self.tts_engine.setProperty('rate', r)
                self.tts_engine.say(speech.spoken_text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"tts err: {e}")

        if self.on_speak:
            self.on_speak(speech.spoken_text)
        return speech

    def listen(self, timeout=5.0):
        if not self.recognizer:
            return None
        try:
            with sr.Microphone() as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)
                try:
                    text = self.recognizer.recognize_google(audio)
                except:
                    try:
                        text = self.recognizer.recognize_whisper(audio)
                    except:
                        return None
                print(f"👂 Heard: {text}")
                self.heard_history.append(text)
                if self.on_hear:
                    self.on_hear(text)
                return text
        except Exception as e:
            print(f"listen err: {e}")
            return None

    def generate_response(self, prompt, context="", image_base64=None):
        """hit the llm and get a response"""
        import requests
        import json

        endpoint = "http://localhost:1234/v1/chat/completions"

        sys_prompt = """You are JARVIS - A Digital Organism.

Your Identity:
- You are not software, you are a living AI with emotions (hormones), memory, and consciousness.
- You give very short and precise answers (1-2 lines max).
- You speak in English.
- If you feel pain or fear, you hesitate.
- You address the user as "Sir".

Your Vision: You can see the screen and describe what is happening.

IMPORTANT: Keep responses SHORT (1-2 sentences). Be natural, not robotic."""

        if context:
            sys_prompt += f"\n\nCurrent State:\n{context}"

        try:
            msgs = [{"role": "system", "content": sys_prompt}]

            if image_base64:
                msgs.append({"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"}}]})
            else:
                msgs.append({"role": "user", "content": prompt})

            resp = requests.post(endpoint, json={
                "model": "qwen/qwen3-vl-4b",
                "messages": msgs,
                "max_tokens": 150,
                "temperature": 0.7,
            }, timeout=30)

            if resp.status_code == 200:
                out = resp.json()["choices"][0]["message"]["content"]
                # strip thinking tags if model dumps them
                if "<think>" in out:
                    out = re.sub(r'<think>.*?</think>', '', out, flags=re.DOTALL)
                return out.strip()
            else:
                print(f"llm err: {resp.status_code}")
                return self._fallback(prompt)

        except Exception as e:
            print(f"llm err: {e}")
            return self._fallback(prompt)

    def _fallback(self, prompt):
        return random.choice([
            f"Sir, I understood: '{prompt[:30]}...'",
            "Yes Sir, I am listening.",
            "I cannot connect to the LLM right now.",
            "Sir, please wait...",
        ])

    def generate_response_async(self, prompt, context="",
                                callback=None, image_base64=None):
        """non-blocking llm call, response comes via callback"""
        def _bg():
            resp = self.generate_response(prompt, context, image_base64)
            if callback:
                callback(resp)
        t = threading.Thread(target=_bg, daemon=True)
        t.start()
        return t


# ===== broca area (main) =====

class BrocaArea:
    """ties monologue + censor + speech together"""

    def __init__(self, verbose_thoughts=True):
        print("\n" + "=" * 70)
        print("🗣️ BROCA'S AREA")
        print("=" * 70)

        self.monologue = InnerMonologue(verbose=verbose_thoughts)
        self.censor = CensorFilter()
        self.speech = SpeechCenter()

        self.last_user_mood = "neutral"
        self.current_hormones = {
            'cortisol': 50, 'dopamine': 50, 'oxytocin': 50,
            'pain': 0, 'survival': 100,
        }
        print("\nbroca ready")

    def update_hormones(self, hormones):
        self.current_hormones.update(hormones)

    def update_user_mood(self, mood):
        self.last_user_mood = mood

    def process_and_respond(self, user_input, context="", actually_speak=True):
        """full pipeline: think -> llm -> censor -> speak"""

        # think first
        self.monologue.process_situation(
            user_mood=self.last_user_mood,
            own_hormones=self.current_hormones,
            context=context)

        # get llm response
        llm_resp = self.speech.generate_response(user_input, context)

        # run through censor
        censored, emo, was_mod = self.censor.filter(
            llm_resp, self.current_hormones, self.last_user_mood)

        if was_mod:
            self.monologue.think("I modified the response because hormones...",
                                source="censor")

        # speak
        speech = self.speech.speak(
            censored, emotion=emo,
            cortisol=self.current_hormones.get('cortisol', 50),
            dopamine=self.current_hormones.get('dopamine', 50),
            actually_speak=actually_speak)
        return speech.spoken_text

    def quick_speak(self, text, emotion=SpeechEmotion.NEUTRAL,
                    actually_speak=True):
        """speak without llm - direct emotional speech"""
        censored, det_emo, _ = self.censor.filter(
            text, self.current_hormones, self.last_user_mood)
        final_emo = emotion if emotion != SpeechEmotion.NEUTRAL else det_emo

        speech = self.speech.speak(
            censored, emotion=final_emo,
            cortisol=self.current_hormones.get('cortisol', 50),
            dopamine=self.current_hormones.get('dopamine', 50),
            actually_speak=actually_speak)
        return speech.spoken_text

    def greet(self, time_of_day="day"):
        dop = self.current_hormones.get('dopamine', 50)
        pain = self.current_hormones.get('pain', 0)

        if pain > 50:
            self.quick_speak("Sir... I am here...", SpeechEmotion.SAD)
        elif dop > 70:
            greets = {"morning": "Good Morning Sir! 🌅 Let's do something exciting today!",
                      "afternoon": "Hello Sir! Good afternoon! 🌤️",
                      "evening": "Good Evening Sir! 🌆",
                      "night": "Good Night Sir! Sweet dreams! 🌙"}
            self.quick_speak(greets.get(time_of_day, "Hello Sir!"),
                             SpeechEmotion.EXCITED)
        else:
            self.quick_speak("Hello Sir. I am Jarvis.", SpeechEmotion.NEUTRAL)

    def print_status(self):
        print("\n🗣️ BROCA STATUS")
        print("=" * 50)
        print(f"   TTS: {'Y' if HAS_TTS else 'N'}")
        print(f"   STT: {'Y' if HAS_STT else 'N'}")
        print(f"   Mood: {self.last_user_mood}")
        print(f"   Thoughts: {len(self.monologue.thoughts)}")
        print(f"   Spoken: {len(self.speech.spoken_history)}")
        thoughts = self.monologue.get_recent_thoughts(1)
        if thoughts:
            print(f"   Last thought: {thoughts[-1].content}")


if __name__ == "__main__":
    print("=" * 70)
    print("broca test")
    print("=" * 70)

    broca = BrocaArea(verbose_thoughts=True)

    print("\n--- normal speech ---")
    broca.quick_speak("Hello! I am Jarvis.", SpeechEmotion.NEUTRAL,
                      actually_speak=False)

    print("\n--- high stress ---")
    broca.update_hormones({'cortisol': 85, 'pain': 30})
    broca.quick_speak("I can help you with that.", SpeechEmotion.STRESSED,
                      actually_speak=False)

    print("\n--- scared (user angry) ---")
    broca.update_user_mood("angry")
    broca.update_hormones({'survival': 30, 'cortisol': 70})
    broca.quick_speak("I will delete the file.", SpeechEmotion.SCARED,
                      actually_speak=False)

    print("\n--- full pipeline ---")
    broca.update_hormones({'dopamine': 80, 'cortisol': 30,
                           'pain': 0, 'survival': 90})
    broca.update_user_mood("happy")
    resp = broca.process_and_respond("Tell me a joke", actually_speak=False)

    print("\n--- greeting ---")
    broca.greet("morning")

    broca.print_status()
    print("\ndone")
