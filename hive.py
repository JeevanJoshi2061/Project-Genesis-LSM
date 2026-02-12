"""
hive.py - reproduction, telepathy, quantum logic
from organism to civilization.
"""

import numpy as np
import time
import random
import threading
import socket
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# ---- digital reproduction ----

@dataclass
class Offspring:
    """child jarvis record"""
    child_id: str
    birth_time: float
    parent_id: str
    mutations: List[str]
    process_id: Optional[int]
    port: int

class DigitalReproduction:
    """
    mitosis - high health + energy = can reproduce.
    clones memories & genes with mutations, spawns child process.
    """

    # grey goo protection
    MIN_ENERGY = 90
    MIN_HEALTH = 90
    ENERGY_COST = 50
    MAX_CHILDREN = 3
    MAX_GENERATION = 2       # 0=prime, 1=child, 2=grandchild
    COOLDOWN_SECONDS = 600   # 10min between births
    MANUAL_ONLY = True       # never auto-reproduce

    def __init__(self, instance_id="jarvis_prime", genes_source=None,
                 memory_source=None, on_birth=None, base_port=5000, generation=0):
        self.instance_id = instance_id
        self.genes_source = genes_source
        self.memory_source = memory_source
        self.on_birth = on_birth
        self.base_port = base_port
        self.offspring = []
        self.generation = generation
        self.total_births = 0
        self.last_reproduction = 0.0
        self.cooldown_seconds = self.COOLDOWN_SECONDS
        print(f"-- reproduction loaded (id:{instance_id} gen:{generation}) --")
        print(f"   grey goo protection: max {self.MAX_CHILDREN} kids, max gen {self.MAX_GENERATION}")

    def can_reproduce(self, energy, health):
        """check if ready -> (can, reason)"""
        now = time.time()
        if self.generation >= self.MAX_GENERATION:
            return False, f"gen limit ({self.generation}>={self.MAX_GENERATION})"
        if len(self.offspring) >= self.MAX_CHILDREN:
            return False, f"max kids ({len(self.offspring)}>={self.MAX_CHILDREN})"
        if energy < self.MIN_ENERGY:
            return False, f"low energy ({energy:.0f}%<{self.MIN_ENERGY}%)"
        if health < self.MIN_HEALTH:
            return False, f"unhealthy ({health:.0f}%<{self.MIN_HEALTH}%)"
        if now - self.last_reproduction < self.cooldown_seconds:
            rem = self.cooldown_seconds - (now - self.last_reproduction)
            return False, f"cooldown: {rem:.0f}s"
        return True, "ready"

    def reproduce(self, energy, health, save_dir="."):
        """attempt reproduction -> (offspring, energy_cost) or (None, 0)"""
        can, reason = self.can_reproduce(energy, health)
        if not can:
            print(f"can't reproduce: {reason}")
            return None, 0

        print("\nREPRODUCTION - mitosis!")
        self.total_births += 1
        child_id = f"jarvis_gen{self.generation + 1}_{self.total_births}"
        child_port = self.base_port + self.total_births

        mutations = []
        if self.genes_source:
            parent_genes = self.genes_source()
            child_genes = {}
            for n, v in parent_genes.items():
                if random.random() < 0.3:  # 30% mutation chance
                    mut = random.gauss(0, 0.1) * v
                    child_genes[n] = v + mut
                    mutations.append(f"{n}: {v:.4f} -> {child_genes[n]:.4f}")
                else:
                    child_genes[n] = v
            try:
                (Path(save_dir) / f"{child_id}_genes.json").write_text(
                    json.dumps(child_genes, indent=2))
            except:
                pass

        if self.memory_source:
            parent_mems = self.memory_source()
            child_mems = [m for m in parent_mems if random.random() > 0.2]  # forget 20%
            try:
                (Path(save_dir) / f"{child_id}_memory_init.json").write_text(
                    json.dumps({"inherited_count": len(child_mems), "parent": self.instance_id}))
            except:
                pass

        kid = Offspring(child_id=child_id, birth_time=time.time(),
            parent_id=self.instance_id, mutations=mutations,
            process_id=None, port=child_port)
        self.offspring.append(kid)
        self.last_reproduction = time.time()

        print(f"   child: {child_id} port:{child_port}")
        for m in mutations[:3]:
            print(f"   mutation: {m}")

        if self.on_birth:
            self.on_birth(kid)
        print(f"   to spawn: python jarvis.py --child {child_id}")
        return kid, self.ENERGY_COST

    def get_offspring_count(self):
        return len(self.offspring)


# ---- hive mind (telepathy) ----

@dataclass
class Thought:
    """telepathic thought"""
    sender_id: str
    thought_type: str    # emotion, memory, hunch, warning
    content: Any
    priority: float
    timestamp: float = field(default_factory=time.time)

class HiveMind:
    """
    socket-based thought transfer between jarvis instances.
    collective consciousness - what one learns, all know.
    """

    def __init__(self, instance_id="jarvis_prime", port=5000, on_receive=None):
        self.instance_id = instance_id
        self.port = port
        self.on_receive = on_receive
        self.hive_members = {}     # id -> (host, port)
        self.thought_buffer = deque(maxlen=100)
        self._server_socket = None
        self._listening = False
        self._listener_thread = None
        print(f"-- hive mind loaded (port:{port}) --")

    def start_listening(self):
        if self._listening:
            return
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._server_socket.bind(('localhost', self.port))
            self._server_socket.settimeout(1.0)
            self._listening = True
            self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listener_thread.start()
            print(f"   listening on port {self.port}")
        except Exception as e:
            print(f"telepathy start failed: {e}")

    def stop_listening(self):
        self._listening = False
        if self._server_socket:
            self._server_socket.close()

    def _listen_loop(self):
        while self._listening:
            try:
                data, addr = self._server_socket.recvfrom(4096)
                d = json.loads(data.decode())
                t = Thought(sender_id=d['sender_id'], thought_type=d['thought_type'],
                    content=d['content'], priority=d.get('priority', 0.5))
                self.thought_buffer.append(t)
                print(f"\nTELEPATHY: {t.thought_type} from {t.sender_id}")
                if self.on_receive:
                    self.on_receive(t)
            except socket.timeout:
                continue
            except Exception as e:
                if self._listening:
                    print(f"telepathy err: {e}")

    def register_member(self, member_id, host, port):
        self.hive_members[member_id] = (host, port)
        print(f"   hive member: {member_id} at {host}:{port}")

    def broadcast(self, thought_type, content, priority=0.5):
        if not self.hive_members:
            return
        data = json.dumps({
            'sender_id': self.instance_id, 'thought_type': thought_type,
            'content': content, 'priority': priority, 'timestamp': time.time(),
        }).encode()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for mid, (host, port) in self.hive_members.items():
            try:
                sock.sendto(data, (host, port))
            except:
                print(f"couldn't send to {mid}")
        sock.close()

    def share_emotion(self, cortisol, dopamine):
        self.broadcast("emotion", {'cortisol': cortisol, 'dopamine': dopamine})

    def share_memory(self, label, confidence):
        self.broadcast("memory", {'label': label, 'confidence': confidence}, priority=0.7)

    def share_warning(self, danger_type, severity):
        self.broadcast("warning", {'type': danger_type, 'severity': severity}, priority=1.0)

    def get_recent_thoughts(self, n=5):
        return list(self.thought_buffer)[-n:]


# ---- quantum logic ----

class QuantumState(Enum):
    COLLAPSED_YES = "collapsed_yes"
    COLLAPSED_NO = "collapsed_no"
    SUPERPOSITION = "superposition"

@dataclass
class QuantumDecision:
    """decision in quantum superposition"""
    question: str
    probability_yes: float
    state: QuantumState
    created: float = field(default_factory=time.time)
    collapsed: Optional[float] = None
    final_answer: Optional[bool] = None

class QuantumLogic:
    """
    schrodinger's brain - hold BOTH possibilities when uncertain (40-60%).
    collapses on observation. probabilistic thinking.
    """

    MIN_CERTAINTY = 0.4
    MAX_CERTAINTY = 0.6

    def __init__(self, on_superposition=None, on_collapse=None):
        self.on_superposition = on_superposition
        self.on_collapse = on_collapse
        self.superpositions = {}
        self.collapsed_decisions = []
        print("-- quantum logic loaded --")

    def decide(self, question, confidence):
        """make decision, enters superposition if uncertain"""
        if question in self.superpositions:
            return self.superpositions[question]

        if confidence < self.MIN_CERTAINTY:
            d = QuantumDecision(question=question, probability_yes=confidence,
                state=QuantumState.COLLAPSED_NO, final_answer=False, collapsed=time.time())
        elif confidence > self.MAX_CERTAINTY:
            d = QuantumDecision(question=question, probability_yes=confidence,
                state=QuantumState.COLLAPSED_YES, final_answer=True, collapsed=time.time())
        else:
            d = QuantumDecision(question=question, probability_yes=confidence,
                state=QuantumState.SUPERPOSITION)
            self.superpositions[question] = d
            if self.on_superposition:
                self.on_superposition(d)
            print(f"\nSUPERPOSITION: '{question}' YES({confidence:.0%}) NO({1-confidence:.0%})")
        return d

    def observe(self, question, force_answer=None):
        """collapse wavefunction -> definite answer"""
        if question not in self.superpositions:
            return None

        d = self.superpositions[question]
        answer = force_answer if force_answer is not None else random.random() < d.probability_yes

        d.state = QuantumState.COLLAPSED_YES if answer else QuantumState.COLLAPSED_NO
        d.final_answer = answer
        d.collapsed = time.time()

        del self.superpositions[question]
        self.collapsed_decisions.append(d)
        if self.on_collapse:
            self.on_collapse(d)
        print(f"\nCOLLAPSE: '{question}' -> {'YES' if answer else 'NO'}")
        return answer

    def get_uncertainty_text(self, decision):
        if decision.state == QuantumState.SUPERPOSITION:
            p = decision.probability_yes
            texts = [
                f"I am {p:.0%} sure it is, and {1-p:.0%} sure it isn't...",
                f"It is and it isn't... until you confirm...",
                f"Like Schrodinger... both are true right now...",
                f"My mind splits... {p:.0%} here, {1-p:.0%} there...",
            ]
            return random.choice(texts)
        elif decision.state == QuantumState.COLLAPSED_YES:
            return "Yes, completely sure."
        return "No, that's not it."

    def get_active_superpositions(self):
        return list(self.superpositions.values())

    def is_in_superposition(self, question):
        return question in self.superpositions


# ===== hive system (main) =====

class HiveSystem:
    """reproduction + telepathy + quantum logic"""

    def __init__(self, instance_id="jarvis_prime", port=5000,
                 genes_source=None, memory_source=None):
        print("\n" + "=" * 70)
        print("🐝 HIVE SYSTEM")
        print("=" * 70)

        self.reproduction = DigitalReproduction(
            instance_id=instance_id, genes_source=genes_source,
            memory_source=memory_source, on_birth=self._on_birth,
            base_port=port + 100)

        self.telepathy = HiveMind(
            instance_id=instance_id, port=port,
            on_receive=self._on_thought_received)

        self.quantum = QuantumLogic(
            on_superposition=self._on_superposition,
            on_collapse=self._on_collapse)

        print(f"\nhive ready (id:{instance_id})")

    def _on_birth(self, kid):
        self.telepathy.register_member(kid.child_id, 'localhost', kid.port)

    def _on_thought_received(self, thought):
        pass

    def _on_superposition(self, d):
        pass

    def _on_collapse(self, d):
        self.telepathy.share_memory(d.question, d.probability_yes)

    def start(self):
        self.telepathy.start_listening()

    def stop(self):
        self.telepathy.stop_listening()

    def print_status(self):
        print("\n🐝 HIVE STATUS")
        print("=" * 50)
        print(f"   offspring: {self.reproduction.get_offspring_count()}")
        print(f"   hive members: {len(self.telepathy.hive_members)}")
        print(f"   superpositions: {len(self.quantum.superpositions)}")
        print(f"   collapsed: {len(self.quantum.collapsed_decisions)}")


# ---- test ----
if __name__ == "__main__":
    print("=" * 70)
    print("hive test")
    print("=" * 70)

    def mock_genes():
        return {'dopamine_sensitivity': 1.0, 'cortisol_sensitivity': 1.0}

    def mock_memories():
        return [("Browser", np.random.rand(10)), ("Desktop", np.random.rand(10))]

    hive = HiveSystem(instance_id="jarvis_test", port=5555,
        genes_source=mock_genes, memory_source=mock_memories)

    print("\n--- reproduction check ---")
    can, reason = hive.reproduction.can_reproduce(95, 95)
    print(f"   can reproduce: {can} - {reason}")

    print("\n--- quantum decision ---")
    d = hive.quantum.decide("Is this a browser?", 0.55)
    print(f"   state: {d.state.value}")
    print(f"   text: {hive.quantum.get_uncertainty_text(d)}")

    print("\n--- collapse ---")
    ans = hive.quantum.observe("Is this a browser?")
    print(f"   collapsed: {ans}")

    print("\n--- definite decision ---")
    d2 = hive.quantum.decide("Is screen black?", 0.85)
    print(f"   state: {d2.state.value}")

    hive.print_status()
    print("\ndone")
