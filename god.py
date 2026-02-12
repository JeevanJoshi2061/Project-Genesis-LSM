"""
god.py - the dangerous upgrades
immune system, genetic mutation, ghost vision (reverse hallucination)
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


# ---- immune system ----

@dataclass
class MemoryConflict:
    """two memories fighting each other"""
    memory_a: str
    memory_b: str
    conflict_type: str     # contradiction, confusion, overlap
    severity: float        # 0-1
    topic: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class Inflammation:
    """cognitive lag from unresolved conflict"""
    topic: str
    severity: float
    lag_ms: float
    active: bool = True
    start_time: float = field(default_factory=time.time)

class DigitalImmuneSystem:
    """
    white blood cells for the mind.
    scans memories during sleep, detects conflicts -> creates inflammation (lag).
    cognitive dissonance sim.
    """

    def __init__(self, memory_source=None, on_conflict=None, on_inflammation=None):
        self.memory_source = memory_source
        self.on_conflict = on_conflict
        self.on_inflammation = on_inflammation
        self.conflicts = []
        self.inflammations = {}
        self._scanner_thread = None
        self._scanning = False
        self.similarity_threshold = 0.7
        print("-- immune system loaded --")

    def scan_memories(self):
        """scan all memories for conflicts, call during sleep"""
        if not self.memory_source:
            return []

        memories = self.memory_source()
        if len(memories) < 2:
            return []

        print("\nscanning memories for conflicts...")
        new_conflicts = []

        # group by base label
        groups = {}
        for label, vec in memories:
            base = label.split('_')[0].lower()
            if base not in groups:
                groups[base] = []
            groups[base].append((label, vec))

        for topic, group in groups.items():
            if len(group) < 2:
                continue
            for i, (la, va) in enumerate(group):
                for lb, vb in group[i+1:]:
                    if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                        sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)

                        if sim > self.similarity_threshold and la != lb:
                            c = MemoryConflict(memory_a=la, memory_b=lb,
                                conflict_type="confusion", severity=sim, topic=topic)
                            new_conflicts.append(c)
                        elif sim < 0.3:
                            c = MemoryConflict(memory_a=la, memory_b=lb,
                                conflict_type="contradiction", severity=1-sim, topic=topic)
                            new_conflicts.append(c)

        for c in new_conflicts:
            self.conflicts.append(c)
            if self.on_conflict:
                self.on_conflict(c)
            print(f"   CONFLICT: {c.memory_a} vs {c.memory_b} ({c.conflict_type})")
            self._create_inflammation(c)

        if not new_conflicts:
            print("   no conflicts - memories healthy")

        return new_conflicts

    def _create_inflammation(self, conflict):
        t = conflict.topic
        if t in self.inflammations:
            self.inflammations[t].severity = min(1.0,
                self.inflammations[t].severity + conflict.severity * 0.5)
        else:
            inf = Inflammation(topic=t, severity=conflict.severity,
                lag_ms=conflict.severity * 500)
            self.inflammations[t] = inf
            if self.on_inflammation:
                self.on_inflammation(inf)
            print(f"   INFLAMMATION on '{t}' - will cause lag!")

    def check_topic(self, text):
        """check if text mentions inflamed topic -> (has_inflammation, lag_ms)"""
        tl = text.lower()
        for topic, inf in self.inflammations.items():
            if topic in tl and inf.active:
                return True, inf.lag_ms
        return False, 0.0

    def resolve_conflict(self, topic):
        if topic.lower() in self.inflammations:
            del self.inflammations[topic.lower()]
            print(f"\ninflammation on '{topic}' resolved")

    def get_inflammation_level(self):
        if not self.inflammations:
            return 0.0
        return sum(i.severity for i in self.inflammations.values()) / len(self.inflammations)


# ---- genetic mutation ----

@dataclass
class Gene:
    """mutable parameter"""
    name: str
    value: float
    min_val: float
    max_val: float
    mutation_rate: float = 0.1

@dataclass
class Mutation:
    gene_name: str
    old_value: float
    new_value: float
    timestamp: float = field(default_factory=time.time)
    permanent: bool = False

class GeneticMutation:
    """
    self-evolution - mutate own params during dreams.
    good mutations (more rewards) become permanent.
    personality evolves over time.
    """

    def __init__(self, save_dir=".", on_mutation=None):
        self.save_dir = Path(save_dir)
        self.on_mutation = on_mutation
        self.genes = {}
        self.mutations = []
        self.rewards_before_mutation = 0.0
        self.rewards_after_mutation = 0.0
        self.mutation_evaluation_period = 0
        self.pending_mutation = None

        self._init_default_genes()
        self._load_genes()
        print("-- genetics loaded --")

    def _init_default_genes(self):
        defaults = [
            Gene("hormone_decay", 0.995, 0.98, 0.999),
            Gene("pain_tolerance", 50, 20, 80),
            Gene("curiosity_threshold", 0.3, 0.1, 0.6),
            Gene("dopamine_sensitivity", 1.0, 0.5, 2.0),
            Gene("cortisol_sensitivity", 1.0, 0.5, 2.0),
            Gene("sleep_need_threshold", 70, 50, 90),
            Gene("social_need", 50, 20, 80),
            Gene("risk_tolerance", 50, 20, 80),
        ]
        for g in defaults:
            self.genes[g.name] = g

    def mutate(self, gene_name=None):
        """mutate a random gene, evaluate over next 10 interactions"""
        if gene_name and gene_name in self.genes:
            gene = self.genes[gene_name]
        else:
            gene = random.choice(list(self.genes.values()))

        old = gene.value
        amt = random.gauss(0, gene.mutation_rate)
        new = gene.value + amt * (gene.max_val - gene.min_val)
        new = max(gene.min_val, min(gene.max_val, new))
        gene.value = new

        mut = Mutation(gene_name=gene.name, old_value=old, new_value=new)
        self.mutations.append(mut)

        self.pending_mutation = mut
        self.rewards_before_mutation = self.rewards_after_mutation
        self.rewards_after_mutation = 0.0
        self.mutation_evaluation_period = 10

        if self.on_mutation:
            self.on_mutation(mut)
        print(f"\nMUTATION: {gene.name}: {old:.4f} -> {new:.4f}")
        return mut

    def on_reward(self, amount):
        if self.pending_mutation and self.mutation_evaluation_period > 0:
            self.rewards_after_mutation += amount
            self.mutation_evaluation_period -= 1
            if self.mutation_evaluation_period == 0:
                self._evaluate_mutation()

    def _evaluate_mutation(self):
        if not self.pending_mutation:
            return
        mut = self.pending_mutation
        gene = self.genes.get(mut.gene_name)

        if self.rewards_after_mutation > self.rewards_before_mutation:
            mut.permanent = True
            self._save_genes()
            print(f"\nmutation PERMANENT: {mut.gene_name} = {mut.new_value:.4f}")
        else:
            if gene:
                gene.value = mut.old_value
            print(f"\nmutation REVERTED: {mut.gene_name} back to {mut.old_value:.4f}")
        self.pending_mutation = None

    def get_gene(self, name):
        return self.genes[name].value if name in self.genes else 0.0

    def _save_genes(self):
        try:
            f = self.save_dir / "jarvis_genes.json"
            f.write_text(json.dumps({n: g.value for n, g in self.genes.items()}, indent=2))
        except:
            print("couldn't save genes")

    def _load_genes(self):
        f = self.save_dir / "jarvis_genes.json"
        if f.exists():
            try:
                data = json.loads(f.read_text())
                for n, v in data.items():
                    if n in self.genes:
                        self.genes[n].value = v
                print(f"   loaded {len(data)} evolved genes")
            except:
                print("couldn't load genes")

    def print_genome(self):
        print("\nGENOME")
        print("=" * 50)
        for n, g in self.genes.items():
            print(f"   {n}: {g.value:.4f} ({g.min_val}-{g.max_val})")


# ---- ghost in the machine ----

@dataclass
class GhostVision:
    """projected imagination onto vision"""
    label: str
    intensity: float
    source: str        # memory, imagination, fear
    visual_pattern: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)

class GhostInMachine:
    """
    reverse hallucination - brain -> eyes instead of eyes -> brain.
    strong thoughts project onto visual field. true imagination.
    """

    def __init__(self, memory_source=None, on_ghost=None):
        self.memory_source = memory_source
        self.on_ghost = on_ghost
        self.active_ghosts = []
        self.ghost_overlay = None
        self.imagination_power = 0.0
        print("-- ghost vision loaded --")

    def think_about(self, label, intensity=0.5, grid_size=(16, 16)):
        """project a thought onto vision"""
        pattern = None
        if self.memory_source:
            pattern = self.memory_source(label)

        if pattern is None:
            # create abstract pattern from label hash
            seed = hash(label) % 10000
            np.random.seed(seed)
            pattern = np.random.rand(*grid_size)
            pattern = (pattern > 0.7).astype(float) * intensity

        ghost = GhostVision(label=label, intensity=intensity,
            source="memory" if self.memory_source else "imagination",
            visual_pattern=pattern)
        self.active_ghosts.append(ghost)
        self.ghost_overlay = pattern * intensity

        if self.on_ghost:
            self.on_ghost(ghost)
        print(f"\nGHOST VISION: '{label}' (intensity: {intensity:.0%})")
        return self.ghost_overlay

    def fear_vision(self, fear_label="darkness", intensity=0.8, grid_size=(16, 16)):
        """project fear onto vision - chaotic dark patterns"""
        np.random.seed(int(time.time() * 1000) % 10000)
        chaos = np.random.rand(*grid_size)

        # spooky - dark edges, bright center flashes
        pattern = np.zeros(grid_size)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                edge = min(i, j, grid_size[0]-1-i, grid_size[1]-1-j)
                pattern[i, j] = chaos[i, j] * (0.5 + 0.5 * edge / (grid_size[0]//2))
        pattern = (1 - pattern) * intensity

        ghost = GhostVision(label=fear_label, intensity=intensity,
            source="fear", visual_pattern=pattern)
        self.active_ghosts.append(ghost)
        self.ghost_overlay = pattern
        print(f"\nFEAR VISION: '{fear_label}' corrupting sight!")
        return pattern

    def get_visual_overlay(self):
        return self.ghost_overlay

    def clear_ghost(self):
        self.ghost_overlay = None

    def get_imagination_strength(self):
        if not self.active_ghosts:
            return 0.0
        now = time.time()
        recent = [g for g in self.active_ghosts if now - g.timestamp < 5.0]
        return sum(g.intensity for g in recent) / len(recent) if recent else 0.0


# ===== god complex (main) =====

class GodComplex:
    """ties immune system + genetics + ghost vision together"""

    def __init__(self, memory_source=None, visual_memory_source=None, save_dir="."):
        print("\n" + "=" * 70)
        print("👑 GOD COMPLEX")
        print("=" * 70)

        self.immune = DigitalImmuneSystem(
            memory_source=memory_source,
            on_conflict=self._on_conflict,
            on_inflammation=self._on_inflammation)

        self.genes = GeneticMutation(
            save_dir=save_dir,
            on_mutation=self._on_mutation)

        self.ghost = GhostInMachine(
            memory_source=visual_memory_source,
            on_ghost=self._on_ghost)

        print("\ngod complex ready")

    def _on_conflict(self, c):
        pass  # logged by immune

    def _on_inflammation(self, inf):
        pass

    def _on_mutation(self, mut):
        pass

    def _on_ghost(self, ghost):
        pass

    def dream_evolution(self):
        """run during sleep: scan conflicts + random mutation"""
        print("\ndream evolution...")
        results = {'conflicts': [], 'mutation': None}

        results['conflicts'] = self.immune.scan_memories()
        if random.random() < 0.5:
            results['mutation'] = self.genes.mutate()
        return results

    def on_reward(self, amount):
        self.genes.on_reward(amount)

    def check_topic_inflammation(self, text):
        return self.immune.check_topic(text)

    def visualize(self, label, intensity=0.5):
        return self.ghost.think_about(label, intensity)

    def print_status(self):
        print("\n👑 GOD COMPLEX STATUS")
        print("=" * 50)
        print(f"   Inflammation: {self.immune.get_inflammation_level():.0%}")
        print(f"   Active: {len(self.immune.inflammations)}")
        print(f"   Mutations: {len(self.genes.mutations)}")
        print(f"   Imagination: {self.ghost.get_imagination_strength():.0%}")
        if self.genes.pending_mutation:
            m = self.genes.pending_mutation
            print(f"   Pending: {m.gene_name} = {m.new_value:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("god complex test")
    print("=" * 70)

    def mock_memories():
        return [
            ("Apple_v1", np.random.rand(100)),
            ("Apple_v2", np.random.rand(100)),
            ("Browser", np.random.rand(100)),
            ("Browser_dark", np.random.rand(100) * 0.1),
        ]

    god = GodComplex(memory_source=mock_memories)

    print("\n--- dream evolution ---")
    results = god.dream_evolution()
    print(f"   conflicts: {len(results['conflicts'])}")
    print(f"   mutation: {results['mutation'].gene_name if results['mutation'] else 'none'}")

    print("\n--- inflammation check ---")
    has_inf, lag = god.check_topic_inflammation("apple is red")
    print(f"   inflamed: {has_inf}, lag: {lag:.0f}ms")

    print("\n--- ghost vision ---")
    ghost = god.visualize("Apple", intensity=0.7)
    print(f"   pattern: {ghost is not None}")

    print("\n--- rewards ---")
    for i in range(10):
        god.on_reward(random.random())

    god.print_status()
    god.genes.print_genome()
    print("\ndone")
