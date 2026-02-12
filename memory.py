"""
memory.py - vector memory + hierarchical temporal memory
pure numpy vector db. cosine similarity, consolidation, persistence.
"""

import numpy as xp
# import cupy as xp

import pickle
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Memory:
    """single memory in the brain"""
    vector: xp.ndarray
    label: str
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    keywords: List[str] = field(default_factory=list)
    feedback_score: float = 0.0  # pos=good, neg=bad

    def reinforce(self, reward=1.0):
        self.strength += reward * 0.1
        self.feedback_score += reward
        self.access_count += 1
        self.last_accessed = time.time()

    def punish(self, penalty=1.0):
        self.strength = max(0.1, self.strength - penalty * 0.1)
        self.feedback_score -= penalty
        self.last_accessed = time.time()

    def decay(self, factor=0.99):
        self.strength *= factor


class VectorMemory:
    """vector db for thought vectors. cosine similarity, consolidation, persistence."""

    def __init__(self, vector_dim=2000, max_memories=1000,
                 similarity_threshold=0.85, save_path=None):
        self.vector_dim = vector_dim
        self.max_memories = max_memories
        self.similarity_threshold = similarity_threshold
        self.save_path = Path(save_path) if save_path else None

        self.memories: List[Memory] = []
        self._memory_matrix = None  # cached normalized matrix
        self._matrix_dirty = True

        self.label_index: Dict[str, List[int]] = {}
        self.total_stores = 0
        self.total_recalls = 0

        # load existing
        if self.save_path and self.save_path.exists():
            self.load()

        print(f"-- memory loaded (dim:{vector_dim} max:{max_memories} existing:{len(self.memories)}) --")

    def _cosine_similarity(self, a, b):
        a = xp.asarray(a).flatten()
        b = xp.asarray(b).flatten()
        na = xp.linalg.norm(a)
        nb = xp.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(xp.dot(a, b) / (na * nb))

    def store(self, vector, label, keywords=None, strength=1.0):
        """store or reinforce existing -> (is_new, memory)"""
        vector = xp.asarray(vector).flatten()
        keywords = keywords or []

        existing = self._find_similar(vector)
        if existing is not None and existing[1] > self.similarity_threshold:
            mem = existing[0]
            mem.reinforce()
            for kw in keywords:
                if kw not in mem.keywords:
                    mem.keywords.append(kw)
            self.total_stores += 1
            return False, mem

        mem = Memory(vector=vector.copy(), label=label,
                     strength=strength, keywords=keywords)
        self.memories.append(mem)
        self._matrix_dirty = True

        if label not in self.label_index:
            self.label_index[label] = []
        self.label_index[label].append(len(self.memories) - 1)
        self.total_stores += 1

        if len(self.memories) > self.max_memories:
            self._prune_weakest()
        return True, mem

    def _find_similar(self, vector):
        if not self.memories:
            return None
        best_mem = None
        best_sim = -1.0
        for m in self.memories:
            s = self._cosine_similarity(vector, m.vector)
            if s > best_sim:
                best_sim = s
                best_mem = m
        if best_mem is None:
            return None
        return (best_mem, best_sim)

    def recall(self, vector, top_k=5):
        """batch cosine recall with cached matrix"""
        vector = xp.asarray(vector).flatten()
        self.total_recalls += 1
        if not self.memories:
            return []

        # lazy rebuild normalized matrix
        if self._matrix_dirty or self._memory_matrix is None:
            mat = xp.vstack([m.vector.flatten() for m in self.memories])
            norms = xp.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
            self._memory_matrix = mat / norms
            self._matrix_dirty = False

        qn = vector / (xp.linalg.norm(vector) + 1e-10)
        sims = self._memory_matrix @ qn
        top_idx = xp.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            idx = int(idx)
            m = self.memories[idx]
            m.last_accessed = time.time()
            m.access_count += 1
            results.append((m, float(sims[idx])))
        return results

    def get_confidence(self, vector):
        """-> (confidence, label, best_match). key method for curiosity engine."""
        matches = self.recall(vector, top_k=3)
        if not matches:
            return 0.0, "Unknown", None

        best, best_sim = matches[0]
        conf = best_sim
        if len(matches) > 1:
            gap = best_sim - matches[1][1]
            conf = best_sim * (0.5 + 0.5 * gap)
        conf *= min(1.0, best.strength)
        return conf, best.label, best

    def reinforce_label(self, label, reward=1.0):
        if label in self.label_index:
            for idx in self.label_index[label]:
                if idx < len(self.memories):
                    self.memories[idx].reinforce(reward)

    def punish_label(self, label, penalty=1.0):
        if label in self.label_index:
            for idx in self.label_index[label]:
                if idx < len(self.memories):
                    self.memories[idx].punish(penalty)

    def _prune_weakest(self, keep_ratio=0.8):
        target = int(self.max_memories * keep_ratio)
        if len(self.memories) <= target:
            return
        sorted_mems = sorted(enumerate(self.memories), key=lambda x: x[1].strength)
        to_rm = len(self.memories) - target
        rm_set = set(idx for idx, _ in sorted_mems[:to_rm])
        self.memories = [m for i, m in enumerate(self.memories) if i not in rm_set]
        self._rebuild_index()
        print(f"pruned {to_rm} weak memories")

    def _rebuild_index(self):
        self.label_index = {}
        for idx, m in enumerate(self.memories):
            if m.label not in self.label_index:
                self.label_index[m.label] = []
            self.label_index[m.label].append(idx)

    def resize_vectors(self, new_dim):
        """pad old memories with zeros after brain growth. MUST call after lsm.resize()"""
        if new_dim <= self.vector_dim:
            return
        pad = new_dim - self.vector_dim
        print(f"\nresizing memory: {self.vector_dim} -> {new_dim}")
        for m in self.memories:
            m.vector = xp.pad(m.vector, (0, pad), 'constant')
        self.vector_dim = new_dim
        self._matrix_dirty = True
        print(f"   resized {len(self.memories)} memories")

    def consolidate(self, similarity_threshold=0.95):
        """dream mode: merge similar memories"""
        print("\nconsolidating memories...")
        merged = 0
        skip = set()
        new_mems = []

        for i, mi in enumerate(self.memories):
            if i in skip:
                continue
            to_merge = [mi]
            for j, mj in enumerate(self.memories[i+1:], start=i+1):
                if j in skip:
                    continue
                s = self._cosine_similarity(mi.vector, mj.vector)
                if s > similarity_threshold and mi.label == mj.label:
                    to_merge.append(mj)
                    skip.add(j)
                    merged += 1
            if len(to_merge) > 1:
                new_mems.append(self._merge_memories(to_merge))
            else:
                new_mems.append(mi)

        self.memories = new_mems
        self._rebuild_index()
        for m in self.memories:
            m.decay()
        print(f"   merged {merged}, total {len(self.memories)}")

    def _merge_memories(self, mems):
        avg_vec = xp.mean([m.vector for m in mems], axis=0)
        kws = list(set(kw for m in mems for kw in m.keywords))
        return Memory(
            vector=avg_vec, label=mems[0].label,
            strength=sum(m.strength for m in mems),
            keywords=kws, feedback_score=sum(m.feedback_score for m in mems),
            access_count=sum(m.access_count for m in mems))

    def save(self):
        if self.save_path is None:
            return
        data = {
            'memories': self.memories, 'label_index': self.label_index,
            'total_stores': self.total_stores, 'total_recalls': self.total_recalls,
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"saved {len(self.memories)} memories")

    def load(self):
        if self.save_path is None or not self.save_path.exists():
            return False
        try:
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)
            self.memories = data['memories']
            self.label_index = data.get('label_index', {})
            self.total_stores = data.get('total_stores', 0)
            self.total_recalls = data.get('total_recalls', 0)
            if not self.label_index:
                self._rebuild_index()
            print(f"loaded {len(self.memories)} memories")
            return True
        except:
            print("failed to load memories")
            return False

    def get_stats(self):
        labels = {}
        for m in self.memories:
            labels[m.label] = labels.get(m.label, 0) + 1
        return {
            'total_memories': len(self.memories),
            'unique_labels': len(labels),
            'label_distribution': labels,
            'total_stores': self.total_stores,
            'total_recalls': self.total_recalls,
            'avg_strength': sum(m.strength for m in self.memories) / max(1, len(self.memories)),
        }

    def print_stats(self):
        st = self.get_stats()
        print("\nMEMORY STATS")
        print("-" * 40)
        print(f"   total: {st['total_memories']}")
        print(f"   labels: {st['unique_labels']}")
        print(f"   avg strength: {st['avg_strength']:.2f}")
        print(f"   stores: {st['total_stores']}")
        print(f"   recalls: {st['total_recalls']}")
        print("\n   labels:")
        for label, cnt in sorted(st['label_distribution'].items()):
            bar = "█" * min(cnt, 20)
            print(f"      {label:15s}: {cnt:3d} {bar}")


# ---- test ----
if __name__ == "__main__":
    print("=" * 60)
    print("memory test")
    print("=" * 60)

    memory = VectorMemory(vector_dim=100, max_memories=50)

    print("\nstoring test memories...")
    for i in range(5):
        vec = xp.random.randn(100)
        vec[i*20:(i+1)*20] += 5
        labels = ["Coding", "Browser", "Video", "Desktop", "Terminal"]
        is_new, mem = memory.store(vec, labels[i], keywords=[f"test_{i}"])
        print(f"   {labels[i]} (new={is_new})")

    print("\nrecall test...")
    query = xp.random.randn(100)
    query[0:20] += 5
    matches = memory.recall(query, top_k=3)
    for m, sim in matches:
        print(f"   {m.label}: {sim:.3f}")

    print("\nconfidence test...")
    conf, label, _ = memory.get_confidence(query)
    print(f"   confidence: {conf:.2f} prediction: {label}")

    unknown = xp.random.randn(100) * 0.1
    conf, label, _ = memory.get_confidence(unknown)
    print(f"   unknown -> confidence: {conf:.2f} prediction: {label}")

    memory.print_stats()
    print("\ndone")


# ---- hierarchical temporal memory ----

from enum import IntEnum
import gzip

class MemoryLevel(IntEnum):
    RAW = 1       # raw sensory
    PATTERN = 2   # edges, lines
    OBJECT = 3    # chair, dog, browser
    CONCEPT = 4   # freedom, love, fear

@dataclass
class HierarchicalMemoryItem:
    """memory at a specific hierarchy level"""
    vector: xp.ndarray
    label: str
    level: MemoryLevel
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    children_ids: List[int] = field(default_factory=list)
    parent_id: Optional[int] = None
    compressed: bool = False

class HierarchicalMemory:
    """HTM - 4 levels: RAW -> PATTERN -> OBJECT -> CONCEPT.
    active forgetting via compression, not deletion."""

    def __init__(self, vector_dim=2000, max_ram_memories=500,
                 compression_threshold=0.3, archive_path=None):
        self.vector_dim = vector_dim
        self.max_ram_memories = max_ram_memories
        self.compression_threshold = compression_threshold
        self.archive_path = Path(archive_path) if archive_path else Path("jarvis_longterm.pkl.gz")

        self.ram = {
            MemoryLevel.RAW: [], MemoryLevel.PATTERN: [],
            MemoryLevel.OBJECT: [], MemoryLevel.CONCEPT: [],
        }
        self._disk_cache = None
        self.total_stores = 0
        self.total_compressions = 0

        print(f"-- HTM loaded (levels:4 ram/level:{max_ram_memories}) --")

    def store(self, vector, label, level, parent_id=None):
        vector = xp.asarray(vector).flatten()
        item = HierarchicalMemoryItem(
            vector=vector.copy(), label=label, level=level, parent_id=parent_id)
        self.ram[level].append(item)
        self.total_stores += 1
        if len(self.ram[level]) > self.max_ram_memories:
            self._compress_weakest(level)
        return item

    def recall(self, vector, level, top_k=5, include_archived=False):
        vector = xp.asarray(vector).flatten()
        cands = self.ram[level].copy()
        if include_archived and self._disk_cache:
            cands.extend(self._disk_cache.get(level, []))
        if not cands:
            return []
        results = [(m, self._cosine_similarity(vector, m.vector)) for m in cands]
        results.sort(key=lambda x: x[1], reverse=True)
        for m, _ in results[:top_k]:
            m.last_accessed = time.time()
            m.access_count += 1
        return results[:top_k]

    def _cosine_similarity(self, a, b):
        a, b = a.flatten(), b.flatten()
        na, nb = xp.linalg.norm(a), xp.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(xp.dot(a, b) / (na * nb))

    def _compress_weakest(self, level, count=10):
        mems = self.ram[level]
        if len(mems) <= count:
            return
        mems.sort(key=lambda m: m.strength)
        to_comp = mems[:count]
        self.ram[level] = mems[count:]
        for m in to_comp:
            m.compressed = True
        self._save_to_disk(to_comp, level)
        self.total_compressions += len(to_comp)
        print(f"compressed {len(to_comp)} memories (level:{level.name}) to disk")

    def _save_to_disk(self, memories, level):
        archive = {}
        if self.archive_path.exists():
            try:
                with gzip.open(self.archive_path, 'rb') as f:
                    archive = pickle.load(f)
            except:
                pass
        if level not in archive:
            archive[level] = []
        archive[level].extend(memories)
        with gzip.open(self.archive_path, 'wb') as f:
            pickle.dump(archive, f)

    def load_disk_cache(self):
        if self.archive_path.exists():
            try:
                with gzip.open(self.archive_path, 'rb') as f:
                    self._disk_cache = pickle.load(f)
                total = sum(len(v) for v in self._disk_cache.values())
                print(f"loaded {total} archived memories")
            except:
                print("failed to load archive")

    def sleep_consolidate(self):
        """decay, compress weak, promote strong objects to concepts"""
        print("\nHTM sleep consolidation...")
        for level in MemoryLevel:
            mems = self.ram[level]
            for m in mems:
                m.strength *= 0.95
            weak = [m for m in mems if m.strength < self.compression_threshold]
            if weak:
                self._save_to_disk(weak, level)
                self.ram[level] = [m for m in mems if m.strength >= self.compression_threshold]
                self.total_compressions += len(weak)
                print(f"   {level.name}: compressed {len(weak)} weak")

        # promote strong objects -> concepts
        strong = [m for m in self.ram[MemoryLevel.OBJECT] if m.strength > 2.0]
        for obj in strong:
            if not any(c.label == obj.label for c in self.ram[MemoryLevel.CONCEPT]):
                concept = HierarchicalMemoryItem(
                    vector=obj.vector.copy(), label=obj.label,
                    level=MemoryLevel.CONCEPT, strength=obj.strength,
                    children_ids=[id(obj)])
                self.ram[MemoryLevel.CONCEPT].append(concept)
                print(f"   promoted '{obj.label}' to CONCEPT")
        print("HTM sleep done")

    def print_stats(self):
        print("\nHTM STATS")
        print("-" * 50)
        for level in MemoryLevel:
            cnt = len(self.ram[level])
            avg = sum(m.strength for m in self.ram[level]) / max(1, cnt)
            print(f"   L{level.value} ({level.name:8s}): {cnt:4d} memories (avg:{avg:.2f})")
        print(f"\n   stores: {self.total_stores}")
        print(f"   compressions: {self.total_compressions}")
        if self.archive_path.exists():
            sz = self.archive_path.stat().st_size / 1024
            print(f"   archive: {sz:.1f} KB")
