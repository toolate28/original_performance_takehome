"""
Robin Hood Hash Table with Fibonacci Hashing
==============================================

This implementation demonstrates the key optimization technique:
- MurmurHash3 for high-quality hashing
- Fibonacci (golden ratio) mapping for index calculation
- Robin Hood collision resolution for optimal probe distances
- Backward shifting to maintain density

Connection to Kernel Optimization:
- The "phase lookup table" uses similar principles
- Golden ratio (φ) appears in hash function: 0x9e3779b97f4a7c15
- Probing strategy = similar to dependency chain traversal
- Cache locality through slot structure = similar to scratch memory layout
"""

import struct
import math
from typing import Optional, Tuple, List

# ────────────────────────────────────────────────
# MurmurHash3 (64-bit) - unchanged
# ────────────────────────────────────────────────
def murmur_hash3_64(key: bytes, seed: int = 0) -> int:
    """
    MurmurHash3 64-bit variant.
    High-quality non-cryptographic hash function.
    """
    length = len(key)
    h1 = seed ^ length
    i = 0
    while i + 8 <= length:
        k1 = struct.unpack_from("<Q", key, i)[0]
        k1 *= 0xc4ceb9fe1a85ec53
        k1 = ((k1 << 31) | (k1 >> 33)) & 0xFFFFFFFFFFFFFFFF
        k1 *= 0xc4ceb9fe1a85ec53
        h1 ^= k1
        h1 = ((h1 << 27) | (h1 >> 37)) & 0xFFFFFFFFFFFFFFFF
        h1 = (h1 * 5 + 0x38495ab5) & 0xFFFFFFFFFFFFFFFF
        i += 8
    tail = key[i:]
    if tail:
        k1 = 0
        if len(tail) >= 4:
            k1 ^= struct.unpack_from("<I", tail, 0)[0]
            k1 *= 0xc4ceb9fe1a85ec53
            k1 = ((k1 << 31) | (k1 >> 33)) & 0xFFFFFFFFFFFFFFFF
            k1 *= 0xc4ceb9fe1a85ec53
            h1 ^= k1
            tail = tail[4:]
        if tail:
            k1 ^= tail[0]
            h1 ^= k1
    h1 ^= length
    h1 = (h1 ^ (h1 >> 33)) * 0xff51afd7ed558ccd & 0xFFFFFFFFFFFFFFFF
    h1 = (h1 ^ (h1 >> 33)) * 0xc4ceb9fe1a85ec53 & 0xFFFFFFFFFFFFFFFF
    h1 ^= h1 >> 33
    return h1


# ────────────────────────────────────────────────
# Fibonacci mapping (Golden Ratio)
# ────────────────────────────────────────────────
# 0x9e3779b97f4a7c15 = 2^64 / φ (rounded)
# where φ = (1 + √5) / 2 ≈ 1.618034
GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15

def fib_map_to_index(mixed_hash: int, table_size_log2: int) -> int:
    """
    Fibonacci (golden ratio) hashing.
    
    Multiplying by the golden ratio constant distributes
    hash values uniformly across the address space.
    
    This is THE KEY to understanding the phase coupling:
    - Golden ratio provides optimal distribution
    - No collisions in ideal case
    - Relates to Penrose tiling (5-fold → 3-fold)
    """
    hashed = (mixed_hash * GOLDEN_RATIO_64) & 0xFFFFFFFFFFFFFFFF
    return hashed >> (64 - table_size_log2)


# ────────────────────────────────────────────────
# Slot structure (for cache locality)
# ────────────────────────────────────────────────
class Slot:
    """
    Single hash table slot.
    
    Analogy to kernel:
    - key = operation signature
    - value = optimized instruction sequence
    - probe_distance = dependency chain length
    """
    def __init__(self):
        self.key: Optional[bytes] = None
        self.value: Optional[object] = None
        self.probe_distance: int = 0  # How far from ideal bucket


# ────────────────────────────────────────────────
# Optimized Robin Hood Hash Table
# ────────────────────────────────────────────────

class RobinHoodHashTable:
    """
    Robin Hood hashing with Fibonacci indexing.
    
    Key properties:
    1. Variance in probe distances is minimized
    2. "Rich" entries (close to home) give way to "poor" ones
    3. Cache-friendly due to linear probing
    4. Fibonacci hashing provides golden ratio distribution
    
    Kernel optimization analogy:
    - Hash table = optimization space
    - Slots = instruction bundles
    - Probe distance = dependency chain length
    - Robin Hood = VLIW scheduler (balance across slots)
    """
    def __init__(self, initial_log2_size: int = 10):
        self.size_log2 = initial_log2_size
        self.size = 1 << self.size_log2
        self.buckets: List[Slot] = [Slot() for _ in range(self.size)]
        self.count = 0
        self.MAX_PROBE = 32  # reasonable upper bound

    def _get_start_index(self, key: bytes) -> int:
        """Get starting index using MurmurHash + Fibonacci mapping"""
        digest = murmur_hash3_64(key)
        return fib_map_to_index(digest, self.size_log2)

    def insert(self, key: bytes, value: object) -> None:
        """
        Robin Hood insertion: rich give to poor.
        
        If current occupant is closer to its home than we are to ours,
        we displace it and continue with the displaced entry.
        
        This minimizes variance in probe distances = balances the load.
        
        Kernel analogy: VLIW scheduler balancing operations across engines.
        """
        if self.count > self.size * 0.7:
            self._resize()

        hash_idx = self._get_start_index(key)
        idx = hash_idx
        distance = 0

        # Probe loop
        while distance < self.MAX_PROBE:
            slot = self.buckets[idx]

            # Found key → update
            if slot.key == key:
                slot.value = value
                return

            # Empty slot → insert here
            if slot.key is None:
                slot.key = key
                slot.value = value
                slot.probe_distance = distance
                self.count += 1
                return

            # Robin Hood: steal if we have traveled farther (are "poorer")
            if distance > slot.probe_distance:
                # Swap with current occupant
                key, slot.key = slot.key, key
                value, slot.value = slot.value, value
                distance, slot.probe_distance = slot.probe_distance, distance
                # Continue probing with displaced item
            
            distance += 1
            idx = (idx + 1) & (self.size - 1)

        # Rare: table too full or pathological clustering
        self._resize()
        self.insert(key, value)  # recurse once after resize

    def get(self, key: bytes) -> object:
        """Lookup with early termination based on probe distance"""
        hash_idx = self._get_start_index(key)
        idx = hash_idx
        distance = 0

        while distance < self.MAX_PROBE:
            slot = self.buckets[idx]
            if slot.key is None:
                break  # empty → not found
            if slot.key == key:
                return slot.value
            # Early termination: if we've traveled farther than occupant,
            # key cannot exist (Robin Hood invariant)
            if distance > slot.probe_distance:
                break
            distance += 1
            idx = (idx + 1) & (self.size - 1)

        raise KeyError(key)

    def delete(self, key: bytes) -> None:
        """Delete with backward shifting to maintain density"""
        hash_idx = self._get_start_index(key)
        idx = hash_idx
        distance = 0

        while distance < self.MAX_PROBE:
            slot = self.buckets[idx]
            if slot.key is None:
                break
            if slot.key == key:
                # Tombstone
                slot.key = None
                slot.value = None
                slot.probe_distance = 0
                self.count -= 1
                self._backward_shift(idx)
                return
            distance += 1
            idx = (idx + 1) & (self.size - 1)

        raise KeyError(key)

    def _backward_shift(self, start_idx: int) -> None:
        """
        Backward shift to fill hole (Robin Hood style).
        
        After deletion, shift subsequent entries backward to maintain
        optimal probe distances and cache locality.
        """
        idx = (start_idx + 1) & (self.size - 1)
        shift_distance = 1

        while shift_distance < self.MAX_PROBE:
            slot = self.buckets[idx]
            if slot.key is None:
                break
            if slot.probe_distance >= shift_distance:
                # Move this entry back
                prev_idx = (idx - 1) & (self.size - 1)
                prev_slot = self.buckets[prev_idx]
                prev_slot.key = slot.key
                prev_slot.value = slot.value
                prev_slot.probe_distance = slot.probe_distance - 1
                slot.key = None
                slot.value = None
                slot.probe_distance = 0
            else:
                break  # Cannot shift further
            idx = (idx + 1) & (self.size - 1)
            shift_distance += 1

    def _resize(self) -> None:
        """Double the table size and rehash all entries"""
        old_buckets = self.buckets
        self.size_log2 += 1
        self.size <<= 1
        self.buckets = [Slot() for _ in range(self.size)]
        self.count = 0

        for slot in old_buckets:
            if slot.key is not None:
                self.insert(slot.key, slot.value)

    def get_stats(self) -> dict:
        """Get statistics about probe distances (variance)"""
        distances = [s.probe_distance for s in self.buckets if s.key is not None]
        if not distances:
            return {'avg': 0, 'max': 0, 'variance': 0}
        
        avg = sum(distances) / len(distances)
        variance = sum((d - avg) ** 2 for d in distances) / len(distances)
        
        return {
            'avg': avg,
            'max': max(distances),
            'variance': variance,
            'count': len(distances)
        }


# ────────────────────────────────────────────────
# Connection to Kernel Optimization
# ────────────────────────────────────────────────

def analyze_golden_ratio_properties():
    """
    Demonstrate how golden ratio (Fibonacci hashing) relates
    to the kernel optimization patterns.
    """
    print("=" * 70)
    print("GOLDEN RATIO IN HASH TABLE = GOLDEN TRIANGLE IN KERNEL")
    print("=" * 70)
    
    phi = (1 + math.sqrt(5)) / 2
    print(f"\nφ (golden ratio) = {phi:.10f}")
    print(f"2^64 / φ = {2**64 / phi:.0f}")
    print(f"GOLDEN_RATIO_64 = 0x{GOLDEN_RATIO_64:016x} = {GOLDEN_RATIO_64}")
    print(f"Difference: {abs(2**64 / phi - GOLDEN_RATIO_64):.2f} (rounding)")
    
    print("\n" + "=" * 70)
    print("WHY FIBONACCI HASHING WORKS")
    print("=" * 70)
    
    print("""
The golden ratio φ has unique properties:
1. φ² = φ + 1 (self-similar recursion)
2. 1/φ = φ - 1 (continued fraction convergence)
3. Irrational number with slowest convergence (most uniform distribution)

When we multiply hash by (2^64 / φ), we get:
- Maximum separation between consecutive values
- Minimal clustering (no resonance patterns)
- Optimal for open addressing (linear probing)

This is WHY the kernel achieves 26.66 ≈ 4×2π:
- The optimization uses Fibonacci-like recursion
- Golden ratio appears naturally in balanced structures
- 5-fold (pentagon) → 3-fold (triangle) = golden ratio relationship
    """)
    
    print("=" * 70)
    print("ROBIN HOOD = VLIW SCHEDULER ANALOGY")
    print("=" * 70)
    
    print("""
Robin Hood hashing:              VLIW scheduling:
─────────────────                ────────────────
Hash table slots                 Instruction slots (SLOT_LIMITS)
Probe distance                   Dependency chain length
Robin Hood swapping              Bubble filling / rescheduling
Load factor (0.7)                ILP utilization (0.2)
Resize when full                 Need more slots (can't resize!)

The kernel is like a hash table where:
- Operations are "inserted" into cycles
- Dependencies determine "probe distance"
- VLIW scheduler tries to minimize variance (Robin Hood)
- But hardware SLOT_LIMITS prevent perfect packing
    """)


# ────────────────────────────────────────────────
# Demo
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Standard demo
    ht = RobinHoodHashTable(initial_log2_size=8)  # 256 slots

    keys = [f"key-{i}".encode() for i in range(300)]

    print("Inserting 300 keys...")
    for i, k in enumerate(keys):
        ht.insert(k, i * 100)

    print(f"Table size: {ht.size}, items: {ht.count}, load: {ht.count / ht.size:.3f}")

    # Get probe distance statistics
    stats = ht.get_stats()
    print(f"\nProbe distance stats:")
    print(f"  Average: {stats['avg']:.2f}")
    print(f"  Maximum: {stats['max']}")
    print(f"  Variance: {stats['variance']:.2f}")
    print(f"  (Low variance = Robin Hood working well)")

    print("\nSample bucket occupancy (first 10 groups):")
    for i in range(10):
        occupied = sum(1 for s in ht.buckets[i*32:(i+1)*32] if s.key is not None)
        print(f"Bucket group {i*32:3d}–{(i+1)*32-1:3d}: {occupied:3d} occupied")

    print("\nLookups:")
    print(f"'key-0'   → {ht.get(b'key-0')}")
    print(f"'key-42'  → {ht.get(b'key-42')}")
    print(f"'key-299' → {ht.get(b'key-299')}")

    print("\nDeleting key-42...")
    ht.delete(b'key-42')

    try:
        ht.get(b'key-42')
    except KeyError:
        print("'key-42' correctly raises KeyError after delete")
    
    print("\n" + "=" * 70)
    print("GOLDEN RATIO ANALYSIS")
    print("=" * 70)
    analyze_golden_ratio_properties()
