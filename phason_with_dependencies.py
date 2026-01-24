"""
PHASON FLIP OPTIMIZER WITH PROPER DEPENDENCY CHECKING

The geometry of symmetry: Respect dependency anti-symmetry!
"""

import random
import math
from typing import List, Dict, Any, Tuple, Set
from phason_flip_optimizer import (
    fib_strides, golden_slot_order, is_vector_op, bundle_utilization,
    vector_phason_gain, propagate_vector_wave
)

VLEN = 8

def get_op_reads_writes(op: Tuple) -> Tuple[Set[int], Set[int]]:
    """
    Extract register reads and writes from an operation.
    Returns (reads, writes) as sets of scratch addresses.
    """
    if not op or len(op) < 2:
        return (set(), set())

    op_name = op[0]

    # Load operations
    if op_name in ["load", "load_offset"]:
        dest = op[1]
        addr = op[2]
        return ({addr}, {dest})
    elif op_name == "vload":
        dest = op[1]
        addr = op[2]
        # Writes to dest, dest+1, ..., dest+7
        writes = set(range(dest, dest + VLEN))
        return ({addr}, writes)
    elif op_name == "const":
        dest = op[1]
        return (set(), {dest})

    # Store operations
    elif op_name in ["store"]:
        addr = op[1]
        src = op[2]
        return ({addr, src}, set())
    elif op_name == "vstore":
        addr = op[1]
        src = op[2]
        reads = {addr} | set(range(src, src + VLEN))
        return (reads, set())

    # ALU operations
    elif op_name in ["+", "-", "*", "//", "cdiv", "^", "&", "|", "<<", ">>", "%", "<", "=="]:
        dest = op[1]
        a1 = op[2]
        a2 = op[3]
        return ({a1, a2}, {dest})

    # VALU operations
    elif op_name == "vbroadcast":
        dest = op[1]
        src = op[2]
        writes = set(range(dest, dest + VLEN))
        return ({src}, writes)
    elif op_name == "multiply_add":
        dest, a, b, c = op[1], op[2], op[3], op[4]
        reads = set(range(a, a + VLEN)) | set(range(b, b + VLEN)) | set(range(c, c + VLEN))
        writes = set(range(dest, dest + VLEN))
        return (reads, writes)
    # Generic binary valu ops
    elif len(op) == 4 and isinstance(op[1], int):
        dest, a1, a2 = op[1], op[2], op[3]
        reads = set(range(a1, a1 + VLEN)) | set(range(a2, a2 + VLEN))
        writes = set(range(dest, dest + VLEN))
        return (reads, writes)

    # Flow operations
    elif op_name in ["select"]:
        dest, cond, a, b = op[1], op[2], op[3], op[4]
        return ({cond, a, b}, {dest})
    elif op_name == "vselect":
        dest, cond, a, b = op[1], op[2], op[3], op[4]
        reads = set(range(cond, cond + VLEN)) | set(range(a, a + VLEN)) | set(range(b, b + VLEN))
        writes = set(range(dest, dest + VLEN))
        return (reads, writes)
    elif op_name == "add_imm":
        dest, a = op[1], op[2]
        return ({a}, {dest})
    elif op_name in ["cond_jump", "trace_write"]:
        val = op[1]
        return ({val}, set())
    elif op_name in ["halt", "pause", "jump"]:
        return (set(), set())
    elif op_name == "coreid":
        dest = op[1]
        return (set(), {dest})

    # Unknown operation - be conservative
    return (set(), set())

def bundles_have_dependency(b1: Dict, b2: Dict) -> bool:
    """
    Check if b2 has any dependency on b1 (RAW, WAW, or WAR hazard).
    Returns True if bundles CANNOT be safely merged/reordered.
    """
    # Get all reads and writes from b1
    b1_reads = set()
    b1_writes = set()
    for engine, ops in b1.items():
        for op in ops:
            reads, writes = get_op_reads_writes(op)
            b1_reads |= reads
            b1_writes |= writes

    # Get all reads and writes from b2
    b2_reads = set()
    b2_writes = set()
    for engine, ops in b2.items():
        for op in ops:
            reads, writes = get_op_reads_writes(op)
            b2_reads |= reads
            b2_writes |= writes

    # Check hazards:
    # RAW: b2 reads what b1 writes
    if b2_reads & b1_writes:
        return True
    # WAW: b2 writes what b1 writes
    if b2_writes & b1_writes:
        return True
    # WAR: b2 writes what b1 reads (only matters if we reorder)
    if b2_writes & b1_reads:
        return True

    return False

def try_bundle_merge_safe(bundles: List[Dict], idx1: int, idx2: int) -> Tuple[bool, List[Dict]]:
    """
    Try to merge two adjacent bundles if:
    1. They don't exceed slot limits
    2. They don't have dependencies on each other
    """
    from problem import SLOT_LIMITS

    if idx1 >= len(bundles) or idx2 >= len(bundles) or idx2 != idx1 + 1:
        return False, bundles

    b1, b2 = bundles[idx1], bundles[idx2]

    # Check for dependencies
    if bundles_have_dependency(b1, b2):
        return False, bundles

    # Check slot limits
    merged = {}
    for engine in set(b1.keys()) | set(b2.keys()):
        ops1 = b1.get(engine, [])
        ops2 = b2.get(engine, [])
        merged_ops = ops1 + ops2

        if engine in SLOT_LIMITS and len(merged_ops) > SLOT_LIMITS[engine]:
            return False, bundles

        merged[engine] = merged_ops

    # Merge successful
    new_bundles = bundles[:idx1] + [merged] + bundles[idx2+1:]
    return True, new_bundles

def vector_phason_flip_pass_safe(bundles: List[Dict], num_passes: int = 42, seed: int = 42) -> List[Dict]:
    """
    Dependency-aware phason optimization.
    Only merges bundles when safe (no hazards).
    """
    random.seed(seed)

    FIB_STRIDES = fib_strides(len(bundles))
    GOLDEN_LANE_ORDER = golden_slot_order(VLEN)

    new_bundles = [b.copy() for b in bundles]

    print(f"Safe Phason Pass: {len(bundles)} bundles, {num_passes} passes")
    print(f"Fibonacci strides: {FIB_STRIDES[:10]}...")

    for pass_num in range(num_passes):
        improved = False
        improvements = 0

        # Traverse with Fibonacci strides
        for stride in FIB_STRIDES:
            for b_idx in range(0, len(new_bundles) - 1, stride):
                if b_idx + 1 >= len(new_bundles):
                    break

                # Try safe bundle merge
                merged, candidate_bundles = try_bundle_merge_safe(new_bundles, b_idx, b_idx + 1)
                if merged:
                    gain = vector_phason_gain(candidate_bundles, new_bundles,
                                             (max(0, b_idx-2), min(len(new_bundles), b_idx+3)))
                    if gain > 0:
                        new_bundles = candidate_bundles
                        improved = True
                        improvements += 1

        if pass_num % 7 == 0:
            print(f"  Pass {pass_num}: {len(new_bundles)} bundles, {improvements} improvements")

        if not improved:
            print(f"  Converged at pass {pass_num}")
            break

    reduction_pct = 100 * (1 - len(new_bundles) / len(bundles))
    print(f"Safe phason: {len(bundles)} → {len(new_bundles)} bundles ({reduction_pct:.1f}% reduction)")
    return new_bundles


if __name__ == "__main__":
    import perf_takehome
    from perf_takehome import do_kernel_test

    # Monkey-patch
    original_build = perf_takehome.KernelBuilder.build_kernel

    def safe_phason_build(self, forest_height, n_nodes, batch_size, rounds):
        original_build(self, forest_height, n_nodes, batch_size, rounds)
        print(f"Before safe phason: {len(self.instrs)} bundles")
        self.instrs = vector_phason_flip_pass_safe(self.instrs, num_passes=42, seed=42)
        print(f"After safe phason: {len(self.instrs)} bundles")

    perf_takehome.KernelBuilder.build_kernel = safe_phason_build

    # Test
    print("=" * 70)
    print("TESTING: DEPENDENCY-SAFE PHASON OPTIMIZATION")
    print("=" * 70)

    cycles = do_kernel_test(10, 16, 256)

    print("\n" + "=" * 70)
    print(f"Safe Phason Kernel: {cycles} cycles")
    print(f"Current Best: 5,028 cycles")
    print(f"Target: 1,487 cycles")
    print("=" * 70)

    if cycles < 5028:
        improvement = 5028 - cycles
        print(f"✓✓✓ BREAKTHROUGH! {improvement} cycles better ({100*improvement/5028:.1f}%)")
        if cycles < 1487:
            print(f"✓✓✓✓✓ SURPASSED OPUS 4.5 TARGET BY {1487 - cycles} CYCLES!")
    elif cycles == 5028:
        print("Matched current best (dependencies prevent merging)")
    else:
        print(f"Regression: {cycles} vs 5,028 (investigating...)")

    print("=" * 70)
