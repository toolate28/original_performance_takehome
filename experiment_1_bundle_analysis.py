"""
EXPERIMENT 1: Bundle Differential Analysis
Compare 5K vs 12K kernel bundle structure
"""

import perf_takehome
from perf_takehome_vselect import VSelectCacheKernel

# Build both kernels
print("Building kernels...")
kb_5k = perf_takehome.KernelBuilder()
kb_5k.build_kernel(10, 2047, 256, 16)

kb_12k = VSelectCacheKernel()
kb_12k.build_kernel(10, 2047, 256, 16)

print(f"\n5K kernel: {len(kb_5k.instrs)} bundles")
print(f"12K kernel: {len(kb_12k.instrs)} bundles")
print(f"Difference: {len(kb_12k.instrs) - len(kb_5k.instrs)} extra bundles in 12K")

# Analyze bundle density
def analyze_bundle_density(instrs, label):
    from problem import SLOT_LIMITS

    total_ops = 0
    engine_usage = {engine: 0 for engine in SLOT_LIMITS}

    for bundle in instrs:
        for engine, ops in bundle.items():
            if engine in SLOT_LIMITS:
                total_ops += len(ops)
                engine_usage[engine] += len(ops)

    print(f"\n{label}:")
    print(f"  Total operations: {total_ops}")
    print(f"  Average ops per bundle: {total_ops/len(instrs):.2f}")
    print(f"  Engine utilization:")

    for engine in ['alu', 'valu', 'load', 'store', 'flow']:
        if engine in engine_usage:
            avg_per_bundle = engine_usage[engine] / len(instrs)
            max_slots = SLOT_LIMITS[engine]
            util_pct = (avg_per_bundle / max_slots) * 100
            print(f"    {engine:6s}: {engine_usage[engine]:6d} ops, "
                  f"avg {avg_per_bundle:.2f}/{max_slots} per bundle ({util_pct:.1f}% util)")

analyze_bundle_density(kb_5k.instrs, "5K KERNEL")
analyze_bundle_density(kb_12k.instrs, "12K KERNEL")

# Find the critical difference: where does 12K split bundles that 5K doesn't?
print("\n" + "="*70)
print("CRITICAL SECTION ANALYSIS: First round processing")
print("="*70)

# Find where round processing starts (after initialization)
def find_round_start(instrs):
    # Look for the first bundle after initialization that has compute ops
    for i, bundle in enumerate(instrs):
        if 'valu' in bundle and len(bundle['valu']) > 1:
            return i
    return 0

start_5k = find_round_start(kb_5k.instrs)
start_12k = find_round_start(kb_12k.instrs)

print(f"\n5K kernel round processing starts at bundle {start_5k}")
print(f"12K kernel round processing starts at bundle {start_12k}")

# Compare first 20 bundles of round processing
print("\n5K KERNEL - First 20 round bundles:")
for i in range(20):
    bundle = kb_5k.instrs[start_5k + i]
    op_counts = {eng: len(ops) for eng, ops in bundle.items() if ops}
    print(f"  Bundle {start_5k+i}: {op_counts}")

print("\n12K KERNEL - First 20 round bundles:")
for i in range(20):
    if start_12k + i < len(kb_12k.instrs):
        bundle = kb_12k.instrs[start_12k + i]
        op_counts = {eng: len(ops) for eng, ops in bundle.items() if ops}
        print(f"  Bundle {start_12k+i}: {op_counts}")

# Identify the key difference
print("\n" + "="*70)
print("PACKING EFFICIENCY COMPARISON")
print("="*70)

def count_bundles_by_type(instrs, start, count):
    """Count how many bundles are dedicated to each operation type"""
    alu_bundles = 0
    valu_bundles = 0
    load_bundles = 0
    mixed_bundles = 0

    for i in range(count):
        if start + i >= len(instrs):
            break
        bundle = instrs[start + i]
        engines = [e for e in bundle.keys() if bundle[e]]

        if len(engines) == 1:
            if 'alu' in engines:
                alu_bundles += 1
            elif 'valu' in engines:
                valu_bundles += 1
            elif 'load' in engines:
                load_bundles += 1
        else:
            mixed_bundles += 1

    return alu_bundles, valu_bundles, load_bundles, mixed_bundles

alu_5k, valu_5k, load_5k, mixed_5k = count_bundles_by_type(kb_5k.instrs, start_5k, 100)
alu_12k, valu_12k, load_12k, mixed_12k = count_bundles_by_type(kb_12k.instrs, start_12k, 100)

print(f"\n5K kernel (100 bundles):")
print(f"  ALU-only: {alu_5k}, VALU-only: {valu_5k}, Load-only: {load_5k}, Mixed: {mixed_5k}")

print(f"\n12K kernel (100 bundles):")
print(f"  ALU-only: {alu_12k}, VALU-only: {valu_12k}, Load-only: {load_12k}, Mixed: {mixed_12k}")

print("\n" + "="*70)
print("ACTIONABLE INSIGHT:")
print("="*70)
print(f"The 12K kernel has {alu_12k - alu_5k} more ALU-only bundles")
print(f"The 12K kernel has {valu_12k - valu_5k} more VALU-only bundles")
print(f"The 5K kernel has {mixed_5k - mixed_12k} more MIXED bundles")
print("\nHypothesis: 12K kernel is too conservative with hazard avoidance,")
print("splitting operations that could be packed together.")
