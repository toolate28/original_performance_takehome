"""
Test the phason-optimized kernel
"""

import perf_takehome
from phason_flip_optimizer import vector_phason_flip_pass

# Build original kernel
print("Building original kernel...")
kb = perf_takehome.KernelBuilder()
kb.build_kernel(forest_height=10, n_nodes=2047, batch_size=256, rounds=16)

print(f"Original bundles: {len(kb.instrs)}")

# Apply phason optimization
print("\nApplying phason flip optimization...")
optimized_instrs = vector_phason_flip_pass(kb.instrs, num_passes=42, seed=42)

print(f"Optimized bundles: {len(optimized_instrs)}")
print(f"Reduction: {100*(1 - len(optimized_instrs)/len(kb.instrs)):.1f}%")

# Monkey-patch to apply phason optimization
original_build_kernel = perf_takehome.KernelBuilder.build_kernel

def phason_build_kernel(self, forest_height, n_nodes, batch_size, rounds):
    # Build original
    original_build_kernel(self, forest_height, n_nodes, batch_size, rounds)
    # Apply phason optimization
    print(f"Before phason: {len(self.instrs)} bundles")
    self.instrs = vector_phason_flip_pass(self.instrs, num_passes=42, seed=42)
    print(f"After phason: {len(self.instrs)} bundles")

# Test it
print("\nTesting phason-optimized kernel...")
perf_takehome.KernelBuilder.build_kernel = phason_build_kernel

from perf_takehome import do_kernel_test
cycles = do_kernel_test(10, 16, 256)

print("="*70)
print(f"PHASON-OPTIMIZED KERNEL: {cycles} cycles")
print(f"Original: 5,028 cycles")
print(f"Target: 1,487 cycles")
print("="*70)

if cycles < 5028:
    improvement = 5028 - cycles
    print(f"✓✓✓ BREAKTHROUGH! Improved by {improvement} cycles ({100*improvement/5028:.1f}%)")
    if cycles < 1487:
        print(f"✓✓✓✓✓ SURPASSED OPUS 4.5 TARGET!")
        print(f"VICTORY! {1487 - cycles} cycles better than target!")
else:
    print(f"Cycles: {cycles} (need to optimize further)")

print("="*70)
