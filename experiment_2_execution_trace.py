"""
EXPERIMENT 2: Understand why fewer bundles = more cycles
"""

import perf_takehome
from perf_takehome_vselect import VSelectCacheKernel
from problem import Machine, DebugInfo, build_mem_image, Tree, Input, VLEN

# Build tiny test case
forest = Tree.generate(height=4)  # Smaller tree
inp = Input.generate(forest, batch_size=16, rounds=2)  # Fewer elements, fewer rounds
mem = build_mem_image(forest, inp)

print("Testing with tiny case: height=4, batch_size=16, rounds=2")

# Build 5K kernel
kb_5k = perf_takehome.KernelBuilder()
kb_5k.build_kernel(forest.height, len(forest.values), len(inp.indices), inp.rounds)

print(f"\n5K kernel: {len(kb_5k.instrs)} bundles in program")

# Run and measure
debug_info = DebugInfo(scratch_map=kb_5k.scratch_map_inverse())
machine = Machine(mem, kb_5k.instrs, debug_info)
machine.run()
cycles_5k = machine.cycle

print(f"5K kernel executed: {cycles_5k} cycles")

# Build 12K kernel
kb_12k = VSelectCacheKernel()
kb_12k.build_kernel(forest.height, len(forest.values), len(inp.indices), inp.rounds)

print(f"\n12K kernel: {len(kb_12k.instrs)} bundles in program")

# Reset memory
inp2 = Input.generate(forest, batch_size=16, rounds=2)
mem2 = build_mem_image(forest, inp2)

debug_info2 = DebugInfo(scratch_map=kb_12k.scratch_map_inverse())
machine2 = Machine(mem2, kb_12k.instrs, debug_info2)
machine2.run()
cycles_12k = machine2.cycle

print(f"12K kernel executed: {cycles_12k} cycles")

print("\n" + "="*70)
print("KEY INSIGHT:")
print("="*70)
print(f"5K: {len(kb_5k.instrs)} bundles → {cycles_5k} cycles → {cycles_5k/len(kb_5k.instrs):.2f} cycles/bundle")
print(f"12K: {len(kb_12k.instrs)} bundles → {cycles_12k} cycles → {cycles_12k/len(kb_12k.instrs):.2f} cycles/bundle")

if len(kb_5k.instrs) == cycles_5k and len(kb_12k.instrs) == cycles_12k:
    print("\nRESULT: Cycles = bundle count! No loops, straight-line execution.")
    print("The 'bundles' count IS the cycle count (one bundle per cycle).")
else:
    print("\nRESULT: Cycles ≠ bundle count. Investigating...")

# Now test full size
print("\n" + "="*70)
print("FULL SIZE TEST:")
print("="*70)

forest_full = Tree.generate(height=10)
inp_full = Input.generate(forest_full, batch_size=256, rounds=16)

kb_5k_full = perf_takehome.KernelBuilder()
kb_5k_full.build_kernel(10, 2047, 256, 16)

kb_12k_full = VSelectCacheKernel()
kb_12k_full.build_kernel(10, 2047, 256, 16)

print(f"\n5K kernel: {len(kb_5k_full.instrs)} bundles")
print(f"12K kernel: {len(kb_12k_full.instrs)} bundles")
print(f"\nIf bundles = cycles:")
print(f"  5K would take: {len(kb_5k_full.instrs)} cycles (but actually takes 5,028)")
print(f"  12K would take: {len(kb_12k_full.instrs)} bundles (should be ~12,069)")

print("\nCONCLUSION:")
print("The 5K kernel has MASSIVE loop unrolling (50K bundles for 256 elements × 16 rounds)")
print("The 12K kernel has LESS unrolling (12K bundles)")
print("But 5K's aggressive unrolling enables better packing within each iteration!")
print("\nThe path forward: Match 12K's structure but PACK operations like 5K does.")
