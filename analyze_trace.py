"""Quick analysis of instruction utilization"""
from perf_takehome import *

kb = KernelBuilder()
kb.build_kernel(10, 1024, 256, 16)

# Count operations by engine
by_engine = {}
for instr in kb.instrs:
    for engine, slots in instr.items():
        if engine not in by_engine:
            by_engine[engine] = []
        by_engine[engine].append(len(slots))

print("\nInstruction count analysis:")
print(f"Total instructions: {len(kb.instrs)}")
for engine in ["alu", "load", "store", "flow"]:
    if engine in by_engine:
        counts = by_engine[engine]
        print(f"{engine}: {len(counts)} bundles, avg {sum(counts)/len(counts):.1f} slots/bundle, max {max(counts)}")

# Look at a few sample bundles
print("\nSample bundles:")
for i in range(min(20, len(kb.instrs))):
    print(f"Cycle {i}: {[(k, len(v)) for k, v in kb.instrs[i].items()]}")
