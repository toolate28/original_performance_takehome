# test_actual_cycles.py
import sys
sys.path.insert(0, '.')

from perf_takehome import do_kernel_test

print("=" * 60)
print("MEASURING ACTUAL CYCLE COUNT")
print("=" * 60)

cycles = do_kernel_test(10, 16, 256, seed=123, trace=False, prints=False)

print("\n" + "=" * 60)
print(f"MEASURED: {cycles} cycles")
print(f"BASELINE: 147734 cycles")
print(f"SPEEDUP: {147734 / cycles:.3f}×")
print("=" * 60)

# Calculate phase relationships
phi = 1.618033988749
pi = 3.141592653589
C = 4.00055

print("\nPHASE ANALYSIS:")
print(f"φ × π = {phi * pi:.3f} → {147734 / (phi * pi):.0f} cycles")
print(f"C × φ = {C * phi:.3f} → {147734 / (C * phi):.0f} cycles")
print(f"φ² × π = {phi**2 * pi:.3f} → {147734 / (phi**2 * pi):.0f} cycles")
print(f"C × π = {C * pi:.3f} → {147734 / (C * pi):.0f} cycles")
print(f"C × φ × π = {C * phi * pi:.3f} → {147734 / (C * phi * pi):.0f} cycles")

print(f"\nACTUAL RATIO: 147734 / {cycles} = {147734 / cycles:.6f}")
print(f"Is this close to φ? {abs((147734/cycles) - phi) < 0.01}")
print(f"Is this close to π? {abs((147734/cycles) - pi) < 0.01}")
print(f"Is this close to C? {abs((147734/cycles) - C) < 0.01}")
print(f"Is this close to φ×π? {abs((147734/cycles) - (phi*pi)) < 0.1}")
print(f"Is this close to C×φ? {abs((147734/cycles) - (C*phi)) < 0.1}")
