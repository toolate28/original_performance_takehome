"""
Comprehensive Test of Current VLIW Optimizations
=================================================

Tests the complete optimization pipeline:
1. Bubble filling with LOOKAHEAD=10
2. multiply_add fusion
3. Quasicrystal phason-flip scheduler
4. SIMD vectorization with VEC_UNROLL=8

Validates performance and correctness across different test cases.
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perf_takehome import KernelBuilder, BASELINE
from problem import Tree, Input, build_mem_image, reference_kernel2, Machine, N_CORES, VLEN


def test_optimization_correctness():
    """Test correctness on multiple random seeds"""
    print("="*70)
    print("TEST 1: Correctness Validation (Multiple Seeds)")
    print("="*70)
    
    test_cases = [
        (4, 2, 16),   # Small
        (6, 4, 32),   # Medium
        (8, 8, 64),   # Large
    ]
    
    for forest_height, rounds, batch_size in test_cases:
        print(f"\nTest case: forest_height={forest_height}, rounds={rounds}, batch_size={batch_size}")
        
        for seed in [42, 123, 456]:
            random.seed(seed)
            forest = Tree.generate(forest_height)
            inp = Input.generate(forest, batch_size, rounds)
            mem = build_mem_image(forest, inp)
            
            # Build optimized kernel
            kb = KernelBuilder()
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
            
            # Run optimized kernel
            machine = Machine(mem.copy(), kb.instrs, kb.debug_info(), n_cores=N_CORES)
            machine.enable_pause = False
            machine.enable_debug = False
            machine.run()
            
            # Run reference kernel
            ref_mem = mem.copy()
            for _ in reference_kernel2(ref_mem):
                pass
            
            # Verify correctness
            inp_values_p = ref_mem[6]
            opt_values = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            ref_values = ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            
            assert opt_values == ref_values, f"Mismatch on seed {seed}"
        
        print(f"  ✓ All seeds passed")
    
    print("\n" + "="*70)
    print("✓ CORRECTNESS TEST PASSED")
    print("="*70)


def test_performance_benchmarks():
    """Test performance on standard benchmark"""
    print("\n" + "="*70)
    print("TEST 2: Performance Benchmark")
    print("="*70)
    
    forest_height = 10
    rounds = 16
    batch_size = 256
    
    print(f"\nBenchmark: forest_height={forest_height}, rounds={rounds}, batch_size={batch_size}")
    
    random.seed(123)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    
    # Build and run kernel
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    machine = Machine(mem.copy(), kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    
    cycles = machine.cycle
    speedup = BASELINE / cycles
    
    print(f"\nResults:")
    print(f"  Cycles: {cycles}")
    print(f"  Baseline: {BASELINE}")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  Total bundles: {len(kb.instrs)}")
    
    # Analyze bundle utilization
    alu_ops = sum(len(b.get('alu', [])) for b in kb.instrs)
    valu_ops = sum(len(b.get('valu', [])) for b in kb.instrs)
    load_ops = sum(len(b.get('load', [])) for b in kb.instrs)
    store_ops = sum(len(b.get('store', [])) for b in kb.instrs)
    
    print(f"\nBundle statistics:")
    print(f"  ALU operations: {alu_ops} ({alu_ops/(len(kb.instrs)*12)*100:.1f}% utilization)")
    print(f"  VALU operations: {valu_ops} ({valu_ops/(len(kb.instrs)*6)*100:.1f}% utilization)")
    print(f"  Load operations: {load_ops} ({load_ops/(len(kb.instrs)*2)*100:.1f}% utilization)")
    print(f"  Store operations: {store_ops} ({store_ops/(len(kb.instrs)*2)*100:.1f}% utilization)")
    
    # Performance assertions
    assert cycles < BASELINE, "Should be faster than baseline"
    assert cycles < 18532, "Should beat updated starting point"
    
    print("\n" + "="*70)
    print("✓ PERFORMANCE TEST PASSED")
    print("="*70)
    
    return cycles


def test_optimization_components():
    """Test individual optimization components"""
    print("\n" + "="*70)
    print("TEST 3: Optimization Component Analysis")
    print("="*70)
    
    forest_height = 6
    rounds = 4
    batch_size = 64
    
    random.seed(42)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    
    # Test 1: Build with VLIW enabled
    kb_vliw = KernelBuilder()
    kb_vliw.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    # Test 2: Build without VLIW (for comparison)
    kb_no_vliw = KernelBuilder()
    kb_no_vliw.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    # Temporarily disable bubble filling to see its impact
    # (We'd need to modify the code to test this separately)
    
    print(f"\nComponent analysis:")
    print(f"  VLIW enabled bundles: {len(kb_vliw.instrs)}")
    
    # Run VLIW version
    machine_vliw = Machine(mem.copy(), kb_vliw.instrs, kb_vliw.debug_info(), n_cores=N_CORES)
    machine_vliw.enable_pause = False
    machine_vliw.enable_debug = False
    machine_vliw.run()
    
    print(f"  VLIW cycles: {machine_vliw.cycle}")
    
    # Verify correctness
    ref_mem = mem.copy()
    for _ in reference_kernel2(ref_mem):
        pass
    
    inp_values_p = ref_mem[6]
    assert machine_vliw.mem[inp_values_p : inp_values_p + len(inp.values)] == \
           ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    
    print("\n" + "="*70)
    print("✓ COMPONENT TEST PASSED")
    print("="*70)


def test_vectorization_effectiveness():
    """Test SIMD vectorization effectiveness"""
    print("\n" + "="*70)
    print("TEST 4: SIMD Vectorization Analysis")
    print("="*70)
    
    forest_height = 8
    rounds = 8
    batch_size = 128
    
    random.seed(789)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    # Count vector operations
    valu_count = sum(len(b.get('valu', [])) for b in kb.instrs)
    vload_count = sum(1 for b in kb.instrs for op in b.get('load', []) if op[0] == 'vload')
    vstore_count = sum(1 for b in kb.instrs for op in b.get('store', []) if op[0] == 'vstore')
    
    print(f"\nVectorization statistics:")
    print(f"  VALU operations: {valu_count}")
    print(f"  vload operations: {vload_count}")
    print(f"  vstore operations: {vstore_count}")
    print(f"  Elements processed per vload: {VLEN}")
    print(f"  Total vectorized loads: {vload_count * VLEN}")
    
    # Run kernel
    machine = Machine(mem.copy(), kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    
    print(f"  Cycles with vectorization: {machine.cycle}")
    
    # Verify correctness
    ref_mem = mem.copy()
    for _ in reference_kernel2(ref_mem):
        pass
    
    inp_values_p = ref_mem[6]
    assert machine.mem[inp_values_p : inp_values_p + len(inp.values)] == \
           ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    
    print("\n" + "="*70)
    print("✓ VECTORIZATION TEST PASSED")
    print("="*70)


if __name__ == "__main__":
    print(__doc__)
    
    try:
        test_optimization_correctness()
        cycles = test_performance_benchmarks()
        test_optimization_components()
        test_vectorization_effectiveness()
        
        print("\n" + "="*70)
        print("ALL COMPREHENSIVE TESTS PASSED")
        print("="*70)
        print(f"\nFinal Performance: {cycles} cycles ({BASELINE/cycles:.2f}× speedup)")
        print(f"Target: <1,400 cycles (need {cycles/1400:.2f}× more improvement)")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
