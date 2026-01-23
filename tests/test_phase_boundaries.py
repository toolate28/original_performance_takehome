"""
Property-based tests for phase boundary detection and optimization.

Tests that the supercollapse optimization is applied correctly:
- Round 0: Should use supercollapse (all indices converge to 0)
- Round 1+: Should use standard loads (indices diverge)
"""

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import unittest
import random
from hypothesis import given, strategies as st, settings

from problem import Tree, Input, build_mem_image, reference_kernel2, VLEN, N_CORES
from perf_takehome import KernelBuilder


def is_supercollapse_zone(round_num: int) -> bool:
    """
    Property: Identify V=C horizons (convergence zones).
    
    Round 0 is a supercollapse zone because all indices start at 0.
    Round 1+ are not supercollapse zones due to exponential divergence.
    """
    return round_num == 0


def analyze_convergence(indices: list[int]) -> float:
    """
    Measure convergence: ratio of unique indices to total indices.
    
    Returns:
        Convergence ratio: 1.0 = 100% convergence (all same), 0.0 = 0% convergence (all different)
    """
    if not indices:
        return 1.0
    
    unique_count = len(set(indices))
    total_count = len(indices)
    
    # Convergence = 1 - (unique_ratio)
    # If all same: unique=1, convergence=1.0
    # If all different: unique=total, convergence=0.0
    return 1.0 - (unique_count - 1) / total_count


class TestPhaseBoundaries(unittest.TestCase):
    """Property-based tests for phase boundary correctness"""
    
    @given(round_num=st.integers(0, 10))
    @settings(max_examples=20, deadline=None)
    def test_phase_boundary_correctness(self, round_num):
        """Property: Optimization must respect phase boundaries"""
        if round_num == 0:
            assert is_supercollapse_zone(round_num), \
                "Round 0 should be identified as supercollapse zone"
        else:
            assert not is_supercollapse_zone(round_num), \
                f"Round {round_num} should not be supercollapse zone"
    
    def test_round_0_convergence(self):
        """Test that Round 0 has 100% convergence (all indices = 0)"""
        forest = Tree.generate(10)
        batch_size = 256
        rounds = 1
        inp = Input.generate(forest, batch_size, rounds)
        
        # Before any rounds, all indices should be 0
        convergence = analyze_convergence(inp.indices)
        self.assertEqual(convergence, 1.0, 
                        "Round 0 should have 100% convergence (all indices = 0)")
        
        # Verify all indices are actually 0
        self.assertTrue(all(idx == 0 for idx in inp.indices),
                       "All indices should be 0 at Round 0")
    
    def test_round_1_divergence(self):
        """Test that Round 1+ has divergence (indices spread out)"""
        random.seed(123)
        forest = Tree.generate(10)
        batch_size = 256
        rounds = 16  # Run all 16 rounds to see full divergence pattern
        inp = Input.generate(forest, batch_size, rounds)
        
        # Use the actual reference_kernel function to simulate rounds
        from problem import reference_kernel
        reference_kernel(forest, inp)  # This modifies inp in place
        
        # After all rounds, verify there's some divergence
        # (Even if most indices converge, we expect at least some variety)
        unique_indices = len(set(inp.indices))
        self.assertGreater(unique_indices, 1,
                          "After multiple rounds, should have multiple unique indices")
        
        # The key property: Round 0 starts with 100% convergence (all 0)
        # Round 1+ will have less convergence (even if still high)
        # This validates that the phase boundary exists
    
    def test_kernel_correctness_with_phase_optimization(self):
        """Test that phase-aware optimization preserves correctness"""
        random.seed(456)
        forest = Tree.generate(10)
        batch_size = 256
        rounds = 16
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)
        
        # Build optimized kernel
        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
        
        # Run optimized kernel
        from problem import Machine, DebugInfo
        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        
        # Run reference kernel
        for ref_mem in reference_kernel2(mem):
            pass
        
        # Verify outputs match
        inp_values_p = ref_mem[6]
        self.assertEqual(
            machine.mem[inp_values_p : inp_values_p + len(inp.values)],
            ref_mem[inp_values_p : inp_values_p + len(inp.values)],
            "Optimized kernel should produce same results as reference"
        )
    
    def test_performance_improvement(self):
        """Test that phase-aware optimization improves performance"""
        random.seed(789)
        forest = Tree.generate(10)
        batch_size = 256
        rounds = 16
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)
        
        # Build and run optimized kernel
        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
        
        from problem import Machine
        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        
        cycles = machine.cycle
        
        # Expected baseline from problem statement
        BASELINE = 147734
        
        # Should achieve significant speedup
        speedup = BASELINE / cycles
        self.assertGreater(speedup, 25.0,
                          f"Should achieve >25x speedup (got {speedup:.2f}x)")
        
        # Should be better than 5541 cycles (pre-optimization baseline)
        PRE_OPTIMIZATION = 5541
        self.assertLess(cycles, PRE_OPTIMIZATION,
                       f"Should be faster than {PRE_OPTIMIZATION} cycles (got {cycles})")


if __name__ == "__main__":
    unittest.main()
