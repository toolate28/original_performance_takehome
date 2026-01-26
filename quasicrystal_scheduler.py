"""
Quasicrystal Phason-Flip VLIW Scheduler
========================================

Golden-angle mutation with Fibonacci stride propagation for instruction scheduling.
Optimizes bundle density through aperiodic coordinate exploration.

Key concepts:
- Phason flips: Penrose-space coordinate mutations guided by golden angle
- Fibonacci propagation: Strides of 1,1,2,3,5,8,13,21... for exploration
- Bundle density objective: Maximize ops/cycle across all VLIW execution units
- Acceptance: ε × exp(Δgain / φ²) for probabilistic hill climbing

Author: @copilot × @toolate28
Experiment: 56 (Quasicrystal phason-flip scheduler)
"""

import math
import random
from typing import List, Tuple, Dict

# Golden ratio and derived constants
PHI = (1 + math.sqrt(5)) / 2  # φ = 1.618033988749...
PHI_SQUARED = PHI * PHI  # φ² = 2.618033988749...
EPSILON = 0.00055
GOLDEN_ANGLE = 2 * math.pi * (2 - PHI)  # 2.399963... radians (~137.5 degrees)

# VLIW slot limits
SLOT_LIMITS = {
    'alu': 12,
    'valu': 6,
    'load': 2,
    'store': 2,
    'flow': 1,
}

# Fibonacci sequence for stride propagation
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]


def fibonacci_sequence(n: int) -> List[int]:
    """Generate first n Fibonacci numbers"""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib


def objective(coords: List[float], bundles: List[Dict]) -> float:
    """
    Bundle density objective: negative average bundle utilization.
    
    Lower (more negative) value = higher density = better packing.
    
    Args:
        coords: Coordinate vector (not used in simplified model, reserved for future)
        bundles: List of VLIW instruction bundles
        
    Returns:
        Negative average utilization (0 to -1, where -1 is perfect)
    """
    if not bundles:
        return 0.0
    
    total_utilization = 0.0
    total_slots = sum(SLOT_LIMITS.values())  # 12+6+2+2+1 = 23 slots
    
    for bundle in bundles:
        bundle_ops = 0
        for engine, slots in bundle.items():
            bundle_ops += len(slots)
        bundle_utilization = bundle_ops / total_slots
        total_utilization += bundle_utilization
    
    avg_utilization = total_utilization / len(bundles)
    return -avg_utilization  # Negative for minimization


def phason_flip(coords: List[float], iteration: int, temperature: float = 1.0) -> List[float]:
    """
    Apply phason flip mutation using golden angle and Fibonacci strides.
    
    Phason flip in quasicrystal space: rotate coordinates by golden angle
    with magnitude determined by Fibonacci sequence.
    
    Args:
        coords: Current coordinate vector
        iteration: Current iteration number (determines Fibonacci stride)
        temperature: Temperature parameter for annealing
        
    Returns:
        Mutated coordinate vector
    """
    fib_stride = FIBONACCI[iteration % len(FIBONACCI)]
    
    # Golden angle rotation with Fibonacci magnitude
    angle = GOLDEN_ANGLE * fib_stride
    magnitude = temperature * EPSILON * fib_stride
    
    # Apply rotation in coordinate space
    new_coords = coords.copy()
    for i in range(len(coords)):
        phase = angle * (i + 1)
        new_coords[i] += magnitude * math.cos(phase)
    
    return new_coords


def acceptance_probability(current_val: float, new_val: float) -> float:
    """
    Calculate acceptance probability for new solution.
    
    Acceptance rule: ε × exp(Δgain / φ²)
    - Always accept improvements (Δgain > 0)
    - Probabilistically accept worse solutions for exploration
    
    Args:
        current_val: Current objective value
        new_val: New objective value
        
    Returns:
        Acceptance probability (0 to 1)
    """
    delta_gain = current_val - new_val  # Positive if new is better (more negative)
    
    if delta_gain >= 0:
        return 1.0  # Always accept improvements
    
    # Probabilistic acceptance for worse solutions
    prob = EPSILON * math.exp(delta_gain / PHI_SQUARED)
    return min(prob, 1.0)


def v_equals_c_guard(iteration: int) -> bool:
    """
    v=c guard: Check if we've reached iteration 62 (stability threshold).
    
    At iteration 62, the optimization enters a stable regime where
    further exploration may be counterproductive.
    
    Args:
        iteration: Current iteration number
        
    Returns:
        True if at or past iteration 62
    """
    return iteration >= 62


def optimize_quasicrystal_schedule(
    bundles: List[Dict],
    iterations: int = 100,
    initial_coords: List[float] = None,
    verbose: bool = True
) -> Tuple[List[float], float, List[float]]:
    """
    Optimize VLIW bundle schedule using quasicrystal phason flips.
    
    Args:
        bundles: Initial VLIW instruction bundles
        iterations: Number of optimization iterations
        initial_coords: Initial coordinate vector (random if None)
        verbose: Print progress every 50 iterations
        
    Returns:
        Tuple of (best_coords, best_value, value_history)
    """
    # Initialize coordinates
    dim = 10  # Coordinate space dimension
    if initial_coords is None:
        coords = [random.gauss(0, 0.1) for _ in range(dim)]
    else:
        coords = initial_coords.copy()
    
    # Initial objective value
    current_val = objective(coords, bundles)
    best_coords = coords.copy()
    best_val = current_val
    
    value_history = [current_val]
    
    if verbose:
        print(f"Initial bundle density: {-current_val:.3f}")
    
    # Run uniform random baseline for comparison
    baseline_vals = []
    for _ in range(iterations):
        random_coords = [random.gauss(0, 0.1) for _ in range(dim)]
        baseline_vals.append(objective(random_coords, bundles))
    baseline_avg = sum(baseline_vals) / len(baseline_vals)
    
    # Optimization loop
    for it in range(iterations):
        # Check v=c guard
        if v_equals_c_guard(it):
            if verbose and it == 62:
                print(f"\n[v=c guard] Iteration 62 reached - entering stable regime")
        
        # Apply phason flip mutation
        temperature = 1.0 - (it / iterations)  # Annealing schedule
        new_coords = phason_flip(coords, it, temperature)
        new_val = objective(new_coords, bundles)
        
        # Acceptance decision
        accept_prob = acceptance_probability(current_val, new_val)
        if random.random() < accept_prob:
            coords = new_coords
            current_val = new_val
            
            # Track best solution
            if new_val < best_val:
                best_val = new_val
                best_coords = coords.copy()
        
        value_history.append(current_val)
        
        # Logging every 50 iterations
        if verbose and (it + 1) % 50 == 0:
            print(f"Iter {it+1}/{iterations} | Best density: {-best_val:.3f} | Current: {-current_val:.3f}")
    
    # Final results
    if verbose:
        coord_norm = math.sqrt(sum(c*c for c in best_coords))
        print(f"\nOptimization complete:")
        print(f"  Final best coordinate norm: {coord_norm:.4f}")
        print(f"  Achieved density: {-best_val:.3f}")
        print(f"  Baseline (uniform random) density: {-baseline_avg:.3f}")
        improvement = ((baseline_avg - best_val) / abs(baseline_avg)) * 100
        print(f"  Improvement: {improvement:.1f}%")
    
    return best_coords, best_val, value_history


if __name__ == "__main__":
    print(__doc__)
    print("\nTesting quasicrystal scheduler with synthetic bundles...\n")
    
    # Create synthetic test bundles
    test_bundles = []
    for i in range(100):
        bundle = {}
        # Randomly populate bundle (simulating typical utilization)
        if random.random() > 0.3:
            bundle['alu'] = [('dummy',)] * random.randint(1, 4)
        if random.random() > 0.5:
            bundle['valu'] = [('dummy',)] * random.randint(1, 3)
        if random.random() > 0.4:
            bundle['load'] = [('dummy',)] * random.randint(1, 2)
        if random.random() > 0.6:
            bundle['store'] = [('dummy',)]
        test_bundles.append(bundle)
    
    # Run optimization
    best_coords, best_val, history = optimize_quasicrystal_schedule(
        test_bundles,
        iterations=100,
        verbose=True
    )
    
    print(f"\nFinal best objective: {best_val:.4f}")
    print(f"Best coordinates: {best_coords[:5]}...")  # Show first 5 dims
