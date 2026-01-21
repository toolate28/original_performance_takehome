"""
Phason Flip Optimization Pass
Uses golden ratio ordering and Fibonacci propagation to heal VLIW bundle defects.
"""

import random
import math
from typing import List, Tuple, Any, Optional, Dict

# Precomputes
PHI = (1 + math.sqrt(5)) / 2
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
EPSILON_CHAOS = 0.00055

def fib_strides(max_len: int) -> List[int]:
    """Get Fibonacci numbers smaller than max_len for stride propagation"""
    return [f for f in FIB if f < max_len and f > 0]

def golden_order(n_slots: int) -> List[int]:
    """Generate non-repeating slot ordering using golden ratio"""
    if n_slots <= 0:
        return []
    
    order = []
    for i in range(n_slots):
        angle = i / PHI
        slot = int(angle * n_slots) % n_slots
        attempts = 0
        while slot in order and attempts < n_slots * 2:
            angle += 1 / PHI
            slot = int(angle * n_slots) % n_slots
            attempts += 1
        if slot not in order:
            order.append(slot)
    
    # Fill any missing slots
    for i in range(n_slots):
        if i not in order:
            order.append(i)
    
    return order[:n_slots]

def estimate_dependency_reduction(bundle: Dict, deps_graph: Optional[Any] = None) -> float:
    """Estimate how much a bundle reduces dependency chains"""
    if not bundle or deps_graph is None:
        return 0.0
    
    # Simple heuristic: more filled engines = better
    filled_engines = sum(1 for engine_ops in bundle.values() if engine_ops)
    return filled_engines * 0.1

def local_flip(bundle: Dict, engine1: str, idx1: int, engine2: str, idx2: int) -> Optional[Dict]:
    """Swap two operations between engines in a bundle"""
    if engine1 not in bundle or engine2 not in bundle:
        return None
    
    ops1 = bundle.get(engine1, [])
    ops2 = bundle.get(engine2, [])
    
    if idx1 >= len(ops1) or idx2 >= len(ops2):
        return None
    
    # Create new bundle with swapped ops
    new_bundle = {k: v[:] for k, v in bundle.items()}
    if ops1 and ops2:
        temp = new_bundle[engine1][idx1]
        new_bundle[engine1][idx1] = new_bundle[engine2][idx2]
        new_bundle[engine2][idx2] = temp
    
    return new_bundle

def flip_gain(new_bundle: Dict, old_bundle: Dict, deps_graph: Optional[Any] = None) -> float:
    """Calculate gain heuristic: filled slots + dependency reduction + chaos"""
    old_fill = sum(len(ops) for ops in old_bundle.values())
    new_fill = sum(len(ops) for ops in new_bundle.values())
    
    dep_reduce = estimate_dependency_reduction(new_bundle, deps_graph)
    chaos_boost = random.random() * EPSILON_CHAOS * math.exp(dep_reduce)
    
    return (new_fill - old_fill) + dep_reduce + chaos_boost

def propagate_flip(bundles: List[Dict], b_idx: int, stride: int):
    """Propagate flip effects in Fibonacci-stride waves"""
    for offset in [-stride, stride]:
        p_idx = b_idx + offset
        if 0 <= p_idx < len(bundles) and bundles[p_idx]:
            # Simple wave: rotate operations slightly
            # This is a placeholder - actual propagation would be more sophisticated
            pass

def phason_flip_pass(bundles: List[Dict], 
                     deps_graph: Optional[Any] = None, 
                     num_passes: int = 42) -> List[Dict]:
    """
    Core phason flip pass using golden ratio ordering and Fibonacci propagation.
    
    Args:
        bundles: List of VLIW instruction bundles
        deps_graph: Optional dependency tracker for smarter gain calculation
        num_passes: Number of optimization passes (default 42)
    
    Returns:
        Optimized bundles with healed defects
    """
    if not bundles:
        return bundles
    
    # Get list of all engines
    all_engines = set()
    for bundle in bundles:
        all_engines.update(bundle.keys())
    engines = sorted(list(all_engines))
    
    if not engines:
        return bundles
    
    # Golden order for engine selection
    engine_order = golden_order(len(engines))
    new_bundles = [{k: v[:] for k, v in b.items()} for b in bundles]
    
    for pass_idx in range(num_passes):
        improved = False
        strides = fib_strides(len(new_bundles))
        if not strides:
            strides = [1]
        
        for stride in strides:
            for b_idx in range(0, len(new_bundles), stride):
                bundle = new_bundles[b_idx]
                
                # Try flips between different engines
                for i_idx, e1_idx in enumerate(engine_order):
                    if e1_idx >= len(engines):
                        continue
                    engine1 = engines[e1_idx]
                    
                    for e2_idx in engine_order[i_idx + 1:]:
                        if e2_idx >= len(engines):
                            continue
                        engine2 = engines[e2_idx]
                        
                        ops1 = bundle.get(engine1, [])
                        ops2 = bundle.get(engine2, [])
                        
                        if not ops1 or not ops2:
                            continue
                        
                        # Try swapping first ops
                        candidate = local_flip(bundle, engine1, 0, engine2, 0)
                        if candidate is None:
                            continue
                        
                        gain = flip_gain(candidate, bundle, deps_graph)
                        
                        # Accept if gain > 0 or with chaos probability
                        if gain > 0 or random.random() < EPSILON_CHAOS * abs(gain):
                            new_bundles[b_idx] = candidate
                            improved = True
                            propagate_flip(new_bundles, b_idx, stride)
        
        if not improved:
            break  # Reached equilibrium
    
    return new_bundles
