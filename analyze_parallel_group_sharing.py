"""Analyze node sharing across 6 parallel groups (48 elements)"""

from problem import Tree, Input, myhash, VLEN
import random

def analyze_parallel_sharing(forest_height=10, batch_size=256, rounds=16, seed=123):
    """Check sharing across 6 parallel groups (48 elements processed together)"""
    random.seed(seed)

    tree = Tree.generate(forest_height)
    n_nodes = 2 ** (forest_height + 1) - 1
    inp = Input(
        indices=[random.randrange(n_nodes) for _ in range(batch_size)],
        values=[random.randrange(2**32) for _ in range(batch_size)],
        rounds=rounds,
    )

    PARALLEL_GROUPS = 6
    n_groups = batch_size // VLEN  # 32 groups total

    total_loads = 0
    unique_loads = 0
    sharing_by_round = []

    for r in range(rounds):
        round_total = 0
        round_unique = 0

        # Process in batches of PARALLEL_GROUPS
        for batch_start in range(0, n_groups, PARALLEL_GROUPS):
            batch_end = min(batch_start + PARALLEL_GROUPS, n_groups)
            batch_size_actual = batch_end - batch_start

            # Get ALL indices across the parallel batch (up to 48 elements)
            indices_in_batch = []
            for bg in range(batch_size_actual):
                g = batch_start + bg
                for vi in range(VLEN):
                    i = g * VLEN + vi
                    indices_in_batch.append(inp.indices[i])

            # Count unique
            unique_in_batch = len(set(indices_in_batch))
            round_total += len(indices_in_batch)
            round_unique += unique_in_batch

            # Process normally
            for bg in range(batch_size_actual):
                g = batch_start + bg
                for vi in range(VLEN):
                    i = g * VLEN + vi
                    idx = inp.indices[i]
                    val = inp.values[i]
                    val = myhash(val ^ tree.values[idx])
                    idx = 2 * idx + (1 if val % 2 == 0 else 2)
                    idx = 0 if idx >= n_nodes else idx
                    inp.values[i] = val
                    inp.indices[i] = idx

        sharing_by_round.append((r, round_total, round_unique, round_total - round_unique))
        total_loads += round_total
        unique_loads += round_unique

    print(f"Processing model: {PARALLEL_GROUPS} groups × {VLEN} elements = {PARALLEL_GROUPS*VLEN} parallel elements\n")
    print(f"Total loads needed: {total_loads}")
    print(f"Unique loads if deduplicated across parallel groups: {unique_loads}")
    print(f"Potential savings: {total_loads - unique_loads} ({100*(total_loads-unique_loads)/total_loads:.1f}%)")
    print(f"**Reduction factor: {total_loads / unique_loads:.2f}x**\n")

    print("Per-round analysis:")
    print("Round | Total | Unique | Saved | Sharing%")
    print("------|-------|--------|-------|----------")
    for r, total, unique, saved in sharing_by_round:
        sharing_pct = 100 * saved / total
        print(f"  {r:2d}  | {total:5d} | {unique:6d} | {saved:5d} | {sharing_pct:6.1f}%")

    best_rounds = sorted(sharing_by_round, key=lambda x: x[3], reverse=True)[:5]
    print(f"\nTop 5 rounds with most sharing:")
    for r, total, unique, saved in best_rounds:
        reduction = total / unique if unique > 0 else 1
        print(f"  Round {r}: {saved} duplicate loads ({100*saved/total:.1f}% sharing, {reduction:.2f}x reduction)")

    # Calculate impact on overall performance
    print(f"\n=== Impact Analysis ===")
    print(f"Current indirect load cycles: 2,107")
    print(f"If deduplicated: {2107 * unique_loads / total_loads:.0f} cycles")
    print(f"Savings: {2107 - 2107 * unique_loads / total_loads:.0f} cycles")
    print(f"New total: {5028 - (2107 - 2107 * unique_loads / total_loads):.0f} cycles")
    print(f"Speedup toward 1,487 target: {5028 / (5028 - (2107 - 2107 * unique_loads / total_loads)):.2f}x")

if __name__ == "__main__":
    analyze_parallel_sharing()
