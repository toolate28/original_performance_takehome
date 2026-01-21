"""Analyze how often elements in same vector access the same tree nodes"""

from problem import Tree, Input, myhash, VLEN
import random

def analyze_intra_vector_sharing(forest_height=10, batch_size=256, rounds=16, seed=123):
    """Check for node access sharing within SIMD vectors"""
    random.seed(seed)

    tree = Tree.generate(forest_height)
    n_nodes = 2 ** (forest_height + 1) - 1
    inp = Input(
        indices=[random.randrange(n_nodes) for _ in range(batch_size)],
        values=[random.randrange(2**32) for _ in range(batch_size)],
        rounds=rounds,
    )

    total_loads = 0
    unique_loads = 0
    sharing_by_round = []

    for r in range(rounds):
        round_total = 0
        round_unique = 0

        # Process in groups of VLEN (8)
        for g in range(batch_size // VLEN):
            base_i = g * VLEN

            # Get indices for this vector group
            indices_in_group = [inp.indices[base_i + vi] for vi in range(VLEN)]

            # Count unique indices
            unique_indices = len(set(indices_in_group))

            round_total += VLEN
            round_unique += unique_indices

            # Now process as normal
            for vi in range(VLEN):
                i = base_i + vi
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

    print(f"Total loads needed: {total_loads}")
    print(f"Unique loads if deduplicated within vectors: {unique_loads}")
    print(f"Potential savings: {total_loads - unique_loads} ({100*(total_loads-unique_loads)/total_loads:.1f}%)")
    print(f"Reduction factor: {total_loads / unique_loads:.2f}x\n")

    print("Per-round analysis:")
    print("Round | Total | Unique | Saved | Sharing%")
    print("------|-------|--------|-------|----------")
    for r, total, unique, saved in sharing_by_round:
        sharing_pct = 100 * saved / total
        print(f"  {r:2d}  | {total:5d} | {unique:6d} | {saved:5d} | {sharing_pct:6.1f}%")

    # Find best rounds
    best_rounds = sorted(sharing_by_round, key=lambda x: x[3], reverse=True)[:5]
    print(f"\nTop 5 rounds with most sharing:")
    for r, total, unique, saved in best_rounds:
        print(f"  Round {r}: {saved} duplicate loads ({100*saved/total:.1f}% sharing)")

if __name__ == "__main__":
    analyze_intra_vector_sharing()
