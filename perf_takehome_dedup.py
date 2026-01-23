"""
New approach: Deduplicate loads by exploiting the fact that many elements
access the same tree nodes in early rounds.

Key insight from analysis:
- Round 0: 1 unique index (all access node 0)
- Round 1: 2 unique indices  
- Round 2: 4 unique indices
- Round 11+: wraps around, starts over with 1, 2, 4 unique indices

Strategy:
1. For early rounds (0-6), use a different algorithm that loads unique nodes once
2. For later rounds, use the regular vectorized approach

This could save thousands of loads!
