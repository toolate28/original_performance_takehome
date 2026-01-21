"""
MINIMAL STRUCTURE - v=c Stable Kernel
Strip all excess. Only essential constraints preserved.
Emergent property: optimal packing arises naturally from minimalism.
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class MinimalKernel(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        MINIMAL: ONE vector, ONE loop, PERFECT packing.
        Everything else eliminated.
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Initialize (minimal)
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        # Loop counters (minimal)
        round_counter = self.alloc_scratch("round_counter")
        group_counter = self.alloc_scratch("group_counter")
        group_offset = self.alloc_scratch("group_offset")

        # Working registers for ONE vector (minimal)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node = self.alloc_scratch("v_node", VLEN)
        v_tmp = self.alloc_scratch("v_tmp", VLEN)
        hash_v1 = self.alloc_scratch("hash_v1", VLEN)
        hash_v2 = self.alloc_scratch("hash_v2", VLEN)

        # Vector constants (minimal)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.emit(valu=[
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ])

        # Hash constants (pre-broadcast once)
        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1 = self.alloc_scratch(f"vc1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"vc3_{hi}", VLEN)
            self.emit(valu=[
                ("vbroadcast", v_c1, self.scratch_const(val1)),
                ("vbroadcast", v_c3, self.scratch_const(val3)),
            ])
            hash_consts.append((v_c1, v_c3))

        # Addressing (minimal - pack these!)
        tmp_addr = self.alloc_scratch("tmp_addr")
        node_addrs = [self.alloc_scratch(f"na{i}") for i in range(VLEN)]

        # Initialize counters
        self.emit(load=[("const", round_counter, 0)])

        # ============ OUTER LOOP: Rounds ============
        round_loop_start = len(self.instrs)
        self.emit(load=[("const", group_counter, 0)])

        # ============ INNER LOOP: Groups ============
        group_loop_start = len(self.instrs)

        # Calculate offset (group_counter << 3)
        self.emit(alu=[("<<", group_offset, group_counter, self.scratch_const(3))])

        # === LOAD PHASE === (pack as much as possible)
        # Load indices + values (can pack address calc with loads)
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], group_offset)])
        self.emit(load=[("vload", v_idx, tmp_addr)])
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_values_p"], group_offset)])
        self.emit(load=[("vload", v_val, tmp_addr)])

        # Calculate all 8 node addresses (pack into minimal bundles)
        # Pack 8 ALU ops across 2 bundles (4 each) since we have 12 ALU slots but dependencies
        for offset in range(0, VLEN, 4):
            ops = [("+", node_addrs[offset+i], self.scratch["forest_values_p"], v_idx + offset + i)
                   for i in range(min(4, VLEN-offset))]
            self.emit(alu=ops)

        # Load all 8 nodes (pack 2 per bundle)
        for offset in range(0, VLEN, 2):
            self.emit(load=[
                ("load", v_node + offset, node_addrs[offset]),
                ("load", v_node + offset + 1, node_addrs[offset + 1]),
            ])

        # === COMPUTE PHASE === (perfectly packed VALU)
        # XOR
        self.emit(valu=[("^", v_val, v_val, v_node)])

        # Hash - pack ops1 and ops3 together where possible
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = hash_consts[hi]
            # Both ops1 and ops3 read v_val - can execute together
            self.emit(valu=[
                (op1, hash_v1, v_val, v_c1),
                (op3, hash_v2, v_val, v_c3),
            ])
            self.emit(valu=[(op2, v_val, hash_v1, hash_v2)])

        # Index calculation (pack independent ops)
        self.emit(valu=[
            ("%", v_tmp, v_val, v_two),
            ("+", v_tmp, v_one, v_tmp),
        ])
        self.emit(valu=[
            ("*", v_idx, v_idx, v_two),
            ("+", v_idx, v_idx, v_tmp),
        ])

        # Bounds check
        self.emit(valu=[
            ("<", v_tmp, v_idx, v_n_nodes),
            ("*", v_idx, v_idx, v_tmp),
        ])

        # === STORE PHASE === (pack address calc with stores)
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], group_offset)])
        self.emit(store=[("vstore", tmp_addr, v_idx)])
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_values_p"], group_offset)])
        self.emit(store=[("vstore", tmp_addr, v_val)])

        # Increment group counter
        self.emit(flow=[("add_imm", group_counter, group_counter, 1)])

        # Loop back if more groups
        self.emit(alu=[("<", tmp1, group_counter, self.scratch_const(32))])
        self.emit(flow=[("cond_jump", tmp1, group_loop_start)])

        # Increment round counter
        self.emit(flow=[("add_imm", round_counter, round_counter, 1)])

        # Loop back if more rounds
        self.emit(alu=[("<", tmp1, round_counter, self.scratch["rounds"])])
        self.emit(flow=[("cond_jump", tmp1, round_loop_start)])

        # Done
        self.instrs.append({"flow": [("pause",)]})

# Test
if __name__ == "__main__":
    import perf_takehome

    # Replace build_kernel
    original = perf_takehome.KernelBuilder
    original.build_kernel = MinimalKernel.build_kernel

    from perf_takehome import do_kernel_test
    cycles = do_kernel_test(10, 16, 256)
    print(f"\nMinimal kernel: {cycles} cycles")
    print(f"Speedup from baseline (147,734): {147734/cycles:.2f}x")

    if cycles < 5028:
        print(f"✓ BREAKTHROUGH! Improved from 5,028 by {5028-cycles} cycles")
    elif cycles < 10000:
        print(f"Competitive: {cycles} cycles (current best: 5,028)")
    else:
        print(f"Slower than optimized version")
