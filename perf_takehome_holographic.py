"""
QUANTUM HOLOGRAPHIC CONSERVATION
All 256 elements as ONE coherent quantum state.
Never break the information - load once, transform 16 times, store once.
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class HolographicKernel(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Holographic conservation: ALL 256 elements stay in registers.
        Load once → transform through 16 rounds → store once.
        Information never leaves the quantum state.
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Initialize
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

        n_groups = batch_size // VLEN  # 32 groups

        # ========== QUANTUM STATE: ALL 256 ELEMENTS ==========
        # Allocate scratch for ALL indices and values (256 each = 512 total)
        all_idx = [self.alloc_scratch(f"idx{g}", VLEN) for g in range(n_groups)]
        all_val = [self.alloc_scratch(f"val{g}", VLEN) for g in range(n_groups)]

        # Working registers for processing ONE group at a time
        v_node = self.alloc_scratch("v_node", VLEN)
        v_tmp = self.alloc_scratch("v_tmp", VLEN)
        hash_v1 = self.alloc_scratch("hash_v1", VLEN)
        hash_v2 = self.alloc_scratch("hash_v2", VLEN)

        # Constants
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.emit(valu=[
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ])

        # Pre-broadcast hash constants
        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1 = self.alloc_scratch(f"vc1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"vc3_{hi}", VLEN)
            self.emit(valu=[
                ("vbroadcast", v_c1, self.scratch_const(val1)),
                ("vbroadcast", v_c3, self.scratch_const(val3)),
            ])
            hash_consts.append((v_c1, v_c3))

        # Addressing
        tmp_addr = self.alloc_scratch("tmp_addr")
        node_addrs = [self.alloc_scratch(f"na{i}") for i in range(VLEN)]

        # ========== LOAD QUANTUM STATE (ONCE) ==========
        print(f"Loading ALL {n_groups} groups into quantum state...")
        for g in range(n_groups):
            # Load indices
            self.emit(flow=[("add_imm", tmp_addr, self.scratch["inp_indices_p"], g * VLEN)])
            self.emit(load=[("vload", all_idx[g], tmp_addr)])

            # Load values
            self.emit(flow=[("add_imm", tmp_addr, self.scratch["inp_values_p"], g * VLEN)])
            self.emit(load=[("vload", all_val[g], tmp_addr)])

        # ========== TRANSFORM THROUGH ALL ROUNDS (IN REGISTERS) ==========
        print(f"Processing {rounds} rounds with quantum state in registers...")
        for r in range(rounds):
            # Process each group
            for g in range(n_groups):
                # Current group's idx/val are in all_idx[g], all_val[g]

                # Calculate node addresses (pack into 2 bundles)
                for offset in range(0, VLEN, 4):
                    ops = [("+", node_addrs[offset+i], self.scratch["forest_values_p"], all_idx[g] + offset + i)
                           for i in range(min(4, VLEN-offset))]
                    self.emit(alu=ops)

                # Load nodes (pack 2 per bundle)
                for offset in range(0, VLEN, 2):
                    self.emit(load=[
                        ("load", v_node + offset, node_addrs[offset]),
                        ("load", v_node + offset + 1, node_addrs[offset + 1]),
                    ])

                # XOR
                self.emit(valu=[("^", all_val[g], all_val[g], v_node)])

                # Hash
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v_c1, v_c3 = hash_consts[hi]
                    self.emit(valu=[
                        (op1, hash_v1, all_val[g], v_c1),
                        (op3, hash_v2, all_val[g], v_c3),
                    ])
                    self.emit(valu=[(op2, all_val[g], hash_v1, hash_v2)])

                # Index calculation
                self.emit(valu=[
                    ("%", v_tmp, all_val[g], v_two),
                    ("+", v_tmp, v_one, v_tmp),
                ])
                self.emit(valu=[
                    ("*", all_idx[g], all_idx[g], v_two),
                    ("+", all_idx[g], all_idx[g], v_tmp),
                ])

                # Bounds check
                self.emit(valu=[
                    ("<", v_tmp, all_idx[g], v_n_nodes),
                    ("*", all_idx[g], all_idx[g], v_tmp),
                ])

        # ========== STORE QUANTUM STATE (ONCE) ==========
        print(f"Storing final quantum state...")
        for g in range(n_groups):
            # Store indices
            self.emit(flow=[("add_imm", tmp_addr, self.scratch["inp_indices_p"], g * VLEN)])
            self.emit(store=[("vstore", tmp_addr, all_idx[g])])

            # Store values
            self.emit(flow=[("add_imm", tmp_addr, self.scratch["inp_values_p"], g * VLEN)])
            self.emit(store=[("vstore", tmp_addr, all_val[g])])

        # Done
        self.instrs.append({"flow": [("pause",)]})
        print(f"Holographic kernel complete. Quantum coherence preserved.")

# Test
if __name__ == "__main__":
    import perf_takehome

    original = perf_takehome.KernelBuilder
    original.build_kernel = HolographicKernel.build_kernel

    from perf_takehome import do_kernel_test
    cycles = do_kernel_test(10, 16, 256)
    print(f"\n{'='*60}")
    print(f"HOLOGRAPHIC KERNEL RESULTS")
    print(f"{'='*60}")
    print(f"Cycles: {cycles}")
    print(f"Baseline: 147,734")
    print(f"Current best: 5,028")
    print(f"Target: 1,487")
    print(f"{'='*60}")
    print(f"Speedup from baseline: {147734/cycles:.2f}x")
    if cycles < 5028:
        print(f"✓✓✓ BREAKTHROUGH! Improved by {5028-cycles} cycles ({100*(5028-cycles)/5028:.1f}%)")
        if cycles < 1487:
            print(f"✓✓✓ SURPASSED OPUS 4.5 TARGET!")
    elif cycles < 10000:
        print(f"Competitive but not breakthrough: {cycles} vs 5,028")
    print(f"{'='*60}")
