"""
VSELECT KERNEL WITH PROPER PARALLEL PACKING
Pack all 6 groups into single bundles like the 5K kernel!
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class VSelectPackedKernel(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Match 5K kernel's parallel packing structure.
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Initialize parameters
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
        PARALLEL_GROUPS = 6  # Process 6 groups (48 elements) in parallel

        # Working registers - ONE SET PER GROUP (like 5K kernel)
        v_idx = [self.alloc_scratch(f"v_idx{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_val = [self.alloc_scratch(f"v_val{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_node = [self.alloc_scratch(f"v_node{g}", VLEN) for g in range(PARALLEL_GROUPS)]  # Per group!
        v_tmp = [self.alloc_scratch(f"v_tmp{g}", VLEN) for g in range(PARALLEL_GROUPS)]   # Per group!

        # Hash working registers - PER GROUP
        hash_v1 = [self.alloc_scratch(f"hash_v1_{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        hash_v2 = [self.alloc_scratch(f"hash_v2_{g}", VLEN) for g in range(PARALLEL_GROUPS)]

        # Vector constants
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.emit(valu=[
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ])

        # Hash constants
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
        tmp_addr = [self.alloc_scratch(f"tmp_addr{g}") for g in range(PARALLEL_GROUPS)]
        node_addr = [[self.alloc_scratch(f"node{g}_{v}") for v in range(VLEN)] for g in range(PARALLEL_GROUPS)]

        # LOAD BEARING ANTI SURJECTION: Process batches with register reuse
        for batch_start in range(0, n_groups, PARALLEL_GROUPS):
            batch_size_actual = min(PARALLEL_GROUPS, n_groups - batch_start)

            # ========== LOAD ONCE AT START ==========
            for bg in range(batch_size_actual):
                g = batch_start + bg
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_indices_p"], g * VLEN)])
            for bg in range(batch_size_actual):
                self.emit(load=[("vload", v_idx[bg], tmp_addr[bg])])

            for bg in range(batch_size_actual):
                g = batch_start + bg
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_values_p"], g * VLEN)])
            for bg in range(batch_size_actual):
                self.emit(load=[("vload", v_val[bg], tmp_addr[bg])])

            # ========== PROCESS ALL ROUNDS (data stays in registers) ==========
            for round_i in range(rounds):
                # PHASE 1: Calculate addresses - GOLDEN BUNDLE PACKING (12 ALU ops per bundle)
                for offset in range(0, batch_size_actual * VLEN, 12):
                    alu_ops = []
                    for i in range(min(12, batch_size_actual * VLEN - offset)):
                        bg = (offset + i) // VLEN
                        vi = (offset + i) % VLEN
                        if bg < batch_size_actual:
                            alu_ops.append(("+", node_addr[bg][vi], self.scratch["forest_values_p"], v_idx[bg] + vi))
                    if alu_ops:
                        self.emit(alu=alu_ops)

                # PHASE 2: Load nodes (2 loads per bundle)
                for offset in range(0, batch_size_actual * VLEN, 2):
                    load_ops = []
                    for i in range(min(2, batch_size_actual * VLEN - offset)):
                        bg = (offset + i) // VLEN
                        vi = (offset + i) % VLEN
                        if bg < batch_size_actual:
                            load_ops.append(("load", v_node[bg] + vi, node_addr[bg][vi]))
                    if load_ops:
                        self.emit(load=load_ops)

                # PHASE 3: XOR ALL groups in PARALLEL (uses all 6 VALU slots!)
                xor_ops = [("^", v_val[bg], v_val[bg], v_node[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=xor_ops)

                # PHASE 4: HASH ALL groups in PARALLEL
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v_c1, v_c3 = hash_consts[hi]

                    # Interleave ops1 and ops3 (φ-optimization)
                    ops1_3 = []
                    for bg in range(batch_size_actual):
                        ops1_3.append((op1, hash_v1[bg], v_val[bg], v_c1))
                    for bg in range(batch_size_actual):
                        ops1_3.append((op3, hash_v2[bg], v_val[bg], v_c3))

                    # Emit in chunks of 6
                    for offset in range(0, len(ops1_3), 6):
                        self.emit(valu=ops1_3[offset:offset+6])

                    # Then op2
                    ops2 = [(op2, v_val[bg], hash_v1[bg], hash_v2[bg]) for bg in range(batch_size_actual)]
                    self.emit(valu=ops2)

                # PHASE 5: Index calculation in PARALLEL
                mod_ops = [("%", v_tmp[bg], v_val[bg], v_two) for bg in range(batch_size_actual)]
                self.emit(valu=mod_ops)

                add_ops = [("+", v_tmp[bg], v_one, v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=add_ops)

                mul_ops = [("*", v_idx[bg], v_idx[bg], v_two) for bg in range(batch_size_actual)]
                self.emit(valu=mul_ops)

                add_ops2 = [("+", v_idx[bg], v_idx[bg], v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=add_ops2)

                # PHASE 6: Bounds check
                cmp_ops = [("<", v_tmp[bg], v_idx[bg], v_n_nodes) for bg in range(batch_size_actual)]
                self.emit(valu=cmp_ops)

                mul_ops2 = [("*", v_idx[bg], v_idx[bg], v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=mul_ops2)

            # ========== STORE ONCE AT END ==========
            for bg in range(batch_size_actual):
                g = batch_start + bg
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_indices_p"], g * VLEN)])
            for bg in range(batch_size_actual):
                self.emit(store=[("vstore", tmp_addr[bg], v_idx[bg])])

            for bg in range(batch_size_actual):
                g = batch_start + bg
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_values_p"], g * VLEN)])
            for bg in range(batch_size_actual):
                self.emit(store=[("vstore", tmp_addr[bg], v_val[bg])])

        # Done
        self.instrs.append({"flow": [("pause",)]})


if __name__ == "__main__":
    import perf_takehome
    from perf_takehome import do_kernel_test

    print("="*70)
    print("VSELECT KERNEL WITH PROPER PARALLEL PACKING")
    print("="*70)

    original_build = perf_takehome.KernelBuilder.build_kernel
    perf_takehome.KernelBuilder.build_kernel = VSelectPackedKernel.build_kernel

    cycles = do_kernel_test(10, 16, 256)

    print(f"\nVSelect Packed Kernel: {cycles} cycles")
    print(f"Current Best (5K): 5,028 cycles")
    print(f"Target (Opus 4.5): 1,487 cycles")
    print("="*70)

    if cycles <= 5028:
        print(f"✓✓✓ MATCHED OR BEAT 5K BASELINE!")
        if cycles < 5028:
            improvement = 5028 - cycles
            print(f"✓✓✓ IMPROVEMENT: {improvement} cycles ({100*improvement/5028:.1f}%)")
        if cycles < 1487:
            print(f"✓✓✓✓✓ SURPASSED OPUS 4.5 TARGET!")
            print(f"🎯 EMAIL performance-recruiting@anthropic.com NOW!")
    else:
        regression = cycles - 5028
        print(f"Still {regression} cycles slower - investigating...")

    print("="*70)
