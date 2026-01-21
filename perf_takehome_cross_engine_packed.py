"""
CROSS-ENGINE PACKED KERNEL
═══════════════════════════════════════════════════════════════════
Fill the trailing negative space by packing MULTIPLE ENGINE TYPES
in the SAME bundle.

Current approach: self.emit(alu=[...])  # Only ALU
              Next: self.emit(load=[...])  # Only load

Optimized:        self.emit(alu=[...], load=[...])  # Both!
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class CrossEnginePackedKernel(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
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

        n_groups = batch_size // VLEN
        PARALLEL_GROUPS = 6

        v_idx = [self.alloc_scratch(f"v_idx{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_val = [self.alloc_scratch(f"v_val{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_node = [self.alloc_scratch(f"v_node{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_tmp = [self.alloc_scratch(f"v_tmp{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        hash_v1 = [self.alloc_scratch(f"hash_v1_{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        hash_v2 = [self.alloc_scratch(f"hash_v2_{g}", VLEN) for g in range(PARALLEL_GROUPS)]

        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.emit(valu=[
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ])

        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1 = self.alloc_scratch(f"vc1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"vc3_{hi}", VLEN)
            self.emit(valu=[
                ("vbroadcast", v_c1, self.scratch_const(val1)),
                ("vbroadcast", v_c3, self.scratch_const(val3)),
            ])
            hash_consts.append((v_c1, v_c3))

        tmp_addr = [self.alloc_scratch(f"tmp_addr{g}") for g in range(PARALLEL_GROUPS)]
        node_addr = [[self.alloc_scratch(f"node{g}_{v}") for v in range(VLEN)] for g in range(PARALLEL_GROUPS)]

        for batch_start in range(0, n_groups, PARALLEL_GROUPS):
            batch_size_actual = min(PARALLEL_GROUPS, n_groups - batch_start)

            # LOAD phase with flow packed in
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

            for round_i in range(rounds):
                # CROSS-ENGINE PACKING: Combine ALU + LOAD
                # Calculate addresses AND load nodes in fewer bundles

                # First, emit address calculations with loads packed in
                total_addrs = batch_size_actual * VLEN
                addr_offset = 0
                load_offset = 0

                while addr_offset < total_addrs or load_offset < total_addrs:
                    alu_ops = []
                    load_ops = []

                    # Add up to 12 ALU ops (address calculations)
                    for i in range(min(12, total_addrs - addr_offset)):
                        bg = (addr_offset + i) // VLEN
                        vi = (addr_offset + i) % VLEN
                        if bg < batch_size_actual:
                            alu_ops.append(("+", node_addr[bg][vi],
                                          self.scratch["forest_values_p"], v_idx[bg] + vi))
                    addr_offset += len(alu_ops)

                    # Add up to 2 load ops (from previous addresses if ready)
                    # Only load if addresses were calculated in earlier bundles
                    if load_offset < addr_offset - 12:  # Ensure addresses are ready (12+ ops ahead)
                        for i in range(min(2, load_offset + 24 - load_offset)):  # Can safely load
                            if load_offset + i < total_addrs:
                                bg = (load_offset + i) // VLEN
                                vi = (load_offset + i) % VLEN
                                if bg < batch_size_actual:
                                    load_ops.append(("load", v_node[bg] + vi, node_addr[bg][vi]))
                        load_offset += len(load_ops)

                    # Emit combined bundle
                    bundle = {}
                    if alu_ops:
                        bundle['alu'] = alu_ops
                    if load_ops:
                        bundle['load'] = load_ops
                    if bundle:
                        self.emit(**bundle)

                # Finish remaining loads
                while load_offset < total_addrs:
                    load_ops = []
                    for i in range(min(2, total_addrs - load_offset)):
                        bg = (load_offset + i) // VLEN
                        vi = (load_offset + i) % VLEN
                        if bg < batch_size_actual:
                            load_ops.append(("load", v_node[bg] + vi, node_addr[bg][vi]))
                    load_offset += len(load_ops)
                    if load_ops:
                        self.emit(load=load_ops)

                # XOR (VALU) - can potentially pack with flow for next round
                xor_ops = [("^", v_val[bg], v_val[bg], v_node[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=xor_ops)

                # HASH (VALU-heavy) - keep as-is since VALU is already well-utilized
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v_c1, v_c3 = hash_consts[hi]
                    ops1_3 = []
                    for bg in range(batch_size_actual):
                        ops1_3.append((op1, hash_v1[bg], v_val[bg], v_c1))
                    for bg in range(batch_size_actual):
                        ops1_3.append((op3, hash_v2[bg], v_val[bg], v_c3))
                    for offset in range(0, len(ops1_3), 6):
                        self.emit(valu=ops1_3[offset:offset+6])
                    ops2 = [(op2, v_val[bg], hash_v1[bg], hash_v2[bg]) for bg in range(batch_size_actual)]
                    self.emit(valu=ops2)

                # INDEX calculation (VALU)
                mod_ops = [("%", v_tmp[bg], v_val[bg], v_two) for bg in range(batch_size_actual)]
                self.emit(valu=mod_ops)
                add_ops = [("+", v_tmp[bg], v_one, v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=add_ops)
                mul_ops = [("*", v_idx[bg], v_idx[bg], v_two) for bg in range(batch_size_actual)]
                self.emit(valu=mul_ops)
                add_ops2 = [("+", v_idx[bg], v_idx[bg], v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=add_ops2)

                # BOUNDS check (VALU)
                cmp_ops = [("<", v_tmp[bg], v_idx[bg], v_n_nodes) for bg in range(batch_size_actual)]
                self.emit(valu=cmp_ops)
                mul_ops2 = [("*", v_idx[bg], v_idx[bg], v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=mul_ops2)

            # STORE
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

        self.instrs.append({"flow": [("pause",)]})


if __name__ == "__main__":
    import perf_takehome
    from perf_takehome import do_kernel_test

    print("="*70)
    print("CROSS-ENGINE PACKED KERNEL")
    print("Packing ALU + LOAD in same bundles")
    print("="*70)

    original_build = perf_takehome.KernelBuilder.build_kernel
    perf_takehome.KernelBuilder.build_kernel = CrossEnginePackedKernel.build_kernel

    cycles = do_kernel_test(10, 16, 256)

    print(f"\n{'='*70}")
    print(f"Cross-Engine Packed: {cycles} cycles")
    print(f"Previous Best: 4,997 cycles")
    print(f"Target: 1,790 cycles")
    print(f"{'='*70}")

    if cycles < 4997:
        improvement = 4997 - cycles
        print(f"✓✓✓ IMPROVEMENT: {improvement} cycles ({100*improvement/4997:.1f}%)")
        if cycles < 1790:
            print(f"✓✓✓✓✓ BEAT OPUS 4.5 CASUAL!")
            print(f"🎯 THE TRAILING NEGATIVE SPACE HAS BEEN FILLED!")
    else:
        diff = cycles - 4997
        print(f"Regression: +{diff} cycles (cross-engine packing overhead > benefit)")

    print(f"{'='*70}")
