"""
VECTORIZED implementation - transcend scalar thinking

Process 8 elements simultaneously using SIMD valu operations.
This is the EMERGENT optimization - using the architecture's vector nature.
"""

from perf_takehome import KernelBuilder as BaseKernelBuilder, HASH_STAGES, VLEN

class VectorKernelBuilder(BaseKernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        VECTOR implementation - 8 elements per instruction.
        Transcends scalar loop unrolling.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Initialize
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        # Vector registers - 8 elements each
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        self.emit(valu=[
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
        ])

        # Scalar temporaries for addressing
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_node_addr = [self.alloc_scratch(f"tmp_node_addr{i}") for i in range(VLEN)]

        # Process batch in groups of VLEN (8)
        n_groups = batch_size // VLEN

        # Pre-allocate base address constants
        base_addrs = [self.scratch_const(g * VLEN) for g in range(n_groups)]

        for round in range(rounds):
            for group in range(n_groups):
                base_i = group * VLEN
                base_const = base_addrs[group]

                # ===== VECTOR LOADS =====
                # Load 8 indices at once
                self.emit(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], base_const)])
                self.emit(load=[("vload", v_idx, tmp_addr)])

                # Load 8 values at once
                self.emit(alu=[("+", tmp_addr, self.scratch["inp_values_p"], base_const)])
                self.emit(load=[("vload", v_val, tmp_addr)])

                # Debug vector indices and values
                for vi in range(VLEN):
                    i = base_i + vi
                    self.emit(debug=[
                        ("compare", v_idx + vi, (round, i, "idx")),
                        ("compare", v_val + vi, (round, i, "val")),
                    ])

                # ===== INDIRECT LOADS (still scalar - bottleneck) =====
                # Load node values - must be scalar due to indirect addressing
                for vi in range(VLEN):
                    self.emit(alu=[("+", tmp_node_addr[vi], self.scratch["forest_values_p"], v_idx + vi)])
                for vi in range(VLEN):
                    self.emit(load=[("load", v_node_val + vi, tmp_node_addr[vi])])

                for vi in range(VLEN):
                    i = base_i + vi
                    self.emit(debug=[("compare", v_node_val + vi, (round, i, "node_val"))])

                # ===== VECTOR XOR =====
                self.emit(valu=[("^", v_val, v_val, v_node_val)])

                # ===== VECTOR HASH =====
                # Each valu instruction processes 8 elements
                hash_v1 = self.alloc_scratch("hash_v1", VLEN)
                hash_v2 = self.alloc_scratch("hash_v2", VLEN)

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v_c1 = self.alloc_scratch(f"v_c1_{hi}", VLEN)
                    v_c3 = self.alloc_scratch(f"v_c3_{hi}", VLEN)
                    c1 = self.scratch_const(val1)
                    c3 = self.scratch_const(val3)

                    # Broadcast constants once
                    self.emit(valu=[
                        ("vbroadcast", v_c1, c1),
                        ("vbroadcast", v_c3, c3),
                    ])

                    # Vector operations - 8 elements in 3 instructions
                    self.emit(valu=[(op1, hash_v1, v_val, v_c1)])
                    self.emit(valu=[(op3, hash_v2, v_val, v_c3)])
                    self.emit(valu=[(op2, v_val, hash_v1, hash_v2)])

                    # Debug
                    for vi in range(VLEN):
                        i = base_i + vi
                        self.emit(debug=[("compare", v_val + vi, (round, i, "hash_stage", hi))])

                for vi in range(VLEN):
                    i = base_i + vi
                    self.emit(debug=[("compare", v_val + vi, (round, i, "hashed_val"))])

                # ===== VECTOR INDEX CALCULATION =====
                # All flow operations eliminated with arithmetic

                # v_tmp1 = v_val % 2
                self.emit(valu=[("%", v_tmp1, v_val, v_two)])

                # v_tmp3 = 1 + (v_val % 2) = {1, 2}
                self.emit(valu=[("+", v_tmp3, v_one, v_tmp1)])

                # v_tmp2 = v_idx * 2
                self.emit(valu=[("*", v_tmp2, v_idx, v_two)])

                # v_idx = 2*idx + {1,2}
                self.emit(valu=[("+", v_idx, v_tmp2, v_tmp3)])

                # Debug next_idx
                for vi in range(VLEN):
                    i = base_i + vi
                    self.emit(debug=[("compare", v_idx + vi, (round, i, "next_idx"))])

                # ===== VECTOR BOUNDS CHECK (arithmetic, no flow) =====
                # v_tmp1 = (v_idx < n_nodes) as 0/1
                v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
                self.emit(valu=[("vbroadcast", v_n_nodes, self.scratch["n_nodes"])])
                self.emit(valu=[("<", v_tmp1, v_idx, v_n_nodes)])

                # v_idx = v_idx * (v_idx < n_nodes)  -- zeros out-of-bounds
                self.emit(valu=[("*", v_idx, v_idx, v_tmp1)])

                # Debug wrapped_idx
                for vi in range(VLEN):
                    i = base_i + vi
                    self.emit(debug=[("compare", v_idx + vi, (round, i, "wrapped_idx"))])

                # ===== VECTOR STORES =====
                self.emit(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], base_const)])
                self.emit(store=[("vstore", tmp_addr, v_idx)])

                self.emit(alu=[("+", tmp_addr, self.scratch["inp_values_p"], base_const)])
                self.emit(store=[("vstore", tmp_addr, v_val)])

        self.instrs.append({"flow": [("pause",)]})

    def emit(self, **engines):
        """Emit a VLIW bundle"""
        self.instrs.append(engines)
