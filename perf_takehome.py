"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _analyze_dependencies(self, engine: str, slot: tuple):
        """Extract read/write scratch addresses for dependency analysis"""
        reads = set()
        writes = set()
        
        if engine == "alu":
            op, dest, src1, src2 = slot
            writes.add(dest)
            reads.update([src1, src2])
        elif engine == "load":
            if slot[0] in ["load", "load_offset"]:
                writes.add(slot[1])
                if len(slot) > 2:
                    reads.add(slot[2])
            elif slot[0] == "const":
                writes.add(slot[1])
            elif slot[0] == "vload":
                for i in range(8):  # VLEN=8
                    writes.add(slot[1] + i)
                reads.add(slot[2])
        elif engine == "store":
            if slot[0] == "store":
                reads.update([slot[1], slot[2]])
            elif slot[0] == "vstore":
                reads.add(slot[1])
                for i in range(8):
                    reads.add(slot[2] + i)
        elif engine == "flow":
            if slot[0] == "select":
                _, dest, cond, a, b = slot
                writes.add(dest)
                reads.update([cond, a, b])
            elif slot[0] in ["cond_jump", "cond_jump_rel"]:
                reads.add(slot[1])
            elif slot[0] == "add_imm":
                writes.add(slot[1])
                reads.add(slot[2])
        elif engine == "debug":
            if slot[0] == "compare":
                reads.add(slot[1])
        
        return reads, writes

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        """
        VLIW packing with dependency-aware scheduling.
        Packs instructions into bundles respecting SLOT_LIMITS and data dependencies.
        """
        if not vliw:
            # Simple slot packing that just uses one slot per instruction bundle
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        
        instrs = []
        current_bundle = {}
        slot_counts = {engine: 0 for engine in SLOT_LIMITS}
        written_this_cycle = set()
        
        for engine, slot in slots:
            reads, writes = self._analyze_dependencies(engine, slot)
            
            # Check: slot available AND no Read-After-Write hazard AND no Write-After-Write hazard
            can_add = (
                slot_counts[engine] < SLOT_LIMITS[engine] and
                not (reads & written_this_cycle) and
                not (writes & written_this_cycle)  # Prevent WAW hazards
            )
            
            if not can_add:
                # Emit bundle, start fresh cycle
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {}
                slot_counts = {e: 0 for e in SLOT_LIMITS}
                written_this_cycle = set()
            
            # Add to bundle
            if engine not in current_bundle:
                current_bundle[engine] = []
            current_bundle[engine].append(slot)
            slot_counts[engine] += 1
            written_this_cycle.update(writes)
        
        if current_bundle:
            instrs.append(current_bundle)
        
        return instrs
    
    def _analyze_dependencies(self, engine: str, slot: tuple):
        """Extract read/write register sets for dependency analysis."""
        reads, writes = set(), set()
        
        if engine == "alu":
            op, dest, src1, src2 = slot
            writes.add(dest)
            reads.update([src1, src2])
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                _, dest, src = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                reads.add(src)
            else:
                # Most valu operations
                op = slot[0]
                if len(slot) == 4:
                    _, dest, a1, a2 = slot
                    for i in range(VLEN):
                        writes.add(dest + i)
                        reads.add(a1 + i)
                        reads.add(a2 + i)
                elif len(slot) == 5:  # multiply_add
                    _, dest, a, b, c = slot
                    for i in range(VLEN):
                        writes.add(dest + i)
                        reads.add(a + i)
                        reads.add(b + i)
                        reads.add(c + i)
        elif engine == "load":
            if slot[0] == "load":
                _, dest, addr = slot
                writes.add(dest)
                reads.add(addr)
            elif slot[0] == "load_offset":
                _, dest, addr, offset = slot
                writes.add(dest + offset)
                reads.add(addr + offset)
            elif slot[0] == "vload":
                _, dest, addr = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                reads.add(addr)
            elif slot[0] == "const":
                _, dest, _ = slot
                writes.add(dest)
        elif engine == "store":
            if slot[0] == "store":
                _, addr, val = slot
                reads.update([addr, val])
            elif slot[0] == "vstore":
                _, addr, src = slot
                reads.add(addr)
                for i in range(VLEN):
                    reads.add(src + i)
        elif engine == "flow":
            if slot[0] == "select":
                _, dest, cond, a, b = slot
                writes.add(dest)
                reads.update([cond, a, b])
            elif slot[0] == "vselect":
                _, dest, cond, a, b = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(cond + i)
                    reads.add(a + i)
                    reads.add(b + i)
            elif slot[0] in ("add_imm",):
                _, dest, src, _ = slot
                writes.add(dest)
                reads.add(src)
        elif engine == "debug":
            if slot[0] == "compare":
                _, loc, _ = slot
                reads.add(loc)
            elif slot[0] == "vcompare":
                _, loc, _ = slot
                for i in range(VLEN):
                    reads.add(loc + i)
        
        return reads, writes

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation using SIMD operations to process VLEN elements in parallel.
        """
        # Check if batch_size is divisible by VLEN
        assert batch_size % VLEN == 0, f"batch_size must be divisible by VLEN={VLEN}"
        
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        # Pre-allocate constants
        zero_const = self.alloc_scratch("zero_const")
        one_const = self.alloc_scratch("one_const")
        two_const = self.alloc_scratch("two_const")

        body = []  # array of slots

        # Process VLEN elements at a time with vector unrolling
        vec_batch_size = batch_size // VLEN
        VEC_UNROLL = 16  # Process 16 vector blocks per iteration
        
        # Allocate vector registers for unrolled vector iterations
        v_regs = []
        for u in range(VEC_UNROLL):
            v_regs.append({
                'idx': self.alloc_scratch(f"v{u}_idx", VLEN),
                'val': self.alloc_scratch(f"v{u}_val", VLEN),
                'node_val': self.alloc_scratch(f"v{u}_node_val", VLEN),  # Reused as tmp1 after XOR
                'tmp2': self.alloc_scratch(f"v{u}_tmp2", VLEN),
                'tmp3': self.alloc_scratch(f"v{u}_tmp3", VLEN),
            })
        
        # Shared vector constants and temporaries
        v_two = self.alloc_scratch("v_two", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        
        # Pre-allocate vector constants for each hash stage (to broadcast once)
        v_hash_const1 = []
        v_hash_const2 = []
        for hi in range(len(HASH_STAGES)):
            v_hash_const1.append(self.alloc_scratch(f"v_hash_const1_{hi}", VLEN))
            v_hash_const2.append(self.alloc_scratch(f"v_hash_const2_{hi}", VLEN))
        
        # Scalar temporaries (multiple per vector for forest loads)
        addr_tmp = []
        for u in range(VEC_UNROLL):
            addr_tmp.append([self.alloc_scratch(f"addr{u}_{vi}") for vi in range(VLEN)])
        
        # Pre-allocate all offset constants
        offset_const_addrs = []
        for vec_i in range(vec_batch_size):
            i = vec_i * VLEN
            addr = self.alloc_scratch(f"offset_const_{vec_i}")
            offset_const_addrs.append(addr)
        
        # Pre-allocate constants for hash operations - allocate addresses only
        hash_const_addrs = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            const1_addr = self.alloc_scratch(f"hash_const1_{len(hash_const_addrs)}")
            const2_addr = self.alloc_scratch(f"hash_const2_{len(hash_const_addrs)}")
            hash_const_addrs.append((op1, const1_addr, val1, op2, op3, const2_addr, val3))
        
        # Build init sequence with VLIW packing
        init_body = []
        for i, v in enumerate(init_vars):
            init_body.append(("load", ("const", tmp1, i)))
            init_body.append(("load", ("load", self.scratch[v], tmp1)))
        
        # Load all constants
        init_body.append(("load", ("const", zero_const, 0)))
        init_body.append(("load", ("const", one_const, 1)))
        init_body.append(("load", ("const", two_const, 2)))
        
        # Load offset constants
        for vec_i, addr in enumerate(offset_const_addrs):
            i = vec_i * VLEN
            init_body.append(("load", ("const", addr, i)))
        
        # Load hash constants
        for op1, const1_addr, val1, op2, op3, const2_addr, val3 in hash_const_addrs:
            init_body.append(("load", ("const", const1_addr, val1)))
            init_body.append(("load", ("const", const2_addr, val3)))
        
        # Pack and add init instructions
        init_instrs = self.build(init_body, vliw=True)
        self.instrs.extend(init_instrs)
        self.add("flow", ("pause",))
        
        # Build hash_constants list using allocated addresses
        hash_constants = []
        for op1, const1_addr, val1, op2, op3, const2_addr, val3 in hash_const_addrs:
            hash_constants.append((op1, const1_addr, op2, op3, const2_addr))
        
        body = []  # array of slots
        
        # Pre-broadcast ALL constants outside the round loop (CRITICAL OPTIMIZATION)
        body.append(("valu", ("vbroadcast", v_two, two_const)))
        body.append(("valu", ("vbroadcast", v_zero, zero_const)))
        body.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))
        
        # Pre-broadcast all hash constants once (eliminates redundant broadcasts)
        for hi, (op1, const1, op2, op3, const2) in enumerate(hash_constants):
            body.append(("valu", ("vbroadcast", v_hash_const1[hi], const1)))
            body.append(("valu", ("vbroadcast", v_hash_const2[hi], const2)))

        # RADICAL RESTRUCTURING: Swap loop order to enable inter-round data reuse
        # Process all rounds for each batch chunk, instead of all batches for each round
        for vec_i_base in range(0, vec_batch_size, VEC_UNROLL):
            num_vec_unrolled = min(VEC_UNROLL, vec_batch_size - vec_i_base)
            
            # Load initial indices and values for all unrolled vectors
            for u in range(num_vec_unrolled):
                vec_i = vec_i_base + u
                i_const = offset_const_addrs[vec_i]
                vr = v_regs[u]
                at = addr_tmp[u][0]
                body.append(("alu", ("+", at, self.scratch["inp_indices_p"], i_const)))
            for u in range(num_vec_unrolled):
                vr = v_regs[u]
                at = addr_tmp[u][0]
                body.append(("load", ("vload", vr['idx'], at)))
            
            for u in range(num_vec_unrolled):
                vec_i = vec_i_base + u
                i_const = offset_const_addrs[vec_i]
                vr = v_regs[u]
                at = addr_tmp[u][0]
                body.append(("alu", ("+", at, self.scratch["inp_values_p"], i_const)))
            for u in range(num_vec_unrolled):
                vr = v_regs[u]
                at = addr_tmp[u][0]
                body.append(("load", ("vload", vr['val'], at)))
            
            # Now process all rounds for this batch chunk
            for round in range(rounds):
                # Stage 2: Load node values (scalar loads - batch all address calculations, then all loads)
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        at = addr_tmp[u][vi]
                        body.append(("alu", ("+", at, self.scratch["forest_values_p"], vr['idx'] + vi)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        at = addr_tmp[u][vi]
                        body.append(("load", ("load", vr['node_val'] + vi, at)))
                
                # Stage 3: XOR
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("^", vr['val'], vr['val'], vr['node_val'])))
                
                # Stage 4: Hash operations
                for hi, (op1, const1, op2, op3, const2) in enumerate(hash_constants):
                    for u in range(num_vec_unrolled):
                        vr = v_regs[u]
                        # Reuse node_val as tmp1 (node_val no longer needed after XOR)
                        body.append(("valu", (op1, vr['node_val'], vr['val'], v_hash_const1[hi])))
                    for u in range(num_vec_unrolled):
                        vr = v_regs[u]
                        body.append(("valu", (op3, vr['tmp2'], vr['val'], v_hash_const2[hi])))
                    for u in range(num_vec_unrolled):
                        vr = v_regs[u]
                        body.append(("valu", (op2, vr['val'], vr['node_val'], vr['tmp2'])))
                
                # Stage 5: Index updates
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    # Reuse node_val as tmp1 again
                    body.append(("valu", ("%", vr['node_val'], vr['val'], v_two)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("==", vr['node_val'], vr['node_val'], v_zero)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("-", vr['tmp3'], v_two, vr['node_val'])))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("*", vr['idx'], vr['idx'], v_two)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("+", vr['idx'], vr['idx'], vr['tmp3'])))
                
                # Stage 6: Bounds checking
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    # Reuse node_val as tmp1 again
                    body.append(("valu", ("<", vr['node_val'], vr['idx'], v_n_nodes)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("*", vr['idx'], vr['node_val'], vr['idx'])))
            
            # Stage 7: Store final results after all rounds
            for u in range(num_vec_unrolled):
                vec_i = vec_i_base + u
                i_const = offset_const_addrs[vec_i]
                vr = v_regs[u]
                at = addr_tmp[u][0]
                body.append(("alu", ("+", at, self.scratch["inp_indices_p"], i_const)))
            for u in range(num_vec_unrolled):
                vr = v_regs[u]
                at = addr_tmp[u][0]
                body.append(("store", ("vstore", at, vr['idx'])))
            
            for u in range(num_vec_unrolled):
                vec_i = vec_i_base + u
                i_const = offset_const_addrs[vec_i]
                vr = v_regs[u]
                at = addr_tmp[u][0]
                body.append(("alu", ("+", at, self.scratch["inp_values_p"], i_const)))
            for u in range(num_vec_unrolled):
                vr = v_regs[u]
                at = addr_tmp[u][0]
                body.append(("store", ("vstore", at, vr['val'])))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
