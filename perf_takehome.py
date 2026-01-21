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
            
            # Check: slot available AND no Read-After-Write hazard
            can_add = (
                slot_counts[engine] < SLOT_LIMITS[engine] and
                not (reads & written_this_cycle)
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

    def build_hash_stage_parallel(self, unroll_regs, num_unrolled, round, i_base, stage_idx):
        """
        Build hash operations for a specific stage across all unrolled iterations.
        This enables better VLIW packing by grouping similar operations.
        """
        slots = []
        op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
        const1 = self.scratch_const(val1)
        const2 = self.scratch_const(val3)
        
        # First operation for all iterations (can execute in parallel)
        for u in range(num_unrolled):
            ur = unroll_regs[u]
            slots.append(("alu", (op1, ur['tmp1'], ur['val'], const1)))
        
        # Second operation for all iterations (can execute in parallel)
        for u in range(num_unrolled):
            ur = unroll_regs[u]
            slots.append(("alu", (op3, ur['tmp2'], ur['val'], const2)))
        
        # Combine results for all iterations (depends on previous operations)
        for u in range(num_unrolled):
            i = i_base + u
            ur = unroll_regs[u]
            slots.append(("alu", (op2, ur['val'], ur['tmp1'], ur['tmp2'])))
        
        # Debug ops grouped after all ALU operations
        for u in range(num_unrolled):
            i = i_base + u
            ur = unroll_regs[u]
            slots.append(("debug", ("compare", ur['val'], (round, i, "hash_stage", stage_idx))))
        
        return slots

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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

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
                'node_val': self.alloc_scratch(f"v{u}_node_val", VLEN),
                'tmp1': self.alloc_scratch(f"v{u}_tmp1", VLEN),
                'tmp2': self.alloc_scratch(f"v{u}_tmp2", VLEN),
                'tmp3': self.alloc_scratch(f"v{u}_tmp3", VLEN),
            })
        
        # Shared vector constants and temporaries
        v_const1 = self.alloc_scratch("v_const1", VLEN)
        v_const2 = self.alloc_scratch("v_const2", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        
        # Scalar temporaries (multiple per vector for forest loads)
        addr_tmp = []
        for u in range(VEC_UNROLL):
            addr_tmp.append([self.alloc_scratch(f"addr{u}_{vi}") for vi in range(VLEN)])
        
        # Pre-allocate constants for hash operations
        hash_constants = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            const1 = self.scratch_const(val1)
            const2 = self.scratch_const(val3)
            hash_constants.append((op1, const1, op2, op3, const2))
        
        # Pre-broadcast common constants outside the loop
        body.append(("valu", ("vbroadcast", v_two, two_const)))
        body.append(("valu", ("vbroadcast", v_zero, zero_const)))
        body.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))

        for round in range(rounds):
            for vec_i_base in range(0, vec_batch_size, VEC_UNROLL):
                num_vec_unrolled = min(VEC_UNROLL, vec_batch_size - vec_i_base)
                
                # Stage 1: Load indices and values for all unrolled vectors
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    i_const = self.scratch_const(i)
                    vr = v_regs[u]
                    at = addr_tmp[u][0]  # Use first address temp for vector loads
                    
                    body.append(("alu", ("+", at, self.scratch["inp_indices_p"], i_const)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    at = addr_tmp[u][0]
                    body.append(("load", ("vload", vr['idx'], at)))
                
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    i_const = self.scratch_const(i)
                    vr = v_regs[u]
                    at = addr_tmp[u][0]
                    body.append(("alu", ("+", at, self.scratch["inp_values_p"], i_const)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    at = addr_tmp[u][0]
                    body.append(("load", ("vload", vr['val'], at)))
                
                # Debug loaded values
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", vr['idx'] + vi, (round, i + vi, "idx"))))
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", vr['val'] + vi, (round, i + vi, "val"))))
                
                # Stage 2: Load node values (scalar loads - batch all address calculations, then all loads)
                # Calculate ALL addresses first (use dedicated address temps for each element)
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        at = addr_tmp[u][vi]
                        body.append(("alu", ("+", at, self.scratch["forest_values_p"], vr['idx'] + vi)))
                # Then do ALL loads
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        at = addr_tmp[u][vi]
                        body.append(("load", ("load", vr['node_val'] + vi, at)))
                
                # Debug node values
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", vr['node_val'] + vi, (round, i + vi, "node_val"))))
                
                # Stage 3: XOR (vectorized for all unrolled iterations)
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("^", vr['val'], vr['val'], vr['node_val'])))
                
                # Stage 4: Hash operations (vectorized for all unrolled iterations)
                for hi, (op1, const1, op2, op3, const2) in enumerate(hash_constants):
                    body.append(("valu", ("vbroadcast", v_const1, const1)))
                    body.append(("valu", ("vbroadcast", v_const2, const2)))
                    for u in range(num_vec_unrolled):
                        vr = v_regs[u]
                        body.append(("valu", (op1, vr['tmp1'], vr['val'], v_const1)))
                    for u in range(num_vec_unrolled):
                        vr = v_regs[u]
                        body.append(("valu", (op3, vr['tmp2'], vr['val'], v_const2)))
                    for u in range(num_vec_unrolled):
                        vr = v_regs[u]
                        body.append(("valu", (op2, vr['val'], vr['tmp1'], vr['tmp2'])))
                    # Debug each hash stage
                    for u in range(num_vec_unrolled):
                        vec_i = vec_i_base + u
                        i = vec_i * VLEN
                        vr = v_regs[u]
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", vr['val'] + vi, (round, i + vi, "hash_stage", hi))))
                
                # Debug hashed values
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", vr['val'] + vi, (round, i + vi, "hashed_val"))))
                
                # Stage 5: Index updates (vectorized for all unrolled iterations)
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("%", vr['tmp1'], vr['val'], v_two)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("==", vr['tmp1'], vr['tmp1'], v_zero)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("-", vr['tmp3'], v_two, vr['tmp1'])))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("*", vr['idx'], vr['idx'], v_two)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("+", vr['idx'], vr['idx'], vr['tmp3'])))
                
                # Debug next_idx
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", vr['idx'] + vi, (round, i + vi, "next_idx"))))
                
                # Stage 6: Bounds checking (vectorized for all unrolled iterations)
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("<", vr['tmp1'], vr['idx'], v_n_nodes)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    body.append(("valu", ("*", vr['idx'], vr['tmp1'], vr['idx'])))
                
                # Debug wrapped_idx
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    vr = v_regs[u]
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", vr['idx'] + vi, (round, i + vi, "wrapped_idx"))))
                
                # Stage 7: Store results (vectorized for all unrolled iterations)
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    i_const = self.scratch_const(i)
                    vr = v_regs[u]
                    at = addr_tmp[u][0]
                    body.append(("alu", ("+", at, self.scratch["inp_indices_p"], i_const)))
                for u in range(num_vec_unrolled):
                    vr = v_regs[u]
                    at = addr_tmp[u][0]
                    body.append(("store", ("vstore", at, vr['idx'])))
                
                for u in range(num_vec_unrolled):
                    vec_i = vec_i_base + u
                    i = vec_i * VLEN
                    i_const = self.scratch_const(i)
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
