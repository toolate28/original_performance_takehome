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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        """
        Optimized hash with parallel computation.
        Groups similar operations for better VLIW packing.
        Reuses tmp1 and tmp2 to save scratch space.
        """
        slots = []
        
        # Process all hash stages with maximum parallelism
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1 = self.scratch_const(val1)
            const2 = self.scratch_const(val3)
            
            # Both operations can execute in parallel (no dependency)
            slots.append(("alu", (op1, tmp1, val_hash_addr, const1)))
            slots.append(("alu", (op3, tmp2, val_hash_addr, const2)))
            # Combine results (depends on previous two)
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized scalar implementation with loop unrolling and flow elimination.
        """
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

        # Unroll factor - using 16 for optimal balance
        UNROLL_FACTOR = 16
        
        # Allocate registers for unrolled iterations
        unroll_regs = []
        for u in range(UNROLL_FACTOR):
            unroll_regs.append({
                'idx': self.alloc_scratch(f"u{u}_idx"),
                'val': self.alloc_scratch(f"u{u}_val"),
                'node_val': self.alloc_scratch(f"u{u}_node_val"),
                'addr': self.alloc_scratch(f"u{u}_addr"),
                'tmp1': self.alloc_scratch(f"u{u}_tmp1"),
                'tmp2': self.alloc_scratch(f"u{u}_tmp2"),
                'tmp3': self.alloc_scratch(f"u{u}_tmp3"),
            })

        for round in range(rounds):
            for i_base in range(0, batch_size, UNROLL_FACTOR):
                num_unrolled = min(UNROLL_FACTOR, batch_size - i_base)
                
                # Stage 1: Load indices (parallel)
                for u in range(num_unrolled):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    ur = unroll_regs[u]
                    body.append(("alu", ("+", ur['addr'], self.scratch["inp_indices_p"], i_const)))
                    body.append(("load", ("load", ur['idx'], ur['addr'])))
                    body.append(("debug", ("compare", ur['idx'], (round, i, "idx"))))
                
                # Stage 2: Load values (parallel)
                for u in range(num_unrolled):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    ur = unroll_regs[u]
                    body.append(("alu", ("+", ur['addr'], self.scratch["inp_values_p"], i_const)))
                    body.append(("load", ("load", ur['val'], ur['addr'])))
                    body.append(("debug", ("compare", ur['val'], (round, i, "val"))))
                
                # Stage 3: Load node values (depends on Stage 1)
                for u in range(num_unrolled):
                    i = i_base + u
                    ur = unroll_regs[u]
                    body.append(("alu", ("+", ur['addr'], self.scratch["forest_values_p"], ur['idx'])))
                    body.append(("load", ("load", ur['node_val'], ur['addr'])))
                    body.append(("debug", ("compare", ur['node_val'], (round, i, "node_val"))))
                
                # Stage 4: XOR operations (parallel after Stage 2&3)
                for u in range(num_unrolled):
                    i = i_base + u
                    ur = unroll_regs[u]
                    body.append(("alu", ("^", ur['val'], ur['val'], ur['node_val'])))
                
                # Stage 5: Hash operations (parallel)
                for u in range(num_unrolled):
                    i = i_base + u
                    ur = unroll_regs[u]
                    body.extend(self.build_hash(ur['val'], ur['tmp1'], ur['tmp2'], round, i))
                    body.append(("debug", ("compare", ur['val'], (round, i, "hashed_val"))))
                
                # Stage 6: Index update with flow elimination
                for u in range(num_unrolled):
                    i = i_base + u
                    ur = unroll_regs[u]
                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    body.append(("alu", ("%", ur['tmp1'], ur['val'], two_const)))
                    body.append(("alu", ("==", ur['tmp1'], ur['tmp1'], zero_const)))
                    # Flow elimination: select(cond, 1, 2) == 2 - cond
                    body.append(("alu", ("-", ur['tmp3'], two_const, ur['tmp1'])))
                    body.append(("alu", ("*", ur['idx'], ur['idx'], two_const)))
                    body.append(("alu", ("+", ur['idx'], ur['idx'], ur['tmp3'])))
                    body.append(("debug", ("compare", ur['idx'], (round, i, "next_idx"))))
                
                # Stage 7: Bounds checking with flow elimination
                for u in range(num_unrolled):
                    i = i_base + u
                    ur = unroll_regs[u]
                    # idx = 0 if idx >= n_nodes else idx
                    body.append(("alu", ("<", ur['tmp1'], ur['idx'], self.scratch["n_nodes"])))
                    # Flow elimination: select(cond, idx, 0) == cond * idx
                    body.append(("alu", ("*", ur['idx'], ur['tmp1'], ur['idx'])))
                    body.append(("debug", ("compare", ur['idx'], (round, i, "wrapped_idx"))))
                
                # Stage 8: Store results (parallel, uses 2 store slots)
                for u in range(num_unrolled):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    ur = unroll_regs[u]
                    body.append(("alu", ("+", ur['addr'], self.scratch["inp_indices_p"], i_const)))
                    body.append(("store", ("store", ur['addr'], ur['idx'])))
                    body.append(("alu", ("+", ur['addr'], self.scratch["inp_values_p"], i_const)))
                    body.append(("store", ("store", ur['addr'], ur['val'])))

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
