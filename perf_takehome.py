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
        """Extract read/write registers from instruction slot"""
        reads = set()
        writes = set()
        
        if engine == "alu":
            op, dest, src1, src2 = slot
            writes.add(dest)
            reads.update([src1, src2])
        elif engine == "load":
            if slot[0] == "load":
                _, dest, addr = slot
                writes.add(dest)
                reads.add(addr)
            elif slot[0] == "const":
                _, dest, _ = slot
                writes.add(dest)
        elif engine == "store":
            _, addr, val = slot
            reads.update([addr, val])
        elif engine == "flow":
            if slot[0] == "select":
                _, dest, cond, a, b = slot
                writes.add(dest)
                reads.update([cond, a, b])
            elif slot[0] == "pause":
                # No dependencies
                pass
        elif engine == "debug":
            # Debug instructions read from registers
            if slot[0] == "compare":
                _, reg, _ = slot
                reads.add(reg)
        
        return reads, writes

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """Dependency-aware VLIW bundling with RAW hazard detection"""
        if not vliw:
            return [{engine: [slot]} for engine, slot in slots]
        
        instrs = []
        current_bundle = {}
        slot_counts = {engine: 0 for engine in SLOT_LIMITS}
        written_this_cycle = set()
        
        for engine, slot in slots:
            reads, writes = self._analyze_dependencies(engine, slot)
            
            # Can add if: slot available AND no RAW hazard
            has_raw_hazard = bool(reads & written_this_cycle)
            slot_available = slot_counts[engine] < SLOT_LIMITS[engine]
            
            if has_raw_hazard or not slot_available:
                # Flush current bundle, start new cycle
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {}
                slot_counts = {engine: 0 for engine in SLOT_LIMITS}
                written_this_cycle = set()
            
            # Add to current bundle
            if engine not in current_bundle:
                current_bundle[engine] = []
            current_bundle[engine].append(slot)
            slot_counts[engine] += 1
            written_this_cycle.update(writes)
        
        if current_bundle:
            instrs.append(current_bundle)
        return instrs

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
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
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

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        UNROLL_FACTOR = 13
        
        # Pre-compute all index constants before the main loop
        i_consts = []
        for i in range(batch_size):
            i_consts.append(self.scratch_const(i))
        
        # Allocate registers for each unrolled iteration
        tmp_regs = []
        for u in range(UNROLL_FACTOR):
            tmp_regs.append({
                'idx': self.alloc_scratch(f"tmp_idx_{u}"),
                'val': self.alloc_scratch(f"tmp_val_{u}"),
                'node_val': self.alloc_scratch(f"tmp_node_val_{u}"),
                'addr': self.alloc_scratch(f"tmp_addr_{u}"),
                'tmp1': self.alloc_scratch(f"tmp1_{u}"),
                'tmp2': self.alloc_scratch(f"tmp2_{u}"),
                'tmp3': self.alloc_scratch(f"tmp3_{u}"),
            })

        for round in range(rounds):
            # Process in batches of UNROLL_FACTOR
            for i_base in range(0, batch_size, UNROLL_FACTOR):
                num_iterations = min(UNROLL_FACTOR, batch_size - i_base)
                
                # Stage 1: Load all indices in parallel
                # Batch address calculations
                for u in range(num_iterations):
                    i = i_base + u
                    i_const = i_consts[i]
                    tr = tmp_regs[u]
                    body.append(("alu", ("+", tr['addr'], self.scratch["inp_indices_p"], i_const)))
                # Batch loads
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("load", ("load", tr['idx'], tr['addr'])))
                # Batch debugs
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("debug", ("compare", tr['idx'], (round, i, "idx"))))
                
                # Stage 2: Load all values in parallel
                for u in range(num_iterations):
                    i = i_base + u
                    i_const = i_consts[i]
                    tr = tmp_regs[u]
                    body.append(("alu", ("+", tr['addr'], self.scratch["inp_values_p"], i_const)))
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("load", ("load", tr['val'], tr['addr'])))
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("debug", ("compare", tr['val'], (round, i, "val"))))
                
                # Stage 3: Load all node values in parallel
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("alu", ("+", tr['addr'], self.scratch["forest_values_p"], tr['idx'])))
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("load", ("load", tr['node_val'], tr['addr'])))
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("debug", ("compare", tr['node_val'], (round, i, "node_val"))))
                
                # Stage 4: Hash computation for all iterations
                # XOR operation for all iterations first
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("alu", ("^", tr['val'], tr['val'], tr['node_val'])))
                
                # Process each hash stage for all iterations
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    # First ALU op for all iterations
                    for u in range(num_iterations):
                        i = i_base + u
                        tr = tmp_regs[u]
                        body.append(("alu", (op1, tr['tmp1'], tr['val'], self.scratch_const(val1))))
                    # Second ALU op for all iterations
                    for u in range(num_iterations):
                        i = i_base + u
                        tr = tmp_regs[u]
                        body.append(("alu", (op3, tr['tmp2'], tr['val'], self.scratch_const(val3))))
                    # Third ALU op for all iterations
                    for u in range(num_iterations):
                        i = i_base + u
                        tr = tmp_regs[u]
                        body.append(("alu", (op2, tr['val'], tr['tmp1'], tr['tmp2'])))
                    # Debug compares for all iterations
                    for u in range(num_iterations):
                        i = i_base + u
                        tr = tmp_regs[u]
                        body.append(("debug", ("compare", tr['val'], (round, i, "hash_stage", hi))))
                
                # Final hash debug for all iterations
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("debug", ("compare", tr['val'], (round, i, "hashed_val"))))
                
                # Stage 5: Update indices for all iterations
                # Batch modulo operations
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("alu", ("%", tr['tmp1'], tr['val'], two_const)))
                # Batch equality checks
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("alu", ("==", tr['tmp1'], tr['tmp1'], zero_const)))
                # Batch selects
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("flow", ("select", tr['tmp3'], tr['tmp1'], one_const, two_const)))
                # Batch multiplications
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("alu", ("*", tr['idx'], tr['idx'], two_const)))
                # Batch additions
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("alu", ("+", tr['idx'], tr['idx'], tr['tmp3'])))
                # Batch debug compares
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("debug", ("compare", tr['idx'], (round, i, "next_idx"))))
                # Batch less-than checks
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("alu", ("<", tr['tmp1'], tr['idx'], self.scratch["n_nodes"])))
                # Batch wrap selects
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("flow", ("select", tr['idx'], tr['tmp1'], tr['idx'], zero_const)))
                # Batch final debug compares
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("debug", ("compare", tr['idx'], (round, i, "wrapped_idx"))))
                
                # Stage 6: Store all results back to memory
                # Batch address calculations for indices
                for u in range(num_iterations):
                    i = i_base + u
                    i_const = i_consts[i]
                    tr = tmp_regs[u]
                    body.append(("alu", ("+", tr['addr'], self.scratch["inp_indices_p"], i_const)))
                # Batch stores for indices
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("store", ("store", tr['addr'], tr['idx'])))
                # Batch address calculations for values
                for u in range(num_iterations):
                    i = i_base + u
                    i_const = i_consts[i]
                    tr = tmp_regs[u]
                    body.append(("alu", ("+", tr['addr'], self.scratch["inp_values_p"], i_const)))
                # Batch stores for values
                for u in range(num_iterations):
                    i = i_base + u
                    tr = tmp_regs[u]
                    body.append(("store", ("store", tr['addr'], tr['val'])))

        body_instrs = self.build(body)  # Use VLIW bundling
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
