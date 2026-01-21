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

    def _analyze_dependencies(self, engine: Engine, slot: tuple):
        """
        Analyze dependencies for a given engine and slot.
        Returns (reads: set, writes: set) of scratch addresses.
        """
        reads = set()
        writes = set()
        
        if engine == "alu":
            # (op, dest, src1, src2)
            op, dest, src1, src2 = slot
            reads.add(src1)
            reads.add(src2)
            writes.add(dest)
        elif engine == "load":
            if slot[0] == "const":
                # ("const", dest, val) - only writes
                writes.add(slot[1])
            elif slot[0] == "load":
                # ("load", dest, addr)
                writes.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vload":
                # ("vload", dest, addr)
                dest, addr = slot[1], slot[2]
                for i in range(8):  # VLEN = 8
                    writes.add(dest + i)
                reads.add(addr)
        elif engine == "store":
            if slot[0] == "store":
                # ("store", addr, src)
                reads.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vstore":
                # ("vstore", addr, src)
                addr, src = slot[1], slot[2]
                reads.add(addr)
                for i in range(8):  # VLEN = 8
                    reads.add(src + i)
        elif engine == "flow":
            if slot[0] == "select":
                # ("select", dest, cond, a, b)
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
            elif slot[0] == "add_imm":
                # ("add_imm", dest, a, imm)
                writes.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vselect":
                # ("vselect", dest, cond, a, b)
                dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
                for i in range(8):  # VLEN = 8
                    writes.add(dest + i)
                    reads.add(cond + i)
                    reads.add(a + i)
                    reads.add(b + i)
            elif slot[0] in ["pause", "halt"]:
                pass  # No dependencies
        elif engine == "debug":
            # Debug operations have no real dependencies
            pass
        
        return (reads, writes)
    
    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """
        Pack slots into VLIW instruction bundles with dependency awareness.
        """
        if not vliw:
            # Fall back to naive packing for debugging
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        
        instrs = []
        current_bundle = {}
        slot_counts = defaultdict(int)
        written_this_cycle = set()
        
        for engine, slot in slots:
            reads, writes = self._analyze_dependencies(engine, slot)
            
            # Check if we can add this slot to the current bundle
            can_add = True
            
            # Check slot limit
            if slot_counts[engine] >= SLOT_LIMITS.get(engine, 1):
                can_add = False
            
            # Check for RAW (Read-After-Write) hazard
            if reads & written_this_cycle:
                can_add = False
            
            # If we can't add to current bundle, emit it and start a new one
            if not can_add:
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {}
                slot_counts = defaultdict(int)
                written_this_cycle = set()
            
            # Add slot to current bundle
            if engine not in current_bundle:
                current_bundle[engine] = []
            current_bundle[engine].append(slot)
            slot_counts[engine] += 1
            written_this_cycle.update(writes)
        
        # Emit final bundle
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
        Optimized with loop unrolling and interleaved operations for maximum VLIW packing.
        """
        UNROLL_FACTOR = 13  # Fibonacci number for golden resonance
        
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

        # Pre-allocate all constants for i values
        i_const_addrs = [self.scratch_const(i) for i in range(batch_size)]
        zero_const = i_const_addrs[0]
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        # Allocate separate scratch registers per unroll iteration
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

        body = []  # array of slots

        for round in range(rounds):
            for i_base in range(0, batch_size, UNROLL_FACTOR):
                iterations = min(UNROLL_FACTOR, batch_size - i_base)
                
                # Process all iterations in parallel as much as possible
                # Instead of stage-by-stage, we interleave operations
                
                # Stage 1: Load indices (address calc + load)
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("+", regs['addr'], self.scratch["inp_indices_p"], i_const_addrs[i])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("load", ("load", regs['idx'], regs['addr'])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['idx'], (round, i, "idx"))))
                
                # Stage 2: Load values (address calc + load)
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("+", regs['addr'], self.scratch["inp_values_p"], i_const_addrs[i])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("load", ("load", regs['val'], regs['addr'])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['val'], (round, i, "val"))))
                
                # Stage 3: Load node values (address calc + load)
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("+", regs['addr'], self.scratch["forest_values_p"], regs['idx'])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("load", ("load", regs['node_val'], regs['addr'])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['node_val'], (round, i, "node_val"))))
                
                # Stage 4: XOR values
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("^", regs['val'], regs['val'], regs['node_val'])))
                
                # Stage 5: Hash computation - interleave hash operations from different iterations
                # Each hash stage has 3 ALU ops, so we can run 4 iterations in parallel (12 ALU slots)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    # Do op1 for all iterations
                    for u in range(iterations):
                        i = i_base + u
                        regs = tmp_regs[u]
                        body.append(("alu", (op1, regs['tmp1'], regs['val'], self.scratch_const(val1))))
                    # Do op3 for all iterations
                    for u in range(iterations):
                        i = i_base + u
                        regs = tmp_regs[u]
                        body.append(("alu", (op3, regs['tmp2'], regs['val'], self.scratch_const(val3))))
                    # Do op2 for all iterations
                    for u in range(iterations):
                        i = i_base + u
                        regs = tmp_regs[u]
                        body.append(("alu", (op2, regs['val'], regs['tmp1'], regs['tmp2'])))
                        body.append(("debug", ("compare", regs['val'], (round, i, "hash_stage", hi))))
                
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['val'], (round, i, "hashed_val"))))
                
                # Stage 6: Update indices - replace flow select with arithmetic
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("%", regs['tmp1'], regs['val'], two_const)))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("==", regs['tmp1'], regs['tmp1'], zero_const)))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    # When cond=1 (even): 2-1=1, When cond=0 (odd): 2-0=2
                    body.append(("alu", ("-", regs['tmp3'], two_const, regs['tmp1'])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("*", regs['idx'], regs['idx'], two_const)))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("+", regs['idx'], regs['idx'], regs['tmp3'])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['idx'], (round, i, "next_idx"))))
                
                # Stage 7: Wrap indices
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("<", regs['tmp1'], regs['idx'], self.scratch["n_nodes"])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    # When cond=1: idx*1 = idx, When cond=0: idx*0 = 0
                    body.append(("alu", ("*", regs['idx'], regs['idx'], regs['tmp1'])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['idx'], (round, i, "wrapped_idx"))))
                
                # Stage 8: Store results (address calc + store)
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("+", regs['addr'], self.scratch["inp_indices_p"], i_const_addrs[i])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("store", ("store", regs['addr'], regs['idx'])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("alu", ("+", regs['addr'], self.scratch["inp_values_p"], i_const_addrs[i])))
                for u in range(iterations):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("store", ("store", regs['addr'], regs['val'])))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
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
