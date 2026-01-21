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
        """Analyze read and write dependencies for an instruction."""
        reads = set()
        writes = set()
        
        if engine == "alu":
            # ALU: (op, dest, src1, src2)
            op, dest, src1, src2 = slot
            writes.add(dest)
            reads.add(src1)
            reads.add(src2)
        elif engine == "load":
            if slot[0] == "const":
                # const: (const, dest, val)
                writes.add(slot[1])
            elif slot[0] == "load":
                # load: (load, dest, addr)
                writes.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vload":
                # vload: (vload, dest, addr)
                writes.add(slot[1])
                reads.add(slot[2])
        elif engine == "store":
            if slot[0] == "store":
                # store: (store, addr, src)
                reads.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vstore":
                # vstore: (vstore, addr, src)
                reads.add(slot[1])
                reads.add(slot[2])
        elif engine == "flow":
            if slot[0] == "select":
                # select: (select, dest, cond, a, b)
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
            elif slot[0] in ["pause", "halt"]:
                pass  # No dependencies
            else:
                # Conservative: assume reads and writes
                pass
        elif engine == "debug":
            # Debug operations don't affect actual execution
            if slot[0] == "compare":
                reads.add(slot[1])
        
        return reads, writes

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """
        Build instruction bundles with dependency-aware packing.
        Packs independent instructions into bundles respecting SLOT_LIMITS.
        """
        if not vliw:
            # Fallback to simple one-per-cycle for debugging
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        
        instrs = []
        current_bundle = {}
        current_writes = set()
        current_reads = set()
        engine_counts = defaultdict(int)
        
        for engine, slot in slots:
            reads, writes = self._analyze_dependencies(engine, slot)
            
            # Check for RAW hazard: reading what hasn't been written yet
            has_hazard = bool(reads & current_writes)
            
            # Check if we've exceeded slot limits
            would_exceed_limit = engine_counts[engine] >= SLOT_LIMITS.get(engine, 1)
            
            # Start new bundle if hazard detected or slot limit reached
            if has_hazard or would_exceed_limit:
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {}
                current_writes = set()
                current_reads = set()
                engine_counts = defaultdict(int)
            
            # Add instruction to current bundle
            if engine not in current_bundle:
                current_bundle[engine] = []
            current_bundle[engine].append(slot)
            engine_counts[engine] += 1
            
            # Update dependency tracking
            current_reads.update(reads)
            current_writes.update(writes)
        
        # Add final bundle
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

    def build_hash(self, val_hash_addr, hash_tmp1_addrs, hash_tmp2_addrs, round, i):
        """
        Optimized hash with operation grouping for ILP within each stage.
        Hash stages are sequential, but op1 and op3 can run in parallel.
        Uses pre-allocated temporaries per unroll iteration.
        """
        slots = []
        
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # op1 and op3 can run in parallel (both read val_hash_addr)
            slots.append(("alu", (op1, hash_tmp1_addrs[hi], val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, hash_tmp2_addrs[hi], val_hash_addr, self.scratch_const(val3))))
            # op2 depends on op1 and op3
            slots.append(("alu", (op2, val_hash_addr, hash_tmp1_addrs[hi], hash_tmp2_addrs[hi])))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Optimized with Fibonacci unrolling and software pipelining.
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

        # Pre-allocate constants
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

        # Fibonacci unrolling factor
        UNROLL_FACTOR = 13
        
        # Allocate separate registers per unrolled iteration
        tmp_regs = []
        for u in range(UNROLL_FACTOR):
            regs = {
                'idx': self.alloc_scratch(f"tmp_idx_{u}"),
                'val': self.alloc_scratch(f"tmp_val_{u}"),
                'node_val': self.alloc_scratch(f"tmp_node_val_{u}"),
                'addr': self.alloc_scratch(f"tmp_addr_{u}"),
                'tmp1': self.alloc_scratch(f"tmp1_{u}"),
                'tmp2': self.alloc_scratch(f"tmp2_{u}"),
                'tmp3': self.alloc_scratch(f"tmp3_{u}"),
            }
            # Allocate hash temporaries per unroll iteration
            regs['hash_tmp1'] = [self.alloc_scratch(f"hash_tmp1_{u}_{hi}") for hi in range(len(HASH_STAGES))]
            regs['hash_tmp2'] = [self.alloc_scratch(f"hash_tmp2_{u}_{hi}") for hi in range(len(HASH_STAGES))]
            tmp_regs.append(regs)

        for round in range(rounds):
            # Process in groups of UNROLL_FACTOR
            for i_base in range(0, batch_size, UNROLL_FACTOR):
                # Software pipeline: Stage 1 - Load all indices in parallel
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    regs = tmp_regs[u]
                    # idx = mem[inp_indices_p + i]
                    body.append(("alu", ("+", regs['addr'], self.scratch["inp_indices_p"], i_const)))
                    body.append(("load", ("load", regs['idx'], regs['addr'])))
                
                # Debug stage 1
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['idx'], (round, i, "idx"))))
                
                # Stage 2 - Load all values in parallel
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    regs = tmp_regs[u]
                    # val = mem[inp_values_p + i]
                    body.append(("alu", ("+", regs['addr'], self.scratch["inp_values_p"], i_const)))
                    body.append(("load", ("load", regs['val'], regs['addr'])))
                
                # Debug stage 2
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['val'], (round, i, "val"))))
                
                # Stage 3 - Load all node values in parallel
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    # node_val = mem[forest_values_p + idx]
                    body.append(("alu", ("+", regs['addr'], self.scratch["forest_values_p"], regs['idx'])))
                    body.append(("load", ("load", regs['node_val'], regs['addr'])))
                
                # Debug stage 3
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['node_val'], (round, i, "node_val"))))
                
                # Stage 4 - Hash operations in parallel
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    # val = myhash(val ^ node_val)
                    body.append(("alu", ("^", regs['val'], regs['val'], regs['node_val'])))
                    body.extend(self.build_hash(regs['val'], regs['hash_tmp1'], regs['hash_tmp2'], round, i))
                
                # Debug hash
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['val'], (round, i, "hashed_val"))))
                
                # Stage 5 - Update calculations in parallel
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    body.append(("alu", ("%", regs['tmp1'], regs['val'], two_const)))
                    body.append(("alu", ("==", regs['tmp1'], regs['tmp1'], zero_const)))
                    # Replace select with arithmetic: tmp3 = 2 - tmp1 (when tmp1 is 0 or 1)
                    body.append(("alu", ("-", regs['tmp3'], two_const, regs['tmp1'])))
                    body.append(("alu", ("*", regs['idx'], regs['idx'], two_const)))
                    body.append(("alu", ("+", regs['idx'], regs['idx'], regs['tmp3'])))
                
                # Debug next_idx
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['idx'], (round, i, "next_idx"))))
                
                # Wrap indices
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    # idx = 0 if idx >= n_nodes else idx
                    body.append(("alu", ("<", regs['tmp1'], regs['idx'], self.scratch["n_nodes"])))
                    body.append(("flow", ("select", regs['idx'], regs['tmp1'], regs['idx'], zero_const)))
                
                # Debug wrapped_idx
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    regs = tmp_regs[u]
                    body.append(("debug", ("compare", regs['idx'], (round, i, "wrapped_idx"))))
                
                # Stage 6 - Store all results in parallel
                for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    regs = tmp_regs[u]
                    # mem[inp_indices_p + i] = idx
                    body.append(("alu", ("+", regs['addr'], self.scratch["inp_indices_p"], i_const)))
                    body.append(("store", ("store", regs['addr'], regs['idx'])))
                    # mem[inp_values_p + i] = val
                    body.append(("alu", ("+", regs['addr'], self.scratch["inp_values_p"], i_const)))
                    body.append(("store", ("store", regs['addr'], regs['val'])))

        body_instrs = self.build(body)
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
