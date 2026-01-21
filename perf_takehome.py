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
        elif engine == "store":
            if slot[0] == "store":
                # ("store", addr, src)
                reads.add(slot[1])
                reads.add(slot[2])
        elif engine == "flow":
            if slot[0] == "select":
                # ("select", dest, cond, a, b)
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
        elif engine == "debug":
            if slot[0] == "compare":
                # ("compare", addr, ...) - reads the value at addr
                reads.add(slot[1])
        
        return (reads, writes)
    
    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """
        Pack slots into VLIW instruction bundles with dependency awareness.
        """
        if not vliw:
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
        Highly optimized kernel with aggressive unrolling and instruction-level parallelism.
        Key optimizations:
        1. Unroll loop by UNROLL_FACTOR to expose ILP
        2. Allocate separate registers per iteration to eliminate dependencies
        3. Group operations by type to maximize VLIW packing (12 ALU, 2 load, 2 store per cycle)
        4. Replace flow ops with arithmetic to increase ALU utilization
        5. Pre-compute and cache hash constants
        6. Batch all constant loads together for maximum VLIW packing
        """
        # Smaller unroll factor for better packing - matches ALU slot limit
        UNROLL_FACTOR = 12
        
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        
        # Allocate runtime variables
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        # Pre-allocate all scratch addresses for constants first
        # This prevents them from being added to instrs during allocation
        const_values_needed = set([0, 1, 2])  # zero, one, two
        const_values_needed.update(range(batch_size))  # i values
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            const_values_needed.add(val1)
            const_values_needed.add(val3)
        
        # Allocate scratch space for all constants
        const_addrs = {}
        for val in sorted(const_values_needed):
            const_addrs[val] = self.alloc_scratch(f"const_{val}")
        
        # Now batch all const loads together
        const_body = []
        for val in sorted(const_values_needed):
            const_body.append(("load", ("const", const_addrs[val], val)))
        
        # Initialize runtime variables
        init_body = []
        for i, v in enumerate(init_vars):
            init_body.append(("load", ("const", tmp1, i)))
            init_body.append(("load", ("load", self.scratch[v], tmp1)))
        
        # Combine and pack all initialization
        self.instrs.extend(self.build(const_body + init_body))

        # Set up const map for easy access
        self.const_map = const_addrs
        i_consts = [const_addrs[i] for i in range(batch_size)]
        zero_const = const_addrs[0]
        one_const = const_addrs[1]
        two_const = const_addrs[2]
        
        # Hash constants
        hash_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_consts.append((const_addrs[val1], const_addrs[val3]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        # Allocate registers for each unrolled iteration
        # Optimize: use fewer address registers per iteration
        regs = []
        for u in range(UNROLL_FACTOR):
            regs.append({
                'idx': self.alloc_scratch(f"idx{u}"),
                'val': self.alloc_scratch(f"val{u}"),
                'nval': self.alloc_scratch(f"nval{u}"),
                'addr1': self.alloc_scratch(f"a1_{u}"),  # Reusable address register
                'addr2': self.alloc_scratch(f"a2_{u}"),  # Reusable address register
                't1': self.alloc_scratch(f"t1_{u}"),
                't2': self.alloc_scratch(f"t2_{u}"),
                't3': self.alloc_scratch(f"t3_{u}"),
            })

        body = []

        for round in range(rounds):
            for i_base in range(0, batch_size, UNROLL_FACTOR):
                n = min(UNROLL_FACTOR, batch_size - i_base)
                
                # PHASE 1: Load indices
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("+", r['addr1'], self.scratch["inp_indices_p"], i_consts[i_base+u])))
                for u in range(n):
                    r = regs[u]
                    body.append(("load", ("load", r['idx'], r['addr1'])))
                
                # PHASE 2: Load values
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("+", r['addr1'], self.scratch["inp_values_p"], i_consts[i_base+u])))
                for u in range(n):
                    r = regs[u]
                    body.append(("load", ("load", r['val'], r['addr1'])))
                
                # PHASE 3: Load node values
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("+", r['addr1'], self.scratch["forest_values_p"], r['idx'])))
                for u in range(n):
                    r = regs[u]
                    body.append(("load", ("load", r['nval'], r['addr1'])))
                
                # PHASE 4: XOR
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("^", r['val'], r['val'], r['nval'])))
                
                # PHASE 5: Hash
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1, c3 = hash_consts[hi]
                    
                    for u in range(n):
                        r = regs[u]
                        body.append(("alu", (op1, r['t1'], r['val'], c1)))
                    
                    for u in range(n):
                        r = regs[u]
                        body.append(("alu", (op3, r['t2'], r['val'], c3)))
                    
                    for u in range(n):
                        r = regs[u]
                        body.append(("alu", (op2, r['val'], r['t1'], r['t2'])))
                
                # PHASE 6: Update indices
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("%", r['t1'], r['val'], two_const)))
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("==", r['t1'], r['t1'], zero_const)))
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("-", r['t3'], two_const, r['t1'])))
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("*", r['idx'], r['idx'], two_const)))
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("+", r['idx'], r['idx'], r['t3'])))
                
                # PHASE 7: Wrap indices
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("<", r['t1'], r['idx'], self.scratch["n_nodes"])))
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("*", r['idx'], r['idx'], r['t1'])))
                
                # PHASE 8: Store results
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("+", r['addr1'], self.scratch["inp_indices_p"], i_consts[i_base+u])))
                for u in range(n):
                    r = regs[u]
                    body.append(("store", ("store", r['addr1'], r['idx'])))
                for u in range(n):
                    r = regs[u]
                    body.append(("alu", ("+", r['addr2'], self.scratch["inp_values_p"], i_consts[i_base+u])))
                for u in range(n):
                    r = regs[u]
                    body.append(("store", ("store", r['addr2'], r['val'])))

        self.instrs.extend(self.build(body))
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
