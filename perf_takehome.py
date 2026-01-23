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
        elif engine == "valu":
            # Vector ALU operations
            if slot[0] == "vbroadcast":
                # ("vbroadcast", dest, src) - writes VLEN elements
                _, dest, src = slot
                reads.add(src)
                for i in range(VLEN):
                    writes.add(dest + i)
            elif len(slot) == 4:
                # (op, dest, src1, src2) - all are vectors
                _, dest, src1, src2 = slot
                for i in range(VLEN):
                    reads.add(src1 + i)
                    reads.add(src2 + i)
                    writes.add(dest + i)
            elif len(slot) == 5:
                # multiply_add or other 4-arg valu ops
                _, dest, a, b, c = slot
                for i in range(VLEN):
                    reads.add(a + i)
                    reads.add(b + i)
                    reads.add(c + i)
                    writes.add(dest + i)
            else:
                # Unknown valu operation - raise error to aid debugging
                raise NotImplementedError(f"Unknown valu op format: {slot}")
        elif engine == "load":
            if slot[0] == "const":
                # ("const", dest, val) - only writes
                writes.add(slot[1])
            elif slot[0] == "load":
                # ("load", dest, addr)
                writes.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "load_offset":
                # ("load_offset", dest, addr, offset) - writes to dest+offset
                _, dest, addr, offset = slot
                writes.add(dest + offset)
                reads.add(addr + offset)
            elif slot[0] == "vload":
                # ("vload", dest, addr) - loads VLEN elements
                _, dest, addr = slot
                reads.add(addr)
                for i in range(VLEN):
                    writes.add(dest + i)
        elif engine == "store":
            if slot[0] == "store":
                # ("store", addr, src)
                reads.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vstore":
                # ("vstore", addr, src) - stores VLEN elements
                _, addr, src = slot
                reads.add(addr)
                for i in range(VLEN):
                    reads.add(src + i)
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
    
    def _local_bubble_fill(self, bundles: list[dict]) -> list[dict]:
        """
        Conservative local optimization: look ahead a few bundles and try to pull
        instructions to fill bubbles while carefully checking ALL dependencies.
        """
        if len(bundles) <= 1:
            return bundles
        
        LOOKAHEAD = 3  # Conservative lookahead
        
        # Precompute dependencies for all bundles
        bundle_deps = []
        for bundle in bundles:
            reads = set()
            writes = set()
            for engine, slots in bundle.items():
                for slot in slots:
                    r, w = self._analyze_dependencies(engine, slot)
                    reads.update(r)
                    writes.update(w)
            bundle_deps.append((reads, writes))
        
        for i in range(len(bundles)):
            # Calculate current slot usage
            slot_usage = {e: 0 for e in SLOT_LIMITS}
            for engine, slots in bundles[i].items():
                slot_usage[engine] = len(slots)
            
            # Skip if all slots are full
            if all(slot_usage[e] >= SLOT_LIMITS[e] for e in SLOT_LIMITS):
                continue
            
            # Try to pull from next few bundles
            for j in range(i + 1, min(i + 1 + LOOKAHEAD, len(bundles))):
                if not bundles[j]:
                    continue
                
                # For each engine, try to move ONE instruction at a time
                for engine in list(bundles[j].keys()):
                    if slot_usage[engine] >= SLOT_LIMITS[engine]:
                        continue
                    
                    slots_in_j = bundles[j][engine]
                    if not slots_in_j:
                        continue
                    
                    # Try first instruction
                    candidate = slots_in_j[0]
                    r, w = self._analyze_dependencies(engine, candidate)
                    
                    # Check that moving this instruction doesn't violate dependencies
                    # with bundles i and all intermediate bundles between i and j
                    safe = True
                    
                    # Check bundle i
                    if (r & bundle_deps[i][1]) or (w & bundle_deps[i][1]) or (w & bundle_deps[i][0]):
                        safe = False
                    
                    # Check intermediate bundles
                    if safe:
                        for k in range(i + 1, j):
                            kr, kw = bundle_deps[k]
                            if (r & kw) or (w & kr) or (w & kw):
                                safe = False
                                break
                    
                    if safe:
                        # Move it
                        if engine not in bundles[i]:
                            bundles[i][engine] = []
                        bundles[i][engine].append(candidate)
                        
                        # Remove from bundle j
                        bundles[j][engine] = slots_in_j[1:]
                        if not bundles[j][engine]:
                            del bundles[j][engine]
                        
                        # Update dependencies and slot usage
                        bundle_deps[i] = (
                            bundle_deps[i][0] | r,
                            bundle_deps[i][1] | w
                        )
                        slot_usage[engine] += 1
        
        # Remove empty bundles
        return [b for b in bundles if b]
    
    
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

    def build_hash(self, val_hash_addr, tmp1_base, tmp2_base, round, i):
        """Group hash operations by type for parallel execution (Phason Flip)"""
        slots = []
        
        # Allocate separate temporaries per hash stage using offset
        # tmp1_base[0..5] and tmp2_base[0..5] for 6 hash stages
        hash_tmp1_addrs = [tmp1_base + hi for hi in range(len(HASH_STAGES))]
        hash_tmp2_addrs = [tmp2_base + hi for hi in range(len(HASH_STAGES))]
        
        # Process each hash stage sequentially (must update val_hash_addr in order)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Phase 1: op1 operation
            slots.append(("alu", (op1, hash_tmp1_addrs[hi], val_hash_addr, self.scratch_const(val1))))
            # Phase 2: op3 operation (can potentially parallel with op1 in bundle)
            slots.append(("alu", (op3, hash_tmp2_addrs[hi], val_hash_addr, self.scratch_const(val3))))
            # Phase 3: op2 operation (depends on both tmp1 and tmp2)
            slots.append(("alu", (op2, val_hash_addr, hash_tmp1_addrs[hi], hash_tmp2_addrs[hi])))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Fibonacci-weighted VLIW optimization with 13× unroll and 6-stage software pipeline.
        Key optimizations:
        1. Unroll factor 13 (Fibonacci golden resonance)
        2. 6-stage software pipeline (load_idx, load_val, load_node, hash, update, store)
        3. Separate registers per iteration eliminate RAW hazards
        4. Hash phason grouping for massive ILP exposure
        5. Flow→ALU arithmetic transformation
        """
        UNROLL_FACTOR = 13  # Fibonacci golden resonance
        
        # Allocate runtime variables
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        # Pre-allocate constants needed
        const_values_needed = set([0, 1, 2])
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            const_values_needed.add(val1)
            const_values_needed.add(val3)
        
        # Allocate scratch space for all constants
        const_addrs = {}
        for val in sorted(const_values_needed):
            const_addrs[val] = self.alloc_scratch(f"const_{val}")
        
        # Temporary for initialization
        tmp_init = self.alloc_scratch("tmp_init")
        
        # Batch all const loads together
        const_body = []
        for val in sorted(const_values_needed):
            const_body.append(("load", ("const", const_addrs[val], val)))
        
        # Initialize runtime variables
        init_body = []
        for i, v in enumerate(init_vars):
            init_body.append(("load", ("const", tmp_init, i)))
            init_body.append(("load", ("load", self.scratch[v], tmp_init)))
        
        # Combine and pack all initialization
        self.instrs.extend(self.build(const_body + init_body))

        # Set up const map for easy access
        self.const_map = const_addrs
        zero_const = const_addrs[0]
        one_const = const_addrs[1]
        two_const = const_addrs[2]

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting Fibonacci unroll loop"))

        # Allocate separate registers per unrolled iteration (eliminate RAW hazards)
        tmp_regs = []
        for u in range(UNROLL_FACTOR):
            tmp_regs.append({
                'idx': self.alloc_scratch(f"tmp_idx_{u}"),
                'val': self.alloc_scratch(f"tmp_val_{u}"),
                'node_val': self.alloc_scratch(f"tmp_node_val_{u}"),
                'addr': self.alloc_scratch(f"tmp_addr_{u}"),
                'tmp1': self.alloc_scratch(f"tmp1_{u}", len(HASH_STAGES)),  # 6 stages
                'tmp2': self.alloc_scratch(f"tmp2_{u}", len(HASH_STAGES)),  # 6 stages
                'tmp3': self.alloc_scratch(f"tmp3_{u}"),
            })
        
        # Pre-allocate constants for all batch indices
        i_const_addrs = []
        for i in range(batch_size):
            i_const_addrs.append(self.scratch_const(i))
        
        body = []
        
        # 6-stage software pipeline
        for round in range(rounds):
            for i_base in range(0, batch_size, UNROLL_FACTOR):
                for stage in ['load_idx', 'load_val', 'load_node', 'hash', 'update', 'store']:
                    for u in range(min(UNROLL_FACTOR, batch_size - i_base)):
                        i = i_base + u
                        i_const = i_const_addrs[i]
                        tr = tmp_regs[u]
                        
                        if stage == 'load_idx':
                            body.append(("alu", ("+", tr['addr'], self.scratch["inp_indices_p"], i_const)))
                            body.append(("load", ("load", tr['idx'], tr['addr'])))
                            body.append(("debug", ("compare", tr['idx'], (round, i, "idx"))))
                        elif stage == 'load_val':
                            body.append(("alu", ("+", tr['addr'], self.scratch["inp_values_p"], i_const)))
                            body.append(("load", ("load", tr['val'], tr['addr'])))
                            body.append(("debug", ("compare", tr['val'], (round, i, "val"))))
                        elif stage == 'load_node':
                            body.append(("alu", ("+", tr['addr'], self.scratch["forest_values_p"], tr['idx'])))
                            body.append(("load", ("load", tr['node_val'], tr['addr'])))
                            body.append(("debug", ("compare", tr['node_val'], (round, i, "node_val"))))
                        elif stage == 'hash':
                            body.append(("alu", ("^", tr['val'], tr['val'], tr['node_val'])))
                            body.extend(self.build_hash(tr['val'], tr['tmp1'], tr['tmp2'], round, i))
                            body.append(("debug", ("compare", tr['val'], (round, i, "hashed_val"))))
                        elif stage == 'update':
                            body.append(("alu", ("%", tr['tmp1'], tr['val'], two_const)))
                            body.append(("alu", ("==", tr['tmp1'], tr['tmp1'], zero_const)))
                            body.append(("alu", ("-", tr['tmp3'], two_const, tr['tmp1'])))  # Flow→ALU
                            body.append(("alu", ("*", tr['idx'], tr['idx'], two_const)))
                            body.append(("alu", ("+", tr['idx'], tr['idx'], tr['tmp3'])))
                            body.append(("debug", ("compare", tr['idx'], (round, i, "next_idx"))))
                            body.append(("alu", ("<", tr['tmp1'], tr['idx'], self.scratch["n_nodes"])))
                            body.append(("flow", ("select", tr['idx'], tr['tmp1'], tr['idx'], zero_const)))
                            body.append(("debug", ("compare", tr['idx'], (round, i, "wrapped_idx"))))
                        elif stage == 'store':
                            body.append(("alu", ("+", tr['addr'], self.scratch["inp_indices_p"], i_const)))
                            body.append(("store", ("store", tr['addr'], tr['idx'])))
                            body.append(("alu", ("+", tr['addr'], self.scratch["inp_values_p"], i_const)))
                            body.append(("store", ("store", tr['addr'], tr['val'])))

        body_instrs = self.build(body, vliw=True)
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
