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
        
        LOOKAHEAD = 100  # Very aggressive lookahead for maximum bubble filling
        
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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        SIMD vectorized kernel using vload/vstore/valu instructions.
        Key optimizations:
        1. Process 8 elements at once using VLEN=8 vector operations
        2. Use vload/vstore for contiguous memory access (indices and values)
        3. Use valu for vectorizable operations (XOR, hash arithmetic)
        4. Unroll by 13 (Fibonacci number) for optimal ILP
        5. Software pipelining in 6 stages to maximize parallelism
        6. Pre-broadcast constants outside loop to reduce redundant operations
        """
        # Optimal unroll factor for balanced ILP and packing efficiency  
        UNROLL_FACTOR = 8
        
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
        
        # Pre-allocate constants needed
        const_values_needed = set([0, 1, 2, VLEN])
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            const_values_needed.add(val1)
            const_values_needed.add(val3)
        
        # Allocate scratch space for all constants
        const_addrs = {}
        for val in sorted(const_values_needed):
            const_addrs[val] = self.alloc_scratch(f"const_{val}")
        
        # Batch all const loads together
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
        zero_const = const_addrs[0]
        one_const = const_addrs[1]
        two_const = const_addrs[2]
        vlen_const = const_addrs[VLEN]
        
        # Hash constants - we'll need to broadcast these for vector ops
        hash_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_consts.append((const_addrs[val1], const_addrs[val3]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting SIMD loop"))

        # Allocate vector registers for unrolled iterations
        # Each vector register holds VLEN=8 elements
        vregs = []
        for u in range(UNROLL_FACTOR):
            vregs.append({
                'idx_vec': self.alloc_scratch(f"idx_vec{u}", VLEN),
                'val_vec': self.alloc_scratch(f"val_vec{u}", VLEN),
                'addr_base': self.alloc_scratch(f"addr_base{u}"),
                't1_vec': self.alloc_scratch(f"t1_vec{u}", VLEN),
                't2_vec': self.alloc_scratch(f"t2_vec{u}", VLEN),
                't3_vec': self.alloc_scratch(f"t3_vec{u}", VLEN),
                'hash_c1': self.alloc_scratch(f"hash_c1_{u}", VLEN),
                'hash_c3': self.alloc_scratch(f"hash_c3_{u}", VLEN),
            })
        
        # Allocate scalar registers for the indexed loads (can't vectorize)
        # We only need addr now since we use load_offset to write directly to vector
        scalar_regs = []
        for s in range(VLEN * UNROLL_FACTOR):
            scalar_regs.append({
                'addr': self.alloc_scratch(f"s_addr{s}"),
            })
        
        # Number of vector groups = batch_size / VLEN
        n_vec_groups = batch_size // VLEN
        
        # Pre-allocate and broadcast common constants outside the loop
        # This avoids redundant broadcasts in each iteration
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        
        # Pre-compute offset constants for all possible vec_base positions
        # This eliminates const loads inside the loop
        offset_consts = []
        for vec_idx in range(n_vec_groups):
            offset_consts.append(self.alloc_scratch(f"offset_{vec_idx}"))
        
        # Allocate hash constant vectors
        hash_const_vecs = []
        for hi, (c1, c3) in enumerate(hash_consts):
            hash_const_vecs.append({
                'c1_vec': self.alloc_scratch(f"hash_c1_vec{hi}", VLEN),
                'c3_vec': self.alloc_scratch(f"hash_c3_vec{hi}", VLEN),
            })
        
        # Load and broadcast constants once before the loop
        pre_body = []
        
        # Load offset constants
        for vec_idx in range(n_vec_groups):
            pre_body.append(("load", ("const", offset_consts[vec_idx], vec_idx * VLEN)))
        
        # Broadcast common vectors
        pre_body.append(("valu", ("vbroadcast", zero_vec, zero_const)))
        pre_body.append(("valu", ("vbroadcast", two_vec, two_const)))
        pre_body.append(("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])))
        
        # Broadcast hash constants
        for hi, (c1, c3) in enumerate(hash_consts):
            hv = hash_const_vecs[hi]
            pre_body.append(("valu", ("vbroadcast", hv['c1_vec'], c1)))
            pre_body.append(("valu", ("vbroadcast", hv['c3_vec'], c3)))
        
        self.instrs.extend(self.build(pre_body))

        #  Allocate address registers for loads/stores (computed once, reused across rounds)
        addr_indices_load = []
        addr_values_load = []
        addr_indices_store = []
        addr_values_store = []
        for vec_idx in range(n_vec_groups):
            addr_indices_load.append(self.alloc_scratch(f"addr_idx_ld{vec_idx}"))
            addr_values_load.append(self.alloc_scratch(f"addr_val_ld{vec_idx}"))
            addr_indices_store.append(self.alloc_scratch(f"addr_idx_st{vec_idx}"))
            addr_values_store.append(self.alloc_scratch(f"addr_val_st{vec_idx}"))
        
        # Pre-compute address vectors for all vector groups (computed once!)
        addr_body = []
        for vec_idx in range(n_vec_groups):
            addr_body.append(("alu", ("+", addr_indices_load[vec_idx], self.scratch["inp_indices_p"], offset_consts[vec_idx])))
            addr_body.append(("alu", ("+", addr_values_load[vec_idx], self.scratch["inp_values_p"], offset_consts[vec_idx])))
            addr_body.append(("alu", ("+", addr_indices_store[vec_idx], self.scratch["inp_indices_p"], offset_consts[vec_idx])))
            addr_body.append(("alu", ("+", addr_values_store[vec_idx], self.scratch["inp_values_p"], offset_consts[vec_idx])))
        
        self.instrs.extend(self.build(addr_body))

        body = []

        for round in range(rounds):
            for vec_base in range(0, n_vec_groups, UNROLL_FACTOR):
                n = min(UNROLL_FACTOR, n_vec_groups - vec_base)
                
                # PHASE 1: Vector load indices using pre-computed addresses
                for u in range(n):
                    vr = vregs[u]
                    body.append(("load", ("vload", vr['idx_vec'], addr_indices_load[vec_base + u])))
                
                # PHASE 2: Vector load values using pre-computed addresses
                for u in range(n):
                    vr = vregs[u]
                    body.append(("load", ("vload", vr['val_vec'], addr_values_load[vec_base + u])))
                
                # PHASE 3: Scalar indexed loads of node_val (can't vectorize)
                # Calculate addresses directly from idx_vec
                for u in range(n):
                    vr = vregs[u]
                    for vi in range(VLEN):
                        sr = scalar_regs[u * VLEN + vi]
                        body.append(("alu", ("+", sr['addr'], self.scratch["forest_values_p"], vr['idx_vec'] + vi)))
                
                # Load node values directly into t1_vec
                for u in range(n):
                    vr = vregs[u]
                    for vi in range(VLEN):
                        sr = scalar_regs[u * VLEN + vi]
                        body.append(("load", ("load", vr['t1_vec'] + vi, sr['addr'])))
                
                # PHASE 4: Vector XOR
                for u in range(n):
                    vr = vregs[u]
                    body.append(("valu", ("^", vr['val_vec'], vr['val_vec'], vr['t1_vec'])))
                
                # PHASE 5: Vector hash using pre-broadcast constants
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    hv = hash_const_vecs[hi]
                    
                    for u in range(n):
                        vr = vregs[u]
                        body.append(("valu", (op1, vr['t1_vec'], vr['val_vec'], hv['c1_vec'])))
                    
                    for u in range(n):
                        vr = vregs[u]
                        body.append(("valu", (op3, vr['t2_vec'], vr['val_vec'], hv['c3_vec'])))
                    
                    for u in range(n):
                        vr = vregs[u]
                        body.append(("valu", (op2, vr['val_vec'], vr['t1_vec'], vr['t2_vec'])))
                
                # PHASE 6: Update indices using pre-broadcast constants
                for u in range(n):
                    vr = vregs[u]
                    body.append(("valu", ("%", vr['t2_vec'], vr['val_vec'], two_vec)))
                
                for u in range(n):
                    vr = vregs[u]
                    body.append(("valu", ("==", vr['t2_vec'], vr['t2_vec'], zero_vec)))
                
                for u in range(n):
                    vr = vregs[u]
                    body.append(("valu", ("-", vr['t3_vec'], two_vec, vr['t2_vec'])))
                
                for u in range(n):
                    vr = vregs[u]
                    body.append(("valu", ("*", vr['idx_vec'], vr['idx_vec'], two_vec)))
                
                for u in range(n):
                    vr = vregs[u]
                    body.append(("valu", ("+", vr['idx_vec'], vr['idx_vec'], vr['t3_vec'])))
                
                # PHASE 7: Wrap indices using pre-broadcast constant
                for u in range(n):
                    vr = vregs[u]
                    body.append(("valu", ("<", vr['t1_vec'], vr['idx_vec'], n_nodes_vec)))
                
                for u in range(n):
                    vr = vregs[u]
                    body.append(("valu", ("*", vr['idx_vec'], vr['idx_vec'], vr['t1_vec'])))
                
                # PHASE 8: Vector store results using pre-computed addresses
                for u in range(n):
                    vr = vregs[u]
                    body.append(("store", ("vstore", addr_indices_store[vec_base + u], vr['idx_vec'])))
                
                for u in range(n):
                    vr = vregs[u]
                    body.append(("store", ("vstore", addr_values_store[vec_base + u], vr['val_vec'])))

        # Build and apply bubble fill optimization
        bundles = self.build(body)
        bundles = self._local_bubble_fill(bundles)
        self.instrs.extend(bundles)
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
