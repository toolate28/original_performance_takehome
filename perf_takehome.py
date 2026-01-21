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
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                # vbroadcast: (vbroadcast, dest, src)
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
            else:
                # valu ops: (op, dest, src1, src2)
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
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
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
        elif engine == "store":
            if slot[0] == "store":
                # store: (store, addr, src)
                reads.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vstore":
                # vstore: (vstore, addr, src)
                reads.add(slot[1])
                for i in range(VLEN):
                    reads.add(slot[2] + i)
        elif engine == "flow":
            if slot[0] == "select":
                # select: (select, dest, cond, a, b)
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
            elif slot[0] == "vselect":
                # vselect: (vselect, dest, cond, a, b)
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
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
        engine_counts = defaultdict(int)
        
        for engine, slot in slots:
            reads, writes = self._analyze_dependencies(engine, slot)
            
            # Check for RAW hazard
            has_raw_hazard = bool(reads & current_writes)
            
            # Check if we've exceeded slot limits
            would_exceed_limit = engine_counts[engine] >= SLOT_LIMITS.get(engine, 1)
            
            # Start new bundle if hazard detected or slot limit reached
            if has_raw_hazard or would_exceed_limit:
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {}
                current_writes = set()
                engine_counts = defaultdict(int)
            
            # Add instruction to current bundle
            if engine not in current_bundle:
                current_bundle[engine] = []
            current_bundle[engine].append(slot)
            engine_counts[engine] += 1
            
            # Update dependency tracking
            current_writes.update(writes)
        
        # Add final bundle
        if current_bundle:
            instrs.append(current_bundle)
        
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})
    
    def add_bundle(self, bundle: dict[Engine, list[tuple]]):
        """Add a pre-constructed bundle directly to instructions."""
        self.instrs.append(bundle)

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
        SIMD-optimized kernel using vload/vstore/valu to process 8 elements at once.
        Unrolled across multiple vector batches for maximum ILP.
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

        # Pre-allocate all constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        
        # Pre-allocate all hash constants
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.scratch_const(val1)
            self.scratch_const(val3)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting SIMD loop"))

        body = []  # array of slots

        # SIMD vectorization - process VLEN elements at once
        assert batch_size % VLEN == 0, f"batch_size {batch_size} must be divisible by VLEN {VLEN}"
        num_vec_batches = batch_size // VLEN
        
        # Unroll across multiple vector batches to maximize parallelism
        # With separate addr_gather per batch (73 regs/batch): max ~18 batches
        # Each unrolled batch allocates:
        #   - 8 VLEN-wide vectors
        #   - 1 scalar addr_base
        #   - VLEN scalar addr_gather entries
        # So per-batch scratch usage in "words" is:
        per_batch_scratch_words = 9 * VLEN + 1
        remaining_scratch = SCRATCH_SIZE - self.scratch_ptr
        # Ensure we do not overrun scratch space when choosing VEC_UNROLL.
        # We require at least one batch; cap by what fits in remaining scratch.
        max_batches_by_scratch = max(1, remaining_scratch // per_batch_scratch_words)
        VEC_UNROLL = min(18, num_vec_batches, max_batches_by_scratch)
        
        # Allocate registers for VEC_UNROLL batches
        vec_regs = []
        for u in range(VEC_UNROLL):
            regs = {
                'vec_idx': self.alloc_scratch(f"vec_idx_{u}", VLEN),
                'vec_val': self.alloc_scratch(f"vec_val_{u}", VLEN),
                'vec_node_val': self.alloc_scratch(f"vec_node_val_{u}", VLEN),
                'vec_tmp1': self.alloc_scratch(f"vec_tmp1_{u}", VLEN),
                'vec_tmp2': self.alloc_scratch(f"vec_tmp2_{u}", VLEN),
                'vec_tmp3': self.alloc_scratch(f"vec_tmp3_{u}", VLEN),
                'vec_hash_tmp1': self.alloc_scratch(f"vec_hash_tmp1_{u}", VLEN),
                'vec_hash_tmp2': self.alloc_scratch(f"vec_hash_tmp2_{u}", VLEN),
                'addr_base': self.alloc_scratch(f"addr_base_{u}"),
                # Each batch gets its own gather addresses for parallel calculation
                'addr_gather': [self.alloc_scratch(f"addr_gather_{u}_{i}") for i in range(VLEN)],
            }
            vec_regs.append(regs)
        
        # Pre-broadcast all hash constants (one time setup)
        hash_val1_vecs = []
        hash_val3_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val1_vec = self.alloc_scratch(f"hash_val1_vec_{hi}", VLEN)
            val3_vec = self.alloc_scratch(f"hash_val3_vec_{hi}", VLEN)
            body.append(("valu", ("vbroadcast", val1_vec, self.scratch_const(val1))))
            body.append(("valu", ("vbroadcast", val3_vec, self.scratch_const(val3))))
            hash_val1_vecs.append(val1_vec)
            hash_val3_vecs.append(val3_vec)
        
        # Broadcast other constants once (one-time setup)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        body.append(("valu", ("vbroadcast", two_vec, two_const)))
        body.append(("valu", ("vbroadcast", zero_vec, zero_const)))
        body.append(("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])))

        # Optimized: exploit early round locality
        # Round 0: all at idx=0, Round 1: at idx 1 or 2, etc.
        # Use conditional select for early rounds instead of gathers
        MAX_EARLY_ROUND = min(7, rounds)  # Up to 128 unique indices
        
        for round in range(rounds):
            # Early rounds: few unique indices - use broadcast/select
            if round < MAX_EARLY_ROUND:
                max_idx = min(2**(round+1) - 1, n_nodes - 1)
                
                # Pre-load all possible node values for this round
                node_vals = []
                for idx in range(max_idx + 1):
                    nv = self.alloc_scratch(f"node_val_r{round}_i{idx}")
                    body.append(("alu", ("+", tmp1, self.scratch["forest_values_p"], self.scratch_const(idx))))
                    body.append(("load", ("load", nv, tmp1)))
                    node_vals.append(nv)
                
                # Broadcast node values to vectors
                node_val_vecs = []
                for idx in range(max_idx + 1):
                    nvv = self.alloc_scratch(f"node_val_vec_r{round}_i{idx}", VLEN)
                    body.append(("valu", ("vbroadcast", nvv, node_vals[idx])))
                    node_val_vecs.append(nvv)
            
            # Process in groups of VEC_UNROLL vector batches
            for vi_base in range(0, num_vec_batches, VEC_UNROLL):
                num_batches = min(VEC_UNROLL, num_vec_batches - vi_base)
                
                # === LOAD PHASE (interleaved) ===
                # Load all indices vectors
                for u in range(num_batches):
                    i_base = (vi_base + u) * VLEN
                    regs = vec_regs[u]
                    body.append(("alu", ("+", regs['addr_base'], self.scratch["inp_indices_p"], self.scratch_const(i_base))))
                    body.append(("load", ("vload", regs['vec_idx'], regs['addr_base'])))
                
                # Load all values vectors
                for u in range(num_batches):
                    i_base = (vi_base + u) * VLEN
                    regs = vec_regs[u]
                    body.append(("alu", ("+", regs['addr_base'], self.scratch["inp_values_p"], self.scratch_const(i_base))))
                    body.append(("load", ("vload", regs['vec_val'], regs['addr_base'])))
                
                # CRITICAL OPTIMIZATION: Batch gather address calculations
                # ALU limit = 12/cycle (not 2!), Load limit = 2/cycle
                # Calculate ALL addresses first (12 at a time), then do ALL loads (2 at a time)
                
                # Early rounds: use conditional select instead of gather
                if round < MAX_EARLY_ROUND:
                    max_idx = min(2**(round+1) - 1, n_nodes - 1)
                    for u in range(num_batches):
                        regs = vec_regs[u]
                        # Start with first node value
                        body.append(("valu", ("vbroadcast", regs['vec_node_val'], node_vals[0])))
                        # Conditionally select based on index
                        for idx in range(1, max_idx + 1):
                            # Create mask: vec_idx == idx
                            idx_vec = self.alloc_scratch(f"idx_vec_{round}_{idx}", VLEN)
                            mask_vec = self.alloc_scratch(f"mask_vec_{round}_{idx}", VLEN)
                            body.append(("valu", ("vbroadcast", idx_vec, self.scratch_const(idx))))
                            body.append(("valu", ("==", mask_vec, regs['vec_idx'], idx_vec)))
                            # Select: node_val = mask ? node_val_vecs[idx] : node_val
                            body.append(("flow", ("vselect", regs['vec_node_val'], mask_vec, node_val_vecs[idx], regs['vec_node_val'])))
                else:
                    # Late rounds: use gather (fallback to original)
                    # Phase 1: Calculate ALL gather addresses for ALL batches
                    # This will pack at 12 ALU ops/cycle instead of 1!
                    for lane in range(VLEN):
                        for u in range(num_batches):
                            regs = vec_regs[u]
                            body.append(("alu", ("+", regs['addr_gather'][lane], self.scratch["forest_values_p"], regs['vec_idx'] + lane)))
                    
                    # Phase 2: Perform ALL gather loads (2/cycle)
                    for lane in range(VLEN):
                        for u in range(num_batches):
                            regs = vec_regs[u]
                            body.append(("load", ("load", regs['vec_node_val'] + lane, regs['addr_gather'][lane])))
                
                # === COMPUTE PHASE (interleaved) ===
                # XOR all batches
                for u in range(num_batches):
                    regs = vec_regs[u]
                    body.append(("valu", ("^", regs['vec_val'], regs['vec_val'], regs['vec_node_val'])))
                
                # Hash operations - use pre-broadcast constants
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    # Apply to all batches using pre-broadcast constants
                    for u in range(num_batches):
                        regs = vec_regs[u]
                        body.append(("valu", (op1, regs['vec_hash_tmp1'], regs['vec_val'], hash_val1_vecs[hi])))
                    
                    for u in range(num_batches):
                        regs = vec_regs[u]
                        body.append(("valu", (op3, regs['vec_hash_tmp2'], regs['vec_val'], hash_val3_vecs[hi])))
                    
                    for u in range(num_batches):
                        regs = vec_regs[u]
                        body.append(("valu", (op2, regs['vec_val'], regs['vec_hash_tmp1'], regs['vec_hash_tmp2'])))
                
                # Index update calculations (interleaved by operation type)
                for u in range(num_batches):
                    regs = vec_regs[u]
                    body.append(("valu", ("%", regs['vec_tmp1'], regs['vec_val'], two_vec)))
                
                for u in range(num_batches):
                    regs = vec_regs[u]
                    body.append(("valu", ("==", regs['vec_tmp1'], regs['vec_tmp1'], zero_vec)))
                
                for u in range(num_batches):
                    regs = vec_regs[u]
                    body.append(("valu", ("-", regs['vec_tmp3'], two_vec, regs['vec_tmp1'])))
                
                for u in range(num_batches):
                    regs = vec_regs[u]
                    body.append(("valu", ("*", regs['vec_idx'], regs['vec_idx'], two_vec)))
                
                for u in range(num_batches):
                    regs = vec_regs[u]
                    body.append(("valu", ("+", regs['vec_idx'], regs['vec_idx'], regs['vec_tmp3'])))
                
                # Wrap indices
                for u in range(num_batches):
                    regs = vec_regs[u]
                    body.append(("valu", ("<", regs['vec_tmp1'], regs['vec_idx'], n_nodes_vec)))
                
                for u in range(num_batches):
                    regs = vec_regs[u]
                    body.append(("flow", ("vselect", regs['vec_idx'], regs['vec_tmp1'], regs['vec_idx'], zero_vec)))
                
                # === STORE PHASE (interleaved) ===
                # Store all indices
                for u in range(num_batches):
                    i_base = (vi_base + u) * VLEN
                    regs = vec_regs[u]
                    body.append(("alu", ("+", regs['addr_base'], self.scratch["inp_indices_p"], self.scratch_const(i_base))))
                    body.append(("store", ("vstore", regs['addr_base'], regs['vec_idx'])))
                
                # Store all values
                for u in range(num_batches):
                    i_base = (vi_base + u) * VLEN
                    regs = vec_regs[u]
                    body.append(("alu", ("+", regs['addr_base'], self.scratch["inp_values_p"], self.scratch_const(i_base))))
                    body.append(("store", ("vstore", regs['addr_base'], regs['vec_val'])))

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
