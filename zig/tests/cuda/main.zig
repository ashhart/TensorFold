//! GPU test runner for the Zig CUDA runtime: `tf-cuda-test <command> [args]`, PASS/FAIL/RESULT lines, exit 1 on failure.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const runtime_tests = @import("runtime_tests.zig");
const bench = @import("bench.zig");
const oracle_tests = @import("oracle_tests.zig");
const libs_tests = @import("libs_tests.zig");
const rank_scatter_tests = @import("rank_scatter_tests.zig");
const roce_gather_tests = @import("roce_gather_tests.zig");
const sample_tests = @import("sample_tests.zig");
const glue_tests = @import("glue_tests.zig");
const window_profile = @import("window_profile.zig");
const tree_accept = @import("tree_accept.zig");
const resume_tests = @import("resume_tests.zig");
const prompt_cuts = @import("prompt_cuts.zig");
const chunk_costs = @import("chunk_costs.zig");
const grouped_tests = @import("grouped_tests.zig");
const fp8_tests = @import("fp8_tests.zig");
const mxfp8_tests = @import("mxfp8_tests.zig");
const projection_f32_tests = @import("projection_f32_tests.zig");
const fp8_experts_tests = @import("fp8_experts_tests.zig");
const nvfp4_experts_tests = @import("nvfp4_experts_tests.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const glm_ref_tests = @import("glm_ref_tests.zig");
const affine8_tests = @import("affine8_tests.zig");
const carveout_tests = @import("carveout_tests.zig");
const nvfp4_tests = @import("nvfp4_tests.zig");
const modelopt_tests = @import("modelopt_tests.zig");
const train_pack = @import("train_pack.zig");
const exlr8_tests = @import("exlr8_tests.zig");
const exlr8_dense_tests = @import("exlr8_dense_tests.zig");
const norms_tests = @import("norms_tests.zig");
const norm_rows_tests = @import("norm_rows_tests.zig");
const pocket_tests = @import("pocket_tests.zig");
const recapture_tests = @import("recapture_tests.zig");
const contract_tests = @import("contract_tests.zig");
const lfm2_attention_tests = @import("lfm2_attention_tests.zig");
const lfm2_kernel_tests = @import("lfm2_kernel_tests.zig");
const lfm2_vision_tests = @import("lfm2_vision_tests.zig");
const radio_tests = @import("radio_tests.zig");
const glm_lanes_tests = @import("glm_lanes_tests.zig");
const glm_synth = @import("glm_synth.zig");
const tree_round_tests = @import("tree_round_tests.zig");
const tree_tests = @import("tree_tests.zig");
const block_tests = @import("block_tests.zig");
const act_rows_tests = @import("act_rows_tests.zig");
const gdn_prep_tests = @import("gdn_prep_tests.zig");
const qk_rows_tests = @import("qk_rows_tests.zig");
const window_tests = @import("window_tests.zig");
const moe_glue_tests = @import("moe_glue_tests.zig");
const dense_rows_tests = @import("dense_rows_tests.zig");

const usage =
    \\usage: tf-cuda-test <command>
    \\  info                      driver, device and library versions
    \\  smoke                     copies, fills, launches, argument packing, module globals
    \\  shared-context            Context.shared and Stream.borrowed as a library inside a host's process sees them
    \\  graph                     stream capture, explicit graphs, node and whole-exec updates
    \\  overhead [n] [reps]       dependent one-thread kernels: plain stream vs one graph
    \\  overhead-pdl [n] [reps]   the same with programmatic dependent launch on every kernel
    \\  launch-ex                 cuLaunchKernelEx: clusters, cooperative grid, PDL on a stream and in a graph
    \\  ptx                       hand-written PTX through the driver JIT, and a refused broken image
    \\  symbols                   every gdn fatbin instantiation resolves by its listed symbol
    \\  cublaslt                  bf16 GEMM through cuBLASLt against an fp64 reference
    \\  nccl                      one-rank NCCL all-reduce and all-gather
    \\  rank-scatter <rank> <n> [id]  rank-order reduce-scatter (NCCL_COMM_ID or an id file) against an all-gather
    \\  roce-gather <rank>        two ranks (NCCL_COMM_ID): the RoCE all-gather against NCCL's, graph replays, time a call
    \\  rank-sum                  the reduce-scatter's sum kernel alone, 2 to 5 ranks as each rank (one GPU, no NCCL)
    \\  nccl-pair <rank> <world> <idfile>  one rank of a multi-host communicator: send/recv with every peer, all-reduce, all-gather
    \\  gdn-replay <dir>          replay_kernel bits against the Python oracle's fixture
    \\  gdn-tree <dir>            tree_kernel bits against the Python oracle's fixture
    \\  triton <dir>              a Triton cubin's bits against the Python oracle's fixture
    \\  norms                     RMS norm and the residual step against float64 references (synthetic)
    \\  norm-rows                 bf16-residual and gated norms, token rows against float64 (synthetic)
    \\  tree-round                tree_round.cu's picks, batched verified trees and commits against core/tree_round
    \\  act-rows                  SwiGLU, output gates, group sums, DFlash2's dynamic conv against float64 (synthetic)
    \\  gdn-prep                  Gated DeltaNet conv, norms, g and beta against float64 (synthetic)
    \\  qk-rows                   QK norm, RoPE and the cache write against float64 references (synthetic)
    \\  sample                    sample.cu's keyed draws against the Metal rule's host references (synthetic rows)
    \\  glue                      Nemotron's glue kernels against their host references at its shapes (synthetic rows)
    \\  lfm2-kernels              LFM2's glue kernels against their host references at d1's shapes (synthetic rows)
    \\  pocket [reps]             the pocket's two kernels against tf_pocket_rows byte for byte; reps > 0 times them too
    \\  lfm2-attention            LFM2's GQA prompt attention (head dim 64), causal and packed trees, against float64 (synthetic rows)
    \\  lfm2-vision [MODEL]       LFM2-VL's vision kernels against host references; MODEL adds features against CPU goldens
    \\  radio REF MODEL           RADIO's tower and projector rows against float32 references on the same pixels
    \\  radio-processor REF PICS MODEL  decoded RGB, processor pixels, float resizes and grids against the reference
    \\  radio-kernels             RADIO's float-patch cast and position rows against host references (synthetic rows)
    \\  radio-attention           the head-dim-80 tower attention's time at five sizes, outputs finite (synthetic rows)
    \\  window-profile MODEL WIDTHS IDS_FILE   a decode window's GPU ms by kernel class (WIDTHS like 1,4,16), beside its graph
    \\  tree-accept MODEL IDS_FILE COUNT OUT   a greedy reply's MTP chain and level-1 runner-up branches at every position (JSON)
    \\  chunk-costs MODEL IDS_FILE WIDTHS   r rows as the decode window graph (r <= 16) and as a prompt chunk (any r)
    \\  resume MODEL IDS_FILE     prompt states kept at off-grid rows: each resumed pass against a fresh one, byte for byte
    \\  marks MODEL IDS_FILE N PATTERNS   states kept inside chunks against chunks cut at the same rows, byte for byte
    \\  prompt-cuts MODEL IDS_FILE   a prompt keeping states inside chunks and at their ends (background passes too), and resumed from each, equals one uncut pass
    \\  recapture MODEL           graphs captured again while another sequence is bound: both keep their sampling rules
    \\  grouped-plan <dir>        the shared expert plan against experts.route's bytes (oracle/grouped_plan.py)
    \\  fp8-lane <dir>            block-FP8 projections against the Python lane matmul's bytes (oracle/fp8_lane.py)
    \\  fp8-experts <dir>         grouped block-FP8 experts against recorded fixture bytes
    \\  nvfp4-experts <dir>       grouped NVFP4 experts against recorded fixture bytes
    \\  projection-f32            FP32 projection outputs against FP64, rounding and row/reduction invariance
    \\  mxfp8                     MXFP8 strided projections against FP64 and row-count invariance
    \\  quant-ref                 FP8 and NVFP4 projections and experts against float64 references (synthetic)
    \\  affine8                   MLX affine 2 to 8-bit tiles and the 8-bit lane_gemv: widened codes, float64, rows (synthetic)
    \\  carveout [MiB] [card]     GB10 display memory: round trips and bandwidth (SKIP without the card)
    \\  nvfp4 <dir>               NVFP4 projections against recorded lane-matmul and prompt-GEMM fixture bytes
    \\  modelopt                  ModelOpt repack, FP8 and BF16 lane rows and relu^2 NVFP4 experts against fp64
    \\  exlr8                     EXLR8 projections: every codebook state, FP64, row windows, the Metal frozen digests
    \\  exlr8-bench DIR [MANIFEST] [LAYER] [SETS]   a real EXLR8 layer: FP64, row windows, then GB/s by stage
    \\  exlr8-dense [DIR]         EXLR8 dense formats: decode bit-equal, FP64, row windows; DIR adds the real matrices
    \\  tree                      tree attention (committed keys and paths) against float64 references (synthetic)
    \\  block                     DFlash block attention over windowed contexts against float64 (synthetic)
    \\  window                    sliding-window attention against float64 references (synthetic)
    \\  moe-glue                  sigmoid top-k routing and the weighted combine against float64 references (synthetic)
    \\  dense-rows                bf16 rows times weight rows (routers, heads) against float64 references (synthetic)
    \\  train-pack MODEL EXAMPLES [COUNT]   training passes holding several sequences against one a pass: differences, time
    \\  kernel-contract           the registered kernels through the contract's gates, each planted control red (needs -Dnvcc)
    \\  kernel-contract-limits DIR   each registered launch against sm_80/86/89/120/121 from the fatbins in DIR
    \\  glm-ref                   GLM projections (matmuls, router, attention, indexer, hc) against float64 (synthetic)
    \\  glm-lanes-cover <ckpt> <dc.json> <rank> <world> <idfile>  copy drafts, cancels, discards, eos and max_new on one lanes.Engine across requests
    \\  glm-lanes-concurrent <ckpt> <dc.json> <rank> <world> <idfile>  concurrent streams on one lanes.Engine, each against its solo run
    \\  glm-shared-check <ckpt> <dc.json> <rank> <world> <idfile>  shared verify windows against lone twins, logits and states byte for byte
    \\  glm-synth <dir>          a synthetic GLM-5.3-Flash checkpoint (2 layers, 16 experts) and its dc.json for the lane checks
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    const cmd = args[1];
    const rest = args[2..];

    run(gpu, cmd, rest) catch |e| {
        std.debug.print("FAIL {s}: {t}\n", .{ cmd, e });
        return 1;
    };
    return 0;
}

fn arg(rest: []const [:0]const u8, i: usize) ![]const u8 {
    if (i >= rest.len) {
        std.debug.print("{s}", .{usage});
        return error.MissingArgument;
    }
    return rest[i];
}

fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "info")) return info(gpu);
    if (std.mem.eql(u8, cmd, "smoke")) return runtime_tests.smoke(gpu);
    if (std.mem.eql(u8, cmd, "shared-context")) return runtime_tests.sharedContext(gpu);
    if (std.mem.eql(u8, cmd, "graph")) return runtime_tests.graphs(gpu);
    if (std.mem.eql(u8, cmd, "overhead") or std.mem.eql(u8, cmd, "overhead-pdl")) {
        const n = if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 1000;
        const reps = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 20;
        return bench.overhead(gpu, n, reps, std.mem.eql(u8, cmd, "overhead-pdl"));
    }
    if (std.mem.eql(u8, cmd, "launch-ex")) return runtime_tests.launchEx(gpu);
    if (std.mem.eql(u8, cmd, "ptx")) return runtime_tests.ptx(gpu);
    if (std.mem.eql(u8, cmd, "symbols")) return runtime_tests.symbols(gpu);
    if (std.mem.eql(u8, cmd, "cublaslt")) return libs_tests.cublaslt(gpu);
    if (std.mem.eql(u8, cmd, "nccl")) return libs_tests.nccl(gpu);
    if (std.mem.eql(u8, cmd, "rank-scatter")) {
        const rank = try std.fmt.parseInt(usize, try arg(rest, 0), 10);
        const world = try std.fmt.parseInt(usize, try arg(rest, 1), 10);
        return rank_scatter_tests.run(gpu, rank, world, if (rest.len > 2) rest[2] else null);
    }
    if (std.mem.eql(u8, cmd, "roce-gather")) return roce_gather_tests.run(gpu, try std.fmt.parseInt(usize, try arg(rest, 0), 10));
    if (std.mem.eql(u8, cmd, "rank-sum")) return rank_scatter_tests.sum(gpu);
    if (std.mem.eql(u8, cmd, "nccl-pair")) return libs_tests.ncclPair(gpu, try std.fmt.parseInt(c_int, try arg(rest, 0), 10), try std.fmt.parseInt(c_int, try arg(rest, 1), 10), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "gdn-replay")) return oracle_tests.gdnReplay(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "gdn-tree")) return oracle_tests.gdnTree(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "triton")) return oracle_tests.tritonKernel(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "fp8-lane")) return fp8_tests.lane(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "fp8-experts")) return fp8_experts_tests.experts(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "nvfp4")) return nvfp4_tests.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "nvfp4-experts")) return nvfp4_experts_tests.experts(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "projection-f32")) return projection_f32_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "mxfp8")) return mxfp8_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "quant-ref")) return quant_ref_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "glm-ref")) return glm_ref_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "affine8")) return affine8_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "modelopt")) return modelopt_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "exlr8")) return exlr8_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "exlr8-dense")) return exlr8_dense_tests.run(gpu, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "exlr8-bench")) return exlr8_tests.bench(gpu, try arg(rest, 0), if (rest.len > 1) rest[1] else "home", if (rest.len > 2) try std.fmt.parseInt(u16, rest[2], 10) else 40, if (rest.len > 3) try std.fmt.parseInt(usize, rest[3], 10) else 4);
    if (std.mem.eql(u8, cmd, "sample")) return sample_tests.draws(gpu);
    if (std.mem.eql(u8, cmd, "norms")) return norms_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "norm-rows")) return norm_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "tree-round")) return tree_round_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "act-rows")) return act_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "gdn-prep")) return gdn_prep_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "qk-rows")) return qk_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "glue")) return glue_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "lfm2-kernels")) return lfm2_kernel_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "pocket"))
        return pocket_tests.run(gpu, if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 0);
    if (std.mem.eql(u8, cmd, "lfm2-attention")) return lfm2_attention_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "lfm2-vision")) return lfm2_vision_tests.run(gpu, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "radio")) return radio_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "radio-processor")) return radio_tests.processor(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "radio-kernels")) return radio_tests.kernels(gpu);
    if (std.mem.eql(u8, cmd, "radio-attention")) return radio_tests.attention(gpu);
    if (std.mem.eql(u8, cmd, "window-profile")) return window_profile.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "tree-accept")) return tree_accept.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "chunk-costs")) return chunk_costs.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "resume")) return resume_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "marks")) return resume_tests.insideChunks(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "prompt-cuts")) return prompt_cuts.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "recapture")) return recapture_tests.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "grouped-plan")) return grouped_tests.plan(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "carveout")) {
        const mib = if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else cuda.carveout.default_bytes >> 20;
        return carveout_tests.run(gpu, if (rest.len > 1) rest[1] else cuda.carveout.default_card, mib);
    }
    if (std.mem.eql(u8, cmd, "kernel-contract")) return contract_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "kernel-contract-limits")) return contract_tests.fatbins(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "tree")) return tree_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "block")) return block_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "window")) return window_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "moe-glue")) return moe_glue_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "dense-rows")) return dense_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "train-pack")) return train_pack.run(gpu, try arg(rest, 0), try arg(rest, 1), if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 24);
    if (std.mem.eql(u8, cmd, "glm-lanes-cover")) return glm_lanes_tests.coverCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "glm-shared-check")) return glm_lanes_tests.sharedCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "glm-synth")) return glm_synth.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "glm-lanes-concurrent")) return glm_lanes_tests.concurrentCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}

fn info(gpu: check.Gpu) !void {
    var name_buf: [256]u8 = undefined;
    const name = try gpu.ctx.name(&name_buf);
    const mem = try gpu.ctx.memInfo();
    std.debug.print("RESULT driver CUDA {d}, device {s}, sm_{d}, {d} SMs, {d} MiB total, {d} MiB free, kernels embedded {}\n", .{
        try gpu.d.version(),      name,
        try gpu.ctx.capability(), try gpu.ctx.attribute(.multiprocessor_count),
        mem.total >> 20,          mem.free >> 20,
        cuda.kernels.available,
    });
}

fn hasFlag(flags: []const [:0]const u8, name: []const u8) bool {
    for (flags) |f| if (std.mem.eql(u8, f, name)) return true;
    return false;
}
