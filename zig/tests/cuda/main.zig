//! GPU test runner for the Zig CUDA runtime: `tf-cuda-test <command> [args]`, PASS/FAIL/RESULT lines, exit 1 on failure.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const runtime_tests = @import("runtime_tests.zig");
const bench = @import("bench.zig");
const oracle_tests = @import("oracle_tests.zig");
const libs_tests = @import("libs_tests.zig");
const sample_tests = @import("sample_tests.zig");
const glue_tests = @import("glue_tests.zig");
const window_profile = @import("window_profile.zig");
const tree_accept = @import("tree_accept.zig");
const chunk_costs = @import("chunk_costs.zig");
const grouped_tests = @import("grouped_tests.zig");
const fp8_tests = @import("fp8_tests.zig");
const fp8_experts_tests = @import("fp8_experts_tests.zig");
const nvfp4_experts_tests = @import("nvfp4_experts_tests.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const carveout_tests = @import("carveout_tests.zig");
const slide_grad = @import("slide_grad.zig");
const slide_bench = @import("slide_bench.zig");
const slide_decode = @import("slide_decode.zig");
const train_mixer_tests = @import("train_mixer_tests.zig");
const train_tests = @import("train_tests.zig");
const nvfp4_tests = @import("nvfp4_tests.zig");

const usage =
    \\usage: tf-cuda-test <command>
    \\  info                      driver, device and library versions
    \\  smoke                     copies, fills, launches, argument packing, module globals
    \\  graph                     stream capture, explicit graphs, node and whole-exec updates
    \\  overhead [n] [reps]       dependent one-thread kernels: plain stream vs one graph
    \\  overhead-pdl [n] [reps]   the same with programmatic dependent launch on every kernel
    \\  launch-ex                 cuLaunchKernelEx: clusters, cooperative grid, PDL on a stream and in a graph
    \\  ptx                       hand-written PTX through the driver JIT, and a refused broken image
    \\  symbols                   every gdn fatbin instantiation resolves by its listed symbol
    \\  cublaslt                  bf16 GEMM through cuBLASLt against an fp64 reference
    \\  nccl                      one-rank NCCL all-reduce and all-gather
    \\  gdn-replay <dir>          replay_kernel bits against the Python oracle's fixture
    \\  gdn-tree <dir>            tree_kernel bits against the Python oracle's fixture
    \\  triton <dir>              a Triton cubin's bits against the Python oracle's fixture
    \\  sample                    sample.cu's keyed draws against the Metal rule's host references (synthetic rows)
    \\  glue                      Nemotron's glue kernels against their host references at its shapes (synthetic rows)
    \\  window-profile MODEL WIDTHS IDS_FILE   a decode window's GPU ms by kernel class (WIDTHS like 1,4,16), beside its graph
    \\  tree-accept MODEL IDS_FILE COUNT OUT   a greedy reply's MTP chain and level-1 runner-up branches at every position (JSON)
    \\  chunk-costs MODEL IDS_FILE WIDTHS   r rows as the decode window graph (r <= 16) and as a prompt chunk (any r)
    \\  grouped-plan <dir>        the shared expert plan against experts.route's bytes (oracle/grouped_plan.py)
    \\  fp8-lane <dir>            block-FP8 projections against the Python lane matmul's bytes (oracle/fp8_lane.py)
    \\  fp8-experts <dir>         grouped block-FP8 experts against recorded fixture bytes
    \\  nvfp4-experts <dir>       grouped NVFP4 experts against recorded fixture bytes
    \\  quant-ref                 FP8 and NVFP4 projections and experts against float64 references (synthetic)
    \\  carveout [MiB] [card]     GB10 display memory: round trips and bandwidth (SKIP without the card)
    \\  train                     Sliding Weights' training kernels against f64 references (synthetic rows)
    \\  train-mixers              Sliding Weights' mixer gradients against f64 references (synthetic rows)
    \\  slide-decode MODEL IDS_FILE        a decode token's milliseconds without --slide and with 0 to 512 ranks
    \\  slide-kernels             the change's window kernels at 512 ranks, a and b in mapped host or device memory
    \\  slide-bench MODEL IDS_FILE START   Sliding Weights' step milliseconds in each mode
    \\  slide-grad MODEL IDS_FILE START [REACH [STARTS]]   Sliding Weights' gradient against a cubic fit of the loss
    \\  nvfp4 <dir>               NVFP4 projections against recorded lane-matmul and prompt-GEMM fixture bytes
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
    if (std.mem.eql(u8, cmd, "gdn-replay")) return oracle_tests.gdnReplay(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "gdn-tree")) return oracle_tests.gdnTree(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "triton")) return oracle_tests.tritonKernel(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "fp8-lane")) return fp8_tests.lane(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "fp8-experts")) return fp8_experts_tests.experts(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "nvfp4")) return nvfp4_tests.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "nvfp4-experts")) return nvfp4_experts_tests.experts(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "quant-ref")) return quant_ref_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "sample")) return sample_tests.draws(gpu);
    if (std.mem.eql(u8, cmd, "glue")) return glue_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "window-profile")) return window_profile.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "tree-accept")) return tree_accept.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "chunk-costs")) return chunk_costs.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "grouped-plan")) return grouped_tests.plan(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "carveout")) {
        const mib = if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else cuda.carveout.default_bytes >> 20;
        return carveout_tests.run(gpu, if (rest.len > 1) rest[1] else cuda.carveout.default_card, mib);
    }
    if (std.mem.eql(u8, cmd, "train")) return train_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "train-mixers")) return train_mixer_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "slide-decode")) return slide_decode.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "slide-kernels")) return slide_decode.kernels(gpu);
    if (std.mem.eql(u8, cmd, "slide-bench")) return slide_bench.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "slide-grad")) return slide_grad.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), if (rest.len > 3) rest[3] else null, if (rest.len > 4) rest[4] else null);
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
