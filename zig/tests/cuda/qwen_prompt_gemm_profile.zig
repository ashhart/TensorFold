//! GB10 opt-in NCU targets: the exact Qwen3.0 27B affine-g64 prompt GEMMs, with served-path output identity.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const many = @import("contract_qmm_prefill_many.zig");

const Shape = struct {
    n: u64,
    k: u64,
    half: bool,
    grid_y: u64,
    threads: u32,
    shared: u32,
    tile: []const u8,
};

fn shape(text: []const u8) !Shape {
    const shapes = [_]Shape{
        .{ .n = 5120, .k = 17408, .half = false, .grid_y = 32, .threads = 256, .shared = 62976, .tile = "2x4x3" },
        .{ .n = 5120, .k = 6144, .half = false, .grid_y = 32, .threads = 256, .shared = 62976, .tile = "2x4x3" },
        .{ .n = 17408, .k = 5120, .half = true, .grid_y = 136, .threads = 128, .shared = 41984, .tile = "2x2x2" },
    };
    var buf: [32]u8 = undefined;
    for (shapes) |s| {
        const name = try std.fmt.bufPrint(&buf, "{d}x{d}", .{ s.n, s.k });
        if (std.mem.eql(u8, text, name)) return s;
    }
    return error.UnknownPromptGemmShape;
}

/// The target launches first. NCU filters to it; the ordinary-tile reference follows.
pub fn run(gpu: check.Gpu, text: []const u8) !void {
    const selected = try shape(text);
    const cap = try gpu.ctx.capability();
    try check.expect(cap == 121, "cuda-prompt-gemm-profile requires GB10 sm_121; found sm_{d}", .{cap});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mods: [4]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ cuda.kernels.qmm_group, cuda.kernels.qmm_prefill, cuda.kernels.lane_gemv, cuda.kernels.affine4_pack }) |image| {
        mods[loaded] = try cuda.Module.load(gpu.d, image);
        loaded += 1;
    }
    var affine = try cuda.qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], 0, true);
    defer affine.deinit();

    var product = many.Product{ .rig = .{ .gpu = gpu, .s = s }, .affine = &affine };
    product.member_widths = &.{selected.n};
    defer product.deinit();
    const fixture = core.contract.spec.Shape{ .rows = 4096, .n = selected.n, .k = selected.k };
    try check.expect(try product.prepare(fixture, .random, 0x120120), "{s}: fixture preparation rejected the served shape", .{text});

    const q = product.qs[0];
    const out = product.outs[0];
    const bytes = product.bytes;
    if (selected.half) {
        try affine.promptHalf(s, product.x, q, out, 4096);
    } else {
        try affine.prompt(s, product.x, q, out, 4096);
    }
    try s.synchronize();
    const got = try gpu.gpa.dupe(u8, try product.rig.outputBytes());
    defer gpu.gpa.free(got);

    // The ordinary prefill is the served path's exact reference for both tiles.
    affine.half_prompt = false;
    try affine.prompt(s, product.x, q, product.base, 4096);
    const want = try product.rig.bytes(product.base, bytes);
    try check.sameBytes(text, got, want);
    try check.expect(try product.rig.intact(), "{s}: a fixture guard changed", .{text});
    try check.expect(!product.rig.overlap(), "{s}: input and output storage overlap", .{text});

    const group = @as(u64, 12 * 1024 * 1024) / (128 * selected.k * 2);
    const flat_grid = 32 * ((selected.n + 127) / 128);
    check.pass(
        "cuda-prompt-gemm-profile M=4096 N={d} K={d} tile={s}: {d} output bytes == ordinary prefill; " ++
            "grid=({d},1,1) group={d} threads={d} shared={d}; seed=0x120120",
        .{ selected.n, selected.k, selected.tile, bytes, flat_grid, group, selected.threads, selected.shared },
    );
}
