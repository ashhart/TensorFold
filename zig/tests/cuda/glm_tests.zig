//! GLM-5.3-Flash on CUDA: each kernel family against float64 and the Python engine's bytes, then whole layers.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");

pub const Rig = quant_ref_tests.RigT;
pub const Run = exl3_tests.Run;

/// Every family's checks, in the order a decode token meets them; `dir` none prints SKIP for fixture checks.
pub fn run(gpu: check.Gpu, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(17);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var state = Run{};
    try @import("glm_glue_tests.zig").run(&r, &state, dir);
    try @import("glm_kda_tests.zig").run(&r, &state, dir);
    try @import("glm_sparse_tests.zig").run(&r, &state, dir);
    try @import("glm_latent_tests.zig").run(&r, &state, dir);
    try @import("glm_layer_tests.zig").run(&r, &state, dir);
    try @import("glm_mlp_tests.zig").run(&r, &state, dir);
    try @import("glm_dsa_block_tests.zig").run(&r, &state, dir);
    try @import("glm_layer_dsa_tests.zig").run(&r, &state, dir);
    try @import("glm_head_tests.zig").run(&r, &state, dir);
    if (state.failed) return error.TestFailed;
}
