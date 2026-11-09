//! Ling-3.1-flash on CUDA: its config as the GLM-5.3-Flash kernels it reuses take it (routing groups, SwiGLU limits).
const std = @import("std");
const glm = @import("glm_cuda");
pub const config = @import("config.zig");
pub const names = @import("names.zig");
pub const weights = @import("weights.zig");
pub const synth = @import("synth.zig");
pub const graph = @import("graph.zig");
pub const ops = @import("ops.zig");
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const engine = @import("engine.zig");
pub const lanes = @import("lanes.zig");
pub const native = @import("native.zig");
pub const tp = @import("tp.zig");
pub const fake_driver = @import("fake_driver.zig");

const Config = config.Config;

/// Layer i's SwiGLU clamps, 0 for none: experts.Layer.limit, the shared expert's, mlp's.
pub const Limits = config.Limits;

/// The lists cover the backbone; the dense layers run unclamped (parse refuses a dense limit).
pub fn limits(c: *const Config, i: u32) Limits {
    return c.layerLimits(i);
}

test {
    _ = @import("config_test.zig");
    _ = @import("tp.zig");
    _ = @import("state.zig");
    _ = @import("engine.zig");
    _ = @import("weights_cut.zig");
    _ = @import("weights_bias_test.zig");
    _ = @import("names_test.zig");
    _ = @import("synth.zig");
    _ = ops;
    std.testing.refAllDecls(weights); // the loader compiles on hosts without a GPU
    std.testing.refAllDecls(@import("weights_proj.zig"));
    std.testing.refAllDecls(forward); // the layer graph compiles on hosts without a GPU
    std.testing.refAllDecls(forward.Forward);
    std.testing.refAllDecls(engine.Engine); // the engine, its lane backend and its server hooks too
    std.testing.refAllDecls(lanes.Lanes);
    _ = native;
    _ = @import("fake_driver.zig");
    _ = lanes;
}

test "config's fixed widths are the kernels' own: the latent cache, the rope" {
    const f = config.fixed;
    const cu = @import("cuda");
    try cu.mla.geometry(f.kv_lora, f.nope, f.v_dim, f.rope);
    try std.testing.expect(f.rope == ops.rope_width and f.kv_lora + f.rope == state.cache_width);
    try std.testing.expect(f.block == cu.fp8.block);
}

test "KDA widths: one check, cuda.kda.geometry at the open; head_dim and taps it does not take refused with the reason" {
    const a = std.testing.allocator;
    var why: config.Why = .{};
    const json = try synth.configJson(a, .bf16, 2);
    defer a.free(json);
    try weights.kdaGeometry(&try config.parse(a, json, &why), &why);
    const cases = [_][3][]const u8{ .{ "\"head_dim\": 128", "\"head_dim\": 64", "head_dim 64" }, .{ "\"head_dim\": 128", "\"head_dim\": 256", "128-wide heads" }, .{ "\"short_conv_kernel_size\": 4", "\"short_conv_kernel_size\": 3", "short_conv_kernel_size 3" }, .{ "\"short_conv_kernel_size\": 4", "\"short_conv_kernel_size\": 5", "4 taps" } };
    for (cases) |x| {
        const edited = try std.mem.replaceOwned(u8, a, json, x[0], x[1]);
        defer a.free(edited);
        try std.testing.expect(!std.mem.eql(u8, edited, json));
        why = .{};
        try std.testing.expectError(error.KdaGeometry, weights.kdaGeometry(&try config.parse(a, edited, &why), &why));
        try std.testing.expect(std.mem.indexOf(u8, why.text(), x[2]) != null);
    }
}

test "Ling's routing as ops.select takes it: 8 groups keeping 4, and select's own bounds hold" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/config.json"), &why);
    const g = c.groups();
    try std.testing.expectEqual(config.Groups{ .n = 8, .keep = 4 }, g);
    try std.testing.expect(c.experts % g.n == 0 and g.keep <= g.n and c.topk <= g.keep * (c.experts / g.n) and g.n <= 32);
}

test "per-layer limits on the MoE layers: routed 10 from 65, shared 10 from 64, none on dense or MTP layers" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/config.json"), &why);
    for (0..c.layers + c.mtp) |i| {
        const l = limits(&c, @intCast(i));
        const moe = i >= 4 and i < 72;
        try std.testing.expectEqual(@as(f32, if (moe and i >= 65) 10 else 0), l.routed);
        try std.testing.expectEqual(@as(f32, if (moe and i >= 64) 10 else 0), l.shared);
        try std.testing.expectEqual(@as(f32, 0), l.dense);
    }
}
