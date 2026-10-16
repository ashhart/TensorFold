//! Ling-3.1-flash on CUDA: its config as the GLM-5.3-Flash kernels it reuses take it (routing groups, SwiGLU limits).
const std = @import("std");
const glm = @import("glm_cuda");
pub const config = @import("config.zig");
pub const names = @import("names.zig");
pub const weights = @import("weights.zig");
pub const synth = @import("synth.zig");
pub const graph = @import("graph.zig");
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const mtp = @import("mtp.zig");
pub const engine = @import("engine.zig");
pub const lanes = @import("lanes.zig");
pub const native = @import("native.zig");
pub const fake_driver = @import("fake_driver.zig");

const Config = config.Config;

/// glue.select's group-limited routing: n_group groups, the best topk_group kept (parse holds select's bounds).
pub fn groups(c: *const Config) glm.glue.Groups {
    return .{ .n = c.n_group, .keep = c.topk_group };
}

/// Layer i's SwiGLU clamps, 0 for none: experts.Layer.limit, moe.Layer.shared_limit, mlp.Layer.limit.
pub const Limits = struct { routed: f32, shared: f32, dense: f32 };

/// The lists cover the backbone; the MTP layer and the dense layers run unclamped (parse refuses a dense limit).
pub fn limits(c: *const Config, i: u32) Limits {
    if (i >= c.layers or !c.isMoe(i)) return .{ .routed = 0, .shared = 0, .dense = 0 };
    return .{ .routed = c.expert_limit[i], .shared = c.shared_limit[i], .dense = 0 };
}

/// The MoE layer's routing fields as moe.Layer holds them (pointers and widths come from the weight loader).
pub fn route(c: *const Config, i: u32, l: *glm.moe.Layer) void {
    const lim = limits(c, i);
    l.top_k = c.topk;
    l.scale = c.routed_scale;
    l.norm = c.norm_topk;
    l.groups = groups(c);
    l.shared_limit = lim.shared;
    l.routed.limit = lim.routed;
}

test {
    _ = @import("config_test.zig");
    _ = @import("names_test.zig");
    _ = @import("synth.zig");
    std.testing.refAllDecls(weights); // the loader compiles on hosts without a GPU
    std.testing.refAllDecls(@import("weights_proj.zig"));
    std.testing.refAllDecls(forward); // the layer graph compiles on hosts without a GPU
    std.testing.refAllDecls(forward.Forward);
    std.testing.refAllDecls(engine.Engine); // the engine, its lane backend and its server hooks too
    std.testing.refAllDecls(lanes.Lanes);
    std.testing.refAllDecls(mtp.Head);
    _ = native;
    _ = @import("fake_driver.zig");
    _ = @import("store_test.zig");
    _ = lanes;
}

test "config's fixed widths are the kernels' own: KDA heads, the latent cache, the rope" {
    const f = config.fixed;
    try std.testing.expect(f.kda_dim == glm.kda.dk and f.kda_dim == glm.kda.dv);
    try std.testing.expect(f.rope == glm.latent.rope_width and f.kv_lora + f.rope == state.cache_width);
    try std.testing.expect(f.block == @import("cuda").fp8.block);
}

test "Ling's routing as glue.select takes it: 8 groups keeping 4, and select's own bounds hold" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/config.json"), &why);
    const g = groups(&c);
    try std.testing.expectEqual(glm.glue.Groups{ .n = 8, .keep = 4 }, g);
    try std.testing.expect(c.experts % g.n == 0 and g.keep <= g.n and c.topk <= g.keep * (c.experts / g.n) and g.n <= 32);
    try std.testing.expect(!std.meta.eql(g, glm.glue.Groups{})); // GLM's one group would skip Ling's group pass
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
    var l: glm.moe.Layer = undefined;
    route(&c, 64, &l);
    try std.testing.expectEqual(@as(f32, 0), l.routed.limit);
    try std.testing.expectEqual(@as(f32, 10), l.shared_limit);
    route(&c, 65, &l);
    try std.testing.expectEqual(@as(f32, 10), l.routed.limit);
    try std.testing.expectEqual(glm.glue.Groups{ .n = 8, .keep = 4 }, l.groups);
    try std.testing.expectEqual(@as(usize, 8), l.top_k);
    try std.testing.expectEqual(@as(f32, 2.5), l.scale);
    try std.testing.expect(l.norm);
}
