//! The pocket in the shards: the first write widens and tells config.json, own bytes kept, later writes in place.
const std = @import("std");
const cfg = @import("config.zig");
const pocket_shards = @import("pocket_shards.zig");
const core = struct {
    pub const shard_edit = @import("../../core/shard_edit.zig");
    pub const safetensors = @import("../../core/safetensors.zig");
    pub const affine4_host = @import("../../core/affine4_host.zig");
};
const Shards = pocket_shards.Io(core);
const host4 = core.affine4_host;

const d = 64;
const own = 64;
const learned = 64;

test "the first write widens the shared expert and tells config.json; a later one lands in place; own neurons keep every byte" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    // a model with one MoE layer whose shared expert has `own` neurons
    var up_w: [own * d / 8]u32 = undefined;
    var down_w: [d * own / 8]u32 = undefined;
    for (&up_w) |*x| x.* = r.int(u32);
    for (&down_w) |*x| x.* = r.int(u32);
    var up_s: [own * d / 64]u16 = undefined;
    var up_b: [own * d / 64]u16 = undefined;
    var down_s: [d * own / 64]u16 = undefined;
    var down_b: [d * own / 64]u16 = undefined;
    for (&up_s, &up_b, &down_s, &down_b) |*a, *b, *c, *e| {
        a.* = host4.bf16of(0.01 + r.float(f32) * 0.01);
        b.* = host4.bf16of(-0.05);
        c.* = host4.bf16of(0.02);
        e.* = host4.bf16of(-0.1);
    }
    const base = "backbone.layers.0.mixer.shared_experts.";
    const shard = try std.fs.path.join(gpa, &.{ dir, "model-00001-of-00001.safetensors" });
    defer gpa.free(shard);
    try core.shard_edit.create(gpa, shard, &.{.{ "format", "mlx" }}, &.{
        .{ .name = base ++ "up_proj.weight", .dtype = .u32, .shape = &.{ own, d / 8 }, .bytes = std.mem.sliceAsBytes(&up_w) },
        .{ .name = base ++ "up_proj.scales", .dtype = .bf16, .shape = &.{ own, d / 64 }, .bytes = std.mem.sliceAsBytes(&up_s) },
        .{ .name = base ++ "up_proj.biases", .dtype = .bf16, .shape = &.{ own, d / 64 }, .bytes = std.mem.sliceAsBytes(&up_b) },
        .{ .name = base ++ "down_proj.weight", .dtype = .u32, .shape = &.{ d, own / 8 }, .bytes = std.mem.sliceAsBytes(&down_w) },
        .{ .name = base ++ "down_proj.scales", .dtype = .bf16, .shape = &.{ d, own / 64 }, .bytes = std.mem.sliceAsBytes(&down_s) },
        .{ .name = base ++ "down_proj.biases", .dtype = .bf16, .shape = &.{ d, own / 64 }, .bytes = std.mem.sliceAsBytes(&down_b) },
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"metadata":{"total_size":1},"weight_map":{
        \\"backbone.layers.0.mixer.shared_experts.up_proj.weight":"model-00001-of-00001.safetensors",
        \\"backbone.layers.0.mixer.shared_experts.up_proj.scales":"model-00001-of-00001.safetensors",
        \\"backbone.layers.0.mixer.shared_experts.up_proj.biases":"model-00001-of-00001.safetensors",
        \\"backbone.layers.0.mixer.shared_experts.down_proj.weight":"model-00001-of-00001.safetensors",
        \\"backbone.layers.0.mixer.shared_experts.down_proj.scales":"model-00001-of-00001.safetensors",
        \\"backbone.layers.0.mixer.shared_experts.down_proj.biases":"model-00001-of-00001.safetensors"}}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"nemotron_h\",\"moe_shared_expert_intermediate_size\":64}" });
    var c: cfg.Config = .{ .hidden = d, .vocab = 16, .layers = 1, .mamba_heads = 1, .mamba_head_dim = 1, .groups = 1, .state = 1, .conv_kernel = 4, .heads = 1, .kv_heads = 1, .head_dim = 1, .experts = 1, .top_k = 1, .expert_width = 64, .shared_width = own, .routed_scaling = 1, .eps = 1e-5 };
    c.kinds[0] = .moe;
    // one learned neuron: its input weights and its output
    var up: [learned * d]f32 = @splat(0);
    var down: [learned * d]f32 = @splat(0);
    for (up[0..d], down[0..d]) |*u, *v| {
        u.* = r.floatNorm(f32) * 0.05;
        v.* = r.floatNorm(f32) * 0.2;
    }
    const asked_up = up;
    var rounded_up = up;
    var rounded_down = down;
    _ = try Shards.write(gpa, io, dir, c, &.{0}, learned, &up, &down);
    // round gives what write hands back, bit for bit, and rounding it again changes nothing
    try Shards.round(gpa, learned, d, &rounded_up, &rounded_down);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&up), std.mem.sliceAsBytes(&rounded_up));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&down), std.mem.sliceAsBytes(&rounded_down));
    try Shards.round(gpa, learned, d, &rounded_up, &rounded_down);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&down), std.mem.sliceAsBytes(&rounded_down));
    try std.testing.expectEqual(@as(?i64, learned), try core.shard_edit.configInt(gpa, io, dir, cfg.pocket_key));
    try std.testing.expectEqual(@as(?i64, own + learned), try core.shard_edit.configInt(gpa, io, dir, "moe_shared_expert_intermediate_size"));
    // handed back as written: within half a step of what was asked
    for (up[0..d], asked_up[0..d]) |got, want| try std.testing.expectApproxEqAbs(want, got, 0.01);
    const size_after_widen = blk: {
        var f = try core.safetensors.File.open(gpa, io, shard);
        defer f.close(io);
        const uw = f.get(base ++ "up_proj.weight").?;
        try std.testing.expect(uw.is(.u32, &.{ own + learned, d / 8 }));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&up_w), uw.bytes[0 .. own * d / 2]);
        const dw = f.get(base ++ "down_proj.weight").?;
        try std.testing.expect(dw.is(.u32, &.{ d, (own + learned) / 8 }));
        const ds = f.get(base ++ "down_proj.scales").?;
        const db = f.get(base ++ "down_proj.biases").?;
        var col: [learned]f32 = undefined;
        for (0..d) |o| {
            // own columns byte for byte, learned ones as the down array now holds them
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(down_w[o * own / 8 ..][0 .. own / 8]), dw.bytes[o * (own + learned) / 2 ..][0 .. own / 2]);
            const w8: []align(1) const u32 = std.mem.bytesAsSlice(u32, dw.bytes[o * (own + learned) / 2 ..][own / 2 ..][0 .. learned / 2]);
            const s8: []align(1) const u16 = std.mem.bytesAsSlice(u16, ds.bytes[o * (own + learned) / 64 * 2 ..][own / 64 * 2 ..][0 .. learned / 64 * 2]);
            const b8: []align(1) const u16 = std.mem.bytesAsSlice(u16, db.bytes[o * (own + learned) / 64 * 2 ..][own / 64 * 2 ..][0 .. learned / 64 * 2]);
            var wa: [learned / 8]u32 = undefined;
            for (&wa, w8) |*x, y| x.* = y;
            host4.dequantize(&wa, &.{s8[0]}, &.{b8[0]}, &col);
            for (col, 0..) |x, k| try std.testing.expectEqual(down[k * d + o], x);
        }
        break :blk (try tmp.dir.statFile(io, "model-00001-of-00001.safetensors", .{})).size;
    };
    // a second neuron: written in place, nothing grows, the first one stays
    for (up[d .. 2 * d], down[d .. 2 * d]) |*u, *v| {
        u.* = r.floatNorm(f32) * 0.05;
        v.* = r.floatNorm(f32) * 0.2;
    }
    const first = up[0..d].*;
    c.pocket = learned;
    _ = try Shards.write(gpa, io, dir, c, &.{0}, learned, &up, &down);
    try std.testing.expectEqual(size_after_widen, (try tmp.dir.statFile(io, "model-00001-of-00001.safetensors", .{})).size);
    try std.testing.expectEqualSlices(f32, &first, up[0..d]);
    var f = try core.safetensors.File.open(gpa, io, shard);
    defer f.close(io);
    const uw = f.get(base ++ "up_proj.weight").?;
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&up_w), uw.bytes[0 .. own * d / 2]);
    var back: [d]f32 = undefined;
    const us = f.get(base ++ "up_proj.scales").?;
    const ub = f.get(base ++ "up_proj.biases").?;
    var wa: [d / 8]u32 = undefined;
    for (&wa, std.mem.bytesAsSlice(u32, uw.bytes[(own + 1) * d / 2 ..][0 .. d / 2])) |*x, y| x.* = y;
    const s1: []align(1) const u16 = std.mem.bytesAsSlice(u16, us.bytes[(own + 1) * 2 ..][0..2]);
    const b1: []align(1) const u16 = std.mem.bytesAsSlice(u16, ub.bytes[(own + 1) * 2 ..][0..2]);
    host4.dequantize(&wa, &.{s1[0]}, &.{b1[0]}, &back);
    try std.testing.expectEqualSlices(f32, up[d .. 2 * d], &back);
}
