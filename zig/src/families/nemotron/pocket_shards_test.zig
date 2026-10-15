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
    try Shards.round(gpa, d, &rounded_up, &rounded_down);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&up), std.mem.sliceAsBytes(&rounded_up));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&down), std.mem.sliceAsBytes(&rounded_down));
    try Shards.round(gpa, d, &rounded_up, &rounded_down);
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

test "a pocket rounded in place is what a write hands back, and writing it again changes no value" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    const base = "backbone.layers.0.mixer.shared_experts.";
    var w: [own * d / 8]u32 = undefined;
    for (&w) |*x| x.* = r.int(u32);
    var sb: [own * d / 64]u16 = undefined;
    for (&sb) |*x| x.* = host4.bf16of(0.01);
    const shard = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(shard);
    const sw = std.mem.sliceAsBytes(&w);
    const ss = std.mem.sliceAsBytes(&sb);
    const parts = [_]core.shard_edit.New{
        .{ .name = base ++ "up_proj.weight", .dtype = .u32, .shape = &.{ own, d / 8 }, .bytes = sw },
        .{ .name = base ++ "up_proj.scales", .dtype = .bf16, .shape = &.{ own, d / 64 }, .bytes = ss },
        .{ .name = base ++ "up_proj.biases", .dtype = .bf16, .shape = &.{ own, d / 64 }, .bytes = ss },
        .{ .name = base ++ "down_proj.weight", .dtype = .u32, .shape = &.{ d, own / 8 }, .bytes = sw },
        .{ .name = base ++ "down_proj.scales", .dtype = .bf16, .shape = &.{ d, own / 64 }, .bytes = ss },
        .{ .name = base ++ "down_proj.biases", .dtype = .bf16, .shape = &.{ d, own / 64 }, .bytes = ss },
    };
    try core.shard_edit.create(gpa, shard, &.{.{ "format", "mlx" }}, &parts);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"nemotron_h\",\"moe_shared_expert_intermediate_size\":64}" });
    var c: cfg.Config = .{ .hidden = d, .vocab = 16, .layers = 1, .mamba_heads = 1, .mamba_head_dim = 1, .groups = 1, .state = 1, .conv_kernel = 4, .heads = 1, .kv_heads = 1, .head_dim = 1, .experts = 1, .top_k = 1, .expert_width = 64, .shared_width = own, .routed_scaling = 1, .eps = 1e-5 };
    c.kinds[0] = .moe;
    // three neurons of mixed sign, one zeroed as a delete leaves it, the rest of the pocket empty
    var up: [learned * d]f32 = @splat(0);
    var down: [learned * d]f32 = @splat(0);
    for (up[0 .. 3 * d], down[0 .. 3 * d]) |*u, *v| {
        u.* = r.floatNorm(f32) * 0.05;
        v.* = r.floatNorm(f32) * 0.2;
    }
    @memset(up[d .. 2 * d], 0);
    @memset(down[d .. 2 * d], 0);
    var rounded_up = up;
    var rounded_down = down;
    try Shards.round(gpa, d, &rounded_up, &rounded_down);
    _ = try Shards.write(gpa, io, dir, c, &.{0}, learned, &up, &down);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&up), std.mem.sliceAsBytes(&rounded_up));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&down), std.mem.sliceAsBytes(&rounded_down));
    try std.testing.expect(std.mem.allEqual(f32, up[d .. 2 * d], 0));
    // a save of what was rounded writes and hands back the same values, so a check before it reads what it writes
    c.pocket = learned;
    _ = try Shards.write(gpa, io, dir, c, &.{0}, learned, &up, &down);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&rounded_up), std.mem.sliceAsBytes(&up));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&rounded_down), std.mem.sliceAsBytes(&down));
}

test "a write that fails at the second shard leaves the first written; writing the old pocket again restores it" {
    if (std.c.geteuid() == 0) return error.SkipZigTest; // a read-only file stops no one running as root
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    var w: [own * d / 8]u32 = undefined;
    for (&w) |*x| x.* = r.int(u32);
    var sb: [own * d / 64]u16 = undefined;
    for (&sb) |*x| x.* = host4.bf16of(0.01);
    const sw = std.mem.sliceAsBytes(&w);
    const ss = std.mem.sliceAsBytes(&sb);
    // MoE layer 0's shared expert in the first shard, layer 1's in the second
    const files = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
    const bases = [_][]const u8{ "backbone.layers.0.mixer.shared_experts.", "backbone.layers.1.mixer.shared_experts." };
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(gpa);
    try index.appendSlice(gpa, "{\"metadata\":{\"total_size\":1},\"weight_map\":{");
    for (files, bases, 0..) |file, base, f| {
        const leaves = [_][]const u8{ "up_proj.weight", "up_proj.scales", "up_proj.biases", "down_proj.weight", "down_proj.scales", "down_proj.biases" };
        var names: [6][]u8 = undefined;
        for (&names, leaves) |*n, leaf| n.* = try std.mem.concat(gpa, u8, &.{ base, leaf });
        defer for (names) |n| gpa.free(n);
        const parts = [_]core.shard_edit.New{
            .{ .name = names[0], .dtype = .u32, .shape = &.{ own, d / 8 }, .bytes = sw },
            .{ .name = names[1], .dtype = .bf16, .shape = &.{ own, d / 64 }, .bytes = ss },
            .{ .name = names[2], .dtype = .bf16, .shape = &.{ own, d / 64 }, .bytes = ss },
            .{ .name = names[3], .dtype = .u32, .shape = &.{ d, own / 8 }, .bytes = sw },
            .{ .name = names[4], .dtype = .bf16, .shape = &.{ d, own / 64 }, .bytes = ss },
            .{ .name = names[5], .dtype = .bf16, .shape = &.{ d, own / 64 }, .bytes = ss },
        };
        const path = try std.fs.path.join(gpa, &.{ dir, file });
        defer gpa.free(path);
        try core.shard_edit.create(gpa, path, &.{.{ "format", "mlx" }}, &parts);
        for (names, 0..) |n, i| try index.print(gpa, "{s}\"{s}\":\"{s}\"", .{ if (f + i == 0) "" else ",", n, file });
    }
    try index.appendSlice(gpa, "}}");
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"nemotron_h\",\"moe_shared_expert_intermediate_size\":64}" });
    var c: cfg.Config = .{ .hidden = d, .vocab = 16, .layers = 2, .mamba_heads = 1, .mamba_head_dim = 1, .groups = 1, .state = 1, .conv_kernel = 4, .heads = 1, .kv_heads = 1, .head_dim = 1, .experts = 1, .top_k = 1, .expert_width = 64, .shared_width = own, .routed_scaling = 1, .eps = 1e-5 };
    c.kinds[0] = .moe;
    c.kinds[1] = .moe;
    // a neuron at each layer, saved: the pocket as the shards hold it
    var up: [2 * learned * d]f32 = @splat(0);
    var down: [2 * learned * d]f32 = @splat(0);
    for ([_]usize{ 0, learned * d }) |at| for (up[at..][0..d], down[at..][0..d]) |*u, *v| {
        u.* = r.floatNorm(f32) * 0.05;
        v.* = r.floatNorm(f32) * 0.2;
    };
    _ = try Shards.write(gpa, io, dir, c, &.{ 0, 1 }, learned, &up, &down);
    c.pocket = learned;
    const saved_up = up;
    const saved_down = down;
    var before: [2][]u8 = undefined;
    for (&before, files) |*b, file| b.* = try tmp.dir.readFileAlloc(io, file, gpa, .limited(1 << 20));
    defer for (before) |b| gpa.free(b);
    // a delete's pocket (both neurons' input weights zeroed), written while the second shard cannot be
    var cut_up = saved_up;
    var cut_down = saved_down;
    @memset(cut_up[0..d], 0);
    @memset(cut_up[learned * d ..][0..d], 0);
    const second = try std.fs.path.joinZ(gpa, &.{ dir, files[1] });
    defer gpa.free(second);
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(second, 0o444));
    defer _ = std.c.chmod(second, 0o644);
    try std.testing.expectError(error.ShardWrite, Shards.write(gpa, io, dir, c, &.{ 0, 1 }, learned, &cut_up, &cut_down));
    const torn = try tmp.dir.readFileAlloc(io, files[0], gpa, .limited(1 << 20));
    defer gpa.free(torn);
    try std.testing.expect(!std.mem.eql(u8, before[0], torn));
    const kept = try tmp.dir.readFileAlloc(io, files[1], gpa, .limited(1 << 20));
    defer gpa.free(kept);
    try std.testing.expectEqualSlices(u8, before[1], kept);
    // the pocket as it was, written again, fails at the same shard, but the first shard is as it was by then
    var old_up = saved_up;
    var old_down = saved_down;
    try std.testing.expectError(error.ShardWrite, Shards.write(gpa, io, dir, c, &.{ 0, 1 }, learned, &old_up, &old_down));
    for (before, files) |b, file| {
        const now = try tmp.dir.readFileAlloc(io, file, gpa, .limited(1 << 20));
        defer gpa.free(now);
        try std.testing.expectEqualSlices(u8, b, now);
    }
}

test "a pocket as learning leaves it rounds once and stays: a second rounding changes no bit" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(872);
    const r = prng.random();
    const width = 128;
    var up: [learned * width]f32 = undefined;
    var down: [learned * width]f32 = undefined;
    for (0..200) |trial| {
        // inputs centred on zero (half with a mild lean), outputs zero past the neurons in use
        const sd = std.math.pow(f32, 10, -3 + 2 * r.float(f32));
        const used = 1 + r.uintLessThan(usize, learned);
        const lean = if (trial % 2 == 0) @as(f32, 0) else sd * 4 * r.float(f32);
        for (0..learned) |k| for (0..width) |o| {
            up[k * width + o] = lean + sd * r.floatNorm(f32);
            down[k * width + o] = if (k < used) sd * r.floatNorm(f32) else 0;
        };
        try Shards.round(gpa, width, &up, &down);
        const once_up = up;
        const once_down = down;
        try Shards.round(gpa, width, &up, &down);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&once_up), std.mem.sliceAsBytes(&up));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&once_down), std.mem.sliceAsBytes(&down));
    }
}

/// A pocket whose groups sit up to 256 of their spreads from zero, seeded: most move when rounded again.
pub fn farFromZero(r: std.Random, up: []f32, down: []f32) void {
    const spread = std.math.pow(f32, 10, -4 + 6 * r.float(f32));
    const offset = spread * 256 * r.float(f32) * @as(f32, if (r.boolean()) 1 else -1);
    for (up, down) |*u, *v| {
        u.* = offset + spread * r.floatNorm(f32);
        v.* = -offset + spread * r.floatNorm(f32);
    }
}

test "a group far from zero next to its spread can move when rounded again, which a delete refuses" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);
    const r = prng.random();
    const width = 64;
    var up: [learned * width]f32 = undefined;
    var down: [learned * width]f32 = undefined;
    var moved = false;
    for (0..20) |_| {
        farFromZero(r, &up, &down);
        try Shards.round(gpa, width, &up, &down);
        const once_up = up;
        const once_down = down;
        try Shards.round(gpa, width, &up, &down);
        moved = moved or !std.mem.eql(u8, std.mem.sliceAsBytes(&once_up), std.mem.sliceAsBytes(&up)) or !std.mem.eql(u8, std.mem.sliceAsBytes(&once_down), std.mem.sliceAsBytes(&down));
    }
    try std.testing.expect(moved);
}
