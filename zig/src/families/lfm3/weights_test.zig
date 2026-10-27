//! Metadata fixtures exercise admission without allocating tensor payloads or a GPU device.
const std = @import("std");
const w = @import("weights.zig");
const c = @import("config.zig");
const st = @import("safetensors");

fn config(bits: u8, group: u16) c.Config {
    return .{ .arena = std.heap.ArenaAllocator.init(std.testing.allocator), .kind = .text, .context = 262144, .quant = .{ .global = .{ .bits = bits, .group = group } } };
}

fn header(a: std.mem.Allocator, cfg: *const c.Config, namespace: w.Namespace) !st.Header {
    var h: st.Header = .empty;
    var offset: usize = 0;
    for (try w.schema(a, cfg, namespace)) |e| {
        var bytes: usize = e.dtype.size();
        for (e.shape[0..e.rank]) |d| bytes *= d;
        try h.put(a, e.name, .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .begin = offset, .end = offset + bytes });
        offset += bytes;
    }
    return h;
}

test "official text4 tensor count and bytes; both namespaces and every target bit width" {
    for ([_]u8{ 4, 6, 8 }) |bits| for ([_]w.Namespace{ .bare, .wrapped }) |namespace| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(bits, 64);
        defer cfg.deinit();
        const h = try header(arena.allocator(), &cfg, namespace);
        const m = try w.validate(std.testing.allocator, &cfg, &h);
        try std.testing.expectEqual(@as(usize, 594), m.tensors);
        try std.testing.expectEqual(namespace, m.namespace);
        if (bits == 4) try std.testing.expectEqual(@as(usize, 973008384), m.bytes);
    };
}

test "VL text uses mixed8 embedding with group32 without admitting its image tower" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cfg = config(4, 32);
    defer cfg.deinit();
    cfg.kind = .vision_language;
    try cfg.quant.modules.put(cfg.arena.allocator(), "embed_tokens", .{ .bits = 8, .group = 32 });
    var h = try header(a, &cfg, .wrapped);
    const embed = h.get("language_model.model.embed_tokens.weight").?;
    try std.testing.expectEqual(@as(usize, 512), embed.shape[1]);
    const gate = h.get("language_model.model.layers.2.self_attn.gate_proj.scales").?;
    try std.testing.expectEqualSlices(usize, &.{ 16, 64 }, gate.shape[0..2]);
    try h.put(a, "vision_tower.unqualified.weight", .{ .dtype = .bf16, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .begin = 0, .end = 2 });
    const m = try w.validate(std.testing.allocator, &cfg, &h);
    try std.testing.expectEqual(@as(usize, 594), m.tensors);
    try std.testing.expectEqual(@as(usize, 1212106240), m.bytes);
}

test "missing head gate, wrong packed width and malformed convolution are rejected" {
    const keys = [_][]const u8{
        "language_model.model.layers.2.self_attn.gate_proj.weight",
        "language_model.model.layers.0.conv.in_proj.weight",
        "language_model.model.layers.0.conv.conv.weight",
        "language_model.model.layers.0.post_operator_norm.weight",
    };
    for (keys, 0..) |key, i| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(4, 64);
        defer cfg.deinit();
        var h = try header(arena.allocator(), &cfg, .wrapped);
        if (i == 0) {
            _ = h.swapRemove(key);
            try std.testing.expectError(error.MissingTensor, w.validate(std.testing.allocator, &cfg, &h));
        } else {
            h.getPtr(key).?.shape[if (i == 3) 0 else 1] += 1;
            try std.testing.expectError(error.TensorShape, w.validate(std.testing.allocator, &cfg, &h));
        }
    }
}

test "bad dtype, byte range, namespace and unexpected text tensors are rejected" {
    for (0..4) |mode| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(4, 64);
        defer cfg.deinit();
        var h = try header(arena.allocator(), &cfg, .wrapped);
        const key = "language_model.model.embed_tokens.weight";
        switch (mode) {
            0 => h.getPtr(key).?.dtype = .bf16,
            1 => h.getPtr(key).?.end -= 1,
            2 => try h.put(arena.allocator(), "model.embed_tokens.weight", h.get(key).?),
            3 => try h.put(arena.allocator(), "language_model.lm_head.weight", h.get(key).?),
            else => unreachable,
        }
        const err = switch (mode) {
            0 => error.TensorDType,
            1 => error.TensorBytes,
            2 => error.MixedNamespaces,
            3 => error.UnexpectedTensor,
            else => unreachable,
        };
        try std.testing.expectError(err, w.validate(std.testing.allocator, &cfg, &h));
    }
}
