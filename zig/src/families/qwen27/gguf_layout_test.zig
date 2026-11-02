//! Measured GGUF layout contracts use independent tagged heads and stored norm multipliers.
const std = @import("std");
const t = std.testing;
const gg = @import("../../core/gguf.zig");
const cfg = @import("gguf_config.zig");
const names = @import("gguf_names.zig");
const image = @import("gguf_image.zig");
const heads = @import("gguf_heads.zig");
const repack = @import("gguf_repack.zig");
const fixture = @import("gguf_fixture.zig");

test "GGUF Qwen: measured 16-key 48-value inverse map is a bijection with three consecutive native heads" {
    const o: heads.Order = .{ .axis = .rows, .key_heads = 16, .value_heads = 48, .width = 128 };
    const expected = [_]usize{ 0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47 };
    var seen: [48]bool = @splat(false);
    for (expected, 0..) |j, i| {
        try t.expectEqual(j, o.sourceHead(i));
        try t.expectEqual(i, (j % 16) * 3 + j / 16);
        try t.expect(!seen[j]);
        seen[j] = true;
    }
}
test "GGUF Qwen: every norm is a stored multiplier with no added or removed offset and no head permutation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try fixture.make(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    const plan = try names.plan(a, &d, try cfg.parse(a, try cfg.build(a, &d)));
    var count: usize = 0;
    for (plan) |entry| {
        if ((entry.transform != .bf16 and entry.transform != .f32) or std.mem.indexOf(u8, entry.name, "norm.weight") == null) continue;
        count += 1;
        var input = entry;
        const raw = try a.alloc(u8, entry.tensor.bytes.len);
        input.tensor.bytes = raw;
        for (0..raw.len / 4) |i| {
            const x: f32 = @bitCast(@as(u32, @as(u16, 0x3f00) + @as(u16, @intCast(i % 128))) << 16);
            std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(x), .little);
        }
        const out = try image.small(t.allocator, input);
        defer t.allocator.free(out);
        for (out, 0..) |value, i| try t.expectEqual(@as(u16, 0x3f00) + @as(u16, @intCast(i % 128)), value);
    }
    try t.expectEqual(@as(usize, 14), count);
}
test "GGUF Qwen: native image inverts scalar heads and only convolution V channels" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try fixture.make(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    const plan = try names.plan(a, &d, try cfg.parse(a, try cfg.build(a, &d)));
    const native_to_gguf = [_]usize{ 0, 2, 1, 3 };
    var checked: usize = 0;
    for (plan) |entry| {
        const conv = std.mem.endsWith(u8, entry.name, "linear_attn.conv1d.weight");
        const bias = std.mem.endsWith(u8, entry.name, "linear_attn.dt_bias");
        const direct = entry.transform == .ssm_a;
        if (!conv and !bias and !direct) continue;
        checked += 1;
        var input = entry;
        const raw = try a.alloc(u8, entry.tensor.bytes.len);
        input.tensor.bytes = raw;
        const coefficients = [_]f32{ -0.1104574353, -1.001953125, -0.7504296303, -0.9551389813 };
        for (0..raw.len / 4) |i| {
            const value: f32 = if (direct) coefficients[i] else @bitCast(@as(u32, @as(u16, 0x3f00) + @as(u16, @intCast(i))) << 16);
            std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(value), .little);
        }
        const out = if (direct) try image.rawF32(a, input) else blk: {
            const narrowed = try image.small(a, input);
            const wide = try a.alloc(f32, narrowed.len);
            for (wide, narrowed) |*v, h| v.* = @bitCast(@as(u32, h) << 16);
            break :blk wide;
        };
        for (out, 0..) |value, i| {
            const row = if (conv) i / 4 else i;
            const sr = if (conv and row < 64) row else if (conv) 64 + native_to_gguf[(row - 64) / 32] * 32 + (row - 64) % 32 else native_to_gguf[row];
            const si = if (conv) sr * 4 + i % 4 else sr;
            const expected: f32 = if (direct) coefficients[si] else @bitCast(@as(u32, @as(u16, 0x3f00) + @as(u16, @intCast(si))) << 16);
            try t.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(value)));
        }
    }
    try t.expectEqual(@as(usize, 9), checked);
}
fn exerciseAffine(a: std.mem.Allocator, bits: u8, group: usize, axis: heads.Axis) !void {
    const n: usize = if (axis == .rows) 8 else 3;
    const k: usize = if (axis == .rows) 128 else 512;
    var p: repack.Affine = blk: {
        const words = try a.alloc(u32, n * k * bits / 32);
        errdefer a.free(words);
        const scales = try a.alloc(f32, n * k / group);
        errdefer a.free(scales);
        const biases = try a.alloc(f32, scales.len);
        break :blk .{ .n = n, .k = k, .bits = bits, .group = group, .words = words, .scales = scales, .biases = biases };
    };
    defer p.deinit(a);
    for (p.words, 0..) |*v, i| v.* = @truncate(i * 2654435761 + 123);
    for (p.scales, p.biases, 0..) |*s, *b, i| {
        s.* = @bitCast(@as(u32, 0x3f000000) + @as(u32, @intCast(i)));
        b.* = -@as(f32, @floatFromInt(i));
    }
    const old_words = try a.dupe(u32, p.words);
    defer a.free(old_words);
    const old_scales = try a.dupe(f32, p.scales);
    defer a.free(old_scales);
    const old_biases = try a.dupe(f32, p.biases);
    defer a.free(old_biases);
    const order: heads.Order = .{ .axis = axis, .key_heads = 2, .value_heads = 6, .width = if (axis == .rows) 1 else 64, .offset = if (axis == .rows) 1 else 64 };
    try heads.affine(a, &p, order);
    const map = [_]usize{ 0, 2, 4, 1, 3, 5 };
    const wk = k * bits / 32;
    const width = if (axis == .rows) 1 else 64 * @as(usize, bits) / 32;
    const offset = if (axis == .rows) 1 else 64 * @as(usize, bits) / 32;
    for (p.words, 0..) |v, i| {
        const x = if (axis == .rows) i / wk else i % wk;
        const sx = if (x < offset or x >= offset + 6 * width) x else offset + map[(x - offset) / width] * width + (x - offset) % width;
        try t.expectEqual(old_words[if (axis == .rows) sx * wk + i % wk else i / wk * wk + sx], v);
    }
    const gk = k / group;
    const gw = if (axis == .rows) 1 else 64 / group;
    const go = if (axis == .rows) 1 else 64 / group;
    for (p.scales, p.biases, 0..) |s, b, i| {
        const x = if (axis == .rows) i / gk else i % gk;
        const sx = if (x < go or x >= go + 6 * gw) x else go + map[(x - go) / gw] * gw + (x - go) % gw;
        const si = if (axis == .rows) sx * gk + i % gk else i / gk * gk + sx;
        try t.expectEqual(old_scales[si], s);
        try t.expectEqual(old_biases[si], b);
    }
}
test "GGUF Qwen: head reorder preserves packed codes and F32 metadata for all four admitted forms" {
    for ([_]struct { u8, usize }{ .{ 4, 32 }, .{ 5, 32 }, .{ 6, 16 }, .{ 8, 32 } }) |f| {
        for ([_]heads.Axis{ .rows, .columns }) |axis| try exerciseAffine(t.allocator, f[0], f[1], axis);
    }
    try t.checkAllAllocationFailures(t.allocator, exerciseAffine, .{ @as(u8, 5), @as(usize, 32), heads.Axis.columns });
}
