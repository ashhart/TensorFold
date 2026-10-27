//! The host oracle of the device K/V table: one layer's slots as the kernel addresses them.

const std = @import("std");
const shape = @import("../pack/exl3_shape.zig");
const cache = @import("../model/exl3_cache.zig");

/// Full-attention layers only. GDN layers have no KV table. The size is `contextKv`.
const full_layers: usize = shape.layers / shape.full_attention_interval;

pub const Table = struct {
    heads: usize,
    tokens: usize,
    dtype: cache.Dtype,
    bytes: []u8,

    pub fn init(gpa: std.mem.Allocator, tokens: usize, dtype: cache.Dtype) error{ HeadDim, OutOfMemory }!Table {
        const n = try cache.contextKv(tokens, dtype);
        const bytes = try gpa.alloc(u8, @intCast(n));
        @memset(bytes, 0);
        return .{ .heads = shape.kv_heads, .tokens = tokens, .dtype = dtype, .bytes = bytes };
    }

    pub fn deinit(self: *Table, gpa: std.mem.Allocator) void {
        gpa.free(self.bytes);
    }

    /// One head, one token, one full-attention layer.
    pub fn write(self: *Table, head: usize, layer: usize, token: usize, src: []const u8) error{ Head, Slot, Length }!void {
        const off = try self.locate(head, layer, token, src.len);
        @memcpy(self.bytes[off..][0..src.len], src);
    }

    pub fn read(self: *const Table, head: usize, layer: usize, token: usize, dest: []u8) error{ Head, Slot, Length }!void {
        const off = try self.locate(head, layer, token, dest.len);
        @memcpy(dest, self.bytes[off..][0..dest.len]);
    }

    fn locate(self: *const Table, head: usize, layer: usize, token: usize, len: usize) error{ Head, Slot, Length }!usize {
        if (head >= self.heads) return error.Head;
        if (self.tokens == 0 or full_layers == 0 or layer >= full_layers or token >= self.tokens) return error.Slot;
        const denom = full_layers * self.tokens * self.heads;
        if (self.bytes.len % denom != 0) return error.Slot;
        const one = self.bytes.len / denom;
        if (len != one) return error.Length;
        return ((layer * self.tokens + token) * self.heads + head) * one;
    }
};

test "the table is contextKv bytes, both heads" {
    const gpa = std.testing.allocator;
    const tokens: usize = 8192;
    for ([_]struct { dtype: cache.Dtype, bytes: u64 }{
        .{ .dtype = .bf16, .bytes = 201_326_592 },
        .{ .dtype = .int8, .bytes = 106_954_752 },
        .{ .dtype = .int4, .bytes = 56_623_104 },
    }) |case| {
        var t = try Table.init(gpa, tokens, case.dtype);
        defer t.deinit(gpa);
        try std.testing.expectEqual(shape.kv_heads, t.heads);
        try std.testing.expectEqual(case.bytes, @as(u64, t.bytes.len));
        try std.testing.expectEqual(try cache.contextKv(tokens, case.dtype), @as(u64, t.bytes.len));
    }
    var none = try Table.init(gpa, 0, .bf16);
    defer none.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), none.bytes.len);
}

test "a write lands in its own head's slot and nowhere else" {
    const gpa = std.testing.allocator;
    const tokens: usize = 2;
    const cases = [_]struct { dtype: cache.Dtype, slot: usize, mark: u8 }{
        .{ .dtype = .bf16, .slot = 1024, .mark = 0xa5 },
        .{ .dtype = .int8, .slot = 544, .mark = 0x11 },
        .{ .dtype = .int4, .slot = 288, .mark = 0x22 },
    };
    var src: [1024]u8 = undefined;
    var got: [1024]u8 = undefined;
    for (cases) |case| {
        var t = try Table.init(gpa, tokens, case.dtype);
        defer t.deinit(gpa);
        const slot = src[0..case.slot];
        @memset(slot, case.mark);
        try std.testing.expectError(error.Head, t.write(shape.kv_heads, 0, 0, slot));
        try std.testing.expectError(error.Length, t.write(0, 0, 0, slot[0 .. case.slot - 1]));
        try std.testing.expectError(error.Slot, t.write(0, 0, tokens, slot));
        try std.testing.expectError(error.Slot, t.write(0, full_layers, 0, slot));
        try std.testing.expect(std.mem.allEqual(u8, t.bytes, 0));

        try t.write(0, 0, 1, slot);
        @memset(&got, 0);
        try t.read(0, 0, 1, got[0..case.slot]);
        try std.testing.expectEqualSlices(u8, slot, got[0..case.slot]);
        try t.read(1, 0, 1, got[0..case.slot]);
        try std.testing.expect(std.mem.allEqual(u8, got[0..case.slot], 0));

        @memset(slot, case.mark +% 1);
        try t.write(1, 1, 0, slot);
        try t.read(1, 1, 0, got[0..case.slot]);
        try std.testing.expectEqualSlices(u8, slot, got[0..case.slot]);
        try t.read(0, 0, 1, got[0..case.slot]);
        try std.testing.expect(got[0] == case.mark);
    }
    var none = try Table.init(gpa, 0, .bf16);
    defer none.deinit(gpa);
    var scratch: [0]u8 = .{};
    try std.testing.expectError(error.Slot, none.write(0, 0, 0, &scratch));
    try std.testing.expectError(error.Slot, none.read(0, 0, 0, &scratch));
}
