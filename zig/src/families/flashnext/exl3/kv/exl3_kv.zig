//! The host oracle of the device K/V table: one layer's slots as the kernel addresses them.

const std = @import("std");
const shape = @import("../pack/exl3_shape.zig");
const cache = @import("../model/exl3_cache.zig");

/// Full-attention layers only. GDN layers have no KV table. The size is `contextKv`.
const full_layers: usize = shape.layers / shape.full_attention_interval;

pub const Table = struct {
    heads: usize,
    tokens: usize,
    bytes: []u8,

    pub fn init(gpa: std.mem.Allocator, tokens: usize) error{OutOfMemory}!Table {
        const n = cache.contextKv(tokens);
        const bytes = try gpa.alloc(u8, @intCast(n));
        @memset(bytes, 0);
        return .{ .heads = shape.kv_heads, .tokens = tokens, .bytes = bytes };
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
    var t = try Table.init(gpa, tokens);
    defer t.deinit(gpa);
    try std.testing.expectEqual(shape.kv_heads, t.heads);
    try std.testing.expectEqual(@as(u64, 201_326_592), @as(u64, t.bytes.len));
    try std.testing.expectEqual(cache.contextKv(tokens), @as(u64, t.bytes.len));
    var none = try Table.init(gpa, 0);
    defer none.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), none.bytes.len);
}

test "a write lands in its own head's slot and nowhere else" {
    const gpa = std.testing.allocator;
    const tokens: usize = 2;
    const slot_bytes: usize = 1024;
    const mark: u8 = 0xa5;
    var src: [slot_bytes]u8 = undefined;
    var got: [slot_bytes]u8 = undefined;
    {
        var t = try Table.init(gpa, tokens);
        defer t.deinit(gpa);
        const slot = src[0..slot_bytes];
        @memset(slot, mark);
        try std.testing.expectError(error.Head, t.write(shape.kv_heads, 0, 0, slot));
        try std.testing.expectError(error.Length, t.write(0, 0, 0, slot[0 .. slot_bytes - 1]));
        try std.testing.expectError(error.Slot, t.write(0, 0, tokens, slot));
        try std.testing.expectError(error.Slot, t.write(0, full_layers, 0, slot));
        try std.testing.expect(std.mem.allEqual(u8, t.bytes, 0));

        try t.write(0, 0, 1, slot);
        @memset(&got, 0);
        try t.read(0, 0, 1, got[0..slot_bytes]);
        try std.testing.expectEqualSlices(u8, slot, got[0..slot_bytes]);
        try t.read(1, 0, 1, got[0..slot_bytes]);
        try std.testing.expect(std.mem.allEqual(u8, got[0..slot_bytes], 0));

        @memset(slot, mark +% 1);
        try t.write(1, 1, 0, slot);
        try t.read(1, 1, 0, got[0..slot_bytes]);
        try std.testing.expectEqualSlices(u8, slot, got[0..slot_bytes]);
        try t.read(0, 0, 1, got[0..slot_bytes]);
        try std.testing.expect(got[0] == mark);
    }
    var none = try Table.init(gpa, 0);
    defer none.deinit(gpa);
    var scratch: [0]u8 = .{};
    try std.testing.expectError(error.Slot, none.write(0, 0, 0, &scratch));
    try std.testing.expectError(error.Slot, none.read(0, 0, 0, &scratch));
}
