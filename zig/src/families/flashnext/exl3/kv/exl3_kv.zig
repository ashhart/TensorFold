//! The buffer is `contextKvRank` for this rank's heads.

const std = @import("std");
const shape = @import("../pack/exl3_shape.zig");
const cache = @import("../model/exl3_cache.zig");

/// Full-attention layers only. GDN layers have no KV table. Same split as `contextKvRank`.
const full_layers: usize = shape.layers / shape.full_attention_interval;

pub const Table = struct {
    rank: usize,
    ranks: usize,
    head: usize,
    /// Two ranks store one head each, not both on rank 0.
    heads: usize,
    tokens: usize,
    dtype: cache.Dtype,
    bytes: []u8,

    /// `contextKvRank` refuses a zero rank count, or a count that does not divide the KV heads, before this divides.
    pub fn init(
        gpa: std.mem.Allocator,
        tokens: usize,
        dtype: cache.Dtype,
        rank: usize,
        ranks: usize,
    ) error{ HeadDim, Ranks, RankOutOfRange, OutOfMemory }!Table {
        const n = try cache.contextKvRank(tokens, dtype, ranks);
        if (rank >= ranks) return error.RankOutOfRange;
        const heads = shape.kv_heads / ranks;
        const bytes = try gpa.alloc(u8, @intCast(n));
        @memset(bytes, 0);
        return .{
            .rank = rank,
            .ranks = ranks,
            .head = rank * heads,
            .heads = heads,
            .tokens = tokens,
            .dtype = dtype,
            .bytes = bytes,
        };
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
        if (self.heads == 0 or head < self.head or head >= self.head + self.heads) return error.Head;
        if (self.tokens == 0 or full_layers == 0 or layer >= full_layers or token >= self.tokens) return error.Slot;
        const denom = full_layers * self.tokens * self.heads;
        if (self.bytes.len % denom != 0) return error.Slot;
        const one = self.bytes.len / denom;
        if (len != one) return error.Length;
        const local = head - self.head;
        return ((layer * self.tokens + token) * self.heads + local) * one;
    }
};

test "one rank stores one head, sized by contextKvRank, not both heads" {
    const gpa = std.testing.allocator;
    const tokens: usize = 8192;
    const ranks: usize = 2;
    const both = try cache.contextKv(tokens, .bf16);
    try std.testing.expectEqual(@as(u64, 201_326_592), both);

    {
        var r0 = try Table.init(gpa, tokens, .bf16, 0, ranks);
        defer r0.deinit(gpa);
        var r1 = try Table.init(gpa, tokens, .bf16, 1, ranks);
        defer r1.deinit(gpa);
        try std.testing.expectEqual(@as(usize, 0), r0.head);
        try std.testing.expectEqual(@as(usize, 1), r0.heads);
        try std.testing.expectEqual(@as(usize, 1), r1.head);
        try std.testing.expectEqual(@as(usize, 1), r1.heads);
        try std.testing.expectEqual(@as(u64, 100_663_296), @as(u64, r0.bytes.len));
        try std.testing.expectEqual(r0.bytes.len, r1.bytes.len);
        try std.testing.expectEqual(try cache.contextKvRank(tokens, .bf16, ranks), @as(u64, r0.bytes.len));
        try std.testing.expect(r0.bytes.len != both);
        try std.testing.expect(r1.bytes.len != both);
    }

    {
        var t = try Table.init(gpa, tokens, .int8, 0, ranks);
        defer t.deinit(gpa);
        try std.testing.expectEqual(@as(usize, 1), t.heads);
        try std.testing.expectEqual(@as(u64, 53_477_376), @as(u64, t.bytes.len));
        try std.testing.expectEqual(try cache.contextKvRank(tokens, .int8, ranks), @as(u64, t.bytes.len));
        try std.testing.expect(t.bytes.len != try cache.contextKv(tokens, .int8));
    }

    {
        var t = try Table.init(gpa, tokens, .int4, 1, ranks);
        defer t.deinit(gpa);
        try std.testing.expectEqual(@as(usize, 1), t.head);
        try std.testing.expectEqual(@as(usize, 1), t.heads);
        try std.testing.expectEqual(@as(u64, 28_311_552), @as(u64, t.bytes.len));
        try std.testing.expectEqual(try cache.contextKvRank(tokens, .int4, ranks), @as(u64, t.bytes.len));
        try std.testing.expect(t.bytes.len != try cache.contextKv(tokens, .int4));
    }

    var none = try Table.init(gpa, 0, .bf16, 0, ranks);
    defer none.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), none.bytes.len);
    try std.testing.expectEqual(@as(usize, 1), none.heads);
}

test "a bad rank is error.Ranks and not a zero-byte table" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.Ranks, Table.init(gpa, 8192, .bf16, 0, 0));
    try std.testing.expectError(error.Ranks, Table.init(gpa, 8192, .int8, 0, 0));
    try std.testing.expectError(error.Ranks, Table.init(gpa, 8192, .int4, 1, 0));
    try std.testing.expectError(error.Ranks, Table.init(gpa, 8192, .bf16, 0, 3));
    try std.testing.expectError(error.Ranks, Table.init(gpa, 8192, .int8, 0, 3));
    try std.testing.expectError(error.RankOutOfRange, Table.init(gpa, 8, .bf16, 2, 2));
    try std.testing.expectError(error.RankOutOfRange, Table.init(gpa, 8, .int4, 1, 1));
}

test "a write on one rank does not change the other rank's head" {
    const gpa = std.testing.allocator;
    const ranks: usize = 2;
    const tokens: usize = 2;
    const cases = [_]struct { dtype: cache.Dtype, slot: usize, mark: u8 }{
        .{ .dtype = .bf16, .slot = 1024, .mark = 0xa5 },
        .{ .dtype = .int8, .slot = 544, .mark = 0x11 },
        .{ .dtype = .int4, .slot = 288, .mark = 0x22 },
    };
    var src: [1024]u8 = undefined;
    var got: [1024]u8 = undefined;

    for (cases) |case| {
        var r0 = try Table.init(gpa, tokens, case.dtype, 0, ranks);
        defer r0.deinit(gpa);
        var r1 = try Table.init(gpa, tokens, case.dtype, 1, ranks);
        defer r1.deinit(gpa);
        const slot = src[0..case.slot];
        @memset(slot, case.mark);

        try std.testing.expectError(error.Head, r0.write(1, 0, 0, slot));
        try std.testing.expectError(error.Head, r1.write(0, 0, 0, slot));
        try std.testing.expectError(error.Length, r0.write(0, 0, 0, slot[0 .. case.slot - 1]));
        try std.testing.expectError(error.Slot, r0.write(0, 0, tokens, slot));
        try std.testing.expectError(error.Slot, r0.write(0, full_layers, 0, slot));
        try std.testing.expect(std.mem.allEqual(u8, r0.bytes, 0));
        try std.testing.expect(std.mem.allEqual(u8, r1.bytes, 0));

        try r0.write(0, 0, 1, slot);
        try std.testing.expect(std.mem.allEqual(u8, r1.bytes, 0));
        @memset(&got, 0);
        try r0.read(0, 0, 1, got[0..case.slot]);
        try std.testing.expectEqualSlices(u8, slot, got[0..case.slot]);
        try std.testing.expectError(error.Head, r0.read(1, 0, 1, got[0..case.slot]));

        const snap = try gpa.dupe(u8, r0.bytes);
        defer gpa.free(snap);
        @memset(slot, case.mark +% 1);
        try r1.write(1, 1, 0, slot);
        try std.testing.expectEqualSlices(u8, snap, r0.bytes);
        try r1.read(1, 1, 0, got[0..case.slot]);
        try std.testing.expectEqualSlices(u8, slot, got[0..case.slot]);
        try std.testing.expectError(error.Head, r1.read(0, 1, 0, got[0..case.slot]));
    }

    var none = try Table.init(gpa, 0, .bf16, 0, ranks);
    defer none.deinit(gpa);
    var scratch: [0]u8 = .{};
    try std.testing.expectError(error.Slot, none.write(0, 0, 0, &scratch));
    try std.testing.expectError(error.Slot, none.read(0, 0, 0, &scratch));
}
