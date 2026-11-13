//! Sparse request scores. -100 is a hard ban; other values add to FP32 logits before sampling.
const std = @import("std");
const sampling = @import("sampling.zig");
const grammar = @import("grammar.zig");
// This module is also built as the independent lanes root, so keep the wire conversion here.
const bf = struct {
    fn value(word: u16) f32 {
        return @bitCast(@as(u32, word) << 16);
    }
    fn bits(x: f32) u16 {
        const u: u32 = @bitCast(x);
        if (std.math.isNan(x)) return @truncate((u >> 16) | 0x40);
        return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
    }
    fn round32(x: f32) f32 {
        return value(bits(x));
    }
};

/// A request's list is sorted by token, each token once, as the server's parse leaves it.
pub const Entry = struct { token: u32, value: f32, ban: bool = false };

/// The entry for `token` in a sorted list: a binary search, since a host draw asks once a vocabulary token.
pub fn find(entries: []const Entry, token: u32) ?Entry {
    var lo: usize = 0;
    var hi: usize = entries.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (entries[mid].token == token) return entries[mid];
        if (entries[mid].token < token) lo = mid + 1 else hi = mid;
    }
    return null;
}

/// The existing subtract kernel's amount: combine once per cell, so no concurrent writers touch one logit.
pub fn subtract(penalty: f32, value: ?Entry) f32 {
    const b = value orelse return penalty;
    return if (b.ban) std.math.inf(f32) else penalty - b.value;
}

pub fn apply(values: []f32, entries: []const Entry) !void {
    for (entries) |e| {
        if (e.token >= values.len) return error.BadLogitBiasToken;
        const x = &values[e.token];
        x.* = if (e.ban) -std.math.inf(f32) else x.* - subtract(0, e);
    }
}

/// Host draws of raw BF16 rows apply a bias and round as the CUDA sparse launch does, without editing cached logits.
pub fn draw(a: std.mem.Allocator, words: []const u16, settings: ?sampling.Sampling, position: u64, mask: []const u32, entries: []const Entry) !u32 {
    const values = try a.alloc(f64, words.len);
    defer a.free(values);
    const ids = try a.alloc(u64, words.len);
    defer a.free(ids);
    var n: usize = 0;
    for (words, 0..) |word, id| {
        var x = bf.value(word);
        if (entries.len > 0) {
            const b = find(entries, @intCast(id));
            if (b != null and b.?.ban) continue;
            if (!std.math.isFinite(x)) return error.NonfiniteLogits;
            if (b != null) x = bf.round32(x - subtract(0, b));
        }
        // Empty entries also draws an already processed CUDA row, with bans encoded as -inf.
        if (x == -std.math.inf(f32)) continue;
        if (!std.math.isFinite(x)) return error.NonfiniteLogits;
        if (mask.len > 0 and !grammar.allows(mask, id)) continue;
        values[n] = x;
        ids[n] = id;
        n += 1;
    }
    if (n == 0) return error.NoAllowedToken;
    return @intCast(try sampling.choose(a, values[0..n], ids[0..n], position, settings orelse .{ .seed = 0, .temperature = 0 }));
}

test "bias shifts greedy picks, bans exclude even a nonfinite maximum and grammar cannot restore a ban" {
    const a = std.testing.allocator;
    const words = [_]u16{ bf.bits(10), bf.bits(9), bf.bits(0) };
    try std.testing.expectEqual(@as(u32, 1), try draw(a, &words, null, 0, &.{}, &.{.{ .token = 1, .value = 2 }}));
    const entries = [_]Entry{ .{ .token = 0, .value = -100, .ban = true }, .{ .token = 2, .value = 20 } };
    for (0..128) |position| {
        const s: sampling.Sampling = .{ .seed = position, .top_k = 0, .top_p = 1 };
        try std.testing.expect(try draw(a, &words, s, position, &.{}, &entries) != 0);
    }
    try std.testing.expectError(error.NoAllowedToken, draw(a, &words, null, 0, &.{1}, &entries));
    try std.testing.expectEqual(@as(u32, 1), try draw(a, &.{ 0x7f80, bf.bits(1) }, null, 0, &.{}, entries[0..1]));
    var values = [_]f32{ 10000, 1, 0 };
    try apply(&values, &entries);
    try std.testing.expectEqual(-std.math.inf(f32), values[0]);
    try std.testing.expectEqual(@as(f32, 20), values[2]);
}

test "bias draws equal main's row draw over the processed row, raw or as the device leaves it, and never take a ban" {
    const a = std.testing.allocator;
    const rowDraw = @import("sampling_row.zig").draw;
    var raw: [64]u16 = undefined;
    for (&raw, 0..) |*w, i| w.* = bf.bits(@as(f32, @floatFromInt((i * 37) % 61)) / 8.0 - 4.0);
    const entries = [_]Entry{ .{ .token = 3, .value = 1.25 }, .{ .token = 17, .value = -100, .ban = true }, .{ .token = 40, .value = -0.75 } };
    var processed = raw; // tf_bias_rows: the amount taken in fp32, rounded once to bf16; a ban is -inf
    var reference = raw; // main's row draw refuses -inf, so a ban there is bf16's lowest finite value
    for (entries) |e| {
        processed[e.token] = if (e.ban) 0xff80 else bf.bits(bf.value(raw[e.token]) - subtract(0, e));
        reference[e.token] = if (e.ban) 0xff7f else processed[e.token];
    }
    const mask = [_]u32{ 0xffff_fffe, 0xffff_ffff }; // token 0 disallowed
    for ([_]u32{ 0, 1, 20 }) |top_k| for (0..32) |position| {
        const s: sampling.Sampling = .{ .seed = 99, .temperature = 0.8, .top_k = top_k, .top_p = 0.95 };
        for ([_]?sampling.Sampling{ null, s }) |settings| for ([_][]const u32{ &.{}, &mask }) |m| {
            const want = try rowDraw(a, &reference, settings, position, m);
            try std.testing.expect(want != 17);
            try std.testing.expectEqual(want, try draw(a, &raw, settings, position, m, &entries));
            try std.testing.expectEqual(want, try draw(a, &processed, settings, position, m, &.{}));
        };
    };
    try std.testing.expectEqual(@as(?Entry, null), find(&entries, 4));
    try std.testing.expectEqual(@as(u32, 40), find(&entries, 40).?.token);
}

fn allocationCase(a: std.mem.Allocator) !void {
    _ = try draw(a, &.{ 0x3f80, 0x4000 }, null, 0, &.{}, &.{.{ .token = 0, .value = 3 }});
}

test "bias host allocations release both arrays on every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}
