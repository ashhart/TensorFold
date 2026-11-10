//! The provider walker preserves the normal walker's nodes, ties and FP64 probability words.
const std = @import("std");
const tree = @import("draft_tree.zig");
const Sampling = @import("sampling.zig").Sampling;
const t = std.testing;
const R = 7;
const W = 43;

const Provider = struct {
    c: tree.Candidates,
    changed: bool = false,
    pub fn choices(self: Provider, row: usize, previous: u32) tree.proposals.Choices {
        var result = tree.proposals.baseChoices(self.c, row, previous);
        if (self.changed) for (result.tokens[0..result.len]) |*token| {
            token.* ^= 16;
        };
        return result;
    }
};

fn fill(words: []u8, topk: usize) void {
    @memset(words, 0);
    for (0..R) |r| for (0..topk) |j| {
        const token: u32 = @intCast(j + 1);
        const logit: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((r + j) % 5)) - 2)) / 4;
        std.mem.writeInt(u32, words[(r * W + 2 + j) * 4 ..][0..4], token, .little);
        std.mem.writeInt(u32, words[(r * W + 22 + j) * 4 ..][0..4], @bitCast(logit), .little);
    };
}

test "draft providers: normal candidates preserve every output byte across sampling, tables, budgets and ties" {
    var storage: [R * W * 4 + 1]u8 = undefined;
    const words = storage[1..]; // Candidate row words need no host alignment.
    var projected: [R * 2]u16 = undefined;
    for (&projected, 0..) |*value, i| value.* = if (i % 2 == 0) 0x3f00 else 0xbf80;
    var predecessor: [32 * 2]u16 = undefined;
    var successor: [32 * 2]u16 = undefined;
    for (&predecessor, &successor, 0..) |*p, *s, i| {
        p.* = if (i % 3 == 0) 0x3f80 else 0x3f00;
        s.* = if (i % 2 == 0) 0x3f00 else 0xbf00;
    }
    var table: [R * 20 * 20]f32 = undefined;
    for ([_]usize{ 1, 2, 4, 7, 20 }) |topk| {
        fill(words, topk);
        for (table[0 .. R * topk * topk], 0..) |*value, i|
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 8;
        for ([_]bool{ false, true }) |tabular| for ([_]?Sampling{ null, .{ .seed = 19, .temperature = 0.7 }, .{ .seed = 19, .temperature = 1.3 } }) |sampling| {
            const candidates: tree.Candidates = .{
                .words = words,
                .projected = if (tabular) std.mem.sliceAsBytes(table[0 .. R * topk * topk]) else std.mem.sliceAsBytes(&projected),
                .topk = topk,
                .rank = 2,
                .predecessor = &predecessor,
                .successor = &successor,
                .sampling = sampling,
                .position = 97,
                .table = tabular,
            };
            for ([_]usize{ 1, 2, R }) |rows| for ([_]usize{ 0, 1, 5, 15, 63 }) |budget| {
                var a_tokens: [64]u32 = @splat(0xa5a5a5a5);
                var b_tokens = a_tokens;
                var a_parents: [64]i32 = @splat(-77);
                var b_parents = a_parents;
                var a_own: [64]f64 = @splat(123.25);
                var b_own = a_own;
                const a = tree.grow(64, candidates, 0, rows, budget, &a_tokens, &a_parents, &a_own);
                const b = tree.proposals.growWith(64, Provider{ .c = candidates }, 0, rows, budget, &b_tokens, &b_parents, &b_own);
                try t.expectEqual(a, b);
                try t.expectEqualSlices(u32, &a_tokens, &b_tokens);
                try t.expectEqualSlices(i32, &a_parents, &b_parents);
                try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&a_own), std.mem.sliceAsBytes(&b_own));
                if (budget > 0) try t.expect(a > 0);
            };
        };
    }
}

test "draft providers: altered candidate tokens fail the byte replacement check" {
    var words: [R * W * 4]u8 = undefined;
    fill(&words, 2);
    const zeros: [64]u16 = @splat(0);
    const candidates: tree.Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(zeros[0 .. R * 2]), .topk = 2, .rank = 2, .predecessor = &zeros, .successor = &zeros };
    var base: [1]u32 = undefined;
    var changed: [1]u32 = undefined;
    var parent: [1]i32 = undefined;
    var own: [1]f64 = undefined;
    try t.expectEqual(@as(usize, 1), tree.grow(64, candidates, 0, 1, 1, &base, &parent, &own));
    try t.expectEqual(@as(usize, 1), tree.proposals.growWith(64, Provider{ .c = candidates, .changed = true }, 0, 1, 1, &changed, &parent, &own));
    try t.expect(!std.mem.eql(u32, &base, &changed));
}
