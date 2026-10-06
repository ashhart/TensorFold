//! decode.CopyIndex: copy drafts' search over a context, by each token's positions (longest match, or the 8-token chain).

const std = @import("std");

pub const min_match = 8;

/// A match's length, and where the tokens that followed it start.
pub const Match = struct { n: usize = 0, at: usize = 0 };

pub const CopyIndex = struct {
    gpa: std.mem.Allocator,
    ctx: std.ArrayList(u32) = .empty,
    seen: std.AutoHashMapUnmanaged(u32, std.ArrayList(u32)) = .empty, // each token's positions in ctx

    pub fn init(gpa: std.mem.Allocator, context: []const u32) !CopyIndex {
        var x: CopyIndex = .{ .gpa = gpa };
        errdefer x.deinit();
        try x.extend(context);
        return x;
    }

    pub fn deinit(x: *CopyIndex) void {
        var it = x.seen.valueIterator();
        while (it.next()) |v| v.deinit(x.gpa);
        x.seen.deinit(x.gpa);
        x.ctx.deinit(x.gpa);
    }

    pub fn extend(x: *CopyIndex, tokens: []const u32) !void {
        for (tokens) |t| {
            const got = try x.seen.getOrPut(x.gpa, t);
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.append(x.gpa, @intCast(x.ctx.items.len));
            try x.ctx.append(x.gpa, t);
        }
    }

    /// The longest earlier match of the context's last tokens, up to `max_n` deep (the latest of equal length).
    pub fn longest(x: *const CopyIndex, max_n: usize) Match {
        const ctx = x.ctx.items;
        var best: Match = .{};
        if (ctx.len < 2 or max_n < 1) return best;
        const last = ctx.len - 1;
        const list = x.seen.get(ctx[last]) orelse return best;
        var i = list.items.len;
        while (i > 0) {
            i -= 1;
            const p: usize = list.items[i];
            if (p >= last) continue;
            var n: usize = 1;
            while (n < max_n and p >= n and ctx[p - n] == ctx[last - n]) n += 1;
            if (n > best.n) {
                best = .{ .n = n, .at = p + 1 };
                if (n == max_n) break;
            }
        }
        return best;
    }

    /// Up to `out.len` tokens that followed the latest earlier copy of the last 8 (none shorter than 8), copied out.
    pub fn chain(x: *const CopyIndex, out: []u32) []const u32 {
        const max_nodes = out.len;
        const ctx = x.ctx.items;
        if (ctx.len < 2 * min_match or max_nodes < 1) return &.{};
        const key = ctx[ctx.len - min_match ..];
        const list = x.seen.get(key[min_match - 1]) orelse return &.{};
        var best: []const u32 = &.{};
        var i = list.items.len;
        while (i > 0) {
            i -= 1;
            const p: usize = list.items[i]; // the copy's last token
            if (p + 1 < min_match or p + 1 >= ctx.len) continue;
            if (!std.mem.eql(u32, ctx[p + 1 - min_match .. p + 1], key)) continue;
            const cont = ctx[p + 1 .. @min(ctx.len, p + 1 + max_nodes)];
            if (cont.len > best.len) {
                best = cont;
                if (best.len == max_nodes) break;
            }
        }
        if (best.len < min_match) return &.{};
        @memcpy(out[0..best.len], best);
        return out[0..best.len];
    }
};

test "copies the continuation of the latest earlier match" {
    const a = std.testing.allocator;
    var seq: [40]u32 = undefined;
    for (&seq, 0..) |*v, i| v.* = @intCast(i % 20);
    var x = try CopyIndex.init(a, &seq);
    defer x.deinit();
    var buf: [15]u32 = undefined;
    const got = x.chain(&buf);
    try std.testing.expectEqual(@as(usize, 15), got.len);
    try std.testing.expectEqual(@as(u32, 0), got[0]);
    try std.testing.expectEqual(@as(usize, 0), x.chain(buf[0..0]).len);
}

// Plain scans of the whole context: what `longest` and `chain` must return.
fn scanLongest(ctx: []const u32, max_n: usize) Match {
    var best: Match = .{};
    if (ctx.len < 2 or max_n < 1) return best;
    const last = ctx.len - 1;
    var p = last;
    while (p > 0) {
        p -= 1;
        var n: usize = 0;
        while (n < max_n and p >= n and ctx[p - n] == ctx[last - n]) n += 1;
        if (n > best.n) {
            best = .{ .n = n, .at = p + 1 };
            if (n == max_n) break;
        }
    }
    return best;
}

fn scanChain(ctx: []const u32, max_nodes: usize) []const u32 {
    if (ctx.len < 2 * min_match or max_nodes < 1) return &.{};
    const key = ctx[ctx.len - min_match ..];
    var best: []const u32 = &.{};
    var s = ctx.len - min_match;
    while (s > 0) {
        s -= 1;
        if (!std.mem.eql(u32, ctx[s .. s + min_match], key)) continue;
        const cont = ctx[s + min_match .. @min(ctx.len, s + min_match + max_nodes)];
        if (cont.len > best.len) {
            best = cont;
            if (best.len == max_nodes) break;
        }
    }
    return if (best.len < min_match) &.{} else best;
}

test "longest and chain match plain scans as the context grows" {
    const a = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    for (0..200) |_| {
        const symbols = r.intRangeAtMost(u32, 1, 4);
        var x = try CopyIndex.init(a, &.{});
        defer x.deinit();
        for (0..r.intRangeAtMost(usize, 1, 300)) |_| {
            try x.extend(&.{r.uintLessThan(u32, symbols)});
            const ctx = x.ctx.items;
            for ([_]usize{ 1, 3, 8 }) |max_n| {
                const want = scanLongest(ctx, max_n);
                const got = x.longest(max_n);
                try std.testing.expectEqual(want.n, got.n);
                if (want.n > 0) try std.testing.expectEqual(want.at, got.at);
            }
            var buf: [20]u32 = undefined;
            for ([_]usize{ 1, 9, 20 }) |m| {
                const want = scanChain(ctx, m);
                const got = x.chain(buf[0..m]);
                try std.testing.expectEqualSlices(u32, want, got);
            }
        }
    }
}
