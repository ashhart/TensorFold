//! Presence and frequency penalties as (row, token, amount) entries, applied as CUDA's sample_penalties.cuh does.
const std = @import("std");
const Stream = @import("stream.zig").Stream;
const Window = @import("backend.zig").Window;
const Penalties = @import("sampling.zig").Penalties;

/// A window's rows at most here: its pending row and its drafts.
pub const most_rows = 256;

/// One forward's entries; a (row, token) pair appears once, so a kernel thread an entry never races another.
pub const Entries = struct {
    rows: std.ArrayList(u32) = .empty,
    toks: std.ArrayList(u32) = .empty,
    subs: std.ArrayList(f32) = .empty,
    /// The current window's counts: the reply's tokens before it, then its pending token.
    counts: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty,

    pub fn deinit(self: *Entries, gpa: std.mem.Allocator) void {
        self.rows.deinit(gpa);
        self.toks.deinit(gpa);
        self.subs.deinit(gpa);
        self.counts.deinit(gpa);
    }

    pub fn len(self: *const Entries) usize {
        return self.rows.items.len;
    }

    pub fn clear(self: *Entries) void {
        self.rows.clearRetainingCapacity();
        self.toks.clearRetainingCapacity();
        self.subs.clearRetainingCapacity();
    }

    /// A window's reply so far: `stream`'s committed tokens before `cache_len`, then `next`, the token at `cache_len`.
    pub fn window(self: *Entries, gpa: std.mem.Allocator, stream: *const Stream, cache_len: usize, next: u32) !void {
        self.counts.clearRetainingCapacity();
        const context = stream.context.items;
        if (cache_len < stream.prompt_len or cache_len > context.len) return error.PenaltyPositionOutOfStep;
        for (context[stream.prompt_len..cache_len]) |t| try bump(gpa, &self.counts, t, 1);
        try bump(gpa, &self.counts, next, 1);
    }

    /// Row `at`'s entries under `p`: the window's counts with `path` (the drafts from the window's first row to it).
    pub fn row(self: *Entries, gpa: std.mem.Allocator, at: u32, p: Penalties, path: []const u32) !void {
        for (path) |t| try bump(gpa, &self.counts, t, 1);
        defer for (path) |t| bump(gpa, &self.counts, t, -1) catch unreachable;
        const n = self.counts.count();
        try self.rows.ensureUnusedCapacity(gpa, n);
        try self.toks.ensureUnusedCapacity(gpa, n);
        try self.subs.ensureUnusedCapacity(gpa, n);
        for (self.counts.keys(), self.counts.values()) |t, c| if (c > 0) {
            self.rows.appendAssumeCapacity(at);
            self.toks.appendAssumeCapacity(t);
            self.subs.appendAssumeCapacity(p.of(c));
        };
    }

    /// A verify window's rows from row `first`: `drafts` are its rows 1.. (held or host), drawn at `w.positions`.
    pub fn stageWindow(self: *Entries, gpa: std.mem.Allocator, w: Window, drafts: []const u32, first: usize) !void {
        if (!w.stream.penalties.active()) return;
        const rows = w.rows();
        if (rows > most_rows or drafts.len + 1 != rows or w.positions.len != rows) return error.PenaltyWindowOutOfStep;
        // a row's draw is keyed one past the cache it reads: row 0's cache is the reply before the pending token
        if (w.positions[0] == 0) return error.PenaltyPositionOutOfStep;
        try self.window(gpa, w.stream, @intCast(w.positions[0] - 1), w.pending);
        var buf: [most_rows]u32 = undefined;
        for (0..rows) |r| try self.row(gpa, @intCast(first + r), w.stream.penalties, pathOf(&buf, drafts, w.parents, r));
    }

    /// A one-token round's row `at`: the reply before `cache_len`, then `fed`, the token the round feeds.
    pub fn stageOne(self: *Entries, gpa: std.mem.Allocator, stream: *const Stream, cache_len: usize, fed: u32, at: u32) !void {
        if (!stream.penalties.active()) return;
        try self.window(gpa, stream, cache_len, fed);
        try self.row(gpa, at, stream.penalties, &.{});
    }

    /// Every entry of row `at` taken from `logits` (that row's bf16 words), as the GPU kernels do.
    pub fn applyRow(self: *const Entries, at: u32, logits: []u16) void {
        for (self.rows.items, self.toks.items, self.subs.items) |r, t, s| if (r == at) {
            logits[t] = lower(logits[t], s);
        };
    }
};

fn bump(gpa: std.mem.Allocator, counts: *std.AutoArrayHashMapUnmanaged(u32, u32), t: u32, by: i2) !void {
    const gop = try counts.getOrPut(gpa, t);
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* = if (by > 0) gop.value_ptr.* + 1 else gop.value_ptr.* - 1;
}

/// Row `r`'s path in a window of `drafts` (row r > 0 feeds drafts[r - 1]) under `parents` (null: a chain).
pub fn pathOf(out: []u32, drafts: []const u32, parents: ?[]const i32, r: usize) []const u32 {
    var n: usize = 0;
    var x = r;
    while (x > 0) {
        out[n] = drafts[x - 1];
        n += 1;
        x = if (parents) |ps| @intCast(ps[x]) else x - 1;
    }
    std.mem.reverse(u32, out[0..n]);
    return out[0..n];
}

/// A bf16 logit less `amount` in fp32, rounded once to nearest even: __float2bfloat16 and Metal's bfloat(float).
pub fn lower(word: u16, amount: f32) u16 {
    const x = @as(f32, @bitCast(@as(u32, word) << 16)) - amount;
    if (std.math.isNan(x)) return 0x7fc0;
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

fn fakeStream(context: []const u32, prompt_len: usize, p: Penalties) Stream {
    var s: Stream = undefined;
    s.context = .fromOwnedSlice(@constCast(context));
    s.prompt_len = prompt_len;
    s.penalties = p;
    return s;
}

fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

test "penalty rows: a row counts the reply, the pending token and its own path once each" {
    const gpa = std.testing.allocator;
    var e: Entries = .{};
    defer e.deinit(gpa);
    // prompt 9 9 (not counted), reply 5 7 5, pending 7 at cache_len 5
    const p: Penalties = .{ .presence = 1, .frequency = 0.5 };
    const s = fakeStream(&.{ 9, 9, 5, 7, 5, 7 }, 2, p);
    try e.window(gpa, &s, 5, 7);
    try e.row(gpa, 0, p, &.{});
    try e.row(gpa, 1, p, &.{ 5, 3 });
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 1, 1 }, e.rows.items);
    try std.testing.expectEqualSlices(u32, &.{ 5, 7, 5, 7, 3 }, e.toks.items);
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2.5, 2, 1.5 }, e.subs.items);
}

test "penalty rows: a verify window keys row 0 one past its cache, and a tree row follows its parents" {
    const gpa = std.testing.allocator;
    var e: Entries = .{};
    defer e.deinit(gpa);
    const p: Penalties = .{ .frequency = 1 };
    var s = fakeStream(&.{ 1, 4, 6 }, 1, p); // reply 4, pending 6 at cache_len 2
    const drafts = [_]u32{ 8, 9, 4 };
    const parents = [_]i32{ -1, 0, 0, 2 }; // row 3 hangs off row 2, whose draft is 9
    const w: Window = .{ .stream = &s, .pending = 6, .held = 0, .tokens = &drafts, .parents = &parents, .positions = &.{ 3, 4, 4, 5 } };
    try e.stageWindow(gpa, w, &drafts, 10);
    // row 0: {4:1, 6:1}; row 1: + 8; row 2: + 9; row 3: + 9, 4 -> 4 counted twice
    try std.testing.expectEqualSlices(u32, &.{ 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13 }, e.rows.items);
    try std.testing.expectEqualSlices(u32, &.{ 4, 6, 4, 6, 8, 4, 6, 9, 4, 6, 9 }, e.toks.items);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1 }, e.subs.items);
    // zero penalties stage nothing: the forward runs no penalty pass and keeps main's tokens
    e.clear();
    s.penalties = .{};
    try e.stageWindow(gpa, w, &drafts, 0);
    try e.stageOne(gpa, &s, 2, 6, 0);
    try std.testing.expectEqual(@as(usize, 0), e.len());
}

test "penalty rows: a position before the reply or past the committed tokens is refused" {
    const gpa = std.testing.allocator;
    var e: Entries = .{};
    defer e.deinit(gpa);
    const s = fakeStream(&.{ 1, 2, 3 }, 2, .{ .presence = 1 });
    try std.testing.expectError(error.PenaltyPositionOutOfStep, e.window(gpa, &s, 1, 0));
    try std.testing.expectError(error.PenaltyPositionOutOfStep, e.window(gpa, &s, 4, 0));
}

test "penalty rows: lower rounds fp32 to the nearest bf16, ties to even, and keeps -inf" {
    var rng = std.Random.DefaultPrng.init(11);
    const r = rng.random();
    for (0..20000) |_| {
        const word: u16 = @intCast(r.int(u16) & 0x7f7f | (if (r.boolean()) @as(u16, 0x8000) else 0)); // finite
        const amount: f32 = @floatCast((r.float(f64) - 0.25) * 6);
        const exact: f64 = @as(f32, @floatCast(value(word))) - amount; // the fp32 difference
        const got = lower(word, amount);
        if (got & 0x7fff == 0) continue; // a zero's neighbours cross the sign
        // no neighbour of the result lies nearer the fp32 difference (a tie keeps an even mantissa)
        for ([_]i32{ -1, 1 }) |step| {
            const other: u16 = @intCast(@as(i32, got) + step);
            if (other & 0x7f80 == 0x7f80) continue; // past the finite range
            const d0 = @abs(value(got) - exact);
            const d1 = @abs(value(other) - exact);
            try std.testing.expect(d0 < d1 or (d0 == d1 and got & 1 == 0));
        }
    }
    try std.testing.expectEqual(@as(u16, 0xff80), lower(0xff80, 1.5)); // a masked token stays masked
    try std.testing.expectEqual(@as(u16, 0x3f80), lower(0x3f80, 0)); // 1.0 less nothing
    try std.testing.expectEqual(@as(u16, 0xbf00), lower(0x3f80, 1.5)); // 1.0 - 1.5 = -0.5
}

test "penalty rows: applyRow takes each of its row's entries once" {
    const gpa = std.testing.allocator;
    var e: Entries = .{};
    defer e.deinit(gpa);
    const p: Penalties = .{ .presence = 1.5, .frequency = 0.5 };
    const s = fakeStream(&.{ 0, 2, 2 }, 1, p); // reply 2, pending 2: count 2
    try e.stageOne(gpa, &s, 2, 2, 3);
    var logits = [_]u16{ 0x4000, 0x4000, 0x4000, 0x4000 }; // 2.0 each
    e.applyRow(4, &logits); // another row's: unchanged
    try std.testing.expectEqualSlices(u16, &.{ 0x4000, 0x4000, 0x4000, 0x4000 }, &logits);
    e.applyRow(3, &logits);
    try std.testing.expectEqualSlices(u16, &.{ 0x4000, 0x4000, 0xbf00, 0x4000 }, &logits); // 2 - 2.5 = -0.5
}

/// Row `at`'s (token, amount bits) pairs in token order.
fn sortedRow(gpa: std.mem.Allocator, e: *const Entries, at: u32) ![]u64 {
    var out: std.ArrayList(u64) = .empty;
    errdefer out.deinit(gpa);
    for (e.rows.items, e.toks.items, e.subs.items) |r, t, s| if (r == at) try out.append(gpa, @as(u64, t) << 32 | @as(u32, @bitCast(s)));
    std.mem.sort(u64, out.items, {}, std.sort.asc(u64));
    return out.toOwnedSlice(gpa);
}

test "penalty rows: a verify row stages what the one-token round at its position stages, chain or tree" {
    const gpa = std.testing.allocator;
    const p: Penalties = .{ .presence = 0.75, .frequency = 0.25 };
    const committed = [_]u32{ 9, 5, 7 }; // prompt 9, reply 5 7; pending 5 at cache_len 3
    var s = fakeStream(&committed, 1, p);
    const drafts = [_]u32{ 7, 3, 4 };
    const parents = [_]i32{ -1, 0, 1, 0 }; // rows 1 and 2 a chain under row 0, row 3 a sibling of row 1
    const w: Window = .{ .stream = &s, .pending = 5, .held = 0, .tokens = &drafts, .parents = &parents, .positions = &.{ 4, 5, 6, 5 } };
    var drafted: Entries = .{};
    defer drafted.deinit(gpa);
    try drafted.stageWindow(gpa, w, &drafts, 0);
    var buf: [most_rows]u32 = undefined;
    for (0..4) |r| {
        // undrafted, row r's draw follows the committed tokens, the pending one and its path, its last token fed
        var ctx: std.ArrayList(u32) = .empty;
        defer ctx.deinit(gpa);
        try ctx.appendSlice(gpa, &committed);
        try ctx.append(gpa, 5);
        try ctx.appendSlice(gpa, pathOf(&buf, &drafts, &parents, r));
        const fed = ctx.pop().?;
        const alone = fakeStream(ctx.items, 1, p);
        var plain: Entries = .{};
        defer plain.deinit(gpa);
        try plain.stageOne(gpa, &alone, ctx.items.len, fed, @intCast(r));
        const want = try sortedRow(gpa, &plain, @intCast(r));
        defer gpa.free(want);
        const got = try sortedRow(gpa, &drafted, @intCast(r));
        defer gpa.free(got);
        try std.testing.expect(want.len > 0);
        try std.testing.expectEqualSlices(u64, want, got);
    }
}
