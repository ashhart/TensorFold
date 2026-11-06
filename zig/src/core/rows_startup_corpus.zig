//! Synthetic kept states for the startup timing: chat conversations of 2K to 64K tokens and a 27B-like image.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Allocator = std.mem.Allocator;
const Span = [2]u32;

const vocab = 151_000;
const im_start = 151_644;
const im_end = 151_645;
const nl = 198;
const roles = [3]u32{ 8948, 872, 77091 }; // system, user, assistant

/// The 27B's 16 MiB of keys and values a block, as 128 row parts of 512 bytes (zig/src/native/qwen27_disk.zig).
pub const kv_parts = 128;
pub const kv_row = 512;
/// The drafter's taps ring: its 2,048-row window (dflash/config.zig) of five 5,120-wide bf16 taps.
pub const ring_window = 2048;
pub const ring_row = 51_200;

/// The image a 27B state with its drafter lays out: the committed end, its own states, rows by head, the taps ring.
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter };

var family_word: u8 = 0;

/// A family the store can open with: its image only (startup copies nothing in or out).
pub fn snapshots() pc.Snapshots {
    return .{ .ptr = &family_word, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &image } };
}

fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    try out.append(a, .{ .bytes = 8 });
    try out.append(a, .{ .bytes = 64 });
    for (0..kv_parts) |_| try out.append(a, .{ .bytes = @as(u64, at) * kv_row, .row = kv_row });
    try out.append(a, .{ .bytes = @as(u64, @min(at, ring_window)) * ring_row, .row = ring_row, .window = ring_window });
}

fn bytes(_: *anyopaque, at: u32) u64 {
    return 72 + @as(u64, at) * kv_parts * kv_row + @as(u64, @min(at, ring_window)) * ring_row;
}
fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
    return error.Unsupported;
}
fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
    return error.Unsupported;
}
fn drop(_: *anyopaque, _: pc.Saved) void {}
fn gather(_: *anyopaque, _: pc.Saved, _: []const rows.image.Range, _: []u8) anyerror!void {
    return error.Unsupported;
}
fn blank(_: *anyopaque, _: u32) anyerror!pc.Saved {
    return error.Unsupported;
}
fn scatter(_: *anyopaque, _: pc.Saved, _: []const rows.image.Range, _: []const u8) anyerror!void {
    return error.Unsupported;
}

/// A kept position of the conversation in `Turns.tokens`, and how many of its decoded spans lie below it.
pub const Mark = struct { at: u32, spans: u32, next: u2 };

/// Conversations one after another: 8 shared system prompts, pasted documents, forks of the previous conversation.
pub const Turns = struct {
    a: Allocator,
    prng: std.Random.DefaultPrng,
    shared: [8][]u32 = undefined,
    tokens: std.ArrayList(u32) = .empty,
    spans: std.ArrayList(Span) = .empty,
    marks: std.ArrayList(Mark) = .empty, // the last conversation's new kept positions (a fork's are past its prefix)
    every: std.ArrayList(Mark) = .empty, // all of its kept positions, the prefix's too (the next fork's choices)

    pub fn init(a: Allocator, seed: u64) !Turns {
        var t: Turns = .{ .a = a, .prng = .init(seed) };
        var made: usize = 0;
        errdefer for (t.shared[0..made]) |s| a.free(s);
        for (&t.shared) |*s| {
            s.* = try a.alloc(u32, t.between(1024, 6143));
            t.words(s.*);
            made += 1;
        }
        return t;
    }

    pub fn deinit(t: *Turns) void {
        for (t.shared) |s| t.a.free(s);
        t.tokens.deinit(t.a);
        t.spans.deinit(t.a);
        t.marks.deinit(t.a);
        t.every.deinit(t.a);
    }

    fn between(t: *Turns, lo: u32, hi: u32) u32 {
        return t.prng.random().intRangeAtMost(u32, lo, hi);
    }

    fn words(t: *Turns, out: []u32) void {
        const r = t.prng.random();
        for (out) |*w| w.* = r.uintLessThan(u32, vocab);
    }

    /// Log-uniform from `lo` to `hi`.
    fn logUniform(t: *Turns, lo: f64, hi: f64) u32 {
        return @intFromFloat(lo * std.math.pow(f64, hi / lo, t.prng.random().float(f64)));
    }

    /// One turn: its header, `n` words and its end; an assistant's words and end are rows the server decoded.
    fn turn(t: *Turns, role: u2, n: u32) !void {
        try t.tokens.appendSlice(t.a, &.{ im_start, roles[role], nl });
        const from: u32 = @intCast(t.tokens.items.len);
        try t.tokens.resize(t.a, from + n);
        t.words(t.tokens.items[from..]);
        try t.tokens.append(t.a, im_end);
        if (role == 2) try t.spans.append(t.a, .{ from, from + n + 1 });
        try t.tokens.append(t.a, nl);
    }

    /// The next conversation: one in 10 continues a kept state of the previous one (a regenerate or an edit).
    pub fn next(t: *Turns) !void {
        var goal = t.logUniform(2048, 65536);
        t.marks.clearRetainingCapacity();
        var role: u2 = 1;
        var opened = false;
        if (t.every.items.len > 0 and t.prng.random().uintLessThan(u32, 10) == 0) {
            const m = t.every.items[t.prng.random().uintLessThan(usize, t.every.items.len)];
            t.tokens.shrinkRetainingCapacity(m.at);
            t.spans.shrinkRetainingCapacity(m.spans);
            var kept: usize = 0;
            while (kept < t.every.items.len and t.every.items[kept].at <= m.at) kept += 1;
            t.every.shrinkRetainingCapacity(kept);
            role = m.next;
            opened = true;
            goal = @max(goal, m.at + 1); // a fork adds a turn at least
        } else {
            t.tokens.clearRetainingCapacity();
            t.spans.clearRetainingCapacity();
            t.every.clearRetainingCapacity();
            const own = t.prng.random().uintLessThan(u32, 4) == 0;
            const k = t.prng.random().uintLessThan(usize, t.shared.len);
            try t.tokens.appendSlice(t.a, &.{ im_start, roles[0], nl });
            if (own) {
                const from = t.tokens.items.len;
                try t.tokens.resize(t.a, from + t.between(1024, 6143));
                t.words(t.tokens.items[from..]);
            } else try t.tokens.appendSlice(t.a, t.shared[k]);
            try t.tokens.appendSlice(t.a, &.{ im_end, nl });
        }
        while (t.tokens.items.len < goal) {
            const doc = !opened and role == 1 and t.prng.random().uintLessThan(u32, 4) == 0;
            const n = if (doc) @min(t.between(4096, 32767), @max(goal -| @as(u32, @intCast(t.tokens.items.len)), 256)) else t.between(256, 4095);
            try t.turn(role, n);
            opened = true;
            role = if (role == 1) 2 else 1;
            const at: u32 = @intCast(t.tokens.items.len);
            if (at < 2048) continue;
            const m: Mark = .{ .at = at, .spans = @intCast(t.spans.items.len), .next = role };
            try t.marks.append(t.a, m);
            try t.every.append(t.a, m);
        }
    }
};

test "conversations keep canonical spans and strictly growing marks past 2,048 tokens" {
    var t = try Turns.init(std.testing.allocator, 3);
    defer t.deinit();
    for (0..40) |_| {
        try t.next();
        var last: u32 = 0;
        for (t.marks.items) |m| {
            try std.testing.expect(m.at >= 2048 and m.at > last and m.at <= t.tokens.items.len);
            try @import("cache_modes.zig").validate(t.spans.items[0..m.spans]);
            if (m.spans > 0) try std.testing.expect(t.spans.items[m.spans - 1][1] < m.at);
            last = m.at;
        }
    }
}
