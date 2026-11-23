//! Ring rows the drafter's last ingest overwrote, put back where a later commit cuts that ingest's window.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const plan_ = @import("dspark_plan.zig");

const R = plan_.max_rows;

/// The rows [start, start + rows) of a slot its last ingest wrote.
const Span = struct { start: u64, rows: u32 };

pub const Undo = struct {
    a: std.mem.Allocator,
    c: *const Config,
    rings: []const mtl.Buffer,
    /// each slot's ring rows as they were before its last ingest: [slots][R][stages][head_dim] bf16
    saved: []u8,
    /// each slot's last ingest (null: nothing to put back)
    spans: []?Span,
    /// rows put back
    restored: u64 = 0,

    pub fn init(a: std.mem.Allocator, c: *const Config, rings: []const mtl.Buffer, slots: u32) !Undo {
        const saved = try a.alloc(u8, @as(usize, slots) * R * rings.len * rowBytes(c));
        errdefer a.free(saved);
        const spans = try a.alloc(?Span, slots);
        @memset(spans, null);
        return .{ .a = a, .c = c, .rings = rings, .saved = saved, .spans = spans };
    }

    pub fn deinit(u: *Undo) void {
        u.a.free(u.saved);
        u.a.free(u.spans);
    }

    fn rowBytes(c: *const Config) usize {
        return @as(usize, c.head_dim) * 2;
    }

    /// Stage `s`'s ring row of `slot` at position `p`.
    fn ring(u: *const Undo, s: usize, slot: u32, p: u64) []u8 {
        const at = @as(usize, plan_.ringRow(u.c, slot, p)) * rowBytes(u.c);
        return u.rings[s].contents()[at..][0..rowBytes(u.c)];
    }

    /// Stage `s`'s kept row `r` (the span's r-th position) of `slot`.
    fn kept(u: *const Undo, s: usize, slot: u32, r: usize) []u8 {
        const row = (@as(usize, slot) * R + r) * u.rings.len + s;
        return u.saved[row * rowBytes(u.c) ..][0..rowBytes(u.c)];
    }

    /// Before an ingest of [start, start + n), the rings idle: those rows kept (more than R rows are committed ones).
    pub fn save(u: *Undo, slot: u32, start: u64, n: u32) void {
        if (n == 0 or n > R) {
            u.spans[slot] = null;
            return;
        }
        for (0..n) |r| for (0..u.rings.len) |s| @memcpy(u.kept(s, slot, r), u.ring(s, slot, start + r));
        u.spans[slot] = .{ .start = start, .rows = n };
    }

    /// The last ingest wrote only its first `m` rows into the rings (its later ones went elsewhere): those kept.
    pub fn shrink(u: *Undo, slot: u32, m: u32) void {
        const sp = u.spans[slot] orelse return;
        u.spans[slot] = if (m > 0) .{ .start = sp.start, .rows = @min(m, sp.rows) } else null;
    }

    /// Whether `settle(slot, end)` writes rows (the GPU must be done with the rings first).
    pub fn pending(u: *const Undo, slot: u32, end: u64) bool {
        const sp = u.spans[slot] orelse return false;
        return end < sp.start + sp.rows;
    }

    /// Before work from position `end` on: the last ingest's rows from `end` put back, those before it still kept.
    pub fn settle(u: *Undo, slot: u32, end: u64) void {
        const sp = u.spans[slot] orelse return;
        const stop = sp.start + sp.rows;
        if (end >= stop) return;
        const from = @max(end, sp.start);
        for (from..stop) |p| for (0..u.rings.len) |s| @memcpy(u.ring(s, slot, p), u.kept(s, slot, p - sp.start));
        u.restored += stop - from;
        u.spans[slot] = if (from > sp.start) .{ .start = sp.start, .rows = @intCast(from - sp.start) } else null;
    }

    /// A new request on the slot: no earlier row of it matters.
    pub fn forget(u: *Undo, slot: u32) void {
        if (slot < u.spans.len) u.spans[slot] = null;
    }
};
