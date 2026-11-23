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
    /// each slot's ring rows as they were before its last ingest: [slots][R][stages][head_dim] bf16 (shared GPU buffer)
    saved: mtl.Buffer,
    /// each slot's last ingest (null: nothing to put back)
    spans: []?Span,
    /// rows put back
    restored: u64 = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, c: *const Config, rings: []const mtl.Buffer, slots: u32) !Undo {
        const bytes = @max(256, @as(usize, slots) * R * rings.len * rowBytes(c));
        const saved = try device.buffer(bytes, mtl.ResourceOptions.shared);
        errdefer saved.deinit();
        @memset(saved.contents()[0..bytes], 0);
        const spans = try a.alloc(?Span, slots);
        @memset(spans, null);
        return .{ .a = a, .c = c, .rings = rings, .saved = saved, .spans = spans };
    }

    pub fn deinit(u: *Undo) void {
        u.saved.deinit();
        u.a.free(u.spans);
    }

    fn rowBytes(c: *const Config) usize {
        return @as(usize, c.head_dim) * 2;
    }

    fn ringOff(u: *const Undo, slot: u32, p: u64) usize {
        return @as(usize, plan_.ringRow(u.c, slot, p)) * rowBytes(u.c);
    }

    fn keptOff(u: *const Undo, s: usize, slot: u32, r: usize) usize {
        const row = (@as(usize, slot) * R + r) * u.rings.len + s;
        return row * rowBytes(u.c);
    }

    fn ring(u: *const Undo, s: usize, slot: u32, p: u64) []u8 {
        return u.rings[s].contents()[u.ringOff(slot, p)..][0..rowBytes(u.c)];
    }

    fn kept(u: *const Undo, s: usize, slot: u32, r: usize) []u8 {
        return u.saved.contents()[u.keptOff(s, slot, r)..][0..rowBytes(u.c)];
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

    /// Encode GPU blit restoring the last ingest's rows from `end` on, then saving [start, start + n) before an ingest.
    pub fn encodeSettleAndSave(u: *Undo, cb: mtl.CommandBuffer, slot: u32, end: u64, start: u64, n: u32) void {
        const need_settle = u.pending(slot, end);
        const need_save = n > 0 and n <= R;
        if (!need_settle and !need_save) {
            if (n == 0 or n > R) u.spans[slot] = null;
            return;
        }
        const b = cb.blit();
        if (need_settle) u.blitSettle(b, slot, end);
        if (need_save) {
            const rb = rowBytes(u.c);
            for (0..n) |r| for (0..u.rings.len) |s| {
                b.copy(u.rings[s], u.ringOff(slot, start + r), u.saved, u.keptOff(s, slot, r), rb);
            };
            u.spans[slot] = .{ .start = start, .rows = n };
        } else {
            u.spans[slot] = null;
        }
        b.end();
    }

    /// Encode GPU blit restoring rows from `end` on into `cb` (no-op if nothing pending).
    pub fn encodeSettle(u: *Undo, cb: mtl.CommandBuffer, slot: u32, end: u64) bool {
        if (!u.pending(slot, end)) return false;
        const b = cb.blit();
        u.blitSettle(b, slot, end);
        b.end();
        return true;
    }

    /// Encode GPU blit restoring any pending undo rows across `asks` (returns true if any blit was encoded).
    pub fn encodeSettleAsks(u: *Undo, queue: mtl.Queue, asks: anytype) ?mtl.CommandBuffer {
        var any = false;
        for (asks) |a| if (a.slot < u.spans.len and u.pending(a.slot, a.start)) {
            any = true;
            break;
        };
        if (!any) return null;
        const cb = queue.commandBuffer();
        for (asks) |a| if (a.slot < u.spans.len) {
            _ = u.encodeSettle(cb, a.slot, a.start);
        };
        return cb;
    }

    fn blitSettle(u: *Undo, b: mtl.BlitEncoder, slot: u32, end: u64) void {
        const sp = u.spans[slot] orelse return;
        const stop = sp.start + sp.rows;
        if (end >= stop) return;
        const from = @max(end, sp.start);
        const rb = rowBytes(u.c);
        for (from..stop) |p| for (0..u.rings.len) |s| {
            b.copy(u.saved, u.keptOff(s, slot, p - sp.start), u.rings[s], u.ringOff(slot, p), rb);
        };
        u.restored += stop - from;
        u.spans[slot] = if (from > sp.start) .{ .start = sp.start, .rows = @intCast(from - sp.start) } else null;
    }

    /// The last ingest wrote only its first `m` rows into the rings (its later ones went elsewhere): those kept.
    pub fn shrink(u: *Undo, slot: u32, m: u32) void {
        const sp = u.spans[slot] orelse return;
        u.spans[slot] = if (m > 0) .{ .start = sp.start, .rows = @min(m, sp.rows) } else null;
    }

    /// Whether `settle(slot, end)` or `encodeSettle(slot, end)` writes rows.
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
