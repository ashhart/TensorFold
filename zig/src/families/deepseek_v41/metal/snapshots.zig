//! Kept prompt states on the host: a slot's regions up to its rows, its ids and DSpark taps, restored into any slot.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const kv = @import("kv.zig");

/// One kept state: the slot after `ids` (every row run), and the choice's logits when it ended a prompt.
pub const Entry = struct {
    ids: []u32,
    /// each region's window, pool rows, index key rows and carry, in layer order
    bytes: []u8,
    /// the taps ring: position p's row at p % ring rows
    taps: []u16,
    logits: ?[]f32,
    tick: u64 = 0,

    fn size(e: *const Entry) u64 {
        return e.ids.len * 4 + e.bytes.len + e.taps.len * 2 + if (e.logits) |l| l.len * 4 else 0;
    }

    fn free(e: *Entry, a: std.mem.Allocator) void {
        a.free(e.ids);
        a.free(e.bytes);
        a.free(e.taps);
        if (e.logits) |l| a.free(l);
        a.destroy(e);
    }
};

/// The pool and key rows a state of `len` positions has written at a KV source (and the row it may be filling).
pub fn poolRowsOf(c: *const Config, slot: *const kv.Slot, layer: u32, len: u64) usize {
    const all = kv.poolRows(c, layer, slot.limit);
    return @min(all, @as(usize, @intCast(len / c.compressRatio(layer))) + 1);
}

/// Every region's bytes a state of `len` positions reads, as (buffer, bytes) runs in order.
fn runs(c: *const Config, slot: *const kv.Slot, len: u64, out: *[4 * 64]Run) []Run {
    var n: usize = 0;
    for (slot.regions, 0..) |r, l| {
        out[n] = .{ .buf = r.window, .len = r.window.length() };
        n += 1;
        if (r.pool) |p| {
            const rows = poolRowsOf(c, slot, @intCast(l), len);
            out[n] = .{ .buf = p, .len = rows * c.head_dim * 2 };
            out[n + 1] = .{ .buf = r.keys.?, .len = rows * kv.index_key_bytes };
            n += 2;
        }
        if (r.carry) |k| {
            out[n] = .{ .buf = k, .len = k.length() };
            n += 1;
        }
    }
    return out[0..n];
}

const Run = struct { buf: mtl.Buffer, len: usize };

/// Zeroes what a state of up to `high` positions may have written: a reused slot reads as a new one.
pub fn clear(c: *const Config, slot: *const kv.Slot, high: u64) void {
    var buf: [4 * 64]Run = undefined;
    for (runs(c, slot, high, &buf)) |r| @memset(r.buf.contents()[0..r.len], 0);
    for (slot.regions) |*r| kv.resetCarry(r, c);
}

pub const Store = struct {
    a: std.mem.Allocator,
    /// bytes the entries may hold (0: no prompt cache)
    cap: u64,
    held: u64 = 0,
    tick: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,

    pub fn deinit(st: *Store) void {
        for (st.entries.items) |e| e.free(st.a);
        st.entries.deinit(st.a);
    }

    pub fn on(st: *const Store) bool {
        return st.cap > 0;
    }

    /// The longest entry that starts `ids`: shorter than them, or as long with the choice's logits.
    pub fn best(st: *Store, ids: []const u32) ?*Entry {
        var out: ?*Entry = null;
        for (st.entries.items) |e| {
            if (e.ids.len > ids.len or !std.mem.eql(u32, e.ids, ids[0..e.ids.len])) continue;
            if (e.ids.len == ids.len and e.logits == null) continue;
            if (out == null or e.ids.len > out.?.ids.len) out = e;
        }
        if (out) |e| {
            st.tick += 1;
            e.tick = st.tick;
        }
        return out;
    }

    /// The slot's state after `ids` kept (the oldest entries make room); one entry a distinct `ids`.
    pub fn save(st: *Store, c: *const Config, slot: *const kv.Slot, ids: []const u32, ring: []const u16, logits: ?[]const f32) !void {
        if (!st.on() or ids.len == 0) return;
        st.tick += 1;
        for (st.entries.items) |e| if (std.mem.eql(u32, e.ids, ids)) {
            e.tick = st.tick;
            if (e.logits != null or logits == null) return;
            e.logits = try st.a.dupe(f32, logits.?);
            st.held += logits.?.len * 4;
            return;
        };
        var buf: [4 * 64]Run = undefined;
        const rs = runs(c, slot, ids.len, &buf);
        var total: usize = 0;
        for (rs) |r| total += r.len;
        const size = ids.len * 4 + total + ring.len * 2 + if (logits) |l| l.len * 4 else 0;
        if (size > st.cap) return;
        while (st.held + size > st.cap) st.evict();
        const e = try st.a.create(Entry);
        errdefer st.a.destroy(e);
        e.* = .{ .ids = try st.a.dupe(u32, ids), .bytes = undefined, .taps = undefined, .logits = null, .tick = st.tick };
        errdefer st.a.free(e.ids);
        e.bytes = try st.a.alloc(u8, total);
        errdefer st.a.free(e.bytes);
        var at: usize = 0;
        for (rs) |r| {
            @memcpy(e.bytes[at..][0..r.len], r.buf.contents()[0..r.len]);
            at += r.len;
        }
        e.taps = try st.a.dupe(u16, ring);
        errdefer st.a.free(e.taps);
        if (logits) |l| e.logits = try st.a.dupe(f32, l);
        try st.entries.append(st.a, e);
        st.held += e.size();
    }

    /// The entry's state into `slot` (cleared first up to `high`, its old rows) and its taps into `ring`.
    pub fn restore(_: *const Store, c: *const Config, e: *const Entry, slot: *const kv.Slot, high: u64, ring: []u16) void {
        clear(c, slot, high);
        var buf: [4 * 64]Run = undefined;
        var at: usize = 0;
        for (runs(c, slot, e.ids.len, &buf)) |r| {
            @memcpy(r.buf.contents()[0..r.len], e.bytes[at..][0..r.len]);
            at += r.len;
        }
        @memcpy(ring, e.taps);
    }

    fn evict(st: *Store) void {
        var old: usize = 0;
        for (st.entries.items, 0..) |e, i| if (e.tick < st.entries.items[old].tick) {
            old = i;
        };
        const e = st.entries.swapRemove(old);
        st.held -= e.size();
        e.free(st.a);
    }
};
