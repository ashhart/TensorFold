//! Ripple's caches on CUDA: every slot's sliding rings and global pages in one buffer, then the page table.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const fwd = @import("cuda_model.zig");

const wa = cuda.window_attention;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;
const Config = cfg.Config;
const ring_rows = fwd.ring_rows;
const page_tokens = fwd.page_tokens;
const buffer_align = fwd.buffer_align;

/// The pages a slot's global layer holds: the context in whole pages.
pub fn pagesOf(context: usize) usize {
    return (context + page_tokens - 1) / page_tokens;
}

/// A layer's caches from slot 0: a sliding layer's rings [slots][ring][kv_heads][128], or (`pg`) a global's pages.
pub const Cache = struct { kc: u64, vc: u64, pg: ?wa.Pages = null };

/// One slot's rows of a layer as a snapshot copies them: position p at ring row p % ring_rows, or global row p.
pub const Span = struct { k: u64, v: u64, rows: usize, row_bytes: usize, ring: bool };

/// One slot's caches: a sliding layer's K and V rings, a global layer's K and V pages for `context`.
pub fn slotBytes(c: Config, context: usize) usize {
    const kvw = c.kvWidth();
    const g = c.globals();
    return (c.layers - g) * 2 * ring_rows * kvw * 2 + g * 2 * pagesOf(context) * page_tokens * kvw * 2;
}

/// Every slot's caches in one buffer, a layer at a time (each slot's rows, K then V), then the page table.
pub const Arena = struct {
    gpa: std.mem.Allocator,
    kv: DeviceBuffer,
    caches: []Cache, // a layer each, at slot 0
    ids: []i32, // the page table's host copy [slots][per_slot]: the call tables gather its rows
    slots: usize,
    per_slot: usize, // pages a slot's global layer holds
    kvw: usize,

    /// `slots` slots' caches for `context` positions and the int32 table (one slot: M1's State.bytes).
    pub fn bytes(c: Config, context: usize, slots: usize) usize {
        return slots * slotBytes(c, context) + std.mem.alignForward(usize, slots * pagesOf(context) * 4, buffer_align);
    }

    /// Zeroed caches (no stale bits under a masked key), the identity table: slot s's page j is page s * per_slot + j.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: Stream, c: Config, context: usize, slots: usize) !Arena {
        const kvw = c.kvWidth();
        const per = pagesOf(context);
        var kv = try DeviceBuffer.alloc(d, bytes(c, context, slots));
        errdefer kv.free();
        try d.check(d.api.cuMemsetD8Async(kv.ptr, 0, kv.len, s.handle), "cuMemsetD8Async");
        const caches = try gpa.alloc(Cache, c.layers);
        errdefer gpa.free(caches);
        const ids = try gpa.alloc(i32, slots * per);
        errdefer gpa.free(ids);
        for (ids, 0..) |*x, i| x.* = @intCast(i);
        const table_at = kv.len - std.mem.alignForward(usize, ids.len * 4, buffer_align);
        try cuda.copy_source.enqueue(d, kv.ptr + table_at, std.mem.sliceAsBytes(ids), s);
        var at: u64 = kv.ptr;
        for (caches, 0..) |*cache, i| {
            const half: usize = slots * if (c.global[i]) per * page_tokens * kvw * 2 else ring_rows * kvw * 2;
            cache.* = .{ .kc = at, .vc = at + half };
            if (c.global[i]) cache.pg = .{ .table = kv.ptr + table_at, .page_elems = page_tokens * kvw, .tokens = page_tokens, .stride = per };
            at += 2 * half;
        }
        return .{ .gpa = gpa, .kv = kv, .caches = caches, .ids = ids, .slots = slots, .per_slot = per, .kvw = kvw };
    }

    pub fn deinit(a: *Arena) void {
        a.gpa.free(a.ids);
        a.gpa.free(a.caches);
        a.kv.free();
        a.* = undefined;
    }

    /// Layer i's caches of one slot as a prompt chunk reads them: its rings, or the pools and its table row (stride 0).
    pub fn slotCache(a: *const Arena, i: usize, slot: usize) Cache {
        const c = a.caches[i];
        if (c.pg) |pg| {
            var row = pg;
            row.table += slot * a.per_slot * 4;
            row.stride = 0;
            return .{ .kc = c.kc, .vc = c.vc, .pg = row };
        }
        const ring = ring_rows * a.kvw * 2;
        return .{ .kc = c.kc + slot * ring, .vc = c.vc + slot * ring };
    }

    /// Layer i's rows of one slot.
    pub fn span(a: *const Arena, i: usize, slot: usize) Span {
        const c = a.caches[i];
        const row = a.kvw * 2;
        const rows: usize = if (c.pg != null) a.per_slot * page_tokens else ring_rows;
        return .{ .k = c.kc + slot * rows * row, .v = c.vc + slot * rows * row, .rows = rows, .row_bytes = row, .ring = c.pg == null };
    }

    /// The V pools' offset from the K pools in bf16 elements: the same in every global layer.
    pub fn vOff(a: *const Arena) usize {
        for (a.caches) |c| if (c.pg != null) return (c.vc - c.kc) / 2;
        return 0;
    }
};
