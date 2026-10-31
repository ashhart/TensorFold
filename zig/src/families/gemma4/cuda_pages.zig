//! Gemma 4's global layers in pages of the core pool: a request's pages, the pool's attach, slots and leases.
const std = @import("std");
const kv = @import("core").kv_pool;
const fwd = @import("cuda_forward.zig");
const be = @import("cuda_backend.zig");

const Gemma = be.Gemma;

/// Pages a request holds at most: its prompt, its reply and one last verify window.
pub fn pagesFor(prompt: usize, max_new: usize) usize {
    return pagesOf(prompt + max_new + be.max_window);
}

/// Pages that hold positions [0, positions).
fn pagesOf(positions: usize) usize {
    return (positions + fwd.page_tokens - 1) / fwd.page_tokens;
}

/// Makes a paged model's arena of `pages` pages for `extra + 1` slots, and the pool that leases them.
pub fn attach(x: *Gemma, extra: usize, pages: usize) !void {
    try x.m.attach(x.s, extra, pages);
    x.pool = try kv.Pool.init(x.gpa, pages, x.m.pageBytes());
}

/// The pool's free pages in a seeded random order, leased in that order (a check's hook).
pub fn shufflePages(x: *Gemma, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    if (x.pool) |*p| prng.random().shuffle(u32, p.free[0..p.nfree]);
}

fn freeSlot(x: *const Gemma) ?usize {
    return std.mem.indexOfScalar(bool, x.free[0..x.m.slots], true);
}

pub fn fitsContext(x: *const Gemma, prompt: usize, max_new: usize) bool {
    return prompt + max_new + be.max_window <= x.m.context;
}

/// Whether a request could start now: a free slot and (paged) its pages; one that never fits passes.
pub fn admits(x: *const Gemma, prompt: usize, max_new: usize) bool {
    if (!fitsContext(x, prompt, max_new)) return true; // its prompt pass refuses it by name
    const p = x.pool orelse return freeSlot(x) != null;
    return freeSlot(x) != null and p.fits(pagesFor(prompt, max_new));
}

const Map = struct {
    x: *Gemma,
    slot: usize,
    fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
        const m: *const Map = @ptrCast(@alignCast(ctx));
        try m.x.m.mapPages(m.x.s, m.slot, ids[from..], from);
    }
};

/// The lane's pages cover positions [0, end): pages taken from its promise as its context grows.
pub fn reach(x: *Gemma, l: anytype, end: usize) !void {
    const p = if (x.pool) |*p| p else return;
    const n = pagesOf(end);
    if (n <= l.lease.ids.items.len) return;
    var map: Map = .{ .x = x, .slot = l.slot };
    try p.cover(&l.lease, n, .{ .ctx = &map, .write = Map.write });
}

/// A free slot for the request, its pages promised now (taken as its rows need them); LaneMemory: not yet.
pub fn takeSlot(x: *Gemma, lease: *kv.Lease, prompt: usize, max_new: usize) !usize {
    const slot = freeSlot(x) orelse return error.NoFreeSlot;
    if (x.pool) |*p| try p.promise(lease, pagesFor(prompt, max_new));
    return slot;
}

/// The idle server's line: nothing promised, every page free (kept states are copies, not pages).
pub fn report(x: *const Gemma) void {
    if (@import("builtin").is_test) return;
    const p = x.pool orelse return;
    const c = p.census();
    const ok = p.idle() and c.free == c.total;
    std.log.info("kv pool idle{s}: promised {d}, free {d} + store {d} == total {d}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.kept, c.total });
}
