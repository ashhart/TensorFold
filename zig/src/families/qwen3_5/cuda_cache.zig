//! A stream's state on CUDA: recurrences, conv tails, keys and values, the drafter's tap ring, the last window's log.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Engine = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;
const recurrent = @import("cuda_recurrent.zig");

/// Rows a drafter's tap ring holds a stream: its window and a prompt chunk more.
pub const tap_rows = 2048 + engine.max_rows;

/// qwen35.cuh's Seg and Path.
pub const Seg = extern struct { rec: u64, conv: u64, kv: u64, tap: u64, log: u64, before: u64, cap: c_int, len: c_int, row0: c_int, rows: c_int, tap_rows: c_int, taps: c_int, delta: c_int, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0 };
pub const Path = extern struct { n: c_int, rows: [engine.most_window]c_int };

/// A cache's keys and values as pages (qwen35.cuh kv_row): page i of the stream's positions at `base` + ids[i] pages.
pub const Paged = struct {
    /// Where page 0 of the arena starts: the core pool's, or the cache's own buffer under TF_QWEN_KV_PAGE_TEST.
    base: u64,
    /// The stream's page table on the device, an int a page, `ids` uploaded.
    table: DeviceBuffer,
    ids: []u32,
    /// Positions a page holds.
    tokens: usize,
    /// The cache owns `ids` (TF_QWEN_KV_PAGE_TEST); a pool's lane maps its lease's.
    own_ids: bool = true,

    /// Elements a page holds (bf16, or bytes for fp8 rows and their scales): every attention layer's of `tokens` rows.
    pub fn elems(p: Paged, g: geometry.Geometry, kv8: bool) usize {
        return if (kv8) g.kvBytes(p.tokens, true) else g.kvBytes(p.tokens, false) / 2;
    }
};

/// TF_QWEN_KV_PAGE_TEST=n: each cache's own buffer read through a table of n-position pages in reverse order (checks).
pub fn pageTest() !usize {
    const v = std.c.getenv("TF_QWEN_KV_PAGE_TEST") orelse return 0;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadQwenPageTest;
    if (n == 0) return error.BadQwenPageTest;
    return n;
}

pub const Cache = struct {
    owner: ?*recurrent.Pool = null,
    log_lease: ?recurrent.Lease = null,
    before_lease: ?recurrent.Lease = null,
    /// Bytes this cache adds to its owner's cap while it lives (recurrent.bound).
    scratch: usize = 0,
    untaught: bool = false,
    /// Keys and values through a page table (null: cap contiguous rows a head).
    paged: ?Paged = null,
    /// A pool lane's drafter ring in its run (the drafter borrows it for the lane's stream).
    drafter_ring: ?u64 = null,
    delta: i32 = 0,
    state: DeviceBuffer,
    /// The exact state before a recorded one-row round advances its stable slot in place.
    before: ?DeviceBuffer = null,
    kv: DeviceBuffer,
    /// A drafter's tapped rows (`Engine.taps`), a ring of tap_rows: the row at position p in slot p % tap_rows.
    tap: ?DeviceBuffer = null,
    /// The newest verify window's records (qwen35.cuh's LogRow), held until its keep completes.
    log: DeviceBuffer,
    cap: usize,
    len: usize = 0,
    /// Positions the newest forward wrote up to: a tap ring holds them until the next forward, whatever `keep` lands.
    written: usize = 0,
    logits: []u16,
    /// The newest verify window, until its path is kept: where it started, its rows.
    last: ?struct { start: usize, rows: usize } = null,
    /// An image prompt's rotary shift (get_rope_index's rope_delta): rows after its prompt rotate at position + delta.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, cap: usize) !Cache {
        const test_pages = try pageTest();
        const rows = if (test_pages > 0) (cap + test_pages - 1) / test_pages * test_pages else cap;
        var kv = try DeviceBuffer.alloc(e.d, e.g.kvBytes(rows, e.kv8));
        errdefer kv.free();
        try kv.fill8(0, null); // touched now, not as the first request's keys land (the driver commits on first touch)
        var paged: ?Paged = null;
        errdefer if (paged) |*p| freePaged(gpa, p);
        if (test_pages > 0) {
            const n = rows / test_pages;
            const ids = try gpa.alloc(u32, n);
            errdefer gpa.free(ids);
            for (ids, 0..) |*id, i| id.* = @intCast(n - 1 - i); // a table no contiguous layout matches
            var table = try DeviceBuffer.alloc(e.d, n * 4);
            errdefer table.free();
            try table.upload(0, std.mem.sliceAsBytes(ids));
            paged = .{ .base = kv.ptr, .table = table, .ids = ids, .tokens = test_pages };
        }
        return fixed(gpa, e, cap, kv, paged);
    }

    /// A pool lane's cache: a table of `pages` pages at `base`; its state, log and tap ring a run mapped at open.
    pub fn initPaged(gpa: std.mem.Allocator, e: *Engine, pages: usize, tokens: usize, base: u64) !Cache {
        var table = try DeviceBuffer.alloc(e.d, pages * 4);
        errdefer table.free();
        try table.fill8(0, null);
        const none = try DeviceBuffer.alloc(e.d, 0);
        return .{ .state = none, .before = none, .kv = none, .paged = .{ .base = base, .table = table, .ids = &.{}, .tokens = tokens, .own_ids = false }, .log = none, .cap = 0, .logits = try gpa.alloc(u16, e.g.vocab) };
    }

    /// Bytes of a pool lane's run: state, before, log, tap ring and a drafter's ring (`ring`), each 256-aligned.
    pub fn runBytes(e: *const Engine, ring: usize) usize {
        var n: usize = 0;
        for (runParts(e, ring)) |len| n += std.mem.alignForward(usize, len, 256);
        return n;
    }

    fn runParts(e: *const Engine, ring: usize) [5]usize {
        return .{ e.g.stateBytes(), e.g.stateBytes(), e.g.logBytes(engine.most_window), e.tapBytes(), ring };
    }

    /// A lane's state, before, log, taps and drafter ring as views of the run at `base`; a short run is refused.
    pub fn mapRun(self: *Cache, e: *const Engine, base: u64, len: usize, ring: usize) !void {
        if (runBytes(e, ring) > len) return error.QwenRunTooShort;
        var at = base;
        var views: [5]DeviceBuffer = undefined;
        for (runParts(e, ring), &views) |n, *v| {
            v.* = .{ .d = e.d, .ptr = at, .len = n, .borrowed = true };
            at += std.mem.alignForward(usize, n, 256);
        }
        self.state = views[0];
        self.before = views[1];
        self.log = views[2];
        self.tap = if (e.n_taps > 0) views[3] else null;
        self.drafter_ring = if (ring > 0) views[4].ptr else null;
        for ([_]DeviceBuffer{ views[2], views[3], views[4] }) |v| if (v.len > 0) try v.fill8(0, e.stream.handle);
    }

    /// A closed lane's views dropped: its run went back to the pool.
    pub fn unmapRun(self: *Cache, e: *const Engine) void {
        const none: DeviceBuffer = .{ .d = e.d, .ptr = 0, .len = 0, .borrowed = true };
        self.state = none;
        self.before = none;
        self.log = none;
        self.tap = null;
        self.drafter_ring = null;
    }

    /// The parts every cache holds itself beside `kv` and `paged` (the caller's to free on failure).
    fn fixed(gpa: std.mem.Allocator, e: *Engine, cap: usize, kv: DeviceBuffer, paged: ?Paged) !Cache {
        var state = try DeviceBuffer.alloc(e.d, e.g.stateBytes());
        errdefer state.free();
        try state.fill8(0, null);
        const adaptive = recurrent.enabled(e);
        var before: ?DeviceBuffer = null;
        errdefer if (before) |*b| b.free();
        if (!adaptive) before = try DeviceBuffer.alloc(e.d, e.g.stateBytes());
        var log: DeviceBuffer = .{ .d = e.d, .ptr = 0, .len = 0, .borrowed = true };
        if (!adaptive) log = try DeviceBuffer.alloc(e.d, e.g.logBytes(engine.most_window));
        errdefer log.free();
        if (!adaptive) try log.fill8(0, null);
        var tap: ?DeviceBuffer = null;
        errdefer if (tap) |*t| t.free();
        if (e.n_taps > 0) {
            tap = try DeviceBuffer.alloc(e.d, e.tapBytes());
            try tap.?.fill8(0, null);
        }
        const logits = try gpa.alloc(u16, e.g.vocab);
        const scratch = if (adaptive) recurrent.bound(e) else 0;
        e.rec_scratch.cap += scratch;
        return .{ .owner = if (adaptive) &e.rec_scratch else null, .scratch = scratch, .state = state, .before = before, .kv = kv, .paged = paged, .tap = tap, .log = log, .cap = cap, .logits = logits };
    }

    /// Publish a completed forward's metadata, including one-row slots that already advanced in place.
    pub fn advanced(self: *Cache, rows: usize, record: bool) void {
        self.last = if (record) .{ .start = self.len, .rows = rows } else null;
        self.len += rows;
        self.written = self.len;
    }

    /// Ready for another stream: no rows, the recurrences and tails zero (on the engine's stream, before its forwards).
    pub fn reset(self: *Cache, e: *const Engine) !void {
        recurrent.release(self);
        self.len = 0;
        self.written = 0;
        self.last = null;
        self.untaught = false;
        self.delta = 0;
        try self.state.fill8(0, e.stream.handle);
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        recurrent.release(self);
        if (self.owner) |p| p.cap -= self.scratch;
        self.state.free();
        if (self.before) |*b| b.free();
        self.kv.free();
        if (self.paged) |*p| freePaged(gpa, p);
        self.log.free();
        if (self.tap) |*t| t.free();
        gpa.free(self.logits);
    }

    /// The stream's window of `rows` rows from forward row `row0`, after `len` cached positions.
    pub fn seg(self: *const Cache, e: *const Engine, len: usize, row0: usize, rows: usize) Seg {
        const g = e.g;
        var sg: Seg = .{ .rec = self.state.ptr, .conv = self.state.ptr + g.deltas() * g.recBytes(), .kv = self.kv.ptr, .tap = if (self.tap) |t| t.ptr else 0, .log = self.log.ptr, .before = if (self.before) |b| b.ptr else 0, .cap = @intCast(self.cap), .len = @intCast(len), .row0 = @intCast(row0), .rows = @intCast(rows), .tap_rows = tap_rows, .taps = @intCast(e.n_taps), .delta = self.delta };
        if (self.paged) |p| {
            sg.kv = p.base;
            sg.pt = p.table.ptr;
            sg.ps = p.elems(g, e.kv8);
            sg.ptok = @intCast(p.tokens);
        }
        return sg;
    }

    /// Keeps `path` (window rows, ascending from 0): the state as that chain would leave it, a tree's keys moved down.
    pub fn keep(self: *Cache, e: *Engine, path: []const u32) !void {
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0) {
            if (last.rows == 1) {
                try recurrent.mark(self, true);
                try self.state.copyFrom(0, (self.before orelse return error.MissingQwenBefore).ptr, e.g.stateBytes(), e.stream.handle);
            }
            self.last = null;
            self.len = last.start;
            return;
        }
        if (path.len > last.rows or path[0] != 0) return error.NothingToKeep;
        var p: Path = .{ .n = @intCast(path.len), .rows = @splat(0) };
        var prefix = true;
        for (path, 0..) |r, i| {
            if (r >= last.rows or (i > 0 and r <= path[i - 1])) return error.NothingToKeep;
            p.rows[i] = @intCast(r);
            prefix = prefix and r == i;
        }
        if (last.rows == 1) {
            self.last = null;
            self.len = last.start + 1;
            return;
        }
        const g = e.g;
        const sg = self.seg(e, last.start, 0, last.rows);
        try recurrent.mark(self, false);
        try e.run(e.keeps.apply, .{ 1, geometry.dk / 4, g.v_heads * g.deltas() }, .{ 32, 4, 1 }, .{ e.geo, sg, p });
        try e.run(e.keeps.conv, .{ g.deltas(), (g.convBytes() / 2 + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ e.geo, sg, @as(c_int, @intCast(path[path.len - 1])) });
        if (!prefix and g.attentions() > 0) try e.run(if (e.kv8) e.fp8.compact else e.keeps.compact, .{ g.attentions(), g.kv_heads, 1 }, .{ geometry.head_dim, 1, 1 }, .{ e.geo, sg, p });
        self.last = null;
        self.len = last.start + path.len;
    }

    /// The newest window kept whole (the round loop keeps nothing when every row landed), before the next forward.
    pub fn settle(self: *Cache, e: *Engine) !void {
        const last = self.last orelse return;
        var all: [engine.most_window]u32 = undefined;
        for (all[0..last.rows], 0..) |*r, i| r.* = @intCast(i);
        try self.keep(e, all[0..last.rows]);
    }
};

fn freePaged(gpa: std.mem.Allocator, p: *Paged) void {
    p.table.free();
    if (p.own_ids) gpa.free(p.ids);
}
