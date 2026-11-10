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

/// Kept rows a cache may hold pending, and the pending count at which a recorded slot scan writes the state back.
pub const pend_cap = 48;
pub const pend_commit = 24;

/// TF_QWEN_KEEP_LATE=1 defers a keep's rows to the next slot scan (#1824): opt-in, its copy failed served chats.
pub fn keepLate() bool {
    const S = struct {
        var on: ?bool = null;
    };
    if (S.on) |v| return v;
    const v = if (std.c.getenv("TF_QWEN_KEEP_LATE")) |x| std.mem.eql(u8, std.mem.span(x), "1") else false;
    S.on = v;
    return v;
}

/// One kept row's decay, write and key in one linear layer: the head of its log row (qwen35.cuh pend_row).
pub fn headBytes(g: anytype) usize {
    return g.v_heads * 4 + g.v_heads * geometry.dk * 4 + g.k_heads * geometry.dk * 2;
}

/// A cache's pending buffer: pend_cap rows of every linear layer's head.
pub fn pendBytes(g: anytype) usize {
    return pend_cap * g.deltas() * headBytes(g);
}

/// qwen35.cuh's Seg and Path: rec, conv, before and log point into the stream's States table.
pub const Seg = extern struct { rec: u64, conv: u64, kv: u64, tap: u64, log: u64, before: u64, cap: c_int, len: c_int, row0: c_int, rows: c_int, tap_rows: c_int, taps: c_int, delta: c_int, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0, pn: c_int = 0, plog: u64 = 0 };
pub const Path = extern struct { n: c_int, rows: [engine.most_window]c_int };

/// A stream's state table: cuda_states.zig.
pub const States = @import("cuda_states.zig").States;

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
    return parsePageTest(std.mem.span(v));
}

pub fn parsePageTest(v: []const u8) !usize {
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadQwenPageTest;
    if (!validPageTokens(n)) return error.BadQwenPageTest;
    return n;
}

pub fn validPageTokens(n: usize) bool {
    return n > 0 and n <= std.math.maxInt(c_int);
}

/// Logical capacity rounded to whole pages, still representable by Seg.cap.
pub fn cacheRows(cap: usize, tokens: usize) !usize {
    if (cap > std.math.maxInt(c_int) or (tokens != 0 and !validPageTokens(tokens))) return error.BadQwenPageTest;
    if (tokens == 0) return cap;
    const rounded = try std.math.add(usize, cap, tokens - 1);
    const rows = rounded / tokens * tokens;
    if (rows > std.math.maxInt(c_int)) return error.BadQwenPageTest;
    return rows;
}

/// Diagnostic paging adds padded cache rows and a device page table to a contiguous cache's budget.
pub fn pageExtra(g: geometry.Geometry, cap: usize, tokens: usize, kv8: bool) !usize {
    if (tokens == 0) return 0;
    const rows = try cacheRows(cap, tokens);
    const padding = try std.math.mul(usize, rows - cap, g.kvBytes(1, kv8));
    return std.math.add(usize, padding, try std.math.mul(usize, rows / tokens, 4));
}

/// Where a cache's before sits in its log's bytes: past the first row, which is all a one-row window logs.
pub fn overlay(e: *const Engine) @import("core").recurrent_workspace.Overlay {
    return .init(e.g.stateBytes(), e.g.logBytes(1), engine.most_window);
}

/// What a cache's log and before take: one overlaid block, or with adaptive scratch the bound its owner holds.
pub fn scratchBytes(e: *const Engine) usize {
    return if (recurrent.enabled(e)) recurrent.bound(e) else overlay(e).bytes;
}

fn logView(e: *const Engine, at: u64) DeviceBuffer {
    return .{ .d = e.d, .ptr = at, .len = e.g.logBytes(engine.most_window), .borrowed = true };
}

fn beforeView(e: *const Engine, at: u64) DeviceBuffer {
    return .{ .d = e.d, .ptr = at + overlay(e).before_at, .len = e.g.stateBytes(), .borrowed = true };
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
    /// A pool lane (its run mapped): its kernels read `states` (the _table kernels); false: its blocks, as on main.
    on_table: bool = false,
    delta: i32 = 0,
    state: DeviceBuffer,
    /// The exact state before a recorded one-row round advances its stable slot in place.
    before: ?DeviceBuffer = null,
    /// A window's log and before in one block (recurrent_workspace.Overlay); runs and adaptive caches have none.
    block: ?DeviceBuffer = null,
    kv: DeviceBuffer,
    /// The state table the kernels read the state, before and log through (null: a test's cache without one).
    states: ?States = null,
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
    /// Kept rows (rows 0..pend-1 of `pbuf`) the state does not hold yet: recorded slot scans replay them.
    pend: c_int = 0,
    /// The pending heads (pendBytes), allocated at the first deferred keep.
    pbuf: ?DeviceBuffer = null,
    /// An image prompt's rotary shift (get_rope_index's rope_delta): rows after its prompt rotate at position + delta.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, cap: usize) !Cache {
        const test_pages = try pageTest();
        const rows = try cacheRows(cap, test_pages);
        var kv = try DeviceBuffer.alloc(e.d, try std.math.mul(usize, e.g.kvBytes(1, e.kv8), rows));
        var own = true;
        errdefer if (own) kv.free();
        var paged: ?Paged = null;
        errdefer if (own) {
            if (paged) |*p| freePaged(gpa, p);
        };
        if (test_pages > 0) {
            const n = rows / test_pages;
            const ids = try gpa.alloc(u32, n);
            errdefer gpa.free(ids);
            for (ids, 0..) |*id, i| id.* = @intCast(n - 1 - i); // a table no contiguous layout matches
            var table = try DeviceBuffer.alloc(e.d, n * 4);
            errdefer table.free();
            paged = .{ .base = kv.ptr, .table = table, .ids = ids, .tokens = test_pages };
        }
        var out = try fixed(gpa, e, cap, kv, paged);
        own = false;
        errdefer out.deinit(gpa);
        errdefer cuda.completion.drain(e.d, e.stream, "qwen3_5: failed cache initialization did not drain");
        try kv.fill8(0, e.stream.handle);
        if (paged) |p| try p.table.uploadAsync(0, std.mem.sliceAsBytes(p.ids), e.stream.handle);
        try e.stream.synchronize();
        return out;
    }

    /// A pool lane's cache: a table of `pages` pages at `base`; its state, log and tap ring a run mapped at open.
    pub fn initPaged(gpa: std.mem.Allocator, e: *Engine, pages: usize, tokens: usize, base: u64) !Cache {
        var table = try DeviceBuffer.alloc(e.d, pages * 4);
        errdefer table.free();
        var states = try States.init(gpa, e);
        errdefer states.deinit(gpa);
        const none = try DeviceBuffer.alloc(e.d, 0);
        return .{ .state = none, .before = none, .kv = none, .paged = .{ .base = base, .table = table, .ids = &.{}, .tokens = tokens, .own_ids = false }, .states = states, .log = none, .cap = 0, .logits = try gpa.alloc(u16, e.g.vocab) };
    }

    /// Bytes of a pool lane's run: state, log with its rollback copy, tap ring and a drafter's ring, each 256-aligned.
    pub fn runBytes(e: *const Engine, ring: usize) usize {
        var n: usize = 0;
        for (runParts(e, ring)) |len| n += std.mem.alignForward(usize, len, 256);
        return n;
    }

    fn runParts(e: *const Engine, ring: usize) [4]usize {
        return .{ e.g.stateBytes(), overlay(e).bytes, e.tapBytes(), ring };
    }

    /// A lane's state, log and before, taps and drafter ring as views of the run at `base`; a short run is refused.
    pub fn mapRun(self: *Cache, e: *const Engine, base: u64, len: usize, ring: usize) !void {
        if (runBytes(e, ring) > len) return error.QwenRunTooShort;
        var at = base;
        var views: [4]DeviceBuffer = undefined;
        for (runParts(e, ring), &views) |n, *v| {
            v.* = .{ .d = e.d, .ptr = at, .len = n, .borrowed = true };
            at += std.mem.alignForward(usize, n, 256);
        }
        self.state = views[0];
        self.pend = 0;
        self.log = logView(e, views[1].ptr);
        self.before = beforeView(e, views[1].ptr);
        self.tap = if (e.n_taps > 0) views[2] else null;
        self.drafter_ring = if (ring > 0) views[3].ptr else null;
        self.on_table = true;
        for ([_]DeviceBuffer{ self.log, views[2], views[3] }) |v| if (v.len > 0) try v.fill8(0, e.stream.handle);
    }

    /// A closed lane's views dropped: its run went back to the pool.
    pub fn unmapRun(self: *Cache, e: *const Engine) void {
        const none: DeviceBuffer = .{ .d = e.d, .ptr = 0, .len = 0, .borrowed = true };
        self.state = none;
        self.pend = 0;
        self.before = none;
        self.log = none;
        self.tap = null;
        self.drafter_ring = null;
        self.on_table = false;
    }

    /// The parts every cache holds itself beside `kv` and `paged` (the caller's to free on failure).
    fn fixed(gpa: std.mem.Allocator, e: *Engine, cap: usize, kv: DeviceBuffer, paged: ?Paged) !Cache {
        var state = try DeviceBuffer.alloc(e.d, e.g.stateBytes());
        errdefer state.free();
        const adaptive = recurrent.enabled(e);
        var block: ?DeviceBuffer = null;
        errdefer if (block) |*b| b.free();
        if (!adaptive) block = try DeviceBuffer.alloc(e.d, overlay(e).bytes);
        const before: ?DeviceBuffer = if (block) |b| beforeView(e, b.ptr) else null;
        const none: DeviceBuffer = .{ .d = e.d, .ptr = 0, .len = 0, .borrowed = true };
        const log = if (block) |b| logView(e, b.ptr) else none;
        var tap: ?DeviceBuffer = null;
        errdefer if (tap) |*t| t.free();
        if (e.n_taps > 0) {
            tap = try DeviceBuffer.alloc(e.d, e.tapBytes());
        }
        var states = try States.init(gpa, e);
        errdefer states.deinit(gpa);
        const logits = try gpa.alloc(u16, e.g.vocab);
        errdefer gpa.free(logits);
        errdefer cuda.completion.drain(e.d, e.stream, "qwen3_5: failed cache state initialization did not drain");
        try state.fill8(0, e.stream.handle);
        if (!adaptive) try log.fill8(0, e.stream.handle);
        if (tap) |t| try t.fill8(0, e.stream.handle);
        try e.stream.synchronize();
        const scratch = if (adaptive) recurrent.bound(e) else 0;
        e.rec_scratch.cap += scratch;
        var c: Cache = .{ .owner = if (adaptive) &e.rec_scratch else null, .scratch = scratch, .state = state, .before = before, .kv = kv, .paged = paged, .tap = tap, .states = states, .log = log, .cap = cap, .logits = logits };
        c.block = block;
        return c;
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
        self.pend = 0;
        self.delta = 0;
        try self.state.fill8(0, e.stream.handle);
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        recurrent.release(self);
        if (self.owner) |p| p.cap -= self.scratch;
        self.state.free();
        if (self.before) |*b| b.free();
        if (self.block) |*b| b.free();
        self.kv.free();
        if (self.paged) |*p| freePaged(gpa, p);
        self.log.free();
        if (self.tap) |*t| t.free();
        if (self.pbuf) |*b| b.free();
        if (self.states) |*t| t.deinit(gpa);
        gpa.free(self.logits);
    }

    /// The state table rewritten (on the engine's stream, before the launches that read it) if a block moved.
    pub fn stateTable(self: *Cache, e: *const Engine) !States {
        const t = &(self.states orelse return error.MissingQwenStates);
        const from: [3]u64 = .{ self.state.ptr, if (self.before) |b| b.ptr else 0, self.log.ptr };
        if (t.from) |f| if (std.mem.eql(u64, &f, &from)) return t.*;
        States.fill(t.host, e.g, from[0], from[1], from[2]);
        try t.dev.uploadAsync(0, std.mem.sliceAsBytes(t.host), e.stream.handle);
        t.from = from;
        return t.*;
    }

    /// The stream's window of `rows` rows from forward row `row0` after `len` cached positions (pool lane: its table).
    pub fn seg(self: *Cache, e: *const Engine, len: usize, row0: usize, rows: usize) !Seg {
        if (self.on_table) return self.segIn(e, (try self.stateTable(e)).at(e.g), len, row0, rows);
        const g = e.g;
        const at: [4]u64 = .{ self.state.ptr, self.state.ptr + g.deltas() * g.recBytes(), if (self.before) |b| b.ptr else 0, self.log.ptr };
        return self.segIn(e, at, len, row0, rows);
    }

    /// `seg` with `t` (rec, conv, before, log): its blocks, or a state table's entries (States.parts).
    pub fn segIn(self: *Cache, e: *const Engine, t: [4]u64, len: usize, row0: usize, rows: usize) Seg {
        const g = e.g;
        var sg: Seg = .{ .rec = t[0], .conv = t[1], .kv = self.kv.ptr, .tap = if (self.tap) |x| x.ptr else 0, .log = t[3], .before = t[2], .cap = @intCast(self.cap), .len = @intCast(len), .row0 = @intCast(row0), .rows = @intCast(rows), .tap_rows = tap_rows, .taps = @intCast(e.n_taps), .delta = self.delta };
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
        const sg = try self.seg(e, last.start, 0, last.rows);
        try recurrent.mark(self, false);
        // the path's rows join the pending ones; without a log, with the switch off or past pend_cap they land now
        const deferred = keepLate() and self.log.len > 0 and path.len <= pend_cap;
        const tab = self.on_table;
        if (!deferred) {
            try self.flush(e);
            try e.run(if (tab) e.keeps.apply_table else e.keeps.apply, .{ 1, geometry.dk / 4, g.v_heads * g.deltas() }, .{ 32, 4, 1 }, .{ e.geo, sg, p });
        }
        try e.run(if (tab) e.keeps.conv_table else e.keeps.conv, .{ g.deltas(), (g.convBytes() / 2 + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ e.geo, sg, @as(c_int, @intCast(path[path.len - 1])) });
        if (!prefix and g.attentions() > 0) try e.run(if (e.kv8) e.fp8.compact else e.keeps.compact, .{ g.attentions(), g.kv_heads, 1 }, .{ geometry.head_dim, 1, 1 }, .{ e.geo, sg, p });
        if (deferred) try self.pendRows(e, path);
        self.last = null;
        self.len = last.start + path.len;
    }

    /// The kept path's heads after the pending rows, a run of consecutive rows one copy; a full buffer lands first.
    fn pendRows(self: *Cache, e: *Engine, path: []const u32) !void {
        const g = e.g;
        if (self.pbuf == null) self.pbuf = try DeviceBuffer.alloc(e.d, pendBytes(g));
        if (@as(usize, @intCast(self.pend)) + path.len > pend_cap) try self.flush(e);
        const hb = headBytes(g);
        const row_log = g.deltas() * g.logRowBytes();
        const tab = self.on_table;
        const t: ?States = if (tab) try self.stateTable(e) else null; // a window stream: main's rows, no table
        const log = self.log.ptr;
        const base = self.pbuf.?.ptr;
        var j: usize = 0;
        while (j < path.len) {
            // consecutive window rows whose records follow each other in the log, one copy
            var run: usize = 1;
            while (j + run < path.len and path[j + run] == path[j] + run and (!tab or t.?.logRow(g, path[j + run]) == t.?.logRow(g, path[j]) + run * row_log)) run += 1;
            const at: usize = @as(usize, @intCast(self.pend)) + j;
            const m: cuda.abi.Memcpy2D = .{ .src = if (tab) t.?.logRow(g, path[j]) else log + path[j] * row_log, .src_pitch = g.logRowBytes(), .dst = base + at * g.deltas() * hb, .dst_pitch = hb, .width = hb, .height = run * g.deltas() };
            // a copy past the log or the pending buffer is refused here, not by the driver mid-reply
            const src_end = if (m.src < log) std.math.maxInt(u64) else m.src - log + (m.height - 1) * m.src_pitch + m.width;
            const dst_end = m.dst - base + (m.height - 1) * m.dst_pitch + m.width;
            if (src_end > self.log.len or dst_end > self.pbuf.?.len) {
                std.log.err("qwen3_5 pending keep: path {d} rows after {d} pending, path row {d} (window row {d}), run {d}: source end {d} of a {d}-byte log, destination end {d} of a {d}-byte buffer; head {d} bytes, log row {d}", .{ path.len, self.pend, j, path[j], run, src_end, self.log.len, dst_end, self.pbuf.?.len, hb, g.logRowBytes() });
                return error.QwenPendingOutOfBounds;
            }
            try e.d.check(e.d.api.cuMemcpy2DAsync_v2(&m, e.stream.handle), "cuMemcpy2DAsync");
            j += run;
        }
        self.pend += @intCast(path.len);
    }

    /// The pending rows landed in the state now (before anything but a recorded slot scan reads it).
    pub fn flush(self: *Cache, e: *Engine) !void {
        if (self.pend == 0) return;
        var sg = try self.seg(e, self.len, 0, 0);
        sg.plog = self.pbuf.?.ptr;
        const n = self.pend;
        self.pend = 0;
        const g = e.g;
        try e.run(if (self.on_table) e.keeps.apply_pend_table else e.keeps.apply_pend, .{ 1, geometry.dk / 4, g.v_heads * g.deltas() }, .{ 32, 4, 1 }, .{ e.geo, sg, n });
    }

    /// This forward's scan entry with the pending rows, written back (a positive count) at one row or pend_commit.
    pub fn fold(self: *Cache, seg_: Seg) Seg {
        var sg = seg_;
        if (self.pend == 0) return sg;
        sg.plog = self.pbuf.?.ptr;
        const commit = seg_.rows == 1 or self.pend >= pend_commit;
        sg.pn = if (commit) self.pend else -self.pend;
        if (commit) self.pend = 0;
        return sg;
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

test "KV page dimensions: diagnostic rows fit the signed kernel argument" {
    try std.testing.expectEqual(@as(usize, 48), try parsePageTest("48"));
    try std.testing.expectError(error.BadQwenPageTest, parsePageTest("0"));
    try std.testing.expectError(error.BadQwenPageTest, parsePageTest("2147483648"));
    try std.testing.expectError(error.BadQwenPageTest, parsePageTest("18446744073709551615"));
}

test "KV page dimensions: diagnostic admission includes its padded rows and device table" {
    const g: geometry.Geometry = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 64, .linear = @splat(false) };
    try std.testing.expectEqual(@as(usize, 96), try cacheRows(65, 48));
    for ([_]bool{ false, true }) |fp8| try std.testing.expectEqual(31 * g.kvBytes(1, fp8) + 8, try pageExtra(g, 65, 48, fp8));
    try std.testing.expectError(error.BadQwenPageTest, cacheRows(2147483647, 48));
    try std.testing.expectError(error.BadQwenPageTest, cacheRows(std.math.maxInt(usize), 48));
}
