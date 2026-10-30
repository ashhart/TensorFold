//! Flash Next's streams beside the first take attention-cache pages from the core pool (TF_FLASHNEXT_KV_PAGES=n).
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const prompt = @import("cuda_prompt.zig");
const pages = @import("cuda_pages.zig");
const mtp = @import("cuda_mtp.zig");
const snap = @import("cuda_snap.zig");
const wt = @import("cuda_prompt_wt.zig");

/// Rows past a request's prompt and reply its pages hold: one drafted window's.
pub const window_rows: u32 = mtp.max_rows;

/// TF_FLASHNEXT_KV_PAGES=n: streams beside the first take n-position pages from one pool (unset or 0: none).
pub fn paging() !usize {
    const v = std.c.getenv("TF_FLASHNEXT_KV_PAGES") orelse return 0;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadFlashNextKvPages;
    if (n != 0 and !pages.valid(n)) return error.BadFlashNextKvPages;
    return n;
}

/// TF_FLASHNEXT_KV_PAGES_ALL=1: every stream on pages, none on its own contiguous caches (checks).
pub fn pagedOnly() bool {
    const v = std.c.getenv("TF_FLASHNEXT_KV_PAGES_ALL") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Pages a stream maps at most: a full window and one more drafted window.
fn mostPages(capacity: usize, tokens: usize) usize {
    return (capacity + window_rows + tokens - 1) / tokens;
}

fn al(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

const conv_bytes = mtp.conv_bytes;
const state_bytes = wt.state_n * 4;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;

/// A running stream's own states in its run: each DeltaNet layer's convolution tail and state, the n-gram tail.
fn seqRunBytes() usize {
    return wt.linear_n * (al(conv_bytes) + al(state_bytes)) + al(tail_bytes);
}

/// The MTP head a drafting lane runs: its draft vocabulary and the widest prompt chunk whose streams it keeps.
pub const Head = struct { draft_n: usize, rows: usize };

/// Fixed: a lane's table and `extra` (image rows); paged: the attention caches (the head's too); a run: the states.
pub fn layout(capacity: usize, tokens: usize, head: ?Head, extra: usize) kv.Layout {
    const layers = wt.attn_n + @intFromBool(head != null);
    const run = seqRunBytes() + if (head) |h| mtp.Draft.runBytes(h.draft_n, h.rows) else 0;
    return .{ .fixed_bytes = mostPages(capacity, tokens) * 4 + extra, .token_bytes = pages.pageBytesOf(tokens, layers) / tokens, .page_tokens = @intCast(tokens), .snap_bytes = snap.bytes(0, head != null), .run_bytes = run };
}

/// Where a stream's caches came from: no pool, the one stream on its own caches beside a pool, or a pool lane.
pub const From = enum { none, own, pool };

/// The backend's paging: the pool `attach` made, its layout from load, and whether the own stream is taken.
pub const State = struct {
    pool: ?*Pool = null,
    layout: ?kv.Layout = null,
    head: ?Head = null,
    paged_only: bool = false,
    own_busy: bool = false,

    pub fn choose(s: *const State) From {
        if (s.pool == null) return .none;
        return if (!s.paged_only and !s.own_busy) .own else .pool;
    }

    pub fn opened(s: *State, from: From) void {
        if (from == .own) s.own_busy = true;
    }

    /// A stream's caches given back: a lane's to the pool (true), else the caller frees them (false).
    pub fn close(s: *State, from: From, seq: *prompt.Seq) bool {
        switch (from) {
            .pool => return s.pool.?.close(seq),
            .own => s.own_busy = false,
            .none => {},
        }
        return false;
    }

    /// The lane host's admission: the own stream is free, or a lane and the request's pages are.
    pub fn fits(s: *const State, prompt_rows: u32, max_tokens: u32) bool {
        const p = s.pool orelse return true;
        if (!s.paged_only and !s.own_busy) return true;
        return p.fits(@as(usize, prompt_rows) + max_tokens + window_rows);
    }
};

/// One pool lane: its cache's views (its run's states, its stream's pages), that stream's lease and its run.
const Lane = struct { seq: prompt.Seq, lease: kv.Lease = .{}, used: bool = false, run: u32 = 0 };

pub const Pool = struct {
    gpa: std.mem.Allocator,
    e: *prompt.Engine,
    layout: kv.Layout,
    capacity: usize,
    head: ?Head,
    pool: kv.Pool,
    arena: cuda.DeviceBuffer,
    lanes: []Lane,

    /// `streams` lanes and `count` pages of layout `l`, made and touched now; a lane holds only its table idle.
    pub fn init(gpa: std.mem.Allocator, e: *prompt.Engine, l: kv.Layout, capacity: usize, head: ?Head, streams: usize, count: usize) !*Pool {
        if (!pages.valid(l.page_tokens) or capacity > prompt.max_tokens or count > @as(usize, std.math.maxInt(c_int)) + 1) return error.BadFlashNextKvPages;
        const page_bytes = try std.math.mul(usize, @intCast(l.token_bytes), l.page_tokens);
        if (page_bytes == 0) return error.BadFlashNextKvPages;
        _ = try std.math.add(u64, l.run_bytes, page_bytes - 1);
        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);
        var arena = try cuda.DeviceBuffer.alloc(e.driver, try std.math.mul(usize, count, page_bytes));
        errdefer arena.free();
        p.* = .{ .gpa = gpa, .e = e, .layout = l, .capacity = capacity, .head = head, .pool = try kv.Pool.init(gpa, count, page_bytes), .arena = arena, .lanes = &.{} };
        errdefer p.pool.deinit();
        p.lanes = try gpa.alloc(Lane, streams);
        errdefer gpa.free(p.lanes);
        var made: usize = 0;
        errdefer for (p.lanes[0..made]) |*x| x.seq.deinit(gpa);
        for (p.lanes) |*x| {
            x.* = .{ .seq = try p.laneSeq() };
            made += 1;
        }
        errdefer cuda.completion.drain(e.driver, e.stream.*, "flashnext: failed pool initialization did not drain");
        try arena.fill8(0, e.stream.handle);
        for (p.lanes) |x| try x.seq.paged.?.table.fill8(0, e.stream.handle);
        try e.stream.synchronize();
        return p;
    }

    /// A lane's cache: views only (states mapped from its run at open, attention caches on the arena) and a table.
    fn laneSeq(p: *Pool) !prompt.Seq {
        const gpa = p.gpa;
        const d = p.e.driver;
        var table = try cuda.DeviceBuffer.alloc(d, mostPages(p.capacity, p.layout.page_tokens) * 4);
        errdefer table.free();
        const paged: pages.Paged = .{ .base = p.arena.ptr, .table = table, .ids = &.{}, .tokens = p.layout.page_tokens, .own_ids = false, .layers = wt.attn_n + @intFromBool(p.head != null) };
        const conv = try gpa.alloc(cuda.DeviceBuffer, wt.linear_n);
        errdefer gpa.free(conv);
        const state = try gpa.alloc(cuda.DeviceBuffer, wt.linear_n);
        errdefer gpa.free(state);
        const kc = try paged.views(gpa, d, .key);
        errdefer gpa.free(kc);
        const vc = try paged.views(gpa, d, .value);
        errdefer gpa.free(vc);
        const ikc = try paged.views(gpa, d, .index);
        errdefer gpa.free(ikc);
        const pooled = try paged.views(gpa, d, .pooled);
        errdefer gpa.free(pooled);
        const none: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0, .borrowed = true };
        @memset(conv, none);
        @memset(state, none);
        return .{ .conv = conv, .state = state, .kc = kc, .vc = vc, .ikc = ikc, .pooled = pooled, .paged = paged, .tail = none, .history = .{ p.e.eos, p.e.eos }, .fed = 0, .logits = try gpa.alloc(u8, prompt.vocabulary * 2), .cap = 0 };
    }

    pub fn deinit(p: *Pool) void {
        cuda.completion.drain(p.e.driver, p.e.stream.*, "flashnext: the pool did not drain at close");
        for (p.lanes) |*x| {
            if (x.used) {
                p.pool.releaseRun(x.run, p.layout.runPages());
                p.pool.finish(&x.lease);
            }
            x.seq.deinit(p.gpa);
        }
        p.gpa.free(p.lanes);
        p.arena.free();
        p.pool.deinit();
        p.gpa.destroy(p);
    }

    /// Whether a lane is free, the pool can promise a cache of `cap` rows and a run now, and a run is free.
    pub fn fits(p: *const Pool, cap: usize) bool {
        if (cap > p.capacity + window_rows) return false;
        const k = p.layout.runPages();
        for (p.lanes) |x| if (!x.used) return p.pool.fits(p.layout.pages(cap) + k) and p.pool.findRun(k) != null;
        return false;
    }

    /// A free lane's cleared cache (a copy) with `cap` rows' pages and its run; LaneMemory: not now.
    pub fn open(p: *Pool, cap: usize) !prompt.Seq {
        if (cap > p.capacity + window_rows) return error.PromptTooLong;
        const x = for (p.lanes) |*x| {
            if (!x.used) break x;
        } else return error.LaneMemory;
        const k = p.layout.runPages();
        const n = p.layout.pages(cap);
        if (n > mostPages(p.capacity, p.layout.page_tokens)) return error.PromptTooLong;
        try p.pool.promise(&x.lease, n + k);
        errdefer p.pool.finish(&x.lease);
        // the run first: pages fill from the bottom, runs from the top; a fragmented pool waits (LaneMemory)
        const run = p.pool.claimRun(&x.lease, k) orelse return error.LaneMemory;
        errdefer p.pool.releaseRun(run, k);
        errdefer cuda.completion.drain(p.e.driver, p.e.stream.*, "flashnext: a failed lane open did not drain");
        var up: Upload = .{ .p = p, .table = x.seq.paged.?.table };
        try p.pool.cover(&x.lease, n, .{ .ctx = &up, .write = Upload.write });
        x.seq.paged.?.ids = x.lease.ids.items;
        x.run = run;
        try p.mapStates(&x.seq);
        try p.e.clearSeq(&x.seq, cap);
        x.used = true;
        return x.seq;
    }

    fn runBase(p: *const Pool, x: *const Lane) u64 {
        return p.arena.ptr + @as(u64, x.run) * p.layout.pageBytes();
    }

    /// The lane's states as views of its run, laid out afresh (a kept window swaps state buffers with the head's).
    fn mapStates(p: *Pool, seq: *prompt.Seq) !void {
        if (p.layout.runPages() * p.layout.pageBytes() < seqRunBytes()) return error.FlashNextRunTooShort;
        const x = p.laneOf(seq).?;
        var at = p.runBase(x);
        for (seq.conv) |*b| {
            b.* = .{ .d = p.e.driver, .ptr = at, .len = conv_bytes, .borrowed = true };
            at += al(conv_bytes);
        }
        for (seq.state) |*b| {
            b.* = .{ .d = p.e.driver, .ptr = at, .len = state_bytes, .borrowed = true };
            at += al(state_bytes);
        }
        seq.tail = .{ .d = p.e.driver, .ptr = at, .len = tail_bytes, .borrowed = true };
    }

    /// The MTP head of the stream on lane cache `seq`: its cache on the stream's pages, its buffers in the run.
    pub fn draft(p: *Pool, seq: *const prompt.Seq, cap: usize) !mtp.Draft {
        const h = p.head orelse return error.FlashNextRunTooShort;
        const x = p.laneOf(seq) orelse return error.UnknownStream;
        const run_len = p.layout.runPages() * p.layout.pageBytes();
        if (run_len < seqRunBytes()) return error.FlashNextRunTooShort;
        return mtp.Draft.initIn(p.gpa, p.e.driver, cap, h.draft_n, h.rows, &x.seq.paged.?, p.runBase(x) + seqRunBytes(), run_len - seqRunBytes());
    }

    fn laneOf(p: *Pool, seq: *const prompt.Seq) ?*Lane {
        const t = (seq.paged orelse return null).table.ptr;
        for (p.lanes) |*x| if (x.seq.paged.?.table.ptr == t) return x;
        return null;
    }

    /// A lane's stream ended: its pages and run go back once the forwards reading them are done. False: not a lane's.
    pub fn close(p: *Pool, seq: *prompt.Seq) bool {
        const x = p.laneOf(seq) orelse return false;
        if (!x.used) return true;
        cuda.completion.drain(p.e.driver, p.e.stream.*, "flashnext: a closed stream did not drain");
        p.pool.releaseRun(x.run, p.layout.runPages());
        p.pool.finish(&x.lease);
        if (seq.media) |*m| m.deinit(); // the stream's image state, held by its copy
        seq.media = null;
        x.seq.paged.?.ids = &.{};
        x.seq.cap = 0;
        x.used = false;
        return true;
    }

    /// A lane's new table entries uploaded to its device table.
    const Upload = struct {
        p: *Pool,
        table: cuda.DeviceBuffer,

        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const u: *Upload = @ptrCast(@alignCast(ctx));
            errdefer cuda.completion.drain(u.p.e.driver, u.p.e.stream.*, "flashnext: a failed table upload did not drain");
            try u.table.uploadAsync(from * 4, std.mem.sliceAsBytes(ids[from..]), u.p.e.stream.handle);
            try u.p.e.stream.synchronize(); // the ids slice may move as the lease grows
        }
    };
};

test "a stream's pages: its rows and a drafted window, no more than a full window's" {
    const l = layout(40960, 256, null, 0);
    try std.testing.expectEqual(pages.pageBytes(256), l.pageBytes());
    const h = layout(40960, 256, .{ .draft_n = 32768, .rows = 1536 }, 0);
    try std.testing.expectEqual(pages.pageBytesOf(256, wt.attn_n + 1), h.pageBytes());
    try std.testing.expect(h.run_bytes >= seqRunBytes() + mtp.Draft.runBytes(32768, 1536));
    try std.testing.expectEqual(@as(usize, 161), mostPages(40960, 256));
    try std.testing.expect(l.pages(40960 + window_rows) <= mostPages(40960, 256));
}

test {
    _ = @import("cuda_kv_init_test.zig");
    _ = @import("cuda_kv_lifetime_test.zig");
}
