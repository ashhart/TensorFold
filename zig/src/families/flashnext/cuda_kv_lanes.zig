//! Flash Next's streams beside the first take attention-cache pages from the core pool (TF_FLASHNEXT_KV_PAGES=n).
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const prompt = @import("cuda_prompt.zig");
const pages = @import("cuda_pages.zig");
const mtp = @import("cuda_mtp.zig");
const snap = @import("cuda_snap.zig");
const wt = @import("cuda_prompt_wt.zig");
const host_tier = @import("exl3/kv/cuda_exl3_kvhost.zig");
const kvt = @import("exl3/kv/cuda_exl3_kvtier.zig");

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
    // a run is whole pages (a host tier: of the K/V plane, which alone takes fewer bytes)
    const page = pages.pageBytesOf(tokens, layers);
    const run_page = if (host_tier.tiered()) pages.kvPageBytes(tokens, layers) else page;
    const run_pages = (run + run_page - 1) / run_page;
    return .{ .fixed_bytes = mostPages(capacity, tokens) * 4 + extra, .token_bytes = page / tokens, .page_tokens = @intCast(tokens), .snap_bytes = snap.bytes(0, head != null), .run_bytes = run_pages * page };
}

/// `l` with every stream's run `k` pages: the ranks of a pair agree the larger, so their pools open the same ids.
pub fn withRunPages(l: kv.Layout, k: usize) kv.Layout {
    var out = l;
    out.run_bytes = k * l.pageBytes();
    return out;
}

/// The lanes' pool on `own`; a pair's ranks agree lanes, pages and run pages, so their page ids are equal.
pub fn attachTo(own: anytype, streams: u32, count: usize) !void {
    var lay = own.kv.layout.?;
    var lane_count = streams;
    var page_count = count;
    if (own.ranks) |r| {
        lane_count = @intCast(try r.room(streams));
        page_count = @intCast(try r.room(count));
        lay = withRunPages(lay, @intCast(~(try r.room(~@as(u64, lay.runPages())))));
    }
    own.kv.layout = lay;
    own.kv.pool = try Pool.init(own.gpa, own.engine, lay, own.context, own.kv.head, lane_count, page_count);
    own.kv.paged_only = pagedOnly() and lane_count > 0;
    if (own.ranks) |r| r.pool = own.kv.pool;
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
const Lane = struct {
    seq: prompt.Seq,
    tl: kv.tier.Lease = .{}, // the stream's reservation: its pages (and, on a host tier, frames)
    used: bool = false,
    run: u32 = 0, // the first page (or frame, on a host tier) of its run
    fids: std.ArrayList(u32) = .empty, // on a host tier: the frame of each page's K/V rows (none: out of frames)
    sel: ?*kvt.LaneSel = null, // on a host tier: its selection of blocks in slots
    itable: cuda.DeviceBuffer = undefined, // on a host tier: the index plane's table, logical page ids
};

pub const Pool = struct {
    gpa: std.mem.Allocator,
    e: *prompt.Engine,
    layout: kv.Layout,
    capacity: usize,
    head: ?Head,
    plain: kv.Pool, // the pool when there is no host tier
    pool: *kv.Pool, // `plain`, or the host tier's logical pages
    host: ?*host_tier.Host = null,
    kern: ?kvt.Kernels = null,
    slot_frames: usize = 0,
    planes: bool = false, // a host tier: the arena holds the K/V plane (frames), then the index plane (logical pages)
    arena: cuda.DeviceBuffer,
    ibase: u64, // where the index plane starts in the arena: after `count` pages of the K/V plane
    lanes: []Lane,

    /// `streams` lanes and `count` pages of layout `l`, made and touched now; a lane holds only its table idle.
    pub fn init(gpa: std.mem.Allocator, e: *prompt.Engine, l: kv.Layout, capacity: usize, head: ?Head, streams: usize, count: usize) !*Pool {
        if (!pages.valid(l.page_tokens) or capacity > prompt.max_tokens or count > @as(usize, std.math.maxInt(c_int)) + 1) return error.BadFlashNextKvPages;
        const page_bytes = try std.math.mul(usize, @intCast(l.token_bytes), l.page_tokens);
        if (page_bytes == 0) return error.BadFlashNextKvPages;
        _ = try std.math.add(u64, l.run_bytes, page_bytes - 1);
        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);
        const layers = wt.attn_n + @intFromBool(head != null);
        const kv_page = pages.kvPageBytes(l.page_tokens, layers);
        const ix_page = pages.indexPageBytes(l.page_tokens, layers);
        // a host tier: `logical` pages in host RAM, `frames` of them on the device at once; else every page is a frame
        const mib = try host_tier.hostMiB();
        const slot_frames = if (mib > 0) try kvt.slotFrames() else 0;
        const per_stream = kvt.framesFor(l.runPages(), slot_frames);
        const mine_pages = if (mib > 0) mib * (1 << 20) / kv_page else count;
        const mine_frames = if (mib > 0) (try host_tier.framesCap()) orelse streams * per_stream else count;
        // two ranks hold the same logical pages and frames: the smaller of each
        const agreed = if (mib > 0) try host_tier.agreeShape(e, mine_pages, mine_frames) else kv.tier.Shape{ .host_pages = mine_pages, .frames = mine_frames };
        const logical = agreed.host_pages;
        const frames = agreed.frames;
        if (logical == 0 or frames == 0 or (mib > 0 and frames < per_stream)) return error.BadFlashNextKvHost;
        var arena = try cuda.DeviceBuffer.alloc(e.driver, try std.math.add(usize, try std.math.mul(usize, frames, kv_page), try std.math.mul(usize, logical, ix_page)));
        errdefer arena.free();
        p.* = .{ .gpa = gpa, .e = e, .layout = l, .capacity = capacity, .head = head, .plain = undefined, .pool = undefined, .arena = arena, .ibase = arena.ptr + frames * kv_page, .lanes = &.{} };
        p.planes = mib > 0;
        if (mib > 0) {
            p.host = try host_tier.Host.create(gpa, e.driver, logical, frames, kv_page);
            p.kern = try kvt.Kernels.load(e.driver, slot_frames * 64);
            p.slot_frames = slot_frames;
            std.log.info("flashnext: host tier: {d} logical pages ({d} MiB pinned), {d} frames on the device", .{ logical, logical * kv_page >> 20, frames });
            p.pool = &p.host.?.tier.pool;
        } else {
            p.plain = try kv.Pool.init(gpa, count, page_bytes);
            p.pool = &p.plain;
        }
        errdefer if (p.host) |h| h.destroy(gpa) else p.plain.deinit();
        errdefer if (p.kern) |*kn| kn.unload();
        if (p.host) |h| try h.lockstep(e, slot_frames);
        p.lanes = try gpa.alloc(Lane, streams);
        errdefer gpa.free(p.lanes);
        var made: usize = 0;
        errdefer for (p.lanes[0..made]) |*x| x.seq.deinit(gpa);
        for (p.lanes) |*x| {
            x.* = .{ .seq = try p.laneSeq() };
            made += 1;
            if (p.host != null) {
                x.itable = try cuda.DeviceBuffer.alloc(e.driver, mostPages(capacity, l.page_tokens) * 4);
                x.seq.paged.?.itable = x.itable.ptr;
                x.sel = try kvt.LaneSel.init(gpa, e.driver, &p.kern.?, capacity + window_rows, mostPages(capacity, l.page_tokens), slot_frames, layers, p.own(x));
                x.seq.sel = x.sel;
            }
        }
        errdefer cuda.completion.drain(e.driver, e.stream.*, "flashnext: failed pool initialization did not drain");
        try arena.fill8(0, e.stream.handle);
        for (p.lanes) |x| try x.seq.paged.?.table.fill8(0, e.stream.handle);
        if (p.host != null) for (p.lanes) |x| try x.itable.fill8(0, e.stream.handle);
        try e.stream.synchronize();
        return p;
    }

    /// A lane's cache: views only (states mapped from its run at open, attention caches on the arena) and a table.
    fn laneSeq(p: *Pool) !prompt.Seq {
        const gpa = p.gpa;
        const d = p.e.driver;
        var table = try cuda.DeviceBuffer.alloc(d, (mostPages(p.capacity, p.layout.page_tokens) + p.slot_frames) * 4);
        errdefer table.free();
        const paged: pages.Paged = .{ .base = p.arena.ptr, .ibase = p.ibase, .planes = p.planes, .hbase = if (p.host) |h| (h.store.device() catch 0) else 0, .table = table, .ids = &.{}, .tokens = p.layout.page_tokens, .own_ids = false, .layers = wt.attn_n + @intFromBool(p.head != null) };
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
            if (x.used) p.giveBack(x);
            if (x.sel) |sel| sel.deinit();
            if (p.host != null) x.itable.free();
            x.fids.deinit(p.gpa);
            x.seq.deinit(p.gpa);
        }
        p.gpa.free(p.lanes);
        p.arena.free();
        if (p.kern) |*kn| kn.unload();
        if (p.host) |h| h.destroy(p.gpa) else p.plain.deinit();
        p.gpa.destroy(p);
    }

    /// Whether a lane is free, the pool can promise a cache of `cap` rows and a run now, and a run is free.
    pub fn fits(p: *const Pool, cap: usize) bool {
        if (cap > p.capacity + window_rows) return false;
        const k = p.layout.runPages();
        const n = p.layout.pages(cap);
        if (p.host) |h| {
            for (p.lanes) |x| if (!x.used) return h.tier.fits(n, n + k);
            return false;
        }
        for (p.lanes) |x| if (!x.used) return p.pool.fits(n + k) and p.pool.findRun(k) != null;
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
        var run: u32 = undefined;
        var up: Upload = .{ .p = p, .x = x, .table = x.seq.paged.?.table };
        if (p.host) |h| {
            // pages in host RAM and a frame each, the run in consecutive frames of its own
            h.tier.reserve(&x.tl, n, kvt.framesFor(k, p.slot_frames)) catch return error.LaneMemory;
        } else try p.pool.promise(&x.tl.pages, n + k);
        errdefer if (p.host) |h| h.tier.finish(&x.tl) else p.pool.finish(&x.tl.pages);
        // the run first: pages fill from the bottom, runs from the top; a fragmented pool waits (LaneMemory)
        if (p.host) |h| {
            run = h.tier.lendRun(&x.tl, k) catch return error.LaneMemory;
            h.tier.lend(&x.tl, p.slot_frames) catch return error.LaneMemory;
        } else run = p.pool.claimRun(&x.tl.pages, k) orelse return error.LaneMemory;
        errdefer if (p.host == null) p.pool.releaseRun(run, k);
        errdefer p.drainFailed();
        if (p.host) |h| {
            x.fids.clearRetainingCapacity();
            try h.tier.cover(&x.tl, n, .{ .ctx = &up, .write = Upload.writeFrames });
            x.sel.?.npages = n;
            x.sel.?.reset_pending = true;
            try x.sel.?.nameSlots(p.e.stream.*);
        } else try p.pool.cover(&x.tl.pages, n, .{ .ctx = &up, .write = Upload.write });
        x.seq.paged.?.ids = x.tl.pages.ids.items;
        x.seq.paged.?.fids = x.fids.items;
        x.run = run;
        try p.mapStates(&x.seq);
        try p.e.clearSeq(&x.seq, cap);
        x.used = true;
        return x.seq;
    }

    /// Bytes a run's pages hold: whole pages of the K/V plane.
    fn runLen(p: *const Pool) usize {
        return p.layout.runPages() * p.runPage();
    }

    /// Bytes of one page a run holds: a host tier's frames hold the K/V plane alone.
    fn runPage(p: *const Pool) usize {
        const layers = wt.attn_n + @intFromBool(p.head != null);
        return if (p.host != null) pages.kvPageBytes(p.layout.page_tokens, layers) else pages.pageBytesOf(p.layout.page_tokens, layers);
    }

    fn runBase(p: *const Pool, x: *const Lane) u64 {
        return p.arena.ptr + @as(u64, x.run) * p.runPage();
    }

    /// The lane's states as views of its run, laid out afresh (a kept window swaps state buffers with the head's).
    fn mapStates(p: *Pool, seq: *prompt.Seq) !void {
        if (p.runLen() < seqRunBytes()) return error.FlashNextRunTooShort;
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
        const run_len = p.runLen();
        if (run_len < seqRunBytes()) return error.FlashNextRunTooShort;
        var d = try mtp.Draft.initIn(p.gpa, p.e.driver, cap, h.draft_n, h.rows, &x.seq.paged.?, p.runBase(x) + seqRunBytes(), run_len - seqRunBytes());
        d.sel = x.sel;
        return d;
    }

    /// Whether `seq` is a lane's cache.
    pub fn owns(p: *Pool, seq: *const prompt.Seq) bool {
        return p.laneOf(seq) != null;
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
        p.giveBack(x);
        if (seq.media) |*m| m.deinit(); // the stream's image state, held by its copy
        seq.media = null;
        x.seq.paged.?.ids = &.{};
        x.seq.paged.?.fids = &.{};
        x.seq.cap = 0;
        x.used = false;
        return true;
    }

    /// What a lane's selection moves: its tables, lease, frames and cache views.
    fn own(p: *Pool, x: *Lane) kvt.Own {
        return .{ .table = x.seq.paged.?.table, .itable = x.itable, .host = p.host.?, .arena = p.arena, .tl = &x.tl, .fids = &x.fids, .kc = x.seq.kc, .vc = x.seq.vc };
    }

    fn drainFailed(p: *Pool) void {
        cuda.completion.drain(p.e.driver, p.e.stream.*, "flashnext: a failed lane open did not drain");
    }

    /// A lane's pages, frames and run go back (a host tier: the pages the prompt store keeps are made canonical first).
    fn giveBack(p: *Pool, x: *Lane) void {
        if (p.host) |h| {
            const t = &h.tier;
            var flushed = false;
            for (x.tl.pages.ids.items) |id| if (t.pool.refs[id] > 1 and t.dirty[id]) {
                h.flush(p.arena, id, p.e.stream.*) catch std.debug.panic("flashnext: a kept page could not reach the host store", .{});
                flushed = true;
            };
            if (flushed) p.e.stream.synchronize() catch std.debug.panic("flashnext: the host store flush did not finish", .{});
            for (x.tl.pages.ids.items) |id| if (t.pool.refs[id] > 1) h.done(id);
            if (x.sel) |sel| if (sel.stats()) |st| {
                const c = t.census();
                std.log.info("flashnext: host tier stream: {d} pages ({d} flushed), {d} lent frames, sparse blocks needed {d} fetched {d}{s}; frames {d} resident {d} free", .{ x.tl.pages.ids.items.len, sel.flushed, x.tl.lent.items.len, st.needed, st.fetched, if (st.overflow) " OVERFLOW" else "", c.resident, c.free });
            } else |_| {};
            t.finish(&x.tl);
            x.fids.clearRetainingCapacity();
        } else {
            p.pool.releaseRun(x.run, p.layout.runPages());
            p.pool.finish(&x.tl.pages);
        }
    }

    /// A lane's new table entries uploaded to its device table.
    const Upload = struct {
        p: *Pool,
        x: *Lane,
        table: cuda.DeviceBuffer,

        /// A host tier: the index table gets the page ids; the pages take frames as the stream's window reaches them.
        fn writeFrames(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const u: *Upload = @ptrCast(@alignCast(ctx));
            for (ids[from..]) |_| try u.x.fids.append(u.p.gpa, std.math.maxInt(u32));
            errdefer cuda.completion.drain(u.p.e.driver, u.p.e.stream.*, "flashnext: a failed table upload did not drain");
            try u.x.itable.uploadAsync(from * 4, std.mem.sliceAsBytes(ids[from..]), u.p.e.stream.handle);
            try u.p.e.stream.synchronize(); // the ids slice may move as the lease grows
        }

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
