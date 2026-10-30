//! A stream's K/V on the host tier: first pages and a window in frames, the rest canonical in host RAM, cached in slots
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const wt = @import("../../cuda_prompt_wt.zig");
const pages = @import("../../cuda_pages.zig");
const host_tier = @import("cuda_exl3_kvhost.zig");

const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
const attn_n = wt.attn_n;
const ctl_n: usize = 8;
const none = std.math.maxInt(u32);

/// Positions a page holds on a host tier: 64 four-row slots a lent frame and layer.
pub const page_rows: usize = 256;
/// Pages kept in frames from the first row: a dense row's keys (the first `dense_keys` positions) are all in them.
pub const head_pages: usize = (wt.dense_keys + page_rows - 1) / page_rows;
/// The most rows one attention call writes (a deep prefill's chunk and a drafted window).
pub const call_rows: usize = 4608;

/// TF_FLASHNEXT_KV_SLOTS=n: n lent frames a stream, 64 four-row slots a frame for each attention layer (default 128).
pub fn slotFrames() !usize {
    const v = std.c.getenv("TF_FLASHNEXT_KV_SLOTS") orelse return 128;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadFlashNextKvHost;
    if (n == 0) return error.BadFlashNextKvHost;
    return n;
}

/// Frames a stream may hold at once: the first pages, the window and a call's rows, the state run, the slot frames.
pub fn framesFor(run_pages: usize, slot_frames: usize) usize {
    return head_pages + (call_rows + 2 * page_rows) / page_rows + run_pages + slot_frames;
}

pub const Kernels = struct {
    module: cuda.Module,
    plan: cuda.Function,
    fetch: cuda.Function,
    remap: cuda.Function,
    /// One call's lists (need, miss, fetch), shared by the layers and streams.
    work: cuda.DeviceBuffer,

    pub fn load(d: *const cuda.Driver, nslots: usize) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.module = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_kvtier);
        errdefer k.module.unload();
        k.plan = try k.module.function("kvt_plan");
        k.fetch = try k.module.function("kvt_fetch");
        k.remap = try k.module.function("kvt_remap");
        k.work = try cuda.DeviceBuffer.alloc(d, 4 * 4 * @max(nslots, 2048));
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.work.free();
        k.module.unload();
    }
};

/// What the pool lends a stream's selection: its tables, tier lease, frame list and cache views.
pub const Own = struct {
    table: cuda.DeviceBuffer,
    itable: cuda.DeviceBuffer,
    host: *host_tier.Host,
    arena: cuda.DeviceBuffer,
    tl: *kv.tier.Lease,
    fids: *std.ArrayList(u32),
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
};

pub const LaneSel = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const Kernels,
    dev: cuda.DeviceBuffer, // blk2slot, mark [layers][nblocks]; ctl [layers][8]; slot2blk, stamp [layers][nslots]
    stage: cuda.HostBuffer, // table entries on their way to the device
    itable: u64, // the index plane's table: logical page ids
    table: u64,
    tbuf: cuda.DeviceBuffer, // the K/V table: a frame a page, then the lent slot frames
    host: *host_tier.Host,
    arena: cuda.DeviceBuffer,
    tl: *kv.tier.Lease,
    fids: *std.ArrayList(u32),
    kc: []cuda.DeviceBuffer, // each layer's K and V in page 0 of the frames
    vc: []cuda.DeviceBuffer,
    nblocks: usize,
    nslots: usize,
    npages: usize, // logical pages of the stream: the lent entries follow them in the table
    layers: usize,
    flushed: usize = 0, // pages [0, flushed) are canonical in the host store and out of the window
    win0: usize = 0, // the first row not canonical in the host store
    done: [attn_n + 1]usize = @splat(0), // rows each layer (the MTP head last) has been given to write
    head_on: bool = false, // the MTP head has written (or will write) rows of this stream
    hw: usize = 0, // pages [0, hw) have been flushed once: their host rows are valid
    reset_pending: bool = true,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const Kernels, cap: usize, npages: usize, slot_frames: usize, layers: usize, own: Own) !*LaneSel {
        const s = try gpa.create(LaneSel);
        errdefer gpa.destroy(s);
        const nblocks = (cap + ratio - 1) / ratio;
        const nslots = slot_frames * 64;
        var dev = try cuda.DeviceBuffer.alloc(d, 4 * layers * (2 * nblocks + ctl_n + 2 * nslots));
        errdefer dev.free();
        var stage = try cuda.HostBuffer.alloc(d, 4 * (npages + slot_frames));
        errdefer stage.free();
        s.* = .{ .gpa = gpa, .d = d, .k = k, .dev = dev, .stage = stage, .itable = own.itable.ptr, .table = own.table.ptr, .tbuf = own.table, .host = own.host, .arena = own.arena, .tl = own.tl, .fids = own.fids, .kc = own.kc, .vc = own.vc, .nblocks = nblocks, .nslots = nslots, .npages = npages, .layers = layers };
        return s;
    }

    pub fn deinit(s: *LaneSel) void {
        s.stage.free();
        s.dev.free();
        s.gpa.destroy(s);
    }

    /// Slots a layer's plan can take: its rows' blocks at most `top` each.
    pub fn rowsPerPlan(s: *const LaneSel) usize {
        return @max(1, s.nslots / top);
    }

    const Tab = struct { blk2slot: u64, mark: u64, ctl: u64, slot2blk: u64, stamp: u64 };

    fn tab(s: *const LaneSel, ai: usize) Tab {
        const nb = s.nblocks * 4;
        const marks = s.dev.ptr + s.layers * nb;
        const ctls = marks + s.layers * nb;
        const s2b = ctls + s.layers * ctl_n * 4;
        return .{ .blk2slot = s.dev.ptr + ai * nb, .mark = marks + ai * nb, .ctl = ctls + ai * ctl_n * 4, .slot2blk = s2b + ai * s.nslots * 4, .stamp = s2b + (s.layers + ai) * s.nslots * 4 };
    }

    fn memset32(s: *const LaneSel, ptr: u64, value: u32, words: usize, stream: cuda.Stream) !void {
        try s.d.check(s.d.api.cuMemsetD32Async(ptr, value, words, stream.handle), "cuMemsetD32Async");
    }

    /// No block is in a slot, every clock and counter at 0.
    pub fn reset(s: *LaneSel, stream: cuda.Stream) !void {
        const t = s.tab(0);
        try s.memset32(t.blk2slot, 0xFFFF_FFFF, s.layers * s.nblocks, stream);
        try s.memset32(t.mark, 0, s.layers * s.nblocks, stream);
        try s.memset32(t.ctl, 0, s.layers * ctl_n, stream);
        try s.memset32(t.slot2blk, 0xFFFF_FFFF, s.layers * s.nslots, stream);
        try s.memset32(s.tab(0).stamp, 0, s.layers * s.nslots, stream);
        s.reset_pending = false;
    }

    fn setEntry(s: *LaneSel, stream: cuda.Stream, index: usize, frame: u32) !void {
        const words = s.stage.slice(u32);
        words[index] = frame;
        try s.tbuf.uploadAsync(index * 4, std.mem.sliceAsBytes(words[index..][0..1]), stream.handle);
    }

    /// The lent frames named at the end of the K/V table: the slots' rows are rows past the stream's.
    pub fn nameSlots(s: *LaneSel, stream: cuda.Stream) !void {
        for (s.tl.lent.items[s.tl.lent.items.len - (s.nslots / 64) ..], 0..) |f, j| try s.setEntry(stream, s.npages + j, f);
    }

    /// Page `pidx` in a frame: pinned, its rows copied in from the host store when they were out.
    fn ensure(s: *LaneSel, stream: cuda.Stream, pidx: usize) !void {
        if (s.fids.items[pidx] != none) return;
        const id = s.tl.pages.ids.items[pidx];
        const fill = try s.host.tier.pin(s.tl, id);
        if (fill.miss and pidx < s.hw) {
            const at = @as(usize, fill.frame) * s.host.kv_page;
            try s.arena.uploadAsync(at, s.host.tier.host_page(id), stream.handle);
        }
        try s.host.tier.markDirty(s.tl, id); // rows of the window are written; a refilled page is flushed again, equal
        s.fids.items[pidx] = fill.frame;
        try s.setEntry(stream, pidx, fill.frame);
    }

    /// Layer `ai` writes [pos0, pos0 + n): a page leaves the window once every layer wrote it; a rewind brings it back.
    pub fn begin(s: *LaneSel, stream: cuda.Stream, ai: usize, pos0: usize, n: usize) !void {
        if (ai >= s.layers) return;
        if (s.reset_pending) {
            // a new, resumed or restored stream: rows before pos0 are canonical in the host store
            try s.reset(stream);
            s.flushed = pos0 / page_rows;
            s.win0 = s.flushed * page_rows;
            s.hw = (pos0 + page_rows - 1) / page_rows;
            s.done = @splat(pos0);
            s.head_on = false;
        }
        const first = pos0 / page_rows;
        if (first < s.flushed) {
            // a resume, a rewind or the head priming: the window restarts at this page (brought in below)
            s.flushed = first;
            s.win0 = first * page_rows;
            try s.reset(stream); // blocks at and past the rewind point hold rows that are about to change
        }
        s.done[ai] = @min(s.done[ai], pos0);
        if (ai == attn_n) s.head_on = true; // from its first call the MTP head holds pages back until it has written them
        var low = s.done[0];
        for (s.done[1..if (s.head_on) attn_n + 1 else attn_n]) |d| low = @min(low, d);
        const target = low / page_rows;
        var p = s.flushed;
        while (p < target) : (p += 1) {
            if (s.fids.items[p] == none) continue;
            const id = s.tl.pages.ids.items[p];
            try s.host.flush(s.arena, id, stream);
            s.host.done(id);
            s.hw = @max(s.hw, p + 1);
            if (p >= head_pages) {
                try s.host.tier.unpin(s.tl, id);
                s.fids.items[p] = none;
            }
        }
        if (target > s.flushed) s.flushed = target;
        s.win0 = s.flushed * page_rows;
        s.done[ai] = pos0 + n;
        const last = (pos0 + n + page_rows - 1) / page_rows;
        if (last > s.npages) return error.PromptTooLong;
        for (0..last) |q| if (q < head_pages or q >= s.flushed) try s.ensure(stream, q);
    }

    /// The snapshot restored the stream's K/V rows into the host store: no page keeps a frame, the window starts over.
    pub fn restored(s: *LaneSel) void {
        const t = &s.host.tier;
        for (s.tl.pages.ids.items, 0..) |id, p| {
            if (p < s.fids.items.len and s.fids.items[p] != none) {
                t.sync(id);
                t.unpin(s.tl, id) catch {};
                s.fids.items[p] = none;
            }
            t.invalidate(id) catch {};
        }
        s.reset_pending = true;
    }

    /// Rows [fed / page_rows * page_rows, fed) and everything before them are canonical in the host store.
    pub fn flushAll(s: *LaneSel, stream: cuda.Stream, fed: usize) !void {
        const upto = (fed + page_rows - 1) / page_rows;
        for (0..@min(upto, s.npages)) |p| {
            if (s.fids.items[p] == none) continue;
            const id = s.tl.pages.ids.items[p];
            try s.host.flush(s.arena, id, stream);
            s.host.done(id);
        }
    }

    /// One plan: `m` rows' selections in layer `ai`: older blocks to slots (fetched if absent), positions to rows.
    pub fn page(s: *LaneSel, stream: cuda.Stream, ai: usize, qids: u64, qnk: u64, qspr: u64, m: usize) !void {
        const t = s.tab(ai);
        const k = s.k;
        const need = k.work.ptr;
        const miss = need + 4 * @max(s.nslots, 2048);
        const fetch = miss + 4 * @max(s.nslots, 2048);
        var a: cuda.Args = .{};
        for ([_]u64{ qids, qnk, qspr }) |v| a.add(v);
        for ([_]usize{ m, idw, top, ratio, s.win0 }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(s.tab(0).ctl);
        a.add(t.blk2slot);
        a.add(t.blk2slot);
        a.add(@as(c_int, 0));
        for ([_]u64{ t.slot2blk, t.stamp, t.mark }) |v| a.add(v);
        a.add(@as(c_int, @intCast(s.nslots)));
        for ([_]u64{ t.ctl, need, miss, fetch }) |v| a.add(v);
        try cuda.launch.launch(k.plan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &a);
        const hk = s.host.store.device() catch return error.NoMappedHostStore;
        const layer_k = pages.planeOffset(page_rows, ai, .key);
        const layer_v = pages.planeOffset(page_rows, ai, .value);
        var f: cuda.Args = .{};
        for ([_]u64{ hk + layer_k, hk + layer_v, s.arena.ptr + layer_k, s.arena.ptr + layer_v, t.ctl, fetch, s.itable, s.table }) |v| f.add(v);
        f.add(@as(c_int, @intCast(s.npages)));
        f.add(@as(i64, @intCast(s.host.kv_page / 16)));
        f.add(@as(c_int, @intCast(pages.rowBytes(.key) / 16)));
        f.add(@as(c_int, ratio));
        try cuda.launch.launch(k.fetch, .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, stream, &f);
        var r: cuda.Args = .{};
        for ([_]u64{ qids, qnk, qspr }) |v| r.add(v);
        for ([_]usize{ idw, s.win0, s.npages * page_rows }) |v| r.add(@as(c_int, @intCast(v)));
        r.add(t.blk2slot);
        r.add(@as(c_int, ratio));
        r.add(t.ctl);
        try cuda.launch.launch(k.remap, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 256 } }, stream, &r);
    }

    /// The blocks the plans needed, and those fetched, since open (read back: a sync).
    pub fn stats(s: *const LaneSel) !struct { needed: u64, fetched: u64, overflow: bool } {
        var needed: u64 = 0;
        var fetched: u64 = 0;
        var overflow = false;
        for (0..s.layers) |ai| {
            var ctl: [ctl_n]i32 = undefined;
            try s.d.check(s.d.api.cuMemcpyDtoH_v2(&ctl, s.tab(ai).ctl, ctl_n * 4), "cuMemcpyDtoH");
            needed += @as(u32, @bitCast(ctl[4]));
            fetched += @as(u32, @bitCast(ctl[3]));
            overflow = overflow or ctl[2] != 0;
        }
        return .{ .needed = needed, .fetched = fetched, .overflow = overflow };
    }
};

/// One block of attention rows: where its queries and outputs are, its rows' selection in the batch's lists.
pub const Block = struct { q: u64, ao: u64, kc: u64, vc: u64, pos: u64, sel: usize, m: usize, ends: usize, first_row: usize };

/// Attention for `a.m` rows: plan, fetch, remap, chunks, merge; sparse rows a few at a time, within the layer's slots.
pub fn attend(s: *LaneSel, b: anytype, ai: usize, a: Block) !void {
    const st = b.s;
    const per = if (a.first_row / ratio > top) s.rowsPerPlan() else a.m;
    var first: usize = 0;
    while (first < a.m) : (first += per) {
        const cnt = @min(per, a.m - first);
        const ids = st.qids + (a.sel + first) * idw * 4;
        const nk = st.qnk + (a.sel + first) * 4;
        const spr = st.qspr + (a.sel + first) * 4;
        if (a.ends / ratio > top) try s.page(b.stream, ai, ids, nk, spr, cnt);
        const chunks = (@min(a.ends, wt.dense_keys) + 511) / 512;
        try b.tri.attnChunks(a.q + first * wt.q_heads * wt.head_dim * 2, a.kc, a.vc, a.pos, st.po, st.pm, st.pl, ids, nk, spr, cnt, wt.kv_heads, chunks, true, b.seq.pg);
        try b.tri.attnMerge(st.po, st.pm, st.pl, a.pos, a.ao + first * wt.out_k * 2, nk, spr, cnt, wt.kv_heads, true);
    }
}
