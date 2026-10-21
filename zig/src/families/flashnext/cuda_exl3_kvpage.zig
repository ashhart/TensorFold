//! Dynamic KV paging for long contexts (docs/plans/exl3-kv-paging.md): a sequence's full K/V of the 12 attention
//! layers in pinned host memory, the card holding a recent dense window and an LRU-like cache of the 4-key blocks
//! the sparse indexer selects. Upstream's attention kernels are unchanged: they read K/V through the rows' key lists,
//! so the paged path rewrites each list into positions of one compact device buffer [window | block slots].
//!
//! Also the selection trace (TF_FLASHNEXT_KVPAGE_TRACE): every attention row's selected blocks, recorded after
//! fn_qsa_select, for the offline cache replay (tools/exl3_kvpage_replay.py).
//!
//! The hook in the pass is one line at the top of cuda_batch.zig attnRows (`takes` / `attnRows` here).

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");
pub const shared_contract = @import("exl3_shared_kv.zig");
pub const shared_device = @import("cuda_exl3_shared_kv.zig");

test "exl3: shared KV host ownership contracts" {
    _ = shared_contract;
    _ = shared_device;
}

const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;
const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
const att_rows = parts.att_rows;
/// One token's K (or V) in one attention layer: [2][256] bf16.
pub const row_bytes: usize = kv_heads * head_dim * 2;

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

// -- the selection trace --------------------------------------------------------------------------------------------

/// Records: u32 layer (attention index), u32 first position, u32 rows, u32 kind (0 prompt, 1 decode round); then a
/// row each: u16 count (0: the row is dense, else 512) and that many u16 block ids (block = key / 4), in block order.
pub const Trace = struct {
    gpa: std.mem.Allocator,
    out: std.ArrayList(u8) = .empty,
    /// Record prompt rows too (else only decode rounds).
    prompt: bool = true,
    rows: usize = 0,
    ids: []i32 = &.{},
    nk: []i32 = &.{},
    sp: []i32 = &.{},

    pub fn init(gpa: std.mem.Allocator, prompt: bool) !Trace {
        return .{ .gpa = gpa, .prompt = prompt, .ids = try gpa.alloc(i32, att_rows * idw), .nk = try gpa.alloc(i32, att_rows), .sp = try gpa.alloc(i32, att_rows) };
    }

    pub fn deinit(t: *Trace) void {
        t.out.deinit(t.gpa);
        t.gpa.free(t.ids);
        t.gpa.free(t.nk);
        t.gpa.free(t.sp);
    }

    fn word(t: *Trace, v: u32) !void {
        try t.out.appendSlice(t.gpa, std.mem.asBytes(&v));
    }

    /// The `m` rows at `pos` the select kernel just listed in s.qids / qnk / qspr (one sync and three small copies).
    fn record(t: *Trace, stream: cuda.Stream, ai: usize, pos: usize, m: usize, decode: bool, qids: u64, qnk: u64, qspr: u64) !void {
        if (!decode and !t.prompt) return;
        try stream.synchronize();
        const d = stream.d;
        try d.check(d.api.cuMemcpyDtoH_v2(t.ids.ptr, qids, m * idw * 4), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(t.nk.ptr, qnk, m * 4), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(t.sp.ptr, qspr, m * 4), "cuMemcpyDtoH");
        try t.word(@intCast(ai));
        try t.word(@intCast(pos));
        try t.word(@intCast(m));
        try t.word(@intFromBool(decode));
        for (0..m) |r| {
            if (t.sp[r] == 0) {
                try t.out.appendSlice(t.gpa, &[2]u8{ 0, 0 });
                continue;
            }
            const cnt: u16 = @intCast(top);
            try t.out.appendSlice(t.gpa, std.mem.asBytes(&cnt));
            const row = t.ids[r * idw ..][0 .. top * ratio];
            for (0..top) |j| {
                const b: u16 = @intCast(@divTrunc(row[j * ratio], @as(i32, ratio)));
                try t.out.appendSlice(t.gpa, std.mem.asBytes(&b));
            }
        }
        t.rows += m;
    }
};

/// The active trace, if any (a test sets it; the pass reads it).
pub var trace: ?*Trace = null;

// -- the pass hook --------------------------------------------------------------------------------------------------

/// Whether attnRows runs here: a trace is on, or the sequence's K/V are paged.
pub fn takes(seq: anytype) bool {
    return trace != null or seq.paged != null;
}

/// cuda_batch.zig attnRows (rope, cache writes, pooling, then each 256-row block's scores, selection and attention)
/// with the trace after each selection, or with the sequence's paged K/V. `at` is the pass's RowsAt.
pub fn attnRows(p: anytype, lw: anytype, ai: usize, pa: u64, n: usize, at: anytype) !void {
    if (p.seq.media != null) return error.KvPageMedia;
    if (p.seq.paged) |pg| return pg.attnRows(p, lw, ai, pa, n, at);
    const s = p.s;
    const kc = p.seq.kc[ai].ptr;
    const vc = p.seq.vc[ai].ptr;
    const ikc = p.seq.ikc[ai].ptr;
    const pooled = p.seq.pooled[ai].ptr;
    try p.tri.attnPrep(pa, at.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, at.q, kc, vc, at.iq, ikc, s.dummy, s.dummy, wt.eps, n, 0);
    try p.tri.qsaPool(ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, wt.eps, n);
    const nb = parts.scoreStride(p.seq.cap);
    var r0: usize = 0;
    var b: usize = 0;
    while (r0 < n) : ({
        r0 += att_rows;
        b += 1;
    }) {
        const m = @min(att_rows, n - r0);
        const ends = p.pos0 + r0 + m;
        if (ends > p.seq.cap) return error.UnexpectedTensor;
        const pos = at.pos + b * 16;
        try select(p, at.iq + r0 * wt.index_heads * wt.index_dim * 2, pooled, pos, nb, m, ends);
        if (trace) |t| try t.record(p.stream, ai, p.pos0 + r0, m, p.lanes.len > 0, s.qids, s.qnk, s.qspr);
        const chunks = cdiv(@min(ends, wt.dense_keys), 512);
        try p.tri.attnChunks(at.q + r0 * wt.q_heads * head_dim * 2, kc, vc, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, m, kv_heads, chunks, true);
        try p.tri.attnMerge(s.po, s.pm, s.pl, pos, at.ao + r0 * wt.out_k * 2, s.qnk, s.qspr, m, kv_heads, true);
    }
}

/// Block scores (rows past `top` complete blocks) and fn_qsa_select into s.qids / qnk / qspr, as upstream.
pub fn select(p: anytype, iq: u64, pooled: u64, pos: u64, nb: usize, m: usize, ends: usize) !void {
    const s = p.s;
    if (ends / ratio > top) try p.tri.qsaScores(iq, pooled, pos, p.seq.scores, nb, m, @min(nb, cdiv(ends, ratio)));
    var a: cuda.Args = .{};
    a.add(p.seq.scores);
    a.add(pos);
    a.add(s.qids);
    a.add(s.qnk);
    a.add(s.qspr);
    for ([_]usize{ nb, idw, top, ratio }) |v| a.add(cint(v));
    try cuda.launch.launch(p.k.select, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1024 } }, p.stream, &a);
}

// -- the paged store ------------------------------------------------------------------------------------------------

/// Shape of a paged sequence's device side (TF_FLASHNEXT_KVPAGE=1 turns it on for every new sequence; the sizes
/// from TF_FLASHNEXT_KVPAGE_WINDOW / _SLOTS / _ROWS).
pub const Config = struct {
    /// Tokens kept dense on the card before a call's first row (>= 2051, so a dense row's keys are all in it).
    window: usize = 4096,
    /// 4-key block slots, one pool for the 12 layers (the prompt's layer-major order gives one layer all of it).
    slots: usize = 49152,
    /// The most rows one attention call writes (the engine's chunk / a deep prefill's sub-block).
    rows: usize = 1536,
    shared_lanes: usize = 0,

    pub fn forChunk(cfg: Config, chunk: usize) Config {
        var out = cfg;
        out.rows = @max(out.rows, chunk);
        return out;
    }

    pub fn validate(cfg: Config) !void {
        if (cfg.window < wt.dense_keys or cfg.slots == 0 or cfg.rows == 0) return error.KvPageConfig;
        if (cfg.shared_lanes > shared_device.max_lanes) return error.KvPageConfig;
    }
};

/// A test's override of the environment (null: read it).
pub var force: ?Config = null;
pub var force_off: bool = false;

fn envUsize(name: [*:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

pub fn configFromEnv() ?Config {
    if (force_off) return null;
    if (force) |c| return c;
    const v = std.c.getenv("TF_FLASHNEXT_KVPAGE") orelse return null;
    if (!std.mem.eql(u8, std.mem.span(v), "1")) return null;
    const d: Config = .{};
    var rows = envUsize("TF_FLASHNEXT_KVPAGE_ROWS", d.rows);
    if (std.mem.eql(u8, std.mem.span(std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL") orelse "1"), "1")) {
        rows = @max(rows, envUsize("TF_FLASHNEXT_EXL3_PREFILL_ROWS", 4096));
    }
    const shared = std.mem.eql(u8, std.mem.span(std.c.getenv("TF_FLASHNEXT_KVPAGE_SHARED") orelse "0"), "1");
    const shared_lanes = if (shared) envUsize("TF_FLASHNEXT_KVPAGE_SHARED_LANES", 8) else 0;
    return .{ .window = envUsize("TF_FLASHNEXT_KVPAGE_WINDOW", d.window), .slots = envUsize("TF_FLASHNEXT_KVPAGE_SLOTS", d.slots), .rows = rows, .shared_lanes = if (shared and shared_lanes == 0) std.math.maxInt(usize) else shared_lanes };
}

pub fn configIdentity(cfg: ?Config) u64 {
    const paging = cfg orelse return 0;
    const shape = [_]u64{ paging.window, paging.rows, paging.slots, paging.shared_lanes, row_bytes, attn_n };
    return std.hash.Wyhash.hash(0, std.mem.asBytes(&shape));
}

pub fn admissionRoom(local: u64, stream_bytes: usize, cfg: ?Config) !u64 {
    const paging = cfg orelse return local;
    try paging.validate();
    if (paging.shared_lanes == 0) return local;
    return @min(local, try std.math.mul(u64, stream_bytes, paging.shared_lanes));
}

const Kern = struct {
    module: cuda.Module,
    plan: cuda.Function,
    fetch: cuda.Function,
    remap: cuda.Function,

    fn load(d: *const cuda.Driver) !Kern {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kern = undefined;
        k.module = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_kvpage);
        errdefer k.module.unload();
        k.plan = try k.module.function("kvp_plan");
        k.fetch = try k.module.function("kvp_fetch");
        k.remap = try k.module.function("kvp_remap");
        return k;
    }
};

const attn_n = wt.attn_n;
const ctl_n: usize = 8;
const work_n: usize = att_rows * top;

pub const SnapshotLayout = struct {
    cap: usize,
    cfg: Config,
    at: usize,
    wr: usize,
    nblocks: usize,
    canonical: usize,
    device: usize,
    tables: usize,
    total: usize,

    pub fn init(cap: usize, cfg: Config, at: usize) !SnapshotLayout {
        try cfg.validate();
        if (cap == 0 or at > cap) return error.KvPageRows;
        _ = try std.math.mul(usize, cap, attn_n * 2 * row_bytes);
        if (cfg.shared_lanes != 0) _ = try shared_device.Layout.init(.{ .cap = cap, .lanes = cfg.shared_lanes, .window = cfg.window, .rows = cfg.rows, .slots = cfg.slots });
        const wr = try std.math.add(usize, try std.math.add(usize, cfg.window, cfg.rows), ratio);
        const nblocks = try std.math.add(usize, (try std.math.add(usize, cap, ratio - 1)) / ratio, 1);
        const canonical = try std.math.mul(usize, at, attn_n * 2 * row_bytes);
        const rows = try std.math.add(usize, try std.math.mul(usize, attn_n, wr), if (cfg.shared_lanes == 0) try std.math.mul(usize, ratio, cfg.slots) else 0);
        const device = try std.math.mul(usize, rows, 2 * row_bytes);
        const words = if (cfg.shared_lanes == 0) try std.math.add(usize, try std.math.mul(usize, attn_n, try std.math.add(usize, try std.math.mul(usize, 2, nblocks), ctl_n)), try std.math.mul(usize, 2, cfg.slots)) else try std.math.mul(usize, attn_n, try std.math.add(usize, nblocks, ctl_n));
        const tables = try std.math.mul(usize, words, 4);
        return .{ .cap = cap, .cfg = cfg, .at = at, .wr = wr, .nblocks = nblocks, .canonical = canonical, .device = device, .tables = tables, .total = try std.math.add(usize, canonical, try std.math.add(usize, device, tables)) };
    }
};

pub const Snapshot = struct {
    layout: SnapshotLayout,
    win0: [attn_n]usize,
    rebases: [attn_n]u64,
    written: [attn_n]u64,

    pub fn validate(image: Snapshot) !void {
        const expected = try SnapshotLayout.init(image.layout.cap, image.layout.cfg, image.layout.at);
        if (!std.meta.eql(expected, image.layout)) return error.KvPageSnapshotLayout;
        for (image.win0) |origin| {
            if (origin % ratio != 0 or origin > image.layout.at or image.layout.at - origin > image.layout.wr) return error.KvPageWindow;
        }
    }
};

/// Counters of one layer (kvp_plan's ctl) plus the host's window moves.
pub const Stats = struct {
    misses: u64 = 0,
    needed: u64 = 0,
    calls: u64 = 0,
    overflow: bool = false,
    rebases: u64 = 0,
    written_rows: u64 = 0,
    epoch: u32 = 0,

    pub fn add(a: *Stats, b: Stats) void {
        a.misses += b.misses;
        a.needed += b.needed;
        a.calls += b.calls;
        a.overflow = a.overflow or b.overflow;
        a.rebases += b.rebases;
        a.written_rows += b.written_rows;
        a.epoch = @max(a.epoch, b.epoch);
    }
};

/// One sequence's K/V of the 12 attention layers: all of it in mapped pinned host memory, [layer][K|V][cap][2][256];
/// on the card, for K and for V, [12 windows of wr rows | 4 * slots rows] [2][256] bf16 and the block tables.
pub const Paged = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: Kern,
    cfg: Config,
    cap: usize,
    /// Window rows: the kept window, one call's rows, and a block's slack.
    wr: usize,
    nblocks: usize,
    host: cuda.HostBuffer,
    host_dev: u64,
    status_host: cuda.HostBuffer,
    dev: cuda.DeviceBuffer,
    /// blk2slot [12][nblocks] i32, mark [12][nblocks] u32, ctl [12][8] i32, slot2blk [slots] i32, stamp [slots] u32.
    tabs: cuda.DeviceBuffer,
    /// need, miss [att_rows * 512] i32 and fetch [2 * att_rows * 512] i32 (one call's, shared by the layers).
    work: cuda.DeviceBuffer,
    win0: [attn_n]usize = @splat(0),
    rebases: [attn_n]u64 = @splat(0),
    written: [attn_n]u64 = @splat(0),
    graph_prepared: bool = false,
    shared_arena: ?*shared_device.Arena = null,
    shared_sequence: shared_contract.Sequence = undefined,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, cfg: Config) !*Paged {
        return initStore(gpa, d, cap, cfg, null);
    }

    pub fn initShared(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, cfg: Config, arena: *shared_device.Arena) !*Paged {
        return initStore(gpa, d, cap, cfg, arena);
    }

    fn initStore(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, cfg: Config, shared: ?*shared_device.Arena) !*Paged {
        const layout = try SnapshotLayout.init(cap, cfg, 0);
        if ((cfg.shared_lanes != 0) != (shared != null)) return error.SharedKvArenaRequired;
        if (shared) |arena| {
            const options = arena.layout.options;
            if (cap != options.cap) return error.SharedKvCommonCap;
            if (d != arena.driver or cfg.shared_lanes != options.lanes or cfg.window != options.window or cfg.rows != options.rows or cfg.slots != options.slots) return error.SharedKvGeometry;
        }
        const pg = try gpa.create(Paged);
        errdefer gpa.destroy(pg);
        pg.* = .{ .gpa = gpa, .d = d, .k = undefined, .cfg = cfg, .cap = cap, .wr = layout.wr, .nblocks = layout.nblocks, .host = undefined, .host_dev = 0, .status_host = undefined, .dev = undefined, .tabs = undefined, .work = undefined };
        pg.shared_arena = shared;
        if (shared) |arena| pg.shared_sequence = try arena.register(cap) else pg.k = try Kern.load(d);
        errdefer if (shared) |arena| arena.release(pg.shared_sequence, .{ .d = d, .handle = null }) catch unreachable else pg.k.module.unload();
        pg.host = try cuda.HostBuffer.allocMapped(d, attn_n * 2 * cap * row_bytes);
        errdefer pg.host.free();
        pg.host_dev = try pg.host.device();
        pg.status_host = try cuda.HostBuffer.alloc(d, attn_n * ctl_n * 4);
        errdefer pg.status_host.free();
        pg.dev = if (shared) |arena| arena.device else try cuda.DeviceBuffer.alloc(d, layout.device);
        if (shared != null) pg.dev.borrowed = true;
        errdefer pg.dev.free();
        pg.tabs = if (shared) |arena| arena.tables else try cuda.DeviceBuffer.alloc(d, layout.tables);
        if (shared != null) pg.tabs.borrowed = true;
        errdefer pg.tabs.free();
        pg.work = if (shared) |arena| arena.work else try cuda.DeviceBuffer.alloc(d, work_n * 4 * 4);
        if (shared != null) pg.work.borrowed = true;
        errdefer pg.work.free();
        if (shared == null) {
            try pg.dev.fill8(0, null);
            try pg.tabs.fill8(0, null);
            try pg.resetTables(null);
        }
        return pg;
    }

    pub fn destroy(pg: *Paged) void {
        if (pg.cfg.shared_lanes != 0) pg.shared_arena.?.release(pg.shared_sequence, .{ .d = pg.d, .handle = null }) catch @panic("shared KV lane retirement failed");
        pg.work.free();
        pg.tabs.free();
        pg.dev.free();
        pg.status_host.free();
        pg.host.free();
        if (pg.cfg.shared_lanes == 0) pg.k.module.unload();
        pg.gpa.destroy(pg);
    }

    /// Device bytes of the K/V side (windows and slots).
    pub fn devBytes(pg: *const Paged) usize {
        if (pg.cfg.shared_lanes != 0) return 0;
        return 2 * pg.rowsDev() * row_bytes;
    }

    /// Device bytes in all: K/V side, tables and the call's work lists (deviceBytesFor, a host test checks).
    pub fn deviceBytes(pg: *const Paged) usize {
        if (pg.cfg.shared_lanes != 0) return 0;
        return pg.devBytes() + pg.tabWords() * 4 + work_n * 4 * 4;
    }

    /// Rows of the K (or V) buffer: the 12 windows, then the slots.
    fn rowsDev(pg: *const Paged) usize {
        if (pg.cfg.shared_lanes != 0) return pg.shared_arena.?.layout.side_rows;
        return attn_n * pg.wr + ratio * pg.cfg.slots;
    }

    fn tabWords(pg: *const Paged) usize {
        return attn_n * (2 * pg.nblocks + ctl_n) + 2 * pg.cfg.slots;
    }

    /// Layer `ai`'s window base in the K buffer (its kernels index from here).
    fn devK(pg: *const Paged, ai: usize) u64 {
        if (pg.cfg.shared_lanes != 0) return pg.dev.ptr + (pg.shared_arena.?.layout.windowOffset(pg.shared_sequence.lane, ai) catch unreachable) * row_bytes;
        return pg.dev.ptr + ai * pg.wr * row_bytes;
    }

    fn devV(pg: *const Paged, ai: usize) u64 {
        if (pg.cfg.shared_lanes != 0) return pg.devK(ai) + pg.rowsDev() * row_bytes;
        return pg.dev.ptr + (pg.rowsDev() + ai * pg.wr) * row_bytes;
    }

    /// Rows from layer `ai`'s window base to the first slot.
    fn slotBase(pg: *const Paged, ai: usize) usize {
        if (pg.cfg.shared_lanes != 0) return pg.shared_arena.?.layout.slotBase(pg.shared_sequence.lane, ai) catch unreachable;
        return (attn_n - ai) * pg.wr;
    }

    fn hostK(pg: *const Paged, ai: usize) u64 {
        return pg.host_dev + (2 * ai) * pg.cap * row_bytes;
    }

    fn hostV(pg: *const Paged, ai: usize) u64 {
        return pg.host_dev + (2 * ai + 1) * pg.cap * row_bytes;
    }

    const Tab = struct { blk2slot: u64, mark: u64, ctl: u64, slot2blk: u64, stamp: u64, blk2slot_all: u64 };

    fn tab(pg: *const Paged, ai: usize) Tab {
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            const layout = arena.layout;
            const key = @as(usize, pg.shared_sequence.lane) * layout.lane_words + ai * layout.blocks;
            return .{ .blk2slot = arena.tables.ptr + key * 4, .mark = arena.tables.ptr + layout.marks + key * 4, .ctl = arena.tables.ptr + layout.controls + (@as(usize, pg.shared_sequence.lane) * attn_n + ai) * ctl_n * 4, .slot2blk = arena.tables.ptr + layout.owners, .stamp = arena.tables.ptr + layout.stamps, .blk2slot_all = arena.tables.ptr };
        }
        const base = pg.tabs.ptr;
        const nb = pg.nblocks * 4;
        const all = base;
        const marks = base + attn_n * nb;
        const ctls = marks + attn_n * nb;
        const s2b = ctls + attn_n * ctl_n * 4;
        return .{ .blk2slot = all + ai * nb, .mark = marks + ai * nb, .ctl = ctls + ai * ctl_n * 4, .slot2blk = s2b, .stamp = s2b + pg.cfg.slots * 4, .blk2slot_all = all };
    }

    fn memset32(pg: *const Paged, ptr: u64, value: u32, words: usize, stream: ?cuda.Stream) !void {
        if (stream) |s| return pg.d.check(pg.d.api.cuMemsetD32Async(ptr, value, words, s.handle), "cuMemsetD32Async");
        try pg.d.check(pg.d.api.cuMemsetD32_v2(ptr, value, words), "cuMemsetD32");
    }

    /// An empty cache with a fresh device clock and every window at 0.
    fn resetTables(pg: *Paged, stream: ?cuda.Stream) !void {
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            try arena.clear(try arena.leaseFor(pg.shared_sequence), pg.shared_sequence, stream orelse .{ .d = pg.d, .handle = null });
            pg.win0 = @splat(0);
            return;
        }
        const t = pg.tab(0);
        try pg.memset32(t.blk2slot_all, 0xFFFF_FFFF, attn_n * pg.nblocks, stream);
        try pg.memset32(t.slot2blk, 0xFFFF_FFFF, pg.cfg.slots, stream);
        try pg.memset32(t.stamp, 0, pg.cfg.slots, stream);
        try pg.memset32(pg.tabs.ptr + attn_n * pg.nblocks * 4, 0, attn_n * pg.nblocks, stream);
        try pg.memset32(t.ctl, 0, attn_n * ctl_n, stream);
        pg.win0 = @splat(0);
    }

    fn copy(pg: *const Paged, stream: cuda.Stream, dst: u64, src: u64, n: usize) !void {
        if (n == 0) return;
        try pg.d.check(pg.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream.handle), "cuMemcpyDtoDAsync");
    }

    /// Moves the window so rows [pos0, pos0 + n) fit: the rows that leave it are written out to host memory, the
    /// last `window` tokens before pos0 move to the window's start (pieces no longer than the shift: no overlap).
    fn rebase(pg: *Paged, ai: usize, pos0: usize, stream: cuda.Stream) !void {
        const w0 = pg.win0[ai];
        const new0 = (pos0 -| pg.cfg.window) / ratio * ratio;
        if (new0 <= w0) return error.KvPageWindow;
        const shift = new0 - w0;
        if (pos0 < w0 or pos0 > w0 + pg.wr or shift > pg.wr) return error.KvPageWindow;
        for ([_][2]u64{ .{ pg.devK(ai), pg.hostK(ai) }, .{ pg.devV(ai), pg.hostV(ai) } }) |dh| {
            try pg.copy(stream, dh[1] + w0 * row_bytes, dh[0], shift * row_bytes);
            const len = pos0 - new0;
            var at: usize = 0;
            while (at < len) {
                const piece = @min(shift, len - at);
                try pg.copy(stream, dh[0] + at * row_bytes, dh[0] + (shift + at) * row_bytes, piece * row_bytes);
                at += piece;
            }
        }
        pg.win0[ai] = new0;
        pg.rebases[ai] += 1;
        pg.written[ai] += shift;
    }

    /// Every row the sequence holds written out to host memory (the window's rows too): host K/V then equal what an
    /// all-on-device cache would hold, for a check or a snapshot.
    pub fn flush(pg: *Paged, stream: cuda.Stream, fed: usize) !void {
        _ = try pg.snapshot(fed);
        for (0..attn_n) |ai| try pg.flushLayer(stream, ai, fed);
    }

    pub fn snapshot(pg: *const Paged, at: usize) !Snapshot {
        try pg.validateSharedGeometry();
        const image: Snapshot = .{ .layout = try SnapshotLayout.init(pg.cap, pg.cfg, at), .win0 = pg.win0, .rebases = pg.rebases, .written = pg.written };
        try image.validate();
        if (pg.wr != image.layout.wr or pg.nblocks != image.layout.nblocks or (pg.cfg.shared_lanes == 0 and (pg.dev.len != image.layout.device or pg.tabs.len != image.layout.tables)) or pg.host.bytes.len != attn_n * 2 * pg.cap * row_bytes) return error.KvPageSnapshotLayout;
        return image;
    }

    pub fn validateSnapshot(pg: *const Paged, image: Snapshot) !void {
        try pg.validateSharedGeometry();
        try image.validate();
        const expected = try SnapshotLayout.init(pg.cap, pg.cfg, image.layout.at);
        if (!std.meta.eql(expected, image.layout) or pg.wr != expected.wr or pg.nblocks != expected.nblocks or (pg.cfg.shared_lanes == 0 and (pg.dev.len != expected.device or pg.tabs.len != expected.tables)) or pg.host.bytes.len != attn_n * 2 * pg.cap * row_bytes) return error.KvPageSnapshotLayout;
    }

    fn validateSharedGeometry(pg: *const Paged) !void {
        if (pg.cfg.shared_lanes == 0) return;
        const arena = pg.shared_arena orelse return error.KvPageSnapshotLayout;
        const expected = try shared_device.Layout.init(.{ .cap = pg.cap, .lanes = pg.cfg.shared_lanes, .window = pg.cfg.window, .rows = pg.cfg.rows, .slots = pg.cfg.slots });
        if (!std.meta.eql(expected, arena.layout) or pg.dev.ptr != arena.device.ptr or pg.dev.len != arena.device.len or pg.tabs.ptr != arena.tables.ptr or pg.tabs.len != arena.tables.len or pg.work.ptr != arena.work.ptr or pg.work.len != arena.work.len) return error.KvPageSnapshotLayout;
    }

    pub fn moveSnapshot(pg: *Paged, stream: cuda.Stream, base: u64, image: Snapshot, saving: bool) !void {
        var offset: usize = 0;
        const prefix = image.layout.at * row_bytes;
        for (0..attn_n) |ai| for ([_]u64{ pg.hostK(ai), pg.hostV(ai) }) |address| {
            if (saving) try pg.copy(stream, base + offset, address, prefix) else try pg.copy(stream, address, base + offset, prefix);
            offset += prefix;
        };
        if (pg.cfg.shared_lanes != 0) {
            for (0..attn_n) |ai| for ([_]u64{ pg.devK(ai), pg.devV(ai) }) |address| {
                const size = pg.wr * row_bytes;
                if (saving) try pg.copy(stream, base + offset, address, size) else try pg.copy(stream, address, base + offset, size);
                offset += size;
            };
            const tables = pg.tab(0);
            for ([_][2]u64{ .{ tables.mark, attn_n * pg.nblocks * 4 }, .{ tables.ctl, attn_n * ctl_n * 4 } }) |section| {
                if (saving) try pg.copy(stream, base + offset, section[0], @intCast(section[1])) else try pg.copy(stream, section[0], base + offset, @intCast(section[1]));
                offset += @intCast(section[1]);
            }
            std.debug.assert(offset == image.layout.total);
            return;
        }
        for ([_]cuda.DeviceBuffer{ pg.dev, pg.tabs }) |buffer| {
            if (saving) try pg.copy(stream, base + offset, buffer.ptr, buffer.len) else try pg.copy(stream, buffer.ptr, base + offset, buffer.len);
            offset += buffer.len;
        }
        std.debug.assert(offset == image.layout.total);
    }

    pub fn publishSnapshot(pg: *Paged, image: Snapshot) void {
        pg.win0 = image.win0;
        pg.rebases = image.rebases;
        pg.written = image.written;
        pg.graph_prepared = false;
    }

    pub fn flushLayer(pg: *Paged, stream: cuda.Stream, ai: usize, fed: usize) !void {
        if (fed > pg.cap) return error.KvPageRows;
        const w0 = pg.win0[ai];
        if (fed <= w0) return;
        if (fed - w0 > pg.wr) return error.KvPageWindow;
        try pg.copy(stream, pg.hostK(ai) + w0 * row_bytes, pg.devK(ai), (fed - w0) * row_bytes);
        try pg.copy(stream, pg.hostV(ai) + w0 * row_bytes, pg.devV(ai), (fed - w0) * row_bytes);
    }

    /// The host copy of layer `ai`'s K (v false) or V rows (after `flush`).
    pub fn hostRows(pg: *const Paged, ai: usize, v: bool) []const u8 {
        const off = (2 * ai + @intFromBool(v)) * pg.cap * row_bytes;
        return pg.host.bytes[off..][0 .. pg.cap * row_bytes];
    }

    pub fn stats(pg: *const Paged, ai: usize) !Stats {
        var c: [ctl_n]i32 = undefined;
        try pg.d.check(pg.d.api.cuMemcpyDtoH_v2(&c, pg.tab(ai).ctl, ctl_n * 4), "cuMemcpyDtoH");
        return .{ .misses = @intCast(c[3]), .needed = @intCast(c[4]), .calls = @intCast(c[5]), .overflow = c[2] != 0, .rebases = pg.rebases[ai], .written_rows = pg.written[ai], .epoch = @bitCast(c[6]) };
    }

    pub fn total(pg: *const Paged) !Stats {
        var t: Stats = .{};
        for (0..attn_n) |ai| t.add(try pg.stats(ai));
        return t;
    }

    pub fn stageStatus(pg: *Paged, stream: cuda.Stream) !void {
        const ctl = pg.tab(0).ctl;
        try pg.d.check(pg.d.api.cuMemcpyDtoHAsync_v2(pg.status_host.bytes.ptr, ctl, attn_n * ctl_n * 4, stream.handle), "cuMemcpyDtoHAsync");
    }

    pub fn checkStatus(pg: *const Paged) !void {
        const controls: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, pg.status_host.bytes[0 .. attn_n * ctl_n * 4]));
        for (0..attn_n) |ai| if (controls[ai * ctl_n + 2] != 0) return error.KvPageOverflow;
    }

    /// One call's paging after fn_qsa_select listed `m` rows' keys in qids / qnk / qspr: the plan (hits stamped, the
    /// misses given the least recently used slots), the misses' rows fetched from host memory, the lists remapped.
    pub fn page(pg: *Paged, stream: cuda.Stream, ai: usize, qids: u64, qnk: u64, qspr: u64, m: usize) !void {
        if (ai >= attn_n) return error.KvPageRows;
        if (try pagingRows(pg.cfg, pg.win0[ai], m) != m) return error.KvPageCapacity;
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            return arena.pageFor(try arena.leaseFor(pg.shared_sequence), pg.shared_sequence, stream, ai, pg.win0[ai], m, qids, qnk, qspr, pg.hostK(ai), pg.hostV(ai));
        }
        const t = pg.tab(ai);
        const w0 = pg.win0[ai];
        const need = pg.work.ptr;
        const miss = need + work_n * 4;
        const fetch = miss + work_n * 4;
        var a: cuda.Args = .{};
        for ([_]u64{ qids, qnk, qspr }) |v| a.add(v);
        for ([_]usize{ m, idw, top, ratio, w0 }) |v| a.add(cint(v));
        a.add(pg.tab(0).ctl);
        a.add(t.blk2slot);
        a.add(t.blk2slot_all);
        a.add(cint(ai * pg.nblocks));
        for ([_]u64{ t.slot2blk, t.stamp, t.mark }) |v| a.add(v);
        a.add(cint(pg.cfg.slots));
        for ([_]u64{ t.ctl, need, miss, fetch }) |v| a.add(v);
        try cuda.launch.launch(pg.k.plan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &a);
        var f: cuda.Args = .{};
        for ([_]u64{ pg.hostK(ai), pg.hostV(ai), pg.devK(ai), pg.devV(ai), t.ctl, fetch }) |v| f.add(v);
        for ([_]usize{ pg.slotBase(ai), row_bytes / 16, ratio }) |v| f.add(cint(v));
        try cuda.launch.launch(pg.k.fetch, .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, stream, &f);
        var r: cuda.Args = .{};
        for ([_]u64{ qids, qnk, qspr }) |v| r.add(v);
        for ([_]usize{ idw, w0, pg.slotBase(ai) }) |v| r.add(cint(v));
        r.add(t.blk2slot);
        r.add(cint(ratio));
        r.add(t.ctl);
        try cuda.launch.launch(pg.k.remap, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 256 } }, stream, &r);
    }

    pub fn validateRange(pg: *const Paged, ai: usize, pos0: usize, n: usize) !void {
        if (ai >= attn_n or n == 0 or n > pg.cfg.rows or pos0 > pg.cap or n > pg.cap - pos0) return error.KvPageRows;
        _ = try windowPlan(pg.cfg, pg.wr, pg.win0[ai], pos0, n);
    }

    pub fn prepareLayer(pg: *Paged, stream: cuda.Stream, ai: usize, pos0: usize, n: usize) !void {
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            try arena.bindStream(try arena.leaseFor(pg.shared_sequence), stream);
        }
        try pg.validateRange(ai, pos0, n);
        if (pos0 == 0 and ai == 0) try pg.resetTables(stream);
        const w0 = try windowPlan(pg.cfg, pg.wr, pg.win0[ai], pos0, n);
        if (w0 != pg.win0[ai]) try pg.rebase(ai, pos0, stream);
    }

    pub fn prepareGraph(pg: *Paged, stream: cuda.Stream, pos0: usize, n: usize) !void {
        for (0..attn_n) |ai| try pg.validateRange(ai, pos0, n);
        for (0..attn_n) |ai| try pg.prepareLayer(stream, ai, pos0, n);
        pg.graph_prepared = true;
    }

    pub fn graphKey(pg: *const Paged) [6]u64 {
        var shape: [9 + attn_n]usize = undefined;
        shape[0..5].* = .{ pg.cap, pg.wr, pg.cfg.window, pg.cfg.rows, pg.cfg.slots };
        shape[5..8].* = if (pg.cfg.shared_lanes == 0) .{ 0, 0, 0 } else .{ pg.cfg.shared_lanes, pg.shared_sequence.lane, @intCast(pg.shared_sequence.generation) };
        shape[8] = if (pg.cfg.shared_lanes == 0) 0 else @intCast(pg.shared_arena.?.id);
        for (pg.win0, 0..) |origin, ai| shape[9 + ai] = pagingRows(pg.cfg, origin, att_rows) catch 0;
        return .{ pg.dev.ptr, pg.tabs.ptr, pg.work.ptr, pg.host_dev, std.hash.Wyhash.hash(0, std.mem.asBytes(&shape)), std.hash.Wyhash.hash(0, std.mem.asBytes(&pg.win0)) };
    }

    pub fn logicalGraphKey(pg: *const Paged) [2]u64 {
        var shape: [6 + attn_n]usize = undefined;
        shape[0..6].* = .{ pg.cap, pg.wr, pg.cfg.window, pg.cfg.rows, pg.cfg.slots, pg.cfg.shared_lanes };
        for (pg.win0, 0..) |origin, layer| shape[6 + layer] = pagingRows(pg.cfg, origin, att_rows) catch 0;
        return .{ std.hash.Wyhash.hash(0, std.mem.asBytes(&shape)), std.hash.Wyhash.hash(0, std.mem.asBytes(&pg.win0)) };
    }

    pub fn validateSharedLease(pg: *const Paged) !void {
        if (pg.cfg.shared_lanes != 0) _ = try pg.shared_arena.?.leaseFor(pg.shared_sequence);
    }

    pub fn validateSharedLeaseOn(pg: *const Paged, stream: cuda.Stream) !void {
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            try arena.bindStream(try arena.leaseFor(pg.shared_sequence), stream);
        }
    }

    /// Test support: the window's start of layer `ai` set by hand (the rows before it then come from host memory).
    pub fn setWindow(pg: *Paged, ai: usize, w0: usize) void {
        pg.win0[ai] = w0;
    }

    /// Test support: the device K (v false) or V buffer from layer `ai`'s window base: rows [0, slotRows(ai)).
    pub fn devRows(pg: *const Paged, ai: usize, v: bool) u64 {
        return if (v) pg.devV(ai) else pg.devK(ai);
    }

    /// Test support: rows from layer `ai`'s window base to the end of the slots, and the first slot's row.
    pub fn slotRows(pg: *const Paged, ai: usize) [2]usize {
        return .{ pg.slotBase(ai) + ratio * pg.cfg.slots, pg.slotBase(ai) };
    }

    /// Test support: the host K/V of layer `ai` to write directly.
    pub fn hostMut(pg: *Paged, ai: usize, v: bool) []u8 {
        const off = (2 * ai + @intFromBool(v)) * pg.cap * row_bytes;
        return pg.host.bytes[off..][0 .. pg.cap * row_bytes];
    }

    /// attnRows over the paged cache: prep writes the rows into the window (its base offset by -win0 rows), and
    /// each 256-row block's key lists go through kvp_plan / kvp_fetch / kvp_remap before upstream's chunks and merge.
    pub fn attnRows(pg: *Paged, p: anytype, lw: anytype, ai: usize, pa: u64, n: usize, at: anytype) !void {
        try pg.validateSharedLeaseOn(p.stream);
        const s = p.s;
        const pos0 = p.pos0;
        if (pg.graph_prepared) {
            try pg.validateRange(ai, pos0, n);
            if (pos0 + n > pg.win0[ai] + pg.wr) return error.KvPageWindow;
        } else try pg.prepareLayer(p.stream, ai, pos0, n);
        const w0 = pg.win0[ai];
        const kd = pg.devK(ai);
        const vd = pg.devV(ai);
        const ikc = p.seq.ikc[ai].ptr;
        const pooled = p.seq.pooled[ai].ptr;
        try p.tri.attnPrep(pa, at.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, at.q, kd -% w0 * row_bytes, vd -% w0 * row_bytes, at.iq, ikc, s.dummy, s.dummy, wt.eps, n, 0);
        try p.tri.qsaPool(ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, wt.eps, n);
        const nb = parts.scoreStride(p.seq.cap);
        var r0: usize = 0;
        var b: usize = 0;
        while (r0 < n) : ({
            r0 += att_rows;
            b += 1;
        }) {
            const m = @min(att_rows, n - r0);
            const ends = pos0 + r0 + m;
            const pos = at.pos + b * 16;
            try select(p, at.iq + r0 * wt.index_heads * wt.index_dim * 2, pooled, pos, nb, m, ends);
            if (trace) |tr| try tr.record(p.stream, ai, pos0 + r0, m, p.lanes.len > 0, s.qids, s.qnk, s.qspr);
            const chunks = cdiv(@min(ends, wt.dense_keys), 512);
            const page_rows = try pagingRows(pg.cfg, w0, m);
            var first: usize = 0;
            while (first < m) {
                const count = @min(page_rows, m - first);
                const ids = s.qids + first * idw * 4;
                const nk = s.qnk + first * 4;
                const spr = s.qspr + first * 4;
                if (ends / ratio > top) {
                    try pg.page(p.stream, ai, ids, nk, spr, count);
                } else if (w0 != 0) return error.KvPageWindow;
                try p.tri.attnChunks(at.q + (r0 + first) * wt.q_heads * head_dim * 2, kd, vd, pos, s.po, s.pm, s.pl, ids, nk, spr, count, kv_heads, chunks, true);
                try p.tri.attnMerge(s.po, s.pm, s.pl, pos, at.ao + (r0 + first) * wt.out_k * 2, nk, spr, count, kv_heads, true);
                first += count;
            }
        }
    }
};

pub fn pagingRows(cfg: Config, origin: usize, rows: usize) !usize {
    try cfg.validate();
    if (rows == 0 or rows > att_rows) return error.KvPageRows;
    if (origin % ratio != 0) return error.KvPageWindow;
    if (origin / ratio <= cfg.slots) return rows;
    const bounded = @min(rows, cfg.slots / top);
    if (bounded == 0) return error.KvPageCapacity;
    return bounded;
}

pub fn windowPlan(cfg: Config, wr: usize, w0: usize, pos0: usize, n: usize) !usize {
    try cfg.validate();
    if (n == 0 or n > cfg.rows) return error.KvPageRows;
    if (w0 % ratio != 0) return error.KvPageWindow;
    if (pos0 == 0) {
        _ = try pagingRows(cfg, 0, @min(n, att_rows));
        return 0;
    }
    if (pos0 < w0) return error.KvPageRewind;
    if (pos0 - w0 > wr) return error.KvPageRows;
    const end = try std.math.add(usize, pos0 - w0, n);
    const next = if (end > wr) (pos0 -| cfg.window) / ratio * ratio else w0;
    if (pos0 - next > wr or n > wr - (pos0 - next)) return error.KvPageRows;
    _ = try pagingRows(cfg, next, @min(n, att_rows));
    return next;
}

/// Device bytes of a paged sequence's K/V side for `cap` tokens under `cfg` (windows, slots, tables, work lists).
pub fn deviceBytesFor(cap: usize, cfg: Config) !usize {
    const layout = try SnapshotLayout.init(cap, cfg, 0);
    if (cfg.shared_lanes != 0) {
        _ = try sharedBytesFor(cap, cfg);
        return 0;
    }
    return std.math.add(usize, try std.math.add(usize, layout.device, layout.tables), work_n * 4 * 4);
}

pub fn sharedBytesFor(cap: usize, cfg: Config) !usize {
    try cfg.validate();
    if (cfg.shared_lanes == 0) return 0;
    return (try shared_device.Layout.init(.{ .cap = cap, .lanes = cfg.shared_lanes, .window = cfg.window, .rows = cfg.rows, .slots = cfg.slots })).common_bytes;
}

pub fn ownedSnapshotBytes(cfg: Config) !usize {
    if (cfg.shared_lanes == 0) return 0;
    try cfg.validate();
    return std.math.mul(usize, cfg.slots, 2 * ratio * row_bytes + @sizeOf(shared_contract.OwnedEntry));
}

/// A new sequence's paged store when the environment (or a test) turns paging on, else null.
pub fn forSeq(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, chunk: usize) !?*Paged {
    const cfg = configFromEnv() orelse return null;
    return try Paged.init(gpa, d, cap, cfg.forChunk(chunk));
}

test "exl3: kvpage tables and device bytes" {
    const cfg: Config = .{};
    try std.testing.expect(cfg.window >= wt.dense_keys);
    // 200k tokens: the device side of the default shape against upstream's K/V for the same sequence
    const wr = cfg.window + cfg.rows + ratio;
    const dev = 2 * (attn_n * wr + ratio * cfg.slots) * row_bytes;
    const full = attn_n * 2 * 200_016 * row_bytes;
    try std.testing.expect(dev * 8 < full);
    try std.testing.expectEqual(@as(usize, 1024), row_bytes);
    // 200k: the default shape's device side is about half a GiB (upstream's K/V: 4.58 GiB)
    try std.testing.expect(try deviceBytesFor(200_016, cfg) < 5 << 27);
}

test "exl3: kvpage allocation and admission reject invalid checked sizes before CUDA access" {
    const cfg: Config = .{};
    const layout = try SnapshotLayout.init(200_016, cfg, 0);
    try std.testing.expectEqual(layout.device + layout.tables + work_n * 4 * 4, try deviceBytesFor(200_016, cfg));
    try std.testing.expectError(error.Overflow, deviceBytesFor(std.math.maxInt(usize), cfg));
    try std.testing.expectError(error.KvPageRows, deviceBytesFor(0, cfg));
    var driver: cuda.Driver = undefined;
    try std.testing.expectError(error.Overflow, Paged.init(std.testing.allocator, &driver, std.math.maxInt(usize), cfg));
    var oversized = cfg;
    oversized.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, Paged.init(std.testing.allocator, &driver, 32, oversized));
}

test "exl3: kvpage window boundaries and conservative capacity" {
    const cfg: Config = .{ .window = 2052, .rows = 256, .slots = 512 };
    const wr = cfg.window + cfg.rows + ratio;
    try std.testing.expectEqual(@as(usize, 0), try windowPlan(cfg, wr, 0, wr - 1, 1));
    try std.testing.expectEqual(@as(usize, 260), try windowPlan(cfg, wr, 0, wr, 1));
    try std.testing.expectEqual(@as(usize, 0), try windowPlan(cfg, wr, 260, 0, 256));
    try std.testing.expectError(error.KvPageRewind, windowPlan(cfg, wr, 260, 259, 1));
    try std.testing.expectError(error.KvPageRows, windowPlan(cfg, wr, 0, wr + 1, 1));
    try std.testing.expectEqual(@as(usize, 8192), try windowPlan(cfg, wr, 8192, 8192, 256));
    try std.testing.expectEqual(@as(usize, 8192), try windowPlan(cfg, wr, 8192, 8192, 1));
}

test "exl3: kvpage graph key changes on rebase and host store replacement" {
    var pg: Paged = undefined;
    pg.dev.ptr = 1;
    pg.tabs.ptr = 2;
    pg.work.ptr = 3;
    pg.host_dev = 4;
    pg.cap = 200_016;
    pg.cfg = .{};
    pg.wr = pg.cfg.window + pg.cfg.rows + ratio;
    pg.win0 = @splat(0);
    const first = pg.graphKey();
    try std.testing.expectEqual(first, pg.graphKey());
    pg.win0[0] = 4;
    try std.testing.expect(!std.meta.eql(first, pg.graphKey()));
    pg.win0[0] = 0;
    pg.host_dev = 5;
    try std.testing.expect(!std.meta.eql(first, pg.graphKey()));
}

test "exl3: kvpage capacity subdivision boundaries and checked admission" {
    var cfg: Config = .{ .window = 4096, .rows = 2048, .slots = 16384 };
    try std.testing.expectEqual(@as(usize, 256), try pagingRows(cfg, 65536, 256));
    try std.testing.expectEqual(@as(usize, 32), try pagingRows(cfg, 65540, 256));
    try std.testing.expectEqual(@as(usize, 31), try pagingRows(cfg, 65540, 31));
    try std.testing.expectEqual(@as(usize, 32), try pagingRows(cfg, 65540, 33));
    cfg.slots = 16383;
    try std.testing.expectEqual(@as(usize, 31), try pagingRows(cfg, 65536, 256));
    cfg.slots = top - 1;
    try std.testing.expectEqual(@as(usize, 256), try pagingRows(cfg, (top - 1) * ratio, 256));
    try std.testing.expectError(error.KvPageCapacity, pagingRows(cfg, top * ratio, 1));
    const wr = cfg.window + cfg.rows + ratio;
    try std.testing.expectError(error.KvPageCapacity, windowPlan(cfg, wr, top * ratio, top * ratio, 256));
    try std.testing.expectError(error.KvPageRows, pagingRows(cfg, 0, 0));
    try std.testing.expectError(error.KvPageRows, pagingRows(cfg, 0, att_rows + 1));
    try std.testing.expectError(error.KvPageWindow, pagingRows(cfg, 1, 1));
    try std.testing.expectError(error.Overflow, windowPlan(cfg, std.math.maxInt(usize), 0, std.math.maxInt(usize), 1));
    cfg.slots = 0;
    try std.testing.expectError(error.KvPageConfig, windowPlan(cfg, wr, 0, 0, 1));
}

test "exl3: kvpage adversarial disjoint old blocks fit every subdivision" {
    for ([_]usize{ 512, 513, 1023, 1024, 16383, 16384 }) |slots| {
        const cfg: Config = .{ .slots = slots };
        const origin = (slots + 1) * ratio;
        for ([_]usize{ 1, 2, 15, 16, 17, 30, 31, 32, 33, 127, 255, 256 }) |rows| {
            const width = try pagingRows(cfg, origin, rows);
            var first: usize = 0;
            var calls: usize = 0;
            while (first < rows) {
                const count = @min(width, rows - first);
                var selected = std.AutoHashMap(usize, void).init(std.testing.allocator);
                defer selected.deinit();
                for (first..first + count) |row| for (0..top) |pick| {
                    const block = (row * top + pick) % (origin / ratio);
                    try selected.put(block, {});
                };
                try std.testing.expect(selected.count() <= slots);
                try std.testing.expectEqual(count * top, selected.count());
                first += count;
                calls += 1;
            }
            try std.testing.expectEqual(rows, first);
            try std.testing.expectEqual(cdiv(rows, width), calls);
            if (width < rows) try std.testing.expect(origin > wt.dense_keys);
        }
    }
    const cfg: Config = .{ .slots = 511 };
    try std.testing.expectEqual(att_rows, try pagingRows(cfg, 511 * ratio, att_rows));
}

test "exl3: kvpage 18k and 200k schedules use identical admission and paging bounds" {
    const cfg: Config = .{ .window = 4096, .rows = 2048, .slots = 16384 };
    const wr = cfg.window + cfg.rows + ratio;
    for ([_]usize{ 18432, 18448, 200000, 200016 }) |cap| {
        var origin: usize = 0;
        var position: usize = 0;
        var split_calls: usize = 0;
        while (position < cap) {
            const rows = @min(cfg.rows, cap - position);
            origin = try windowPlan(cfg, wr, origin, position, rows);
            var first: usize = 0;
            while (first < rows) : (first += att_rows) {
                const count = @min(att_rows, rows - first);
                const width = try pagingRows(cfg, origin, count);
                try std.testing.expect(@min(width * top, origin / ratio) <= cfg.slots);
                if (origin / ratio <= cfg.slots) try std.testing.expectEqual(count, width);
                if (width < count) split_calls += 1;
            }
            position += rows;
        }
        try std.testing.expectEqual(cap > 65536 + cfg.window, split_calls > 0);
        try std.testing.expectEqual(@as(usize, 1), try pagingRows(cfg, origin, 1));
        try std.testing.expectEqual(@as(usize, 16), try pagingRows(cfg, origin, 16));
    }
}

test "exl3: kvpage graph topology hash distinguishes subdivision across origins" {
    var pg: Paged = undefined;
    pg.dev.ptr = 1;
    pg.tabs.ptr = 2;
    pg.work.ptr = 3;
    pg.host_dev = 4;
    pg.cap = 200016;
    pg.cfg = .{ .window = 4096, .rows = 2048, .slots = 16384 };
    pg.wr = pg.cfg.window + pg.cfg.rows + ratio;
    pg.win0 = @splat(65536);
    const full = pg.graphKey();
    pg.win0[0] += ratio;
    const split = pg.graphKey();
    try std.testing.expect(full[4] != split[4]);
    try std.testing.expect(full[5] != split[5]);
    pg.win0[0] += ratio;
    const moved = pg.graphKey();
    try std.testing.expectEqual(split[4], moved[4]);
    try std.testing.expect(split[5] != moved[5]);
}

test "exl3: kvpage real attention dispatch preserves selection and slices sparse rows" {
    const Fake = struct {
        const ids_base: u64 = 0x100000;
        const nk_base: u64 = 0x200000;
        const spr_base: u64 = 0x300000;
        const query_base: u64 = 0x400000;
        const output_base: u64 = 0x500000;
        const pos_base: u64 = 0x600000;
        var selected: usize = 0;
        var consumed: usize = 0;
        var planned: usize = 0;
        var width: usize = 0;
        var launches: usize = 0;

        fn value(parameters: ?[*]?*anyopaque, index: usize, comptime Value: type) Value {
            const pointer: *const Value = @ptrCast(@alignCast(parameters.?[index].?));
            return pointer.*;
        }

        fn launch(kernel: cuda.abi.Function, grid_x: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, parameters: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            launches += 1;
            switch (@intFromPtr(kernel.?)) {
                1 => {
                    std.debug.assert(value(parameters, 1, u64) == pos_base + selected * 16);
                    std.debug.assert(value(parameters, 2, u64) == ids_base);
                    std.debug.assert(grid_x == @min(att_rows, 531 - consumed));
                    selected += 1;
                },
                2 => {
                    const offset = consumed % att_rows;
                    std.debug.assert(value(parameters, 0, u64) == ids_base + offset * idw * 4);
                    std.debug.assert(value(parameters, 1, u64) == nk_base + offset * 4);
                    std.debug.assert(value(parameters, 2, u64) == spr_base + offset * 4);
                    planned = @intCast(value(parameters, 3, c_int));
                    std.debug.assert(planned == @min(width, @min(att_rows - offset, 531 - consumed)));
                },
                3 => {},
                4 => std.debug.assert(grid_x == planned),
                else => unreachable,
            }
            return cuda.abi.success;
        }

        fn attnPrep(_: @This(), _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: f32, _: usize, _: usize) !void {}
        fn qsaPool(_: @This(), _: u64, _: u64, _: u64, _: u64, _: u64, _: f32, _: usize) !void {}
        fn qsaScores(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize) !void {}
        fn attnChunks(_: @This(), query: u64, _: u64, _: u64, pos: u64, _: u64, _: u64, _: u64, ids: u64, nk: u64, spr: u64, rows: usize, _: usize, chunks: usize, qsa: bool) !void {
            try std.testing.expectEqual(query_base + consumed * wt.q_heads * head_dim * 2, query);
            try std.testing.expectEqual(pos_base + (selected - 1) * 16, pos);
            try std.testing.expectEqual(ids_base + (consumed % att_rows) * idw * 4, ids);
            try std.testing.expectEqual(nk_base + (consumed % att_rows) * 4, nk);
            try std.testing.expectEqual(spr_base + (consumed % att_rows) * 4, spr);
            try std.testing.expectEqual(planned, rows);
            try std.testing.expectEqual(@as(usize, 5), chunks);
            try std.testing.expect(qsa);
        }
        fn attnMerge(_: @This(), _: u64, _: u64, _: u64, pos: u64, output: u64, _: u64, _: u64, rows: usize, _: usize, qsa: bool) !void {
            try std.testing.expectEqual(pos_base + (selected - 1) * 16, pos);
            try std.testing.expectEqual(output_base + consumed * wt.out_k * 2, output);
            try std.testing.expect(qsa);
            consumed += rows;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const function: cuda.Function = .{ .d = &driver, .handle = @ptrFromInt(1) };
    var pg: Paged = undefined;
    pg.d = &driver;
    pg.cfg = .{ .window = 4096, .rows = 2048, .slots = 16384 };
    pg.cap = 200016;
    pg.wr = pg.cfg.window + pg.cfg.rows + ratio;
    pg.nblocks = cdiv(pg.cap, ratio) + 1;
    pg.dev.ptr = 0x700000;
    pg.tabs.ptr = 0x800000;
    pg.work.ptr = 0x900000;
    pg.host_dev = 0xa00000;
    pg.graph_prepared = true;
    pg.k.plan = .{ .d = &driver, .handle = @ptrFromInt(2) };
    pg.k.fetch = .{ .d = &driver, .handle = @ptrFromInt(3) };
    pg.k.remap = .{ .d = &driver, .handle = @ptrFromInt(4) };
    const Buffer = struct { ptr: u64 = 0 };
    const Seq = struct { cap: usize, ikc: [attn_n]Buffer = @splat(.{}), pooled: [attn_n]Buffer = @splat(.{}), scores: u64 = 0 };
    const State = struct { inv: u64 = 0, dummy: u64 = 0, qids: u64 = Fake.ids_base, qnk: u64 = Fake.nk_base, qspr: u64 = Fake.spr_base, po: u64 = 0, pm: u64 = 0, pl: u64 = 0 };
    const Selection = struct { select: cuda.Function };
    const Pass = struct { seq: *Seq, s: *State, stream: cuda.Stream, tri: Fake = .{}, k: *const Selection, pos0: usize, lanes: []const u8 = &.{} };
    var seq: Seq = .{ .cap = pg.cap };
    var state: State = .{};
    const kernels: Selection = .{ .select = function };
    const layer = .{ .q_scale = Buffer{}, .k_scale = Buffer{}, .i_scale = Buffer{}, .ik_scale = Buffer{} };
    const at = .{ .pos = Fake.pos_base, .q = Fake.query_base, .iq = @as(u64, 0), .ao = Fake.output_base };
    for ([_]usize{ 14336, 65536, 65540, 194560 }) |origin| {
        pg.win0 = @splat(origin);
        Fake.selected = 0;
        Fake.consumed = 0;
        Fake.planned = 0;
        Fake.launches = 0;
        Fake.width = try pagingRows(pg.cfg, origin, att_rows);
        var pass: Pass = .{ .seq = &seq, .s = &state, .stream = stream, .k = &kernels, .pos0 = origin + pg.cfg.window };
        try pg.attnRows(&pass, layer, 0, 0, 531, at);
        try std.testing.expectEqual(@as(usize, 3), Fake.selected);
        try std.testing.expectEqual(@as(usize, 531), Fake.consumed);
        const launches = Fake.launches;
        pg.cfg.slots = top - 1;
        try std.testing.expectError(error.KvPageCapacity, pg.attnRows(&pass, layer, 0, 0, 531, at));
        try std.testing.expectEqual(launches, Fake.launches);
        pg.cfg.slots = 16384;
    }
}

test "exl3: kvpage host preparation writes back and compacts once before replay" {
    const Fake = struct {
        var copies: usize = 0;
        fn copy(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.debug.assert(dst + bytes <= src or src + bytes <= dst);
            const target: [*]u8 = @ptrFromInt(dst);
            const source: [*]const u8 = @ptrFromInt(src);
            @memcpy(target[0..bytes], source[0..bytes]);
            copies += 1;
            return cuda.abi.success;
        }
    };
    Fake.copies = 0;
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoDAsync_v2 = Fake.copy;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var pg: Paged = undefined;
    pg.d = &driver;
    pg.cfg = .{ .window = 2052, .rows = 256, .slots = 512 };
    pg.cap = 8192;
    pg.wr = pg.cfg.window + pg.cfg.rows + ratio;
    pg.win0 = @splat(0);
    pg.rebases = @splat(0);
    pg.written = @splat(0);
    const device = try std.testing.allocator.alloc(u8, pg.devBytes());
    defer std.testing.allocator.free(device);
    pg.dev.ptr = @intFromPtr(device.ptr);
    const host = try std.testing.allocator.alloc(u8, 2 * pg.cap * row_bytes);
    defer std.testing.allocator.free(host);
    @memset(host, 0);
    pg.host_dev = @intFromPtr(host.ptr);
    for (0..2) |side| {
        const start = if (side == 0) pg.devK(0) else pg.devV(0);
        const words: [*]u32 = @ptrFromInt(start);
        for (0..pg.wr) |position| @memset(words[position * (row_bytes / 4) ..][0 .. row_bytes / 4], @as(u32, @intCast(position + side * pg.cap)));
    }
    const before = pg.win0[0];
    const pos0 = pg.wr;
    try pg.prepareLayer(stream, 0, pos0, 1);
    const moved = pg.win0[0];
    try std.testing.expect(moved > before);
    try std.testing.expectEqual(@as(u64, 1), pg.rebases[0]);
    try std.testing.expectEqual(@as(u64, moved), pg.written[0]);
    for (0..2) |side| {
        const written: [*]const u32 = @ptrFromInt(pg.host_dev + side * pg.cap * row_bytes);
        const kept: [*]const u32 = @ptrFromInt(if (side == 0) pg.devK(0) else pg.devV(0));
        for (0..moved) |position| for (written[position * (row_bytes / 4) ..][0 .. row_bytes / 4]) |word| try std.testing.expectEqual(@as(u32, @intCast(position + side * pg.cap)), word);
        for (moved..pos0) |position| for (kept[(position - moved) * (row_bytes / 4) ..][0 .. row_bytes / 4]) |word| try std.testing.expectEqual(@as(u32, @intCast(position + side * pg.cap)), word);
    }
    const copies = Fake.copies;
    try pg.prepareLayer(stream, 0, pos0, 1);
    try std.testing.expectEqual(copies, Fake.copies);
    try std.testing.expectEqual(@as(u64, 1), pg.rebases[0]);
}
