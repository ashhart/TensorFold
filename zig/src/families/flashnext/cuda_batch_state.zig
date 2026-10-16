//! The batched pass's kernels, scratch, a sequence's view, and the hooks a verify window or the MTP head passes.

const std = @import("std");
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_prompt_load.zig");

const layers_n = wt.layers_n;
const linear_n = wt.linear_n;
const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const low = wt.low;
const eps = wt.eps;
const vocab = wt.vocab;
const experts_n = wt.experts_n;
const slots = wt.slots;
const moe_w = wt.moe_w;
const tile = wt.tile;
const q_heads = wt.q_heads;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_heads = wt.index_heads;
const index_dim = wt.index_dim;
const half = wt.half;
const proj_n = wt.proj_n;
const out_k = wt.out_k;
const nch = wt.nch;
const ple_layer = wt.ple_layer;
const ple_tail_n = wt.ple_tail_n;
const state_n = wt.state_n;
const hc_sk = wt.hc_sk;
const out_sk = wt.out_sk;
const Q4 = wt.Q4;
const HcW = wt.HcW;
const Layer = load.Layer;

/// Normed-stream partial sums a row: D / 256 blocks of each stream.
pub const pss_n = (dims / 256) * streams;
/// The widest projection row: the DeltaNet's q|k|v|z|b|a (attention's q|k|v|indexer is 13952).
pub const proj_row = @max(gdn.proj_width, proj_n);
/// Attention rows a _chunks launch takes (its scratch: rows x nch x heads x dim fp32).
pub const att_rows: usize = 256;
/// Rows a pass holds at most unless asked for more.
pub const default_chunk: usize = 1536;
const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
const cubin_qmm: [:0]const u8 = "_ZN6tf_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii";
const qmm_smem: u32 = 9472;
/// Pairs an expert item takes: the core rows kernel's tile.
const rows_tile: usize = cuda.experts.RowsExperts.tile;
const plan_small: usize = cuda.grouped.small;

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// Which rows get the head: the last (a prompt), every row (a verify window or a round), or none.
pub const Logits = enum { last, all, none };

/// A DeltaNet layer's window keep: projection rows, conv history before the chunk, and where its end state goes.
pub const LayerKeep = struct {
    proj: u64 = 0,
    hist: u64 = 0,
    /// The chain's end state goes here, the sequence's own left as before the chunk; 0: in place.
    state_out: u64 = 0,
};

/// What a drafted window keeps for its rollback (cuda_mtp.zig): each DeltaNet layer's rows and states.
pub const Keep = struct {
    layers: [layers_n]LayerKeep = @splat(.{}),
};

/// A caller's view into the pass. Without any, the pass is the server's prompt pass.
pub const Hooks = struct {
    logits: Logits = .last,
    /// Device [rows, vocab] bf16 for `.all` (else the pass's own buffer for the last row).
    logits_out: u64 = 0,
    keep: ?*Keep = null,
    /// Device [rows, wide] bf16: every row's streams after the last MoE write-back (the MTP head's input); 0: none.
    streams: u64 = 0,
};

/// The kernels the pass launches besides the captured Triton ones, loaded once.
pub const Kernels = struct {
    qmm: cuda.Module,
    experts: cuda.Module,
    rows: cuda.Module,
    batch: cuda.Module,
    /// The core's expert plan and its rows kernels.
    router: cuda.grouped.Router,
    moe: cuda.experts.RowsExperts,
    lane: cuda.Function,
    native_lane: cuda.qmmf.Lane,
    front: cuda.Function,
    chain: cuda.Function,
    chain_to: cuda.Function,
    gnorm: cuda.Function,
    sum: cuda.Function,
    shift: cuda.Function,
    /// Each sparse row's top blocks (attention._select / _select_tiles), native.
    select: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.qmm = try cuda.Module.load(d, cuda.kernels.qmm);
        errdefer k.qmm.unload();
        k.experts = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.experts.unload();
        k.rows = try cuda.Module.load(d, cuda.kernels.experts_rows);
        errdefer k.rows.unload();
        k.batch = try cuda.Module.load(d, cuda.kernels.flashnext_batch);
        errdefer k.batch.unload();
        k.lane = try k.qmm.function(cubin_qmm);
        try k.lane.allowDynamicShared(qmm_smem);
        k.router = try cuda.grouped.Router.resolve(k.experts);
        k.moe = try cuda.experts.RowsExperts.resolve(k.rows);
        k.native_lane = try cuda.qmmf.Lane.load(d, (try d.capability(0)) / 10);
        errdefer k.native_lane.unload();
        k.sum = try k.batch.function("fn_sum_slices");
        k.shift = try k.batch.function("fn_shift_window");
        k.select = try k.batch.function("fn_qsa_select");
        k.front = try k.batch.function("fn_gdn_front");
        k.chain = try k.batch.function("fn_gdn_rows");
        k.chain_to = try k.batch.function("fn_gdn_rows_to");
        k.gnorm = try k.batch.function("fn_gdn_norm");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.native_lane.unload();
        k.batch.unload();
        k.rows.unload();
        k.experts.unload();
        k.qmm.unload();
    }
};

/// The compute stream as a pass's pinned staging uses it: copies from it, an event behind them, and a drain.
pub const Uploads = struct {
    stream: cuda.Stream,
    event: cuda.Event,

    pub fn copy(u: Uploads, dst: u64, bytes: []const u8) !void {
        const d = u.stream.d;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, u.stream.handle), "cuMemcpyHtoDAsync");
    }
    pub fn record(u: Uploads) !void {
        try u.event.record(u.stream);
    }
    pub fn wait(u: Uploads) !void {
        try u.event.synchronize();
    }
    pub fn drain(u: Uploads) !void {
        try u.stream.synchronize();
    }
};

/// Pinned bytes' uploads and the fence that says when the bytes may change (`Q`: `Uploads`, or a test's stand-in).
pub fn Staging(comptime Q: type) type {
    return struct {
        const Self = @This();
        q: Q,
        /// An upload was queued after the event's last record (a copy or the record failed): only a drain covers it.
        open: bool = false,

        /// Queues a copy from the staged bytes; until `fence` records behind it, `ready` drains the stream.
        pub fn upload(g: *Self, dst: u64, bytes: []const u8) !void {
            g.open = true;
            try g.q.copy(dst, bytes);
        }

        /// Records the event behind every upload queued so far.
        pub fn fence(g: *Self) !void {
            try g.q.record();
            g.open = false;
        }

        /// Returns once no queued upload reads the staged bytes: the event's wait, or a drain if one came after it.
        pub fn ready(g: *Self) !void {
            if (!g.open) return g.q.wait();
            try g.q.drain();
            g.open = false;
        }
    };
}

/// Device scratch for chunks of up to `cap` rows.
pub const Scratch = struct {
    cap: usize,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    ids: u64 = 0,
    pos: u64 = 0,
    h: u64 = 0,
    pss: u64 = 0,
    norm: u64 = 0,
    dn: u64 = 0,
    hc_part: u64 = 0,
    hc_up: u64 = 0,
    native_part: u64 = 0,
    act: u64 = 0,
    xs_act: u64 = 0,
    inj: u64 = 0,
    mixed: u64 = 0,
    xsm: u64 = 0,
    proj: u64 = 0,
    part: u64 = 0,
    gout: u64 = 0,
    gxs: u64 = 0,
    ys: u64 = 0,
    gq: u64 = 0,
    gk: u64 = 0,
    gv: u64 = 0,
    ggb: u64 = 0,
    q: u64 = 0,
    iq: u64 = 0,
    po: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    /// An attention block's rows' key lists, key counts and sparse flags (attention.AttnScratch ids/nk/sparse).
    qids: u64 = 0,
    qnk: u64 = 0,
    qspr: u64 = 0,
    ao: u64 = 0,
    gated: u64 = 0,
    gxs_attn: u64 = 0,
    router_l: u64 = 0,
    picks: u64 = 0,
    wts: u64 = 0,
    members: u64 = 0,
    items: u64 = 0,
    counts: u64 = 0,
    rank: u64 = 0,
    hist: u64 = 0,
    moe_act: u64 = 0,
    y: u64 = 0,
    branch: u64 = 0,
    ple_emb: u64 = 0,
    ple_xs: u64 = 0,
    ple_keys: u64 = 0,
    ple_vals: u64 = 0,
    ple_part: u64 = 0,
    ple_gated: u64 = 0,
    ple_pss: u64 = 0,
    ple_nrow: u64 = 0,
    logits: u64 = 0,
    inv: u64 = 0,
    dummy: u64 = 0,
    /// Pinned host copies of what a pass uploads (ids, positions, n-gram rows and sums), owned until the copies land.
    host: ?cuda.HostBuffer = null,
    /// The uploads from `host` and the event recorded after a pass's last one: the next pass waits before writing it.
    staging: ?Staging(Uploads) = null,

    fn add(s: *Scratch, gpa: std.mem.Allocator, d: *const cuda.Driver, len: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d, @max(len, 256));
        errdefer b.free();
        try s.bufs.append(gpa, b);
        return b.ptr;
    }

    /// Byte offsets of the staged ids, positions, n-gram rows and their sums in `host`, and its length.
    fn hostAt(cap: usize) [5]usize {
        const pos = std.mem.alignForward(usize, cap * 4, 16);
        const emb = pos + (cap + 1) * 16;
        const xs = emb + cap * dims * 2;
        return .{ 0, pos, emb, xs, xs + cap * (dims / 32) * 4 };
    }

    /// Waits until the last pass's uploads have read `host` (they are queued behind its kernels), so it may change.
    pub fn ready(s: *Scratch) !void {
        if (s.staging) |*g| try g.ready();
    }

    /// A queued copy from `host` to the device, which must stay unchanged until it lands.
    pub fn upload(s: *Scratch, dst: u64, src: []const u8) !void {
        try s.staging.?.upload(dst, src);
    }

    /// Marks `host` read once the stream reaches here: every upload queued before this point has copied its bytes.
    pub fn uploaded(s: *Scratch) !void {
        if (s.staging) |*g| try g.fence();
    }

    /// The staged ids [n], positions [n] (in 16-byte rows), n-gram rows [n, dims] and their 32-group sums.
    pub fn hostIds(s: *const Scratch, n: usize) []i32 {
        return @alignCast(std.mem.bytesAsSlice(i32, s.host.?.bytes[hostAt(s.cap)[0]..][0 .. n * 4]));
    }
    pub fn hostPos(s: *const Scratch, n: usize) []i32 {
        return @alignCast(std.mem.bytesAsSlice(i32, s.host.?.bytes[hostAt(s.cap)[1]..][0 .. n * 4]));
    }
    pub fn hostEmb(s: *const Scratch, n: usize) []u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, s.host.?.bytes[hostAt(s.cap)[2]..][0 .. n * dims * 2]));
    }
    pub fn hostXs(s: *const Scratch, n: usize) []f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, s.host.?.bytes[hostAt(s.cap)[3]..][0 .. n * (dims / 32) * 4]));
    }

    /// Bytes `alloc(cap)` takes on the device.
    pub fn bytes(cap: usize) usize {
        const pairs = cap * slots;
        const nblk = cdiv(pairs, plan_small);
        const items = cdiv(pairs, tile) + experts_n + 1;
        var n: usize = 0;
        n += cap * 4 + (cap + 1) * 16;
        n += cap * wide * 2 * 2 + cap * pss_n * 4 + cap * (low + 4) * 2 + hc_sk * cap * (low + 4) * 4;
        n += cap * low * 2 + cap * (low / 32) * 4 + cap * streams * 2 + cap * dims * 2 + cap * (dims / 32) * 4;
        n += cap * proj_row * 2 + out_sk * cap * dims * 4 + cap * gdn.value_dim * (2 + 4 + 4) + cap * (gdn.value_dim / 32) * 4;
        n += cap * (2 * gdn.nk * gdn.dk + 2 * gdn.nv) * 4;
        n += cap * q_heads * head_dim * 2 + cap * index_heads * index_dim * 2;
        n += att_rows * nch * q_heads * (head_dim + 2) * 4 + cap * out_k * 2 * 2 + cap * (out_k / 32) * 4;
        n += att_rows * (idw + 2) * 4;
        n += cap * (experts_n + 1) * 4 + pairs * 4 * 2 + pairs * 4 + items * 3 * 4 + 8 + pairs * 4 + nblk * (experts_n + 1) * 4;
        n += pairs * moe_w * 2 + pairs * dims * 4 + cap * dims * 2;
        n += cap * dims * 2 + cap * (dims / 32) * 4 + cap * wide * 2 + cap * dims * 2 + 4 * cap * dims * 4;
        n += cap * wide * 2 * 2 + cap * streams * 4 + vocab * 2 + half * 4 + 4096;
        return n;
    }

    pub fn alloc(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, cap: usize) !Scratch {
        var s: Scratch = .{ .cap = cap };
        errdefer s.free(gpa);
        const pairs = cap * slots;
        const nblk = cdiv(pairs, plan_small);
        const items = cdiv(pairs, tile) + experts_n + 1;
        s.ids = try s.add(gpa, d, cap * 4);
        // a position every 16 bytes: one a block of att_rows rows, or one a row of a decode round
        s.pos = try s.add(gpa, d, (cap + 1) * 16);
        s.h = try s.add(gpa, d, cap * wide * 2);
        s.pss = try s.add(gpa, d, cap * pss_n * 4);
        s.norm = try s.add(gpa, d, cap * wide * 2);
        s.dn = try s.add(gpa, d, cap * (low + 4) * 2);
        s.hc_part = try s.add(gpa, d, hc_sk * cap * (low + 4) * 4);
        s.hc_up = try s.add(gpa, d, cap * wide * 2);
        s.native_part = try s.add(gpa, d, 8 * cap * dims * 4);
        s.act = try s.add(gpa, d, cap * low * 2);
        s.xs_act = try s.add(gpa, d, cap * (low / 32) * 4);
        s.inj = try s.add(gpa, d, cap * streams * 2);
        s.mixed = try s.add(gpa, d, cap * dims * 2);
        s.xsm = try s.add(gpa, d, cap * (dims / 32) * 4);
        s.proj = try s.add(gpa, d, cap * proj_row * 2);
        s.part = try s.add(gpa, d, out_sk * cap * dims * 4);
        s.gout = try s.add(gpa, d, cap * gdn.value_dim * 2);
        s.gxs = try s.add(gpa, d, cap * (gdn.value_dim / 32) * 4);
        s.ys = try s.add(gpa, d, cap * gdn.value_dim * 4);
        s.gq = try s.add(gpa, d, cap * gdn.nk * gdn.dk * 4);
        s.gk = try s.add(gpa, d, cap * gdn.nk * gdn.dk * 4);
        s.gv = try s.add(gpa, d, cap * gdn.value_dim * 4);
        s.ggb = try s.add(gpa, d, cap * gdn.nv * 2 * 4);
        s.q = try s.add(gpa, d, cap * q_heads * head_dim * 2);
        s.iq = try s.add(gpa, d, cap * index_heads * index_dim * 2);
        s.po = try s.add(gpa, d, att_rows * nch * q_heads * head_dim * 4);
        s.pm = try s.add(gpa, d, att_rows * nch * q_heads * 4);
        s.pl = try s.add(gpa, d, att_rows * nch * q_heads * 4);
        s.qids = try s.add(gpa, d, att_rows * idw * 4);
        s.qnk = try s.add(gpa, d, att_rows * 4);
        s.qspr = try s.add(gpa, d, att_rows * 4);
        s.ao = try s.add(gpa, d, cap * out_k * 2);
        s.gated = try s.add(gpa, d, cap * out_k * 2);
        s.gxs_attn = try s.add(gpa, d, cap * (out_k / 32) * 4);
        s.router_l = try s.add(gpa, d, cap * (experts_n + 1) * 4);
        s.picks = try s.add(gpa, d, pairs * 4);
        s.wts = try s.add(gpa, d, pairs * 4);
        s.members = try s.add(gpa, d, pairs * 4);
        s.items = try s.add(gpa, d, items * 3 * 4);
        s.counts = try s.add(gpa, d, 8);
        s.rank = try s.add(gpa, d, pairs * 4);
        s.hist = try s.add(gpa, d, nblk * (experts_n + 1) * 4);
        s.moe_act = try s.add(gpa, d, pairs * moe_w * 2);
        s.y = try s.add(gpa, d, pairs * dims * 4);
        s.branch = try s.add(gpa, d, cap * dims * 2);
        s.ple_emb = try s.add(gpa, d, cap * dims * 2);
        s.ple_xs = try s.add(gpa, d, cap * (dims / 32) * 4);
        s.ple_keys = try s.add(gpa, d, cap * wide * 2);
        s.ple_vals = try s.add(gpa, d, cap * dims * 2);
        s.ple_part = try s.add(gpa, d, 4 * cap * dims * 4);
        s.ple_gated = try s.add(gpa, d, cap * wide * 2);
        s.ple_pss = try s.add(gpa, d, cap * streams * 4);
        s.ple_nrow = try s.add(gpa, d, cap * wide * 2);
        s.logits = try s.add(gpa, d, vocab * 2);
        var inv: [half]f32 = undefined;
        for (&inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / @as(f64, half)));
        s.inv = try s.add(gpa, d, half * 4);
        try s.bufs.items[s.bufs.items.len - 1].uploadAsync(0, std.mem.sliceAsBytes(&inv), stream.handle);
        s.dummy = try s.add(gpa, d, 4096);
        try s.bufs.items[s.bufs.items.len - 1].fill8(0, stream.handle);
        try stream.synchronize();
        s.host = try cuda.HostBuffer.alloc(d, hostAt(cap)[4]);
        s.staging = .{ .q = .{ .stream = stream, .event = try cuda.Event.init(d, false) } };
        return s;
    }

    pub fn free(s: *Scratch, gpa: std.mem.Allocator) void {
        if (s.staging) |*g| {
            // no queued upload may still read `host`; a wait that fails has lost the context, and its copies with it
            g.ready() catch {};
            g.q.event.deinit();
        }
        if (s.host) |*h| h.free();
        for (s.bufs.items) |*b| b.free();
        s.bufs.deinit(gpa);
        s.* = undefined;
    }
};

/// A sequence's state the pass reads and leaves as the row-by-row path would.
pub const SeqState = struct {
    conv: []cuda.DeviceBuffer,
    state: []cuda.DeviceBuffer,
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
    ikc: []cuda.DeviceBuffer,
    /// Each attention layer's pooled indexer keys [cap / 4, 128] bf16.
    pooled: []cuda.DeviceBuffer,
    tail: cuda.DeviceBuffer,
    history: *[2]i64,
    /// Tokens the caches hold.
    cap: usize,
    /// Block scores for an attention block's rows [min(rows, att_rows), scoreStride(cap)] fp32 (the engine's).
    scores: u64,
};

/// The block scores' row stride for a cache of `cap` tokens: its blocks rounded up to 16 (the captured _scores').
pub fn scoreStride(cap: usize) usize {
    return (cdiv(cap, ratio) + 15) / 16 * 16;
}

/// Bytes of block scores a pass over a cache of `cap` tokens needs for chunks of up to `rows` rows.
pub fn scoreBytes(cap: usize, rows: usize) usize {
    return @min(rows, att_rows) * scoreStride(cap) * 4;
}

/// One row of a decode round: its stream's next token, on that stream's own caches at its own position.
pub const Lane = struct { seq: SeqState, pos: usize };

/// The engine's resident weights the pass reads.
pub const Model = struct {
    table_w: u64,
    table_s: u64,
    table_b: u64,
    table_dense: bool = false,
    layers: []const Layer,
    mixer: HcW,
    head: Q4,
    ngram: *const ple.Ngram,
};

/// A stand-in for the stream: which copies are still queued, and a failure at a chosen copy or at the record.
const FakeQueue = struct {
    queued: usize = 0,
    covered: usize = 0,
    copies: usize = 0,
    fail_copy: ?usize = null,
    fail_record: bool = false,

    pub fn copy(q: *FakeQueue, _: u64, _: []const u8) !void {
        q.copies += 1;
        if (q.fail_copy) |k| if (k == q.copies) return error.CopyFailed;
        q.queued += 1;
    }
    pub fn record(q: *FakeQueue) !void {
        if (q.fail_record) return error.RecordFailed;
        q.covered = q.queued;
    }
    pub fn wait(q: *FakeQueue) !void {
        q.queued -= q.covered; // the event's copies have landed; any queued after it may still be reading
        q.covered = 0;
    }
    pub fn drain(q: *FakeQueue) !void {
        q.queued = 0;
        q.covered = 0;
    }
};

test "a pass whose second upload or whose record fails leaves no copy reading the staging when it is reused" {
    const Case = struct { fail_copy: ?usize, fail_record: bool };
    for ([_]Case{ .{ .fail_copy = 2, .fail_record = false }, .{ .fail_copy = null, .fail_record = true }, .{ .fail_copy = null, .fail_record = false } }) |c| {
        var fq: FakeQueue = .{};
        var g: Staging(*FakeQueue) = .{ .q = &fq };
        // a pass that stages and fences cleanly, then one that fails part way, as `stage()` uploads ids and positions
        try g.ready();
        try g.upload(0, "ids");
        try g.upload(0, "pos");
        try g.fence();
        fq.fail_copy = if (c.fail_copy) |k| fq.copies + k else null;
        fq.fail_record = c.fail_record;
        try g.ready();
        const failed = blk: {
            g.upload(0, "ids") catch break :blk true;
            g.upload(0, "pos") catch break :blk true;
            g.fence() catch break :blk true;
            break :blk false;
        };
        try std.testing.expectEqual(c.fail_copy != null or c.fail_record, failed);
        // the next pass (or `free`) waits before it writes the bytes: nothing may still be queued
        try g.ready();
        try std.testing.expectEqual(@as(usize, 0), fq.queued);
    }
}
