//! The batched pass's kernels, scratch, a sequence's view, and the hooks a verify window or the MTP head passes.

const std = @import("std");
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const wt = @import("cuda_prompt_wt.zig");
const tk = @import("cuda_tile_kernels.zig");
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
/// Neurons a pocket's activations take a row (fn_back's max_pocket).
pub const pocket_stride: usize = 128;
/// Prompt rows one fused attention launch takes (their selections staged att_rows at a time).
pub const fused_rows: usize = 1024;
/// Rows a pass holds at most unless asked for more.
pub const default_chunk: usize = 1536;
/// Rows of `head_local` (a vocabulary-split head's local logits) a scratch holds at most: every row by default; an
pub var head_local_rows: usize = std.math.maxInt(usize);
/// The most rows TF_FLASHNEXT_CHUNK may ask for (the largest measured pass; its scratch is about 13 GiB more).
pub const max_chunk: usize = 16384;
const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
pub const cubin_qmm: [:0]const u8 = "_ZN6tf_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii";
pub const qmm_smem: u32 = 9472;
/// fn_gdn_chain_tpr's staged rows: 2 x 16 steps of q, k, v (128 fp32 each) and g, beta.
pub const chain_tpr_smem: u32 = 2 * 16 * (3 * 128 * 4 + 16);
/// fn_gdn_fused's shared memory (its SMEM): two chain stages, two readout batches, the conv rows and 16 norm scales.
pub const gdn_fused_smem: u32 = chain_tpr_smem + 2 * 16 * 128 * 4 + 16 * 2 * 128 * 4 + 16 * 4;
/// fn_gdn_fused's block: 256 chain threads and 128 producer threads.
pub const gdn_fused_threads: u32 = 384;
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

/// A read-out's kept tensors (fn_back's HcKept): h and normed [rows, wide], v fp32 [rows, low (+ 4)], up, inject.
pub const HcKeep = struct { h: u64 = 0, normed: u64 = 0, v: u64 = 0, up: u64 = 0, inject: u64 = 0 };

/// A layer's kept tensors, row strides their own widths; 0 leaves one out.
pub const LayerKeep = struct {
    /// DeltaNet: projection rows [rows, proj] and the conv history before the chunk.
    proj: u64 = 0,
    hist: u64 = 0,
    /// The chain's end state goes here, the sequence's own left as before the chunk; 0: in place.
    state_out: u64 = 0,
    attn_hc: HcKeep = .{},
    mlp_hc: HcKeep = .{},
    /// The mixer's and the MoE's branches [rows, dims] bf16 as the write-backs add them (the MoE's with its pocket).
    mix_branch: u64 = 0,
    moe_branch: u64 = 0,
    /// The MoE layer (fn_back's MoeKept): its input [rows, dims] bf16, experts and weights [rows, 11].
    moe_x: u64 = 0,
    moe_ids: u64 = 0,
    moe_wts: u64 = 0,
    /// Attention (fn_back's AttnKept): q|gate, k and v projections, and o_proj's input, bf16.
    pq: u64 = 0,
    pk: u64 = 0,
    pv: u64 = 0,
    gated: u64 = 0,
};

/// What a pass keeps: a drafted window's rollback (cuda_mtp.zig), or a training pass's tensors with the final mixer's.
pub const Keep = struct {
    layers: [layers_n]LayerKeep = @splat(.{}),
    mixer: HcKeep = .{},
};

/// A sequence of a training pass (Pass.spans): its first row and its rows.
pub const Span = struct { at: usize, rows: usize };

/// Adds to a layer's fp32 MoE output `out` [rows, dims] (the slots' sum) before its rounding, from its input and slots.
pub const MoeAdd = struct {
    ctx: *anyopaque,
    add: *const fn (ctx: *anyopaque, layer: usize, x: u64, ids: u64, wts: u64, out: u64, rows: usize) anyerror!void,
    /// A bit a layer it adds to; the others write back as without it.
    layers: u64 = std.math.maxInt(u64),

    pub fn at(m: MoeAdd, layer: usize) bool {
        return m.layers >> @intCast(layer) & 1 != 0;
    }
};

comptime {
    std.debug.assert(layers_n <= 64); // MoeAdd.layers holds a bit a layer
}

/// A caller's view into the pass. Without any, the pass is the server's prompt pass.
pub const Hooks = struct {
    logits: Logits = .last,
    /// Device [rows, vocab] bf16 for `.all` (else the pass's own buffer for the last row).
    logits_out: u64 = 0,
    keep: ?*Keep = null,
    /// Device [rows, wide] bf16: every row's streams after the last MoE write-back (the MTP head's input); 0: none.
    streams: u64 = 0,
    /// A training pass's adds to each layer's MoE output, in place of the model's own (Model.pocket).
    moe_add: ?MoeAdd = null,
};

/// The kernels the pass launches besides the captured Triton ones, loaded once.
pub const Kernels = @import("cuda_batch_kernels.zig").Kernels;

/// The compute stream as a pass's pinned staging uses it: copies from it, an event behind them, and a drain.
pub const Uploads = struct {
    stream: cuda.Stream,
    event: cuda.Event,

    pub fn copy(u: Uploads, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(u.stream.d, dst, bytes, u.stream);
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
pub const ScratchFormat = enum { classic, exl3 };

pub const Scratch = struct {
    cap: usize,
    format: ScratchFormat = .classic,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    ids: u64 = 0,
    pos: u64 = 0,
    h: u64 = 0,
    pss: u64 = 0,
    norm: u64 = 0,
    rinv: u64 = 0,
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
    /// part's second cap-stride slice is zeros (native lanes write only the first): a writeback adds it once.
    part_zero: bool = false,
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
    shared_gate: u64 = 0,
    shared_up: u64 = 0,
    shared_act: u64 = 0,
    shared_out: u64 = 0,
    y: u64 = 0,
    branch: u64 = 0,
    /// A layer's MoE output in fp32 [cap, dims] when something adds to it, and the pocket's activations [cap, 128].
    moe_f32: u64 = 0,
    pocket_act: u64 = 0,
    ple_emb: u64 = 0,
    ple_xs: u64 = 0,
    ple_keys: u64 = 0,
    ple_vals: u64 = 0,
    ple_part: u64 = 0,
    ple_gated: u64 = 0,
    ple_pss: u64 = 0,
    ple_nrow: u64 = 0,
    logits: u64 = 0,
    head_local: u64 = 0,
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

    /// The buffers in the order `allocationSizes` lists their sizes.
    const order = [_][]const u8{
        "ids",       "pos",        "h",          "pss",     "norm",     "rinv",    "dn",         "hc_part",    "hc_up",   "native_part", "act",
        "xs_act",    "inj",        "mixed",      "xsm",     "proj",     "part",    "gout",       "gxs",        "ys",      "gq",          "gk",
        "gv",        "ggb",        "q",          "iq",      "po",       "pm",      "pl",         "qids",       "qnk",     "qspr",        "ao",
        "gated",     "gxs_attn",   "router_l",   "picks",   "wts",      "members", "items",      "counts",     "rank",    "hist",        "shared_gate",
        "shared_up", "shared_act", "shared_out", "moe_act", "y",        "branch",  "moe_f32",    "pocket_act", "ple_emb", "ple_xs",      "ple_keys",
        "ple_vals",  "ple_part",   "ple_gated",  "ple_pss", "ple_nrow", "logits",  "head_local", "inv",        "dummy",
    };

    pub fn allocationSizes(cap: usize, format: ScratchFormat) [64]?usize {
        const pairs = cap * slots;
        const nblk = cdiv(pairs, plan_small);
        const items = cdiv(pairs, tile) + experts_n + 1;
        return .{
            cap * 4,
            (cap + 1) * 16,
            cap * wide * 2,
            cap * pss_n * 4,
            cap * wide * 2,
            cap * streams * 4,
            cap * (low + 4) * 2,
            hc_sk * cap * (low + 4) * 4,
            cap * wide * 2,
            8 * cap * dims * 4,
            cap * low * 2,
            cap * (low / 32) * 4,
            cap * streams * 2,
            cap * dims * 2,
            cap * (dims / 32) * 4,
            cap * proj_row * 2,
            out_sk * cap * dims * 4,
            cap * gdn.value_dim * 2,
            cap * (gdn.value_dim / 32) * 4,
            cap * gdn.value_dim * 4,
            cap * gdn.nk * gdn.dk * 4,
            cap * gdn.nk * gdn.dk * 4,
            cap * gdn.value_dim * 4,
            cap * gdn.nv * 2 * 4,
            cap * q_heads * head_dim * 2,
            cap * index_heads * index_dim * 2,
            att_rows * nch * q_heads * head_dim * 4,
            att_rows * nch * q_heads * 4,
            att_rows * nch * q_heads * 4,
            fused_rows * idw * 4,
            fused_rows * 4,
            fused_rows * 4,
            cap * out_k * 2,
            cap * out_k * 2,
            cap * (out_k / 32) * 4,
            cap * (experts_n + 1) * 4,
            pairs * 4,
            pairs * 4,
            if (format == .classic) pairs * 4 else null,
            if (format == .classic) items * 3 * 4 else null,
            if (format == .classic) 8 else null,
            if (format == .classic) pairs * 4 else null,
            if (format == .classic) nblk * (experts_n + 1) * 4 else null,
            if (format == .classic) cap * moe_w * 2 else null,
            if (format == .classic) cap * moe_w * 2 else null,
            if (format == .classic) cap * moe_w * 2 else null,
            if (format == .classic) cap * dims * 4 else null,
            if (format == .classic) pairs * moe_w * 2 else null,
            pairs * dims * 4,
            cap * dims * 2,
            cap * dims * 4,
            cap * pocket_stride * 4,
            cap * dims * 2,
            cap * (dims / 32) * 4,
            cap * wide * 2,
            cap * dims * 2,
            4 * cap * dims * 4,
            cap * wide * 2,
            cap * streams * 4,
            cap * wide * 2,
            vocab * 2,
            @min(cap, head_local_rows) * vocab,
            half * 4,
            4096,
        };
    }

    pub fn bytes(cap: usize) usize {
        return bytesForFormat(cap, .classic);
    }

    pub fn bytesForFormat(cap: usize, format: ScratchFormat) usize {
        var total: usize = 0;
        for (allocationSizes(cap, format)) |optional| if (optional) |len| {
            total += @max(len, 256);
        };
        return total;
    }

    pub fn alloc(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, cap: usize) !Scratch {
        return allocForFormat(gpa, d, stream, cap, .classic);
    }

    pub fn allocForFormat(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, cap: usize, format: ScratchFormat) !Scratch {
        var s: Scratch = .{ .cap = cap, .format = format };
        errdefer {
            cuda.completion.drain(d, stream, "Flash Next scratch work did not drain before rollback");
            s.free(gpa);
        }
        const sizes = allocationSizes(cap, format);
        // a row of a decode round or a block of att_rows rows takes `pos`'s 16 bytes
        var inv: [half]f32 = undefined;
        for (&inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / @as(f64, half)));
        inline for (order, 0..) |name, i| {
            @field(s, name) = if (sizes[i]) |len| try s.add(gpa, d, len) else 0;
            if (sizes[i] != null) {
                const last = &s.bufs.items[s.bufs.items.len - 1];
                // part's second cap-stride slice stays zeros (native lanes write only the first); dummy is all zeros
                if (comptime std.mem.eql(u8, name, "part")) {
                    try last.fill8(0, stream.handle);
                    s.part_zero = true;
                }
                if (comptime std.mem.eql(u8, name, "inv")) try last.uploadAsync(0, std.mem.sliceAsBytes(&inv), stream.handle);
                if (comptime std.mem.eql(u8, name, "dummy")) try last.fill8(0, stream.handle);
            }
        }
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
    media: ?@import("cuda_media.zig").View = null,
    /// An int8 K/V cache (kc/vc hold its codes; cuda_exl3_kv.zig); null: bf16 kc/vc.
    kv8: ?@import("exl3/kv/cuda_exl3_kv.zig").SeqKv8 = null,
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
    /// The K/V paged to host memory with a device block cache (cuda_exl3_kvpage.zig), else kc/vc hold it all.
    paged: ?*@import("exl3/kv/cuda_exl3_kvpage.zig").Paged = null,
    /// The attention caches' pages (pt 0: kc, vc, ikc and pooled are contiguous rows).
    pg: @import("cuda_glue.zig").Pg = .{},
};

/// The block scores' row stride for a cache of `cap` tokens: its blocks rounded up to 16 (the captured _scores').
pub fn scoreStride(cap: usize) usize {
    return (cdiv(cap, ratio) + 15) / 16 * 16;
}

/// Bytes of block scores a pass over a cache of `cap` tokens needs for chunks of up to `rows` rows.
pub fn scoreBytes(cap: usize, rows: usize) usize {
    return @min(rows, att_rows) * scoreStride(cap) * 4;
}

/// One stream's rows in a shared pass from `pos` on its own caches, and its window's rollback, hidden rows and scores.
pub const Lane = struct { seq: SeqState, pos: usize, rows: usize = 1, keep: ?*const Keep = null, streams: u64 = 0, logits: u64 = 0 };

/// The engine's resident weights the pass reads.
pub const Model = struct {
    collective: ?*@import("cuda_collective.zig").Collective = null,
    table_w: u64,
    table_s: u64,
    table_b: u64,
    table_dense: bool = false,
    layers: []const Layer,
    mixer: HcW,
    head: Q4,
    ngram: *const ple.Ngram,
    /// The engine's own adds to the MoE outputs: a taught folder's pockets (null: none).
    pocket: ?MoeAdd = null,
};

pub const Gather = @import("cuda_batch_gather.zig").Gather;

test {
    _ = @import("cuda_batch_state_test.zig");
    _ = @import("cuda_nvfp4_tile.zig");
    _ = @import("cuda_scratch_layout_test.zig");
}

pub const Pg = @import("cuda_glue.zig").Pg;
