//! The loaded engine and one stream's caches: prompts and windows on the batched pass, decode rounds over streams.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const weights = @import("cuda_prompt_wt.zig");
const load = @import("cuda_prompt_load.zig");
pub const batch = @import("cuda_batch.zig");

const layers_n = weights.layers_n;
const linear_n = weights.linear_n;
const attn_n = weights.attn_n;
const dims = weights.dims;
const streams = weights.streams;
const wide = weights.wide;
const vocab = weights.vocab;
const experts_n = weights.experts_n;
const slots = weights.slots;
const moe_w = weights.moe_w;
const head_dim = weights.head_dim;
const index_dim = weights.index_dim;
const proj_n = weights.proj_n;
const out_k = weights.out_k;
const ple_layer = weights.ple_layer;
const inj_stride = weights.inj_stride;
const wts_stride = weights.wts_stride;
const ple_tail_n = weights.ple_tail_n;
const state_n = weights.state_n;
const kv_heads = weights.kv_heads;
const Store = weights.Store;
const Face = weights.Face;
const Q4 = weights.Q4;
const HcW = weights.HcW;
const linearLayer = weights.linearLayer;
const linearIndex = weights.linearIndex;
const attnIndex = weights.attnIndex;
const promote = weights.promote;
const asF32 = weights.asF32;
const uploadQ4 = weights.uploadQ4;
const join3 = weights.join3;
const loadHc = weights.loadHc;
const stitch = weights.stitch;
const Layer = load.Layer;
const loadLayer = load.loadLayer;

pub fn argmax(raw: []const u8) u32 {
    var best = promote(std.mem.readInt(u16, raw[0..2], .little));
    var at: usize = 0;
    var i: usize = 1;
    while (i < vocab) : (i += 1) {
        const v = promote(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
        if (v > best) {
            best = v;
            at = i;
        }
    }
    return @intCast(at);
}

/// `n` zeroed buffers, the fill on the compute stream: a legacy-stream memset returns before it lands.
fn zeros(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, n: usize, len: usize) ![]cuda.DeviceBuffer {
    const Zeroed = struct {
        driver: *cuda.Driver,
        stream: cuda.Stream,
        len: usize,
        fn make(z: @This()) !cuda.DeviceBuffer {
            return cuda.DeviceBuffer.alloc(z.driver, z.len);
        }
        fn fill(z: @This(), b: *cuda.DeviceBuffer) !void {
            try b.fill8(0, z.stream.handle);
        }
    };
    return made(cuda.DeviceBuffer, gpa, n, Zeroed{ .driver = driver, .stream = stream, .len = len });
}

/// `n` buffers from `src.make`, each through `src.fill`; on a failure every buffer made so far is freed, the last too.
fn made(comptime B: type, gpa: std.mem.Allocator, n: usize, src: anytype) ![]B {
    const out = try gpa.alloc(B, n);
    var held: usize = 0;
    errdefer {
        for (out[0..held]) |*b| b.free();
        gpa.free(out);
    }
    while (held < n) {
        out[held] = try src.make();
        held += 1; // counted before its fill, so a failed fill frees it with the rest
        try src.fill(&out[held - 1]);
    }
    return out;
}

pub const vocabulary: usize = vocab;
/// Prompt plus reply tokens a sequence can hold: the model's window.
pub const max_tokens: usize = weights.window;

/// One position's keys and values in one attention layer, bf16.
pub const kv_row_bytes: usize = kv_heads * head_dim * 2;

/// Device bytes `Engine.newSeq(cap)` allocates: conv tails, recurrent states, KV with indexer keys, the n-gram tail.
pub fn seqBytes(cap: usize) usize {
    const linear = linear_n * ((gdn.taps - 1) * gdn.conv_dim * 2 + state_n * 4);
    const attn = attn_n * (cap * (2 * kv_row_bytes + index_dim * 2) + pooledBytes(cap));
    return linear + attn + ple_tail_n * wide * 2;
}

/// One attention layer's pooled indexer keys for a cache of `cap` tokens: a bf16 row of 128 a block of 4.
pub fn pooledBytes(cap: usize) usize {
    return (cap + weights.index_ratio - 1) / weights.index_ratio * index_dim * 2;
}

const round_graph = @import("cuda_round_graph.zig");

/// One loaded checkpoint: the embedding table, every layer, the mixer and head, and the batched pass's scratch.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: *cuda.Stream,
    store: Store,
    model_dir: []u8,
    table: embed.Table,
    mixer: HcW,
    head_q: Q4,
    eos: i64,
    layers: [layers_n]Layer,
    n_layers: usize,
    batch_k: batch.Kernels,
    /// The n-gram table, open for the engine's life (the batched pass gathers its rows from it).
    ngram: ple.Ngram,
    /// The pass's scratch, grown to the widest chunk run.
    batch_s: ?batch.Scratch = null,
    /// Block scores for the sparse rows of an attention block, grown to the longest sequence's need.
    scores: ?cuda.DeviceBuffer = null,
    /// Rows a batched chunk takes at most (TF_FLASHNEXT_CHUNK; batch.default_chunk by default).
    chunk: usize,
    round_graph: round_graph.Cache = .{},
    collective: ?*@import("cuda_collective.zig").Collective = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, model_dir: []const u8) !*Engine {
        return initRank(gpa, io, driver, stream, model_dir, 1, 0);
    }

    pub fn initRank(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, model_dir: []const u8, world: usize, rank: usize) !*Engine {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.io = io;
        e.driver = driver;
        e.stream = stream;
        e.n_layers = 0;
        e.collective = null;
        e.round_graph = .{};
        errdefer for (e.layers[0..e.n_layers]) |*l| l.free(gpa);
        e.model_dir = try gpa.dupe(u8, model_dir);
        errdefer gpa.free(e.model_dir);
        e.store = try Store.open(gpa, io, e.model_dir);
        errdefer e.store.deinit();
        try e.store.detectNorms();
        const weight = try e.store.tensor(embed.weight_name);
        if (weight.dtype == .bf16) {
            e.table = try embed.Table.uploadDense(driver, weight);
        } else {
            const scales = try e.store.tensor(embed.scales_name);
            const biases = try e.store.tensor(embed.biases_name);
            e.table = try embed.Table.upload(driver, weight, scales, biases);
        }
        errdefer e.table.deinit();
        e.eos = try ple.eosOf(gpa, io, e.model_dir);
        e.mixer = try loadHc(gpa, driver, &e.store, "language_model.model.hyper_connection_mixer", false);
        errdefer e.mixer.free();
        const head_tensor = try e.store.tensor("language_model.lm_head.weight");
        if (head_tensor.dtype == .bf16) {
            if (!head_tensor.is(.bf16, &.{ vocab, dims })) return error.UnexpectedTensor;
            e.head_q = try @import("cuda_projection.zig").dense(gpa, driver, head_tensor);
        } else {
            var buf: [80]u8 = undefined;
            const head_w = try join3(gpa, &e.store, &buf, "language_model.lm_head", "");
            defer gpa.free(head_w.w);
            defer gpa.free(head_w.s);
            defer gpa.free(head_w.b);
            if (head_w.w.len != vocab * (dims / 8) * 4) return error.UnexpectedTensor;
            e.head_q = try uploadQ4(gpa, driver, head_w.w, head_w.s, head_w.b, vocab, dims / 8);
        }
        errdefer e.head_q.free();
        var pack_mod = try cuda.Module.load(driver, cuda.kernels.experts_pack);
        defer pack_mod.unload();
        const pack_fn = try pack_mod.function(weights.pack_symbol);
        for (0..layers_n) |i| {
            e.layers[i] = try load.loadLayerRank(gpa, driver, stream, &e.store, pack_fn, i, world, rank);
            e.n_layers += 1;
            std.log.info("flashnext: rank {d}/{d} loaded layer {d}/{d}", .{ rank + 1, world, i + 1, layers_n });
        }
        e.store.release();
        e.store.closeFile();
        e.ngram = try ple.Ngram.init(gpa, io, e.model_dir);
        errdefer e.ngram.deinit();
        e.batch_k = try batch.Kernels.load(driver);
        e.batch_s = null;
        e.scores = null;
        e.chunk = batch.default_chunk;
        if (std.c.getenv("TF_FLASHNEXT_CHUNK")) |v| e.chunk = std.fmt.parseInt(usize, std.mem.span(v), 10) catch batch.default_chunk;
        if (e.chunk == 0) e.chunk = batch.default_chunk;
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.round_graph.clear();
        if (e.batch_s) |*bs| bs.free(e.gpa);
        if (e.scores) |*b| b.free();
        e.batch_k.deinit();
        e.ngram.deinit();
        for (e.layers[0..e.n_layers]) |*l| l.free(e.gpa);
        e.head_q.free();
        e.mixer.free();
        e.table.deinit();
        e.store.deinit();
        e.gpa.free(e.model_dir);
        e.gpa.destroy(e);
    }

    /// A fresh cache. `cap` is how many tokens it can hold, prompt plus generated.
    pub fn newSeq(e: *Engine, cap: usize) !Seq {
        e.round_graph.clear();
        if (cap == 0) return error.UnexpectedTensor;
        const logits = try e.gpa.alloc(u8, vocab * 2);
        errdefer e.gpa.free(logits);
        const conv = try zeros(e.gpa, e.driver, e.stream.*, linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
        errdefer freeOwned(e.gpa, conv);
        const state = try zeros(e.gpa, e.driver, e.stream.*, linear_n, state_n * 4);
        errdefer freeOwned(e.gpa, state);
        const kc = try zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * kv_heads * head_dim * 2);
        errdefer freeOwned(e.gpa, kc);
        const vc = try zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * kv_heads * head_dim * 2);
        errdefer freeOwned(e.gpa, vc);
        const ikc = try zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * index_dim * 2);
        errdefer freeOwned(e.gpa, ikc);
        const pooled = try zeros(e.gpa, e.driver, e.stream.*, attn_n, pooledBytes(cap));
        errdefer freeOwned(e.gpa, pooled);
        var tail = try cuda.DeviceBuffer.alloc(e.driver, ple_tail_n * wide * 2);
        errdefer tail.free();
        try tail.fill8(0, e.stream.handle);
        return .{
            .conv = conv,
            .state = state,
            .kc = kc,
            .vc = vc,
            .ikc = ikc,
            .pooled = pooled,
            .tail = tail,
            .history = .{ e.eos, e.eos },
            .fed = 0,
            .logits = logits,
            .cap = cap,
        };
    }

    /// Run `ids` on from `seq.fed` and leave the next token's scores in `seq.logits`.
    pub fn predict(e: *Engine, comptime Tri: type, tri: Tri, seq: *Seq, ids: []const u32) !void {
        return e.forward(Tri, tri, seq, ids, .{});
    }

    fn complete(e: *Engine) !void {
        if (e.collective) |c| return c.finish(e.stream.*);
        try e.stream.synchronize();
    }

    /// Block scores for a `cap`-token sequence's chunks of up to `rows` rows, reallocated only to grow.
    pub fn reserveScores(e: *Engine, cap: usize, rows: usize) !u64 {
        const need = batch.scoreBytes(cap, rows);
        if (e.scores) |*b| {
            if (b.len >= need) return b.ptr;
            try e.complete();
            e.round_graph.clear();
            b.free();
            e.scores = null;
        }
        e.scores = try cuda.DeviceBuffer.alloc(e.driver, need);
        return e.scores.?.ptr;
    }

    /// Scratch for chunks of `rows` rows, reallocated only to grow.
    pub fn reserve(e: *Engine, rows: usize) !*batch.Scratch {
        if (e.batch_s) |*bs| {
            if (bs.cap >= rows) return bs;
            try e.complete();
            e.round_graph.clear();
            bs.free(e.gpa);
            e.batch_s = null;
        }
        e.batch_s = try batch.Scratch.alloc(e.gpa, e.driver, e.stream.*, rows);
        return &e.batch_s.?;
    }

    /// The batched pass over `ids` from `seq.fed` in chunks of at most `chunk` rows; a window's hooks take one chunk.
    pub fn forward(e: *Engine, comptime Tri: type, tri: Tri, seq: *Seq, ids: []const u32, hooks: batch.Hooks) !void {
        if (ids.len == 0 or seq.fed + ids.len > seq.cap) return error.UnexpectedTensor;
        if ((hooks.keep != null or hooks.logits == .all) and ids.len > e.chunk) return error.UnexpectedTensor;
        const s = try e.reserve(@min(ids.len, e.chunk));
        const scores = try e.reserveScores(seq.cap, s.cap);
        var p: batch.Pass(Tri) = .{
            .gpa = e.gpa,
            .io = e.io,
            .stream = e.stream.*,
            .tri = tri,
            .k = &e.batch_k,
            .s = s,
            .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = e.layers[0..e.n_layers], .mixer = e.mixer, .head = e.head_q, .ngram = &e.ngram, .collective = e.collective },
            .seq = seq.view(scores),
            .hooks = hooks,
        };
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(s.cap, ids.len - at);
            const last = at + n == ids.len;
            p.hooks.logits = if (last) hooks.logits else .none;
            p.hooks.streams = if (hooks.streams != 0) hooks.streams + at * wide * 2 else 0;
            try p.chunk(ids[at..][0..n], seq.fed + at, if (last and hooks.logits != .none) seq.logits else null);
            at += n;
        }
        if (hooks.logits == .none) try e.complete();
        seq.fed += ids.len;
    }

    /// One decode round: each of `seqs` takes its token of `ids` in one batched pass; scores as if alone.
    pub fn round(e: *Engine, comptime Tri: type, tri: Tri, seqs: []const *Seq, ids: []const u32, out: u64) !void {
        if (seqs.len == 0 or seqs.len != ids.len) return error.UnexpectedTensor;
        var cap: usize = 0;
        for (seqs) |q| {
            if (q.fed + 1 > q.cap) return error.UnexpectedTensor;
            cap = @max(cap, q.cap);
        }
        const rows = try e.gpa.alloc(batch.Lane, seqs.len);
        defer e.gpa.free(rows);
        const s = try e.reserve(seqs.len);
        // each row's attention runs alone on the block scores
        const scores = try e.reserveScores(cap, 1);
        for (rows, seqs) |*l, q| l.* = .{ .seq = q.view(scores), .pos = q.fed };
        var p: batch.Pass(Tri) = .{
            .gpa = e.gpa,
            .io = e.io,
            .stream = e.stream.*,
            .tri = tri,
            .k = &e.batch_k,
            .s = s,
            .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = e.layers[0..e.n_layers], .mixer = e.mixer, .head = e.head_q, .ngram = &e.ngram, .collective = e.collective },
            .seq = rows[0].seq,
            .lanes = rows,
            .hooks = .{ .logits = .all, .logits_out = out },
        };
        if (e.collective != null and round_graph.enabled()) {
            const key = try round_graph.Key.init(seqs, s.ids, scores, out);
            try e.round_graph.run(key, &p, ids);
        } else try p.chunk(ids, 0, null);
        try e.complete();
        const d = e.stream.d;
        for (seqs, 0..) |q, r| {
            try d.check(d.api.cuMemcpyDtoH_v2(q.logits.ptr, out + r * vocab * 2, vocab * 2), "cuMemcpyDtoH");
            q.fed += 1;
        }
    }
};

/// One stream's caches. `fed` tokens have already gone through the model.
pub const Seq = struct {
    rank_id: u64 = 0,
    conv: []cuda.DeviceBuffer,
    state: []cuda.DeviceBuffer,
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
    ikc: []cuda.DeviceBuffer,
    pooled: []cuda.DeviceBuffer,
    tail: cuda.DeviceBuffer,
    history: [2]i64,
    fed: usize,
    logits: []u8,
    cap: usize,

    /// The caches a batched pass reads and leaves; `scores` the engine's block scores (Engine.reserveScores).
    pub fn view(s: *Seq, scores: u64) batch.SeqState {
        return .{ .conv = s.conv, .state = s.state, .kc = s.kc, .vc = s.vc, .ikc = s.ikc, .pooled = s.pooled, .tail = s.tail, .history = &s.history, .cap = s.cap, .scores = scores };
    }

    pub fn deinit(s: *Seq, gpa: std.mem.Allocator) void {
        freeBufs(s.conv);
        freeBufs(s.state);
        freeBufs(s.kc);
        freeBufs(s.vc);
        freeBufs(s.ikc);
        freeBufs(s.pooled);
        gpa.free(s.conv);
        gpa.free(s.state);
        gpa.free(s.kc);
        gpa.free(s.vc);
        gpa.free(s.ikc);
        gpa.free(s.pooled);
        if (s.tail.ptr != 0) s.tail.free();
        gpa.free(s.logits);
    }
};

fn freeBufs(bs: []cuda.DeviceBuffer) void {
    for (bs) |*b| b.free();
}

fn freeOwned(gpa: std.mem.Allocator, bs: []cuda.DeviceBuffer) void {
    freeBufs(bs);
    gpa.free(bs);
}

test "linear layers skip every fourth index and attention starts at 3" {
    try std.testing.expect(linearLayer(0));
    try std.testing.expect(!linearLayer(3));
    try std.testing.expect(!linearLayer(47));
    try std.testing.expectEqual(@as(usize, 0), linearIndex(0));
    try std.testing.expectEqual(@as(usize, 2), linearIndex(2));
    try std.testing.expectEqual(@as(usize, 35), linearIndex(46));
    try std.testing.expectEqual(@as(usize, 0), attnIndex(3));
    try std.testing.expectEqual(@as(usize, 11), attnIndex(47));
}

test "a sequence's device bytes grow by the twelve attention layers' keys, values, indexer keys and pooled blocks" {
    const per_token = attn_n * (2 * kv_heads * head_dim * 2 + index_dim * 2);
    try std.testing.expectEqual(per_token, seqBytes(6) - seqBytes(5));
    try std.testing.expectEqual(per_token + attn_n * index_dim * 2, seqBytes(5) - seqBytes(4));
    try std.testing.expect(seqBytes(max_tokens) > max_tokens * per_token);
}

test "a failed fill frees the buffer it was filling and every one before it" {
    const Fake = struct {
        live: *usize,
        pub fn free(b: *@This()) void {
            b.live.* -= 1;
        }
    };
    const Src = struct {
        live: *usize,
        fills: *usize,
        fail_at: usize,
        fn make(s: @This()) !Fake {
            s.live.* += 1;
            return .{ .live = s.live };
        }
        fn fill(s: @This(), _: *Fake) !void {
            s.fills.* += 1;
            if (s.fills.* == s.fail_at) return error.FillFailed;
        }
    };
    for ([_]usize{ 1, 2, 4 }) |fail_at| {
        var live: usize = 0;
        var fills: usize = 0;
        try std.testing.expectError(error.FillFailed, made(Fake, std.testing.allocator, 4, Src{ .live = &live, .fills = &fills, .fail_at = fail_at }));
        try std.testing.expectEqual(@as(usize, 0), live);
    }
    var live: usize = 0;
    var fills: usize = 0;
    const all = try made(Fake, std.testing.allocator, 3, Src{ .live = &live, .fills = &fills, .fail_at = 0 });
    defer std.testing.allocator.free(all);
    try std.testing.expectEqual(@as(usize, 3), live);
}
