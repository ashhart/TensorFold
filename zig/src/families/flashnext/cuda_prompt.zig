//! The loaded engine and one stream's caches: prompts and windows on the batched pass, decode rounds over streams.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const weights = @import("cuda_prompt_wt.zig");
const load = @import("cuda_prompt_load.zig");
const kvpage = @import("cuda_exl3_kvpage.zig");
const shared_kv = @import("cuda_prompt_shared_kv.zig");
pub const batch = @import("cuda_batch.zig");
const fb = @import("cuda_fn_back.zig");
const pocket = @import("cuda_pocket.zig");

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
    scratch_format: @import("cuda_batch_state.zig").ScratchFormat = .classic,
    /// Block scores for the sparse rows of an attention block, grown to the longest sequence's need.
    scores: ?cuda.DeviceBuffer = null,
    /// Rows a batched chunk takes at most (TF_FLASHNEXT_CHUNK; batch.default_chunk by default).
    chunk: usize,
    round_graph: round_graph.Cache = .{},
    image_factory: ?*@import("cuda_images.zig").Vision = null,
    collective: ?*@import("cuda_collective.zig").Collective = null,
    page_config: ?kvpage.Config = null,
    shared_arena: ?*kvpage.shared_device.Arena = null,
    /// Each layer's Living Weights pocket, from a taught folder's shards or the learner: every pass adds it.
    pockets: pocket.Pockets = .{},
    /// During a round of taught and untaught streams: its rows' pocket scales (0 on an untaught row); 0: none.
    taught_rows: u64 = 0,

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
        e.scratch_format = .classic;
        e.collective = null;
        e.image_factory = null;
        e.page_config = kvpage.configFromEnv();
        e.shared_arena = null;
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
        const exl3_head = try @import("cuda_exl3_load.zig").head(gpa, driver, &e.store, world, rank);
        const head_tensor = if (exl3_head == null) try e.store.tensor("language_model.lm_head.weight") else undefined;
        if (exl3_head) |q| e.head_q = q else if (head_tensor.dtype == .bf16) {
            if (!head_tensor.is(.bf16, &.{ vocab, dims })) return error.UnexpectedTensor;
            e.head_q = try @import("cuda_projection.zig").dense(gpa, driver, if (try @import("cuda_head.zig").enabled()) try @import("cuda_head.zig").shard(head_tensor, world, rank) else head_tensor);
        } else {
            if (try @import("cuda_head.zig").enabled()) return error.BadHeadSharding;
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
        e.pockets = .{};
        errdefer e.pockets.deinit();
        const rows = try gpa.alloc(f32, 3 * fb.max_pocket * dims);
        defer gpa.free(rows);
        var learned: usize = 0;
        for (0..layers_n) |i| {
            const cap = fb.max_pocket * dims;
            var pk: load.Pocket = .{ .gate = rows[0..cap], .up = rows[cap..][0..cap], .down = rows[2 * cap ..][0..cap] };
            e.layers[i] = try load.loadLayerRank(gpa, driver, stream, &e.store, pack_fn, i, world, rank, &pk);
            e.n_layers += 1;
            std.log.info("flashnext: rank {d}/{d} loaded layer {d}/{d}", .{ rank + 1, world, i + 1, layers_n });
            if (pk.n == 0) continue;
            const used = pocket.usedOf(pk.up, pk.n);
            try e.pockets.set(driver, stream, i, pk.gate[0 .. pk.n * dims], pk.up[0 .. pk.n * dims], pk.down[0 .. pk.n * dims], used);
            learned += used;
        }
        if (learned > 0) std.log.info("flashnext: the folder's Living Weights pockets hold {d} learned neurons beside the shared experts", .{learned});
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
        @import("cuda_exl3_prefill.zig").releaseFor(e);
        @import("cuda_exl3_graph.zig").releaseFor(e);
        @import("cuda_exl3_recycle.zig").drain(e.gpa);
        if (e.page_config) |cfg| if (cfg.shared_lanes != 0) {
            if (e.shared_arena) |arena| arena.destroy() catch @panic("shared KV arena still owns live lanes");
        };
        e.round_graph.clear();
        e.pockets.deinit();
        if (e.batch_s) |*bs| bs.free(e.gpa);
        if (e.scores) |*b| b.free();
        e.batch_k.deinit();
        @import("cuda_exl3_kv.zig").forget(e.gpa, e);
        e.ngram.deinit();
        for (e.layers[0..e.n_layers]) |*l| l.free(e.gpa);
        e.head_q.free();
        e.mixer.free();
        e.table.deinit();
        e.store.deinit();
        e.gpa.free(e.model_dir);
        e.gpa.destroy(e);
    }

    /// Layer `layer`'s pocket for every pass: gate rows, up rows and outputs fp32 [n, dims], the first `used` live.
    pub fn setPocket(e: *Engine, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
        try e.pockets.set(e.driver, e.stream, layer, gate, up, down, used);
    }

    /// Layer `layer`'s pocket into gate, up and down ([n, dims] each); the neurons in use (0: none).
    pub fn pocketRead(e: *Engine, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
        return e.pockets.read(e.stream, layer, gate, up, down);
    }

    /// Layer `layer`'s pocket as the kernels take it, or null with no neuron in use.
    pub fn pocketOf(e: *const Engine, layer: usize) ?fb.Ops.Pocket {
        return e.pockets.of(layer, if (e.batch_s) |bs| bs.pocket_act else 0);
    }

    /// The pocket's kernels on the engine's stream (loaded with the first pocket).
    pub fn pocketOps(e: *const Engine) ?fb.Ops {
        return if (e.pockets.k) |*k| .{ .k = k, .s = e.stream.* } else null;
    }

    /// Every pass's adds to the MoE outputs: each taught layer's pocket behind its shared gate.
    fn servePocket(ctx: *anyopaque, layer: usize, x: u64, ids: u64, wts: u64, out: u64, rows: usize) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        var pk = e.pocketOf(layer) orelse return;
        pk.scale = e.taught_rows;
        try e.pocketOps().?.pocketRows(pk, .{ .x = x, .ids = ids, .wts = wts }, out, true, rows, dims, slots - 1);
    }

    /// The resident weights a pass reads, with the taught layers' pockets.
    pub fn model(e: *Engine) batch.Model {
        const m = e.pockets.mask();
        return .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .collective = e.collective, .layers = e.layers[0..e.n_layers], .mixer = e.mixer, .head = e.head_q, .ngram = &e.ngram, .pocket = if (m != 0) .{ .ctx = e, .add = servePocket, .layers = m } else null };
    }

    /// The weights a pass of streams that read the model as shipped reads (untaught): the pockets left out.
    fn shipped(e: *Engine) batch.Model {
        var m = e.model();
        m.pocket = null;
        return m;
    }

    /// A fresh cache. `cap` is how many tokens it can hold, prompt plus generated.
    pub fn newSeq(e: *Engine, cap: usize) !Seq {
        e.round_graph.clear();
        if (cap == 0) return error.UnexpectedTensor;
        if (e.page_config) |cfg| if (cfg.shared_lanes != 0) {
            const arena = e.shared_arena orelse return error.SharedKvArenaRequired;
            if (cap != arena.layout.options.cap) return error.SharedKvCommonCap;
        };
        if (e.page_config == null) if (try @import("cuda_exl3_recycle.zig").takeSeq(e, cap)) |kept| return kept;
        if (@import("cuda_exl3_kv.zig").of(e).placed()) return @import("cuda_exl3_kv.zig").newSeq(e, Seq, cap); // K10 placement
        const logits = try e.gpa.alloc(u8, vocab * 2);
        errdefer e.gpa.free(logits);
        const conv = try zeros(e.gpa, e.driver, e.stream.*, linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
        errdefer freeOwned(e.gpa, conv);
        const state = try zeros(e.gpa, e.driver, e.stream.*, linear_n, state_n * 4);
        errdefer freeOwned(e.gpa, state);
        // K/V paged to host memory (TF_FLASHNEXT_KVPAGE): kc/vc stay one row (cuda_exl3_kvpage.zig holds the rest)
        const paged = if (e.page_config) |cfg| if (cfg.shared_lanes != 0) try kvpage.Paged.initShared(e.gpa, e.driver, cap, cfg.forChunk(e.chunk), e.shared_arena orelse return error.SharedKvArenaRequired) else try kvpage.Paged.init(e.gpa, e.driver, cap, cfg.forChunk(e.chunk)) else null;
        errdefer if (paged) |pg| pg.destroy();
        const kc = try zeros(e.gpa, e.driver, e.stream.*, attn_n, (if (paged != null) 1 else cap) * kv_heads * head_dim * 2);
        errdefer freeOwned(e.gpa, kc);
        const vc = try zeros(e.gpa, e.driver, e.stream.*, attn_n, (if (paged != null) 1 else cap) * kv_heads * head_dim * 2);
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
            .paged = paged,
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

    pub fn reserveSharedKv(e: *Engine, cap: usize) !usize {
        return shared_kv.reserve(e, cap);
    }

    fn beginShared(e: *Engine, seqs: []const *Seq) !?kvpage.shared_device.Lease {
        return shared_kv.begin(e, seqs);
    }

    fn endShared(e: *Engine, lease: ?kvpage.shared_device.Lease) void {
        shared_kv.end(e, lease);
    }

    fn failShared(e: *Engine, lease: ?kvpage.shared_device.Lease, seqs: []const *Seq) void {
        shared_kv.fail(e, lease, seqs);
    }

    fn finishPaged(e: *Engine, seqs: []const *Seq) !void {
        return shared_kv.finish(e, seqs, complete);
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
            if (bs.format != e.scratch_format) return error.ScratchFormatChanged;
            if (bs.cap >= rows) return bs;
            try e.complete();
            e.round_graph.clear();
            bs.free(e.gpa);
            e.batch_s = null;
        }
        e.batch_s = try batch.Scratch.allocForFormat(e.gpa, e.driver, e.stream.*, rows, e.scratch_format);
        return &e.batch_s.?;
    }

    /// The batched pass over `ids` from `seq.fed` in chunks of at most `chunk` rows; a window's hooks take one chunk.
    pub fn forward(e: *Engine, comptime Tri: type, tri: Tri, seq: *Seq, ids: []const u32, hooks: batch.Hooks) !void {
        if (seq.invalidated) return error.SnapshotDestinationInvalid;
        if (ids.len == 0 or seq.fed > seq.cap or ids.len > seq.cap - seq.fed) return error.UnexpectedTensor;
        const shared_sequences = [_]*Seq{seq};
        const shared_lease = try e.beginShared(&shared_sequences);
        defer e.endShared(shared_lease);
        errdefer e.failShared(shared_lease, &shared_sequences);
        if (try @import("cuda_exl3_prefill.zig").forward(e, Tri, tri, seq, ids, hooks)) {
            const one = [_]*Seq{seq};
            return e.finishPaged(&one);
        }
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
            .m = if (seq.untaught) e.shipped() else e.model(),
            .seq = seq.view(scores),
            .hooks = hooks,
        };
        if (try @import("cuda_exl3_graph.zig").forward(e, &p, seq, ids)) {
            const one = [_]*Seq{seq};
            return e.finishPaged(&one);
        }
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(s.cap, ids.len - at);
            const last = at + n == ids.len;
            p.hooks.logits = if (last) hooks.logits else .none;
            p.hooks.streams = if (hooks.streams != 0) hooks.streams + at * wide * 2 else 0;
            try p.chunk(ids[at..][0..n], seq.fed + at, if (last and hooks.logits != .none) seq.logits else null);
            at += n;
        }
        if (seq.paged != null) {
            const one = [_]*Seq{seq};
            try e.finishPaged(&one);
        } else if (hooks.logits == .none) try e.complete();
        seq.fed += ids.len;
    }

    /// One decode round: each of `seqs` takes its token of `ids` in one batched pass; scores as if alone.
    pub fn round(e: *Engine, comptime Tri: type, tri: Tri, seqs: []const *Seq, ids: []const u32, out: u64) !void {
        if (seqs.len == 0 or seqs.len != ids.len) return error.UnexpectedTensor;
        var cap: usize = 0;
        for (seqs) |q| {
            if (q.invalidated) return error.SnapshotDestinationInvalid;
            if (q.fed + 1 > q.cap) return error.UnexpectedTensor;
            cap = @max(cap, q.cap);
        }
        const shared_lease = try e.beginShared(seqs);
        defer e.endShared(shared_lease);
        errdefer e.failShared(shared_lease, seqs);
        const rows = try e.gpa.alloc(batch.Lane, seqs.len);
        defer e.gpa.free(rows);
        const s = try e.reserve(seqs.len);
        // each row's attention runs alone on the block scores
        const scores = try e.reserveScores(cap, 1);
        for (rows, seqs) |*l, q| l.* = .{ .seq = q.view(scores), .pos = q.fed };
        // an untaught stream's row reads no pocket: a round of them alone leaves it out, a mixed one scales it by 0
        const untaught = try e.gpa.alloc(bool, seqs.len);
        defer e.gpa.free(untaught);
        const scales = try e.gpa.alloc(f32, seqs.len);
        defer e.gpa.free(scales);
        for (untaught, seqs) |*u, q| u.* = q.untaught;
        const reads = pocket.reads(untaught, scales);
        defer e.taught_rows = 0;
        if (reads == .some and e.pockets.mask() != 0) e.taught_rows = try e.pockets.rowScales(e.driver, e.stream, scales);
        var p: batch.Pass(Tri) = .{
            .gpa = e.gpa,
            .io = e.io,
            .stream = e.stream.*,
            .tri = tri,
            .k = &e.batch_k,
            .s = s,
            .m = if (reads == .none) e.shipped() else e.model(),
            .seq = rows[0].seq,
            .lanes = rows,
            .hooks = .{ .logits = .all, .logits_out = out },
        };
        const has_media = for (seqs) |q| {
            if (q.media != null) break true;
        } else false;
        const has_paged = for (seqs) |q| {
            if (q.paged != null) break true;
        } else false;
        if (try @import("cuda_exl3_graph.zig").round(e, &p, seqs, ids, scores, out)) {} else if (e.collective != null and round_graph.enabled() and !has_media and !has_paged) {
            const key = try round_graph.Key.init(seqs, s.ids, scores, out);
            try e.round_graph.run(key, &p, ids);
        } else try p.chunk(ids, 0, null);
        try e.finishPaged(seqs);
        if (!has_paged) try e.complete();
        const d = e.stream.d;
        for (seqs, 0..) |q, r| {
            try d.check(d.api.cuMemcpyDtoH_v2(q.logits.ptr, out + r * vocab * 2, vocab * 2), "cuMemcpyDtoH");
            q.fed += 1;
        }
    }
};

test "exl3: Engine graph-hit lease covers every owner without paging callbacks or extra completion waits" {
    try @import("cuda_prompt_test.zig").graphLease(Engine, Seq, .{
        .begin = Engine.beginShared,
        .finish = Engine.finishPaged,
        .end = Engine.endShared,
        .fail = Engine.failShared,
    });
}

/// One stream's caches. `fed` tokens have already gone through the model.
pub const Seq = struct {
    invalidated: bool = false,
    media: ?@import("cuda_media.zig").State = null,
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
    /// int8 K/V scales beside the codes in kc/vc, and the indexer keys' mapped host buffer (cuda_exl3_kv.newSeq).
    kv8: ?@import("cuda_exl3_kv.zig").SeqKv8 = null,
    ikc_host: ?cuda.HostBuffer = null,
    paged: ?*kvpage.Paged = null,
    /// The model as shipped: no learned neuron acts on its rows (a lesson writer's and its judge's reads).
    untaught: bool = false,

    /// The caches a batched pass reads and leaves; `scores` the engine's block scores (Engine.reserveScores).
    pub fn view(s: *Seq, scores: u64) batch.SeqState {
        return .{ .media = if (s.media) |*m| m.view() else null, .conv = s.conv, .state = s.state, .kc = s.kc, .vc = s.vc, .ikc = s.ikc, .pooled = s.pooled, .tail = s.tail, .history = &s.history, .cap = s.cap, .scores = scores, .kv8 = s.kv8, .paged = s.paged };
    }

    pub fn deinit(s: *Seq, gpa: std.mem.Allocator) void {
        if (s.invalidated) cuda.completion.drain(s.tail.d, .{ .d = s.tail.d, .handle = null }, "invalid snapshot destination did not drain");
        if (@import("cuda_exl3_recycle.zig").giveSeq(s)) return; // graphs: kept for the next stream (stable addresses)
        @import("cuda_exl3_graph.zig").releaseSeq(s);
        if (s.media) |*m| m.deinit();
        @import("cuda_exl3_kv.zig").deinitSeq(gpa, &s.kv8, &s.ikc_host);
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
        if (s.paged) |pg| pg.destroy();
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
    try @import("cuda_prompt_test.zig").layerIndices();
}

test "a sequence's device bytes grow by the twelve attention layers' keys, values, indexer keys and pooled blocks" {
    try @import("cuda_prompt_test.zig").sequenceBytes(seqBytes);
}

test "a failed fill frees the buffer it was filling and every one before it" {
    try @import("cuda_prompt_test.zig").failedFill(made);
}
