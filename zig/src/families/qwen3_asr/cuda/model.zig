//! Qwen3-ASR on CUDA: the shared tower schedule over this directory's binding, the Qwen3 layers through qwen3's pass.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const q3 = @import("qwen3_pass");
const cfg = @import("../config.zig");
const mel = @import("../mel.zig");
const names = @import("../names.zig");
const tower = @import("../tower.zig");
const shared = @import("../lm.zig");
const store = @import("store.zig");
const ctower = @import("tower.zig");

pub const Dump = ctower.Dump;
pub const Splice = q3.lm.Splice;

pub const Options = struct {
    group: usize = 16, // chunks a convolution pass
    audio_rows: usize, // the longest clip's tower rows
    pass_rows: usize, // the most ids a language model pass takes
    positions: usize, // the most positions the caches hold
    head_rows: usize = 1, // the most rows a pass's head scores
};

/// The Qwen3 spec of the checkpoint's text config: tied, as the loader checks.
pub fn spec(t: cfg.Text) q3.config.Spec {
    return .{ .layers = t.layers, .hidden = t.hidden, .heads = t.heads, .kv_heads = t.kv_heads, .head_dim = t.head_dim, .inter = t.intermediate, .vocab = t.vocab, .eps = t.eps, .theta = t.theta, .max_positions = t.context, .tied = true };
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    c: cfg.Config,
    st: store.Store,
    weight_bytes: u64,
    tw: tower.Weights,
    tb: tower.Buffers,
    tr: ctower.Tower,
    qm: q3.weights.Model,
    lm: q3.lm.Lm,
    a: q3.lm.Addrs,
    cap: usize, // positions the caches hold
    pass_rows: usize,
    head_rows: usize,
    cache: q3.lm.Cache,
    logits_dev: u64,
    logits: []u16,
    ids: []u32,
    last_margin: f32 = 0, // the last pick's lead over the runner-up
    keep: ?Keep = null, // each head row's logits, appended
    sink: ?Dump = null,

    /// Where the head's logits go, a row of vocab bf16 values at a time.
    pub const Keep = struct { gpa: std.mem.Allocator, rows: *std.ArrayList(u16) };

    /// The checkpoint at `dir` on the device (every tensor used), with scratch for `o`'s clips and passes.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8, c: cfg.Config, o: Options) !*Model {
        const sp = spec(c.text);
        const cap = std.mem.alignForward(usize, o.positions, q3.lm.chunk_keys);
        if (cap > c.text.context) return error.ContextTooLong;
        if (o.pass_rows == 0 or o.pass_rows > o.positions or o.head_rows == 0 or o.head_rows > o.pass_rows) return error.PassRows;
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .d = d, .s = s, .c = c, .st = .{ .gpa = gpa, .d = d }, .weight_bytes = 0, .tw = undefined, .tb = undefined, .tr = undefined, .qm = undefined, .lm = undefined, .a = .{}, .cap = cap, .pass_rows = o.pass_rows, .head_rows = o.head_rows, .cache = undefined, .logits_dev = 0, .logits = &.{}, .ids = &.{} };
        errdefer m.st.deinit();
        {
            var ck = try core.Checkpoint.openModel(gpa, io, dir);
            defer ck.close();
            m.tw = try tower.Weights.load(gpa, c.audio, store.Source{ .ck = &ck }, &m.st, o.group);
            errdefer m.tw.deinit(gpa);
            ck.scope = names.lm;
            m.qm = try q3.weights.Model.fromCheckpoint(gpa, d, &ck, sp);
            errdefer m.qm.deinit();
            ck.scope = "";
            // the tied head: the stored lm_head must be the embedding table, bit for bit
            const embed = (try ck.expect(names.lm ++ "embed_tokens.weight", .bf16, &.{ sp.vocab, sp.hidden })).bytes;
            if (!std.mem.eql(u8, embed, (try ck.expect(names.head, .bf16, &.{ sp.vocab, sp.hidden })).bytes)) return error.UntiedHead;
            if (ck.unused() != 0) return error.UnexpectedTensors;
        }
        errdefer m.tw.deinit(gpa);
        errdefer m.qm.deinit();
        for (m.qm.st.list.items) |b| m.weight_bytes += b.len;
        m.weight_bytes += m.st.bytes;
        m.tb = try tower.Buffers.alloc(c.audio, o.group, o.audio_rows, &m.st);
        m.tr = try ctower.Tower.init(gpa, d, s, c.audio, o.group, o.audio_rows, &m.st);
        errdefer m.tr.deinit();
        m.lm = try q3.lm.Lm.init(gpa, d, s, sp);
        errdefer m.lm.deinit();
        try m.lm.prepare(sp);
        try m.lm.g.prepare(&.{.{ sp.vocab, sp.hidden }}); // the tied head: one more (n, k), which embeddings never run
        try m.lm.g.checkRows(1);
        try m.lm.g.checkRows(o.pass_rows);
        // the scratch sized to the positions: its RoPE, position tables and chunk partials cover the caches
        const sc = q3.lm.Scratch.of(sp, cap, 1);
        m.a = sc.at(try m.st.alloc(sc.total));
        try q3.lm.tables(gpa, d, m.a, sp, cap);
        const stride = cap * sp.kv_heads * sp.head_dim * 2;
        m.cache = .{ .kc = try m.st.alloc(sp.layers * stride), .vc = try m.st.alloc(sp.layers * stride), .stride = stride };
        m.logits_dev = try m.st.alloc(2 * o.head_rows * sp.vocab);
        m.logits = try gpa.alloc(u16, o.head_rows * sp.vocab);
        errdefer gpa.free(m.logits);
        m.ids = try gpa.alloc(u32, o.pass_rows);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.s.synchronize() catch {};
        m.gpa.free(m.ids);
        m.gpa.free(m.logits);
        m.lm.deinit();
        m.qm.deinit();
        m.tr.deinit();
        m.tw.deinit(m.gpa);
        m.st.deinit();
        m.gpa.destroy(m);
    }

    /// The tower over `feats` into `out` [rows, audio.out] bf16, each stage to `sink` when given; the rows returned.
    pub fn encode(m: *Model, feats: mel.Features, out: u64, sink: ?Dump) !usize {
        m.tr.sink = sink;
        defer m.tr.sink = null;
        return tower.encode(&m.tr, m.gpa, m.c.audio, &m.tw, m.tb, feats, out);
    }

    /// One language model pass over `ids` at [pos, pos + ids.len), `audio` spliced in: the greedy next id.
    pub fn forward(m: *Model, ids: []const u32, pos: usize, audio: ?Splice) !u32 {
        return m.pass(ids, pos, audio, 1, null);
    }

    /// forward with the head over the last `heads` rows: each row's logits to `keep` when set.
    pub fn forwardHeads(m: *Model, ids: []const u32, pos: usize, audio: ?Splice, heads: usize) !u32 {
        return m.pass(ids, pos, audio, heads, null);
    }

    /// forward with its stages (lm_embed, lm0_*, lm_layer{i}, lm_norm) to `sink`.
    pub fn forwardDump(m: *Model, ids: []const u32, pos: usize, audio: ?Splice, sink: ?Dump) !u32 {
        m.sink = sink;
        defer m.sink = null;
        return m.pass(ids, pos, audio, 1, if (sink != null) .{ .ctx = m, .put = tapPut } else null);
    }

    fn tapPut(ctx: *anyopaque, stage: []const u8, ptr: u64, rows: usize, cols: usize) anyerror!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (m.sink) |dm| try dm.put(dm.ctx, stage, ptr, rows, cols, null);
    }

    fn pass(m: *Model, ids: []const u32, pos: usize, audio: ?Splice, heads: usize, tap: ?q3.lm.Tap) !u32 {
        const rows = ids.len;
        if (rows == 0 or rows > m.pass_rows) return error.PassRows;
        if (heads == 0 or heads > rows or heads > m.head_rows) return error.PassHeads;
        if (pos + rows > m.cap) return error.ContextFull;
        for (ids) |t| if (t >= m.c.text.vocab) return error.InvalidBatch;
        try m.s.synchronize(); // the last pass may still read them
        @memcpy(m.ids[0..rows], ids);
        try store.view(m.d, m.a.ids, 4 * rows).upload(0, std.mem.sliceAsBytes(m.ids[0..rows]));
        const x: q3.lm.Extra = .{ .start = pos, .cache = m.cache, .splice = audio, .head = .{ .rows = heads, .logits = m.logits_dev }, .tap = tap };
        try m.lm.forwardFrom(&m.qm, m.a, m.cap, &.{@intCast(rows)}, x);
        const got = m.logits[0 .. heads * m.c.text.vocab];
        try store.view(m.d, m.logits_dev, got.len * 2).downloadAsync(0, std.mem.sliceAsBytes(got), m.s.handle);
        try m.s.synchronize();
        if (m.keep) |k| try k.rows.appendSlice(k.gpa, got);
        const last = got[(heads - 1) * m.c.text.vocab ..];
        const best = shared.argmax(last);
        m.last_margin = shared.margin(last, best);
        return best;
    }

    /// A planted fault for tf-qwen3-asr-forced: every layer's q_norm and k_norm swapped (again: back).
    pub fn swapQkNorms(m: *Model) void {
        for (m.qm.layers) |*l| std.mem.swap(u64, &l.q_norm, &l.k_norm);
    }

    /// Positions [0, n) of every layer's k and v caches, downloaded: k rows then v rows, layer by layer.
    pub fn cacheRows(m: *Model, gpa: std.mem.Allocator, n: usize) ![]u16 {
        const t = m.c.text;
        const row = t.kv_heads * t.head_dim;
        const out = try gpa.alloc(u16, 2 * t.layers * n * row);
        errdefer gpa.free(out);
        try m.s.synchronize();
        for (0..t.layers) |l| for ([_]u64{ m.cache.kc, m.cache.vc }, 0..) |base, j| {
            const dst = out[(2 * l + j) * n * row ..][0 .. n * row];
            try m.d.check(m.d.api.cuMemcpyDtoH_v2(dst.ptr, base + l * m.cache.stride, dst.len * 2), "cuMemcpyDtoH");
        };
        return out;
    }
};
