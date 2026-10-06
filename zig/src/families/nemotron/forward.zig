//! One Nemotron-H forward over streams' row segments, encoded as FusedDecode's kernels with their geometry.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const kern = @import("kernels.zig");
const layers = @import("layers.zig");
const tree = @import("tree.zig");
const encoder = @import("encoder.zig");
const Sampling = @import("lanes").sampling.Sampling;
const fullVocabulary = @import("lanes").gpu_full.fullVocabulary;
const row = @import("../../core/row_projection.zig");

const Buffer = mtl.Buffer;
pub const Enc = encoder.Enc;
pub const Profiler = encoder.Profiler;

/// Which rows the head runs on: none, each segment's last (prompt chunks), or all (windows, decode steps).
pub const Head = enum { none, last, all };

/// A keyed sampled draw: row r at `position + r` with the stream's settings.
pub const Sampled = struct { sampling: Sampling, position: u64 };

/// How a row is drawn: argmax (ties to the lowest id), or the keyed GPU sampler.
pub const Draw = union(enum) { greedy, sampled: Sampled };

/// Where a segment's Mamba state goes: after all its rows (prompts, steps), its replayed rows (windows), or each row.
pub const Store = enum { full, lag, all };

/// One stream's rows in a forward: its cache (its replayed rows run first), the state slot it stores (-1: none), its draw.
pub const Seg = struct {
    rows: usize,
    cache: *st.Cache,
    store: Store = .lag,
    slot: i32 = -1,
    slots: []const i32 = &.{},
    parents: ?[]const i32 = null, // a tree's rows: each row's parent row (-1: after the replayed rows)
    draw: Draw = .greedy,
    gpu: ?layers.GpuArgs = null, // its kv write and attention dims written on the GPU (a GPU-side round)
    gpu_tree: ?tree.GpuTree = null, // its rows a tree whose tables and dims the GPU writes (a GPU-side round)
};

pub fn ints8(values: []const i32) [8]i32 {
    var out: [8]i32 = @splat(0);
    @memcpy(out[0..values.len], values);
    return out;
}

pub const Forward = struct {
    c: cfg.Config,
    k: *const kern.Kernels,
    w: *const wts.Weights,
    s: *st.Scratch,
    pool: *st.Pool,
    geometry: usize = 0, // routed experts' (rows, simdgroups) variant: 0 generated, i: kernels.geometries[i - 1]
    members: usize = 0, // routed experts' member rows a pass (2 or 4: the shared-unpack kernels; 0: one at a time)
    round: ?layers.RoundArgs = null, // a GPU-side round's Mamba tables and routed rows, written on the GPU (one segment)

    pub fn mp(rows: usize) usize {
        return 16 * ((rows + 15) / 16);
    }

    pub fn mdims(rows: usize) [8]i32 {
        return ints8(&.{ @intCast(rows), @intCast(mp(rows)) });
    }

    /// lane_qmm's coop kernel for this projection shape and row count (key: in, out, down, qkv, head, eh, draft).
    pub fn coop(self: Forward, e: *Enc, comptime key: []const u8, lin: wts.Linear, x: Buffer, x_off: usize, xs: Buffer, y: Buffer, rows: usize) void {
        if (self.rowCall(e, lin, x, x_off, y, 0, rows, false)) return;
        const m = mp(rows);
        const block = if (m <= 32) m else 32;
        const tmr = block / 16;
        const edge = m % block != 0;
        const sk = splitK(lin.n, lin.k);
        if (comptime splits(key)) {
            // the partials fit this scratch's rows (a prompt chunk of up to 128 rows takes the one-pass kernel)
            if (sk > 1 and splitOn() and m <= mp(self.s.rows)) return self.coopSplit(e, key, lin, x, x_off, xs, y, rows, sk);
        }
        const p = if (tmr == 1) self.k.get("coop_" ++ key ++ "_1_0") else if (!edge) self.k.get("coop_" ++ key ++ "_2_0") else self.k.get("coop_" ++ key ++ "_2_1");
        e.pipe(p);
        e.buf(x, x_off, 0);
        e.buf(xs, 0, 1);
        e.buf(lin.w, 0, 2);
        e.buf(lin.sbt, 0, 3);
        e.bytes(mdims(rows), 4);
        e.buf(y, 0, 5);
        e.run(.{ lin.n / 64 * 64 * sk, (m + block - 1) / block, 1 }, .{ 64 * sk, 1, 1 });
    }

    /// The core's row kernels on a chip without tensor units (the projection in its row layout); false on the tensor path.
    pub fn rowCall(self: Forward, e: *Enc, lin: wts.Linear, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize, relu2: bool) bool {
        const w = lin.rows orelse return false;
        const p = &(self.k.rows orelse return false);
        const c = row.call(p, w, rows, relu2) catch unreachable;
        e.pipe(c.pipeline);
        e.buf(x, x_off, 0);
        e.buf(w.w, w.w_off, 1);
        e.buf(w.scales, w.s_off, 2);
        e.buf(w.biases, w.b_off, 3);
        e.bytes(c.dims, 4);
        e.buf(y, y_off, 5);
        e.run(.{ c.groups * c.threads, 1, 1 }, .{ c.threads, 1, 1 });
        return true;
    }

    /// The same coop math with each K slice its own threadgroup (more threadgroups than cores), summed in slice order.
    fn coopSplit(self: Forward, e: *Enc, comptime key: []const u8, lin: wts.Linear, x: Buffer, x_off: usize, xs: Buffer, y: Buffer, rows: usize, sk: usize) void {
        self.coopParts(e, key, lin, x, x_off, xs, rows, sk);
        e.pipe(self.k.get("tf_coop_combine"));
        e.buf(self.s.part, 0, 0);
        e.bytes(ints8(&.{ @intCast(lin.n), @intCast(rows), @intCast(mp(rows)), @intCast(sk) }), 1);
        e.buf(y, 0, 2);
        e.run(.{ lin.n, rows, 1 }, .{ @min(lin.n, 256), 1, 1 });
    }

    /// coopSplit's partials alone (scratch.part [sk, MP, N]), for a kernel that sums them itself.
    pub fn coopParts(self: Forward, e: *Enc, comptime key: []const u8, lin: wts.Linear, x: Buffer, x_off: usize, xs: Buffer, rows: usize, sk: usize) void {
        const m = mp(rows);
        const block = if (m <= 32) m else 32;
        const tmr = block / 16;
        const edge = m % block != 0;
        const p = if (tmr == 1) self.k.get("coop_" ++ key ++ "_1_0_sk") else if (!edge) self.k.get("coop_" ++ key ++ "_2_0_sk") else self.k.get("coop_" ++ key ++ "_2_1_sk");
        e.pipe(p);
        e.buf(x, x_off, 0);
        e.buf(xs, 0, 1);
        e.buf(lin.w, 0, 2);
        e.buf(lin.sbt, 0, 3);
        e.bytes(mdims(rows), 4);
        e.buf(self.s.part, 0, 5);
        e.run(.{ lin.n / 64 * 64, (m + block - 1) / block, sk }, .{ 64, 1, 1 });
    }

    /// Whether coop() splits this projection's K slices at `rows` rows (and how many).
    pub fn splitsAt(self: Forward, comptime key: []const u8, lin: wts.Linear, rows: usize) usize {
        const sk = splitK(lin.n, lin.k);
        return if (comptime splits(key)) (if (sk > 1 and splitOn() and mp(rows) <= mp(self.s.rows)) sk else 0) else 0;
    }

    pub fn xsum(self: Forward, e: *Enc, comptime key: []const u8, k: usize, x: Buffer, x_off: usize, xs: Buffer, rows: usize) void {
        e.pipe(self.k.get(key));
        e.buf(x, x_off, 0);
        e.bytes(mdims(rows), 1);
        e.buf(xs, 0, 2);
        e.run(.{ k / 64, mp(rows), 1 }, .{ @min(k / 64, 256), 1, 1 });
    }

    /// Embedding to the final norm over the segments' rows (ids in segment order), into scratch.x and scratch.xs.
    pub fn body(self: Forward, e: *Enc, segs: []const Seg, ids: Buffer, ids_off: usize) void {
        const c = self.c;
        const s = self.s;
        const d = c.hidden;
        var rows: usize = 0;
        for (segs) |g| rows += g.rows;
        std.debug.assert(rows >= 1 and rows <= s.rows);
        self.embed(e, rows, ids, ids_off, s.h);
        self.rms(e, rows, s.h, 0, d, self.w.norms[0], s.x, 0, d);
        self.xsum(e, "xsum_2688", d, s.x, 0, s.xs, rows);
        const t = layers.Tables.of(segs);
        var deepest: usize = 0;
        for (segs) |g| if (g.parents) |pp| {
            std.debug.assert(segs.len == 1);
            deepest = tree.tables(s, pp);
        };
        var mamba: usize = 0;
        var attention: usize = 0;
        for (0..c.layers) |i| {
            const next = if (i + 1 < c.layers) self.w.norms[i + 1] else self.w.norm_f;
            switch (self.w.layers[i]) {
                .mamba => |m| {
                    layers.mamba(self, e, m, mamba, &t, rows);
                    self.norm(e, s.delta, next, rows, c.eps);
                    mamba += 1;
                },
                .moe => |m| layers.moe(self, e, m, next, rows, c.eps),
                .attention => |a| {
                    var kvs: [st.max_rows]layers.Kvs = undefined;
                    for (segs, 0..) |g, j| kvs[j] = .{ .kv = g.cache.kv[attention], .len = g.cache.len, .rows = g.rows, .tree = g.parents != null or g.gpu_tree != null, .deepest = deepest, .gpu = g.gpu, .gpu_tree = g.gpu_tree, .layer = attention };
                    layers.attention(self, e, a, kvs[0..segs.len], rows);
                    self.norm(e, s.delta, next, rows, c.eps);
                    attention += 1;
                },
            }
        }
    }

    /// The head and draws over each segment's last row (`.last`) or all rows, tokens consecutive from `out_off`.
    pub fn draw(self: Forward, e: *Enc, segs: []const Seg, head: Head, out: Buffer, out_off: usize) void {
        const s = self.s;
        const d = self.c.hidden;
        var at: usize = 0;
        var put = out_off;
        for (segs) |g| {
            switch (head) {
                .none => {},
                .last => {
                    const off = (at + g.rows - 1) * d * 2;
                    self.xsum(e, "xsum_2688", d, s.x, off, s.hx, 1);
                    self.coop(e, "head", self.w.head, s.x, off, s.hx, s.logits, 1);
                    self.pick(e, 1, self.c.vocab, null, g.draw, out, put);
                    put += 4;
                },
                .all => {},
            }
            at += g.rows;
        }
        if (head != .all) return;
        self.coop(e, "head", self.w.head, s.x, 0, s.xs, s.logits, at);
        at = 0;
        for (segs) |g| {
            self.pickAt(e, g.rows, self.c.vocab, at, null, g.draw, out, put);
            put += 4 * g.rows;
            at += g.rows;
        }
    }

    /// Each row of scratch.logits drawn: argmax, or the keyed sampler at the row's position (ids through `map`).
    pub fn pick(self: Forward, e: *Enc, rows: usize, vocab: usize, map: ?Buffer, how: Draw, out: Buffer, out_off: usize) void {
        self.pickAt(e, rows, vocab, 0, map, how, out, out_off);
    }

    fn pickAt(self: Forward, e: *Enc, rows: usize, vocab: usize, row0: usize, map: ?Buffer, how: Draw, out: Buffer, out_off: usize) void {
        const logits_off = row0 * vocab * 2;
        switch (how) {
            .greedy => {
                e.pipe(self.k.get("tf_argmax_bf16"));
                e.buf(self.s.logits, logits_off, 0);
                e.buf(out, out_off, 1);
                e.bytes(@as(u32, @intCast(vocab)), 2);
                e.buf(map orelse out, 0, 3);
                e.bytes(@as(u32, @intFromBool(map != null)), 4);
                e.run(.{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 });
            },
            .sampled => |d| self.sample(e, rows, vocab, logits_off, map, d, out, out_off),
        }
    }

    /// tf_gpu_sample over `rows` rows (tf_sample_full past its 1,024 candidates): a 1024-thread threadgroup a row.
    fn sample(self: Forward, e: *Enc, rows: usize, vocab: usize, logits_off: usize, map: ?Buffer, d: Sampled, out: Buffer, out_off: usize) void {
        var seeds: [2 * st.max_rows]u32 = @splat(0);
        var positions: [st.max_rows]u32 = @splat(0);
        var cfgs: [4 * st.max_rows]f32 = @splat(0);
        var caps: [st.max_rows]u32 = @splat(0);
        const t = d.sampling;
        for (0..rows) |r| {
            seeds[2 * r] = @truncate(t.seed);
            seeds[2 * r + 1] = @truncate(t.seed >> 32);
            positions[r] = @intCast(d.position + r);
            cfgs[4 * r] = @floatCast(1.0 / @max(t.temperature, 1e-6));
            cfgs[4 * r + 1] = @floatCast(t.top_p);
            cfgs[4 * r + 2] = 20.0;
            cfgs[4 * r + 3] = @floatCast(t.minLog());
            caps[r] = t.top_k;
        }
        const n = @max(rows, 8);
        const whole = fullVocabulary(t);
        if (whole) {
            e.pipe(if (map != null) self.k.get("tf_sample_full_ids") else self.k.get("tf_sample_full"));
        } else e.pipe(if (map != null) self.k.get("sample_ids") else self.k.get("sample"));
        e.buf(self.s.logits, logits_off, 0);
        e.e.setBytes(std.mem.sliceAsBytes(seeds[0 .. 2 * n]), 1);
        e.e.setBytes(std.mem.sliceAsBytes(positions[0..n]), 2);
        e.e.setBytes(std.mem.sliceAsBytes(cfgs[0 .. 4 * n]), 3);
        e.e.setBytes(std.mem.sliceAsBytes(caps[0..n]), 4);
        if (whole) {
            e.buf(out, out_off, 5);
            e.buf(map orelse out, 0, 6);
            e.bytes(@as(u32, @intCast(vocab)), 7);
        } else if (map) |m| {
            e.buf(m, 0, 5);
            e.buf(out, out_off, 6);
        } else e.buf(out, out_off, 5);
        e.run(.{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 });
    }

    /// The embedding rows of the token ids at `ids` + `ids_off` into `out` [rows, D].
    pub fn embed(self: Forward, e: *Enc, rows: usize, ids: Buffer, ids_off: usize, out: Buffer) void {
        const d = self.c.hidden;
        e.pipe(self.k.get("tf_embed_q4"));
        e.buf(ids, ids_off, 0);
        e.buf(self.w.embed[0].buffer, self.w.embed[0].offset, 1);
        e.buf(self.w.embed[1].buffer, self.w.embed[1].offset, 2);
        e.buf(self.w.embed[2].buffer, self.w.embed[2].offset, 3);
        e.buf(out, 0, 4);
        e.bytes(@as(u32, @intCast(d)), 5);
        e.run(.{ d / 2, rows, 1 }, .{ 256, 1, 1 });
    }

    /// MLX's RMS norm of `rows` rows of D (`stride` and `out_stride` elements apart) with weight `w`.
    pub fn rms(self: Forward, e: *Enc, rows: usize, x: Buffer, x_off: usize, stride: usize, w: anytype, out: Buffer, out_off: usize, out_stride: usize) void {
        const d = self.c.hidden;
        const threads = (d / 4 + 31) / 32 * 32;
        e.pipe(self.k.get("tf_rms_mlx"));
        e.buf(x, x_off, 0);
        e.buf(w.buffer, w.offset, 1);
        e.buf(out, out_off, 2);
        e.bytes(self.c.eps, 3);
        e.bytes(@as(u32, @intCast(d)), 4);
        e.bytes(@as(u32, 1), 5);
        e.bytes([2]u32{ @intCast(stride), @intCast(out_stride) }, 6);
        e.run(.{ threads * rows, 1, 1 }, .{ threads, 1, 1 });
    }

    /// Residual add of `delta` into h, the next input norm (`next`), and its 64-group sums into xs.
    pub fn norm(self: Forward, e: *Enc, delta: Buffer, next: anytype, rows: usize, eps: f32) void {
        const s = self.s;
        e.pipe(self.k.get("add_norm_xs"));
        e.buf(s.h, 0, 0);
        e.buf(delta, 0, 1);
        e.buf(next.buffer, next.offset, 2);
        e.bytes(eps, 3);
        e.bytes(mdims(rows), 4);
        e.buf(s.h, 0, 5);
        e.buf(s.x, 0, 6);
        e.buf(s.xs, 0, 7);
        e.run(.{ 896 * mp(rows), 1, 1 }, .{ 896, 1, 1 });
    }

    /// u32 rows of `src` from `src_off` into `dst` at `dst_off` (window ids into a shared round's ids).
    pub fn copyIds(self: Forward, e: *Enc, src: Buffer, src_off: usize, dst: Buffer, dst_off: usize, n: usize) void {
        e.pipe(self.k.get("tf_copy_u32"));
        e.buf(src, src_off, 0);
        e.buf(dst, dst_off, 1);
        e.run(.{ n, 1, 1 }, .{ 64, 1, 1 });
    }
};

/// K slices for an (n, k) weight: fixed by the shape, never by the row count (lane_qmm.split_k).
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 31) / 32;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 1024 and (k / 64) / (sk * 2) >= 8) sk *= 2;
    return sk;
}

test "split_k matches the Python shapes" {
    try std.testing.expectEqual(@as(usize, 4), splitK(10304, 2688));
    try std.testing.expectEqual(@as(usize, 8), splitK(2688, 4096));
    try std.testing.expectEqual(@as(usize, 4), splitK(2688, 3712));
    try std.testing.expectEqual(@as(usize, 4), splitK(4608, 2688));
    try std.testing.expectEqual(@as(usize, 1), splitK(131072, 2688));
    try std.testing.expectEqual(@as(usize, 4), splitK(3712, 2688));
}

/// Projections whose 64-column tiles are too few to fill the GPU, so forward.coop splits their K slices.
fn splits(comptime key: []const u8) bool {
    inline for (.{ "in", "out", "eh" }) |k| if (comptime std.mem.eql(u8, k, key)) return true;
    return false;
}

var split_mode: enum(u8) { unread, on, off } = .unread;

/// TF_COOP_SPLIT=0 keeps one threadgroup a column tile, as the Python engine dispatches.
fn splitOn() bool {
    if (split_mode == .unread) {
        const v = std.c.getenv("TF_COOP_SPLIT");
        split_mode = if (v != null and v.?[0] == '0') .off else .on;
    }
    return split_mode == .on;
}
