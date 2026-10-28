//! The release's DSpark drafter at TP: tapped residuals as each layer's context, a 7-row block, a Markov walk.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const kern = @import("cuda_kernels.zig");

/// The checkpoint's DSpark tensors sit under this prefix, in its dspark/ folder.
const prefix = "language_model.model.dspark.";

pub const Config = struct {
    layers: u32 = 5,
    block: u32 = 7, // rows a block runs: the anchor and its noise tokens; each row drafts one token
    window: u32 = 4096, // context positions a block sees before its anchor
    noise: u32 = 200058,
    taps: [5]u32 = .{ 3, 17, 31, 45, 59 },
    rank: u32 = 256, // the Markov head's width

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "dspark", "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        const o = doc.value.object;
        var c: Config = .{};
        const int = struct {
            fn of(x: std.json.Value) !u32 {
                return if (x == .integer and x.integer > 0) @intCast(x.integer) else error.BadDSparkConfig;
            }
        }.of;
        if (o.get("dspark_block_size")) |v| c.block = try int(v);
        if (o.get("dspark_noise_token_id")) |v| c.noise = try int(v);
        if (o.get("dspark_markov_rank")) |v| c.rank = try int(v);
        if (o.get("text_config")) |t| {
            if (t.object.get("sliding_window")) |v| c.window = try int(v);
            if (t.object.get("num_hidden_layers")) |v| c.layers = try int(v);
        }
        if (o.get("dspark_target_layer_ids")) |v| {
            if (v != .array or v.array.items.len != 5) return error.BadDSparkConfig;
            for (v.array.items, &c.taps) |x, *t| t.* = if (x == .integer and x.integer >= 0) @intCast(x.integer) else return error.BadDSparkConfig;
        }
        if (c.rank != 256 or c.block > state.window_rows or c.layers > 8) return error.BadDSparkConfig;
        return c;
    }
};

/// A draft layer: q | k | v of this rank's heads, k | v alone (the context's), o by columns, the dense MLP.
pub const Layer = struct { in_norm: u64, post_norm: u64, q_norm: u64, k_norm: u64, qkv: weights.Mx, kv: weights.Mx, o: weights.Mx, mlp: weights.MxMlp };

pub const Weights = struct {
    c: Config,
    fc: weights.Mx, // bf16 [hidden][this rank's share of the taps' inputs]: a partial each rank sums
    hidden_norm: u64,
    final_norm: u64,
    w1: u64, // bf16 [vocab][256], whole
    w2: u64, // bf16 [this rank's vocabulary][256]
    layers: [8]Layer = undefined,
};

/// A sequence's draft caches: each draft layer's keys and values by position (this rank's key heads).
pub const Caches = struct {
    buf: cuda.DeviceBuffer,
    layer_bytes: usize,
    capacity: usize,
    kv_heads: usize,

    pub fn init(d: *const cuda.Driver, layers: usize, kv_heads: usize, capacity: usize) !Caches {
        const per = capacity * kv_heads * 128 * 2 * 2;
        var buf = try cuda.DeviceBuffer.alloc(d, layers * per);
        errdefer buf.free();
        try buf.fill8(0, null);
        return .{ .buf = buf, .layer_bytes = per, .capacity = capacity, .kv_heads = kv_heads };
    }

    pub fn deinit(c: *Caches) void {
        c.buf.free();
    }

    pub fn keys(c: *const Caches, l: usize) u64 {
        return c.buf.ptr + l * c.layer_bytes;
    }

    pub fn values(c: *const Caches, l: usize) u64 {
        return c.keys(l) + c.layer_bytes / 2;
    }

    pub fn bytes(layers: usize, kv_heads: usize, capacity: usize) usize {
        return layers * capacity * kv_heads * 128 * 2 * 2;
    }
};

/// Loads the rank's share of dspark/ (beside the target's weights; uploads owned by `owner`).
pub fn load(gpa: std.mem.Allocator, io: std.Io, tools: weights.Tools, dir: []const u8, target: *weights.Weights) !Weights {
    const c = try Config.read(gpa, io, dir);
    const sub = try std.fs.path.join(gpa, &.{ dir, "dspark" });
    defer gpa.free(sub);
    var ck = try core.Checkpoint.openModel(gpa, io, sub);
    defer ck.close();
    var L: weights.Loader = .{ .gpa = gpa, .t = tools, .w = target, .ck = &ck };
    const tc = &target.config;
    const s = target.share;
    const h: usize = tc.hidden;
    const D: usize = tc.head_dim;
    var w: Weights = .{ .c = c, .fc = undefined, .hidden_norm = 0, .final_norm = 0, .w1 = 0, .w2 = 0 };
    // fc's inputs (5 x hidden) cut evenly by rank: each multiplies its share of the taps, the ranks sum
    const taps_in = 5 * h;
    if (taps_in % s.world != 0) return error.UnexpectedMiniMaxTensor;
    const cut = taps_in / s.world;
    const klo = s.rank * cut;
    {
        const fc = try L.expect(prefix ++ "fc.weight", .{}, .bf16, &.{ h, taps_in });
        const all: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, fc.bytes));
        const part = try gpa.alloc(u16, h * cut);
        defer gpa.free(part);
        for (0..h) |r| @memcpy(part[r * cut ..][0..cut], all[r * taps_in + klo ..][0..cut]);
        const tiles = try gpa.alloc(u16, cuda.mxfp8.padded(h) * cut);
        defer gpa.free(tiles);
        cuda.qmmf.packBf16(tiles, part, h, cut);
        w.fc = cuda.modelopt.bf16Weight(try L.put(std.mem.sliceAsBytes(tiles)), h, cut);
    }
    w.hidden_norm = try L.put((try L.expect(prefix ++ "hidden_norm.weight", .{}, .bf16, &.{h})).bytes);
    w.final_norm = try L.put((try L.expect(prefix ++ "final_norm.weight", .{}, .bf16, &.{h})).bytes);
    w.w1 = try L.put((try L.expect(prefix ++ "markov_head.markov_w1.weight", .{}, .bf16, &.{ tc.vocab, c.rank })).bytes);
    const w2 = try L.expect(prefix ++ "markov_head.markov_w2.weight", .{}, .bf16, &.{ tc.vocab, c.rank });
    w.w2 = try L.put(w2.bytes[s.vocab.lo * c.rank * 2 .. s.vocab.hi * c.rank * 2]);
    const q = s.queries(tc);
    for (0..c.layers) |i| {
        var names: [8][160]u8 = undefined;
        const nm = struct {
            fn of(buf: *[160]u8, li: usize, comptime rel: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, prefix ++ "layers.{d}.decoder_layer." ++ rel, .{li});
            }
        }.of;
        const qn = try nm(&names[0], i, "self_attn.q_proj");
        const kn = try nm(&names[1], i, "self_attn.k_proj");
        const vn = try nm(&names[2], i, "self_attn.v_proj");
        const on = try nm(&names[3], i, "self_attn.o_proj");
        const gn = try nm(&names[4], i, "mlp.gate_proj");
        const un = try nm(&names[5], i, "mlp.up_proj");
        const dn = try nm(&names[6], i, "mlp.down_proj");
        const kv_rows = [2]usize{ s.kv.lo * D, s.kv.hi * D };
        // the drafter's layers cut as the target's are: their widths must be the target's
        const qd = [2]usize{ tc.heads * D, h };
        const kvd = [2]usize{ tc.kv_heads * D, h };
        const di: usize = tc.dense_intermediate;
        w.layers[i] = .{
            .in_norm = try L.put((try L.expect(prefix ++ "layers.{d}.decoder_layer.input_layernorm.weight", .{i}, .bf16, &.{h})).bytes),
            .post_norm = try L.put((try L.expect(prefix ++ "layers.{d}.decoder_layer.post_attention_layernorm.weight", .{i}, .bf16, &.{h})).bytes),
            .q_norm = try L.put((try L.expect(prefix ++ "layers.{d}.decoder_layer.self_attn.q_norm.weight", .{i}, .bf16, &.{D})).bytes),
            .k_norm = try L.put((try L.expect(prefix ++ "layers.{d}.decoder_layer.self_attn.k_norm.weight", .{i}, .bf16, &.{D})).bytes),
            .qkv = try L.mx(&.{ .{ .name = qn, .rows = .{ q.lo * D, q.hi * D }, .dims = qd }, .{ .name = kn, .rows = kv_rows, .dims = kvd }, .{ .name = vn, .rows = kv_rows, .dims = kvd } }),
            .kv = try L.mx(&.{ .{ .name = kn, .rows = kv_rows, .dims = kvd }, .{ .name = vn, .rows = kv_rows, .dims = kvd } }),
            .o = try L.mx(&.{.{ .name = on, .cols = .{ q.lo * D, q.hi * D }, .dims = .{ h, tc.heads * D } }}),
            .mlp = .{
                .gu = try L.mx(&.{ .{ .name = gn, .rows = .{ s.dense.lo, s.dense.hi }, .dims = .{ di, h } }, .{ .name = un, .rows = .{ s.dense.lo, s.dense.hi }, .dims = .{ di, h } } }),
                .down = try L.mx(&.{.{ .name = dn, .cols = .{ s.dense.lo, s.dense.hi }, .dims = .{ h, di } }}),
            },
        };
    }
    try ck.skip(prefix ++ "confidence_head");
    try tools.stream.synchronize();
    return w;
}

/// The target layers' tap slots: each tapped layer's index among the five, else -1 (Forward.tap_at).
pub fn tapAt(c: *const Config, layers: usize, out: []i32) void {
    for (out[0..layers]) |*x| x.* = -1;
    for (c.taps, 0..) |t, i| if (t < layers) {
        out[t] = @intCast(i);
    };
}

/// A rank's (value, id) in the picks' gather, padded so two ranks' gathers can go by RoCE writes (16-byte rows).
const pick_stride = 16;

/// Runs the drafter's steps on one rank's forward machinery (`f.b` holds at least `rows` rows).
pub const Drafter = struct {
    w: Weights,
    picks: cuda.DeviceBuffer, // every rank's (value, id) of a Markov pick at pick_stride bytes, then the walk: i32 [17]
    scores: cuda.DeviceBuffer, // fp32 [this rank's vocabulary]: one row's Markov scores
    zero: cuda.DeviceBuffer, // fp32 [prompt rows][hidden] zeros: the residual a context row's norm adds to

    pub fn init(d: *const cuda.Driver, w: Weights, world: usize, vocab_share: usize, hidden: usize) !Drafter {
        var picks = try cuda.DeviceBuffer.alloc(d, world * pick_stride + 17 * 4);
        errdefer picks.free();
        var scores = try cuda.DeviceBuffer.alloc(d, vocab_share * 4);
        errdefer scores.free();
        const zero = try cuda.DeviceBuffer.alloc(d, state.prompt_rows * hidden * 4);
        return .{ .w = w, .picks = picks, .scores = scores, .zero = zero };
    }

    pub fn deinit(dr: *Drafter) void {
        dr.zero.free();
        dr.scores.free();
        dr.picks.free();
    }

    /// Context rows at p0.. from their taps: fc's rank share summed and normed, then each layer's keys and values.
    pub fn absorb(dr: *Drafter, f: *const fw.Forward, caches: *const Caches, taps: u64, rows: usize, p0: usize) !void {
        const c = &f.w.config;
        const s = f.w.share;
        const b = f.b;
        const h: usize = c.hidden;
        const cut = 5 * h / s.world;
        if (rows == 0 or rows > b.rows) return error.BadWindow;
        try f.mx(dr.w.fc, taps + s.rank * cut * 2, 5 * h, rows, f.slot(rows, 1), .f32);
        try f.gather(rows, 1);
        try f.d.check(f.d.api.cuMemsetD8Async(dr.zero.ptr, 0, rows * h * 4, f.stream.handle), "cuMemsetD8Async");
        try kern.addNorm(&f.m.k, f.stream, dr.zero.ptr, b.gathered, s.world, 1, 1.0, dr.w.hidden_norm, b.x, rows, h, c.eps);
        const hk = s.kv.len();
        for (dr.w.layers[0..dr.w.c.layers], 0..) |l, i| {
            try f.mx(l.kv, b.x, h, rows, b.qkv, .bf16);
            const kv = kern.stacked(b.qkv, 2 * hk * c.head_dim, 0, hk, c.head_dim);
            try kern.prep(&f.m.k, f.stream, .{ .xq = kv[0], .xk = kv[1], .xv = kv[2], .xi = kv[3], .qw = 0, .kw = l.k_norm, .iqw = 0, .ikw = 0, .q = 0, .iq = 0, .kc = caches.keys(i), .vc = caches.values(i), .ik = 0, .rows = rows, .pos = p0, .hq = 0, .hk = hk, .hi = 0, .rotary = c.rotary, .eps = c.eps, .theta = c.theta });
        }
    }

    /// A block at anchor position `p` (its context absorbed below p): `n` greedy drafts into `out`, every rank alike.
    pub fn block(dr: *Drafter, f: *const fw.Forward, caches: *const Caches, anchor: u32, p: usize, from: usize, n: usize, out: []u32) !void {
        const c = &f.w.config;
        const s = f.w.share;
        const b = f.b;
        const h: usize = c.hidden;
        const D: usize = c.head_dim;
        const T: usize = dr.w.c.block;
        if (n > T or p + T > caches.capacity) return error.BadDraft;
        var ids: [16]i32 = undefined;
        ids[0] = @intCast(anchor);
        for (1..T) |j| ids[j] = @intCast(dr.w.c.noise);
        try cuda.copy_source.enqueue(f.d, b.ids, std.mem.sliceAsBytes(ids[0..T]), f.stream);
        try kern.embed(&f.m.k, f.stream, b.ids, f.w.embed, b.resid, T, h);
        const hq = s.queries(c).len();
        const hk = s.kv.len();
        const lo = @max((p + 1) -| dr.w.c.window, from); // the anchor's window, within the context absorbed
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        const L = dr.w.layers[0..dr.w.c.layers];
        try kern.norm(&f.m.k, f.stream, b.resid, L[0].in_norm, b.x, T, h, c.eps);
        for (L, 0..) |l, i| {
            try f.mx(l.qkv, b.x, h, T, b.qkv, .bf16);
            const in = kern.stacked(b.qkv, (hq + 2 * hk) * D, hq, hk, D);
            try kern.prep(&f.m.k, f.stream, .{ .xq = in[0], .xk = in[1], .xv = in[2], .xi = in[3], .qw = l.q_norm, .kw = l.k_norm, .iqw = 0, .ikw = 0, .q = b.q, .iq = 0, .kc = caches.keys(i), .vc = caches.values(i), .ik = 0, .rows = T, .pos = p, .hq = hq, .hk = hk, .hi = 0, .rotary = c.rotary, .eps = c.eps, .theta = c.theta });
            try kern.attnWindow(&f.m.k, f.stream, b.q, caches.keys(i), caches.values(i), b.o, T, hq, hk, lo, p + T - 1, scale);
            try f.mx(l.o, b.o, hq * D, T, f.slot(T, 1), .f32);
            try f.gather(T, 1);
            try kern.addNorm(&f.m.k, f.stream, b.resid, b.gathered, s.world, 1, 1.0, l.post_norm, b.x, T, h, c.eps);
            try f.mlpMx(l.mlp, s.dense.len(), T, f.slot(T, 1), .f32);
            try f.gather(T, 1);
            const next = if (i + 1 < L.len) L[i + 1].in_norm else dr.w.final_norm;
            try kern.addNorm(&f.m.k, f.stream, b.resid, b.gathered, s.world, 1, 1.0, next, b.x, T, h, c.eps);
        }
        try f.x3(f.w.head, b.x, T, b.logits, .bf16);
        // the Markov walk: row j's pick biased by the pick before it (the anchor first), every rank's best gathered
        const nv = s.vocab.len();
        // the walk stays on the device: walk[0] the anchor, walk[j + 1] row j's pick; one download at the end
        const walk = dr.picks.ptr + s.world * pick_stride;
        const head = [1]i32{@intCast(anchor)};
        try cuda.copy_source.enqueue(f.d, walk, std.mem.sliceAsBytes(&head), f.stream);
        for (0..n) |j| {
            try kern.markov(&f.m.k, f.stream, b.logits + j * nv * 2, dr.w.w2, dr.w.w1, walk + j * 4, dr.scores.ptr, nv);
            const mine = dr.picks.ptr + s.rank * pick_stride;
            try kern.top1f(&f.m.k, f.stream, dr.scores.ptr, nv, s.vocab.lo, mine, mine + 4);
            switch (f.exchange) {
                .local => {},
                .nccl => |x| if (x.group) |g| try g.allGather(f.stream, mine, dr.picks.ptr, pick_stride) else try x.lib.check(x.lib.api.ncclAllGather(mine, dr.picks.ptr, pick_stride, .u8, x.comm, f.stream.handle), "ncclAllGather"),
            }
            try kern.pickMerge(&f.m.k, f.stream, dr.picks.ptr, s.world, pick_stride, walk, j);
        }
        var got: [17]i32 = undefined;
        try f.stream.synchronize();
        switch (f.exchange) {
            .local => {},
            .nccl => |x| if (x.group) |g| try g.check(),
        }
        try cuda.DeviceBuffer.download(.{ .d = f.d, .ptr = walk, .len = (n + 1) * 4 }, 0, std.mem.sliceAsBytes(got[0 .. n + 1]));
        for (out[0..n], got[1 .. n + 1]) |*o, g| o.* = @intCast(g);
    }
};

test "tap slots: the five tapped layers in order, -1 elsewhere" {
    const c = Config{};
    var at: [60]i32 = undefined;
    tapAt(&c, 60, &at);
    try std.testing.expectEqual(@as(i32, 0), at[3]);
    try std.testing.expectEqual(@as(i32, 4), at[59]);
    try std.testing.expectEqual(@as(i32, -1), at[4]);
    try std.testing.expectEqual(@as(usize, 5 * 32896 * 2 * 128 * 2 * 2), Caches.bytes(5, 2, 32896));
}
