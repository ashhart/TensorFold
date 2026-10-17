//! The 27B's Living Weights trainer: packed passes through Forward.chunks and the loss back to the lowest nudged layer.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const living = @import("living");
const c = @import("cuda_learn_shape.zig");
const st = struct {
    const max_streams = 8;
};
const hooks = @import("cuda_learn_hooks.zig");
const to = @import("cuda_train_ops.zig");
const engine = @import("cuda_engine.zig");

const load = @import("cuda_train_load.zig");

const train_more = @import("cuda_train_more.zig");

const gt = cuda.gdn_train;
const train_mode = living.train_mode;
const lesson_route = living.lesson_route;
const Count = living.pocket_clock.Count;

pub const Mode = train_mode.Mode;
pub const Result = train_mode.Result;
pub const Seq = train_mode.Seq;
pub const Span = train_mode.Span;
pub const Answer = train_mode.Answer;
pub const Turn = lesson_route.Turn;

/// Rows a sequence learns from at most (a question, its answer and the template around them).
pub const max_rows = 512;
/// Rows a pass holds at most, its sequences side by side.
pub const pack_rows = 1024;
/// Sequences a pass holds at most: the batched prompt pass's streams.
pub const max_seqs = st.max_streams;
/// The lowest layer a pass keeps what a backward reads: a pocket's neurons sit at layers / 5 and up.
pub fn keepLow(layers: usize) usize {
    return layers / 5;
}

const prefix = "language_model.";

/// Device bytes the trainer takes at most beside the engine (its MLX copies, keeps, logits and scratch).
pub fn memory(g: c.Geometry) usize {
    const R = pack_rows;
    const kept = g.layers - keepLow(g.layers);
    const proj = g.convDim() + g.vInner() + 2 * g.linear_v_heads;
    const per_row = 2 * g.hidden * 2 + 2 * g.intermediate * 2 + @max(proj, 2 * g.qInner() + 2 * g.kvInner() + g.qInner()) * 2;
    var n: usize = R * kept * per_row + R * g.layers * g.hidden * 2; // keeps, every layer's MLP input
    n += R * c.vocab * 2 + 3 * R * g.hidden * 4 + (R + 512) * g.hidden * 4 + 3 * R * g.intermediate * 4;
    n += gt.Scratch.bytes(max_rows, g.linear_k_heads, g.linear_v_heads, g.hidden) + gt.Wide.bytes(R, g.linear_k_heads, g.linear_v_heads, g.hidden);
    n += 2 * g.query_heads * max_rows * max_rows * 4 + max_rows * (g.qInner() * 16 + g.kvInner() * 16) + R * (g.qInner() * 12 + g.kvInner() * 8);
    const q = struct {
        fn f(rows: usize, cols: usize) usize {
            return rows * cols / 2 + rows * cols / 16;
        }
    }.f;
    var w: usize = q(c.vocab, g.hidden);
    for (keepLow(g.layers)..g.layers) |l| {
        w += 3 * q(g.intermediate, g.hidden);
        w += if (c.linear(l)) q(proj, g.hidden) + q(g.hidden, g.vInner()) else q(2 * g.qInner(), g.hidden) + 2 * q(g.kvInner(), g.hidden) + q(g.hidden, g.qInner());
    }
    n += w + max_seqs * ((g.deltaBytes() + g.convBytes()) * g.layers + g.layers * 2 * g.kvInner() * (max_rows + 1) * 2);
    return n + (512 << 20);
}

/// Host memory the GPU maps: kernels write it at `dev`, the host reads it through `slice`.
const Mapped = struct {
    host: cuda.HostBuffer,
    dev: u64,

    fn init(d: *const cuda.Driver, bytes: usize) !Mapped {
        var h = try cuda.HostBuffer.allocMapped(d, @max(bytes, 16));
        errdefer h.free();
        @memset(h.bytes, 0);
        return .{ .host = h, .dev = try h.device() };
    }

    pub fn deinit(m: *Mapped) void {
        m.host.free();
    }

    fn slice(m: Mapped, comptime T: type, n: usize) []T {
        return m.host.slice(T)[0..n];
    }
};

const LayerW = struct { mlp: to.Mlp = undefined, gdn: ?gt.Layer = null, attn: ?to.Attn = null };

pub const Trainer = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    g: c.Geometry,
    k: to.Kernels,
    gk: gt.Kernels,
    states: [max_seqs]engine.Cache = undefined,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    w: []LayerW = &.{},
    head: to.Q4 = undefined,
    full: []hooks.LayerKeep = &.{},
    light: []hooks.LayerKeep = &.{},
    keep: hooks.Keep = undefined,
    logits: u64 = 0, // bf16 [pack_rows, vocab]: the logits, then their gradient
    last: u64 = 0,
    last_xs: u64 = 0,
    final_h: u64 = 0,
    // the backward's scratch, fp32
    grad: u64 = 0, // [R, hidden] dL/d the residual stream where the backward stands
    dy: u64 = 0, // [R, hidden] a block's input gradient before its norm
    dlast: u64 = 0, // [R, hidden] the head's input gradient
    parts: u64 = 0,
    ms: to.MlpScratch = undefined,
    as: to.AttnScratch = undefined,
    gs: gt.Scratch = undefined,
    gw: gt.Wide = undefined, // the DeltaNet backward's projections over every row of a pass
    host: Mapped = undefined, // targets u32, weights f32, stats [2]f32, scales f32: [pack_rows] each
    ys: Mapped = undefined, // bf16 [layers, pack_rows, hidden]: each layer's MLP input as the last pass read it
    delta: Mapped = undefined, // f32 [max_seqs, 3, layers, hidden]: each sequence's nudges
    push: Mapped = undefined, // f32 [max_seqs, 3, layers, hidden]: the loss's gradient at each nudged row
    tails: Mapped = undefined, // i32 [pack_rows] the backward's tail rows, then u8 [pack_rows] their mask
    tail: ?to.Tail = null, // the running backward's tail (null: every row), which the DeltaNet's products read too
    resid: []u16 = &.{},
    // the learner's view
    spans: [max_seqs]Span = undefined,
    answers: [max_seqs]Answer = undefined,
    seqs: [max_seqs]Seq = undefined,
    nseq: usize = 0,
    rows: usize = 0,
    pack: usize = pack_rows,
    count: Count = .{},
    prefixes: struct {
        pub const most = 0; // no prompt prefix is run once and kept: every pass runs its sequences whole
    } = .{},
    nudge: train_mode.Nudge = .{},
    from: [living.family.max_layers]u32 = @splat(std.math.maxInt(u32)),
    keys_top: ?usize = null, // a keys pass's highest layer whose MLP inputs are read (null: every layer)
    pocket_back: bool = true, // the backward runs through the pocket's inputs (false: its activations held as they are)
    hook_at: usize = 0, // this owned forward's row in the packed pass

    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine) !*Trainer {
        return initWithWeights(gpa, io, e, load.weights);
    }

    fn initWithWeights(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, comptime readWeights: anytype) !*Trainer {
        const geometry = try c.from(e.g);
        const t = try gpa.create(Trainer);
        t.* = .{ .gpa = gpa, .e = e, .g = geometry, .k = undefined, .gk = undefined };
        var partial: @import("cuda_train_init.zig").Partial(Trainer) = .{ .t = t };
        errdefer partial.deinit();
        const d = e.d;
        t.k = try to.Kernels.load(d);
        partial.k = true;
        t.gk = try gt.Kernels.load(d);
        partial.gk = true;
        while (partial.states < max_seqs) : (partial.states += 1)
            t.states[partial.states] = try engine.Cache.init(gpa, e, max_rows + 1);
        t.w = try gpa.alloc(LayerW, t.g.layers);
        @memset(t.w, .{});
        t.resid = try gpa.alloc(u16, t.g.hidden);
        const t0 = std.Io.Clock.awake.now(io);
        try readWeights(t, io);
        const t1 = std.Io.Clock.awake.now(io);
        inline for (.{ "host", "ys", "delta", "push", "tails" }, .{
            pack_rows * 20,                             t.g.layers * pack_rows * t.g.hidden * 2,
            max_seqs * 3 * t.g.layers * t.g.hidden * 4, max_seqs * 3 * t.g.layers * t.g.hidden * 4,
            pack_rows * 5,
        }) |field, bytes| {
            @field(t, field) = try Mapped.init(d, bytes);
            partial.mapped += 1;
        }
        try t.carve();
        try e.stream.synchronize();
        var bytes: usize = 0;
        for (t.owned.items) |buf| bytes += buf.len;
        const secs = @as(f64, @floatFromInt(t1.toNanoseconds() - t0.toNanoseconds())) / 1e9;
        std.log.info("slide: the trainer's MLX copies read in {d:.1} s; its device buffers {d:.2} GiB", .{ secs, @as(f64, @floatFromInt(bytes)) / (1 << 30) });
        return t;
    }

    pub fn drain(t: *Trainer) void {
        @import("qwen_vision").completion.drain(t.e.d, t.e.stream);
    }

    pub fn deinit(t: *Trainer) void {
        const gpa = t.gpa;
        t.drain();

        inline for (.{ "host", "ys", "delta", "push", "tails" }) |f| @field(t, f).deinit();
        for (t.owned.items) |*b| b.free();
        t.owned.deinit(gpa);
        for (&t.states) |*s| s.deinit(gpa);
        gpa.free(t.full);
        gpa.free(t.light);
        gpa.free(t.w);
        gpa.free(t.resid);
        t.gk.deinit();
        t.k.deinit();
        gpa.destroy(t);
    }

    pub fn device(t: *Trainer, bytes: usize) !u64 {
        try t.owned.ensureUnusedCapacity(t.gpa, 1);
        var b = try cuda.DeviceBuffer.alloc(t.e.d, @max(bytes, 256));
        errdefer {
            t.drain();
            b.free();
        }
        try b.fill8(0, t.e.stream.handle);
        t.owned.appendAssumeCapacity(b);
        return b.ptr;
    }

    pub fn devBytes(t: *Trainer, bytes: []const u8) !u64 {
        try t.owned.ensureUnusedCapacity(t.gpa, 1);
        var b = try cuda.DeviceBuffer.alloc(t.e.d, bytes.len);
        errdefer {
            t.drain();
            b.free();
        }
        try b.upload(0, bytes);
        t.owned.appendAssumeCapacity(b);
        return b.ptr;
    }

    pub fn ops(t: *const Trainer) to.Ops {
        return .{ .k = &t.k, .s = t.e.stream };
    }

    fn gops(t: *const Trainer) gt.Ops {
        return .{ .k = &t.gk, .s = t.e.stream, .q4 = if (t.k.wide) .{ .ctx = t, .f = gdnProduct } else null };
    }

    /// The DeltaNet backward's two projections through the 27B's own 4-bit products.
    fn gdnProduct(ctx: *const anyopaque, x: u64, m: gt.Q4, y: u64, rows: usize, add: bool) anyerror!void {
        const t: *const Trainer = @ptrCast(@alignCast(ctx));
        const o = t.ops();
        try o.productTail(x, false, m.n, .{ .w = m.w, .s = m.s, .b = m.b, .n = m.n, .k = m.k, .group = m.group }, y, m.k, rows, add, 0, t.tail);
        // the chain reads every row: the ones the tail left out as zeros
        if (t.tail) |tl| if (!add) try o.zeroRows(y, m.k, rows, tl);
    }

    /// Pocket neurons' input weights fitted on the device: each job's Adam steps.
    pub fn fitDevice(t: *Trainer, jobs: []const living.pocket_fit.DeviceFit) !void {
        return train_more.fitDevice(t, jobs);
    }

    /// The rows the backward still needs, each nudged sequence's from its first nudged row on (null: every row).
    fn tailRows(t: *Trainer, rows: usize) ?to.Tail {
        if (!t.k.wide) return null;
        if (@import("sys").c.getenv("TF_Q27_TAIL")) |v| if (std.mem.eql(u8, std.mem.span(v), "off")) return null;
        const list = t.tails.slice(i32, pack_rows);
        const mask = t.tails.host.bytes[pack_rows * 4 ..][0..pack_rows];
        @memset(mask[0..rows], 0);
        var n: usize = 0;
        for (t.seqs[0..t.nseq], t.spans[0..t.nseq]) |sq, sp| {
            var r0: usize = sp.rows;
            for (sq.nudge.layers, sq.nudge.rows) |m, r| if (m != 0) {
                r0 = @min(r0, r);
            };
            for (r0..sp.rows) |r| {
                list[n] = @intCast(sp.at + r);
                mask[sp.at + r] = 1;
                n += 1;
            }
        }
        if (n == rows or n == 0) return null;
        return .{ .list = t.tails.dev, .mask = t.tails.dev + pack_rows * 4, .n = n };
    }

    /// Sequence s's first row the backward still needs (0: all).
    fn tailFrom(t: *const Trainer, s: usize) usize {
        if (t.tail == null) return 0;
        var r0: usize = t.spans[s].rows;
        for (t.seqs[s].nudge.layers, t.seqs[s].nudge.rows) |m, r| if (m != 0) {
            r0 = @min(r0, r);
        };
        return r0;
    }

    /// The projections a backward reads (train_load.zig).
    fn loadWeights(t: *Trainer, io: std.Io) !void {
        return load.weights(t, io);
    }

    /// The keep lists and the backward's scratch.
    fn carve(t: *Trainer) !void {
        return train_more.carve(t);
    }

    // ---- the learner's hooks --------------------------------------------------------------------------------------

    fn targets(t: *const Trainer) []u32 {
        return t.host.slice(u32, pack_rows);
    }

    fn weights(t: *const Trainer) []f32 {
        return t.host.slice(f32, 2 * pack_rows)[pack_rows..];
    }

    fn stats(t: *const Trainer) [][2]f32 {
        return t.host.slice([2]f32, 2 * pack_rows)[pack_rows..];
    }

    fn scales(t: *const Trainer) []f32 {
        return t.host.slice(f32, 5 * pack_rows)[4 * pack_rows ..];
    }

    /// Layer l's MLP input (its post norm's output, what the pocket reads) at the pass's `row`, bf16 [hidden].
    pub fn normed(t: *const Trainer, l: usize, row: usize) []const u16 {
        const D = t.g.hidden;
        return t.ys.slice(u16, t.g.layers * pack_rows * D)[(l * pack_rows + row) * D ..][0..D];
    }

    /// What layer l's pocket reads is its post norm's output, its weight times the residual over its RMS.
    pub fn readScale(t: *const Trainer, l: usize, out: []f32) !void {
        const raw = try t.gpa.alloc(u16, out.len);
        defer t.gpa.free(raw);
        const d = t.e.d;
        try t.e.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(raw.ptr, t.e.layers[l].post_norm, out.len * 2), "cuMemcpyDtoH");
        for (out, raw) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
    }

    /// The residual stream after layer l - 1 at `row` as the last target pass kept it (zeros where it kept none).
    pub fn residual(t: *Trainer, l: usize, row: usize) []const u16 {
        @memset(t.resid, 0);
        if (l == 0 or l > t.g.layers) return t.resid;
        const at = if (l < t.g.layers) t.keep.layers[l].h_in else t.final_h;
        if (at == 0) return t.resid;
        const d = t.e.d;
        t.e.stream.synchronize() catch return t.resid;
        d.check(d.api.cuMemcpyDtoH_v2(t.resid.ptr, at + row * t.g.hidden * 2, t.g.hidden * 2), "cuMemcpyDtoH") catch return t.resid;
        return t.resid;
    }

    fn vec(t: *const Trainer, s: usize, i: usize, l: usize) usize {
        return ((s * 3 + i) * t.g.layers + l) * t.g.hidden;
    }

    /// The loss's gradient sequence s's target pass read at its key i's row after layer l (dL/d the MLP output).
    pub fn pushAt(t: *const Trainer, s: usize, i: usize, l: usize) []const f32 {
        return t.push.slice(f32, max_seqs * 3 * t.g.layers * t.g.hidden)[t.vec(s, i, l)..][0..t.g.hidden];
    }

    /// Sequence s's nudge for key i after layer l, which its target pass adds to that layer's MLP output.
    pub fn deltaAt(t: *const Trainer, s: usize, i: usize, l: usize) []f32 {
        return t.delta.slice(f32, max_seqs * 3 * t.g.layers * t.g.hidden)[t.vec(s, i, l)..][0..t.g.hidden];
    }

    pub fn packs(_: *const Trainer) bool {
        return true;
    }

    pub fn prefixSlots(_: *Trainer, n: usize) !void {
        if (n > 0) return error.TooManyPrefixes;
    }

    /// The MLP inputs of prompt `ids` at `rows` (ascending) at each of `layers` into `out` [rows, layers.
    pub fn tapRows(t: *Trainer, ids: []const u32, context: usize, rows: []const u32, layers: []const usize, out: []u16) !void {
        return train_more.tapRows(t, ids, context, rows, layers, out);
    }

    /// Whether each of `seqs`' answers is its prompt's greedy reply under the model as it is now.
    pub fn greedy(t: *Trainer, seqs: []const Seq, same: []bool) !void {
        return train_more.greedy(t, seqs, same);
    }

    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode, turn: lesson_route.Turn) !Result {
        var out: [1]Result = undefined;
        try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = t.nudge }}, mode, turn, &out);
        return out[0];
    }

    // ---- the pass ---------------------------------------------------------------------------------------------------

    /// `seqs` side by side in one pass: keys (MLP inputs), loss (each answer's loss, recall) or target (and push).
    pub fn pass(t: *Trainer, seqs: []const Seq, mode: Mode, turn: lesson_route.Turn, out: []Result) !void {
        _ = turn;
        if (seqs.len == 0 or seqs.len > max_seqs or out.len < seqs.len) return error.BadPass;
        switch (mode) {
            .keys, .loss, .target => {},
            .prime => {
                for (out[0..seqs.len]) |*r| r.* = .{ .loss = 0, .recalled = false };
                return;
            },
            else => return error.RoutedUnsupported,
        }
        const e = t.e;
        const g = t.g;
        var rows: usize = 0;
        for (seqs, t.spans[0..seqs.len]) |sq, *sp| {
            if (sq.ids.len < 2 or sq.start < 1 or sq.start >= sq.ids.len or sq.ids.len - 1 > max_rows) return error.BadExample;
            if (sq.skip != 0) return error.BadPrefix;
            sp.* = .{ .at = rows, .rows = sq.ids.len - 1 };
            rows += sp.rows;
        }
        if (rows > pack_rows) return error.PassTooLong;
        const spans = t.spans[0..seqs.len];
        @memcpy(t.seqs[0..seqs.len], seqs);
        t.nseq = seqs.len;
        t.rows = rows;
        t.count = .{ .passes = t.count.passes + 1, .rows = t.count.rows + rows };
        try e.stream.synchronize(); // the host rows below are memory the last pass's kernels may still read
        @memset(t.scales()[0..rows], 1);
        @memset(t.targets()[0..rows], 0);
        @memset(t.weights()[0..rows], 0);
        _ = train_mode.answers(seqs, spans, t.answers[0..seqs.len]);

        for (seqs, spans, t.answers[0..seqs.len], 0..) |sq, sp, a, k| {
            @memset(t.scales()[sp.at..][0..sp.rows], sq.scale);
            const share = 1 / @as(f32, @floatFromInt(a.rows));
            for (0..a.rows) |r| {
                t.targets()[a.first + r] = sq.ids[sq.start + r];
                t.weights()[a.first + r] = share;
            }
            try t.states[k].reset(e);
        }
        var low: usize = g.layers;
        if (mode == .target) {
            var mask: u64 = 0;
            for (seqs) |sq| mask |= sq.nudge.layers[0] | sq.nudge.layers[1] | sq.nudge.layers[2];
            if (mask != 0) low = @ctz(mask);
            if (low < keepLow(g.layers)) return error.NudgeBelowKept;
        }
        // the forward: from the lowest nudge up everything a backward reads (with none, every layer a residual reads)
        const keep_list = try t.gpa.dupe(hooks.LayerKeep, if (mode == .target) t.full else t.light);
        defer t.gpa.free(keep_list);
        if (mode == .target) for (0..if (low == g.layers) keepLow(g.layers) else low) |l| {
            keep_list[l] = t.light[l];
        };
        var keep = t.keep;
        keep.layers = keep_list;
        if (mode == .keys) keep.logits = 0;
        const prior_keep = e.learning;
        const prior_mlp = e.mlp;
        defer {
            e.learning = prior_keep;
            e.mlp = prior_mlp;
        }
        e.mlp = .{ .ctx = t, .f = mlpAdd };
        for (seqs, spans, 0..) |sq, sp, si| for (0..sp.rows) |row| {
            t.hook_at = sp.at + row;
            keep.at = t.hook_at;
            e.learning = &keep;
            try e.forward(&.{.{ .cache = &t.states[si], .rows = 1 }}, sq.ids[row..][0..1], .logits);
        };
        if (mode == .keys) {
            try e.stream.synchronize();
            for (out[0..seqs.len]) |*r| r.* = .{ .loss = 0, .recalled = false };
            return;
        }
        const o = t.ops();
        const hd = t.host.dev;
        try o.softmax(t.logits, hd, hd + pack_rows * 4, hd + pack_rows * 8, rows, c.vocab);
        if (mode == .target and low < g.layers) try t.backward(keep_list, rows, low);
        try e.stream.synchronize();
        for (out[0..seqs.len], t.answers[0..seqs.len]) |*r, a| {
            r.* = .{ .loss = 0, .recalled = true };
            for (t.stats()[a.first..][0..a.rows], t.weights()[a.first..][0..a.rows]) |s, w| {
                r.loss += s[0] * w;
                r.recalled = r.recalled and s[1] > 0.5;
            }
        }
    }

    /// The pass's MLP hook: each layer's pocket (the open fact's neurons scaled per sequence), then the nudges.
    fn mlpAdd(ctx: *anyopaque, layer: usize, y: u64, out: u64, rows: usize, _: u64) anyerror!void {
        const t: *Trainer = @ptrCast(@alignCast(ctx));
        const o = t.ops();
        const D = t.g.hidden;
        if (t.e.pockets.of(layer)) |pk0| {
            var pk = pk0;
            pk.scale = t.host.dev + 4 * pack_rows * 4 + t.hook_at * 4;
            pk.from = t.from[layer];
            try o.pocketRows(pk, y, out, rows, D);
        }
        for (t.seqs[0..t.nseq], t.spans[0..t.nseq], 0..) |sq, sp, si| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| {
            if (mask >> @intCast(layer) & 1 == 0 or sp.row(row) != t.hook_at) continue;
            try o.addBf16(out, t.delta.dev + t.vec(si, i, layer) * 4, D);
        };
    }

    fn copyPush(t: *Trainer, l: usize) !void {
        const d = t.e.d;
        const D = t.g.hidden;
        for (t.seqs[0..t.nseq], t.spans[0..t.nseq], 0..) |sq, sp, si| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| {
            if (mask >> @intCast(l) & 1 == 0) continue;
            try d.check(d.api.cuMemcpyDtoDAsync_v2(t.push.dev + t.vec(si, i, l) * 4, t.grad + sp.row(row) * D * 4, D * 4, t.e.stream.handle), "cuMemcpyDtoDAsync");
        };
    }

    /// The loss back from the head to layer `low`'s MLP output, its nudged rows' gradients into `push` on the way.
    fn backward(t: *Trainer, keep: []const hooks.LayerKeep, rows: usize, low: usize) !void {
        const o = t.ops();
        const e = t.e;
        const g = t.g;
        const D = g.hidden;
        t.tail = t.tailRows(rows);
        defer t.tail = null;
        const tail = t.tail;
        // the head (its bf16 rows' gradient in place), then the final norm onto the residual stream
        try o.productTail(t.logits, true, c.vocab, t.head, t.dlast, D, rows, false, t.parts, tail);
        try o.zero(t.grad, rows * D);
        try o.rmsBack(t.final_h, e.final_norm, t.dlast, t.grad, rows, D, c.eps);
        var l = g.layers;
        while (l > low) {
            l -= 1;
            const kp = keep[l];
            const lw = t.w[l];
            const layer = e.layers[l];
            try t.copyPush(l);
            if (l == low) return;
            // the MLP and its pocket, then the post norm: grad becomes dL/d the residual into the post norm
            var pk = if (t.pocket_back) e.pockets.of(l) else null;
            if (pk) |*x| {
                x.scale = t.host.dev + 4 * pack_rows * 4;
                x.from = t.from[l];
            }
            try o.mlpBackTail(lw.mlp, .{ .y = kp.y, .gate = kp.gate, .up = kp.up }, pk, t.grad, t.dy, t.ms, rows, t.parts, tail);
            try o.rmsBack(kp.h_mid, layer.post_norm, t.dy, t.grad, rows, D, c.eps);
            if (tail) |tl| try o.zeroRows(t.grad, D, rows, tl); // the mixers read every row: the rest as zeros
            // the mixer (a DeltaNet's projections once for every sequence.
            try o.zero(t.dy, rows * D);
            if (lw.gdn) |gl| {
                const pw = g.convDim() + g.vInner() + 2 * g.linear_v_heads;
                var runs: [max_seqs]gt.Run = undefined;
                for (t.spans[0..t.nseq], runs[0..t.nseq]) |sp, *r| r.* = .{ .p = kp.proj + sp.at * pw * 2, .rows = sp.rows, .at = sp.at, .from = 0 };
                try t.gops().backwardAll(gl, t.gs, t.gw, runs[0..t.nseq], t.grad, t.dy, rows, false);
            } else {
                // the projections once for every sequence's rows (the tail's), the middle sequence by sequence
                const at = lw.attn.?;
                const qw = g.qInner();
                const kw = g.kvInner();
                try o.productTail(t.grad, false, D, at.o, t.as.dg, qw, rows, false, 0, tail);
                if (tail) |tl| try o.zeroRows(t.as.dg, qw, rows, tl);
                for (t.spans[0..t.nseq], 0..) |sp, si| {
                    if (t.tailFrom(si) >= sp.rows) continue; // nothing of it reaches a nudge
                    const kept: to.AttnKept = .{ .pq = kp.pq + sp.at * 2 * qw * 2, .pk = kp.pk + sp.at * kw * 2, .pv = kp.pv + sp.at * kw * 2, .gated = kp.gated + sp.at * qw * 2, .pos0 = 0 };
                    var sc = t.as;
                    sc.dg = t.as.dg + sp.at * qw * 4;
                    sc.dpq = t.as.dpq + sp.at * 2 * qw * 4;
                    sc.dpk = t.as.dpk + sp.at * kw * 4;
                    sc.dv = t.as.dv + sp.at * kw * 4;
                    try o.attnPrep(at, kept, sc.qkv, sp.rows);
                    try o.attnCore(at, kept, sc, sp.rows);
                }
                try o.productTail(t.as.dpq, false, 2 * qw, at.q, t.dy, D, rows, false, 0, tail);
                try o.productTail(t.as.dpk, false, kw, at.k, t.dy, D, rows, true, 0, tail);
                try o.productTail(t.as.dv, false, kw, at.v, t.dy, D, rows, true, 0, tail);
            }
            try o.rmsBack(kp.h_in, layer.input_norm, t.dy, t.grad, rows, D, c.eps);
        }
    }
};

test "dense constructor: registration OOM never frees an accepted memset destination" {
    try @import("cuda_train_init_test.zig").registrationFailure(Trainer);
}
test "dense constructor: allocation failure after one and seven registrations frees buffers before their list" {
    if (!cuda.kernels.available) return error.SkipZigTest;
    try @import("cuda_train_init_test.zig").listFailure(Trainer.initWithWeights);
}
test "dense constructor: pending work drains or exits before any owner is released" {
    if (!cuda.kernels.available) return error.SkipZigTest;
    try @import("cuda_train_init_test.zig").pendingFailure(Trainer.initWithWeights);
}
