//! Learning in the weights: sequences side by side forward through every layer, their losses, back to each change.
const std = @import("std");
const train_mode = @import("living").train_mode;
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const pl = @import("prefill_launch.zig");
const ops = @import("train_ops.zig");
const back = @import("train_back.zig");
const job = @import("train_job.zig");
const adapters = @import("adapters.zig");
const backend = @import("backend.zig");
const lesson_route = @import("living").lesson_route;
const prefix = @import("train_prefix.zig");
const Count = @import("living").pocket_clock.Count;

const Metal = backend.Metal;
const At = pl.At;
const Buffer = mtl.Buffer;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Rows a sequence learns from at most: a question and its answer (attention's backward reads at most 256).
pub const max_rows = 256;

/// Rows a pass holds at most, its sequences side by side.
pub const pack_rows = 1024;

/// Sequences a pass holds at most.
pub const max_seqs = 32;

pub const Mode = train_mode.Mode;
pub const Result = train_mode.Result;
pub const Nudge = train_mode.Nudge;
pub const Seq = train_mode.Seq;
pub const Span = train_mode.Span;
pub const Answer = train_mode.Answer;
const reads = train_mode.reads;

pub const Trainer = struct {
    gpa: std.mem.Allocator,
    b: *Metal,
    sites: adapters.Sites,
    bufs: back.Bufs,
    tmp: ops.Tmp,
    saved: Buffer, // bf16 [layers + 1, pack_rows, D]: the residual entering each layer, then the last one
    acts: Buffer, // what each layer's backward reads of its forward's scratch, from slots[layer]
    slots: [cfg.max_layers]usize,
    logits: Buffer, // bf16 [pack_rows, V]: the answer rows' logits, then their gradient
    stats: Buffer, // f32 [pack_rows, 2]: each answer row's loss and target probability
    targets: Buffer, // u32 [pack_rows]: each answer row's next token
    weights: Buffer, // f32 [pack_rows]: each answer row's share of its answer, 1 / its tokens
    ans: Buffer, // bf16 [pack_rows, D]: the final-normed answer rows, the head's input
    scale: Buffer, // f32 [pack_rows]: each row's scale on the open fact's pocket neurons
    dhn: Buffer, // bf16 [pack_rows, D]
    head_t: Buffer, // bf16 [D, V]: the LM head transposed, for the loss's backward product
    head: [3]At,
    cache: st.Cache,
    fresh: u32,
    sketched: usize = 0, // rows sketched since the sketches were cleared, each row's place in them
    proj: Buffer, // f32 [layers, max_rows, candidates]: each row along each layer's candidate directions
    nudge: Nudge = .{}, // where a one-sequence target step adds `delta`
    delta: Buffer, // f32 [max_seqs, 3, layers, D]: each sequence's nudges after each layer it names, a row each
    push: Buffer, // f32 [max_seqs, 3, layers, D]: the loss's gradient at each nudged row after each of them
    inputs: Buffer, // bf16 [layers, pack_rows, D]: each MoE layer's normed input, as the last keys or target pass read it
    from: [cfg.max_layers]u32 = @splat(std.math.maxInt(u32)), // each layer's open pocket neurons, from here on
    spans: [max_seqs]Span = undefined, // where the last pass's sequences sat
    answers: [max_seqs]Answer = undefined, // and their answer rows, packed for the head
    answered: usize = 0, // the head's rows
    seqs: usize = 0,
    pack: usize = pack_rows, // rows a pass may hold here (the prefill scratch's at most)
    prefixes: prefix.Store, // sequences' prompt prefixes, run once and their states kept
    profile: ?*fwd.Profiler = null, // times each dispatch alone (a check's breakdown), else steps run as one command buffer
    count: Count = .{}, // passes run and the rows they held
    residency: ?mtl.ResidencySet.Follow = null,

    /// Every buffer a step needs and a change at every layer, attached to the forward (zero, so nothing moves yet).
    pub fn init(gpa: std.mem.Allocator, b: *Metal) !*Trainer {
        const m = b.m;
        const c = m.config;
        if (b.wide == null or c.mamba_head_dim != 64 or c.state != 128) return error.UnsupportedShape;
        const t = try gpa.create(Trainer);
        errdefer gpa.destroy(t);
        t.gpa = gpa;
        t.b = b;
        t.residency = null;
        t.prefixes = prefix.Store.init(m.device, c, max_rows);
        t.profile = null;
        t.sketched = 0;
        t.count = .{};
        t.nudge = .{};
        t.from = @splat(std.math.maxInt(u32));
        t.seqs = 0;
        t.pack = @min(pack_rows, b.wide.?.rows, b.o.capacity);
        // the saved lessons' blocks, which this trainer learns on beside, else room for the first
        const saved = b.living != null;
        t.sites = if (b.living) |s| s else try adapters.Sites.init(gpa, m.device, c, @max(b.o.chunk, 64), 0);
        b.living = null;
        errdefer if (saved) {
            b.living = t.sites;
        } else t.sites.deinit();
        t.bufs = try back.Bufs.init(m.device, c, pack_rows, max_rows);
        errdefer t.bufs.deinit();
        const widest = @max(c.projDim(), c.heads * c.head_dim, c.shared_width, c.hidden);
        t.tmp = .{ .xb = try m.device.buffer(pack_rows * widest * 2, opts), .wt = try m.device.buffer(widest * c.hidden * 2, opts), .acc = try m.device.buffer(pack_rows * widest * 4, opts), .tiles = try m.device.buffer(ops.tileCap(pack_rows * c.top_k, c.experts) * 8, opts) };
        errdefer inline for (.{ "xb", "wt", "acc", "tiles" }) |f| @field(t.tmp, f).deinit();
        const d = c.hidden;
        const R = pack_rows;
        const vecs = max_seqs * 3 * c.layers * d * 4;
        const sizes = [_]usize{ (c.layers + 1) * R * d * 2, R * c.vocab * 2, R * 8, R * 4, R * 4, R * d * 2, R * 4, R * d * 2, d * c.vocab * 2, c.layers * max_rows * adapters.candidates * 4, vecs, vecs, c.layers * R * d * 2 };
        const fields = [_]*Buffer{ &t.saved, &t.logits, &t.stats, &t.targets, &t.weights, &t.ans, &t.scale, &t.dhn, &t.head_t, &t.proj, &t.delta, &t.push, &t.inputs };
        var made: usize = 0;
        errdefer for (fields[0..made]) |buf| buf.deinit();
        for (fields, sizes) |buf, n| {
            buf.* = try m.device.buffer(n, opts);
            made += 1;
        }
        inline for (.{ "weight", "scales", "biases" }, 0..) |part, i| {
            const tensor = try m.checkpoint.get("lm_head." ++ part);
            t.head[i] = .{ .b = tensor.buffer, .off = tensor.offset };
        }
        var at: usize = 0;
        for (0..c.layers) |l| {
            t.slots[l] = at;
            at += job.slot(c, c.kinds[l], pack_rows);
        }
        t.acts = try m.device.buffer(at, opts);
        errdefer t.acts.deinit();
        t.cache = try st.Cache.init(m.device, c, max_rows + 1, false);
        errdefer t.cache.deinit(&b.pool);
        t.fresh = try b.pool.take();
        const Head = struct {
            t: *Trainer,
            pub fn encode(j: @This(), mb: *Metal, e: *fwd.Enc) !void {
                e.pipe(mb.m.kernels.get("tf_train_head_t"));
                for (j.t.head, 0..) |h, i| e.buf(h.b, h.off, i);
                e.buf(j.t.head_t, 0, 3);
                e.run(.{ mb.m.config.vocab, mb.m.config.hidden, 1 }, .{ 256, 1, 1 });
            }
        };
        try b.drain();
        try b.submit(.prefill, b.next, Head{ .t = t });
        try b.drain();
        t.sites.attach(&m.weights, true);
        @import("residency.zig").trainer(t, b.residency);
        return t;
    }

    pub fn deinit(t: *Trainer, gpa: std.mem.Allocator) void {
        @import("residency.zig").trainer(t, null);
        t.prefixes.deinit(gpa);
        t.sites.attach(&t.b.m.weights, false);
        t.sites.deinit();
        t.bufs.deinit();
        inline for (.{ "xb", "wt", "acc", "tiles" }) |f| @field(t.tmp, f).deinit();
        inline for (.{ "saved", "acts", "logits", "stats", "targets", "weights", "ans", "scale", "dhn", "head_t", "proj", "delta", "push", "inputs" }) |f| @field(t, f).deinit();
        t.b.pool.give(t.fresh);
        t.cache.deinit(&t.b.pool);
        gpa.destroy(t);
    }

    /// The mean loss of `ids`' answer from `start`, and what `mode` adds to it; `turn`: which rows carry which lessons.
    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode, turn: lesson_route.Turn) !Result {
        var out: [1]Result = undefined;
        try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = t.nudge }}, mode, turn, &out);
        return out[0];
    }

    /// Whether a pass may hold several sequences: no lesson block is open, so no row's lessons depend on its stream.
    pub fn packs(t: *const Trainer) bool {
        return t.sites.rank == 0;
    }

    /// Room for `n` prompt prefixes, each a pass's sequence's states after its first rows (none: their memory freed).
    pub fn prefixSlots(t: *Trainer, n: usize) !void {
        if (n == 0) t.prefixes.drop(t.gpa) else try t.prefixes.ensure(t.gpa, n);
    }

    /// `seqs` side by side in one pass (several: keys, prefixes, losses or targets), each's mean loss into `out`.
    pub fn pass(t: *Trainer, seqs: []const Seq, mode: Mode, turn: lesson_route.Turn, out: []Result) !void {
        const b = t.b;
        if (seqs.len == 0 or seqs.len > max_seqs or out.len < seqs.len) return error.BadPass;
        if (seqs.len > 1 and (!t.packs() or !(mode == .keys or mode == .loss or mode == .target or mode == .prime))) return error.BadPass;
        if (!reads(mode) and mode != .loss and mode != .target and t.sites.rank == 0) return error.NoOpenBlock;
        var rows: usize = 0;
        for (seqs, t.spans[0..seqs.len]) |sq, *sp| {
            if (sq.ids.len < 2 or sq.start < 1 or sq.start >= sq.ids.len or sq.ids.len - 1 > max_rows) return error.BadExample;
            try check(t, sq, mode);
            sp.* = .{ .at = rows, .rows = sq.ids.len - 1 - sq.skip, .skip = sq.skip, .slot = sq.slot };
            rows += sp.rows;
        }
        if (rows > t.pack) return error.PassTooLong;
        t.seqs = seqs.len;
        t.count = .{ .passes = t.count.passes + 1, .rows = t.count.rows + rows };
        t.answered = train_mode.answers(seqs, t.spans[0..seqs.len], t.answers[0..seqs.len]);
        const prompt = b.prompt.slice(u32, rows);
        const targets = t.targets.slice(u32, t.answered);
        const weights = t.weights.slice(f32, t.answered);
        const scale = t.scale.slice(f32, rows);
        for (seqs, t.spans[0..seqs.len], t.answers[0..seqs.len]) |sq, sp, a| {
            @memcpy(prompt[sp.at..][0..sp.rows], sq.ids[sq.skip..][0..sp.rows]);
            @memset(scale[sp.at..][0..sp.rows], sq.scale);
            const share = 1 / @as(f32, @floatFromInt(a.rows));
            for (0..a.rows) |r| {
                targets[a.at + r] = sq.ids[sq.start + r];
                weights[a.at + r] = share;
            }
        }
        t.cache.len = 0;
        try b.drain();
        lesson_route.flag(t.sites.latch.slice(u32, lesson_route.max_slots * lesson_route.max_blocks), lesson_route.trainer_slot, turn.bits);
        @memset(t.score(), 0);
        var tt = turn;
        tt.decide = @intCast(t.spans[0].row(seqs[0].decide));
        const j: job.Job = .{ .t = t, .rows = rows, .seqs = seqs, .mode = mode, .route = lesson_route.trained(rows, tt) };
        if (t.profile) |p| {
            var e: fwd.Enc = .{ .e = p.begin(), .prof = p };
            try j.encode(b, &e);
            e.e.end();
            p.cb.?.commit();
            p.cb.?.wait();
        } else {
            try b.submit(.prefill, b.next, j);
            try b.drain();
        }
        if (mode == .avoid or mode == .seek) t.sketched += rows;
        const stats = t.stats.slice([2]f32, t.answered);
        for (out[0..seqs.len], t.answers[0..seqs.len]) |*o, a| {
            o.* = .{ .loss = 0, .recalled = !reads(mode) };
            if (reads(mode)) continue;
            for (stats[a.at..][0..a.rows], weights[a.at..][0..a.rows]) |s, w| {
                o.loss += s[0] * w;
                o.recalled = o.recalled and s[1] > 0.5;
            }
        }
    }

    /// Whether `sq` may run in `mode`: a prefix ends before its answer, decision and nudges, in a slot that exists.
    fn check(t: *const Trainer, sq: Seq, mode: Mode) !void {
        if (mode == .prime and (sq.skip > 0 or sq.slot >= t.prefixes.slots.items.len)) return error.BadPrefix;
        if (sq.skip == 0) return;
        if (sq.skip >= sq.start or sq.decide < sq.skip or sq.slot >= t.prefixes.slots.items.len or !t.packs()) return error.BadPrefix;
        for (sq.nudge.layers, sq.nudge.rows) |mask, row| if (mask != 0 and row < sq.skip) return error.BadPrefix;
    }

    /// Each block's router score at the last step's decision row.
    pub fn score(t: *const Trainer) []f32 {
        return t.sites.score.slice(f32, lesson_route.max_slots * lesson_route.max_blocks)[lesson_route.trainer_slot * lesson_route.max_blocks ..][0..lesson_route.max_blocks];
    }

    /// MoE layer l's shared-expert activations (bf16 [W]) at `row`, the key its down projection reads, as last kept.
    pub fn key(t: *const Trainer, l: usize, row: usize) []const u16 {
        const c = t.b.m.config;
        const at = t.slots[l] + std.mem.alignForward(usize, job.bytes(c, .shared, pack_rows), 256) + row * c.shared_width * 2;
        return @as([*]const u16, @ptrCast(@alignCast(t.acts.contents() + at)))[0..c.shared_width];
    }

    /// The model's shape, as the pocket and the probe read it on either backend.
    pub fn config(t: *const Trainer) cfg.Config {
        return t.b.m.config;
    }

    /// MoE layer l's normed input (bf16 [D]) at the pass's `row`, as the last keys or target pass kept it.
    pub fn normed(t: *const Trainer, l: usize, row: usize) []const u16 {
        const d = t.b.m.config.hidden;
        return t.inputs.slice(u16, t.b.m.config.layers * pack_rows * d)[(l * pack_rows + row) * d ..][0..d];
    }

    /// The residual (bf16 [D]) at the pass's `row` entering layer l (l == layers: after the last), as last saved.
    pub fn residual(t: *const Trainer, l: usize, row: usize) []const u16 {
        const d = t.b.m.config.hidden;
        return t.saved.slice(u16, (t.b.m.config.layers + 1) * pack_rows * d)[(l * pack_rows + row) * d ..][0..d];
    }

    /// The loss's gradient sequence s's target pass read at its key i's row after layer l.
    pub fn pushAt(t: *const Trainer, s: usize, i: usize, l: usize) []const f32 {
        const c = t.b.m.config;
        return t.push.slice(f32, max_seqs * 3 * c.layers * c.hidden)[job.vec(c, s, i, l)..][0..c.hidden];
    }

    /// Sequence s's nudge for key i after layer l, which its target pass adds at that key's row.
    pub fn deltaAt(t: *const Trainer, s: usize, i: usize, l: usize) []f32 {
        const c = t.b.m.config;
        return t.delta.slice(f32, max_seqs * 3 * c.layers * c.hidden)[job.vec(c, s, i, l)..][0..c.hidden];
    }

    /// Each row along layer l's candidate directions ([rows, candidates]), as the last project step left them.
    pub fn projected(t: *const Trainer, l: usize, rows: usize) []const f32 {
        const k = adapters.candidates;
        return t.proj.slice(f32, t.b.m.config.layers * max_rows * k)[l * max_rows * k ..][0 .. rows * k];
    }

    /// The residual entering layer l in `saved`.
    pub fn savedAt(t: *const Trainer, l: usize) At {
        return At.of(t.saved).plus(l * pack_rows * t.b.m.config.hidden * 2);
    }
};
