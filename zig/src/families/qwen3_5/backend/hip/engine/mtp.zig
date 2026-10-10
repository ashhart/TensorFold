//! The MTP head drafting chains: one head forward a step over a row each; a round costs one chain's steps, one sync.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const view = @import("../model/view.zig");
const weights = @import("../model/weights.zig");
const bridge = @import("../model/bridge.zig");
const fwd = @import("../forward/forward.zig");
const draw = @import("draw.zig");
const plan = @import("../forward/plan.zig");
const graph_cache = @import("graph_cache.zig");

const pops = hip.plan_ops;
const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;

/// Most drafts a chain (the Python engine's depth).
pub const max_depth = 3;

/// One stream's chain: its last kept final row, the token after it and its slot, the drafts asked for and their draw.
pub const Job = struct {
    hidden: u64,
    token: u32,
    position: usize,
    depth: usize,
    sampling: ?lanes.Sampling,
    stop_under: f64,
    out: []u32,
    kept: usize = 0,
};

/// What a greedy batch's launches depend on: its chain bucket, drafts per chain, whether the confidence cut runs.
const ChainKey = struct { rows: u32, most: u32, cut: bool };

/// A greedy chain reads the head over the first ids of the vocabulary alone (the common tokens): a draft only guesses.
const draft_vocab = 65536;

/// Most chains one batch runs; a head serving `rows` rows a round holds caches for this many at most.
pub const max_chains = 64;

pub const Head = struct {
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    w: *const weights.MtpHead,
    heads: usize,
    kv_heads: usize,
    mlp: ?view.Mlp,
    logits_head: view.Projection,
    /// The same weights over the first `draft_vocab` ids.
    draft_head: view.Projection,
    /// Most chains a batch runs together; each has its own cache of `max_depth + 1` slots.
    cap: usize,
    cache_bytes: usize,
    k: hip.DeviceBuffer,
    v: hip.DeviceBuffer,
    arena: hip.Arena,
    /// The chains' last kept final rows gathered, (cap, hidden).
    rows: hip.DeviceBuffer,
    /// A row of zeros: the padding chains' input.
    zero: hip.DeviceBuffer,
    scalars: hip.HostBuffer, // `Layout`'s words, pinned
    scalars_dev: hip.DeviceBuffer,
    last: u64 = 0, // the last step's final rows, the next step's hidden input
    /// Greedy batches captured by shape: the head's launches over the scalars alone, replayed.
    graphs: graph_cache.Cache(ChainKey, void),

    /// The head of `m` with no device memory yet, or null when the checkpoint has none: what `deviceBytes` counts.
    fn describe(gpa: std.mem.Allocator, d: *const hip.Runtime, m: *const weights.Model, model: *const view.Model, cap: usize) !?Head {
        const w = if (m.mtp) |*x| x else return null;
        const s = m.spec;
        const qa = try bridge.projection(w.q);
        const ka = try bridge.projection(w.k);
        const kv_heads = ka.n / s.head_dim;
        const full = if (w.head) |hp| try bridge.projection(hp) else model.head;
        var narrow = full;
        narrow.n = @min(full.n, draft_vocab);
        const none: hip.DeviceBuffer = .{ .r = d, .ptr = null, .len = 0 };
        return .{
            .gpa = gpa,
            .d = d,
            .w = w,
            .heads = qa.n / s.head_dim / @as(usize, if (w.gated) 2 else 1),
            .kv_heads = kv_heads,
            .mlp = if (w.mlp) |x| try bridge.mlpView(x, false) else null,
            .logits_head = full,
            .draft_head = narrow,
            .cap = cap,
            .cache_bytes = kv_heads * (max_depth + 1) * s.head_dim * model.act.size(),
            .k = none,
            .v = none,
            .arena = hip.Arena.counting(),
            .rows = none,
            .zero = none,
            .graphs = graph_cache.Cache(ChainKey, void).init(gpa),
            .scalars = undefined,
            .scalars_dev = none,
        };
    }

    /// Device bytes the head of `m` holds for `cap` chains (0 without one): caches, rows, scalars and its arena.
    pub fn deviceBytes(gpa: std.mem.Allocator, d: *const hip.Runtime, lib: *const hip.Launcher, m: *const weights.Model, model: *const view.Model, cap: usize) !usize {
        var h = (try describe(gpa, d, m, model, cap)) orelse return 0;
        defer h.graphs.deinit();
        const row = m.spec.hidden * model.act.size();
        return 2 * h.cache_bytes * cap + cap * row + row + h.layout().total + try h.scratch(lib, model);
    }

    /// The head of `m`, or null when the checkpoint has none; `cap` chains at most run in one batch.
    pub fn init(gpa: std.mem.Allocator, d: *const hip.Runtime, lib: *const hip.Launcher, m: *const weights.Model, model: *const view.Model, cap: usize) !?*Head {
        const described = (try describe(gpa, d, m, model, cap)) orelse return null;
        const s = m.spec;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.* = described;
        errdefer h.graphs.deinit();
        h.k = try hip.DeviceBuffer.alloc(d, h.cache_bytes * cap);
        errdefer h.k.free();
        h.v = try hip.DeviceBuffer.alloc(d, h.cache_bytes * cap);
        errdefer h.v.free();
        h.rows = try hip.DeviceBuffer.alloc(d, cap * s.hidden * model.act.size());
        errdefer h.rows.free();
        h.zero = try hip.DeviceBuffer.alloc(d, s.hidden * model.act.size());
        errdefer h.zero.free();
        try h.zero.fill8(0);
        h.scalars = try hip.HostBuffer.alloc(d, h.layout().total);
        errdefer h.scalars.free();
        h.scalars_dev = try hip.DeviceBuffer.alloc(d, h.layout().total);
        errdefer h.scalars_dev.free();
        h.arena = try hip.Arena.init(d, try h.scratch(lib, model));
        return h;
    }

    /// The arena a batch of `cap` chains takes at full depth, greedy or sampled, counted on the counting stream.
    fn scratch(h: *Head, lib: *const hip.Launcher, m: *const view.Model) !usize {
        var most: usize = 0;
        for ([_]view.Projection{ h.draft_head, h.logits_head }) |head| {
            var arena = hip.Arena.counting();
            const o: Ops = .{ .l = lib, .bf16 = m.act == .bf16, .stream = hip.counting, .arena = &arena };
            try h.steps(o, m, head, h.cap, max_depth, true);
            most = @max(most, arena.peak);
        }
        return most;
    }

    pub fn deinit(h: *Head) void {
        h.graphs.deinit();
        h.scalars_dev.free();
        h.scalars.free();
        h.zero.free();
        h.rows.free();
        h.arena.deinit();
        h.v.free();
        h.k.free();
        h.gpa.destroy(h);
    }

    /// Scalar byte offsets: uploads (tokens, slots, positions, input row addresses), then downloads (drafts, probs).
    const Layout = struct { slots: usize, pos: usize, ptrs: usize, drafts: usize, probs: usize, total: usize };

    fn layout(h: *const Head) Layout {
        const slots = 4 * h.cap;
        const pos = slots + 4 * max_depth;
        const ptrs = std.mem.alignForward(usize, pos + 4 * max_depth * h.cap, 8);
        const drafts = ptrs + 8 * h.cap;
        const probs = drafts + 4 * max_depth * h.cap;
        return .{ .slots = slots, .pos = pos, .ptrs = ptrs, .drafts = drafts, .probs = probs, .total = probs + 4 * max_depth * h.cap };
    }

    /// A projection of the head in whichever format: an unquantized fp32 weight an MTP fc may keep, or a quantized one.
    fn project(o: Ops, p: weights.Projection, x: Tensor, rows: usize) !Tensor {
        return o.project(x, try bridge.projection(p), rows, false);
    }

    /// Every job's chain, `cap` at a time, up to `depth` drafts each, cut after a draft under `stop_under` (0: never).
    pub fn chains(h: *Head, lib: *const hip.Launcher, stream: hip.Stream, drawer: *draw.Drawer, m: *const view.Model, jobs: []Job, graphs: bool) !void {
        var at: usize = 0;
        while (at < jobs.len) : (at += h.cap) try h.batch(lib, stream, drawer, m, jobs[at..][0..@min(h.cap, jobs.len - at)], graphs);
    }

    fn batch(h: *Head, lib: *const hip.Launcher, stream: hip.Stream, drawer: *draw.Drawer, m: *const view.Model, jobs: []Job, graphs: bool) !void {
        const o: Ops = .{ .l = lib, .bf16 = m.act == .bf16, .stream = stream.handle, .arena = &h.arena };
        h.arena.reset();
        const rows = jobs.len;
        const at = h.layout();
        const width = m.spec.hidden * m.act.size();
        var most: usize = 0;
        var greedy = true;
        var cut = false;
        for (jobs) |j| {
            most = @max(most, @min(j.depth, max_depth, j.out.len));
            if (j.sampling) |s| greedy = greedy and s.temperature <= 0.0;
            cut = cut or j.stop_under > 0.0;
        }
        const sc = h.scalars.slice(i32);
        const srcs = h.scalars.slice(u64);
        // greedy chains run in a bucket of rows, the padding ones over zeros, so one graph serves any streams
        const run = if (greedy) plan.bucketOf(rows, h.cap) else rows;
        for (0..max_depth) |n| sc[h.cap + n] = @intCast(n);
        for (0..run) |r| {
            const real = r < jobs.len;
            sc[r] = if (real) @intCast(jobs[r].token) else 0;
            for (0..max_depth) |n| sc[at.pos / 4 + n * h.cap + r] = @intCast(n + if (real) jobs[r].position else 0);
            srcs[at.ptrs / 8 + r] = if (real) jobs[r].hidden else h.zero.base();
        }
        try hip.raw.upload(h.scalars_dev, 0, h.scalars.bytes[0..at.drafts], o.stream);
        const dev = h.scalars_dev.base();
        if (greedy) {
            try h.chain(stream, o, m, .{ .rows = @intCast(run), .most = @intCast(most), .cut = cut }, graphs);
        } else {
            try pops.gather(o, dev + at.ptrs, h.rows.base(), width / 4, rows);
            var cur: Tensor = .{ .ptr = h.rows.base(), .kind = m.act };
            var reqs: [64]draw.Request = undefined;
            var drawn: [64]u32 = undefined;
            for (0..most) |n| {
                const logits = try h.step(o, m, h.logits_head, cur, dev, rows, dev + at.slots + 4 * n, dev + at.pos + 4 * h.cap * n, n);
                for (jobs, 0..) |j, r| reqs[r] = .{ .sampling = j.sampling, .position = j.position + n + 1 };
                try drawer.draw(o, stream, logits, reqs[0..rows], drawn[0..rows], false);
                for (jobs, 0..) |j, r| if (n < j.out.len) {
                    j.out[n] = drawn[r];
                };
                if (n + 1 < most) {
                    for (0..rows) |r| sc[r] = @intCast(drawn[r]);
                    try hip.raw.upload(h.scalars_dev, 0, h.scalars.bytes[0 .. 4 * rows], o.stream);
                    if (cut) try o.tokenProb(logits, rows, h.logits_head.n, dev, dev + at.probs + 4 * h.cap * n);
                }
                cur = .{ .ptr = h.last, .kind = m.act };
            }
        }
        if (greedy or cut) {
            try hip.raw.download(h.scalars_dev, at.drafts, h.scalars.bytes[at.drafts..at.total], o.stream);
            try stream.synchronize();
        }
        const words = h.scalars.slice(u32);
        const chance = h.scalars.slice(f32);
        for (jobs, 0..) |*j, r| {
            const count = @min(j.depth, max_depth, j.out.len);
            if (greedy) for (0..count) |n| {
                j.out[n] = words[at.drafts / 4 + n * h.cap + r];
            };
            j.kept = count;
            if (j.stop_under > 0.0) for (0..count -| 1) |n| if (chance[at.probs / 4 + n * h.cap + r] < j.stop_under) {
                j.kept = n + 1;
                break;
            };
        }
    }

    const Recording = struct { h: *Head, o: Ops, m: *const view.Model, key: ChainKey };

    fn record(c: Recording) anyerror!void {
        try c.h.launches(c.o, c.m, c.key);
    }

    /// A greedy batch's launches: replayed from the shape's graph, captured the first time it is met, or eager.
    fn chain(h: *Head, stream: hip.Stream, o: Ops, m: *const view.Model, key: ChainKey, graphs: bool) !void {
        h.graphs.rounds += 1;
        const entry = if (graphs) try h.graphs.find(key) else null;
        if (entry) |e| switch (e.state) {
            .ready => {
                try e.exec.?.launchOn(stream);
                h.graphs.replayed += 1;
                return;
            },
            .seen => {
                if (try graph_cache.Cache(ChainKey, void).record(stream, Recording{ .h = h, .o = o, .m = m, .key = key }, record)) |rec| {
                    if (h.graphs.keep(e, rec.graph, stream, {})) {
                        try e.exec.?.launchOn(stream);
                        return;
                    } else |_| {}
                }
                e.state = .failed;
                h.arena.reset();
            },
            .failed => {},
        };
        try h.launches(o, m, key);
    }

    /// The chains' input rows gathered, then every step.
    fn launches(h: *Head, o: Ops, m: *const view.Model, key: ChainKey) !void {
        try pops.gather(o, h.scalars_dev.base() + h.layout().ptrs, h.rows.base(), m.spec.hidden * m.act.size() / 4, key.rows);
        try h.steps(o, m, h.draft_head, key.rows, key.most, key.cut);
    }

    /// The greedy chains' steps on the device alone: each step's drafts feed the next, with probabilities for the cut.
    fn steps(h: *Head, o: Ops, m: *const view.Model, head: view.Projection, rows: usize, most: usize, cut: bool) !void {
        const at = h.layout();
        const dev = h.scalars_dev.base();
        var cur: Tensor = .{ .ptr = h.rows.base(), .kind = m.act };
        for (0..most) |n| {
            const drafts = dev + at.drafts + 4 * h.cap * n;
            const ids = if (n == 0) dev else drafts - 4 * h.cap;
            const logits = try h.step(o, m, head, cur, ids, rows, dev + at.slots + 4 * n, dev + at.pos + 4 * h.cap * n, n);
            try o.argmaxRows(logits, rows, head.n, drafts);
            if (cut and n + 1 < most) try o.tokenProb(logits, rows, head.n, 0, dev + at.probs + 4 * h.cap * n);
            cur = .{ .ptr = h.last, .kind = m.act };
        }
    }

    /// One head step over `rows` chains at their rope positions, writing each chain's slot; the logits rows come back.
    fn step(h: *Head, o: Ops, m: *const view.Model, head: view.Projection, hidden: Tensor, ids: u64, rows: usize, slot_at: u64, pos: u64, slot: usize) !Tensor {
        const s = m.spec;
        const w = h.w;
        const eps: f32 = @floatCast(s.eps);
        const hd = s.head_dim;
        const emb = try fwd.take(o, m.act, rows * s.hidden);
        try o.embedRows(m.embed, ids, rows, emb);
        const emb_e = try fwd.take(o, m.act, rows * s.hidden);
        try o.rms(emb, w.fc_e_norm.ptr, emb_e, rows, s.hidden, eps);
        const emb_h = try fwd.take(o, m.act, rows * s.hidden);
        try o.rms(hidden, w.fc_h_norm.ptr, emb_h, rows, s.hidden, eps);
        const e_proj = try project(o, w.fc_e, emb_e, rows);
        const h_proj = try project(o, w.fc_h, emb_h, rows);
        const x = try fwd.take(o, m.act, rows * s.hidden);
        try o.add(e_proj, h_proj, x, rows * s.hidden);
        var normed = x;
        if (w.input_norm) |n| {
            normed = try fwd.take(o, m.act, rows * s.hidden);
            try o.rms(x, n.ptr, normed, rows, s.hidden, eps);
        }
        // the head's gated attention over each chain's cache (slots 0 .. slot), rope at the chain's position
        const qg = try project(o, w.q, normed, rows);
        const keys = try project(o, w.k, normed, rows);
        const values = try project(o, w.v, normed, rows);
        const qc = try fwd.take(o, m.act, rows * h.heads * hd);
        try o.copyCols(qg, if (w.gated) 2 * hd else hd, 0, qc.ptr, rows * h.heads, hd);
        const qn = try fwd.take(o, m.act, rows * h.heads * hd);
        try o.rms(qc, w.q_norm.ptr, qn, rows * h.heads, hd, eps);
        const kn = try fwd.take(o, m.act, rows * h.kv_heads * hd);
        try o.rms(keys, w.k_norm.ptr, kn, rows * h.kv_heads, hd, eps);
        const q32 = try ropeRows(o, m, qn, rows, h.heads, pos);
        const k32 = try ropeRows(o, m, kn, rows, h.kv_heads, pos);
        const kr = try fwd.take(o, m.act, rows * h.kv_heads * hd);
        try o.cast(.{ .ptr = k32, .kind = .f32 }, kr, rows * h.kv_heads * hd);
        const att = try o.arena.of(f32, rows * h.heads * hd);
        for (0..rows) |r| {
            const c: Ops.Cache = .{ .k = h.k.base() + r * h.cache_bytes, .v = h.v.base() + r * h.cache_bytes, .kind = m.act, .kv_heads = h.kv_heads, .total = max_depth + 1, .d = hd };
            try o.kvWrite(fwd.at(kr, r * h.kv_heads * hd), c.k, 1, h.kv_heads, hd, c.total, slot);
            try o.kvWrite(fwd.at(values, r * h.kv_heads * hd), c.v, 1, h.kv_heads, hd, c.total, slot);
            try o.causalAt(q32 + r * h.heads * hd * 4, c, att + r * h.heads * hd * 4, 1, h.heads, fwd.scaleOf(hd), slot_at);
        }
        const gated = try fwd.take(o, m.act, rows * h.heads * hd);
        if (w.gated) {
            try o.attnGate(att, qg, gated, rows, h.heads, hd, true);
        } else try o.cast(.{ .ptr = att, .kind = .f32 }, gated, rows * h.heads * hd);
        const attn_out = try project(o, w.o, gated, rows);
        try o.add(x, attn_out, x, rows * s.hidden);
        if (w.post_norm) |pn| if (h.mlp) |mlp| {
            const xn = try fwd.take(o, m.act, rows * s.hidden);
            try o.rms(x, pn.ptr, xn, rows, s.hidden, eps);
            const y = try fwd.mlpRows(o, m, mlp, xn, rows, false);
            try o.add(x, y, x, rows * s.hidden);
        };
        const residual = try fwd.take(o, m.act, rows * s.hidden);
        try o.rms(x, w.final_norm.ptr, residual, rows, s.hidden, eps);
        h.last = residual.ptr;
        return o.project(residual, head, rows, false);
    }
};

/// The decode RoPE of `rows` rows of `heads` heads, each at its device position, in the activation dtype.
fn ropeRows(o: Ops, m: *const view.Model, x: Tensor, rows: usize, heads: usize, pos: u64) !u64 {
    const s = m.spec;
    const n = rows * heads * s.head_dim;
    const wide = try o.arena.of(f32, n);
    try o.cast(x, .{ .ptr = wide, .kind = .f32 }, n);
    const turned = try o.arena.of(f32, n);
    try o.ropeDecode(wide, turned, rows * heads, s.head_dim, s.rotary_dim, 0, @floatCast(s.rope_theta), pos, heads);
    const narrow = try fwd.take(o, m.act, n);
    try o.cast(.{ .ptr = turned, .kind = .f32 }, narrow, n);
    const back = try o.arena.of(f32, n);
    try o.cast(narrow, .{ .ptr = back, .kind = .f32 }, n);
    return back;
}
