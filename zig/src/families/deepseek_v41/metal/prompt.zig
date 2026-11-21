//! The prompt path: encoder chunks over every row, then the decoder's exact suffix cone, the head and the DSpark taps.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const ctx_ = @import("ctx.zig");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const forward = @import("forward.zig");
const weights = @import("weights.zig");
const pipelines = @import("pipelines.zig");
const ps = @import("prompt_scratch.zig");
const pb = @import("prompt_bufs.zig");
const pl = @import("prompt_layer.zig");
const vrows = @import("vision_rows.zig");
const prep = @import("vision_prep.zig");
const pp = @import("prompt_prof.zig");
const pc = @import("prompt_chunk.zig");
const Chunk = pc.Chunk;
const submit = pc.submit;

const Config = host.config.Config;
const ced = host.ced;
const Ref = ctx_.Ref;
const g = pipelines.glue;
const cl = @import("calls.zig");
const eng = pipelines.engram;
const cv = ctx_.cv;
const plus = ctx_.plus;
const whole = ctx_.whole;

pub const Bufs = pb.Bufs;
pub const Tail = pb.Tail;
pub const Scratch = ps.Scratch;

pub const Opts = struct {
    /// rows a chunk carries (encoder and decoder passes alike)
    chunk: u32 = 4096,
    /// the cone's kept rows: the last layer's outputs the head and drafting read (at least the ring's spare rows + 1)
    kept: u32 = 128,
    /// logits rows and DSpark tap rows at the prompt's end
    head_rows: u32 = 1,
    taps_rows: u32 = pipelines.attn.window,
    /// index rows scored at once (the scores' memory)
    ix_rows: u32 = 64,
    /// every decoder layer on every row (the reference the cone must equal)
    full: bool = false,
    /// routed experts on the decode row kernels instead of the grouped kernels (the reference for them)
    expert_loop: bool = false,
    /// a cold chunk of seg_min rows or more runs its first Engram span on its first seg % of rows first (0: one run)
    seg: u32 = 30,
    seg_min: u32 = 512,
};

/// What a prompt reads: the config, kernels, RoPE tables, weights, Engram rows and the queue it commits on.
pub const Env = struct {
    c: *const Config,
    p: *const pipelines.Pipelines,
    tables: *const rope.Tables,
    model: forward.Model,
    lookup: ?forward.Lookup,
    queue: mtl.Queue,
};

/// The kept rows the cone runs with: enough that every decoder layer's keys cover its whole window ring.
pub fn keptOf(c: *const Config, o: Opts) u32 {
    return @max(o.kept, kv.windowSlots(c) - c.window + 1);
}

/// The tail rows a slot keeps: the decoder's first layer's input cone.
pub fn tailRows(c: *const Config, o: Opts) !u32 {
    return ced.suffixRows(c, keptOf(c, o));
}

/// The layers ced's cone needs: no KV source or Engram after the decoder's first layer, taps only in the decoder.
pub fn check(c: *const Config) !u32 {
    const d = try ced.decoderStart(c);
    for (d + 1..c.layers) |l| if (c.isKvSource(@intCast(l))) return error.KvSourceInDecoder;
    for (c.dspark_targets.items()) |t| if (t < d) return error.TapInEncoder;
    if (c.candidate_source >= 0 and c.candidate_source < d) {
        for (d..c.layers) |l| if (c.usesCandidates(@intCast(l))) return error.CandidatesAcrossPasses;
    }
    return d;
}

pub const Prof = pp.Prof;
/// The monotonic clock in ns (the prompt's profile and wall time).
pub const now = pp.now;

/// A chunk's inputs prepared ahead into input set `set`: the tail (slot) they are for and the chunk's rows.
const Stage = struct { tail: *const Tail, a: u32, b: u32, set: usize };

pub const Prompt = struct {
    o: Opts,
    d: u32,
    s: Scratch,
    b: Bufs,
    /// rows each layer ran in the last finish: window keys and whole rows
    keys: [64]u32 = @splat(0),
    outs: [64]u32 = @splat(0),
    /// rows of logits and taps the last finish wrote (the prompt's last rows)
    head: u32 = 0,
    taps: u32 = 0,
    /// the image rows of the prompt being run (null: text only); its image positions carry virtual ids
    images: ?*const vrows.Rows = null,
    prof: Prof = .{},
    /// the next chunk's inputs an `encode` that stopped short of the prompt's end prepared for the next call
    stage: ?Stage = null,
    /// calls whose first chunk's inputs the previous call prepared (`stage`)
    stage_hits: u64 = 0,
    /// each set's Engram row ids of every layer ([rows][cols], one hash pass a chunk), then the hashes' scratch
    ix: []i64 = &.{},

    pub fn init(device: mtl.Device, c: *const Config, o: Opts, limit: u32) !Prompt {
        const d = try check(c);
        if (o.chunk == 0 or o.head_rows == 0 or o.head_rows > keptOf(c, o) or o.taps_rows > c.window) return error.BadRows;
        if (c.layers > 64) return error.TooManyLayers;
        var s = try Scratch.init(device, c, o.chunk, o.ix_rows, limit);
        errdefer s.deinit();
        var b = try Bufs.init(device, c, o.chunk, o.head_rows, o.taps_rows);
        errdefer b.deinit();
        const ix = try std.heap.page_allocator.alloc(i64, 3 * @as(usize, o.chunk) * c.engram_layers.len * c.hashCols());
        return .{ .o = o, .d = d, .s = s, .b = b, .ix = ix };
    }

    pub fn deinit(p: *Prompt) void {
        p.s.deinit();
        p.b.deinit();
        std.heap.page_allocator.free(p.ix);
    }

    /// The whole prompt: positions [tail.end, hist.len) through the encoder, then the decoder's cone and the head.
    pub fn run(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *Tail, hist: []const u32) !void {
        p.prof = .{ .t0 = now() };
        p.stage = null;
        const t0 = p.prof.t0;
        try p.encode(env, slot, tail, hist, @intCast(hist.len));
        const td = now();
        try p.finish(env, slot, tail);
        p.prof.dec_ns = now() - td;
        p.report(env.c, hist.len, now() - t0);
    }

    /// The prompt's profile and wall time on the log when TF_DSV41_PROMPT_PROF is set.
    pub fn report(p: *const Prompt, c: *const Config, rows: usize, wall_ns: u64) void {
        if (std.c.getenv("TF_DSV41_PROMPT_PROF") != null) p.prof.report(c, rows, wall_ns);
    }

    /// The prepared next chunk is dropped if it was `tail`'s (that slot's prompt ended or restarts).
    pub fn forget(p: *Prompt, tail: *const Tail) void {
        if (p.stage) |st| if (st.tail == tail) {
            p.stage = null;
        };
    }

    /// Encoder chunks over [tail.end, upto): the next chunk fills ahead; an unstaged one between its command buffers.
    pub fn encode(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *Tail, hist: []const u32, upto: u32) !void {
        if (upto > hist.len or upto < tail.end) return error.BadRows;
        var cur: usize = 0;
        var staged = false;
        if (p.stage) |st| if (st.tail == tail and st.a == tail.end and tail.end < upto and st.b == try p.chunkEnd(tail.end, upto)) {
            cur = st.set;
            staged = true;
            p.stage_hits += 1;
        };
        p.stage = null;
        const from = tail.end;
        while (tail.end < upto) {
            const a = tail.end;
            const b = try p.chunkEnd(a, upto);
            const late: ?Late = if (staged) null else .{ .set = cur, .m = p.segRows(b - a), .n = b - a };
            if (late) |l| try p.prepIds(env, hist, a, b, l);
            staged = false;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const te = now();
            const filled = p.prof.prep_ns;
            var ch = try p.encoderChunk(env, slot, tail, a, b, cur, late);
            p.prof.enc_ns += (now() - te) - (p.prof.prep_ns - filled);
            var ahead: anyerror!void = {};
            // the next chunk of this call, else the next call's first, that call taking as many rows as this one
            const lim: u32 = if (b < upto) upto else @intCast(@min(hist.len, b + (upto - from)));
            var nb: u32 = b;
            if (b < lim) {
                if (p.chunkEnd(b, lim)) |e| {
                    nb = e;
                    ahead = p.prepTimed(env, hist, b, nb, 1 - cur);
                    staged = true;
                } else |err| ahead = err;
            }
            const t0 = now();
            const ran = ch.wait();
            p.prof.stall_ns += now() - t0;
            try ran;
            p.prof.gpu_s += ch.gpuSeconds();
            p.prof.enc_late_s += ch.lateSeconds();
            p.prof.chunks += 1;
            tail.end = b;
            try ahead;
            if (b == upto and nb > b) p.stage = .{ .tail = tail, .a = b, .b = nb, .set = 1 - cur };
            cur = 1 - cur;
        }
    }

    /// A cold chunk's first run of rows: `seg` percent of them (0: the chunk runs whole).
    fn segRows(p: *const Prompt, rows: u32) u32 {
        if (p.o.seg == 0 or rows < p.o.seg_min) return 0;
        const m: u32 = @intCast(@as(u64, rows) * p.o.seg / 100);
        return if (m == 0 or m >= rows) 0 else m;
    }

    /// The end of the chunk starting at `a`: `chunk` rows, cut so an image's span runs inside one chunk.
    fn chunkEnd(p: *const Prompt, a: u32, upto: u32) !u32 {
        var b = @min(upto, a + p.o.chunk);
        if (p.images) |im| b = try im.chunkEnd(a, b);
        if (b > upto) return error.ImageSpanSplit;
        return b;
    }

    /// Commits the chunk [a, b) reading input set `set` (its ids written; its Engram rows too unless `late`).
    fn encoderChunk(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *const Tail, a: u32, b: u32, set: usize, late: ?Late) !Chunk {
        const c = env.c;
        try p.s.writeRows(c, a, b - a, tail.cap);
        p.imageFlags(a, b);
        p.b.use(set);
        var ch: Chunk = .{};
        ch.open(env.queue);
        const done = p.encoderPass(env, slot, tail, &ch, a, b, late);
        if (done) |_| ch.close() else |_| ch.abandon();
        try done;
        return ch;
    }

    /// The chunk's ids (image positions embed the image token, the Engram gate shuts there) and Engram rows.
    fn prepTimed(p: *Prompt, env: Env, hist: []const u32, a: u32, b: u32, set: usize) !void {
        try p.writeIds(env, hist, a, b, set);
        if (env.c.engram_layers.len == 0) return;
        const t0 = now();
        try p.hashRows(env, hist, a, b, set);
        p.prof.prep_ns += now() - t0;
        for (0..env.c.engram_layers.len) |li| try p.rowsTimed(env, set, li, 0, b - a);
    }

    fn encoderPass(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *const Tail, ch: *Chunk, a: u32, b: u32, late: ?Late) !void {
        const c = env.c;
        const rows = b - a;
        const x = p.px(env, slot, a, b);
        const bb = &p.b;
        try g.embed(&env.p.glue, ch.e, env.model.top.embed, bb.ids, bb.streams[0], bb.pres[0], rows, c.hidden, c.vocab);
        if (p.images) |im| im.merge(ch.e, bb.streams[0], a, b);
        var i: u32 = 0;
        while (i < p.d) : (i += 1) {
            const w = &env.model.layers[i];
            if (w.layer != i) return error.LayerOrder;
            const at = i % 2;
            if (std.mem.indexOfScalar(u32, c.engram_layers.items(), w.layer)) |li| {
                if (late) |l| {
                    if (l.m > 0 and li == firstEngram(c)) {
                        i = try p.runs(env, slot, ch, a, b, l, li, i);
                        continue;
                    }
                    // the layers before run while this layer's rows are read and written
                    try p.between(env, ch, l, li, 0, rows);
                }
                try p.engram(&x, ch.e, w.engram orelse return error.EngramWeightsMissing, li, bb.streams[at], 0, rows);
            }
            try pl.encode(&x, ch.e, w, p.io(at), .{ .k = a, .o = a });
        }
        const e = ch.e;
        try pl.publish(&x, e, &env.model.layers[p.d], p.io(p.d % 2));
        // the streams entering layer d, the chunk's last `cap` rows, into the tail ring
        const m = @min(rows, tail.cap);
        const dest = p.s.metaAt(p.s.rec.tdest + b - m - a);
        const at = p.d % 2;
        const H = c.hidden;
        g.scatter(&env.p.glue, e, x.at(bb.streams[at], b - m, 8 * H), whole(tail.x), dest, m, 8 * H, 1);
        g.scatter(&env.p.glue, e, x.at(bb.pres[at], b - m, 16), whole(tail.pre), dest, m, 16, 1);
    }

    /// Between command buffers: the open one committed, Engram layer index `li`'s rows [r0, r1) read and written.
    fn between(p: *Prompt, env: Env, ch: *Chunk, l: Late, li: usize, r0: u32, r1: u32) !void {
        const c = env.c;
        ch.close();
        if (p.prof.first_ns == 0) p.prof.first_ns = now() - p.prof.t0;
        if (li == firstEngram(c) and r0 == 0) {
            // the chunk's first commit is out: the rest of this layer's reads, then the other layers', queue
            if (l.m > 0) p.issue(env, l.set, li, l.m, l.n);
            for (0..c.engram_layers.len) |lj| if (lj != li) p.issue(env, l.set, lj, 0, l.n);
        }
        try p.rowsTimed(env, l.set, li, r0, r1);
        ch.open(env.queue);
    }

    /// A cold chunk's first Engram span (layer i up to the next) on rows [0, m), then [m, rows); the last layer run.
    fn runs(p: *Prompt, env: Env, slot: *const kv.Slot, ch: *Chunk, a: u32, b: u32, l: Late, li: usize, i: u32) !u32 {
        const c = env.c;
        var end = p.d;
        for (c.engram_layers.items()) |el| if (el > i and el < end) {
            end = el;
        };
        for ([2][2]u32{ .{ 0, l.m }, .{ l.m, b - a } }) |span| {
            try p.between(env, ch, l, li, span[0], span[1]);
            const x = p.px(env, slot, a, a + span[1]);
            const w = &env.model.layers[i];
            try p.engram(&x, ch.e, w.engram orelse return error.EngramWeightsMissing, li, p.b.streams[i % 2], span[0], span[1] - span[0]);
            for (i..end) |j| try pl.encode(&x, ch.e, &env.model.layers[j], p.io(j % 2), .{ .k = a + span[0], .o = a + span[0] });
        }
        return end - 1;
    }

    /// The decoder layers d.. over their cone (every row if `full`) from the tail; the last rows' head and taps.
    pub fn finish(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *Tail) !void {
        const c = env.c;
        const n = tail.end;
        const f = tail.done;
        if (n == f) return error.NothingToFinish;
        @memset(&p.keys, 0);
        @memset(&p.outs, 0);
        p.head = @min(p.o.head_rows, n - f);
        p.taps = @min(p.o.taps_rows, n - f);
        const st = p.keyStart(c, n, f, p.d);
        if (n - st > tail.cap or st < f) return error.TailShort;
        var a = st;
        while (a < n) : (a = @min(n, a + p.o.chunk)) {
            try p.decoderChunk(env, slot, tail, a, @min(n, a + p.o.chunk));
        }
        tail.done = n;
    }

    /// The first position layer L's window keys (k) or whole rows (o) start at, for a prompt ending at n.
    fn keyStart(p: *const Prompt, c: *const Config, n: u32, f: u32, L: u32) u32 {
        if (p.o.full) return f;
        return @max(f, n -| (p.outRows(c, L) + c.window - 1));
    }

    fn outStart(p: *const Prompt, c: *const Config, n: u32, f: u32, L: u32) u32 {
        if (p.o.full) return f;
        return @max(f, n -| p.outRows(c, L));
    }

    fn outRows(p: *const Prompt, c: *const Config, L: u32) u32 {
        return ced.layerRows(c, keptOf(c, p.o), L) catch unreachable;
    }

    fn decoderChunk(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *const Tail, a: u32, b: u32) !void {
        try p.s.writeRows(env.c, a, b - a, tail.cap);
        p.imageFlags(a, b);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = env.queue.commandBuffer();
        const e = cb.compute(.serial);
        const done = p.decoderPass(env, slot, tail, e, a, b);
        e.end();
        try done;
        const tc = mtl.clock.seconds();
        try submit(cb);
        p.prof.dec_gpu_s += cb.gpuSeconds();
        p.prof.dec_late_s += cb.gpuStart() - tc;
    }

    fn decoderPass(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *const Tail, e: mtl.ComputeEncoder, a: u32, b: u32) !void {
        const c = env.c;
        const n = tail.end;
        const f = tail.done;
        const rows = b - a;
        const H = c.hidden;
        const x = p.px(env, slot, a, b);
        const bb = &p.b;
        // the tail's rows [a, b) into the first streams, as at most two ring runs
        const r0 = a % tail.cap;
        const first = @min(rows, tail.cap - r0);
        for ([2][3]u32{ .{ r0, 0, first }, .{ 0, first, rows - first } }) |span| {
            if (span[2] == 0) continue;
            const dest = p.s.metaAt(p.s.rec.iota + span[1]);
            g.scatter(&env.p.glue, e, plus(whole(tail.x), @as(usize, span[0]) * 8 * H), bb.streams[0], dest, span[2], 8 * H, 1);
            g.scatter(&env.p.glue, e, plus(whole(tail.pre), @as(usize, span[0]) * 16), bb.pres[0], dest, span[2], 16, 1);
        }
        const T: u32 = @intCast(c.dspark_targets.len);
        var ran: u32 = p.d;
        for (p.d..c.layers) |li| {
            const L: u32 = @intCast(li);
            const w = &env.model.layers[li];
            if (w.layer != L) return error.LayerOrder;
            const ka = @max(a, p.keyStart(c, n, f, L));
            if (ka >= b) break;
            const oa = @min(b, @max(a, p.outStart(c, n, f, L)));
            const at = (L - p.d) % 2;
            if (std.mem.indexOfScalar(u32, c.dspark_targets.items(), L)) |t| {
                // the tap of rows [n - taps, n) of this chunk, and the last row's into the slot (a one-row launch's)
                const t0 = @max(ka, n - p.taps);
                if (t0 < b) g.hcmean(&env.p.glue, e, x.at(bb.streams[at], t0, 8 * H), plus(bb.taps, (@as(usize, t0 - (n - p.taps)) * T + t) * H * 2), b - t0, H, T * H);
                if (b == n) g.hcmean(&env.p.glue, e, x.at(bb.streams[at], n - 1, 8 * H), .{ .buf = slot.taps.?, .off = t * H * 2 }, 1, H, T * H);
            }
            try pl.encode(&x, e, w, p.io(at), .{ .k = ka, .o = oa, .compress = false });
            p.keys[li] += b - ka;
            p.outs[li] += b - oa;
            ran = L + 1;
        }
        if (ran == c.layers) {
            const h0 = @max(a, n - p.head);
            if (h0 < b) {
                const last = (c.layers - p.d) % 2;
                const hr = plus(bb.hf, @as(usize, h0 - (n - p.head)) * H * 2);
                const hn = plus(bb.hfn, @as(usize, h0 - (n - p.head)) * H * 2);
                const lg = plus(bb.logits, @as(usize, h0 - (n - p.head)) * c.vocab * 4);
                cl.hcpre(env.p, e, x.at(bb.streams[last], h0, 8 * H), x.at(bb.pres[last], h0, 16), hr, b - h0, H);
                g.rmsnorm(&env.p.glue, e, hr, env.model.top.norm, hn, b - h0, H, c.eps);
                try g.mv(&env.p.glue, e, hn, env.model.top.head, lg, b - h0, H, c.vocab, true);
            }
        }
    }

    /// The logits of the prompt's last `head` rows, row r at [r * vocab].
    pub fn logits(p: *const Prompt, c: *const Config) []const f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, p.b.bytes(p.b.logits, @as(usize, p.head) * c.vocab * 4)));
    }

    /// The DSpark taps of the prompt's last `taps` rows: row r, target t at (r * targets + t) * hidden.
    pub fn tapRows(p: *const Prompt, c: *const Config) []const u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, p.b.bytes(p.b.taps, @as(usize, p.taps) * c.dspark_targets.len * c.hidden * 2)));
    }

    fn px(p: *const Prompt, env: Env, slot: *const kv.Slot, a: u32, b: u32) pl.Px {
        return .{ .c = env.c, .p = env.p, .s = &p.s, .slot = slot, .tables = env.tables, .a = a, .b = b, .expert_loop = p.o.expert_loop };
    }

    fn io(p: *const Prompt, at: usize) pl.Io {
        return .{ .x = p.b.streams[at], .pre = p.b.pres[at], .x_out = p.b.streams[1 - at], .pre_out = p.b.pres[1 - at] };
    }

    /// Each row's image flag (the route's bias_vl) for rows [a, b), and the merge's sources.
    fn imageFlags(p: *const Prompt, a: u32, b: u32) void {
        const flags = p.s.bytes(p.s.image, b - a);
        if (p.images) |im| im.describe(a, b, flags) else @memset(flags, 0);
    }

    /// The chunk's ids and Engram row ids; the first Engram layer's reads started for the chunk's first run of rows.
    fn prepIds(p: *Prompt, env: Env, hist: []const u32, a: u32, b: u32, l: Late) !void {
        const t0 = now();
        defer p.prof.prep_ns += now() - t0;
        try p.writeIds(env, hist, a, b, l.set);
        const c = env.c;
        if (c.engram_layers.len == 0) return;
        try p.hashRows(env, hist, a, b, l.set);
        p.issue(env, l.set, firstEngram(c), 0, if (l.m > 0) l.m else b - a);
        if (p.prof.issue_ns == 0) p.prof.issue_ns = now() - t0;
    }

    /// Set `set`'s Engram rows' ids of every layer for the chunk [a, b), one hash pass.
    fn hashRows(p: *Prompt, env: Env, hist: []const u32, a: u32, b: u32, set: usize) !void {
        const c = env.c;
        const l = env.lookup orelse return error.EngramLookupMissing;
        const L = c.engram_layers.len;
        if (l.host.layer_ids.len != L) return error.EngramLayers;
        var outs: [host.engram_host.max_layers][]i64 = undefined;
        for (outs[0..L], 0..) |*o, hl| o.* = p.ixOf(c, set, hl, b - a);
        const hs = p.ix[2 * @as(usize, p.o.chunk) * L * c.hashCols() ..][0 .. @as(usize, b - a) * L * c.hashCols()];
        try l.host.shardIndexAll(hist[a..b], hist[a -| (c.engram_max_ngram - 1)..a], 0, 1, hs, outs[0..L]);
    }

    /// Host layer index `hl`'s ids of set `set`'s first `n` rows.
    fn ixOf(p: *const Prompt, c: *const Config, set: usize, hl: usize, n: u32) []i64 {
        const per = @as(usize, p.o.chunk) * c.hashCols();
        return p.ix[(set * c.engram_layers.len + hl) * per ..][0 .. @as(usize, n) * c.hashCols()];
    }

    /// Starts reading Engram layer index `li`'s rows of set `set`'s rows [r0, r1); a failure is only logged.
    fn issue(p: *const Prompt, env: Env, set: usize, li: usize, r0: u32, r1: u32) void {
        const l = env.lookup orelse return;
        const s = l.source orelse return;
        const hl = hostIndex(l, env.c, li) orelse return;
        const C = env.c.hashCols();
        s.vtable.issue(s.ptr, hl, p.ixOf(env.c, set, hl, r1)[r0 * C ..]) catch |e| if (!@import("builtin").is_test) std.log.scoped(.dsv41).warn("engram: prefetch of layer {d} failed ({t})", .{ env.c.engram_layers.items()[li], e });
    }

    fn rowsTimed(p: *Prompt, env: Env, set: usize, li: usize, r0: u32, r1: u32) !void {
        const t0 = now();
        const s0 = pp.statsOf(env.lookup);
        defer {
            const dt = now() - t0;
            p.prof.prep_ns += dt;
            p.prof.eng[li].add(dt, s0, pp.statsOf(env.lookup), @as(u64, r1 - r0) * env.c.hashCols());
        }
        return p.engramRows(env, set, li, r0, r1);
    }

    fn writeIds(p: *const Prompt, env: Env, hist: []const u32, a: u32, b: u32, set: usize) !void {
        const ids = hist[a..b];
        const st = &p.b.sets[set];
        const emb: []u32 = @alignCast(std.mem.bytesAsSlice(u32, p.b.bytes(st.ids, 4 * ids.len)));
        const mask = p.b.bytes(st.mask, ids.len);
        for (ids, emb, mask) |id, *o, *m| {
            const image = prep.isVid(id);
            if (image and p.images == null) return error.BadToken;
            if (!image and id >= env.c.vocab) return error.BadToken;
            o.* = if (image) p.images.?.embed_id else id;
            m.* = @intFromBool(!image);
        }
    }

    /// Engram layer index `li`'s rows of set `set`'s rows [r0, r1) (their ids from `hashRows`) into the set.
    fn engramRows(p: *const Prompt, env: Env, set: usize, li: usize, r0: u32, r1: u32) !void {
        const c = env.c;
        const l = env.lookup orelse return error.EngramLookupMissing;
        const K = c.hashCols() * c.engram_head_dim;
        const all: []u16 = @alignCast(std.mem.bytesAsSlice(u16, p.b.bytes(p.b.sets[set].rows[li], 2 * @as(usize, r1) * K)));
        const out = all[r0 * K ..];
        const hl = hostIndex(l, c, li) orelse return error.NotEngram;
        const idx = p.ixOf(c, set, hl, r1)[r0 * c.hashCols() ..];
        const s = l.source orelse return host.engram_host.indexRows(idx, c.engram_head_dim, out);
        try s.vtable.rows(s.ptr, hl, idx, c.engram_head_dim, out);
    }

    /// forward.zig's step (bf16 rows in, from act_quant), wkv on the prompt MMA, into chunk rows [r0, r0 + rows).
    fn engram(p: *const Prompt, x: *const pl.Px, e: mtl.ComputeEncoder, w: weights.Engram, li: usize, streams: Ref, r0: u32, rows: u32) !void {
        const ep = &x.p.engram;
        const b = &p.b;
        const call = forward.engramCall(x.c, rows);
        if (w.wkv.k != call.cols * eng.head or w.wkv.n != call.n) return error.BadEngramShape;
        const st = plus(streams, @as(usize, r0) * 8 * x.c.hidden);
        try eng.encodeAq(ep, e, cv(eng.Ref, plus(b.rows[li], @as(usize, r0) * w.wkv.k * 2)), cv(eng.Ref, b.emb), call);
        const mma = pipelines.fp8_mma;
        try mma.encode(&x.p.mma, e, cv(mma.Ref, b.emb), cv(mma.Ref, w.wkv.w), cv(mma.Ref, w.wkv.s), cv(mma.Ref, b.ekv), .{ .k = w.wkv.k, .n = w.wkv.n, .rows = rows, .out = .bf16 });
        try eng.encodeGate(ep, e, cv(eng.Ref, st), cv(eng.Ref, b.ekv), cv(eng.Ref, w.q), cv(eng.Ref, w.k), cv(eng.Ref, plus(b.mask, r0)), cv(eng.Ref, b.gate), call);
        try eng.encodeAdd(ep, e, cv(eng.Ref, st), cv(eng.Ref, b.ekv), cv(eng.Ref, b.gate), cv(eng.Ref, st), call);
    }
};

/// A cold chunk of `n` rows in input set `set`: its Engram rows fill late; `m` > 0: its first Engram span in two runs.
const Late = struct { set: usize, m: u32, n: u32 };

/// The Engram layer the encoder meets first (its reads start before the first commit).
fn firstEngram(c: *const Config) usize {
    return std.mem.indexOfMin(u32, c.engram_layers.items());
}

/// The host's index of the config's Engram layer index `li`.
fn hostIndex(l: forward.Lookup, c: *const Config, li: usize) ?usize {
    return std.mem.indexOfScalar(u32, l.host.layer_ids, c.engram_layers.items()[li]);
}

test "the cone's rows by layer under the release config: kept 128, the ring's spare rows" {
    const c: Config = .{};
    const o: Opts = .{};
    try std.testing.expectEqual(@as(u32, 128), keptOf(&c, o));
    try std.testing.expectEqual(@as(u32, 17), keptOf(&c, .{ .kept = 1 }));
    try std.testing.expectEqual(@as(u32, 128 + 20 * 127), try tailRows(&c, o));
    try std.testing.expectEqual(@as(u32, 20), try check(&c));
}
