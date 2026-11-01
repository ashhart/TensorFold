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

/// Where a prompt's encoder time went: host fills, host encoding, host waits, GPU time (TF_DSV41_PROMPT_PROF).
pub const Prof = struct { prep_ns: u64 = 0, enc_ns: u64 = 0, stall_ns: u64 = 0, gpu_s: f64 = 0, chunks: u32 = 0 };

fn now() u64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
}

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
    prof: Prof = .{},

    pub fn init(device: mtl.Device, c: *const Config, o: Opts, limit: u32) !Prompt {
        const d = try check(c);
        if (o.chunk == 0 or o.head_rows == 0 or o.head_rows > keptOf(c, o) or o.taps_rows > c.window) return error.BadRows;
        if (c.layers > 64) return error.TooManyLayers;
        var s = try Scratch.init(device, c, o.chunk, o.ix_rows, limit);
        errdefer s.deinit();
        return .{ .o = o, .d = d, .s = s, .b = try Bufs.init(device, c, o.chunk, o.head_rows, o.taps_rows) };
    }

    pub fn deinit(p: *Prompt) void {
        p.s.deinit();
        p.b.deinit();
    }

    /// The whole prompt: positions [tail.end, hist.len) through the encoder, then the decoder's cone and the head.
    pub fn run(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *Tail, hist: []const u32) !void {
        p.prof = .{};
        const t0 = now();
        try p.encode(env, slot, tail, hist, @intCast(hist.len));
        try p.finish(env, slot, tail);
        if (std.c.getenv("TF_DSV41_PROMPT_PROF") == null) return;
        const ms = struct {
            fn of(ns: u64) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.of;
        std.log.scoped(.dsv41).info("prompt {d} rows: {d:.0} ms, encoder {d} chunks: gpu {d:.0} ms, host fill {d:.0} ms, host encode {d:.0} ms, host wait {d:.0} ms", .{ hist.len, ms(now() - t0), p.prof.chunks, p.prof.gpu_s * 1e3, ms(p.prof.prep_ns), ms(p.prof.enc_ns), ms(p.prof.stall_ns) });
    }

    /// Encoder chunks over [tail.end, upto): layers 0..d-1, d's publish, the tail; next chunk's inputs fill meanwhile.
    pub fn encode(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *Tail, hist: []const u32, upto: u32) !void {
        if (upto > hist.len or upto < tail.end) return error.BadRows;
        var cur: usize = 0;
        var staged = false;
        while (tail.end < upto) {
            const a = tail.end;
            const b = try p.chunkEnd(a, upto);
            if (!staged) try p.prepTimed(env, hist, a, b, cur);
            staged = false;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const te = now();
            const cb = try p.encoderChunk(env, slot, tail, a, b, cur);
            p.prof.enc_ns += now() - te;
            var ahead: anyerror!void = {};
            if (b < upto) {
                if (p.chunkEnd(b, upto)) |nb| {
                    ahead = p.prepTimed(env, hist, b, nb, 1 - cur);
                    staged = true;
                } else |err| ahead = err;
            }
            const t0 = now();
            const ran = submitted(cb);
            p.prof.stall_ns += now() - t0;
            try ran;
            p.prof.gpu_s += cb.gpuSeconds();
            p.prof.chunks += 1;
            tail.end = b;
            try ahead;
            cur = 1 - cur;
        }
    }

    /// The end of the chunk starting at `a`: `chunk` rows, at most `upto`.
    fn chunkEnd(p: *const Prompt, a: u32, upto: u32) !u32 {
        return @min(upto, a + p.o.chunk);
    }

    /// Commits the chunk [a, b) reading input set `set` (prepared); the caller waits on the command buffer.
    fn encoderChunk(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *const Tail, a: u32, b: u32, set: usize) !mtl.CommandBuffer {
        const c = env.c;
        try p.s.writeRows(c, a, b - a, tail.cap);
        p.imageFlags(a, b);
        p.b.use(set);
        const cb = env.queue.commandBuffer();
        const e = cb.compute(.serial);
        const done = p.encoderPass(env, slot, tail, e, a, b);
        e.end();
        try done;
        cb.commit();
        return cb;
    }

    fn prepTimed(p: *Prompt, env: Env, hist: []const u32, a: u32, b: u32, set: usize) !void {
        const t0 = now();
        defer p.prof.prep_ns += now() - t0;
        return p.prepare(env, hist, a, b, set);
    }

    fn encoderPass(p: *Prompt, env: Env, slot: *const kv.Slot, tail: *const Tail, e: mtl.ComputeEncoder, a: u32, b: u32) !void {
        const c = env.c;
        const rows = b - a;
        const x = p.px(env, slot, a, b);
        const bb = &p.b;
        g.embed(&env.p.glue, e, env.model.top.embed, bb.ids, bb.streams[0], bb.pres[0], rows, c.hidden);
        for (0..p.d) |i| {
            const w = &env.model.layers[i];
            if (w.layer != i) return error.LayerOrder;
            const at = i % 2;
            if (std.mem.indexOfScalar(u32, c.engram_layers.items(), w.layer)) |li| {
                try p.engram(&x, e, w.engram orelse return error.EngramWeightsMissing, li, bb.streams[at], rows);
            }
            try pl.encode(&x, e, w, p.io(at), .{ .k = a, .o = a });
        }
        try pl.publish(&x, e, &env.model.layers[p.d], p.io(p.d % 2));
        // the streams entering layer d, the chunk's last `cap` rows, into the tail ring
        const m = @min(rows, tail.cap);
        const dest = p.s.metaAt(p.s.rec.tdest + b - m - a);
        const at = p.d % 2;
        const H = c.hidden;
        g.scatter(&env.p.glue, e, x.at(bb.streams[at], b - m, 8 * H), whole(tail.x), dest, m, 8 * H, 1);
        g.scatter(&env.p.glue, e, x.at(bb.pres[at], b - m, 16), whole(tail.pre), dest, m, 16, 1);
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
        try submit(cb);
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
                g.mv(&env.p.glue, e, hn, env.model.top.head, lg, b - h0, H, c.vocab, true);
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

    /// Each row's image flag (the route's bias_vl) for rows [a, b): text rows, all clear.
    fn imageFlags(p: *const Prompt, a: u32, b: u32) void {
        @memset(p.s.bytes(p.s.image, b - a), 0);
    }

    /// The chunk's ids (every row's Engram gate open) and Engram rows.
    fn prepare(p: *const Prompt, env: Env, hist: []const u32, a: u32, b: u32, set: usize) !void {
        const c = env.c;
        const ids = hist[a..b];
        const st = &p.b.sets[set];
        const emb: []u32 = @alignCast(std.mem.bytesAsSlice(u32, p.b.bytes(st.ids, 4 * ids.len)));
        const mask = p.b.bytes(st.mask, ids.len);
        for (ids, emb, mask) |id, *o, *m| {
            if (id >= c.vocab) return error.BadToken;
            o.* = id;
            m.* = 1;
        }
        if (c.engram_layers.len == 0) return;
        const l = env.lookup orelse return error.EngramLookupMissing;
        const look = hist[a -| (c.engram_max_ngram - 1)..a];
        const K = c.hashCols() * c.engram_head_dim;
        for (c.engram_layers.items(), 0..) |id, li| {
            const out: []u16 = @alignCast(std.mem.bytesAsSlice(u16, p.b.bytes(st.rows[li], 2 * ids.len * K)));
            try l.host.windowRows(l.a, l.source, ids, look, id, 0, 1, c.engram_head_dim, out);
        }
    }

    /// As forward.zig's (rows come as bf16, the step starts at act_quant), wkv on the prompt MMA; adds into streams.
    fn engram(p: *const Prompt, x: *const pl.Px, e: mtl.ComputeEncoder, w: weights.Engram, li: usize, streams: Ref, rows: u32) !void {
        const ep = &x.p.engram;
        const b = &p.b;
        const call = forward.engramCall(x.c, rows);
        if (w.wkv.k != call.cols * eng.head or w.wkv.n != call.n) return error.BadEngramShape;
        try eng.encodeAq(ep, e, cv(eng.Ref, b.rows[li]), cv(eng.Ref, b.emb), call);
        const mma = pipelines.fp8_mma;
        try mma.encode(&x.p.mma, e, cv(mma.Ref, b.emb), cv(mma.Ref, w.wkv.w), cv(mma.Ref, w.wkv.s), cv(mma.Ref, b.ekv), .{ .k = w.wkv.k, .n = w.wkv.n, .rows = rows, .out = .bf16 });
        try eng.encodeGate(ep, e, cv(eng.Ref, streams), cv(eng.Ref, b.ekv), cv(eng.Ref, w.q), cv(eng.Ref, w.k), cv(eng.Ref, b.mask), cv(eng.Ref, b.gate), call);
        try eng.encodeAdd(ep, e, cv(eng.Ref, streams), cv(eng.Ref, b.ekv), cv(eng.Ref, b.gate), cv(eng.Ref, streams), call);
    }
};

fn submit(cb: mtl.CommandBuffer) !void {
    cb.commit();
    return submitted(cb);
}

/// Waits for a committed command buffer and reports its failure.
fn submitted(cb: mtl.CommandBuffer) !void {
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("prompt command buffer failed: {s}\n", .{msg});
        return error.CommandBufferFailed;
    }
}

test "the cone's rows by layer under the release config: kept 128, the ring's spare rows" {
    const c: Config = .{};
    const o: Opts = .{};
    try std.testing.expectEqual(@as(u32, 128), keptOf(&c, o));
    try std.testing.expectEqual(@as(u32, 17), keptOf(&c, .{ .kept = 1 }));
    try std.testing.expectEqual(@as(u32, 128 + 20 * 127), try tailRows(&c, o));
    try std.testing.expectEqual(@as(u32, 20), try check(&c));
}
