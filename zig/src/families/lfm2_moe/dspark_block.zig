//! DSpark's passes on the target's queue: context rows into a lane's keys, and a block's rows into Markov drafts.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const tile = @import("../../core/tile_qmm.zig");
const am = @import("../../core/affine_mm.zig");
const round = @import("tree_round_gpu");
const ds = @import("dspark.zig");
const dw = @import("dspark_weights.zig");
const Drafter = ds.Drafter;
const Lane = ds.Lane;

const kw = c.kv_heads * c.head_dim;
const QkArgs = extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32 };
const AttnArgs = extern struct { len: u32, cap: u32, scale: f32, subs: u32 };
const Pick = round.contract.Pick;
const Bf16Args = extern struct { rows: u32, n: u32, k: u32 };
const MarkovArgs = extern struct { stride: u32, rank: u32, vocab: u32, prev_stride: u32 };
const ConfidenceArgs = extern struct { n: u32, pass: u32, vocab: u32, rank: u32, hidden: u32 };
/// Tokens a Markov scan threadgroup takes (the kernel's MARKOV_SPAN), and threadgroups a stream's row takes.
pub const scan_span = 512;
pub const scan_groups = (c.vocab + scan_span - 1) / scan_span;

/// A buffer and byte offset.
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// One stream's block in a hold pass: rows [pending, mask x (block - 1)] at positions anchor .. anchor + block - 1.
pub const Block = struct { lane: *Lane, pending: u32, anchor: u64, depth: u32 };

const Enc = struct {
    d: *Drafter,
    e: mtl.ComputeEncoder,

    fn pipe(x: Enc, comptime key: []const u8) void {
        x.e.setPipeline(x.d.target.kernels.get(key));
    }

    /// y[rows] = x[rows] W^T: the twin up to its rows, past them the M5's tensor-unit kernels, else the tiles.
    fn project(x: Enc, lin: dw.Linear, in: Ref, out: Ref, rows: usize) !void {
        const w = switch (lin) {
            .codes => |m| m,
            .bf16 => |m| return x.bf16(m, in, out, rows),
        };
        const t = &x.d.target.kernels.tiles;
        if (rows > tile.twin_rows) if (x.d.target.kernels.naxPipes(w.bits)) |p| {
            const sums = Ref{ .buf = x.d.s.sums };
            am.rowSums(x.e, p, c.group, in, sums, @intCast(rows), @intCast(w.k));
            const q = .{ .w = Ref{ .buf = w.w, .off = w.w_off }, .s = Ref{ .buf = w.scales, .off = w.s_off }, .b = Ref{ .buf = w.biases, .off = w.b_off }, .n = @as(u32, @intCast(w.n)), .k = @as(u32, @intCast(w.k)) };
            return am.dense(x.e, p, in, sums, q, out, @intCast(rows));
        };
        if (rows <= tile.twin_rows and x.d.target.kernels.twin) return tile.encodeTwin(x.e, t, w, in.buf, in.off, out.buf, out.off, rows);
        tile.encodeSums(x.e, t, w.k, in.buf, in.off, x.d.s.sums, rows);
        try tile.encode(x.e, t, w, in.buf, in.off, x.d.s.sums, 0, out.buf, out.off, rows);
    }

    /// y[rows] = x[rows] W^T over the released bf16 matrix: 16 rows a pass of W up to 32 rows, else 64x64 tiles.
    fn bf16(x: Enc, m: dw.Bf16, in: Ref, out: Ref, rows: usize) void {
        const args = Bf16Args{ .rows = @intCast(rows), .n = @intCast(m.n), .k = @intCast(m.k) };
        const few = rows <= 32;
        if (few) x.pipe("lfm2_bf16_rows") else x.pipe("lfm2_bf16_mma");
        x.e.setBuffer(in.buf, in.off, 0);
        x.e.setBuffer(m.w, 0, 1);
        x.e.setBytes(std.mem.asBytes(&args), 2);
        x.e.setBuffer(out.buf, out.off, 3);
        if (few) {
            x.e.dispatchGroups(mtl.Size.of((m.n + 7) / 8, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
        } else x.e.dispatchGroups(mtl.Size.of((m.n + 63) / 64, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    }

    /// h += r when `add`, then out = norm(h) * gain: LFM2's own norm (the drafter's width and eps are the target's).
    fn norm(x: Enc, h: mtl.Buffer, r: mtl.Buffer, gain: mtl.Buffer, add: bool, out: mtl.Buffer, rows: usize) void {
        x.pipe("lfm2_norm");
        x.e.setBuffer(h, 0, 0);
        x.e.setBuffer(r, 0, 1);
        x.e.setBuffer(gain, 0, 2);
        x.e.setValue(c.eps, 3);
        x.e.setValue(@as(u32, @intFromBool(add)), 4);
        x.e.setBuffer(out, 0, 5);
        x.e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(c.hidden / 4, 1, 1));
    }

    /// Rows' queries normed and rotated into `queries`, keys and values into the lane's layer at positions from `len`.
    fn qk(x: Enc, q: Ref, k: Ref, v: Ref, layer: usize, lane: *const Lane, len: u64, queries: Ref, rows: usize) void {
        const ly = &x.d.w.layers[layer];
        x.pipe("lfm2_qk");
        x.e.setBuffer(q.buf, q.off, 0);
        x.e.setBuffer(k.buf, k.off, 1);
        x.e.setBuffer(v.buf, v.off, 2);
        x.e.setBuffer(ly.q_norm, 0, 3);
        x.e.setBuffer(ly.k_norm, 0, 4);
        const args = QkArgs{ .len = @intCast(len), .cap = @intCast(lane.cap), .eps = c.eps, .log2_theta = std.math.log2(x.d.config.theta) };
        x.e.setBytes(std.mem.asBytes(&args), 5);
        x.e.setBuffer(queries.buf, queries.off, 6);
        x.e.setBuffer(lane.keys[layer], 0, 7);
        x.e.setBuffer(lane.values[layer], 0, 8);
        x.e.dispatchGroups(mtl.Size.of(c.query_heads + c.kv_heads, rows, 1), mtl.Size.of(32, 1, 1));
    }

    /// Query rows [row, row + rows) against the lane's keys [0, keys), all of them visible to every row.
    fn attend(x: Enc, layer: usize, lane: *const Lane, row: usize, rows: usize, keys: u64) void {
        const s = &x.d.s;
        const args = AttnArgs{ .len = @intCast(keys - 1), .cap = @intCast(lane.cap), .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .subs = st.subs };
        const used = (keys + st.chunk_keys - 1) / st.chunk_keys * st.subs;
        x.pipe("lfm2_attn_block");
        x.e.setBuffer(s.queries, row * c.hidden * 2, 0);
        x.e.setBuffer(lane.keys[layer], 0, 1);
        x.e.setBuffer(lane.values[layer], 0, 2);
        x.e.setBytes(std.mem.asBytes(&args), 3);
        x.e.setBuffer(s.pm, 0, 4);
        x.e.setBuffer(s.pl, 0, 5);
        x.e.setBuffer(s.po, 0, 6);
        x.e.setValue(@as(u32, @intCast(rows)), 7);
        x.e.dispatchGroups(mtl.Size.of((rows + 7) / 8, c.kv_heads, @intCast(used)), mtl.Size.of(128, 1, 1));
        x.pipe("lfm2_attn_block_merge");
        x.e.setBuffer(s.pm, 0, 0);
        x.e.setBuffer(s.pl, 0, 1);
        x.e.setBuffer(s.po, 0, 2);
        x.e.setBytes(std.mem.asBytes(&args), 3);
        x.e.setBuffer(s.y, row * c.hidden * 2, 4);
        x.e.dispatchGroups(mtl.Size.of(c.query_heads, rows, 1), mtl.Size.of(32, 1, 1));
    }
};

fn failed(msg: [*:0]const u8) anyerror {
    std.log.err("lfm2 dspark: command buffer failed: {s}", .{msg});
    return error.GpuFailed;
}

fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| return failed(msg);
}

/// One stream's tapped rows in an absorb pass: `rows` rows at `src`, the context from position `p0`.
pub const Piece = struct { lane: *Lane, src: Ref, rows: usize, p0: u64 };

/// Each piece's rows into its lane's keys and values (fc, hidden_norm, k, v, k_norm, RoPE) in passes on `e`.
pub fn absorbPass(d: *Drafter, e: mtl.ComputeEncoder, pieces: []const Piece) !void {
    const s = &d.s;
    const x = Enc{ .d = d, .e = e };
    var rows: usize = 0;
    for (pieces) |pc| {
        try x.project(d.w.fc, pc.src, .{ .buf = s.fcx, .off = rows * c.hidden * 2 }, pc.rows);
        rows += pc.rows;
    }
    if (rows == 0 or rows > ds.absorb_rows) return error.DraftContextGap;
    x.norm(s.fcx, s.fcx, d.w.hidden_norm, false, s.hctx, rows);
    for (0..d.config.layers) |i| {
        const ly = &d.w.layers[i];
        try x.project(ly.k, .{ .buf = s.hctx }, .{ .buf = s.k }, rows);
        try x.project(ly.v, .{ .buf = s.hctx }, .{ .buf = s.v }, rows);
        var at: usize = 0;
        for (pieces) |pc| {
            // the kernel rotates query heads too: they read the context rows and land in x, which no one reads
            const q = Ref{ .buf = s.hctx, .off = at * c.hidden * 2 };
            x.qk(q, .{ .buf = s.k, .off = at * kw * 2 }, .{ .buf = s.v, .off = at * kw * 2 }, i, pc.lane, pc.p0, .{ .buf = s.x }, pc.rows);
            at += pc.rows;
        }
    }
}

/// A command buffer for absorb passes, and its wait.
pub fn begin(d: *Drafter) struct { cb: mtl.CommandBuffer, e: mtl.ComputeEncoder } {
    const cb = d.target.queue.commandBuffer();
    return .{ .cb = cb, .e = cb.compute(.serial) };
}

pub fn end(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    return finish(cb, e);
}

pub fn takeHoldMs(ptr: *anyopaque) f64 {
    const d: *Drafter = @ptrCast(@alignCast(ptr));
    const ms = d.hold_spent_ms;
    d.hold_spent_ms = 0;
    return ms;
}

/// A GPU event a hold signals instead of being waited on, and the value it sets.
pub const Signal = struct { event: mtl.Event, value: u64 };

/// The blocks' drafts (backbone, then Markov chain) into the lanes' held chains; with `signal`, left for readPicks.
pub fn hold(d: *Drafter, blocks: []const Block, signal: ?Signal) !?mtl.CommandBuffer {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const s = &d.s;
    var depth: u32 = 0;
    for (blocks) |b| depth = @max(depth, b.depth);
    // a block of the deepest chain's rows, at most the 9 the drafter was trained on
    const n: u32 = depth;
    const rows = blocks.len * n;
    if (rows == 0 or rows > ds.pass_rows) return error.DraftTooManyStreams;
    const ids = s.ids.slice(u32, ds.pass_rows);
    for (0..blocks.len) |i| {
        ids[i * n] = blocks[i].pending;
        @memset(ids[i * n + 1 ..][0 .. n - 1], d.config.mask);
    }
    const steps = depth;
    // past the last layer's attention a lone block carries only the rows its chain reads; several keep them all
    const tail = if (blocks.len == 1) steps else rows;
    const cb = d.target.queue.commandBuffer();
    const x = Enc{ .d = d, .e = cb.compute(.serial) };
    const e = x.e;
    // the blocks' staged context rows first, in the same command buffer
    var staged: [ds.pass_rows]Piece = undefined;
    var count: usize = 0;
    var held: usize = 0;
    for (blocks) |b| if (ds.Drafter.stagedPiece(b.lane)) |pc| {
        if (held + pc.rows > ds.absorb_rows) {
            try absorbPass(d, e, staged[0..count]);
            count = 0;
            held = 0;
        }
        staged[count] = pc;
        count += 1;
        held += pc.rows;
    };
    if (count > 0) try absorbPass(d, e, staged[0..count]);
    for (blocks) |b| {
        b.lane.end += b.lane.staged;
        b.lane.staged = 0;
    }
    const t = d.target;
    const emb = t.weights.embedding;
    e.setPipeline(t.kernels.embed(emb.bits));
    e.setBuffer(s.ids, 0, 0);
    e.setBuffer(emb.w, emb.w_off, 1);
    e.setBuffer(emb.scales, emb.s_off, 2);
    e.setBuffer(emb.biases, emb.b_off, 3);
    e.setBuffer(s.h, 0, 4);
    const units = c.hidden / @as(usize, if (emb.bits == 6) 4 else 8 / emb.bits);
    e.dispatchThreads(mtl.Size.of(units, rows, 1), mtl.Size.of(256, 1, 1));
    for (0..d.config.layers) |i| {
        const ly = &d.w.layers[i];
        const out = if (i + 1 == d.config.layers) tail else rows;
        x.norm(s.h, s.r, ly.input_norm, i > 0, s.x, rows);
        try x.project(ly.q, .{ .buf = s.x }, .{ .buf = s.q }, rows);
        try x.project(ly.k, .{ .buf = s.x }, .{ .buf = s.k }, rows);
        try x.project(ly.v, .{ .buf = s.x }, .{ .buf = s.v }, rows);
        for (blocks, 0..) |b, bi| {
            const at = bi * n;
            x.qk(.{ .buf = s.q, .off = at * c.hidden * 2 }, .{ .buf = s.k, .off = at * kw * 2 }, .{ .buf = s.v, .off = at * kw * 2 }, i, b.lane, b.anchor, .{ .buf = s.queries, .off = at * c.hidden * 2 }, n);
            x.attend(i, b.lane, at, if (blocks.len == 1) out else n, b.anchor + n);
        }
        try x.project(ly.o, .{ .buf = s.y }, .{ .buf = s.r }, out);
        x.norm(s.h, s.r, ly.post_norm, true, s.x, out);
        try x.project(ly.gate, .{ .buf = s.x }, .{ .buf = s.gate }, out);
        try x.project(ly.up, .{ .buf = s.x }, .{ .buf = s.up }, out);
        x.pipe("lfm2_swiglu");
        e.setBuffer(s.gate, 0, 0);
        e.setBuffer(s.up, 0, 1);
        e.setBuffer(s.act, 0, 2);
        e.setValue(@as(u32, @intCast(out * d.config.intermediate)), 3);
        e.dispatchThreads(mtl.Size.of(out * d.config.intermediate / 8, 1, 1), mtl.Size.of(256, 1, 1));
        try x.project(ly.down, .{ .buf = s.act }, .{ .buf = s.r }, out);
    }
    x.norm(s.h, s.r, d.w.norm, true, s.x, tail);
    try x.project(.{ .codes = fwd.matrix(emb) }, .{ .buf = s.x }, .{ .buf = s.logits }, tail);
    try markov(x, blocks.len, n, steps);
    if (d.confident()) confidence(x, blocks.len, n, steps);
    if (signal) |sg| { // committed unwaited, the command buffer returned retained
        e.end();
        cb.signalEvent(sg.event, sg.value);
        const kept = cb.retain();
        cb.commit();
        return kept;
    }
    try finish(cb, e);
    d.hold_spent_ms += (cb.gpuEnd() - cb.gpuStart()) * 1000;
    const picks = s.picks.slice(Pick, d.config.block * ds.pass_rows);
    for (blocks, 0..) |b, bi| {
        for (b.lane.held[0..steps], 0..) |*h, i| {
            const p = picks[i * ds.pass_rows + bi];
            if (p.nonfinite != 0 or p.token >= c.vocab) return error.DraftNonfinite;
            h.* = p.token;
        }
        b.lane.held_len = b.depth;
    }
    for (blocks, 0..) |b, bi| readConfidence(d, b.lane, b.depth, bi);
    return null;
}

/// Lane `l`'s drafts' chances from slot `bi` of the last hold's logits: sigmoid(logit + bias); none: left as they are.
fn readConfidence(d: *Drafter, l: *Lane, depth: u32, bi: usize) void {
    if (!d.confident()) return;
    const logits = d.s.conf.slice(f32, d.config.block * ds.pass_rows);
    for (l.conf[0..depth], 0..) |*p, i| p.* = 1 / (1 + @exp(-(logits[i * ds.pass_rows + bi] + d.w.confidence_bias)));
    l.conf_len = depth;
}

/// A finished hold's picks into the lanes' held chains; a nonfinite or invalid pick is refused, as when synchronous.
pub fn readPicks(d: *Drafter, lanes: []const *Lane, depths: []const u32) !void {
    const picks = d.s.picks.slice(Pick, d.config.block * ds.pass_rows);
    var chains: [ds.pass_rows][]u32 = undefined;
    for (lanes, chains[0..lanes.len]) |l, *ch| ch.* = &l.held;
    try copyPicks(picks, chains[0..lanes.len], depths);
    for (lanes, depths, 0..) |l, depth, bi| {
        l.held_len = depth;
        readConfidence(d, l, depth, bi);
    }
}

/// Step i of block bi is picks[i * ds.pass_rows + bi]: each block's first `depth` picks into its chain.
pub fn copyPicks(picks: []const Pick, chains: []const []u32, depths: []const u32) !void {
    for (chains, depths, 0..) |chain, depth, bi| {
        for (chain[0..depth], 0..) |*h, i| {
            const p = picks[i * ds.pass_rows + bi];
            if (p.nonfinite != 0 or p.token >= c.vocab) return error.DraftNonfinite;
            h.* = p.token;
        }
    }
}

/// Each stream's draft i confidence logit (the final norm's row i, W1 of the token before draft i), for readConfidence.
fn confidence(x: Enc, streams: usize, n: u32, depth: u32) void {
    const d = x.d;
    const e = x.e;
    x.pipe("lfm2_dspark_confidence");
    e.setBuffer(d.s.x, 0, 0);
    e.setBuffer(d.w.w1, 0, 1);
    e.setBuffer(d.s.ids, 0, 2);
    e.setBuffer(d.s.picks, 0, 3);
    e.setBuffer(d.w.confidence.?, 0, 4);
    const args = ConfidenceArgs{ .n = n, .pass = ds.pass_rows, .vocab = c.vocab, .rank = d.config.rank, .hidden = c.hidden };
    e.setBytes(std.mem.asBytes(&args), 5);
    e.setBuffer(d.s.conf, 0, 6);
    e.dispatchGroups(mtl.Size.of(depth, streams, 1), mtl.Size.of(256, 1, 1));
}

/// Step i of each stream: its row i's logits plus W2 W1[previous draft] (the pending token first), then the argmax.
fn markov(x: Enc, streams: usize, n: u32, depth: u32) !void {
    const d = x.d;
    const s = &d.s;
    const e = x.e;
    for (0..depth) |i| {
        // step i's previous token: the pending one (each block's first id), then step i - 1's pick
        switch (d.w.w2) {
            .codes => x.pipe("lfm2_markov_scan"),
            .bf16 => x.pipe("lfm2_markov_scan_bf16"),
        }
        e.setBuffer(s.logits, i * c.vocab * 2, 0);
        e.setBuffer(d.w.w1, 0, 1);
        e.setBuffer(switch (d.w.w2) {
            .codes => |m| m.w,
            .bf16 => |m| m.w,
        }, switch (d.w.w2) {
            .codes => |m| m.w_off,
            .bf16 => 0,
        }, 2);
        if (i == 0) e.setBuffer(s.ids, 0, 3) else e.setBuffer(s.picks, (i - 1) * ds.pass_rows * @sizeOf(Pick), 3);
        const args = MarkovArgs{ .stride = n * c.vocab, .rank = d.config.rank, .vocab = c.vocab, .prev_stride = if (i == 0) n else @sizeOf(Pick) / 4 };
        e.setBytes(std.mem.asBytes(&args), 4);
        e.setBuffer(s.best, 0, 5);
        e.setBuffer(s.at, 0, 6);
        if (d.w.w2 == .codes) {
            e.setBuffer(d.w.w2.codes.scales, d.w.w2.codes.s_off, 7);
            e.setBuffer(d.w.w2.codes.biases, d.w.w2.codes.b_off, 8);
        }
        e.dispatchGroups(mtl.Size.of(scan_groups, streams, 1), mtl.Size.of(256, 1, 1));
        x.pipe("lfm2_markov_pick");
        e.setBuffer(s.best, 0, 0);
        e.setBuffer(s.at, 0, 1);
        e.setValue(@as(u32, scan_groups), 2);
        e.setBuffer(s.picks, i * ds.pass_rows * @sizeOf(Pick), 3);
        e.dispatchGroups(mtl.Size.of(streams, 1, 1), mtl.Size.of(256, 1, 1));
    }
}
