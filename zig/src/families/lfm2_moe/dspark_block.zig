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
/// Partials a query row takes in the scratch (the kernels' QH x CHUNKS x SUBS).
const part_row = c.query_heads * st.chunks * st.subs;

fn attnArgs(lane: *const Lane, keys: u64) AttnArgs {
    return .{ .len = @intCast(keys - 1), .cap = @intCast(lane.cap), .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .subs = st.subs };
}
const Pick = round.contract.Pick;
const Bf16Args = extern struct { rows: u32, n: u32, k: u32 };
const MarkovArgs = extern struct { stride: u32, rank: u32, vocab: u32, prev_stride: u32 };
const ConfidenceArgs = extern struct { n: u32, pass: u32, vocab: u32, rank: u32, hidden: u32 };
/// Tokens a Markov scan threadgroup takes (the kernel's MARKOV_SPAN), and threadgroups a stream's row takes.
pub const scan_span = 512;
pub const scan_groups = (c.vocab + scan_span - 1) / scan_span;

/// Rows a drafter projection takes on the scalar twin at most (faster there than the window or tensor units).
const scalar_rows = 8;
/// Rows from which the target's head (128,000 outputs) runs on the tensor units, where the GPU has them.
const head_rows = 5;

/// Rows from which `w` runs on the tensor units: outputs of the hidden width or fewer stay on the twin to its rows.
fn naxFrom(w: tile.Matrix) usize {
    if (w.n <= c.hidden) return tile.twin_rows + 1;
    return if (w.n == c.vocab) head_rows else scalar_rows + 1;
}

/// A buffer and byte offset.
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// One stream's block in a hold pass: rows [pending, mask x (block - 1)] at positions anchor .. anchor + block - 1.
pub const Block = struct { lane: *Lane, pending: u32, anchor: u64, depth: u32 };

const Enc = struct {
    d: *Drafter,
    e: mtl.ComputeEncoder,
    bf16_rows_max: usize = 16,
    conc: bool = false, // a concurrent encoder: each step waits for the writes before it at `sync`

    /// Later dispatches see the writes before this point (a no-op on a serial encoder).
    fn sync(x: Enc) void {
        if (x.conc) x.e.barrier();
    }

    fn pipe(x: Enc, comptime key: []const u8) void {
        x.e.setPipeline(x.d.target.kernels.get(key));
    }

    /// y[rows] = x[rows] W^T: the M5's tensor units from naxFrom rows, else the twin (scalar to scalar_rows) or tiles.
    fn project(x: Enc, w: dw.Linear, in: Ref, out: Ref, rows: usize) !void {
        return x.projectAll(&.{w}, in, &.{out}, rows);
    }

    /// y_i[rows] = x[rows] W_i^T on one input and shape: bf16 matrices straight, else one sums pass, products overlap.
    fn projectAll(x: Enc, lins: []const dw.Linear, in: Ref, outs: []const Ref, rows: usize) !void {
        if (lins[0] == .bf16) {
            for (lins, outs) |lin, out| x.bf16(lin.bf16, in, out, rows);
            return;
        }
        var buf: [4]tile.Matrix = undefined;
        for (lins, 0..) |lin, i| buf[i] = lin.codes;
        const ws = buf[0..lins.len];
        const t = &x.d.target.kernels.tiles;
        const sums = Ref{ .buf = x.d.s.sums };
        const k = ws[0].k;
        const nax = x.d.target.kernels.naxPipes(ws[0].bits);
        if (rows >= naxFrom(ws[0])) if (nax) |p| {
            am.rowSums(x.e, p, c.group, in, sums, @intCast(rows), @intCast(k));
            x.sync();
            for (ws, outs) |w, out| {
                const q = .{ .w = Ref{ .buf = w.w, .off = w.w_off }, .s = Ref{ .buf = w.scales, .off = w.s_off }, .b = Ref{ .buf = w.biases, .off = w.b_off }, .n = @as(u32, @intCast(w.n)), .k = @as(u32, @intCast(w.k)) };
                am.dense(x.e, p, in, sums, q, out, @intCast(rows));
            }
            return;
        };
        if (rows <= tile.twin_rows and x.d.target.kernels.twin) {
            // beside the tensor units the scalar twin takes every row count it has; elsewhere the window from 9 rows
            const window: usize = if (nax != null) tile.twin_rows + 1 else scalar_rows + 1;
            for (ws, outs) |w, out| try tile.encodeTwinFrom(x.e, t, w, in.buf, in.off, out.buf, out.off, rows, window);
            return;
        }
        tile.encodeSums(x.e, t, k, in.buf, in.off, x.d.s.sums, rows);
        x.sync();
        for (ws, outs) |w, out| try tile.encode(x.e, t, w, in.buf, in.off, x.d.s.sums, 0, out.buf, out.off, rows);
    }

    /// y[rows] = x[rows] W^T over the released bf16 matrix: one 16-row pass of W, else 64x64 tiles.
    fn bf16(x: Enc, m: dw.Bf16, in: Ref, out: Ref, rows: usize) void {
        const args = Bf16Args{ .rows = @intCast(rows), .n = @intCast(m.n), .k = @intCast(m.k) };
        const few = rows <= x.bf16_rows_max;
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
        x.e.setBuffer(lane.mem, lane.kv(layer, false).off, 7);
        x.e.setBuffer(lane.mem, lane.kv(layer, true).off, 8);
        x.e.dispatchGroups(mtl.Size.of(c.query_heads + c.kv_heads, rows, 1), mtl.Size.of(32, 1, 1));
    }

    /// The chunks' partials of query rows [row, row + rows) against the lane's keys, at partial row `part`.
    fn partials(x: Enc, layer: usize, lane: *const Lane, row: usize, rows: usize, keys: u64, part: usize) void {
        const s = &x.d.s;
        const args = attnArgs(lane, keys);
        const used = (keys + st.chunk_keys - 1) / st.chunk_keys * st.subs;
        const at = part * part_row;
        x.pipe("lfm2_attn_block");
        x.e.setBuffer(s.queries, row * c.hidden * 2, 0);
        x.e.setBuffer(lane.mem, lane.kv(layer, false).off, 1);
        x.e.setBuffer(lane.mem, lane.kv(layer, true).off, 2);
        x.e.setBytes(std.mem.asBytes(&args), 3);
        x.e.setBuffer(s.pm, at * 4, 4);
        x.e.setBuffer(s.pl, at * 4, 5);
        x.e.setBuffer(s.po, at * c.head_dim * 4, 6);
        x.e.setValue(@as(u32, @intCast(rows)), 7);
        x.e.dispatchGroups(mtl.Size.of((rows + 7) / 8, c.kv_heads, @intCast(used)), mtl.Size.of(128, 1, 1));
    }

    /// The partials from partial row `part` merged into the attention's output rows [row, row + rows).
    fn merge(x: Enc, lane: *const Lane, row: usize, rows: usize, keys: u64, part: usize) void {
        const s = &x.d.s;
        const args = attnArgs(lane, keys);
        const at = part * part_row;
        x.pipe("lfm2_attn_block_merge");
        x.e.setBuffer(s.pm, at * 4, 0);
        x.e.setBuffer(s.pl, at * 4, 1);
        x.e.setBuffer(s.po, at * c.head_dim * 4, 2);
        x.e.setBytes(std.mem.asBytes(&args), 3);
        x.e.setBuffer(s.y, row * c.hidden * 2, 4);
        x.e.dispatchGroups(mtl.Size.of(c.query_heads, rows, 1), mtl.Size.of(32, 1, 1));
    }

    /// `rows` hidden-wide bf16 rows copied between buffers on the GPU.
    fn copyRows(x: Enc, from: Ref, to: Ref, rows: usize) void {
        x.pipe("lfm2_tap");
        x.e.setBuffer(from.buf, from.off, 0);
        x.e.setBuffer(to.buf, to.off, 1);
        x.e.setValue([2]u32{ c.hidden, 0 }, 2);
        x.e.dispatchThreads(mtl.Size.of(c.hidden / 8, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// Project pieces into fcx, grouping adjacent small pieces in stable scratch without changing a row's sum.
    fn fc(x: Enc, pieces: []const Piece, gather: ?*usize) !usize {
        const d = x.d;
        const s = &d.s;
        const width = d.tapBytes();
        // A scalar context piece must not become an MMA window just because another lane joins it.
        const scalar = d.w.fc == .codes and d.target.kernels.twin and d.target.kernels.naxPipes(d.w.fc.codes.bits) == null;
        const cap = if (scalar) scalar_rows else @min(x.bf16_rows_max, tile.twin_rows);
        const is_bf16 = d.w.fc == .bf16;
        var rows: usize = 0;
        var at: usize = 0;
        while (at < pieces.len) {
            var until = at + 1;
            var run = pieces[at].rows;
            if (gather != null) while (until < pieces.len and run + pieces[until].rows <= cap) : (until += 1) {
                run += pieces[until].rows;
            };
            const together = until - at > 1 and gather.?.* + run <= ds.gather_rows;
            if (!together) {
                until = at + 1;
                run = pieces[at].rows;
            }
            const out = Ref{ .buf = s.fcx, .off = rows * c.hidden * 2 };
            if (together) {
                const used = gather.?;
                var off = used.* * width;
                for (pieces[at..until]) |pc| {
                    x.copyRows(pc.src, .{ .buf = s.gather, .off = off }, pc.rows * d.config.tap_count);
                    off += pc.rows * width;
                }
                x.sync();
                try x.project(d.w.fc, .{ .buf = s.gather, .off = used.* * width }, out, run);
                used.* += run;
            } else try x.project(d.w.fc, pieces[at].src, out, run);
            if (!is_bf16) x.sync(); // the codes' projections share the sums scratch
            rows += run;
            at = until;
        }
        if (is_bf16) x.sync(); // the calls write their own rows of fcx
        return rows;
    }
};

/// Rows a bf16 call takes on the few-rows kernel in an absorb pass.
const absorb_bf16_rows = 32;
fn failed(msg: [*:0]const u8) anyerror {
    std.log.err("lfm2 dspark: command buffer failed: {s}", .{msg});
    return error.GpuFailed;
}

/// Commits and waits, recording GPU time only for successful command buffers before release.
fn finish(d: *Drafter, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !f64 {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| return failed(msg);
    const ms = cb.gpuSeconds() * 1000;
    d.hold_spent_ms += ms;
    return ms;
}

/// One stream's tapped rows in an absorb pass: `rows` rows at `src`, the context from position `p0`.
pub const Piece = struct { lane: *Lane, src: Ref, rows: usize, p0: u64 };

/// Each piece's rows into its lane's keys and values (fc, hidden_norm, k, v, k_norm, RoPE) in passes on `e`.
pub fn absorbPass(d: *Drafter, e: mtl.ComputeEncoder, pieces: []const Piece) !void {
    return absorbOn(.{ .d = d, .e = e }, pieces, null);
}

/// Project an absorb pass's fc rows; a hold's monotone gather cursor permits batching its small pieces.
pub fn fcRows(d: *Drafter, e: mtl.ComputeEncoder, conc: bool, pieces: []const Piece, gather: ?*usize) !usize {
    var x = Enc{ .d = d, .e = e, .conc = conc };
    x.bf16_rows_max = absorb_bf16_rows;
    return x.fc(pieces, gather);
}

fn absorbOn(encoder: Enc, pieces: []const Piece, gather: ?*usize) !void {
    var x = encoder;
    x.bf16_rows_max = absorb_bf16_rows; // preserve the context cache's original projection arithmetic
    const d = x.d;
    const s = &d.s;
    const rows = try x.fc(pieces, gather);
    if (rows == 0 or rows > ds.absorb_rows) return error.DraftContextGap;
    x.norm(s.fcx, s.fcx, d.w.hidden_norm, false, s.fcx, rows); // in place: a thread reads its values first
    x.sync();
    for (0..d.config.layers) |i| {
        const ly = &d.w.layers[i];
        try x.projectAll(&.{ ly.k, ly.v }, .{ .buf = s.fcx }, &.{ .{ .buf = s.k }, .{ .buf = s.v } }, rows);
        x.sync();
        var at: usize = 0;
        for (pieces) |pc| {
            const q = Ref{ .buf = s.fcx, .off = at * c.hidden * 2 };
            x.qk(q, .{ .buf = s.k, .off = at * kw * 2 }, .{ .buf = s.v, .off = at * kw * 2 }, i, pc.lane, pc.p0, .{ .buf = s.dump }, pc.rows);
            at += pc.rows;
        }
        x.sync();
    }
}

/// A command buffer for absorb passes, and its wait.
pub fn begin(d: *Drafter) struct { cb: mtl.CommandBuffer, e: mtl.ComputeEncoder } {
    const cb = d.target.queue.commandBuffer();
    return .{ .cb = cb, .e = cb.compute(.serial) };
}

/// The passes on `e` committed and waited on; their GPU ms are taken with the holds' (takeHoldMs).
pub fn end(d: *Drafter, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    _ = try finish(d, cb, e);
}

/// Layer `layer`'s q, k, v and every block's attention (a lone block's last layer: `out` rows), on a concurrent `x`.
fn attendBlocks(x: Enc, layer: usize, blocks: []const Block, n: u32, rows: usize, out: usize) !void {
    const s = &x.d.s;
    const ly = &x.d.w.layers[layer];
    // the three projections, every block's keys, then waves of as many blocks as the partials hold at once
    try x.projectAll(&.{ ly.q, ly.k, ly.v }, .{ .buf = s.x }, &.{ .{ .buf = s.q }, .{ .buf = s.k }, .{ .buf = s.v } }, rows);
    x.sync();
    for (blocks, 0..) |b, bi| {
        const at = bi * n;
        x.qk(.{ .buf = s.q, .off = at * c.hidden * 2 }, .{ .buf = s.k, .off = at * kw * 2 }, .{ .buf = s.v, .off = at * kw * 2 }, layer, b.lane, b.anchor, .{ .buf = s.queries, .off = at * c.hidden * 2 }, n);
    }
    x.sync();
    const each = if (blocks.len == 1) out else n;
    const wave = @max(1, x.d.partRows() / n);
    var first: usize = 0;
    while (first < blocks.len) : (first += wave) {
        const last = @min(blocks.len, first + wave);
        for (blocks[first..last], first..) |b, bi| x.partials(layer, b.lane, bi * n, each, b.anchor + n, (bi - first) * n);
        x.sync();
        for (blocks[first..last], first..) |b, bi| x.merge(b.lane, bi * n, each, b.anchor + n, (bi - first) * n);
        x.sync();
    }
}

/// `pieces` absorbed in a command buffer of their own (the encoder closed when encoding fails).
pub fn absorbNow(d: *Drafter, pieces: []const Piece) !void {
    const run = begin(d);
    absorbPass(d, run.e, pieces) catch |err| {
        run.e.end();
        return err;
    };
    return end(d, run.cb, run.e);
}

/// Drains already-settled GPU time once, without submitting, waiting or allocating.
pub fn takeHoldMs(ptr: *anyopaque) f64 {
    const d: *Drafter = @ptrCast(@alignCast(ptr));
    const ms = d.hold_spent_ms;
    d.hold_spent_ms = 0;
    return ms;
}

/// A GPU event a hold signals instead of being waited on, and the value it sets.
pub const Signal = struct { event: mtl.Event, value: u64 };

/// The blocks' staged rows are their lanes' context now: the command buffer that absorbs them was committed.
fn absorbed(blocks: []const Block) void {
    for (blocks) |b| {
        b.lane.end += b.lane.staged;
        b.lane.staged = 0;
    }
}

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
    // one concurrent encoder: independent dispatches overlap, each dependent step waits at a barrier
    const x = Enc{ .d = d, .e = cb.compute(.concurrent), .conc = true };
    var open = true; // close the encoder if encoding fails before commit
    errdefer if (open) x.e.end();
    // the blocks' staged context rows first, in the same command buffer
    var staged: [ds.pass_rows]Piece = undefined;
    var count: usize = 0;
    var held: usize = 0;
    var gathered: usize = 0; // rows of `gather` this hold's absorb calls took
    for (blocks) |b| if (ds.Drafter.stagedPiece(b.lane)) |pc| {
        if (held + pc.rows > ds.absorb_rows) {
            try absorbOn(x, staged[0..count], &gathered);
            count = 0;
            held = 0;
        }
        staged[count] = pc;
        count += 1;
        held += pc.rows;
    };
    if (count > 0) try absorbOn(x, staged[0..count], &gathered);
    const t = d.target;
    const emb = t.weights.embedding;
    x.e.setPipeline(t.kernels.embed(emb.bits));
    x.e.setBuffer(s.ids, 0, 0);
    x.e.setBuffer(emb.w, emb.w_off, 1);
    x.e.setBuffer(emb.scales, emb.s_off, 2);
    x.e.setBuffer(emb.biases, emb.b_off, 3);
    x.e.setBuffer(s.h, 0, 4);
    const units = c.hidden / @as(usize, if (emb.bits == 6) 4 else 8 / emb.bits);
    x.e.dispatchThreads(mtl.Size.of(units, rows, 1), mtl.Size.of(256, 1, 1));
    x.sync();
    for (0..d.config.layers) |i| {
        const ly = &d.w.layers[i];
        const out = if (i + 1 == d.config.layers) tail else rows;
        x.norm(s.h, s.r, ly.input_norm, i > 0, s.x, rows);
        x.sync();
        try attendBlocks(x, i, blocks, n, rows, out);
        try x.project(ly.o, .{ .buf = s.y }, .{ .buf = s.r }, out);
        x.sync();
        x.norm(s.h, s.r, ly.post_norm, true, s.x, out);
        x.sync();
        try x.projectAll(&.{ ly.gate, ly.up }, .{ .buf = s.x }, &.{ .{ .buf = s.gate }, .{ .buf = s.up } }, out);
        x.sync();
        x.pipe("lfm2_swiglu");
        x.e.setBuffer(s.gate, 0, 0);
        x.e.setBuffer(s.up, 0, 1);
        x.e.setBuffer(s.act, 0, 2);
        x.e.setValue(@as(u32, @intCast(out * d.config.intermediate)), 3);
        x.e.dispatchThreads(mtl.Size.of(out * d.config.intermediate / 8, 1, 1), mtl.Size.of(256, 1, 1));
        x.sync();
        try x.project(ly.down, .{ .buf = s.act }, .{ .buf = s.r }, out);
        x.sync();
    }
    x.norm(s.h, s.r, d.w.norm, true, s.x, tail);
    x.sync();
    try x.project(.{ .codes = fwd.matrix(emb) }, .{ .buf = s.x }, .{ .buf = s.logits }, tail);
    x.sync();
    try markov(x, blocks.len, n, steps);
    if (d.confident()) confidence(x, blocks.len, n, steps);
    const e = x.e;
    if (signal) |sg| { // committed unwaited, the command buffer returned retained
        open = false;
        e.end();
        cb.signalEvent(sg.event, sg.value);
        const kept = cb.retain();
        cb.commit();
        absorbed(blocks);
        return kept;
    }
    open = false;
    const ms = try finish(d, cb, e);
    absorbed(blocks);
    d.priceHold(blocks.len, depth, ms); // this group's pass, its staged rows absorbed too, as completed
    const picks = s.picks.slice(Pick, d.config.block * ds.pass_rows);
    for (blocks, 0..) |b, bi| {
        for (b.lane.held[0..b.depth], 0..) |*h, i| {
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
        x.sync();
        x.pipe("lfm2_markov_pick");
        e.setBuffer(s.best, 0, 0);
        e.setBuffer(s.at, 0, 1);
        e.setValue(@as(u32, scan_groups), 2);
        e.setBuffer(s.picks, i * ds.pass_rows * @sizeOf(Pick), 3);
        e.dispatchGroups(mtl.Size.of(streams, 1, 1), mtl.Size.of(256, 1, 1));
        x.sync();
    }
}
