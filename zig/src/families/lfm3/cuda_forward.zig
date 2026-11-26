//! LFM3's forward on CUDA: a prompt chunk or a round of windows, each row's bits its own (solo == batched).
const std = @import("std");
const cuda = @import("cuda");
const c = @import("config.zig");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const mdl = @import("cuda_model.zig");

const Model = mdl.Model;
const Kernels = kern.Kernels;
const Stream = cuda.Stream;
/// HF's RMSNorm points: bf16(w * bf16(h * inv)), the residual rounded at each add.
const round = true;

/// A stream's rows in one forward: its slot, first position, tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

const Pass = struct {
    m: *const Model,
    k: *const Kernels,
    s: Stream,
    chains: []const Chain,
    rows: usize,
    prompt: bool,

    /// out = x w: FP4 prompt rows on the prompt GEMM, the rest on the lane a round at a time (bits free of chunk cuts).
    fn project(p: Pass, w: wts.Proj, x: u64, out: u64, rows: usize) !void {
        if (w == .affine4) return affine(p.m, p.k, p.s, w.affine4, x, out, rows, p.prompt);
        const lin = p.k.linear();
        const in: cuda.qlinear.Rows = .{ .x = x, .part = p.m.s.part };
        if (p.prompt and w == .nvfp4) return lin.prompt(p.s, w, in, out, rows);
        var at: usize = 0;
        while (at < rows) : (at += mdl.round_rows) {
            const n = @min(mdl.round_rows, rows - at);
            const slice: cuda.qlinear.Rows = .{ .x = x + at * w.inputs() * 2, .part = in.part };
            try lin.decode(p.s, w, slice, out + at * w.outputs() * 2, n);
        }
    }

    /// h += post(block) (null: h is the embedding), x = norm(h).
    fn norm(p: Pass, post: ?u64, w: u64) !void {
        const x = p.m.s;
        const tail: cuda.norms.Tail = .{ .round = round, .branch = post };
        if (post == null) return p.k.norms.addRmsTail(p.s, x.h, null, w, x.h, x.x, null, p.rows, c.hidden, c.eps, tail);
        try p.k.norms.addRmsTail(p.s, x.r, x.h, w, x.h, x.x, null, p.rows, c.hidden, c.eps, tail);
    }

    fn conv(p: Pass, cv: wts.Conv, index: usize) !void {
        const x = p.m.s;
        try p.project(cv.in, x.x, x.wide, p.rows);
        const state = p.m.states + index * mdl.state_rows * c.hidden * 2;
        const snaps = x.snaps + index * mdl.round_rows * mdl.state_rows * c.hidden * 2;
        // a prompt chunk keeps its last row's state, a round each row's (a window keeps any prefix of its rows)
        const snap0: usize = if (p.prompt) p.rows - 1 else 0;
        try p.k.convRows(p.s, x.wide, cv.weight, state, mdl.slot_elems, x.slot, x.off, x.y, snaps, snap0, p.rows);
        try p.project(cv.out, x.y, x.r, p.rows);
    }

    fn attention(p: Pass, a: wts.Attention, full_at: usize, sliding_at: usize) !void {
        const x = p.m.s;
        const m = p.m;
        const qw = c.query_heads * c.head_dim;
        try p.project(a.qkvg, x.x, x.wide, p.rows);
        const ld: i64 = @intCast(a.qkvg.outputs());
        const dh: i64 = c.head_dim;
        const src: cuda.qk_rows.Src = .{ .q = x.wide, .q_ld = ld, .q_hs = dh, .k = x.wide + qw * 2, .k_ld = ld, .k_hs = dh, .v = x.wide + (qw + mdl.kv_width) * 2, .v_ld = ld, .v_hs = dh };
        const shape: cuda.qk_rows.Shape = .{ .norms = a.norms, .norms_bf16 = true, .inv = m.inv[@intFromBool(a.sliding)], .rot = c.head_dim, .eps = c.eps, .round = true };
        var out: cuda.qk_rows.Out = .{ .q = x.q, .k = x.kv, .v = x.kv + mdl.kv_width * 2, .kv_ld = 2 * mdl.kv_width };
        if (a.sliding) {
            out.kc = m.ringOf(sliding_at, false);
            out.vc = m.ringOf(sliding_at, true);
            out.len = mdl.ring;
        }
        try p.k.qk.rowsFrom(p.s, src, .{ .h = c.query_heads, .hk = c.kv_heads, .dim = c.head_dim }, shape, x.pos, x.slot, out, p.rows);
        if (a.sliding) try p.sliding(sliding_at) else try p.full(full_at);
        // HF's gate: bf16(o * bf16(sigmoid(gate_proj(x)))), one scalar a query head; on its own, into the merge's sums
        var gate = x.wide + (qw + 2 * mdl.kv_width) * 2;
        var gate_ld = a.qkvg.outputs();
        if (a.gate) |g| {
            try p.project(g, x.x, x.xs, p.rows);
            gate = x.xs;
            gate_ld = c.query_heads;
        }
        try p.k.act.headGate(p.s, x.att, gate, gate_ld, x.y, null, p.rows, c.query_heads, c.head_dim);
        try p.project(a.out, x.y, x.r, p.rows);
    }

    /// Keys (pos - 256, pos] of each row's slot's ring.
    fn sliding(p: Pass, l: usize) !void {
        const x = p.m.s;
        const m = p.m;
        const w: cuda.window_attention.Window = .{ .h = c.query_heads, .hk = c.kv_heads, .d = c.head_dim, .win = c.sliding_window, .ring = mdl.ring, .scale = kern.scale };
        if (p.prompt) {
            const off = p.chains[0].slot * mdl.ring * mdl.kv_width * 2;
            return p.k.win.prompt(p.s, x.q, m.ringOf(l, false) + off, m.ringOf(l, true) + off, x.att, p.chains[0].p0, p.rows, w);
        }
        try p.k.win.decode(p.s, x.q, m.ringOf(l, false), m.ringOf(l, true), x.att, x.pos, x.slot, w, 0, p.rows);
    }

    /// Every key [0, pos] of the stream's pages.
    fn full(p: Pass, index: usize) !void {
        const x = p.m.s;
        const m = p.m;
        const kc = m.cache(index, false);
        const vc = m.cache(index, true);
        var row0: usize = 0;
        var at: usize = 0;
        while (at < p.chains.len) : (at += kern.max_segs) {
            const group = p.chains[at..@min(p.chains.len, at + kern.max_segs)];
            var sg = std.mem.zeroes(kern.AttnSegs);
            var rows: usize = 0;
            for (group, 0..) |ch, j| {
                sg.kc[j] = kc;
                sg.vc[j] = vc;
                sg.meta[j] = x.meta + (at + j) * 4;
                sg.pt[j] = m.tableOf(ch.slot);
                sg.row0[j] = @intCast(rows);
                rows += ch.tokens.len;
            }
            sg.ps = mdl.page_elems;
            sg.n = @intCast(group.len);
            try p.k.kvWrite(p.s, sg, x.kv + row0 * 2 * mdl.kv_width * 2, rows);
            const qa = x.q + row0 * c.hidden * 2;
            const ya = x.att + row0 * c.hidden * 2;
            if (p.prompt) {
                try p.k.prompt(p.s, qa, kc, vc, ya, group[0].p0, rows, m.tableOf(group[0].slot), mdl.page_elems);
            } else {
                const xa = x.xs + row0 * (c.hidden / 64) * 4;
                try p.k.attend(p.s, sg, qa, x.po, x.pm, x.pl, ya, xa, rows, m.chunks(), m.chunkBlocks());
            }
            row0 += rows;
        }
    }

    /// w2(bf16(silu(w1 x)) * w3 x): w1 | w3 one FP4 projection.
    fn ffn(p: Pass, l: wts.Layer) !void {
        const x = p.m.s;
        try p.project(l.gate_up, x.x, x.wide, p.rows);
        try p.k.siluMul(p.s, x.wide, x.y, p.rows, c.intermediate);
        try p.project(l.down, x.y, x.r, p.rows);
    }
};

/// Every chain's rows (a prompt chunk, or windows); logits rows [head0, head0 + heads); conv states end at last rows.
pub fn forward(m: *const Model, k: *const Kernels, s: Stream, chains: []const Chain, prompt: bool, head0: usize, heads: usize) !void {
    var rows: usize = 0;
    for (chains) |ch| {
        if (ch.tokens.len == 0 or ch.slot >= m.slots or ch.p0 + ch.tokens.len > m.context) return error.BadChain;
        if (ch.p0 + ch.tokens.len > m.caps[ch.slot]) return error.PastPages;
        if (!prompt and ch.tokens.len + c.sliding_window > mdl.ring) return error.BadWindow;
        rows += ch.tokens.len;
    }
    if (rows > @as(usize, if (prompt) mdl.prompt_rows else mdl.round_rows) or (prompt and chains.len != 1)) return error.BadWindow;
    if (chains.len > mdl.round_rows or heads > mdl.round_rows or head0 + heads > rows) return error.BadWindow;
    try upload(m, s, chains, rows);
    const p: Pass = .{ .m = m, .k = k, .s = s, .chains = chains, .rows = rows, .prompt = prompt };
    const x = m.s;
    if (m.w.embed_affine) |e| {
        try k.embed.affine4(s, x.ids, e[0], e[1], e[2], x.h, rows, c.hidden);
    } else try k.embed.bf16(s, x.ids, m.w.embed, x.h, rows, c.hidden);
    var conv_at: usize = 0;
    var full_at: usize = 0;
    var sliding_at: usize = 0;
    var post: ?u64 = null; // the previous FFN's post norm: its output is added before this layer's operator norm
    for (m.w.layers) |l| {
        try p.norm(post, l.operator_norm);
        switch (l.mixer) {
            .conv => |cv| {
                try p.conv(cv, conv_at);
                conv_at += 1;
            },
            .attention => |a| {
                try p.attention(a, full_at, sliding_at);
                if (a.sliding) sliding_at += 1 else full_at += 1;
            },
        }
        try p.norm(l.post_operator_norm, l.ffn_norm);
        try p.ffn(l);
        post = l.post_ffn_norm;
    }
    try p.norm(post, m.w.final_norm);
    try commitAll(m, k, s, chains, prompt);
    if (heads == 0) return;
    const hx = x.x + head0 * c.hidden * 2;
    if (m.w.head == .affine4) return affine(m, k, s, m.w.head.affine4, hx, x.logits, heads, false);
    try k.linear().decode(s, m.w.head, .{ .x = hx, .part = x.part }, x.logits, heads);
}

/// out = x q for MLX affine-4: a prompt chunk's GEMM, or decode rows on their 64-value sums, a verify tile at a time.
fn affine(m: *const Model, k: *const Kernels, s: Stream, q: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize, prompt: bool) !void {
    if (prompt) return k.affine.prompt(s, x, q, out, rows);
    try k.act.groupSums(s, x, q.k, m.s.sums, rows, q.k);
    const tile = k.affine.verifyRows();
    var at: usize = 0;
    while (at < rows) : (at += tile) {
        const n = @min(tile, rows - at);
        try k.affine.verify(s, x + at * q.k * 2, m.s.sums + at * (q.k / 64) * 4, q, out + at * q.n * 2, n);
    }
}

/// Each stream's conv states (slots[i]) set to its state after snapshot row rows[i]; a row < 0 keeps them.
pub fn commit(m: *const Model, k: *const Kernels, s: Stream, slots: []const i32, rows: []const i32) !void {
    if (slots.len != rows.len or slots.len > mdl.round_rows) return error.BadWindow;
    try put(m, s, m.s.commit_slot, std.mem.sliceAsBytes(slots));
    try put(m, s, m.s.commit_row, std.mem.sliceAsBytes(rows));
    const sw = mdl.state_rows * c.hidden;
    try k.commitRows(s, m.states, m.s.snaps, m.s.commit_slot, m.s.commit_row, slots.len, mdl.conv_layers, sw, mdl.round_rows * sw, mdl.slot_elems);
}

/// Each chain's conv states at its last row.
fn commitAll(m: *const Model, k: *const Kernels, s: Stream, chains: []const Chain, prompt: bool) !void {
    var slots: [mdl.round_rows]i32 = undefined;
    var rows: [mdl.round_rows]i32 = undefined;
    var base: usize = 0;
    for (chains, 0..) |ch, i| {
        base += ch.tokens.len;
        slots[i] = @intCast(ch.slot);
        rows[i] = if (prompt) 0 else @intCast(base - 1);
    }
    try commit(m, k, s, slots[0..chains.len], rows[0..chains.len]);
}

fn put(m: *const Model, s: Stream, dst: u64, bytes: []const u8) !void {
    try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
}

/// Tokens, positions, slots and offsets a row, each chain's first position, staged for the launches.
fn upload(m: *const Model, s: Stream, chains: []const Chain, rows: usize) !void {
    const ids = try m.gpa.alloc(i32, 4 * rows + chains.len);
    defer m.gpa.free(ids);
    var at: usize = 0;
    for (chains, 0..) |ch, j| {
        for (ch.tokens, 0..) |t, i| {
            ids[at + i] = @intCast(t);
            ids[rows + at + i] = @intCast(ch.p0 + i);
            ids[2 * rows + at + i] = @intCast(ch.slot);
            ids[3 * rows + at + i] = @intCast(i);
        }
        ids[4 * rows + j] = @intCast(ch.p0);
        at += ch.tokens.len;
    }
    const x = m.s;
    for ([_]u64{ x.ids, x.pos, x.slot, x.off }, 0..) |dst, i| try put(m, s, dst, std.mem.sliceAsBytes(ids[i * rows ..][0..rows]));
    try put(m, s, x.meta, std.mem.sliceAsBytes(ids[4 * rows ..]));
}

test "the forward's launches type-check against the shared operators" {
    std.testing.refAllDecls(Pass);
}
