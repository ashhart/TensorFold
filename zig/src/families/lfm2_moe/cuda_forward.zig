//! LFM2.5-24B-A2B's forward on CUDA: a prompt chunk or a round of windows, each row's bits its own (solo == batched).
const std = @import("std");
const cuda = @import("cuda");
const c = @import("config.zig");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const mdl = @import("cuda_model.zig");

const Model = mdl.Model;
const Kernels = kern.Kernels;
const Stream = cuda.Stream;
const tail: cuda.norms.Tail = .{ .round = true }; // the norms' torch points: bf16(w * bf16(h * inv))

/// A stream's rows in one forward: its slot, first position, tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

const Pass = struct {
    m: *const Model,
    k: *const Kernels,
    s: Stream,
    chains: []const Chain,
    rows: usize,
    prompt: bool,

    /// x [rows, k] through `l` into out [rows, n]: a prompt chunk's GEMM, or decode rows on their group sums.
    fn project(p: Pass, l: wts.Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool) !void {
        const k = p.k;
        if (prompt) return switch (l) {
            .a4 => |q| k.affine.prompt(p.s, x, q, out, rows),
            .an => |q| k.wide.prompt(p.s, x, q, out, rows),
        };
        const tile: usize = switch (l) {
            .a4 => k.affine.verifyRows(),
            .an => 16,
        };
        var at: usize = 0;
        while (at < rows) : (at += tile) {
            const n = @min(tile, rows - at);
            const xa = x + at * l.k() * 2;
            const sa = xs + at * (l.k() / 64) * 4;
            const oa = out + at * l.n() * 2;
            switch (l) {
                .a4 => |q| try k.affine.verify(p.s, xa, sa, q, oa, n),
                .an => |q| try k.wide.decode(p.s, xa, sa, q, oa, n),
            }
        }
    }

    /// The 64-value sums of x [rows, width] a decode projection reads (prompt GEMMs take none).
    fn sums(p: Pass, x: u64, width: usize) !void {
        if (!p.prompt) try p.k.act.groupSums(p.s, x, width, p.m.s.xs, p.rows, width);
    }

    /// h = bf16(h + r) (r null: as is), x = bf16(w * bf16(h * inv_rms)) and its sums.
    fn norm(p: Pass, r: ?u64, w: u64) !void {
        const x = p.m.s;
        try p.k.norms.addRmsTail(p.s, x.h, r, w, x.h, x.x, x.xs, p.rows, c.hidden, c.eps, tail);
    }

    fn conv(p: Pass, cv: wts.Conv, index: usize) !void {
        const x = p.m.s;
        try p.project(cv.in, x.x, x.xs, x.wide, p.rows, p.prompt);
        const state = p.m.states + index * 2 * c.hidden * 2;
        const snaps = x.snaps + index * mdl.round_rows * 2 * c.hidden * 2;
        // a prompt chunk keeps its last row's state, a round each row's (a window keeps any prefix of its rows)
        const snap0: usize = if (p.prompt) p.rows - 1 else 0;
        try p.k.convRows(p.s, x.wide, cv.weight, state, mdl.slot_elems, x.slot, x.off, x.y, snaps, snap0, p.rows);
        try p.sums(x.y, c.hidden);
        try p.project(cv.out, x.y, x.xs, x.r, p.rows, p.prompt);
    }

    fn attention(p: Pass, a: wts.Attention, index: usize) !void {
        const x = p.m.s;
        const m = p.m;
        const qkv_w = c.hidden + 2 * mdl.kv_width;
        try p.project(a.qkv, x.x, x.xs, x.wide, p.rows, p.prompt);
        const ld: i64 = @intCast(qkv_w);
        const dh: i64 = c.head_dim;
        const src: cuda.qk_rows.Src = .{ .q = x.wide, .q_ld = ld, .q_hs = dh, .k = x.wide + c.hidden * 2, .k_ld = ld, .k_hs = dh, .v = x.wide + (c.hidden + mdl.kv_width) * 2, .v_ld = ld, .v_hs = dh };
        const shape: cuda.qk_rows.Shape = .{ .norms = a.norms, .norms_bf16 = true, .inv = m.inv, .rot = c.head_dim, .eps = c.eps, .round = true };
        const out: cuda.qk_rows.Out = .{ .q = x.q, .k = x.kv, .v = x.kv + mdl.kv_width * 2, .kv_ld = 2 * mdl.kv_width };
        try p.k.qk.rowsFrom(p.s, src, .{ .h = c.query_heads, .hk = c.kv_heads, .dim = c.head_dim }, shape, x.pos, null, out, p.rows);
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
            const ya = x.y + row0 * c.hidden * 2;
            if (p.prompt) {
                try p.k.prompt(p.s, qa, kc, vc, ya, group[0].p0, rows, m.tableOf(group[0].slot), mdl.page_elems);
            } else {
                const xa = x.xs + row0 * (c.hidden / 64) * 4;
                try p.k.attend(p.s, sg, qa, x.po, x.pm, x.pl, ya, xa, rows, m.chunks(), m.chunkBlocks());
            }
            row0 += rows;
        }
        try p.project(a.out, x.y, x.xs, x.r, p.rows, p.prompt);
    }

    fn dense(p: Pass, dn: wts.Dense) !void {
        const x = p.m.s;
        try p.project(dn.gate_up, x.x, x.xs, x.wide, p.rows, p.prompt);
        try p.k.siluMul(p.s, x.wide, x.y, p.rows, c.intermediate);
        try p.sums(x.y, c.intermediate);
        try p.project(dn.down, x.y, x.xs, x.r, p.rows, p.prompt);
    }

    fn moe(p: Pass, e: wts.Moe) !void {
        const x = p.m.s;
        const k = p.k;
        const pairs = p.rows * c.topk;
        try k.routeRows(p.s, x.x, e.router, e.bias, x.picks, x.wts, null, p.rows, p.m.w.config.scale);
        const tile: usize = if (p.prompt) cuda.experts.promptTile(.affine4) else 16;
        const items = cuda.grouped.maxItems(pairs, c.experts, tile);
        const plan: cuda.grouped.Plan = .{ .members = x.members, .items = x.items, .counts = x.counts, .rank = x.rank, .hist = x.hist };
        try k.plan.route(p.s, x.picks, pairs, c.experts, tile, plan);
        const grouped: cuda.experts.Grouped = .{ .affine4 = &k.experts };
        const rows_in: cuda.experts.Rows = .{ .x = x.x, .stride = c.hidden, .slots = c.topk };
        const act_in: cuda.experts.Rows = .{ .x = x.eact, .stride = c.moe_intermediate };
        if (p.prompt) {
            try grouped.prompt(p.s, .up, rows_in, e.experts, plan, x.eact, items, -1);
            try grouped.prompt(p.s, .down_bf16, act_in, e.experts, plan, x.ey, items, -1);
        } else {
            try grouped.decode(p.s, .up, rows_in, e.experts, plan, x.eact, items, -1);
            try grouped.decode(p.s, .down_f32, act_in, e.experts, plan, x.ey, items, -1);
        }
        try k.moe.combine(p.s, x.ey, !p.prompt, x.wts, x.r, p.rows, c.topk, c.hidden, 0);
    }
};

/// Every chain's rows (a prompt chunk, or windows); logits rows [head0, head0 + heads); conv states end at last rows.
pub fn forward(m: *const Model, k: *const Kernels, s: Stream, chains: []const Chain, prompt: bool, head0: usize, heads: usize) !void {
    var rows: usize = 0;
    for (chains) |ch| {
        if (ch.tokens.len == 0 or ch.slot >= m.slots or ch.p0 + ch.tokens.len > m.context) return error.BadChain;
        if (ch.p0 + ch.tokens.len > m.caps[ch.slot]) return error.PastPages;
        rows += ch.tokens.len;
    }
    if (rows > @as(usize, if (prompt) mdl.prompt_rows else mdl.round_rows) or (prompt and chains.len != 1)) return error.BadWindow;
    if (chains.len > mdl.round_rows or heads > mdl.round_rows or head0 + heads > rows) return error.BadWindow;
    try upload(m, s, chains, rows);
    const p: Pass = .{ .m = m, .k = k, .s = s, .chains = chains, .rows = rows, .prompt = prompt };
    const x = m.s;
    try k.embed.affine(s, x.ids, m.w.embed_w, m.w.embed_s, m.w.embed_b, x.h, rows, c.hidden, .{ .bits = @intCast(m.w.config.embed_bits), .group = c.group });
    var conv_at: usize = 0;
    var attn_at: usize = 0;
    for (m.w.layers, 0..) |l, i| {
        try p.norm(if (i == 0) null else x.r, l.operator_norm);
        switch (l.mixer) {
            .conv => |cv| {
                try p.conv(cv, conv_at);
                conv_at += 1;
            },
            .attention => |a| {
                try p.attention(a, attn_at);
                attn_at += 1;
            },
        }
        try p.norm(x.r, l.ffn_norm);
        switch (l.ffn) {
            .dense => |dn| try p.dense(dn),
            .moe => |e| try p.moe(e),
        }
    }
    try p.norm(x.r, m.w.final_norm);
    try commitAll(m, k, s, chains, prompt);
    if (heads == 0) return;
    try p.project(m.w.head, x.x + head0 * c.hidden * 2, x.xs + head0 * (c.hidden / 64) * 4, x.logits, heads, false);
}

/// Each stream's conv states (slots[i]) set to its state after snapshot row rows[i]; a row < 0 keeps them.
pub fn commit(m: *const Model, k: *const Kernels, s: Stream, slots: []const i32, rows: []const i32) !void {
    if (slots.len != rows.len or slots.len > mdl.round_rows) return error.BadWindow;
    try put(m, s, m.s.commit_slot, std.mem.sliceAsBytes(slots));
    try put(m, s, m.s.commit_row, std.mem.sliceAsBytes(rows));
    try k.commitRows(s, m.states, m.s.snaps, m.s.commit_slot, m.s.commit_row, slots.len, mdl.conv_layers, 2 * c.hidden, mdl.round_rows * 2 * c.hidden, mdl.slot_elems);
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
