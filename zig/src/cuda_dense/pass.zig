//! One dense forward: the engine's operators a recipe's layers call, and the pass from token ids to bf16 logits.
const std = @import("std");
const cuda = @import("cuda");
const recipe = @import("recipe.zig");
const kern = @import("kernels.zig");
const mdl = @import("model.zig");

const Model = mdl.Model;
const Kernels = kern.Kernels;
const Stream = cuda.Stream;
const Weight = cuda.qmmf.Weight;
const tail: cuda.norms.Tail = .{ .round = true }; // the norms' torch points: bf16(w * bf16(h * inv))

/// A stream's rows in one forward: its slot, first position, tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

/// What a forward runs on: the model's state, the shared and the recipe's kernels, the weights, the CUDA stream.
pub fn Engine(comptime R: type) type {
    return struct { m: *const Model, k: *const Kernels, own: *const R.Kernels, w: *const R.Weights, s: Stream };
}

/// A forward's rows (a prompt chunk, or a round of windows) and the operators a recipe's layers call on them.
pub fn Pass(comptime R: type) type {
    return struct {
        const P = @This();
        m: *const Model,
        k: *const Kernels,
        own: *const R.Kernels,
        w: *const R.Weights,
        s: Stream,
        g: recipe.Geometry,
        chains: []const Chain,
        rows: usize,
        prompt: bool,

        fn part(p: *const P, w: Weight) !u64 {
            if (p.k.lane.partBytes(p.rows, w) > p.m.part_bytes) return error.LanePartTooSmall;
            return p.m.s.part;
        }

        /// out [rows, n] bf16 = x [rows, k] times `w`, fp32 sums rounded once.
        pub fn project(p: *const P, w: Weight, x: u64, out: u64) !void {
            try p.k.lane.matmul(p.s, x, w.k, p.rows, w, out, try p.part(w));
        }

        /// out [rows, n] fp32 = x [rows, k] times `w`: the sums a recipe's own kernel rounds.
        pub fn projectF32(p: *const P, w: Weight, x: u64, out: u64) !void {
            try p.k.lane.matmulF32(p.s, x, w.k, p.rows, w, out, try p.part(w));
        }

        /// out [rows, n] = bf16(x times `w` + b): the fp32 sums and the bias rounded once (vit.cu's bias).
        pub fn projectBias(p: *const P, w: Weight, b: u64, x: u64, out: u64) !void {
            if (w.n > p.m.acc_width) return error.AccTooNarrow;
            try p.projectF32(w, x, p.m.s.acc);
            try p.k.vit.bias(p.s, p.m.s.acc, b, out, p.rows, w.n);
        }

        /// h = bf16(h + r) (r null: as is), x = bf16(w * bf16(h * inv_rms)).
        pub fn norm(p: *const P, r: ?u64, w: u64) !void {
            const x = p.m.s;
            try p.k.norms.addRmsTail(p.s, x.h, r, w, x.h, x.x, null, p.rows, p.g.hidden, p.g.eps, tail);
        }

        /// out [rows, f] = bf16(bf16(silu(gate)) * up) from gate|up [rows, 2 f].
        pub fn siluMul(p: *const P, gu: u64, out: u64, f: usize) !void {
            try p.k.siluMul(p.s, gu, out, p.rows, f);
        }

        /// Layer `layer`'s keys and values (kv [rows, 2 kv width]) into the pages, then q's attention into out.
        pub fn attend(p: *const P, layer: usize, q: u64, kv: u64, out: u64) !void {
            if (layer >= p.g.layers) return error.BadLayer;
            const m = p.m;
            const x = m.s;
            const kc = m.cache(layer, false);
            const vc = m.cache(layer, true);
            const qw = p.g.qWidth();
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
                sg.ps = @intCast(m.pageElems());
                sg.n = @intCast(group.len);
                try p.k.kvWrite(p.s, sg, kv + row0 * 2 * p.g.kvWidth() * 2, rows);
                const qa = q + row0 * qw * 2;
                const oa = out + row0 * qw * 2;
                if (p.prompt) {
                    try p.k.prompt(p.s, qa, kc, vc, oa, group[0].p0, rows, m.tableOf(group[0].slot), m.pageElems());
                } else {
                    try p.k.attend(p.s, sg, qa, x.po, x.pm, x.pl, oa, x.xs + row0 * (qw / 64) * 4, rows, m.chunks(), m.chunkBlocks());
                }
                row0 += rows;
            }
        }
    };
}

/// Every chain's rows (a prompt chunk, or windows): the recipe's layers, then logits rows [head0, head0 + heads).
pub fn forward(comptime R: type, e: Engine(R), chains: []const Chain, prompt: bool, head0: usize, heads: usize) !void {
    const m = e.m;
    var rows: usize = 0;
    for (chains) |ch| {
        if (ch.tokens.len == 0 or ch.slot >= m.slots or ch.p0 + ch.tokens.len > m.caps[ch.slot]) return error.BadChain;
        rows += ch.tokens.len;
    }
    if (rows > @as(usize, if (prompt) mdl.prompt_rows else mdl.round_rows) or (prompt and chains.len != 1)) return error.BadWindow;
    if (chains.len > mdl.round_rows or heads > mdl.round_rows or head0 + heads > rows) return error.BadWindow;
    try upload(m, e.s, chains, rows);
    const p: Pass(R) = .{ .m = m, .k = e.k, .own = e.own, .w = e.w, .s = e.s, .g = m.g, .chains = chains, .rows = rows, .prompt = prompt };
    try R.embed(&p);
    for (0..m.g.layers) |i| try R.layer(&p, i);
    try R.final(&p);
    if (heads == 0) return;
    const head: Weight = R.head(e.w);
    const x = m.s.x + head0 * m.g.hidden * 2;
    if (e.k.lane.partBytes(heads, head) > m.part_bytes) return error.LanePartTooSmall;
    try e.k.lane.matmul(e.s, x, m.g.hidden, heads, head, m.s.logits, m.s.part);
}

fn put(m: *const Model, s: Stream, dst: u64, bytes: []const u8) !void {
    try mdl.put(m.d, s, dst, bytes);
}

/// Tokens and positions a row, each chain's first position, staged for the launches.
fn upload(m: *const Model, s: Stream, chains: []const Chain, rows: usize) !void {
    const ids = try m.gpa.alloc(i32, 2 * rows + chains.len);
    defer m.gpa.free(ids);
    var at: usize = 0;
    for (chains, 0..) |ch, j| {
        for (ch.tokens, 0..) |t, i| {
            ids[at + i] = @intCast(t);
            ids[rows + at + i] = @intCast(ch.p0 + i);
        }
        ids[2 * rows + j] = @intCast(ch.p0);
        at += ch.tokens.len;
    }
    const x = m.s;
    try put(m, s, x.ids, std.mem.sliceAsBytes(ids[0..rows]));
    try put(m, s, x.pos, std.mem.sliceAsBytes(ids[rows .. 2 * rows]));
    try put(m, s, x.meta, std.mem.sliceAsBytes(ids[2 * rows ..]));
}
