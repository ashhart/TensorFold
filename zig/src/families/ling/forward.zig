//! Ling-3.1-flash's TP=1 forward on CUDA: embedding, the layer stack, final norm and head, and a window's commit.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("graph.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");
const block_kda = @import("block_kda.zig");
const block_mla = @import("block_mla.zig");
const block_ffn = @import("block_ffn.zig");

const glue = graph.glue;
pub const Seg = block_kda.Seg;
pub const Modules = graph.Modules;
pub const Graph = graph.Graph;
pub const Buffers = state.Buffers;
pub const State = state.State;
pub const window_rows = state.window_rows;
pub const cache_width = state.cache_width;

pub const Forward = struct {
    m: *const graph.Modules,
    g: *const graph.Graph,
    s: cuda.Stream,
    b: *const state.Buffers,
    picks: ?u64 = null, // i32 [MoE layers][rows][top_k + 1]: every MoE layer's picks, for checks
    kda_wide: bool = true, // GLM's TF_GLM_KDA_DECODE_WIDE default: decode windows on the three-kernel chain

    pub fn ops(f: *const Forward) graph.Ops {
        return .{ .m = f.m, .s = f.s, .prompt = f.b.prompt, .part = f.b.part, .kda_wide = f.kda_wide };
    }

    /// Segments in row order from row 0, each on its own committed sequence with room for its rows; windows <= 16 rows.
    fn rowsOf(f: *const Forward, segs: []const Seg) !usize {
        var rows: usize = 0;
        for (segs, 0..) |sg, i| {
            if (sg.row0 != rows or sg.rows == 0 or sg.st.pending != 0) return error.BadSegments;
            if (sg.st.pos + sg.rows > sg.st.capacity) return error.CacheFull; // before a prompt's KDA layers commit
            if (!f.b.prompt and sg.rows > window_rows) return error.WindowTooWide;
            for (segs[0..i]) |o| if (o.st == sg.st) return error.BadSegments;
            rows += sg.rows;
        }
        if (rows == 0 or rows > f.b.rows) return error.TooManyRows;
        return rows;
    }

    /// `ids` (one per row of `segs`) through the embedding, every loaded layer, the final norm and the head.
    pub fn run(f: *const Forward, segs: []const Seg, ids: []const i32) !void {
        const rows = try f.rowsOf(segs);
        if (ids.len != rows) return error.TooManyRows;
        try f.embed(ids);
        try f.stack(segs);
        try f.head(rows);
    }

    /// The tokens' bf16 embedding rows into b.x.
    pub fn embed(f: *const Forward, ids: []const i32) !void {
        const w = f.g.w;
        const d = f.b.arena.buf.d;
        if (ids.len > f.b.rows) return error.TooManyRows;
        for (ids) |t| if (t < 0 or t >= w.config.vocab) return error.BadToken;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(f.b.ids, ids.ptr, ids.len * 4, f.s.handle), "cuMemcpyHtoDAsync");
        try glue.embedB16(f.ops().set(), f.s, f.b.ids, w.embed, f.b.x, ids.len, w.config.hidden, 1);
    }

    /// Every loaded layer on b.x in place; each sequence then holds its rows pending until commit().
    pub fn stack(f: *const Forward, segs: []const Seg) !void {
        const rows = try f.rowsOf(segs);
        var moe: usize = 0;
        for (f.g.w.layers, 0..) |*l, j| {
            const slots: usize = f.g.w.config.topk + 1;
            const p: ?u64 = if (f.picks) |x| x + moe * rows * slots * 4 else null;
            try f.layer(segs, j, rows, p);
            moe += @intFromBool(l.ffn == .moe);
        }
        for (segs) |sg| {
            sg.st.pending = sg.rows;
            sg.st.prompt = f.b.prompt;
        }
    }

    /// Loaded layer j: input norm, mixer, residual; post-attention norm, MLP or MoE, residual.
    pub fn layer(f: *const Forward, segs: []const Seg, j: usize, rows: usize, picks: ?u64) !void {
        const o = f.ops();
        const b = f.b;
        const g = f.g;
        const c = &g.w.config;
        const D: usize = c.hidden;
        const l = g.w.layers[j];
        try glue.rmsnorm(o.set(), o.s, b.x, D, l.attn_norm, b.normed, D, null, c.eps, rows, D);
        switch (l.mixer) {
            .kda => |a| try block_kda.block(o, b, a, c, g.slot[j], segs, rows),
            .mla => |a| try block_mla.block(o, b, a, g.absorb[g.slot[j]], g, g.slot[j], segs, rows),
        }
        try o.add(b.x, b.y, b.x, rows * D);
        try glue.rmsnorm(o.set(), o.s, b.x, D, l.mlp_norm, b.normed, D, null, c.eps, rows, D);
        switch (l.ffn) {
            .dense => |m| try block_ffn.mlp(o, b, m, b.normed, D, rows, 0, b.y),
            .moe => |e| try block_ffn.moe(o, b, e, g, l.index, rows, picks),
        }
        try o.add(b.x, b.y, b.x, rows * D);
    }

    /// The final norm and lm_head of b.x's rows into b.logits (bf16 [rows][vocab]).
    pub fn head(f: *const Forward, rows: usize) !void {
        try f.final(rows);
        try f.logitsOf(0, rows);
    }

    /// The final norm of b.x's rows into b.normed: the rows lm_head and the MTP layer read.
    pub fn final(f: *const Forward, rows: usize) !void {
        const o = f.ops();
        const w = f.g.w;
        const D: usize = w.config.hidden;
        try glue.rmsnorm(o.set(), o.s, f.b.x, D, w.norm, f.b.normed, D, null, w.config.eps, rows, D);
    }

    /// lm_head of b.normed's rows row0 .. row0 + n into b.logits from its first row.
    pub fn logitsOf(f: *const Forward, row0: usize, n: usize) !void {
        const D: usize = f.g.w.config.hidden;
        try f.ops().mm(f.g.w.head orelse return error.TiedHeadNotBuilt, f.b.normed + row0 * D * 2, D, n, f.b.logits);
    }

    /// Keep the first `keep` rows of the sequence's last forward (a prompt keeps them all): KDA states, position.
    pub fn commit(f: *const Forward, st: *state.State, keep: usize) !void {
        if (keep == 0 or keep > st.pending or (st.prompt and keep != st.pending)) return error.BadCommit;
        if (!st.prompt) try block_kda.commit(f.ops(), st, keep);
        st.pending = 0;
        try st.setPos(f.s, st.pos + keep);
    }
};
