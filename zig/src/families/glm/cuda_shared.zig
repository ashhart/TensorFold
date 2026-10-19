//! GLM-5.3-Flash's shared windows: several streams' verify windows or whole prompts in one forward, chains per segment.

const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const kda = @import("cuda_kda.zig");
const latent = @import("cuda_latent.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const glue = @import("cuda_glue.zig");
const projection = @import("cuda_proj.zig");

/// Checks a shared window's segments (a prompt buffer's: whole prompts) against its `rows` rows; each holds its own.
pub fn begin(r: fw.Rank, rows: usize, stream: cuda.Stream) !void {
    const most: usize = if (r.b.prefill) fw.prefill_rows else fw.window_rows;
    if (r.b.prefill and r.segs.len > @import("cuda_state.zig").prompt_heads) return error.BadWindow;
    var at: usize = 0;
    for (r.segs) |g| {
        if (g.row0 != at or g.rows == 0 or g.rows > most) return error.BadWindow;
        try g.st.ready(stream, g.rows);
        at += g.rows;
    }
    if (at != rows) return error.BadWindow;
    for (r.segs) |g| {
        g.st.pending = g.rows;
        g.st.prompt = r.b.prefill;
    }
}

/// Forward.finish's head on a prompt buffer: each segment's last final-norm row, logits row k for segment k.
pub fn heads(f: *const fw.Forward, r: fw.Rank) !void {
    const D = r.w.config.hidden;
    const b = r.b;
    for (r.segs, 0..) |g, k| try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(b.normed + k * D * 2, b.fnormed + (g.row0 + g.rows - 1) * D * 2, D * 2, f.stream.handle), "cuMemcpyDtoDAsync");
    try projection.matmul(f.set(), f.stream, r.w.head, b.normed, D, b.logits, b.kda.mm_part, r.segs.len, r.w.share.vocab.len(), D, false);
}

/// Forward.mixer on a shared window: all rows' projections, each segment's chain or attention, o_proj (null: pieced).
pub fn mixer(f: *const fw.Forward, r: fw.Rank, l: weights.Layer, li: usize, di: usize, rows: usize, out: ?u64) !void {
    const c = &r.w.config;
    const b = r.b;
    switch (l.mixer) {
        .kda => |k| {
            const n = k.heads * kda.dk;
            const w = k.width();
            var s = b.kda;
            s.p = b.kproj;
            try kda_block.project(f.set(), f.stream, k, s, b.normed, c.hidden, rows);
            for (r.segs) |g| {
                const st = g.st;
                const cur = st.cur[li];
                var v = s;
                v.ka += g.row0 * n * 2;
                v.kg += g.row0 * n * 2;
                v.saves.out += g.row0 * n * 2;
                v.saves.state_out = st.recAt(1 - cur, li);
                const conv = st.conv + li * 3 * st.conv_c * 2;
                if (b.prefill) { // a whole prompt: its rows in place, the layer committed now (as Forward.mixer)
                    v.p = b.kproj + g.row0 * w * 2;
                    try kda_block.chainRows(&f.m.kda, f.stream, k, v, g.rows, conv, st.recAt(cur, li), false);
                    try glue.convShift(f.set(), f.stream, conv, v.p, g.rows, g.rows * w, w, 1, 3 * n);
                    st.cur[li] = 1 - cur;
                    continue;
                }
                v.p = st.proj + li * st.proj_layer * 2; // the window's rows stay with the layer until its commit
                try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(v.p, b.kproj + g.row0 * w * 2, g.rows * w * 2, f.stream.handle), "cuMemcpyDtoDAsync");
                const kv = li * fw.window_rows * st.heads * kda.dk;
                v.saves.k = st.save_k + kv * 4;
                v.saves.v = st.save_v + kv * 2;
                v.saves.g = st.save_g + kv * 4;
                v.saves.b = st.save_b + li * fw.window_rows * st.heads * 4;
                try kda_block.chainRows(&f.m.kda, f.stream, k, v, g.rows, conv, st.recAt(cur, li), f.kda_wide);
            }
            if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
        },
        .mla => |a| {
            var sparse_any = false;
            for (r.segs) |g| sparse_any = sparse_any or dsa_block.sparseRows(a, g.st.pos, g.rows);
            try dsa_block.prepare(f.set(), f.stream, a, b.dsa, b.normed, c.hidden, rows, sparse_any);
            for (r.segs) |g| {
                const nch = @min(latent.chunksFor(g.st.pos + g.rows), b.nch);
                try dsa_block.attend(.{ .set = f.set(), .ops = f.m.ops() }, f.stream, a, g.st.caches[di], b.dsa, g.row0, g.rows, g.st.pos_dev, g.st.pos, nch, b.part_rows);
            }
            try latent.expandV(f.set(), f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw);
            if (out) |o| try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, o);
        },
    }
}
