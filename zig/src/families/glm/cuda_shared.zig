//! GLM-5.3-Flash's shared verify windows: several streams' rows in one forward, chains per segment.

const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const kda = @import("cuda_kda.zig");
const latent = @import("cuda_latent.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");

/// Checks a shared window's segments against its `rows` rows, then each sequence holds its rows until its commit.
pub fn begin(r: fw.Rank, rows: usize, stream: cuda.Stream) !void {
    if (r.b.prefill) return error.BadWindow;
    var at: usize = 0;
    for (r.segs) |g| {
        if (g.row0 != at or g.rows == 0 or g.rows > fw.window_rows) return error.BadWindow;
        try g.st.ready(stream, g.rows);
        at += g.rows;
    }
    if (at != rows) return error.BadWindow;
    for (r.segs) |g| {
        g.st.pending = g.rows;
        g.st.prompt = false;
    }
}

/// Forward.mixer on a shared window: all rows' projections, each segment's chain or attention alone, all o_proj.
pub fn mixer(f: *const fw.Forward, r: fw.Rank, l: weights.Layer, li: usize, di: usize, rows: usize, out: u64) !void {
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
                v.p = st.proj + li * st.proj_layer * 2; // the window's rows stay with the layer until its commit
                try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(v.p, b.kproj + g.row0 * w * 2, g.rows * w * 2, f.stream.handle), "cuMemcpyDtoDAsync");
                v.ka += g.row0 * n * 2;
                v.kg += g.row0 * n * 2;
                v.saves.out += g.row0 * n * 2;
                v.saves.state_out = st.recAt(1 - cur, li);
                const kv = li * fw.window_rows * st.heads * kda.dk;
                v.saves.k = st.save_k + kv * 4;
                v.saves.v = st.save_v + kv * 2;
                v.saves.g = st.save_g + kv * 4;
                v.saves.b = st.save_b + li * fw.window_rows * st.heads * 4;
                try kda_block.chainRows(&f.m.kda, f.stream, k, v, g.rows, st.conv + li * 3 * st.conv_c * 2, st.recAt(cur, li), f.kda_wide);
            }
            try kda_block.fill(f.set(), f.stream, k, s, 0, rows, out);
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
            try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, out);
        },
    }
}
