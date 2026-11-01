//! The draft head's passes as kernels alone (their ids staged first), so a graph can hold them on one GPU.
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const window_graph = @import("cuda_window_graph.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const eps = wt.eps;
const fc_sk = mtp.fc_sk;
const pss_n = mtp.pss_n;

/// One piece of forwardHead: the head's layer over `m` rows from `input` (main-model streams), scored when `score`.
pub fn Piece(comptime O: type, comptime P: type) type {
    return struct {
        o: O,
        p: *P,
        d: *mtp.Draft,
        input: u64,
        m: usize,
        score: bool,

        pub fn run(r: @This()) !void {
            const pp = r.p;
            const ps = pp.s;
            const ee = r.o.e;
            const t = r.o.tri;
            const hh = r.o.h;
            const m = r.m;
            pp.pending = null;
            // the embedding branch: the token's row (stream 0 of the four copies), normed, fc_embedding
            if (ee.table.dense) {
                try t.embedDense(ps.ids, ee.table.w.ptr, ps.ple_keys, dims, streams, m);
            } else {
                try t.embed(ps.ids, ee.table.w.ptr, ee.table.s.ptr, ee.table.b.ptr, ps.ple_keys, dims, streams, m);
            }
            try @import("cuda_media.zig").injectNext(r.d.media, ee.stream.*, ps.ple_keys, r.d.len, m);
            try t.rmsnorm(ps.ple_keys, hh.norm_e.ptr, ps.mixed, ps.xsm, eps, wide, m, dims, dims);
            const esk = try pp.lane(ps.mixed, ps.xsm, hh.fc_e, ps.dummy, ps.part, m, fc_sk);
            try pp.sumSlices(ps.part, ps.branch, true, m * dims, esk);
            // the hidden branch: the four streams normed as one row, fc_hidden on each stream
            try t.rmsnorm(r.input, hh.norm_h.ptr, ps.norm, ps.gq, eps, wide, m, wide, wide);
            const hsk = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.dummy, ps.part, streams * m, fc_sk);
            try pp.sumSlices(ps.part, ps.ple_gated, true, streams * m * dims, hsk);
            try t.addStreams(ps.branch, ps.ple_gated, ps.h, m, dims, streams);
            // the layer: attention over the head's own cache, its experts
            try pp.writeback(0, m);
            try pp.readout(hh.layer.attn_hc, 0, m);
            try pp.attnBlock(&hh.layer, 0);
            try t.hcWriteback(ps.h, ps.h, ps.pss, ps.part, "*fp32", ps.inj, ps.dummy, "*bf16", ps.dummy, "*bf16", m * dims, m, dims, streams, 4, 1, 1, pp.out_slices);
            try pp.readout(hh.layer.mlp_hc, 0, m);
            try pp.moeBlock(&hh.layer);
            pp.pending = 0;
            try pp.writeback(0, m);
            try pp.copy(r.d.out.ptr, ps.h + (m - 1) * wide * 2, wide * 2);
            if (!r.score) return;
            // the last row's streams and sums at row 0, where the captured kernels' pointers stay aligned
            if (m > 1) {
                try pp.copy(ps.h, ps.h + (m - 1) * wide * 2, wide * 2);
                try pp.copy(ps.pss, ps.pss + (m - 1) * pss_n * 4, pss_n * 4);
            }
            try pp.readout(hh.mixer, 0, 1);
            try hh.project(pp, ps.mixed, ps.xsm, r.d.dlogits.ptr, 1);
        }
    };
}

/// A piece on one GPU as its graph, keyed by what its kernels read (eagerly with media or tensor parallelism).
pub fn runPiece(one: anytype, score_rows: u64) !void {
    const e = one.o.e;
    const d = one.d;
    const s = one.p.s;
    if (!window_graph.enabled() or d.media != null or e.collective != null) return one.run();
    const key: window_graph.HeadKey = .{ .gen = d.gen, .rows = @intCast(one.m), .geometry = .at(d.len + one.m, d.cap), .scores = one.score, .cap = d.cap, .kc = d.kc.ptr, .input = one.input, .out = d.out.ptr, .dlogits = d.dlogits.ptr, .scratch = s.h, .scratch_cap = s.cap, .score_rows = score_rows };
    try e.graphs.head.run(e.gpa, e.stream.*, key, one);
}

/// A group of lanes' rows in one head pass (headGroup), `total` rows staged: each lane's last row its own.
pub fn Group(comptime O: type, comptime P: type, comptime G: type) type {
    return struct {
        o: O,
        p: *P,
        group: G,
        total: usize,

        pub fn run(r: @This()) !void {
            const pp = r.p;
            const ps = pp.s;
            const ee = r.o.e;
            const t = r.o.tri;
            const hh = r.o.h;
            const tot = r.total;
            pp.pending = null;
            if (ee.table.dense) {
                try t.embedDense(ps.ids, ee.table.w.ptr, ps.ple_keys, dims, streams, tot);
            } else {
                try t.embed(ps.ids, ee.table.w.ptr, ee.table.s.ptr, ee.table.b.ptr, ps.ple_keys, dims, streams, tot);
            }
            try t.rmsnorm(ps.ple_keys, hh.norm_e.ptr, ps.mixed, ps.xsm, eps, wide, tot, dims, dims);
            const esk = try pp.lane(ps.mixed, ps.xsm, hh.fc_e, ps.dummy, ps.part, tot, fc_sk);
            try pp.sumSlices(ps.part, ps.branch, true, tot * dims, esk);
            // each lane's rows of main-model streams, side by side (the head reads them as one block of rows)
            var row0: usize = 0;
            for (r.group.rows, r.group.nexts) |rw, nx| {
                try pp.copy(ps.ple_nrow + row0 * wide * 2, rw, nx.len * wide * 2);
                row0 += nx.len;
            }
            try t.rmsnorm(ps.ple_nrow, hh.norm_h.ptr, ps.norm, ps.gq, eps, wide, tot, wide, wide);
            const hsk = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.dummy, ps.part, streams * tot, fc_sk);
            try pp.sumSlices(ps.part, ps.ple_gated, true, streams * tot * dims, hsk);
            try t.addStreams(ps.branch, ps.ple_gated, ps.h, tot, dims, streams);
            try pp.writeback(0, tot);
            try pp.readout(hh.layer.attn_hc, 0, tot);
            try pp.attnBlock(&hh.layer, 0);
            try t.hcWriteback(ps.h, ps.h, ps.pss, ps.part, "*fp32", ps.inj, ps.dummy, "*bf16", ps.dummy, "*bf16", tot * dims, tot, dims, streams, 4, 1, 1, pp.out_slices);
            try pp.readout(hh.layer.mlp_hc, 0, tot);
            try pp.moeBlock(&hh.layer);
            pp.pending = 0;
            try pp.writeback(0, tot);
            // each lane's last row: its d.out, and gathered to row i for the draft head (row i <= its last row)
            row0 = 0;
            for (r.group.ds, r.group.nexts, 0..) |d, nx, i| {
                const last = row0 + nx.len - 1;
                try pp.copy(d.out.ptr, ps.h + last * wide * 2, wide * 2);
                if (r.group.scores and last != i) {
                    try pp.copy(ps.h + i * wide * 2, ps.h + last * wide * 2, wide * 2);
                    try pp.copy(ps.pss + i * pss_n * 4, ps.pss + last * pss_n * 4, pss_n * 4);
                }
                row0 += nx.len;
            }
            if (!r.group.scores) return;
            const lanes_n = r.group.ds.len;
            try pp.readout(hh.mixer, 0, lanes_n);
            const n_ids = hh.ids.len;
            _ = try pp.lane(ps.mixed, ps.xsm, hh.draft, ps.head_local, ps.dummy, lanes_n, 1);
            for (r.group.ds, 0..) |d, i| try pp.copy(d.dlogits.ptr, ps.head_local + i * n_ids * 2, n_ids * 2);
        }
    };
}

/// A group on one GPU as its graph, keyed by each lane's draft buffers, rows and geometry, then the scratch.
pub fn runGroup(run: anytype, score_rows: u64) !void {
    const e = run.o.e;
    const s = run.p.s;
    if (!window_graph.enabled() or e.collective != null) return run.run();
    var h = std.hash.Wyhash.init(0);
    for (run.group.ds, run.group.nexts, run.group.rows) |d, nx, rw| {
        const g: window_graph.Geometry = .at(d.len + nx.len, d.cap);
        for ([_]u64{ d.gen, d.kc.ptr, d.cap, nx.len, g.chunks, g.score_grid, rw, d.out.ptr, d.dlogits.ptr }) |v| h.update(std.mem.asBytes(&v));
    }
    const key: window_graph.HeadManyKey = .{ .lanes = @intCast(run.group.ds.len), .rows = @intCast(run.total), .scores = run.group.scores, .scratch = s.h, .scratch_cap = s.cap, .score_rows = score_rows, .each = h.final() };
    try e.graphs.head_many.run(e.gpa, e.stream.*, key, run);
}
