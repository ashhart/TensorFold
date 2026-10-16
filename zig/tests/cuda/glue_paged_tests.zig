//! Paged keys and values: the attention kernels through a page table give the bytes they give over contiguous rows.
const std = @import("std");
const nemotron = @import("nemotron");
const gt = @import("glue_tests.zig");
const glue = nemotron.glue;
const Rig = gt.Rig;
const at = gt.at;

const page = 512; // rows a page holds: one attention chunk

/// Positions and rows: one row, a whole window, page edges and one row either side, far into the window.
const shapes = [_][2]usize{ .{ 0, 1 }, .{ 0, 16 }, .{ 511, 1 }, .{ 511, 2 }, .{ 512, 1 }, .{ 1023, 3 }, .{ 1535, 16 }, .{ 5000, 3 }, .{ 20000, 5 }, .{ 40000, 2 } };

/// One shape: its contiguous planes, its pages' ids in a shared arena, and both runs' outputs.
const Case = struct { kc: u64, vc: u64, ids: []u32, table: u64, meta: u64, qkv: []u16, out: u64, xs: u64, rows: usize, keys: usize };

/// Every shape over contiguous rows and over pages in a shuffled order, alone and in shared windows.
pub fn attentionPaged(r: *Rig) !void {
    const kvd = at.kv_heads * at.dim;
    const qd = at.heads * at.dim;
    const ps = 2 * page * kvd; // a page: one layer's K rows, then its V rows
    var total: usize = 0;
    for (shapes) |c| total += (c[0] + c[1] + page - 1) / page;
    const order = try r.a.alloc(u32, total);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    r.rng.shuffle(u32, order);
    const arena = try r.a.alloc(u16, total * ps);
    for (arena) |*v| v.* = 0x7fc1; // a NaN in every row no case writes: a read of one shows in the output
    var cases: [shapes.len]Case = undefined;
    var next: usize = 0;
    for (shapes, &cases) |c, *x| {
        const keys = c[0] + c[1];
        const kc = try r.bfs(keys * kvd, 1.0);
        const vc = try r.bfs(keys * kvd, 1.0);
        const n = (keys + page - 1) / page;
        x.ids = order[next..][0..n];
        next += n;
        for (0..c[0]) |k| { // the rows before the window, through the table
            const at_ = x.ids[k / page] * ps + (k % page) * kvd;
            @memcpy(arena[at_..][0..kvd], kc[k * kvd ..][0..kvd]);
            @memcpy(arena[at_ + page * kvd ..][0..kvd], vc[k * kvd ..][0..kvd]);
        }
        x.* = .{ .kc = try r.dev(u16, kc), .vc = try r.dev(u16, vc), .ids = x.ids, .table = try r.dev(u32, x.ids), .meta = try r.dev(i32, &.{ @intCast(c[0]), 0, 0, 0 }), .qkv = try r.bfs(c[1] * at.nqkv, 1.0), .out = try r.zeros(u16, c[1] * qd), .xs = try r.zeros(f32, c[1] * qd / 64), .rows = c[1], .keys = keys };
    }
    const pages = try r.dev(u16, arena);
    var paged = gt.ashape();
    paged.ps = ps;
    for (&cases) |*x| { // contiguous rows: the reference
        const q = try r.dev(u16, x.qkv);
        try r.g().kvWrite(q, x.kc, x.vc, x.meta, x.rows, gt.ashape());
        try r.g().attention(q, x.kc, x.vc, x.meta, try r.zeros(f32, x.rows * at.nch * qd), try r.zeros(f32, x.rows * at.nch * at.heads), try r.zeros(f32, x.rows * at.nch * at.heads), x.out, x.xs, x.rows, gt.ashape());
    }
    for (&cases) |*x| { // the same windows through each case's table, one at a time
        const q = try r.dev(u16, x.qkv);
        paged.pt = x.table;
        const out = try r.zeros(u16, x.rows * qd);
        const xs = try r.zeros(f32, x.rows * qd / 64);
        try r.g().kvWrite(q, pages, pages + page * kvd * 2, x.meta, x.rows, paged);
        try r.g().attention(q, pages, pages + page * kvd * 2, x.meta, try r.zeros(f32, x.rows * at.nch * qd), try r.zeros(f32, x.rows * at.nch * at.heads), try r.zeros(f32, x.rows * at.nch * at.heads), out, xs, x.rows, paged);
        try r.same("paged attention out", u16, out, try r.get(u16, x.out, x.rows * qd));
        try r.same("paged attention xs", f32, xs, try r.get(f32, x.xs, x.rows * qd / 64));
    }
    const host = try r.get(u16, pages, total * ps); // the written rows sit where the tables say
    for (&cases) |*x| {
        const want_k = try r.get(u16, x.kc, x.keys * kvd);
        const want_v = try r.get(u16, x.vc, x.keys * kvd);
        const got_k = try r.a.alloc(u16, x.keys * kvd);
        const got_v = try r.a.alloc(u16, x.keys * kvd);
        for (0..x.keys) |k| {
            const at_ = x.ids[k / page] * ps + (k % page) * kvd;
            @memcpy(got_k[k * kvd ..][0..kvd], host[at_..][0..kvd]);
            @memcpy(got_v[k * kvd ..][0..kvd], host[at_ + page * kvd ..][0..kvd]);
        }
        try r.same("paged kv_write k", u16, try r.dev(u16, got_k), want_k);
        try r.same("paged kv_write v", u16, try r.dev(u16, got_v), want_v);
    }
    try shared(r, &cases, pages, ps);
}

/// Shared windows of eight streams, paged and contiguous alternating: each stream's rows as it ran alone.
fn shared(r: *Rig, cases: []const Case, pages: u64, ps: usize) !void {
    const kvd = at.kv_heads * at.dim;
    const qd = at.heads * at.dim;
    for (0..2) |flip| {
        var segs: [glue.max_segs]glue.AttnSeg = undefined;
        var pick: [glue.max_segs]usize = undefined;
        var q: std.ArrayList(u16) = .empty;
        var row0: usize = 0;
        for (&segs, &pick, 0..) |*s, *k, i| {
            k.* = (i * 3 + flip) % cases.len;
            const x = cases[k.*];
            try q.appendSlice(r.a, x.qkv);
            // the keys a window wrote are written again with the same values: every case's planes stay as they were
            s.* = if ((i + flip) % 2 == 0) .{ .kc = pages, .vc = pages + page * kvd * 2, .meta = x.meta, .row0 = row0, .rows = x.rows, .pt = x.table } else .{ .kc = x.kc, .vc = x.vc, .meta = x.meta, .row0 = row0, .rows = x.rows };
            row0 += x.rows;
        }
        var shape = gt.ashape();
        shape.ps = ps;
        const outd = try r.zeros(u16, row0 * qd);
        const xsd = try r.zeros(f32, row0 * qd / 64);
        try r.g().attentionSegs(&segs, try r.dev(u16, q.items), try r.zeros(f32, row0 * at.nch * qd), try r.zeros(f32, row0 * at.nch * at.heads), try r.zeros(f32, row0 * at.nch * at.heads), outd, xsd, shape);
        for (segs, pick) |s, k| {
            const x = cases[k];
            try r.same("paged shared out", u16, outd + s.row0 * qd * 2, try r.get(u16, x.out, x.rows * qd));
            try r.same("paged shared xs", f32, xsd + s.row0 * (qd / 64) * 4, try r.get(f32, x.xs, x.rows * qd / 64));
        }
    }
}
