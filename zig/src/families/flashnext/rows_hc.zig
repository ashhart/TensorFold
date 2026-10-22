//! Hyper-connection projections each row in its own threadgroups (0.6.6's rows._hc_rows) without tensor units.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const rows_w = @import("rows_w.zig");
const Buf = @import("replay.zig").Buf;

const S = 4; // streams
const D = 2560; // dims a stream
const W = S * D;
const KS = W / 32 / 32; // the down projection's input splits

/// One compiled library: down rows and up inputs, both widths, and how 4-bit dots make nibbles.
pub const Spec = struct { nd: usize, low: usize, down: [2]usize, up: [2]usize, nib: u8 };

/// The per-row kernels, and for 4-bit group 32 their 8-row matrix-unit tiles (0.6.6's hc_tiles, the same bits).
pub const Pipes = struct {
    down: mtl.Pipeline,
    up: mtl.Pipeline,
    tiles: ?[2]mtl.Pipeline = null,
    pub fn deinit(p: Pipes) void {
        p.down.deinit();
        p.up.deinit();
        if (p.tiles) |ts| for (ts) |v| v.deinit();
    }
};

/// Rows from which a 4/32 pair takes the tiles (HC_MMA_FROM).
pub const tiles_from = 8;

pub const Cache = std.AutoHashMapUnmanaged(Spec, Pipes);

fn four(w: [2]usize) bool {
    return w[0] == 4 and w[1] == 32;
}

/// Both 4-bit group 32: the original 4-bit dots; else both kernels read their widths (0.6.6's `generic`).
pub fn q4(s: Spec) bool {
    return four(s.down) and four(s.up);
}

/// A hyper-connection's spec at `rows` rows: down and up widths (6/32 when null), rows read off the weights' sizes.
pub fn specOf(dw: Buf, uw: Buf, w: ?[2][2]usize, generation: u32, rows: usize) !Spec {
    const ws = w orelse .{ .{ 6, 32 }, .{ 6, 32 } };
    const nd = dw.b.length() * 8 / (W * ws[0][0]);
    const low = uw.b.length() * 8 / (W * ws[1][0]);
    if (nd == 0 or low == 0 or low % 32 != 0 or nd < low or nd > low + S) return error.RowsHcShape;
    var s: Spec = .{ .nd = nd, .low = low, .down = ws[0], .up = ws[1], .nib = 0 };
    if (q4(s)) s.nib = rows_w.nib(generation, rows);
    return s;
}

/// The source compiled for `s`: fn_rows_hc.metal after its defines.
pub fn source(a: std.mem.Allocator, s: Spec) ![]const u8 {
    const fmt = "#define FZ_S {d}\n#define FZ_D {d}\n#define FZ_ND {d}\n#define FZ_LOW {d}\n" ++
        "#define FZ_KS {d}\n#define FZ_DB {d}\n" ++
        "#define FZ_DG {d}\n#define FZ_UB {d}\n#define FZ_UG {d}\n#define FZ_Q4 {d}\n#define FZ_NIB {d}\n{s}";
    return std.fmt.allocPrint(a, fmt, .{
        S,
        D,
        s.nd,
        s.low,
        KS,
        s.down[0],
        s.down[1],
        s.up[0],
        s.up[1],
        @intFromBool(q4(s)),
        s.nib,
        ks.flashnext_rows_hc,
    });
}

pub fn pipes(a: std.mem.Allocator, device: mtl.Device, cache: *Cache, s: Spec) !Pipes {
    if (cache.get(s)) |p| return p;
    const lib = try mtl.Library.fromSource(device, try source(a, s), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const down = try mtl.Pipeline.init(device, lib, "qa_hc_down_split", false);
    errdefer down.deinit();
    const up = try mtl.Pipeline.init(device, lib, "qa_hc_up2", false);
    errdefer up.deinit();
    var p: Pipes = .{ .down = down, .up = up };
    if (q4(s)) {
        const td = try mtl.Pipeline.init(device, lib, "q4_hc_down_tiles", false);
        errdefer td.deinit();
        const tu = try mtl.Pipeline.init(device, lib, "q4_hc_up_tiles", false);
        p.tiles = .{ td, tu };
    }
    errdefer if (p.tiles) |ts| for (ts) |v| v.deinit();
    try cache.put(a, s, p);
    return p;
}

/// The weights a hyper-connection reads: norm scale, down and up words, scales, biases.
pub const Weights = struct { scale: Buf, dw: Buf, ds: Buf, db: Buf, uw: Buf, us: Buf, ub: Buf, w: ?[2][2]usize };

/// Mixed rows and inject gates from normed streams `hn`; `part` takes the down's fp32 split partials [KS, rows, nd].
pub fn encode(
    enc: mtl.ComputeEncoder,
    a: std.mem.Allocator,
    device: mtl.Device,
    cache: *Cache,
    h: Weights,
    hn: Buf,
    ssp: Buf,
    eps: Buf,
    rows_buf: Buf,
    rows: usize,
    part: Buf,
    mixed: Buf,
    inj: Buf,
    concurrent: bool,
    tiles: bool,
) !void {
    const s = try specOf(h.dw, h.uw, h.w, device.generation(), rows);
    const p = try pipes(a, device, cache, s);
    if (tiles and rows >= tiles_from) if (p.tiles) |tp| { // `tiles`: a 4/32 pair from 8 rows takes the tiles
        enc.setPipeline(tp[0]);
        for (
            [_]Buf{ hn, ssp, h.scale, h.dw, h.ds, h.db, eps, rows_buf, part },
            0..,
        ) |b, j| enc.setBuffer(b.b, b.off, j);
        enc.dispatchThreads(mtl.Size.of((s.nd + 7) / 8 * 256, KS, (rows + 7) / 8), mtl.Size.of(256, 1, 1));
        if (concurrent) enc.barrier();
        enc.setPipeline(tp[1]);
        for (
            [_]Buf{ hn, ssp, h.scale, part, h.uw, h.us, h.ub, eps, rows_buf, mixed, inj },
            0..,
        ) |b, j| enc.setBuffer(b.b, b.off, j);
        enc.dispatchThreads(mtl.Size.of(D / 32 * 32 * S, (rows + 7) / 8, 1), mtl.Size.of(32 * S, 1, 1));
        return;
    };
    enc.setPipeline(p.down);
    for ([_]Buf{ hn, ssp, h.scale, h.dw, h.ds, h.db, eps, rows_buf, part }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
    enc.dispatchThreads(mtl.Size.of((s.nd + 7) / 8 * 256, KS, rows), mtl.Size.of(256, 1, 1));
    if (concurrent) enc.barrier();
    const threads = S * 8 * (s.low / 32);
    enc.setPipeline(p.up);
    for (
        [_]Buf{ hn, ssp, h.scale, part, h.uw, h.us, h.ub, eps, rows_buf, mixed, inj },
        0..,
    ) |b, j| enc.setBuffer(b.b, b.off, j);
    enc.dispatchThreads(mtl.Size.of(D / 8 * threads, rows, 1), mtl.Size.of(threads, 1, 1));
}
