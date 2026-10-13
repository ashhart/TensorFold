//! Dense products one input row a simdgroup (0.6.6's rows.qmv_rows) for GPUs without tensor units, at any MLX width.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const Buf = @import("replay.zig").Buf;

/// One matrix in MLX layout: words [n, k bits / 32], bf16 scales and biases [n, k / group].
pub const Product = struct { w: Buf, s: Buf, b: Buf, n: usize, k: usize, bits: u8, group: u16 };

/// Input rows a qmv_rows call takes, one simdgroup each.
pub const rows_a_call = 32;
/// Rows from which other widths than 4/32 take the matrix-unit twin (MMA_FROM), and its simdgroups.
pub const mma_from = 4;
const mma_sg = 8;

/// One compiled library: a shape at a width, its output stride, outputs a simdgroup and how 4-bit dots make nibbles.
pub const Spec = struct { n: usize, k: usize, bits: u8, group: u16, ld: usize, rps: u8, nib: u8 };

/// The library's pipelines: the per-row kernel, and for other widths than 4/32 the block sums and the MMA twin.
pub const Pipes = struct { rows: mtl.Pipeline, sums: ?mtl.Pipeline = null, mma: ?mtl.Pipeline = null };

pub const Cache = std.AutoHashMapUnmanaged(Spec, Pipes);

pub fn q4(p: Product) bool {
    return p.bits == 4 and p.group == 32;
}

/// Values a lane step reads from one row (lane_values).
pub fn vpt(bits: u8) usize {
    return if (bits == 6 or bits == 8) 8 else 16;
}

/// Outputs a simdgroup: 4, else the largest of 2 and 1 that divides n (a row's bits never depend on it).
pub fn rps(n: usize) u8 {
    return if (n % 4 == 0) 4 else if (n % 2 == 0) 2 else 1;
}

/// How 4-bit dots make nibbles at `rows` rows: converts at 1 row, then half (M3, M4) or the exponent trick.
pub fn nib(generation: u32, rows: usize) u8 {
    if (rows < 2 or generation >= 17) return 0;
    return if (generation == 15 or generation == 16) 1 else 2;
}

/// Whether the product takes the matrix-unit twin at `rows` rows.
pub fn mma(p: Product, rows: usize) bool {
    return rows >= mma_from and twinOk(p);
}

/// Shapes the kernels read: 4/32 needs K % 512 and n % 4; other widths K % 16 and 2..8 bits in groups of 32 to 128.
pub fn check(p: Product) !void {
    if (p.n == 0 or p.k % p.group != 0 or p.group % 32 != 0 or p.bits < 2 or p.bits > 8 or p.bits == 7) return error.RowsShape;
    if (q4(p) and (p.k % 512 != 0 or p.n % 4 != 0)) return error.RowsShape;
    if (p.k % 16 != 0) return error.RowsShape;
}

/// The source compiled for `s`: fn_rows.metal after its defines.
pub fn source(a: std.mem.Allocator, s: Spec) ![]const u8 {
    return std.fmt.allocPrint(a, "#define FZ_K {d}\n#define FZ_N {d}\n#define FZ_LD {d}\n#define FZ_BITS {d}\n#define FZ_GS {d}\n#define FZ_RPS {d}\n#define FZ_SG {d}\n#define FZ_NIB {d}\n{s}", .{ s.k, s.n, s.ld, s.bits, s.group, s.rps, mma_sg, s.nib, ks.flashnext_rows });
}

/// The pipelines for `s` on `device`, compiled once and kept in `cache`.
pub fn pipes(a: std.mem.Allocator, device: mtl.Device, cache: *Cache, s: Spec) !Pipes {
    if (cache.get(s)) |p| return p;
    const lib = try mtl.Library.fromSource(device, try source(a, s), mtl.CompileOptions.mlx());
    const four = s.bits == 4 and s.group == 32;
    var p: Pipes = .{ .rows = try mtl.Pipeline.init(device, lib, if (four) "q4_qmv_rows" else "qa_qmv_rows", false) };
    if (!four and s.k % (32 * vpt(s.bits)) == 0) {
        p.sums = try mtl.Pipeline.init(device, lib, "qa_row_block_sums", false);
        p.mma = try mtl.Pipeline.init(device, lib, "qa_qmv_rows_mma", false);
    }
    try cache.put(a, s, p);
    return p;
}

/// The spec `p` compiles under at `rows` rows, writing rows `ld` columns apart.
pub fn specOf(p: Product, generation: u32, rows: usize, ld: usize) Spec {
    return .{ .n = p.n, .k = p.k, .bits = p.bits, .group = p.group, .ld = ld, .rps = if (q4(p)) 4 else rps(p.n), .nib = if (q4(p)) nib(generation, rows) else 0 };
}

/// Bytes of block sums the MMA twin needs at `rows` rows of k inputs (scratch the caller owns).
pub fn sumsBytes(k: usize, bits: u8, rows: usize) usize {
    return rows * (k / vpt(bits)) * 4;
}

/// How a launch runs its rows: by row count (0.6.6's rule), or forced to the per-row loop or the MMA twin.
pub const Mode = enum { auto, loop, twin };

/// The block sums of x's rows at `p`'s lane step into `sums` (the MMA twin's input).
fn encodeSums(enc: mtl.ComputeEncoder, pp: Pipes, p: Product, x: Buf, rows: usize, sums: Buf) void {
    const blocks = p.k / vpt(p.bits);
    enc.setPipeline(pp.sums.?);
    enc.setBuffer(x.b, x.off, 0);
    enc.setBuffer(sums.b, sums.off, 1);
    enc.dispatchThreads(mtl.Size.of((blocks + 31) / 32 * 32, rows, 1), mtl.Size.of(32, 1, 1));
}

/// The product alone, block sums already in `sums` when it takes the MMA twin.
fn encodeProduct(enc: mtl.ComputeEncoder, pp: Pipes, p: Product, s: Spec, x: Buf, rows: usize, y: Buf, sums: ?Buf, twin: bool) !void {
    if (twin) {
        enc.setPipeline(pp.mma orelse return error.RowsNoTwin);
        for ([_]Buf{ x, sums orelse return error.RowsNoScratch, p.w, p.s, p.b, y }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
        enc.setValue(@as(i32, @intCast(rows)), 6);
        enc.dispatchThreads(mtl.Size.of((p.n + 7) / 8 * 32 * mma_sg, (rows + 7) / 8, 1), mtl.Size.of(32 * mma_sg, 1, 1));
        return;
    }
    enc.setPipeline(pp.rows);
    for ([_]Buf{ p.w, p.s, p.b }, 1..) |b, j| enc.setBuffer(b.b, b.off, j);
    var lo: usize = 0;
    while (lo < rows) : (lo += rows_a_call) {
        const m: usize = @min(rows_a_call, rows - lo); // usize: 32 * m must not wrap in @min's narrowed type
        enc.setBuffer(x.b, x.off + lo * p.k * 2, 0);
        enc.setBuffer(y.b, y.off + lo * s.ld * 2, 4);
        enc.dispatchThreads(mtl.Size.of(32 * m, p.n / s.rps, 1), mtl.Size.of(32 * m, 1, 1));
    }
}

/// Whether `p` has an MMA twin: other widths than 4/32 whose K is whole lane steps of 32 lanes.
pub fn twinOk(p: Product) bool {
    return !q4(p) and p.k % (32 * vpt(p.bits)) == 0;
}

fn twinOf(p: Product, rows: usize, mode: Mode) bool {
    return switch (mode) {
        .auto => mma(p, rows),
        .loop => false,
        .twin => twinOk(p),
    };
}

/// y[rows, ld] from its first column = x[rows, k] W^T, every row its one-row bits; `sums` is the MMA scratch.
pub fn encode(enc: mtl.ComputeEncoder, a: std.mem.Allocator, device: mtl.Device, cache: *Cache, p: Product, x: Buf, rows: usize, y: Buf, ld: usize, sums: ?Buf, mode: Mode, concurrent: bool) !void {
    try check(p);
    const s = specOf(p, device.generation(), rows, ld);
    const pp = try pipes(a, device, cache, s);
    const twin = twinOf(p, rows, mode);
    if (twin) {
        encodeSums(enc, pp, p, x, rows, sums orelse return error.RowsNoScratch);
        if (concurrent) enc.barrier();
    }
    try encodeProduct(enc, pp, p, s, x, rows, y, sums, twin);
}

/// A member of a fused projection: its matrix, its first fused column, and whether its MMA twin may run.
pub const Member = struct { p: Product, col: usize, twin: bool = true };

/// What the row kernels of a run share: compiled pipelines and the block-sum scratch (lane steps 8 and 16).
pub const Shared = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    cache: Cache = .empty,
    hc: @import("rows_hc.zig").Cache = .empty,
    hc_tiles: bool = true, // 4/32 hyper-connections take their 8-row tiles (off when the load check finds them inexact)
    sums: [2]Buf,

    /// Rows a launch takes at most (the MTP head's fc_hidden runs four streams of a 16-row window).
    pub const max_rows = 64;
    /// Inputs a row has at most.
    pub const max_k = 6144;
};

/// A fused projection on the row kernels: members in output order, n columns over k inputs.
pub const Dense = struct { members: []Member, n: usize, k: usize, shared: *Shared };

fn slot(bits: u8) usize {
    return if (vpt(bits) == 8) 0 else 1;
}

/// y[rows, d.n] = x[rows, d.k] W^T: the block sums once per lane step the members' MMA twins need, then every member.
pub fn encodeDense(enc: mtl.ComputeEncoder, d: *const Dense, x: Buf, rows: usize, y: Buf, mode: Mode, concurrent: bool) !void {
    const sh = d.shared;
    if (rows > Shared.max_rows or d.k > Shared.max_k) return error.RowsWindow;
    var summed = [2]bool{ false, false };
    for (d.members) |mb| if (mb.twin and twinOf(mb.p, rows, mode) and !summed[slot(mb.p.bits)]) {
        const s = specOf(mb.p, sh.device.generation(), rows, d.n);
        encodeSums(enc, try pipes(sh.a, sh.device, &sh.cache, s), mb.p, x, rows, sh.sums[slot(mb.p.bits)]);
        summed[slot(mb.p.bits)] = true;
    };
    if (concurrent and (summed[0] or summed[1])) enc.barrier();
    for (d.members) |mb| {
        const s = specOf(mb.p, sh.device.generation(), rows, d.n);
        const pp = try pipes(sh.a, sh.device, &sh.cache, s);
        try encodeProduct(enc, pp, mb.p, s, x, rows, y.at(mb.col * 2), sh.sums[slot(mb.p.bits)], mb.twin and twinOf(mb.p, rows, mode));
    }
}
