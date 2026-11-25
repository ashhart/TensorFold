//! Tensor-unit GEMMs of the image tower (vision_gemm.metal): 64 x 128 x 256 tiles, fp32 sums.
const std = @import("std");
const mtl = @import("metal");

/// A buffer and the byte offset a view starts at.
pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, bytes: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + bytes };
    }
};

/// bf16 x w^T to bf16, bf16 x w^T to fp32 (an epilogue rounds once), fp32 x w^T, fp32 x w.
pub const Kind = enum(u8) { bf16_nt, bf16f_nt, f32_nt, f32_nn };

/// y [M, N] = x [M, K] w, rows `lda` / `ldb` / `ldd` apart ([N, K] w for nt, [K, N] for the fp32 nn), `batch` times.
pub const Shape = struct {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldd: u32,
    batch: u32 = 1,
    sa: u32 = 0,
    sb: u32 = 0,
    sd: u32 = 0,
};

pub const Gemm = struct {
    pipes: [4][2][2]?mtl.Pipeline = @splat(@splat(@splat(null))),
    /// Without tensor units, bf16 x w^T for up to `rows_max` rows on 8x8 MMAs with the tile kernel's bits (null: tiles)
    rows: ?mtl.Pipeline = null,

    /// The pipelines of `wanted` from `lib` (the library holds every kind; unused ones cost nothing to leave out).
    pub fn init(device: mtl.Device, lib: mtl.Library, wanted: []const Kind) !Gemm {
        var g: Gemm = .{};
        errdefer g.deinit();
        for (wanted) |kind| for (0..2) |m| for (0..2) |n| {
            var name: [48]u8 = undefined;
            const text = try std.fmt.bufPrint(&name, "ik2_gemm_{s}_{d}{d}", .{ @tagName(kind), m, n });
            g.pipes[@backingInt(kind)][m][n] = try mtl.Pipeline.init(device, lib, text, false);
        };
        for (wanted) |kind| if (kind == .bf16_nt and !device.tensorUnits()) {
            g.rows = try mtl.Pipeline.init(device, lib, "ik2_gemm_bf16_nt_rows", false);
        };
        return g;
    }

    /// Most rows the small GEMM takes (one 8x8 MMA row block).
    pub const rows_max = 8;

    pub fn deinit(g: *Gemm) void {
        if (g.rows) |p| p.deinit();
        g.rows = null;
        for (&g.pipes) |*a| for (a) |*b| for (b) |*p| if (p.*) |pipe| {
            pipe.deinit();
            p.* = null;
        };
    }

    /// Encodes d = a w for `s` (the kernel's parameter array rides in the encoder's bytes).
    pub fn run(g: *const Gemm, e: mtl.ComputeEncoder, kind: Kind, a: Ref, b: Ref, d: Ref, s: Shape) void {
        if (kind == .f32_nt or kind == .f32_nn) {
            g.runExact(e, kind, a, b, d, s);
            return;
        }
        if (kind == .bf16_nt and s.m <= rows_max and s.batch == 1 and s.n % 16 == 0 and s.k % 8 == 0) if (g.rows) |small| {
            e.setPipeline(small);
            e.setBuffer(a.buf, a.off, 0);
            e.setBuffer(b.buf, b.off, 1);
            const p = [16]i32{ @intCast(s.m), @intCast(s.n), @intCast(s.k), @intCast(s.lda), @intCast(s.ldb), @intCast(s.ldd), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            e.setBytes(std.mem.asBytes(&p), 2);
            e.setBuffer(d.buf, d.off, 3);
            e.dispatchGroups(mtl.Size.of(s.n / 16, 1, 1), mtl.Size.of(32, 1, 1));
            return;
        };
        const tn: u32 = (s.n + 127) / 128;
        const tm: u32 = (s.m + 63) / 64;
        const pipe = g.pipes[@backingInt(kind)][@intFromBool(s.m % 64 == 0)][@intFromBool(s.n % 128 == 0)].?;
        e.setPipeline(pipe);
        e.setBuffer(a.buf, a.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        const p = [16]i32{ @intCast(s.m), @intCast(s.n), @intCast(s.k), @intCast(s.lda), @intCast(s.ldb), @intCast(s.ldd), @intCast(tn), @intCast(tm), 2, @intCast(s.k / 256), @intCast(s.sa), @intCast(s.sb), @intCast(s.sd), 0, 0, 0 };
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(d.buf, d.off, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, @as(usize, s.batch) * 2), mtl.Size.of(32, 4, 2));
    }

    /// One thread, one output, a full fp32 dot. No tensor-unit operand cut.
    fn runExact(g: *const Gemm, e: mtl.ComputeEncoder, kind: Kind, a: Ref, b: Ref, d: Ref, s: Shape) void {
        e.setPipeline(g.pipes[@backingInt(kind)][0][0].?);
        e.setBuffer(a.buf, a.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        const p = [16]i32{ @intCast(s.m), @intCast(s.n), @intCast(s.k), @intCast(s.lda), @intCast(s.ldb), @intCast(s.ldd), 0, 0, 0, 0, @intCast(s.sa), @intCast(s.sb), @intCast(s.sd), 0, 0, 0 };
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(d.buf, d.off, 3);
        if (s.m == 0 or s.n == 0 or s.batch == 0) return;
        e.dispatchThreads(mtl.Size.of(s.n, s.m, s.batch), mtl.Size.of(32, 8, 1));
    }
};
