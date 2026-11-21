//! Tensor-unit GEMMs of the image tower and draft head (gemm.metal): 64 x 128 x 256 tiles, fp32 sums.
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

/// bf16 x w^T to bf16, bf16 x w to bf16, bf16 x w^T to fp32 (an epilogue rounds once), fp32 x w^T, fp32 x w.
pub const Kind = enum(u8) { bf16_nt, bf16_nn, bf16f_nt, f32_nt, f32_nn };

/// y [M, N] = x [M, K] w, rows `lda` / `ldb` / `ldd` apart ([N, K] w for nt, [K, N] for nn), `batch` times.
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
    pipes: [5][2][2]?mtl.Pipeline = @splat(@splat(@splat(null))),

    /// The pipelines of `wanted` from `lib` (the library holds every kind; unused ones cost nothing to leave out).
    pub fn init(device: mtl.Device, lib: mtl.Library, wanted: []const Kind) !Gemm {
        var g: Gemm = .{};
        errdefer g.deinit();
        for (wanted) |kind| for (0..2) |m| for (0..2) |n| {
            var name: [48]u8 = undefined;
            const text = try std.fmt.bufPrint(&name, "ik2_gemm_{s}_{d}{d}", .{ @tagName(kind), m, n });
            g.pipes[@backingInt(kind)][m][n] = try mtl.Pipeline.init(device, lib, text, false);
        };
        return g;
    }

    pub fn deinit(g: *Gemm) void {
        for (&g.pipes) |*a| for (a) |*b| for (b) |*p| if (p.*) |pipe| {
            pipe.deinit();
            p.* = null;
        };
    }

    /// Encodes d = a w for `s` (the kernel's parameter array rides in the encoder's bytes).
    pub fn run(g: *const Gemm, e: mtl.ComputeEncoder, kind: Kind, a: Ref, b: Ref, d: Ref, s: Shape) void {
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
};
