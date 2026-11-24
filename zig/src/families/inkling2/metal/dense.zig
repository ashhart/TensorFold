//! Prompt projections on simdgroup matrices, for GPUs without tensor units: the bf16 matvec's sums, bit for bit.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// Rows of x [rows][k] against a bf16 weight [n][k] into y [rows][ldy]; n is a multiple of 8 and k of 256.
pub const Call = struct { rows: u32, n: u32, k: u32, ldy: u32 };

/// Under this many rows a launch reads the weights for too few of them: the matvec kernels are as fast.
pub const min_rows: u32 = 16;

const tile = 16;
const threads = 128;

pub const Mma = struct {
    pipe: mtl.Pipeline,

    /// Null where the GPU has tensor units, which take a prompt's projections themselves.
    pub fn load(device: mtl.Device) !?Mma {
        if (device.tensorUnits()) return null;
        return try loadSource(device, src.dense_mma);
    }

    /// The kernel from `text`; the contract's controls pass an edited one.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Mma {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{ .pipe = try mtl.Pipeline.init(device, lib, "tf_ink2_dense_mma", false) };
    }

    pub fn deinit(m: Mma) void {
        m.pipe.deinit();
    }

    /// A call the kernel takes.
    pub fn valid(c: Call) bool {
        return c.rows != 0 and c.n != 0 and c.n % 8 == 0 and c.k != 0 and c.k % 256 == 0 and c.ldy >= c.n;
    }

    /// A call worth the kernel: the matvec kernels serve fewer rows than `min_rows`.
    pub fn serves(c: Call) bool {
        return c.rows >= min_rows and valid(c);
    }

    pub fn launch(m: Mma, e: mtl.ComputeEncoder, x: Ref, w: Ref, out: Ref, c: Call) void {
        std.debug.assert(valid(c));
        for ([_]Ref{ x, w }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setValue(c, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.setPipeline(m.pipe);
        e.dispatchGroups(mtl.Size.of((c.n + tile - 1) / tile, (c.rows + tile - 1) / tile, 1), mtl.Size.of(threads, 1, 1));
    }
};
