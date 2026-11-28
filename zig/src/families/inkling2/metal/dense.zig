//! Prompt projections on simdgroup matrices (pre-M5) and register-tiled outer products (M5+), bit-equal to matvec.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// Rows of x [rows][k] against a bf16 weight [n][k] into y [rows][ldy]; n is a multiple of 8 and k of 256.
pub const Call = struct { rows: u32, n: u32, k: u32, ldy: u32 };

/// Under this many rows a launch reads the weights for too few of them: the matvec kernels are as fast.
pub const min_rows: u32 = 16;

/// Under this many rows on M5+, matvec keeps more threadgroups active with no staging overhead.
pub const tiled_min_rows: u32 = 8;

const tile = 16;
const threads = 128;
pub const tiled_tile: u32 = 16;
pub const tiled_threads: u32 = 256;

pub const Mma = struct {
    pipe: mtl.Pipeline,

    /// Null where the GPU has tensor units, which take register-tiled projections instead.
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

pub const Tiled = struct {
    pipe: mtl.Pipeline,

    /// Loaded where the GPU has tensor units (M5+), where `Mma`'s 32 fragments spill registers.
    pub fn load(device: mtl.Device) !?Tiled {
        if (!device.tensorUnits()) return null;
        return try loadSource(device, src.dense_tiled);
    }

    /// The kernel from `text`; contracts and tests pass edited or unedited source on any GPU.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Tiled {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "tf_ink2_dense_tiled", false);
        errdefer pipe.deinit();
        if (pipe.maxThreads() < tiled_threads) return error.Inkling2Threads;
        return .{ .pipe = pipe };
    }

    pub fn deinit(t: Tiled) void {
        t.pipe.deinit();
    }

    /// A call the kernel takes: n is a multiple of 8 and k of 256.
    pub fn valid(c: Call) bool {
        return Mma.valid(c);
    }

    /// A call worth the kernel on M5+: fewer rows than `tiled_min_rows` take matvec.
    pub fn serves(c: Call) bool {
        return c.rows >= tiled_min_rows and valid(c);
    }

    /// Grid `(row_tiles, col_tiles, 1)` keeps each 16-column weight strip hot in L2 across row tiles.
    pub fn grid(c: Call) mtl.Size {
        return mtl.Size.of((c.rows + tiled_tile - 1) / tiled_tile, (c.n + tiled_tile - 1) / tiled_tile, 1);
    }

    pub fn launch(t: Tiled, e: mtl.ComputeEncoder, x: Ref, w: Ref, out: Ref, c: Call) void {
        encodeLaunch(mtl.ComputeEncoder, t.pipe, e, x, w, out, c);
    }

    /// Encoder-generic launch so host tests verify exact grid, threadgroup, buffer offsets, and arguments.
    pub fn encodeLaunch(comptime Encoder: type, pipe: anytype, e: Encoder, x: Ref, w: Ref, out: Ref, c: Call) void {
        std.debug.assert(valid(c));
        for ([_]Ref{ x, w }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setValue(c, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.setPipeline(pipe);
        e.dispatchGroups(grid(c), mtl.Size.of(tiled_threads, 1, 1));
    }
};
