//! DeepSeek V4.1 release FP8 dense prompt rows: 8-row MMA tiles and their exact group tree.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Out = enum { f32, bf16 };
pub const Call = struct { k: u32, n: u32, rows: u32, out: Out = .f32 };

const Args = extern struct { K: u32, N: u32, R: u32 };

/// A threadgroup's rows (the source's TF_RS * 8) and threads (TF_RS * 32); it carries 32 columns.
pub const row_tile = 32;
pub const threads = 128;

pub const Pipelines = struct {
    f32: mtl.Pipeline,
    bf16: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.fp8_mma);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{
            .f32 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_mma_f32", false),
            .bf16 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_mma_bf16", false),
        };
    }

    pub fn deinit(p: *Pipelines) void {
        p.f32.deinit();
        p.bf16.deinit();
    }
};

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k == 0 or c.k % 32 != 0 or c.n == 0 or c.n % 32 != 0) return error.BadFp8Shape;
}

/// X [rows][K] bf16, W [N][K] E4M3, S [N/32][K/32] E8M0, Y [rows][N]; each output depends only on its own row and K.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, c: Call) !void {
    try check(c);
    const args = Args{ .K = c.k, .N = c.n, .R = c.rows };
    e.setPipeline(if (c.out == .f32) p.f32 else p.bf16);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(s.buf, s.off, 2);
    e.setBuffer(y.buf, y.off, 3);
    e.setValue(args, 4);
    e.dispatchGroups(mtl.Size.of(c.n / 32, (c.rows + row_tile - 1) / row_tile, 1), mtl.Size.of(threads, 1, 1));
}
