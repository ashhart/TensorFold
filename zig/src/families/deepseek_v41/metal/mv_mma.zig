//! DeepSeek V4.1 prompt matvecs on simdgroup MMA: bit-equal to mv.zig at >= 16 rows and K >= 32.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Out = enum { f32, bf16 };
pub const Call = struct { k: u32, n: u32, rows: u32, out: Out = .f32 };

const Args = extern struct { k: u32, n: u32, rows: u32 };

pub const row_tile = 16;
pub const col_tile = 16;
pub const threads = 128;

pub const names = [2][:0]const u8{ "tf_dsv41_mv_mma_f32", "tf_dsv41_mv_mma_bf16" };

pub const Pipelines = struct {
    f32: mtl.Pipeline,
    bf16: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.mv_mma);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const f32_pipe = try mtl.Pipeline.init(device, lib, names[0], false);
        errdefer f32_pipe.deinit();
        const bf16_pipe = try mtl.Pipeline.init(device, lib, names[1], false);
        return .{
            .f32 = f32_pipe,
            .bf16 = bf16_pipe,
        };
    }

    pub fn deinit(p: *Pipelines) void {
        p.f32.deinit();
        p.bf16.deinit();
    }
};

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k == 0 or c.n == 0) return error.BadMvMmaShape;
}

/// x [rows][k] bf16, w [n][k] bf16, y [rows][n]; each output equals mv.zig bit for bit.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, c: Call) !void {
    try check(c);
    e.setPipeline(if (c.out == .f32) p.f32 else p.bf16);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(y.buf, y.off, 2);
    e.setValue(Args{ .k = c.k, .n = c.n, .rows = c.rows }, 3);
    e.dispatchGroups(mtl.Size.of((c.n + col_tile - 1) / col_tile, (c.rows + row_tile - 1) / row_tile, 1), mtl.Size.of(threads, 1, 1));
}
