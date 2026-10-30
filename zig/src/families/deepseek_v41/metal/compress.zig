//! DeepSeek V4.1's compressor on Metal: the ratio-2 pool, the compressed KV's FP4 round trip, the index-key packing.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = @import("glue.zig").Ref;

pub const Pipelines = struct {
    pool: mtl.Pipeline,
    rt: mtl.Pipeline,
    pack: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.deepseek_v41_compress);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.pool = try mtl.Pipeline.init(device, lib, "tf_dsv41_c_pool2", false);
        p.rt = try mtl.Pipeline.init(device, lib, "tf_dsv41_c_fp4rt", false);
        p.pack = try mtl.Pipeline.init(device, lib, "tf_dsv41_c_fp4pack", false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for ([_]mtl.Pipeline{ p.pool, p.rt, p.pack }) |x| x.deinit();
    }
};

pub const threads = 32;

fn grid(e: mtl.ComputeEncoder, x: usize, y: usize) void {
    e.dispatchThreads(mtl.Size.of(x, y, 1), mtl.Size.of(threads, 1, 1));
}

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// Ratio-2 pool of fp32 kv, score [rows][dim] at pos [rows]; carry `state`: kv[2][dim], score[2][dim]; odd rows to y.
pub fn pool2(p: *const Pipelines, e: mtl.ComputeEncoder, kv: Ref, score: Ref, state: Ref, pos: Ref, y: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.pool);
    bind(e, &.{ kv, score, state, pos, y });
    e.setValue([2]u32{ rows, dim }, 5);
    grid(e, dim, 1);
}

/// FP4 round trip (16-value blocks, E4M3 scales) of `rows` rows of `dim` bf16 into row dest[i] of y; dest < 0 skips.
pub fn fp4rt(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, dest: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.rt);
    bind(e, &.{ x, y, dest });
    e.setValue([3]u32{ dim / 16, dim, dim }, 3);
    grid(e, dim / 16, rows);
}

/// Where fp4pack writes: codes `code_row` bytes a destination row, scales `scale_row` a row from `scale_off`.
pub const Pack = struct { blocks: u32, in_stride: u32, code_row: u32, scale_row: u32, scale_off: u32 = 0, dest: ?Ref = null };

/// fp4_act_quant (E2M1, 32-value blocks, power-of-two scales) of `rows` rows packed into codes and E8M0 scale bytes.
pub fn fp4pack(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, codes: Ref, scales: Ref, rows: u32, k: Pack) void {
    e.setPipeline(p.pack);
    bind(e, &.{ x, codes, scales, k.dest orelse x });
    e.setValue([6]u32{ k.blocks, k.in_stride, k.code_row, k.scale_row, k.scale_off, @intFromBool(k.dest != null) }, 4);
    grid(e, k.blocks, rows);
}
