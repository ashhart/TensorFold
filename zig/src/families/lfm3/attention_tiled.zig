//! Development M5 prompt experiment. Broad kernel/model qualification remains open.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub fn load(device: mtl.Device) !?mtl.Pipeline {
    if (!device.tensorUnits()) return null;
    const source = try std.mem.concat(std.heap.c_allocator, u8, &.{ sources.lfm3_attention, "\n", sources.lfm3_attention_tiled });
    defer std.heap.c_allocator.free(source);
    const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer library.deinit();
    const pipeline = try mtl.Pipeline.init(device, library, "lfm3_attn_tiled_bf16", false);
    errdefer pipeline.deinit();
    if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < 128 or
        mtl.objc.msg(usize, pipeline.id, "staticThreadgroupMemoryLength", .{}) != 0)
        return error.TiledAttentionReflection;
    return pipeline;
}

pub fn eligible(rows: usize, window: usize) bool {
    return rows > 8 and rows <= 2048 and (window == 0 or window == 256);
}

pub fn eligiblePrompt(rows: usize, window: usize, allow_tail: bool) bool {
    return eligible(rows, window) or (allow_tail and rows > 0 and rows <= 8 and (window == 0 or window == 256));
}
