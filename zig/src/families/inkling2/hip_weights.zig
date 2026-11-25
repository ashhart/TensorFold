//! Inkling 2 Mini's text checkpoint on one HIP rank, through the loader CUDA shares (weights.zig).
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("config.zig");
const plan_ = @import("plan.zig");
const weights = @import("weights.zig");

pub const Controller = weights.Controller;
pub const Attention = weights.Attention;
pub const Layer = weights.Layer;
/// Per layer: attention, convs, router, `experts` (format; w13, w2 codes, scales, amax, denominator), `controllers`.
pub const Weights = weights.Weights(Buffer);

/// One HIP allocation as the shared loader takes it: `ptr` its device address, host bytes uploaded at an offset.
pub const Buffer = struct {
    buffer: hip.DeviceBuffer,
    ptr: u64,
    len: usize,

    pub fn alloc(r: *const hip.Runtime, len: usize) hip.Error!Buffer {
        const b = try hip.DeviceBuffer.alloc(r, len);
        return .{ .buffer = b, .ptr = b.base(), .len = b.len };
    }

    pub fn upload(b: Buffer, offset: usize, bytes: []const u8) hip.Error!void {
        return b.buffer.upload(offset, bytes);
    }

    pub fn free(b: *Buffer) void {
        b.buffer.free();
        b.* = undefined;
    }
};

/// Load every text tensor into one HIP allocation, as loop_weights.load does on CUDA.
pub fn load(gpa: std.mem.Allocator, r: *const hip.Runtime, ck: *core.Checkpoint, c: *const config.Config, p: plan_.Plan) !Weights {
    return weights.load(Buffer, gpa, r, ck, c, p);
}

/// A core layer's two cross-pass query rows in each matrix, as the loop's messages read them.
pub const Queries = struct { qm: u64, kv: u64 };

/// Rows 2j and 2j + 1 of both cross-pass query matrices, j the layer's place in the core.
pub fn crossQueries(w: *const Weights, layer: u32) error{Inkling2NotCoreLayer}!Queries {
    const lp = w.c.text.loop orelse return error.Inkling2NotCoreLayer;
    if (layer < lp.core_start or layer >= lp.core_end) return error.Inkling2NotCoreLayer;
    const row = 2 * @as(u64, layer - lp.core_start) * w.c.text.hidden * @sizeOf(f32);
    return .{ .qm = w.cross_qm + row, .kv = w.cross_kv + row };
}

test {
    _ = @import("hip_weights_test.zig");
}
