//! Borrowed target resources let the existing DFlash2 graph serve another family without another draft runtime.
const mtl = @import("metal");
const core = @import("core");
const projection = @import("../projection.zig");
pub const Target = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    checkpoint: *core.checkpoint_metal.Checkpoint,
    config: struct { hidden: usize, vocab: usize, layers: usize },
    weights: struct { embed: projection.Triple, head: projection.Linear },
    kernels: *@import("../quant_gpu.zig").Kernels,
    glue: *@import("../glue_gpu.zig").Kernels,
    dims: projection.Ref,

    pub fn qwen(m: *@import("../model.zig").Model) Target {
        return .{ .device = m.device, .queue = m.queue, .checkpoint = &m.checkpoint, .config = .{ .hidden = m.config.hidden, .vocab = m.config.vocab, .layers = m.config.layers }, .weights = .{ .embed = m.weights.embed, .head = m.weights.head }, .kernels = &m.kernels, .glue = &m.glue, .dims = m.frame.get(.dims) };
    }
};

pub fn wordView(t: @import("../checkpoint.zig").Tensor) !@import("../checkpoint.zig").Tensor {
    if (t.dtype != .u8) return t;
    if (t.rank != 2 or t.shape[1] % 4 != 0 or @intFromPtr(t.bytes.ptr) % 4 != 0) return error.ProjectionShape;
    var words = t;
    words.dtype = .u32;
    words.shape[1] /= 4;
    return words;
}

test "Flint DFlash2 byte-packed affine embedding borrows the same aligned words" {
    const std = @import("std");
    var bytes: [256]u8 align(4) = @splat(0xa5);
    const t = @import("../checkpoint.zig").Tensor{ .dtype = .u8, .rank = 2, .shape = .{ 2, 128, 1, 1, 1 }, .bytes = &bytes };
    const words = try wordView(t);
    try std.testing.expectEqual(@import("../checkpoint.zig").DType.u32, words.dtype);
    try std.testing.expectEqual(@as(usize, 32), words.shape[1]);
    try std.testing.expectEqual(@intFromPtr(t.bytes.ptr), @intFromPtr(words.bytes.ptr));
    try std.testing.expectEqual(t.bytes.len, words.bytes.len);
    var bad = t;
    bad.shape[1] = 127;
    try std.testing.expectError(error.ProjectionShape, wordView(bad));
}
