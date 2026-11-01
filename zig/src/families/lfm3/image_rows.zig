//! Projected image spans replace placeholder embeddings at absolute prompt positions.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const c = @import("config.zig");
const lanes = @import("lanes");

pub const Provider = struct {
    ctx: *anyopaque,
    /// The image rows a prompt chunk [from, to) reads, null when none overlap it.
    prepare: *const fn (*anyopaque, *const lanes.Stream, usize, usize) anyerror!?Images,
};

pub const Span = struct { at: u32, rows: u32, source: u32 };
pub const Images = struct {
    buffer: mtl.Buffer,
    pipeline: mtl.Pipeline,
    spans: []const Span,

    pub fn encode(images: Images, e: mtl.ComputeEncoder, out: mtl.Buffer, at: usize, rows: usize) !void {
        for (images.spans) |span| {
            const first = @max(at, span.at);
            const end = @min(at + rows, @as(usize, span.at) + span.rows);
            if (first >= end) continue;
            const n = end - first;
            const src = (@as(usize, span.source) + first - span.at) * c.hidden * 2;
            const dst = (first - at) * c.hidden * 2;
            const bytes = n * c.hidden * 2;
            if (src > images.buffer.length() or bytes > images.buffer.length() - src or
                dst > out.length() or bytes > out.length() - dst) return error.BadLfm3ImageRows;
            e.setPipeline(images.pipeline);
            e.setBuffer(images.buffer, src, 0);
            e.setBuffer(out, dst, 1);
            e.setValue(@as(u32, @intCast(n * c.hidden)), 2);
            e.dispatchThreads(mtl.Size.of(n * c.hidden, 1, 1), mtl.Size.of(256, 1, 1));
        }
    }
};

pub fn load(device: mtl.Device) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, sources.lfm3_image_rows, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, "lfm3_image_rows", false);
}
