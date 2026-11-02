//! Owned mixed image/text embeddings and rotary coordinates, built without reference tensors.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const c = @import("config.zig");
const processor = @import("processor.zig");
const positions = @import("positions.zig");
const Graph = @import("hip_text.zig").Graph;
const Vision = @import("hip_vision.zig").Vision;
const schema = @import("schema.zig");

pub const Input = struct {
    stream: hip.Stream,
    buffer: hip.DeviceBuffer,
    rows: usize,
    rope: positions.Result,

    pub fn init(a: std.mem.Allocator, graph: *const Graph, ids: []const u32, images: []const processor.Prepared, budget: usize) !Input {
        if (ids.len == 0 or ids.len > c.context) return error.Invalid;
        for (ids) |id| if (id >= c.vocab) return error.Invalid;
        const bytes = ids.len * (c.hidden * 2 + 12);
        if (bytes > budget) return error.GlmOcrInputDoesNotFit;
        const geometry = try a.alloc(processor.Geometry, images.len);
        defer a.free(geometry);
        for (images, geometry) |image, *g| {
            g.* = image.geometry;
            if (g.h == 0 or g.w == 0 or g.h > 16384 or g.w > 16384 or g.h % 28 != 0 or g.w % 28 != 0 or
                g.tokens() > 6144 or image.pixels.len != g.patches() * c.patch_values) return error.Invalid;
            for (image.pixels) |v| if (!std.math.isFinite(v)) return error.Invalid;
            if (image.pixels.len * 2 > budget - bytes) return error.GlmOcrInputDoesNotFit;
        }
        const coordinates = try a.alloc([3]i32, ids.len);
        defer a.free(coordinates);
        const rope = try positions.fill(ids, geometry, coordinates);
        const stream = graph.stream;
        const table = try graph.weights.get(schema.text ++ "embed_tokens.weight");
        var buffer = try hip.DeviceBuffer.alloc(stream.r, bytes);
        errdefer buffer.free();
        errdefer stream.synchronize() catch {};
        try buffer.upload(ids.len * c.hidden * 2, std.mem.sliceAsBytes(coordinates));
        var row: usize = 0;
        var image_index: usize = 0;
        while (row < ids.len) {
            if (ids[row] != c.image_token) {
                try buffer.copyFromAsync(row * c.hidden * 2, table + ids[row] * c.hidden * 2, c.hidden * 2, stream);
                row += 1;
                continue;
            }
            const image = images[image_index];
            image_index += 1;
            const pixels = try a.alloc(u16, image.pixels.len);
            defer a.free(pixels);
            for (pixels, image.pixels) |*v, x| v.* = core.quant.convert.bf16(x);
            var device_pixels = try hip.DeviceBuffer.fromHost(stream.r, std.mem.sliceAsBytes(pixels));
            defer device_pixels.free();
            var vision = try Vision.init(stream, graph.weights, graph.ops, graph.blas, graph.flash, image.geometry.patches(), image.geometry.w / 14, budget - bytes - pixels.len * 2);
            defer vision.deinit();
            try vision.encode(device_pixels.base(), try buffer.address(row * c.hidden * 2), null);
            row += image.geometry.tokens();
        }
        try stream.synchronize();
        return .{ .stream = stream, .buffer = buffer, .rows = ids.len, .rope = rope };
    }
    pub fn embeddings(input: *const Input) u64 {
        return input.buffer.base();
    }
    pub fn rotary(input: *const Input) !u64 {
        return input.buffer.address(input.rows * c.hidden * 2);
    }
    pub fn deinit(input: *Input) void {
        input.stream.synchronize() catch {};
        input.buffer.free();
        input.* = undefined;
    }
};

test "GLM-OCR HIP input: malformed tokens, geometry and budgets fail before device allocation" {
    const graph: Graph = undefined;
    const a = std.testing.allocator;
    try std.testing.expectError(error.Invalid, Input.init(a, &graph, &.{}, &.{}, 0));
    try std.testing.expectError(error.Invalid, Input.init(a, &graph, &.{c.vocab}, &.{}, 4096));
    try std.testing.expectError(error.GlmOcrInputDoesNotFit, Input.init(a, &graph, &.{1}, &.{}, 0));
    try std.testing.expectError(error.Invalid, Input.init(a, &graph, &.{1}, &.{.{ .geometry = .{ .h = 28, .w = 28 }, .pixels = &.{}, .hash = 0 }}, 4096));
    try std.testing.expectError(error.BadImageSpan, Input.init(a, &graph, &.{c.image_token}, &.{}, 4096));
}
