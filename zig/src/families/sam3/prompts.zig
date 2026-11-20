const std = @import("std");

pub const Point = struct { x: f32, y: f32, positive: bool = true };
pub const Box = struct { x0: f32, y0: f32, x1: f32, y1: f32, positive: bool = true };

pub const Size = struct {
    width: usize,
    height: usize,

    pub fn pixels(self: Size) !usize {
        if (self.width == 0 or self.height == 0) return error.EmptyImage;
        return std.math.mul(usize, self.width, self.height) catch error.ImageTooLarge;
    }

    pub fn point(self: Size, p: Point) !void {
        _ = try self.pixels();
        if (!std.math.isFinite(p.x) or !std.math.isFinite(p.y)) return error.NonFinitePoint;
        if (p.x < 0 or p.y < 0 or p.x >= @as(f32, @floatFromInt(self.width)) or
            p.y >= @as(f32, @floatFromInt(self.height))) return error.PointOutsideImage;
    }

    pub fn box(self: Size, b: Box) !void {
        _ = try self.pixels();
        for ([_]f32{ b.x0, b.y0, b.x1, b.y1 }) |v| if (!std.math.isFinite(v)) return error.NonFiniteBox;
        if (b.x0 < 0 or b.y0 < 0 or b.x1 > @as(f32, @floatFromInt(self.width)) or
            b.y1 > @as(f32, @floatFromInt(self.height)) or b.x0 >= b.x1 or b.y0 >= b.y1)
            return error.InvalidBox;
    }
};

pub const Image = struct {
    size: Size,
    rgb: []const u8,
    row_stride: usize = 0,

    pub fn stride(self: Image) !usize {
        const row = std.math.mul(usize, self.size.width, 3) catch return error.ImageTooLarge;
        if (self.row_stride != 0 and self.row_stride < row) return error.InvalidImageStride;
        return if (self.row_stride == 0) row else self.row_stride;
    }

    pub fn validate(self: Image) !void {
        _ = try self.size.pixels();
        const rows = std.math.mul(usize, self.size.height - 1, try self.stride()) catch return error.ImageTooLarge;
        const row = std.math.mul(usize, self.size.width, 3) catch return error.ImageTooLarge;
        const needed = std.math.add(usize, rows, row) catch return error.ImageTooLarge;
        if (self.rgb.len < needed) return error.ShortImage;
    }
};

pub const Mask = struct {
    size: Size,
    values: []const f32,
    format: enum { probabilities, logits } = .probabilities,

    pub fn validate(self: Mask) !void {
        if (self.values.len != try self.size.pixels()) return error.InvalidMaskShape;
        for (self.values) |value| {
            if (!std.math.isFinite(value)) return error.NonFiniteMask;
            if (self.format == .probabilities and (value < 0 or value > 1)) return error.InvalidMaskProbability;
        }
    }
};

pub const Prompt = struct {
    points: []const Point = &.{},
    boxes: []const Box = &.{},
    mask: ?Mask = null,

    pub fn validate(self: Prompt, size: Size) !void {
        _ = try size.pixels();
        if (self.points.len == 0 and self.boxes.len == 0 and self.mask == null) return error.EmptyPrompt;
        for (self.points) |p| try size.point(p);
        for (self.boxes) |b| try size.box(b);
        if (self.mask) |m| {
            try m.validate();
            if (m.format == .probabilities and (m.size.width != size.width or m.size.height != size.height))
                return error.MaskImageSizeMismatch;
        }
    }
};

pub const Transform = struct {
    source: Size,
    side: usize = 1008,

    pub fn point(self: Transform, p: Point) ![2]f32 {
        try self.source.point(p);
        if (self.side == 0) return error.EmptyImage;
        const side: f32 = @floatFromInt(self.side);
        return .{ p.x / @as(f32, @floatFromInt(self.source.width)) * side, p.y / @as(f32, @floatFromInt(self.source.height)) * side };
    }

    pub fn box(self: Transform, b: Box) ![4]f32 {
        try self.source.box(b);
        if (self.side == 0) return error.EmptyImage;
        const width: f32 = @floatFromInt(self.source.width);
        const height: f32 = @floatFromInt(self.source.height);
        const side: f32 = @floatFromInt(self.side);
        return .{ b.x0 / width * side, b.y0 / height * side, b.x1 / width * side, b.y1 / height * side };
    }
};

test "SAM image views accept row padding and refuse short or overflowing buffers" {
    const padded = Image{ .size = .{ .width = 2, .height = 2 }, .rgb = &@as([14]u8, @splat(0)), .row_stride = 8 };
    try padded.validate();
    try std.testing.expectError(error.ShortImage, (Image{ .size = padded.size, .rgb = padded.rgb[0..13], .row_stride = 8 }).validate());
    try std.testing.expectError(error.InvalidImageStride, (Image{ .size = padded.size, .rgb = padded.rgb, .row_stride = 5 }).validate());
    try std.testing.expectError(error.ImageTooLarge, (Size{ .width = std.math.maxInt(usize), .height = 2 }).pixels());
}

test "SAM source-coordinate prompts preserve labels and non-square geometry" {
    const size = Size{ .width = 960, .height = 540 };
    const t = Transform{ .source = size };
    const p = try t.point(.{ .x = 480, .y = 270, .positive = false });
    try std.testing.expectEqual([2]f32{ 504, 504 }, p);
    const b = try t.box(.{ .x0 = 0, .y0 = 0, .x1 = 960, .y1 = 540 });
    try std.testing.expectEqual([4]f32{ 0, 0, 1008, 1008 }, b);
    try std.testing.expectError(error.PointOutsideImage, size.point(.{ .x = 960, .y = 0 }));
    try std.testing.expectError(error.NonFinitePoint, size.point(.{ .x = std.math.nan(f32), .y = 0 }));
    try std.testing.expectError(error.InvalidBox, size.box(.{ .x0 = 2, .y0 = 0, .x1 = 1, .y1 = 1 }));
    try std.testing.expectError(error.EmptyPrompt, (Prompt{}).validate(size));
    try (Prompt{ .points = &.{.{ .x = 480, .y = 270, .positive = false }} }).validate(size);
}

test "SAM masks refuse nonfinite values and a different image geometry" {
    const one = Size{ .width = 1, .height = 1 };
    try std.testing.expectError(error.NonFiniteMask, (Mask{ .size = one, .values = &.{std.math.inf(f32)} }).validate());
    try std.testing.expectError(error.MaskImageSizeMismatch, (Prompt{ .mask = .{ .size = one, .values = &.{1} } }).validate(.{ .width = 2, .height = 1 }));
}
