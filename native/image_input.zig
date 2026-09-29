const std = @import("std");
const mx = @import("mlx.zig");
const Grid = @import("vision_positions.zig").Grid;
const Ref = *const anyopaque;
const Rect = extern struct { origin: extern struct { x: f64, y: f64 }, size: extern struct { width: f64, height: f64 } };
extern "c" fn CFDataCreate(?Ref, [*]const u8, isize) ?Ref;
extern "c" fn CFRelease(Ref) void;
extern "c" fn CFDictionaryGetValue(Ref, Ref) ?Ref;
extern "c" fn CFNumberGetValue(Ref, c_int, *i32) bool;
extern "c" var kCGImagePropertyOrientation: Ref;
extern "c" fn CGImageSourceCreateWithData(Ref, ?Ref) ?Ref;
extern "c" fn CGImageSourceGetCount(Ref) usize;
extern "c" fn CGImageSourceCopyPropertiesAtIndex(Ref, usize, ?Ref) ?Ref;
extern "c" fn CGImageSourceCreateImageAtIndex(Ref, usize, ?Ref) ?Ref;
extern "c" fn CGImageGetWidth(Ref) usize;
extern "c" fn CGImageGetHeight(Ref) usize;
extern "c" fn CGImageRelease(Ref) void;
extern "c" fn CGColorSpaceCreateDeviceRGB() ?Ref;
extern "c" fn CGColorSpaceRelease(Ref) void;
extern "c" fn CGBitmapContextCreate([*]u8, usize, usize, usize, usize, Ref, u32) ?Ref;
extern "c" fn CGContextRelease(Ref) void;
extern "c" fn CGContextDrawImage(Ref, Rect, Ref) void;
extern "c" fn CGContextSetRGBFillColor(Ref, f64, f64, f64, f64) void;
extern "c" fn CGContextFillRect(Ref, Rect) void;
extern "c" fn tjInitDecompress() ?*anyopaque;
extern "c" fn tjDestroy(*anyopaque) c_int;
extern "c" fn tjDecompressHeader3(*anyopaque, [*]const u8, c_ulong, *c_int, *c_int, *c_int, *c_int) c_int;
extern "c" fn tjDecompress2(*anyopaque, [*]const u8, c_ulong, [*]u8, c_int, c_int, c_int, c_int, c_int) c_int;

fn jpeg(encoded: []const u8, bitmap: []u8, width: usize, height: usize) !void {
    const decoder = tjInitDecompress() orelse return error.ImageDecodeFailed;
    defer _ = tjDestroy(decoder);
    var w: c_int = 0;
    var h: c_int = 0;
    var sampling: c_int = 0;
    var color: c_int = 0;
    if (tjDecompressHeader3(decoder, encoded.ptr, encoded.len, &w, &h, &sampling, &color) != 0 or w != width or h != height) return error.ImageDecodeFailed;
    const cmyk = color == 3 or color == 4;
    // Pillow uses accurate IDCT and fancy chroma upsampling. ImageIO differs.
    if (tjDecompress2(decoder, encoded.ptr, encoded.len, bitmap.ptr, w, @intCast(width * 4), h, if (cmyk) 11 else 2, 4096 | 8192) != 0) return error.ImageDecodeFailed;
    if (cmyk) for (0..width * height) |i| {
        const k: u32 = bitmap[i * 4 + 3];
        for (0..3) |ch| {
            const value = @as(u32, bitmap[i * 4 + ch]) * k + 128;
            bitmap[i * 4 + ch] = @intCast((value + (value >> 8)) >> 8);
        }
    };
}

pub const Image = struct {
    rgb: []u8,
    width: usize,
    height: usize,
    pub fn deinit(image: Image) void {
        mx.allocator.free(image.rgb);
    }
    pub fn decode(encoded: []const u8) !Image {
        if (encoded.len == 0 or encoded.len > 10 * 1024 * 1024) return error.InvalidImageBytes;
        const is_jpeg = std.mem.startsWith(u8, encoded, "\xff\xd8\xff");
        const is_png = std.mem.startsWith(u8, encoded, "\x89PNG\r\n\x1a\n");
        const is_webp = encoded.len >= 12 and std.mem.eql(u8, encoded[0..4], "RIFF") and std.mem.eql(u8, encoded[8..12], "WEBP");
        if (!is_jpeg and !is_png and !is_webp) return error.UnsupportedImageFormat;
        const data = CFDataCreate(null, encoded.ptr, @intCast(encoded.len)) orelse return error.ImageDecodeFailed;
        defer CFRelease(data);
        const source = CGImageSourceCreateWithData(data, null) orelse return error.ImageDecodeFailed;
        defer CFRelease(source);
        if (CGImageSourceGetCount(source) != 1) return error.AnimatedImageNotSupported;
        const image = CGImageSourceCreateImageAtIndex(source, 0, null) orelse return error.ImageDecodeFailed;
        defer CGImageRelease(image);
        const w = CGImageGetWidth(image);
        const h = CGImageGetHeight(image);
        if (w == 0 or h == 0 or @max(w, h) > 8192 or w > 16 * 1024 * 1024 / h or @max(w, h) > 200 * @min(w, h)) return error.InvalidImageDimensions;
        const bitmap = try mx.allocator.alloc(u8, w * h * 4);
        defer mx.allocator.free(bitmap);
        const color = CGColorSpaceCreateDeviceRGB() orelse return error.ImageDecodeFailed;
        defer CGColorSpaceRelease(color);
        // Big-endian RGBX; composite transparency onto white like the upstream loader.
        const context = CGBitmapContextCreate(bitmap.ptr, w, h, 8, w * 4, color, 0x4000 | 5) orelse return error.ImageDecodeFailed;
        defer CGContextRelease(context);
        const rect = Rect{ .origin = .{ .x = 0, .y = 0 }, .size = .{ .width = @floatFromInt(w), .height = @floatFromInt(h) } };
        CGContextSetRGBFillColor(context, 1, 1, 1, 1);
        CGContextFillRect(context, rect);
        CGContextDrawImage(context, rect, image);
        if (is_jpeg) try jpeg(encoded, bitmap, w, h);
        var orientation: i32 = 1;
        if (CGImageSourceCopyPropertiesAtIndex(source, 0, null)) |properties| {
            defer CFRelease(properties);
            if (CFDictionaryGetValue(properties, kCGImagePropertyOrientation)) |value| _ = CFNumberGetValue(value, 3, &orientation);
        }
        const swapped = orientation >= 5 and orientation <= 8;
        const width = if (swapped) h else w;
        const height = if (swapped) w else h;
        const rgb = try mx.allocator.alloc(u8, w * h * 3);
        for (0..height) |y| for (0..width) |x| {
            const source_xy: [2]usize = switch (orientation) {
                2 => .{ w - 1 - x, y },
                3 => .{ w - 1 - x, h - 1 - y },
                4 => .{ x, h - 1 - y },
                5 => .{ y, x },
                6 => .{ y, h - 1 - x },
                7 => .{ w - 1 - y, h - 1 - x },
                8 => .{ w - 1 - y, x },
                else => .{ x, y },
            };
            const offset = (source_xy[1] * w + source_xy[0]) * 4;
            @memcpy(rgb[(y * width + x) * 3 ..][0..3], bitmap[offset..][0..3]);
        };
        return .{ .rgb = rgb, .width = width, .height = height };
    }
};

fn roundEven(x: f64) f64 {
    const floor = @floor(x);
    return if (x - floor < 0.5 or (x - floor == 0.5 and @mod(floor, 2) == 0)) floor else floor + 1;
}
pub fn resizeGrid(h: usize, w: usize, min_pixels: usize, max_pixels: usize) !Grid {
    if (h == 0 or w == 0 or min_pixels < 1024 or max_pixels < min_pixels or max_pixels > 4096 * 1024) return error.InvalidImageDimensions;
    const fh: f64 = @floatFromInt(h);
    const fw: f64 = @floatFromInt(w);
    const low: f64 = @floatFromInt(min_pixels);
    const high: f64 = @floatFromInt(max_pixels);
    var rh = roundEven(fh / 32) * 32;
    var rw = roundEven(fw / 32) * 32;
    if (rh * rw > high) {
        const beta = @sqrt(fh * fw / high);
        rh = @max(32, @floor(fh / beta / 32) * 32);
        rw = @max(32, @floor(fw / beta / 32) * 32);
    } else if (rh * rw < low) {
        const beta = @sqrt(low / (fh * fw));
        rh = @ceil(fh * beta / 32) * 32;
        rw = @ceil(fw * beta / 32) * 32;
    }
    if (rh * rw > high) return error.VisualTokenLimitExceeded;
    return .{ .height = @intFromFloat(rh / 16), .width = @intFromFloat(rw / 16) };
}
const Coefficients = struct {
    starts: []usize,
    counts: []usize,
    weights: []i32,
    stride: usize,
    fn deinit(c: Coefficients) void {
        mx.allocator.free(c.starts);
        mx.allocator.free(c.counts);
        mx.allocator.free(c.weights);
    }
    fn init(input: usize, output: usize) !Coefficients {
        const scale = @as(f64, @floatFromInt(input)) / @as(f64, @floatFromInt(output));
        const filter = @max(scale, 1);
        const support = 2 * filter;
        const stride: usize = @intFromFloat(@ceil(support) * 2 + 1);
        const starts = try mx.allocator.alloc(usize, output);
        errdefer mx.allocator.free(starts);
        const counts = try mx.allocator.alloc(usize, output);
        errdefer mx.allocator.free(counts);
        const weights = try mx.allocator.alloc(i32, output * stride);
        errdefer mx.allocator.free(weights);
        const taps = try mx.allocator.alloc(f64, stride);
        defer mx.allocator.free(taps);
        for (0..output) |i| {
            const center = (@as(f64, @floatFromInt(i)) + 0.5) * scale;
            const start: usize = @intFromFloat(@max(0, @trunc(center - support + 0.5)));
            const end: usize = @min(input, @as(usize, @intFromFloat(@max(0, @trunc(center + support + 0.5)))));
            starts[i] = start;
            counts[i] = end - start;
            var total: f64 = 0;
            for (0..end - start) |j| {
                const x = @abs((@as(f64, @floatFromInt(start + j)) + 0.5 - center) / filter);
                taps[j] = if (x < 1) ((1.5 * x - 2.5) * x) * x + 1 else if (x < 2) ((-0.5 * x + 2.5) * x - 4) * x + 2 else 0;
                total += taps[j];
            }
            for (0..end - start) |j| {
                const value = taps[j] / total * 4194304;
                weights[i * stride + j] = @intFromFloat(value + if (value < 0) @as(f64, -0.5) else 0.5);
            }
        }
        return .{ .starts = starts, .counts = counts, .weights = weights, .stride = stride };
    }
};

pub fn patches(image: Image, grid: Grid) ![]f32 {
    const count = try grid.count();
    if (count > 4096 or image.rgb.len != image.width * image.height * 3) return error.InvalidImageDimensions;
    const h: usize = @intCast(grid.height * 16);
    const w: usize = @intCast(grid.width * 16);
    const horizontal = try mx.allocator.alloc(u8, image.height * w * 3);
    defer mx.allocator.free(horizontal);
    const resized = try mx.allocator.alloc(u8, h * w * 3);
    defer mx.allocator.free(resized);
    const xc = try Coefficients.init(image.width, w);
    defer xc.deinit();
    const yc = try Coefficients.init(image.height, h);
    defer yc.deinit();
    for (0..image.height) |y| for (0..w) |x| for (0..3) |ch| {
        var sum: i64 = 2097152;
        for (0..xc.counts[x]) |j| sum += @as(i64, image.rgb[(y * image.width + xc.starts[x] + j) * 3 + ch]) * xc.weights[x * xc.stride + j];
        horizontal[(y * w + x) * 3 + ch] = @intCast(std.math.clamp(sum >> 22, 0, 255));
    };
    for (0..h) |y| for (0..w) |x| for (0..3) |ch| {
        var sum: i64 = 2097152;
        for (0..yc.counts[y]) |j| sum += @as(i64, horizontal[((yc.starts[y] + j) * w + x) * 3 + ch]) * yc.weights[y * yc.stride + j];
        resized[(y * w + x) * 3 + ch] = @intCast(std.math.clamp(sum >> 22, 0, 255));
    };
    const out = try mx.allocator.alloc(f32, count * 4 * 1536);
    var index: usize = 0;
    for (0..h / 32) |by| for (0..w / 32) |bx| for (0..2) |dy| for (0..2) |dx| for (0..3) |ch| for (0..2) |_| for (0..16) |py| for (0..16) |px| {
        const value = resized[(((by * 2 + dy) * 16 + py) * w + (bx * 2 + dx) * 16 + px) * 3 + ch];
        // transformers rescales in float64, then normalizes in float32.
        const scaled: f32 = @floatCast(@as(f64, @floatFromInt(value)) * (1.0 / 255.0));
        out[index] = (scaled - 0.5) / 0.5;
        index += 1;
    };
    return out;
}

test "image grid rounding and visual budgets" {
    try std.testing.expectEqual(Grid{ .height = 16, .width = 16 }, try resizeGrid(256, 256, 65536, 262144));
    try std.testing.expectEqual(@as(f64, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f64, 4), roundEven(3.5));
    try std.testing.expectError(error.InvalidImageDimensions, resizeGrid(0, 512, 65536, 262144));
    const grid = try resizeGrid(3000, 4000, 65536, 262144);
    try std.testing.expect(try grid.count() <= 256);
}
