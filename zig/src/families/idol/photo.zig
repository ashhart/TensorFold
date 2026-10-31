//! A raw photo to IdolPipeline.preprocess's 1024 x 1024 picture (BiRefNet's alpha unless real), and a PNG writer.
const std = @import("std");
const mtl = @import("metal");
const pre = @import("preprocess.zig");
const bn = @import("birefnet.zig");

/// Seconds of each step (preprocess: all of it but the load, set by the caller), the Metal bytes BiRefNet held.
pub const Timing = struct { decode: f64 = 0, load: f64 = 0, birefnet: f64 = 0, birefnet_gpu: f64 = 0, finish: f64 = 0, preprocess: f64 = 0, real_alpha: bool = false, metal_peak: u64 = 0 };

/// Whether `path` names a photo (.png, .jpg, .jpeg or .webp) rather than a safetensors picture.
pub fn isPhoto(path: []const u8) bool {
    const ext = std.fs.path.extension(path);
    for ([_][]const u8{ ".png", ".jpg", ".jpeg", ".webp" }) |e| if (std.ascii.eqlIgnoreCase(ext, e)) return true;
    return false;
}

/// The picture [1024 * 1024 * 3] of the photo at `path`; BiRefNet is loaded from `weights` only if needed.
pub fn preprocess(a: std.mem.Allocator, io: std.Io, device: mtl.Device, weights: [:0]const u8, path: []const u8, crop_margin: f64, out: []u8, t: *Timing) !void {
    var started = mtl.clock.seconds();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
    defer a.free(bytes);
    var photo = try pre.decode(io, a, bytes);
    defer photo.deinit(a);
    var resized = try pre.resizeFirst(a, photo);
    defer resized.deinit(a);
    t.decode = mtl.clock.seconds() - started;
    t.real_alpha = pre.hasRealAlpha(resized);
    var alpha: ?[]u8 = null;
    defer if (alpha) |al| a.free(al);
    if (!t.real_alpha) {
        started = mtl.clock.seconds();
        const net = try bn.Net.load(std.heap.page_allocator, device, weights);
        defer net.deinit();
        t.load = mtl.clock.seconds() - started;
        const logits = try a.alloc(f32, bn.size * bn.size);
        defer a.free(logits);
        alpha = try a.alloc(u8, resized.width * resized.height);
        started = mtl.clock.seconds();
        const tm = try net.removeBackground(a, resized, logits, alpha.?);
        t.birefnet = mtl.clock.seconds() - started;
        t.birefnet_gpu = tm.gpu;
        t.metal_peak = device.allocated();
    }
    started = mtl.clock.seconds();
    try pre.finish(a, resized, alpha, crop_margin, out);
    t.finish = mtl.clock.seconds() - started;
}

/// An 8-bit RGB PNG of `rgb` [h, w, 3], its image data in stored (uncompressed) deflate blocks.
pub fn png(a: std.mem.Allocator, rgb: []const u8, w: usize, h: usize) ![]u8 {
    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(a);
    for (0..h) |y| {
        try raw.append(a, 0);
        try raw.appendSlice(a, rgb[y * w * 3 ..][0 .. w * 3]);
    }
    var z: std.ArrayList(u8) = .empty;
    defer z.deinit(a);
    try z.appendSlice(a, &.{ 0x78, 0x01 });
    var at: usize = 0;
    while (true) {
        const n = @min(raw.items.len - at, 65535);
        const last = at + n == raw.items.len;
        try z.append(a, @intFromBool(last));
        var len: [4]u8 = undefined;
        std.mem.writeInt(u16, len[0..2], @intCast(n), .little);
        std.mem.writeInt(u16, len[2..4], ~@as(u16, @intCast(n)), .little);
        try z.appendSlice(a, &len);
        try z.appendSlice(a, raw.items[at..][0..n]);
        at += n;
        if (last) break;
    }
    var adler: [4]u8 = undefined;
    std.mem.writeInt(u32, &adler, std.hash.Adler32.hash(raw.items), .big);
    try z.appendSlice(a, &adler);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "\x89PNG\r\n\x1a\n");
    var hdr: [13]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], @intCast(w), .big);
    std.mem.writeInt(u32, hdr[4..8], @intCast(h), .big);
    hdr[8..13].* = .{ 8, 2, 0, 0, 0 };
    try chunk(a, &out, "IHDR", &hdr);
    try chunk(a, &out, "IDAT", z.items);
    try chunk(a, &out, "IEND", "");
    return out.toOwnedSlice(a);
}

fn chunk(a: std.mem.Allocator, out: *std.ArrayList(u8), kind: *const [4]u8, body: []const u8) !void {
    var be: [4]u8 = undefined;
    std.mem.writeInt(u32, &be, @intCast(body.len), .big);
    try out.appendSlice(a, &be);
    try out.appendSlice(a, kind);
    try out.appendSlice(a, body);
    var crc = std.hash.Crc32.init();
    crc.update(kind);
    crc.update(body);
    std.mem.writeInt(u32, &be, crc.final(), .big);
    try out.appendSlice(a, &be);
}

test "the PNG writer's picture decodes back to its bytes" {
    const a = std.testing.allocator;
    const w = 300;
    const h = 90;
    const rgb = try a.alloc(u8, w * h * 3);
    defer a.free(rgb);
    for (rgb, 0..) |*v, i| v.* = @truncate(i * 7);
    const bytes = try png(a, rgb, w, h);
    defer a.free(bytes);
    const s = try @import("../../core/picture/png.zig").decode(a, bytes);
    defer a.free(s.data);
    try std.testing.expectEqual(@as(usize, w), s.width);
    for (0..w * h) |i| try std.testing.expectEqualSlices(u8, rgb[i * 3 ..][0..3], s.data[i * s.channels ..][0..3]);
}

test "a photo is named by its extension, any case" {
    try std.testing.expect(isPhoto("a/b.PNG") and isPhoto("x.jpeg") and isPhoto("y.webp") and isPhoto("z.JpG"));
    try std.testing.expect(!isPhoto("case/vae.safetensors") and !isPhoto("-"));
}
