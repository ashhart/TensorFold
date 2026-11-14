//! JPEG output via libjpeg-turbo, loaded on first use: PDF page images (pdf.zig) and OCR image crops (encodeRgb).
const std = @import("std");
const sys = @import("sys");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;

/// The TurboJPEG 3 entry points the encoders call.
pub const Fns = struct {
    tj3Init: *const fn (c_int) callconv(.c) ?*anyopaque,
    tj3Destroy: *const fn (?*anyopaque) callconv(.c) void,
    tj3Set: *const fn (?*anyopaque, c_int, c_int) callconv(.c) c_int,
    tj3Compress8: *const fn (?*anyopaque, [*]const u8, c_int, c_int, c_int, c_int, *?[*]u8, *usize) callconv(.c) c_int,
    tj3Free: *const fn (?*anyopaque) callconv(.c) void,
};

pub const Error = error{ OutOfMemory, EncodeFailed };

const Loaded = union(enum) { unknown, missing, ready: Fns };

var state: Loaded = .unknown;
var mutex: std.Io.Mutex = .init;

/// Looks up every function field of `T` in optional library `id`, optional ones allowed to miss; PDFium uses it too.
pub fn loadDynLib(comptime T: type, comptime id: sys.optional_libraries.Id, a: Allocator) ?T {
    const exe = std.process.executableDirPathAlloc(log.io(), a) catch null;
    defer if (exe) |p| a.free(p);
    var dl = sys.optional_libraries.open(sys.optional_libraries.spec(id), a, sys, exe) orelse return null;
    var f: T = undefined;
    const info = @typeInfo(T).@"struct";
    inline for (info.field_names, info.field_types) |name, FT| {
        if (@typeInfo(FT) == .optional) {
            @field(f, name) = dl.lookup(@typeInfo(FT).optional.child, name);
        } else {
            @field(f, name) = dl.lookup(FT, name) orelse {
                dl.close();
                return null;
            };
        }
    }
    return f;
}

/// libjpeg-turbo's entry points, or null when the library is missing; the first call loads it.
pub fn get(a: Allocator) ?Fns {
    const io = log.io();
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    if (state == .unknown) state = if (loadDynLib(Fns, .jpeg, a)) |f| .{ .ready = f } else .missing;
    return if (state == .ready) state.ready else null;
}

/// Encodes PDFium's BGRx bitmap (rows `stride` bytes apart) as a 4:2:0 JPEG at quality 85.
pub fn encodeBgra(a: Allocator, jf: Fns, bgra: []const u8, width: usize, height: usize, stride: usize) Error![]const u8 {
    return encode(a, jf, bgra, width, height, stride, 8, 85); // 8: TJPF_BGRA, PDFium's BGRx bitmap
}

/// Encodes packed RGB rows as a 4:2:0 JPEG at `quality`; NoLibrary when libjpeg-turbo is missing.
pub fn encodeRgb(a: Allocator, rgb: []const u8, width: usize, height: usize, quality: u8) (Error || error{NoLibrary})![]const u8 {
    const jf = get(a) orelse return error.NoLibrary;
    const pitch = std.math.mul(usize, width, 3) catch return error.EncodeFailed;
    if (pitch == 0 or rgb.len / pitch < height) return error.EncodeFailed;
    return encode(a, jf, rgb, width, height, pitch, 0, quality); // 0: TJPF_RGB
}

fn encode(a: Allocator, jf: Fns, pixels: []const u8, width: usize, height: usize, pitch: usize, format: c_int, quality: u8) Error![]const u8 {
    const max: usize = std.math.maxInt(c_int);
    if (width == 0 or height == 0 or width > max or height > max or pitch > max) return error.EncodeFailed;
    const handle = jf.tj3Init(0) orelse return error.EncodeFailed; // 0 = TJINIT_COMPRESS
    defer jf.tj3Destroy(handle);
    _ = jf.tj3Set(handle, 3, quality); // TJPARAM_QUALITY = 3
    _ = jf.tj3Set(handle, 4, 2); // TJPARAM_SUBSAMP = 4, TJSAMP_420 = 2
    var out_buf: ?[*]u8 = null;
    var out_size: usize = 0;
    const rc = jf.tj3Compress8(handle, pixels.ptr, @intCast(width), @intCast(pitch), @intCast(height), format, &out_buf, &out_size);
    defer if (out_buf) |p| jf.tj3Free(p);
    if (rc != 0 or out_buf == null or out_size == 0) return error.EncodeFailed;
    return a.dupe(u8, out_buf.?[0..out_size]);
}

test "jpeg out: encodeRgb round-trips a small image through core.picture" {
    const picture = @import("core").picture;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = 24;
    const h = 10;
    var rgb: [w * h * 3]u8 = undefined;
    for (&rgb, 0..) |*px, i| px.* = @truncate(i * 7);
    const bytes = encodeRgb(a, &rgb, w, h, 90) catch |err| switch (err) {
        error.NoLibrary => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(picture.Format.jpeg, picture.sniff(bytes));
    const img = picture.decode(std.testing.io, a, bytes) catch |err| switch (err) {
        error.NoJpegLibrary, error.JpegLibraryVersion => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(usize, w), img.width);
    try std.testing.expectEqual(@as(usize, h), img.height);
    try std.testing.expectError(error.EncodeFailed, encodeRgb(a, rgb[0 .. rgb.len - 1], w, h, 90));
}
