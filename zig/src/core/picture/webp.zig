//! WebP through one pinned libwebp decoder, loaded at the first WebP: the pixels Pillow 13.0.0's bundled libwebp gives.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const picture = @import("picture.zig");
const tags = @import("tags.zig");

/// The libwebp release Pillow 13.0.0 bundles (features.version("webp")), as WebPGetDecoderVersion packs it.
pub const pinned = "1.6.0";
const pinned_code = 0x010600;

/// The environment variable naming the libwebpdecoder file to load.
pub const env = "TENSORFOLD_LIBWEBP";

const needs = "WebP images need libwebp " ++ pinned ++ " (libwebpdecoder)";
pub const missing = @import("sys").optional_libraries.spec(.webp).message;
pub const wrong_version = needs ++ "; the library has another release, whose pixels can differ; " ++ @import("sys").optional_libraries.spec(.webp).message;

const library = if (builtin.os.tag == .macos) "libwebpdecoder.3.dylib" else "libwebpdecoder.so.3";

const Fns = struct {
    WebPGetDecoderVersion: *const fn () callconv(.c) c_int,
    WebPGetInfo: *const fn ([*]const u8, usize, *c_int, *c_int) callconv(.c) c_int,
    WebPDecodeRGBAInto: *const fn ([*]const u8, usize, [*]u8, usize, c_int) callconv(.c) ?[*]u8,
};

const Loaded = union(enum) { unknown, missing, wrong, ready: Fns };
var loaded: Loaded = .unknown;
var load_mutex: std.atomic.Mutex = .unlocked;

/// The pinned library: the file TENSORFOLD_LIBWEBP names, else ../lib, beside the server, then the loader's path.
fn load(io: std.Io, a: Allocator) error{ NoWebpLibrary, WebpLibraryVersion }!Fns {
    while (!load_mutex.tryLock()) std.atomic.spinLoopHint();
    defer load_mutex.unlock();
    if (loaded == .unknown) loaded = find(io, a);
    return switch (loaded) {
        .ready => |f| f,
        .wrong => error.WebpLibraryVersion,
        .missing, .unknown => error.NoWebpLibrary,
    };
}

fn find(io: std.Io, a: Allocator) Loaded {
    if (std.c.getenv(env)) |named| return open(std.mem.span(named)) orelse .missing;
    var seen_wrong = false;
    if (std.process.executableDirPathAlloc(io, a)) |exe| {
        defer a.free(exe);
        for ([_][]const []const u8{ &.{ exe, "..", "lib", library }, &.{ exe, library } }) |parts| {
            const p = std.fs.path.join(a, parts) catch continue;
            defer a.free(p);
            switch (open(p) orelse continue) {
                .ready => |f| return .{ .ready = f },
                else => seen_wrong = true,
            }
        }
    } else |_| {}
    const brew = [_][]const u8{ "/opt/homebrew/opt/webp/lib/" ++ library, "/usr/local/opt/webp/lib/" ++ library };
    const fixed = if (builtin.os.tag == .macos) brew ++ [_][]const u8{library} else [_][]const u8{library};
    for (fixed) |p| switch (open(p) orelse continue) {
        .ready => |f| return .{ .ready = f },
        else => seen_wrong = true,
    };
    return if (seen_wrong) .wrong else .missing;
}

/// One candidate: null when it does not load or lacks a function; `.wrong` when its version is not the pin.
fn open(path: []const u8) ?Loaded {
    var dl = @import("sys").DynLib.open(path) catch return null;
    var f: Fns = undefined;
    const info = @typeInfo(Fns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, name) orelse {
            dl.close();
            return null;
        };
    }
    if (f.WebPGetDecoderVersion() != pinned_code) {
        std.log.warn("{s} is not libwebp {s}; WebP images are refused until it is", .{ path, pinned });
        dl.close();
        return .wrong;
    }
    return .{ .ready = f };
}

/// Whether the pinned library loads here (tests skip WebP pixels without it).
pub fn available(io: std.Io, a: Allocator) bool {
    check(io, a) catch return false;
    return true;
}

/// Check the decoder's symbols and exact version without reading pixels.
pub fn check(io: std.Io, a: Allocator) !void {
    _ = try load(io, a);
}

/// What Pillow's open read from the RIFF chunks: the VP8X animation flag and the EXIF and XMP chunks.
const Chunks = struct {
    animated: bool = false,
    exif: ?[]const u8 = null,
    xmp: ?[]const u8 = null,
};

/// The container's chunks after "RIFF size WEBP"; error.Invalid when a chunk runs past the bytes.
fn chunks(data: []const u8) picture.Error!Chunks {
    if (data.len < 12) return error.Invalid;
    const riff = std.mem.readInt(u32, data[4..8], .little);
    const end = @min(data.len, @as(usize, riff) + 8);
    var c: Chunks = .{};
    var at: usize = 12;
    while (at + 8 <= end) {
        const id = data[at..][0..4];
        const n = std.mem.readInt(u32, data[at + 4 ..][0..4], .little);
        const body = at + 8;
        if (n > end - body) return error.Invalid;
        const bytes = data[body..][0..n];
        if (std.mem.eql(u8, id, "VP8X")) {
            if (n < 10) return error.Invalid;
            c.animated = bytes[0] & 0x02 != 0;
        } else if (std.mem.eql(u8, id, "ANIM") or std.mem.eql(u8, id, "ANMF")) {
            c.animated = true;
        } else if (std.mem.eql(u8, id, "EXIF")) {
            c.exif = bytes;
        } else if (std.mem.eql(u8, id, "XMP ")) {
            c.xmp = bytes;
        }
        at = body + n + (n & 1); // chunks pad to an even size
    }
    return c;
}

/// WebP bytes as Pillow loaded them: one frame, RGBA (alpha opaque without one), and the orientation its tags give.
pub fn decode(io: std.Io, a: Allocator, data: []const u8) picture.Error!picture.Samples {
    const c = try chunks(data);
    if (c.animated) return error.Animated;
    const f = try load(io, a);
    var w: c_int = 0;
    var h: c_int = 0;
    if (f.WebPGetInfo(data.ptr, data.len, &w, &h) == 0) return error.Invalid;
    try picture.fits(@intCast(w), @intCast(h));
    const width: usize = @intCast(w);
    const height: usize = @intCast(h);
    const rgba = try a.alloc(u8, width * height * 4);
    errdefer a.free(rgba);
    if (f.WebPDecodeRGBAInto(data.ptr, data.len, rgba.ptr, rgba.len, @intCast(width * 4)) == null) return error.Invalid;
    const o = try tags.orientation(c.exif, c.xmp, .refuse);
    return .{ .width = width, .height = height, .channels = 4, .data = rgba, .orientation = o };
}

test "a WebP's chunks: the animation flag, its EXIF, and a chunk past the bytes refused" {
    const still = "RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    try std.testing.expect(!(try chunks(still)).animated);
    const anim = "RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    try std.testing.expect((try chunks(anim)).animated);
    const exif = "RIFF\x18\x00\x00\x00WEBPEXIF\x03\x00\x00\x00abc\x00XMP \x00\x00\x00\x00";
    const got = try chunks(exif);
    try std.testing.expectEqualStrings("abc", got.exif.?);
    try std.testing.expectEqualStrings("", got.xmp.?);
    try std.testing.expectError(error.Invalid, chunks("RIFF\x10\x00\x00\x00WEBPVP8 \xff\x00\x00\x00"));
}
