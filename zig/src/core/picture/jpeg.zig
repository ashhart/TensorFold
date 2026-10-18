//! JPEG through one pinned libjpeg-turbo, loaded at the first JPEG and run at Pillow's settings: 0.6.6's pixels.

const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const picture = @import("picture.zig");
const tags = @import("tags.zig");

/// The libjpeg-turbo release Pillow 12.3 decoded with for TensorFold 0.6.6 (its build's own version string).
pub const pinned = "3.1.4.1";

/// The environment variable naming the libturbojpeg file to load.
pub const env = "TENSORFOLD_LIBTURBOJPEG";

const needs = "JPEG images need libjpeg-turbo " ++ pinned ++ " (libturbojpeg)";
pub const missing = @import("sys").optional_libraries.spec(.jpeg).message;
pub const wrong_version = needs ++ "; the library has another release, whose pixels can differ; " ++ @import("sys").optional_libraries.spec(.jpeg).message;

const library = if (builtin.os.tag == .macos) "libturbojpeg.0.dylib" else if (builtin.os.tag == .windows) "turbojpeg.dll" else "libturbojpeg.so.0";

// TurboJPEG 3's enums: an init type, pixel formats, parameters and error codes
const init_decompress = 1;
const pf_rgb = 0;
const pf_gray = 6;
const param_stop_on_warning = 0;
const param_width = 5;
const param_height = 6;
const param_precision = 7;
const param_fast_upsample = 9;
const param_fast_dct = 10;
const err_warning = 0;

const Fns = struct {
    tj3Init: *const fn (c_int) callconv(.c) ?*anyopaque,
    tj3Destroy: *const fn (?*anyopaque) callconv(.c) void,
    tj3Set: *const fn (?*anyopaque, c_int, c_int) callconv(.c) c_int,
    tj3Get: *const fn (?*anyopaque, c_int) callconv(.c) c_int,
    tj3DecompressHeader: *const fn (?*anyopaque, [*]const u8, usize) callconv(.c) c_int,
    tj3Decompress8: *const fn (?*anyopaque, [*]const u8, usize, [*]u8, c_int, c_int) callconv(.c) c_int,
    tj3GetErrorCode: *const fn (?*anyopaque) callconv(.c) c_int,
};

const Loaded = union(enum) { unknown, missing, wrong, ready: Fns };
var loaded: Loaded = .unknown;
var load_mutex: std.atomic.Mutex = .unlocked;

const DlInfo = extern struct { fname: ?[*:0]const u8, fbase: ?*anyopaque, sname: ?[*:0]const u8, saddr: ?*anyopaque };
extern "c" fn dladdr(addr: *const anyopaque, info: *DlInfo) c_int;

/// The pinned library: the file TENSORFOLD_LIBTURBOJPEG names, else ../lib, beside the server, then the loader's path.
fn load(io: std.Io, a: Allocator) error{ NoJpegLibrary, JpegLibraryVersion }!Fns {
    while (!load_mutex.tryLock()) std.atomic.spinLoopHint();
    defer load_mutex.unlock();
    if (loaded == .unknown) loaded = find(io, a);
    return switch (loaded) {
        .ready => |f| f,
        .wrong => error.JpegLibraryVersion,
        .missing, .unknown => error.NoJpegLibrary,
    };
}

fn find(io: std.Io, a: Allocator) Loaded {
    if (platform.c.getenv(env)) |named| return open(io, a, std.mem.span(named)) orelse .missing;
    var seen_wrong = false;
    if (std.process.executableDirPathAlloc(io, a)) |exe| {
        defer a.free(exe);
        for ([_][]const []const u8{ &.{ exe, "..", "lib", library }, &.{ exe, library } }) |parts| {
            const p = std.fs.path.join(a, parts) catch continue;
            defer a.free(p);
            switch (open(io, a, p) orelse continue) {
                .ready => |f| return .{ .ready = f },
                else => seen_wrong = true,
            }
        }
    } else |_| {}
    const brew = [_][]const u8{ "/opt/homebrew/opt/jpeg-turbo/lib/" ++ library, "/usr/local/opt/jpeg-turbo/lib/" ++ library };
    const fixed = if (builtin.os.tag == .macos) brew ++ [_][]const u8{library} else [_][]const u8{library};
    for (fixed) |p| switch (open(io, a, p) orelse continue) {
        .ready => |f| return .{ .ready = f },
        else => seen_wrong = true,
    };
    return if (seen_wrong) .wrong else .missing;
}

/// One candidate: null when it does not load or lacks TurboJPEG 3; `.wrong` when its version string is not the pin.
fn open(io: std.Io, a: Allocator, path: []const u8) ?Loaded {
    var dl = platform.DynLib.open(path) catch return null;
    var f: Fns = undefined;
    const info = @typeInfo(Fns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, name) orelse {
            dl.close();
            return null;
        };
    }
    if (!versionIs(io, a, @ptrCast(f.tj3Init))) {
        std.log.warn("{s} is not libjpeg-turbo {s}; JPEG images are refused until it is", .{ path, pinned });
        dl.close();
        return .wrong;
    }
    return .{ .ready = f };
}

/// The version string libjpeg-turbo builds into every binary ("libjpeg-turbo version X (build Y)") equals the pin.
fn versionIs(io: std.Io, a: Allocator, symbol: *const anyopaque) bool {
    var path: [1100]u8 = undefined;
    const file = if (comptime builtin.os.tag == .windows) platform.libraryOf(symbol, &path) orelse return false else blk: {
        var info: DlInfo = undefined;
        if (dladdr(symbol, &info) == 0) return false;
        break :blk std.mem.span(info.fname orelse return false);
    };
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(64 << 20)) catch return false;
    defer a.free(bytes);
    return versionOf(bytes) != null and std.mem.eql(u8, versionOf(bytes).?, pinned);
}

/// The version in a libjpeg-turbo binary's string table.
fn versionOf(bytes: []const u8) ?[]const u8 {
    const key = "libjpeg-turbo version ";
    const at = std.mem.indexOf(u8, bytes, key) orelse return null;
    const rest = bytes[at + key.len ..];
    return rest[0 .. std.mem.indexOfAny(u8, rest, " \x00") orelse return null];
}

/// Whether the pinned library loads here (tests skip JPEG pixels without it).
pub fn available(io: std.Io, a: Allocator) bool {
    _ = load(io, a) catch return false;
    return true;
}

/// What Pillow's open read from the markers before the first scan, and whether the data reaches its end marker.
const Markers = struct {
    width: u16 = 0,
    height: u16 = 0,
    components: u8 = 0,
    exif: std.ArrayList(u8) = .empty,
    xmp: ?[]const u8 = null,
    dpi: bool = false, // a JFIF density in dots: Pillow's open then read no EXIF, so a bad block failed later
    ended: bool = false,
};

pub fn decode(io: std.Io, a: Allocator, data: []const u8) picture.Error!picture.Samples {
    var m: Markers = .{};
    defer m.exif.deinit(a);
    try walk(a, data, &m);
    try picture.fits(m.width, m.height);
    if (m.components == 4) return error.CmykJpeg;
    if (!m.ended) return error.Invalid; // libjpeg would read past the end: Pillow's load refused the truncated bytes
    const f = try load(io, a);
    const h = f.tj3Init(init_decompress) orelse return error.OutOfMemory;
    defer f.tj3Destroy(h);
    // Pillow's libjpeg settings: the accurate integer IDCT and smooth chroma upsampling; warnings are not errors
    for ([_][2]c_int{ .{ param_fast_dct, 0 }, .{ param_fast_upsample, 0 }, .{ param_stop_on_warning, 0 } }) |p| {
        if (f.tj3Set(h, p[0], p[1]) != 0) return error.Invalid;
    }
    if (f.tj3DecompressHeader(h, data.ptr, data.len) != 0 and f.tj3GetErrorCode(h) != err_warning) return error.Invalid;
    const same = f.tj3Get(h, param_width) == m.width and f.tj3Get(h, param_height) == m.height;
    if (!same or f.tj3Get(h, param_precision) != 8) return error.Invalid;
    const w: usize = m.width;
    const ht: usize = m.height;
    const rgb = try a.alloc(u8, w * ht * 3);
    errdefer a.free(rgb);
    const gray = m.components == 1;
    const rc = f.tj3Decompress8(h, data.ptr, data.len, rgb.ptr, 0, if (gray) pf_gray else pf_rgb);
    if (rc != 0 and f.tj3GetErrorCode(h) != err_warning) return error.Invalid;
    if (gray) { // Pillow's L to RGB: each value three times, widened in place from the end
        var i = w * ht;
        while (i > 0) {
            i -= 1;
            @memset(rgb[i * 3 ..][0..3], rgb[i]);
        }
    }
    const exif: ?[]const u8 = if (m.exif.items.len > 0) m.exif.items else null;
    const o = try tags.orientation(exif, m.xmp, if (m.dpi) .refuse else .ignore);
    return .{ .width = w, .height = ht, .channels = 3, .data = rgb, .orientation = o };
}

/// Pillow's open: markers up to the first scan (EXIF, XMP, JFIF density, the frame), then on to the end marker.
fn walk(a: Allocator, data: []const u8, m: *Markers) picture.Error!void {
    var at: usize = 2;
    while (true) {
        if (at >= data.len) return error.Invalid;
        if (data[at] != 0xFF) { // junk before a marker
            at += 1;
            continue;
        }
        if (at + 1 >= data.len) return error.Invalid;
        const marker = data[at + 1];
        switch (marker) {
            0xFF => at += 1, // fill: the second byte starts the marker
            0x00 => at += 2,
            0xC8, 0xD0...0xD9, 0xF0...0xFD => at += 2, // standalone in Pillow's table
            0xC0...0xC7, 0xC9...0xCF, 0xDA...0xEF, 0xFE => {
                const body = try segment(data, at);
                at += 2 + 2 + body.len;
                switch (marker) {
                    0xC0...0xC3, 0xC5...0xC7, 0xC9...0xCB, 0xCD...0xCF, 0xDE => try frame(m, body),
                    0xDB => try quantTables(body),
                    0xE0 => try jfif(m, body),
                    0xE1 => try appExif(a, m, body),
                    0xEE => if (std.mem.startsWith(u8, body, "Adobe") and body.len < 7) return error.Invalid,
                    else => {},
                }
                if (marker == 0xDA) break;
            },
            else => return error.Invalid, // Pillow found no marker it knew
        }
    }
    m.ended = reachesEnd(data, at); // without a frame the size stays 0 x 0, which 0.6.6 refused as too large
}

/// A segment's body after its two length bytes; Pillow read nothing for a length under 2.
fn segment(data: []const u8, at: usize) error{Invalid}![]const u8 {
    if (data.len - at < 4) return error.Invalid;
    const len = std.mem.readInt(u16, data[at + 2 ..][0..2], .big);
    if (len < 2) return data[at + 4 ..][0..0];
    if (data.len - at - 4 < len - 2) return error.Invalid;
    return data[at + 4 ..][0 .. len - 2];
}

fn frame(m: *Markers, b: []const u8) error{Invalid}!void {
    if (b.len < 6 or b[0] != 8) return error.Invalid; // Pillow handled 8-bit samples only
    if (b[5] != 1 and b[5] != 3 and b[5] != 4) return error.Invalid;
    m.height = std.mem.readInt(u16, b[1..3], .big);
    m.width = std.mem.readInt(u16, b[3..5], .big);
    m.components = b[5];
}

fn quantTables(b: []const u8) error{Invalid}!void {
    var rest = b;
    while (rest.len > 0) {
        const n: usize = if (rest[0] >> 4 == 0) 65 else 129;
        if (rest.len < n) return error.Invalid;
        rest = rest[n..];
    }
}

fn jfif(m: *Markers, b: []const u8) error{Invalid}!void {
    if (!std.mem.startsWith(u8, b, "JFIF")) return;
    if (b.len < 7) return error.Invalid;
    if (b.len >= 12 and (b[7] == 1 or b[7] == 2)) m.dpi = true;
}

/// EXIF segments joined as Pillow joined them (the first whole, later ones after their header), and the last XMP.
fn appExif(a: Allocator, m: *Markers, b: []const u8) Allocator.Error!void {
    if (std.mem.startsWith(u8, b, "Exif\x00\x00")) {
        try m.exif.appendSlice(a, if (m.exif.items.len == 0) b else b[6..]);
    } else if (std.mem.startsWith(u8, b, "http://ns.adobe.com/xap/1.0/\x00")) {
        m.xmp = b[29..];
    }
}

/// Whether the data after the first scan's header reaches an end marker: scans, stuffed bytes and segments skipped.
fn reachesEnd(data: []const u8, from: usize) bool {
    var at = from;
    while (at + 1 < data.len) {
        if (data[at] != 0xFF) {
            at += 1;
            continue;
        }
        switch (data[at + 1]) {
            0x00 => at += 2,
            0xFF => at += 1,
            0xD9 => return true,
            0x01, 0xD0...0xD8 => at += 2,
            else => {
                if (data.len - at < 4) return false;
                at += 2 + std.mem.readInt(u16, data[at + 2 ..][0..2], .big);
            },
        }
    }
    return false;
}

test "the version string libjpeg-turbo builds in is read up to its build stamp" {
    try std.testing.expectEqualStrings("3.1.4.1", versionOf("\x00xx libjpeg-turbo version 3.1.4.1 (build 20260701)\x00").?);
    try std.testing.expect(versionOf("libjpeg version 9f") == null);
}

test "the end marker is found past stuffed bytes, restarts and segments, and missing when the data stops" {
    const scan = "\x12\xff\x00\x34\xff\xd0\x56\xff\xc4\x00\x04\xff\xd9\x78";
    try std.testing.expect(reachesEnd(scan ++ "\xff\xd9", 0));
    try std.testing.expect(!reachesEnd(scan, 0));
    try std.testing.expect(!reachesEnd("\x12\xff\xda\x00\x10", 0));
}
