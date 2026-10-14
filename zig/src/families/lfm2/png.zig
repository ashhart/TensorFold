//! PNG to 8-bit RGBA as Pillow opens and converts it: every bit depth and colour type, tRNS, Adam7, eXIf, CRCs checked.

const std = @import("std");
const Allocator = std.mem.Allocator;
const image = @import("image.zig");

pub const signature = "\x89PNG\r\n\x1a\n";

const Header = struct { width: usize, height: usize, depth: u8, color: u8, interlace: bool };

/// Adam7's passes as (x0, y0, dx, dy).
const adam7 = [_][4]u8{ .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 }, .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 } };
const whole = [_][4]u8{.{ 0, 0, 1, 1 }};

pub fn decode(a: Allocator, data: []const u8) image.DecodeError!image.Decoded {
    var hdr: ?Header = null;
    var palette: []const u8 = &.{};
    var trns: ?[]const u8 = null;
    var exif: ?[]const u8 = null;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(a);
    var at: usize = signature.len;
    while (true) {
        if (data.len - at < 12) return error.Invalid;
        const len = std.mem.readInt(u32, data[at..][0..4], .big);
        if (len > data.len - at - 12) return error.Invalid;
        const kind = data[at + 4 ..][0..4];
        const body = data[at + 8 ..][0..len];
        if (std.hash.Crc32.hash(data[at + 4 .. at + 8 + len]) != std.mem.readInt(u32, data[at + 8 + len ..][0..4], .big)) return error.Invalid;
        at += 12 + len;
        if (hdr == null and !std.mem.eql(u8, kind, "IHDR")) return error.Invalid;
        if (std.mem.eql(u8, kind, "IHDR")) {
            if (len != 13) return error.Invalid;
            hdr = .{ .width = std.mem.readInt(u32, body[0..4], .big), .height = std.mem.readInt(u32, body[4..8], .big), .depth = body[8], .color = body[9], .interlace = body[12] == 1 };
            if (body[10] != 0 or body[11] != 0 or body[12] > 1 or bitsPerPixel(hdr.?) == 0) return error.Invalid;
            try image.fits(hdr.?.width, hdr.?.height);
        } else if (std.mem.eql(u8, kind, "PLTE")) {
            palette = body;
        } else if (std.mem.eql(u8, kind, "tRNS")) {
            trns = body;
        } else if (std.mem.eql(u8, kind, "eXIf")) {
            exif = exif orelse body;
        } else if (std.mem.eql(u8, kind, "acTL")) {
            if (len < 8) return error.Invalid;
            if (std.mem.readInt(u32, body[0..4], .big) != 1) return error.Animated;
        } else if (std.mem.eql(u8, kind, "IDAT")) {
            try idat.appendSlice(a, body);
        } else if (std.mem.eql(u8, kind, "IEND")) break;
    }
    const h = hdr.?;
    if (h.color == 3 and palette.len == 0) return error.Invalid;
    const bits = bitsPerPixel(h);
    const passes: []const [4]u8 = if (h.interlace) &adam7 else &whole;
    var size: usize = 0;
    for (passes) |p| {
        const pw, const ph = passSize(h, p);
        if (pw > 0 and ph > 0) size += ph * (1 + (pw * bits + 7) / 8);
    }
    const raw = try a.alloc(u8, size);
    defer a.free(raw);
    var in: std.Io.Reader = .fixed(idat.items);
    var flate: std.compress.flate.Decompress = .init(&in, .zlib, &.{});
    var out: std.Io.Writer = .fixed(raw);
    flate.reader.streamExact(&out, size) catch return error.Invalid;
    const rgba = try a.alloc(u8, h.width * h.height * 4);
    errdefer a.free(rgba);
    var rest = raw;
    for (passes) |p| {
        const pw, const ph = passSize(h, p);
        if (pw == 0 or ph == 0) continue;
        const stride = (pw * bits + 7) / 8;
        var prev: []const u8 = &.{};
        for (0..ph) |y| {
            const row = rest[1 .. 1 + stride];
            try unfilter(rest[0], row, prev, @max(1, bits / 8));
            for (0..pw) |x| convert(h, row, x, palette, trns, rgba[((p[1] + y * p[3]) * h.width + p[0] + x * p[2]) * 4 ..][0..4]);
            prev = row;
            rest = rest[1 + stride ..];
        }
    }
    return .{ .width = h.width, .height = h.height, .rgba = rgba, .exif = exif };
}

/// Bits a pixel for Pillow's supported depth and colour type pairs; 0 for any other pair.
fn bitsPerPixel(h: Header) usize {
    const channels: usize = switch (h.color) {
        0, 3 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        else => return 0,
    };
    const ok = switch (h.color) {
        0 => h.depth == 1 or h.depth == 2 or h.depth == 4 or h.depth == 8 or h.depth == 16,
        3 => h.depth == 1 or h.depth == 2 or h.depth == 4 or h.depth == 8,
        else => h.depth == 8 or h.depth == 16,
    };
    return if (ok) channels * h.depth else 0;
}

fn passSize(h: Header, p: [4]u8) [2]usize {
    return .{ (h.width + p[2] - 1 - p[0]) / p[2], (h.height + p[3] - 1 - p[1]) / p[3] };
}

fn unfilter(kind: u8, row: []u8, prev: []const u8, bpp: usize) error{Invalid}!void {
    for (row, 0..) |*v, i| {
        const left: u8 = if (i >= bpp) row[i - bpp] else 0;
        const up: u8 = if (prev.len > 0) prev[i] else 0;
        const corner: u8 = if (prev.len > 0 and i >= bpp) prev[i - bpp] else 0;
        v.* +%= switch (kind) {
            0 => 0,
            1 => left,
            2 => up,
            3 => @intCast((@as(u9, left) + up) / 2),
            4 => paeth(left, up, corner),
            else => return error.Invalid,
        };
    }
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p = @as(i16, a) + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    return if (pa <= pb and pa <= pc) a else if (pb <= pc) b else c;
}

/// Pixel `x` of a row in Pillow's mode for it (16-bit gray clipped, other 16-bit high bytes); alpha, palette or key.
fn convert(h: Header, row: []const u8, x: usize, palette: []const u8, trns: ?[]const u8, px: *[4]u8) void {
    switch (h.color) {
        0 => {
            const raw = sample(row, x, h.depth);
            const v: u8 = switch (h.depth) {
                1 => if (raw != 0) 255 else 0,
                2 => @intCast(raw * 85),
                4 => @intCast(raw * 17),
                else => @intCast(@min(raw, 255)),
            };
            const key: ?u16 = if (trns) |t| (if (t.len >= 2) std.mem.readInt(u16, t[0..2], .big) else null) else null;
            const keyed = if (key) |k| (if (h.depth == 1) @as(u16, if (k != 0) 255 else 0) else k) == v else false;
            px.* = .{ v, v, v, if (keyed) 0 else 255 };
        },
        2 => {
            for (px[0..3], 0..) |*c, i| c.* = high(row, 3 * x + i, h.depth);
            const keyed = if (trns) |t| t.len >= 6 and for (px[0..3], 0..) |c, i| {
                if (std.mem.readInt(u16, t[2 * i ..][0..2], .big) != c) break false;
            } else true else false;
            px[3] = if (keyed) 0 else 255;
        },
        3 => {
            const i = sample(row, x, h.depth);
            px.* = if (3 * i + 3 <= palette.len) .{ palette[3 * i], palette[3 * i + 1], palette[3 * i + 2], 255 } else .{ 0, 0, 0, 255 };
            if (trns) |t| if (i < t.len) {
                px[3] = t[i];
            };
        },
        4 => {
            const v = high(row, 2 * x, h.depth);
            px.* = .{ v, v, v, high(row, 2 * x + 1, h.depth) };
        },
        else => for (px, 0..) |*c, i| {
            c.* = high(row, 4 * x + i, h.depth);
        },
    }
}

/// Sample `i` of a row at `depth` bits.
fn sample(row: []const u8, i: usize, depth: u8) u16 {
    return switch (depth) {
        16 => std.mem.readInt(u16, row[2 * i ..][0..2], .big),
        8 => row[i],
        else => (row[i * depth / 8] >> @intCast(8 - depth - i * depth % 8)) & ((@as(u8, 1) << @intCast(depth)) - 1),
    };
}

/// An 8- or 16-bit sample as eight bits: the high byte.
fn high(row: []const u8, i: usize, depth: u8) u8 {
    return if (depth == 16) row[2 * i] else row[i];
}
