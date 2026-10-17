//! PNG to RGBA from the PNG specification (chunks, zlib, filters, Adam7), each layout widened as Pillow's mode for it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const picture = @import("picture.zig");
const tags = @import("tags.zig");

pub const signature = "\x89PNG\r\n\x1a\n";

const Header = struct { width: u32, height: u32, depth: u8, color: u8, interlaced: bool };

/// Adam7's passes as (first column, first row, column step, row step); a plain image is one pass over every pixel.
const adam7 = [7][4]u8{ .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 }, .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 } };
const whole = [1][4]u8{.{ 0, 0, 1, 1 }};

/// Pillow's bounds on text: one compressed chunk's 1 MiB, and 64 MiB in all.
const text_chunk_most = 1 << 20;
const text_most = 64 << 20;

/// What the chunks around the image data said, as Pillow's PngStream kept it.
const Info = struct {
    hdr: ?Header = null,
    palette: []const u8 = &.{},
    trns: ?[]const u8 = null,
    exif: ?[]const u8 = null,
    exif_text: bool = false, // a compressed or international text chunk named "exif": Pillow failed reading it
    profile: ?[]const u8 = null, // the "Raw profile type exif" text
    xmp: ?[]const u8 = null,
    frames: ?u32 = null, // acTL's frame count while it is valid
    frame_box: bool = false, // an fcTL came before the image data
    text: usize = 0,
};

const Chunk = struct { kind: *const [4]u8, body: []const u8, next: usize };

/// The chunk at `at`, its checksum checked; null at IEND, whose body and checksum Pillow never read.
fn chunk(data: []const u8, at: usize) error{Invalid}!?Chunk {
    if (data.len - at < 8) return error.Invalid;
    const len = std.mem.readInt(u32, data[at..][0..4], .big);
    const kind = data[at + 4 ..][0..4];
    for (kind) |c| if (!std.ascii.isAlphanumeric(c) and c != '_') return error.Invalid;
    if (std.mem.eql(u8, kind, "IEND")) return null;
    if (len > data.len - at - 8 or data.len - at - 8 - len < 4) return error.Invalid;
    const crc = std.mem.readInt(u32, data[at + 8 + len ..][0..4], .big);
    if (std.hash.Crc32.hash(data[at + 4 .. at + 8 + len]) != crc) return error.Invalid;
    return .{ .kind = kind, .body = data[at + 8 ..][0..len], .next = at + 12 + len };
}

pub fn decode(a: Allocator, data: []const u8) picture.Error!picture.Samples {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var info: Info = .{};
    // the header chunks, up to the first image data (0.6.6's Image.open)
    var at: usize = signature.len;
    const first = while (true) {
        // the image data's own checksum is read later, by verify
        if (data.len - at >= 8 and std.mem.eql(u8, data[at + 4 ..][0..4], "IDAT")) break at;
        const c = (try chunk(data, at)) orelse return error.Invalid;
        if (std.mem.eql(u8, c.kind, "fdAT")) return error.Invalid; // frame data before any image data
        try header(arena, &info, c);
        at = c.next;
    };
    const h = info.hdr orelse return error.Invalid;
    const bits = bitsPerPixel(h);
    if (bits == 0) return error.Invalid; // no mode: Pillow's open identified no image
    try picture.fits(h.width, h.height);
    const frames = if (info.frames) |n| @as(u64, n) + @intFromBool(!info.frame_box) else 1;
    if (frames != 1) return error.Animated;
    // every chunk to IEND checked (verify), the consecutive image data gathered
    var idat: std.ArrayList(u8) = .empty;
    at = first;
    var after: ?usize = null;
    while (try chunk(data, at)) |c| : (at = c.next) {
        if (after == null and std.mem.eql(u8, c.kind, "IDAT")) {
            try idat.appendSlice(arena, c.body);
        } else if (after == null) after = at;
    }
    const rgba = try pixels(a, arena, h, bits, &info, idat.items);
    errdefer a.free(rgba);
    // the chunks after the image data (load_end), then the orientation the tags give
    if (after) |from| {
        at = from;
        while (try chunk(data, at)) |c| : (at = c.next) try trailer(arena, &info, c);
    }
    if (info.exif_text) return error.Invalid; // Pillow's EXIF reader raised on a decoded text string
    const exif = info.exif orelse if (info.profile) |p| try profileBytes(arena, p) else null;
    const o = try tags.orientation(exif, info.xmp, .refuse);
    return .{ .width = h.width, .height = h.height, .channels = 4, .data = rgba, .orientation = o };
}

/// A chunk before the image data, as Pillow's handler took it; the mode is the one the header said so far.
fn header(arena: Allocator, info: *Info, c: Chunk) picture.Error!void {
    const k = c.kind;
    const b = c.body;
    if (std.mem.eql(u8, k, "IHDR")) {
        if (b.len < 13) return error.Invalid;
        if (b[11] != 0) return error.Invalid; // an unknown filter method
        const width = std.mem.readInt(u32, b[0..4], .big);
        const h: Header = .{ .width = width, .height = std.mem.readInt(u32, b[4..8], .big), .depth = b[8], .color = b[9], .interlaced = b[12] != 0 };
        // an unknown depth and colour pair keeps the mode an earlier header gave; interlacing, once set, stays
        const was = info.hdr orelse h;
        const mode = if (bitsPerPixel(h) != 0) h else was;
        info.hdr = .{ .width = h.width, .height = h.height, .depth = mode.depth, .color = mode.color, .interlaced = h.interlaced or was.interlaced };
    } else if (std.mem.eql(u8, k, "PLTE")) {
        if (info.hdr) |h| if (h.color == 3) {
            info.palette = b;
        };
    } else if (std.mem.eql(u8, k, "acTL")) {
        if (b.len < 8) return error.Invalid;
        if (info.frames != null) {
            info.frames = null; // a second acTL: Pillow kept the default image alone
        } else {
            const n = std.mem.readInt(u32, b[0..4], .big);
            if (n != 0 and n <= 0x80000000) info.frames = n;
        }
    } else if (std.mem.eql(u8, k, "fcTL")) {
        try frameControl(info, b);
    } else try trailer(arena, info, c);
}

/// The default image's frame control: Pillow drew the image data into its box, so only the whole image is accepted.
fn frameControl(info: *Info, b: []const u8) error{Invalid}!void {
    if (b.len < 26) return error.Invalid;
    const h = info.hdr orelse return error.Invalid;
    if (info.frame_box or std.mem.readInt(u32, b[0..4], .big) != 0) return error.Invalid;
    const w = std.mem.readInt(u32, b[4..8], .big);
    const ht = std.mem.readInt(u32, b[8..12], .big);
    const x = std.mem.readInt(u32, b[12..16], .big);
    const y = std.mem.readInt(u32, b[16..20], .big);
    if (x != 0 or y != 0 or w != h.width or ht != h.height) return error.Invalid;
    info.frame_box = true;
}

/// A chunk Pillow handled the same before and after the image data: transparency, EXIF and text.
fn trailer(arena: Allocator, info: *Info, c: Chunk) picture.Error!void {
    const k = c.kind;
    const b = c.body;
    if (std.mem.eql(u8, k, "tRNS")) {
        const h = info.hdr orelse return;
        switch (h.color) {
            0 => if (b.len < 2) return error.Invalid,
            2 => if (b.len < 6) return error.Invalid,
            3 => {},
            else => return,
        }
        info.trns = b;
    } else if (std.mem.eql(u8, k, "eXIf")) {
        info.exif = b;
        info.exif_text = false;
    } else if (std.mem.eql(u8, k, "tEXt")) {
        const key, const value = splitKey(b);
        if (key.len > 0) try text(info, key, value, value.len, false);
    } else if (std.mem.eql(u8, k, "zTXt")) {
        const key, const rest = splitKey(b);
        if (rest.len > 0 and rest[0] != 0) return error.Invalid; // an unknown compression method
        const value = (try inflate(arena, if (rest.len > 0) rest[1..] else rest)) orelse "";
        if (key.len > 0) try text(info, key, value, value.len, true);
    } else if (std.mem.eql(u8, k, "iTXt")) {
        const key = b[0 .. std.mem.indexOfScalar(u8, b, 0) orelse return];
        const r = b[key.len + 1 ..];
        if (r.len < 2) return;
        const lang_end = std.mem.indexOfScalar(u8, r[2..], 0) orelse return;
        const tk_end = std.mem.indexOfScalarPos(u8, r[2..], lang_end + 1, 0) orelse return;
        var value = r[2..][tk_end + 1 ..];
        if (r[0] != 0) {
            if (r[1] != 0) return;
            value = (try inflate(arena, value)) orelse return;
        }
        if (std.mem.eql(u8, key, "XML:com.adobe.xmp")) info.xmp = value;
        if (!std.unicode.utf8ValidateSlice(r[2..][0..lang_end]) or !std.unicode.utf8ValidateSlice(r[2..][lang_end + 1 .. tk_end])) return;
        const chars = std.unicode.utf8CountCodepoints(value) catch return;
        try text(info, key, value, chars, true);
    }
}

/// A text chunk's keyword and value at its first zero byte (the whole chunk is the keyword without one).
fn splitKey(b: []const u8) struct { []const u8, []const u8 } {
    const z = std.mem.indexOfScalar(u8, b, 0) orelse return .{ b, "" };
    return .{ b[0..z], b[z + 1 ..] };
}

/// One text value under Pillow's 64 MiB bound; the keywords the orientation is read from are kept.
fn text(info: *Info, key: []const u8, value: []const u8, chars: usize, decoded: bool) error{Invalid}!void {
    if (std.mem.eql(u8, key, "exif")) {
        info.exif = value;
        info.exif_text = decoded; // Pillow kept a text string there, which its EXIF reader refused
    } else if (std.mem.eql(u8, key, "Raw profile type exif")) {
        info.profile = value;
    } else if (std.mem.eql(u8, key, "XML:com.adobe.xmp")) {
        info.xmp = value;
    }
    info.text += chars;
    if (info.text > text_most) return error.Invalid;
}

/// A zlib text body; null when it does not inflate (Pillow kept no text), error.Invalid past 1 MiB.
fn inflate(arena: Allocator, body: []const u8) picture.Error!?[]const u8 {
    var in: std.Io.Reader = .fixed(body);
    const window = try arena.alloc(u8, std.compress.flate.max_window_len);
    var z: std.compress.flate.Decompress = .init(&in, .zlib, window);
    return z.reader.allocRemaining(arena, .limited(text_chunk_most)) catch |err| switch (err) {
        error.StreamTooLong => error.Invalid,
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

/// "Raw profile type exif": hex after the first three lines, whitespace ignored, as bytes.fromhex read it.
fn profileBytes(arena: Allocator, p: []const u8) picture.Error![]const u8 {
    var lines = std.mem.splitScalar(u8, p, '\n');
    for (0..3) |_| _ = lines.next() orelse return "";
    const hex = lines.rest();
    var out: std.ArrayList(u8) = .empty;
    var hi: ?u8 = null;
    for (hex) |ch| {
        if (ch == '\n') continue;
        if (std.ascii.isWhitespace(ch)) {
            if (hi != null) return error.Invalid;
            continue;
        }
        const v = std.fmt.charToDigit(ch, 16) catch return error.Invalid;
        if (hi) |x| {
            try out.append(arena, x << 4 | v);
            hi = null;
        } else hi = v;
    }
    if (hi != null) return error.Invalid;
    return out.items;
}

/// Bits a pixel for the depth and colour pairs Pillow opens; 0 for any other pair.
fn bitsPerPixel(h: Header) usize {
    const ok = switch (h.color) {
        0 => h.depth == 1 or h.depth == 2 or h.depth == 4 or h.depth == 8 or h.depth == 16,
        3 => h.depth == 1 or h.depth == 2 or h.depth == 4 or h.depth == 8,
        2, 4, 6 => h.depth == 8 or h.depth == 16,
        else => false,
    };
    const channels: usize = switch (h.color) {
        0, 3 => 1,
        4 => 2,
        2 => 3,
        6 => 4,
        else => 0,
    };
    return if (ok) channels * h.depth else 0;
}

fn passSize(h: Header, p: [4]u8) [2]usize {
    const w: usize = h.width;
    const ht: usize = h.height;
    return .{ (w + p[2] - 1 - p[0]) / p[2], (ht + p[3] - 1 - p[1]) / p[3] };
}

/// The image data inflated, each row unfiltered and widened to RGBA in its place.
fn pixels(a: Allocator, arena: Allocator, h: Header, bits: usize, info: *const Info, idat: []const u8) picture.Error![]u8 {
    if (idat.len == 0) return error.Invalid;
    if (info.palette.len / 3 > 256) return error.Invalid; // Pillow's putpalette refused more than 256 entries
    const passes: []const [4]u8 = if (h.interlaced) &adam7 else &whole;
    var size: usize = 0;
    for (passes) |p| {
        const pw, const ph = passSize(h, p);
        if (pw > 0 and ph > 0) size += ph * (1 + (pw * bits + 7) / 8);
    }
    const raw = try arena.alloc(u8, size);
    var in: std.Io.Reader = .fixed(idat);
    var z: std.compress.flate.Decompress = .init(&in, .zlib, &.{});
    var out: std.Io.Writer = .fixed(raw);
    z.reader.streamExact(&out, size) catch return error.Invalid;
    const width: usize = h.width;
    const rgba = try a.alloc(u8, width * @as(usize, h.height) * 4);
    errdefer a.free(rgba);
    const keys = Keys.of(h, info.trns);
    var rest = raw;
    for (passes) |p| {
        const pw, const ph = passSize(h, p);
        if (pw == 0 or ph == 0) continue;
        const stride = (pw * bits + 7) / 8;
        var prev: ?[]const u8 = null;
        for (0..ph) |y| {
            const row = rest[1..][0..stride];
            try unfilter(rest[0], row, prev, @max(1, bits / 8));
            for (0..pw) |x| widen(h, row, x, info, keys, rgba[((p[1] + y * p[3]) * width + p[0] + x * p[2]) * 4 ..][0..4]);
            prev = row;
            rest = rest[1 + stride ..];
        }
    }
    return rgba;
}

/// The PNG filters reversed: None, Sub, Up, Average and Paeth, `bpp` bytes back for the pixel to the left.
fn unfilter(kind: u8, row: []u8, prev: ?[]const u8, bpp: usize) error{Invalid}!void {
    if (kind > 4) return error.Invalid;
    for (row, 0..) |*v, i| {
        const left: u8 = if (i >= bpp) row[i - bpp] else 0;
        const up: u8 = if (prev) |p| p[i] else 0;
        const corner: u8 = if (prev != null and i >= bpp) prev.?[i - bpp] else 0;
        v.* +%= switch (kind) {
            1 => left,
            2 => up,
            3 => @intCast((@as(u16, left) + up) >> 1),
            4 => paeth(left, up, corner),
            else => 0,
        };
    }
}

/// The predictor nearest to left + up - corner, ties in that order.
fn paeth(left: u8, up: u8, corner: u8) u8 {
    const p = @as(i16, left) + up - corner;
    const dl = @abs(p - left);
    const du = @abs(p - up);
    const dc = @abs(p - corner);
    if (dl <= du and dl <= dc) return left;
    return if (du <= dc) up else corner;
}

/// The transparency tRNS gave for a gray or RGB image, compared as Pillow did: with the 8-bit values of its mode.
const Keys = struct {
    gray: ?u16 = null,
    rgb: ?[3]u16 = null,

    fn of(h: Header, trns: ?[]const u8) Keys {
        const t = trns orelse return .{};
        const k = struct {
            fn at(b: []const u8, i: usize) u16 {
                return std.mem.readInt(u16, b[2 * i ..][0..2], .big);
            }
        }.at;
        return switch (h.color) {
            0 => if (t.len < 2) .{} else .{ .gray = if (h.depth == 1) @as(u16, if (k(t, 0) != 0) 255 else 0) else k(t, 0) },
            2 => if (t.len < 6) .{} else .{ .rgb = .{ k(t, 0), k(t, 1), k(t, 2) } },
            else => .{},
        };
    }
};

/// Pixel `x` of an unfiltered row as Pillow's mode for the layout holds it, as RGBA.
fn widen(h: Header, row: []const u8, x: usize, info: *const Info, keys: Keys, px: *[4]u8) void {
    switch (h.color) {
        0 => {
            const s = sample(row, x, h.depth);
            const v: u8 = switch (h.depth) {
                1 => if (s != 0) 255 else 0, // mode 1
                2 => @intCast(s * 85), // L;2
                4 => @intCast(s * 17), // L;4
                8 => @intCast(s),
                else => @intCast(@min(s, 255)), // I;16 to RGB clips
            };
            px.* = .{ v, v, v, if (keys.gray != null and keys.gray.? == v) 0 else 255 };
        },
        2 => {
            for (px[0..3], 0..) |*c, i| c.* = high(row, 3 * x + i, h.depth);
            const keyed = if (keys.rgb) |k| k[0] == px[0] and k[1] == px[1] and k[2] == px[2] else false;
            px[3] = if (keyed) 0 else 255;
        },
        3 => {
            const i = sample(row, x, h.depth);
            const p = info.palette;
            px.* = if (3 * @as(usize, i) + 3 <= p.len) .{ p[3 * i], p[3 * i + 1], p[3 * i + 2], 255 } else .{ 0, 0, 0, 255 };
            if (info.trns) |t| if (i < t.len) {
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

/// Sample `i` of a row at `depth` bits, most significant bits first.
fn sample(row: []const u8, i: usize, depth: u8) u16 {
    return switch (depth) {
        16 => std.mem.readInt(u16, row[2 * i ..][0..2], .big),
        8 => row[i],
        else => blk: {
            const bit = i * depth;
            const shift: u3 = @intCast(8 - depth - bit % 8);
            break :blk (row[bit / 8] >> shift) & ((@as(u8, 1) << @intCast(depth)) - 1);
        },
    };
}

/// An 8- or 16-bit channel as 8 bits: a 16-bit one's high byte.
fn high(row: []const u8, i: usize, depth: u8) u8 {
    return if (depth == 16) row[2 * i] else row[i];
}
