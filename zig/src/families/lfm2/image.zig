//! LFM2-VL pictures on the CPU: data URL to RGB, then Lfm2VlImageProcessor's tiles, patches and placeholder ids.

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core");
const Tokenizer = core.tokenizer.Tokenizer;
const picture = core.picture;
const resample = picture.resample;

// tensorfold's ImageLimits for one picture's encoded bytes
const max_bytes = 10 << 20;

// d1-3B's processor_config.json (Lfm2VlImageProcessor), and the d1 recipe's bound on a picture's pixels
const patch = 16; // encoder_patch_size
const downsample = 2; // downsample_factor
const tile = 512; // tile_size
const min_tokens = 64; // min_image_tokens
const max_tokens = 256; // max_image_tokens
const tolerance = 2.0; // max_pixels_tolerance
const cap_pixels = 1024 * 1024; // the d1 recipe's MAX_PIXELS

pub const max_grid = 10; // max_tiles: a grid is at most 10 tiles a side
pub const patch_width = 3 * patch * patch;

/// The processor's candidate grids (columns, rows) for 2 to 10 tiles, in the order it tries them.
const grids = [_][2]u8{ .{ 1, 2 }, .{ 2, 1 }, .{ 3, 1 }, .{ 1, 3 }, .{ 2, 2 }, .{ 4, 1 }, .{ 1, 4 }, .{ 5, 1 }, .{ 1, 5 }, .{ 1, 6 }, .{ 6, 1 }, .{ 3, 2 }, .{ 2, 3 }, .{ 7, 1 }, .{ 1, 7 }, .{ 4, 2 }, .{ 2, 4 }, .{ 1, 8 }, .{ 8, 1 }, .{ 1, 9 }, .{ 3, 3 }, .{ 9, 1 }, .{ 2, 5 }, .{ 5, 2 }, .{ 10, 1 }, .{ 1, 10 } };

/// Oriented RGB rows, three bytes a pixel.
pub const Rgb = picture.Rgb;

/// A data URL's PNG, JPEG or WebP as 0.6.6 loaded it (core's picture decoder), or null with tensorfold's refusal.
pub fn load(io: std.Io, a: Allocator, url: []const u8, problem: *[]const u8) Allocator.Error!?Rgb {
    const bytes = (try dataBytes(a, url, problem)) orelse return null;
    defer a.free(bytes);
    return picture.decode(io, a, bytes) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        problem.* = picture.problem(err);
        return null;
    };
}

/// tensorfold's `_data_bytes`: an optional `;base64` and nothing else in the header; the payload's bytes.
fn dataBytes(a: Allocator, url: []const u8, problem: *[]const u8) Allocator.Error!?[]u8 {
    if (!std.mem.startsWith(u8, url, "data:")) {
        problem.* = if (std.mem.startsWith(u8, url, "https://"))
            "image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls"
        else
            "images require data URLs or public HTTPS URLs";
        return null;
    }
    if (url.len > max_bytes * 3 + 256) return refuse(problem, "image data URL exceeds the encoded byte limit");
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return refuse(problem, "invalid image data URL");
    if (comma > 256) return refuse(problem, "invalid image data URL");
    const media, const payload = .{ url[5..comma], url[comma + 1 ..] };
    const base64 = if (std.mem.indexOfScalar(u8, media, ';')) |semi| blk: {
        if (!std.ascii.eqlIgnoreCase(media[semi + 1 ..], "base64")) return refuse(problem, "image data URL supports only optional base64 encoding");
        break :blk true;
    } else false;
    const most: usize = if (base64) 4 * ((max_bytes + 2) / 3) else max_bytes * 3;
    if (payload.len > most) return refuse(problem, "image data URL exceeds the encoded byte limit");
    for (payload) |c| if (!std.ascii.isAscii(c)) return refuse(problem, "invalid image data URL encoding");
    const data = if (base64) blk: {
        const decoder = std.base64.standard.Decoder;
        const out = try a.alloc(u8, decoder.calcSizeForSlice(payload) catch return refuse(problem, "invalid image data URL encoding"));
        decoder.decode(out, payload) catch {
            a.free(out);
            return refuse(problem, "invalid image data URL encoding");
        };
        break :blk out;
    } else try unquote(a, payload);
    if (data.len == 0 or data.len > max_bytes) {
        a.free(data);
        return refuse(problem, "image is empty or exceeds the encoded byte limit");
    }
    return data;
}

fn refuse(problem: *[]const u8, why: []const u8) ?[]u8 {
    problem.* = why;
    return null;
}

/// urllib's unquote_to_bytes: `%XX` is a byte, any other `%` stays as written.
fn unquote(a: Allocator, text: []const u8) Allocator.Error![]u8 {
    var out = try a.alloc(u8, text.len);
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len) : (n += 1) {
        const byte = if (text[i] == '%' and i + 2 < text.len) hexByte(text[i + 1 ..][0..2]) else null;
        out[n] = byte orelse text[i];
        i += if (byte != null) 3 else 1;
    }
    return a.realloc(out, n);
}

fn hexByte(pair: *const [2]u8) ?u8 {
    const hi = std.fmt.charToDigit(pair[0], 16) catch return null;
    const lo = std.fmt.charToDigit(pair[1], 16) catch return null;
    return hi << 4 | lo;
}

/// A tile's patch rows and columns.
pub const Grid = struct {
    h: usize,
    w: usize,

    /// The image tokens the projector makes of it (2x2 patches each).
    pub fn tokens(g: Grid) usize {
        return (g.h + downsample - 1) / downsample * ((g.w + downsample - 1) / downsample);
    }
};

/// What the processor makes of one picture: normalized patches, each tile's grid, and the tile layout.
pub const Picture = struct {
    patches: []f32, // [n, patch_width], (y, x, channel) within a patch: the tiles row by row, the thumbnail last
    tiles: []Grid,
    rows: usize, // 1 x 1: the whole picture is one tile and there is no thumbnail
    cols: usize,

    pub fn deinit(p: *Picture, a: Allocator) void {
        a.free(p.patches);
        a.free(p.tiles);
        p.* = undefined;
    }

    /// The `<image>` tokens it takes in the prompt.
    pub fn tokens(p: *const Picture) usize {
        var n: usize = 0;
        for (p.tiles) |g| n += g.tokens();
        return n;
    }

    /// The ids one `<image>` expands to: start, each tile's mark and tokens, the thumbnail's, end.
    pub fn appendIds(p: *const Picture, a: Allocator, ids: *std.ArrayList(u32), m: *const Marks) Allocator.Error!void {
        try ids.append(a, m.start);
        if (p.tiles.len == 1) {
            try ids.appendNTimes(a, m.image, p.tiles[0].tokens());
        } else {
            for (p.tiles[0 .. p.tiles.len - 1], 0..) |g, i| {
                try ids.append(a, m.tiles[i / p.cols][i % p.cols]);
                try ids.appendNTimes(a, m.image, g.tokens());
            }
            try ids.append(a, m.thumbnail);
            try ids.appendNTimes(a, m.image, p.tiles[p.tiles.len - 1].tokens());
        }
        try ids.append(a, m.end);
    }
};

/// The processor's placeholder tokens: `<image>`, the picture's bounds, the thumbnail's and each tile's mark.
pub const Marks = struct {
    image: u32,
    start: u32,
    end: u32,
    thumbnail: u32,
    tiles: [max_grid][max_grid]u32, // <|img_row_{r+1}_col_{c+1}|>

    /// Every picture marker must be excluded from matching in request text.
    pub fn firstNonAdded(m: *const Marks, t: *const Tokenizer) ?u32 {
        for ([_]u32{ m.image, m.start, m.end, m.thumbnail }) |id| if (!t.isAdded(id)) return id;
        for (m.tiles) |row| for (row) |id| {
            if (!t.isAdded(id)) return id;
        };
        return null;
    }

    /// The marks from the tokenizer's special tokens, or null when one is missing.
    pub fn resolve(t: *const Tokenizer) ?Marks {
        var m: Marks = undefined;
        inline for (.{ .{ "image", "<image>" }, .{ "start", "<|image_start|>" }, .{ "end", "<|image_end|>" }, .{ "thumbnail", "<|img_thumbnail|>" } }) |f| {
            @field(m, f[0]) = t.specialTokenId(f[1]) orelse return null;
        }
        var name: [32]u8 = undefined;
        for (&m.tiles, 1..) |*row, r| for (row, 1..) |*id, c| {
            id.* = t.specialTokenId(std.fmt.bufPrint(&name, "<|img_row_{d}_col_{d}|>", .{ r, c }) catch unreachable) orelse return null;
        };
        return m;
    }
};

/// The size cap, then the processor: 512-pixel tiles and a thumbnail, or one tile; torch's uint8 bicubic.
pub fn prepare(a: Allocator, rgb: Rgb) Allocator.Error!Picture {
    const capped = try cap(a, rgb);
    defer if (capped) |c| a.free(c.pixels);
    const src = capped orelse rgb;
    const w, const h = .{ src.width, src.height };
    const sw, const sh = smartSize(w, h);
    const cols: usize, const rows: usize = if (tooLarge(w, h)) closestGrid(w, h) else .{ 1, 1 };
    const count = rows * cols + @intFromBool(rows * cols > 1);
    const tiles = try a.alloc(Grid, count);
    errdefer a.free(tiles);
    for (tiles[0 .. rows * cols]) |*g| g.* = .{ .h = tile / patch, .w = tile / patch };
    tiles[count - 1] = .{ .h = sh / patch, .w = sw / patch };
    var n: usize = 0;
    for (tiles) |g| n += g.h * g.w;
    const patches = try a.alloc(f32, n * patch_width);
    errdefer a.free(patches);
    var at: usize = 0;
    const view = resample.View.of(src.pixels, w, h);
    if (count > 1) {
        const canvas = try resample.resize(a, .torch, view, cols * tile, rows * tile);
        defer a.free(canvas);
        for (0..rows) |r| for (0..cols) |c| {
            at = cut(patches, at, canvas, cols * tile, c * tile, r * tile, tile, tile);
        };
    }
    const resized = try resample.resize(a, .torch, view, sw, sh);
    defer a.free(resized);
    _ = cut(patches, at, resized, sw, 0, 0, sw, sh);
    return .{ .patches = patches, .tiles = tiles, .rows = rows, .cols = cols };
}

/// A picture over `cap_pixels` shrunk to fit with Pillow's bicubic; null when it already fits.
fn cap(a: Allocator, rgb: Rgb) Allocator.Error!?Rgb {
    if (rgb.width * rgb.height <= cap_pixels) return null;
    const s = @sqrt(@as(f64, cap_pixels) / @as(f64, @floatFromInt(rgb.width * rgb.height)));
    const w = @max(1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(rgb.width)) * s)));
    const h = @max(1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(rgb.height)) * s)));
    return .{ .width = w, .height = h, .pixels = try resample.resize(a, .pillow, .of(rgb.pixels, rgb.width, rgb.height), w, h) };
}

/// round(x / factor) * factor, Python's round: halves go to even.
fn roundBy(x: usize, factor: usize) usize {
    const q = @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(factor));
    const r = if (q - @floor(q) == 0.5) 2 * @round(q / 2) else @round(q);
    return @as(usize, @intFromFloat(r)) * factor;
}

/// smart_resize: (width, height), multiples of 32 holding 64 to 256 image tokens, near the picture's aspect.
fn smartSize(w: usize, h: usize) [2]usize {
    const f = patch * downsample;
    const fw: f64, const fh: f64 = .{ @floatFromInt(w), @floatFromInt(h) };
    var hb = @max(f, roundBy(h, f));
    var wb = @max(f, roundBy(w, f));
    const most = max_tokens * patch * patch * downsample * downsample;
    const least = min_tokens * patch * patch * downsample * downsample;
    if (hb * wb > most) {
        const beta = @sqrt(fh * fw / most);
        hb = @max(f, @as(usize, @intFromFloat(@floor(fh / beta / f))) * f);
        wb = @max(f, @as(usize, @intFromFloat(@floor(fw / beta / f))) * f);
    } else if (hb * wb < least) {
        const beta = @sqrt(@as(f64, least) / (fh * fw));
        hb = @as(usize, @intFromFloat(@ceil(fh * beta / f))) * f;
        wb = @as(usize, @intFromFloat(@ceil(fw * beta / f))) * f;
    }
    return .{ wb, hb };
}

/// _is_image_too_large: past the single-tile pixel budget times the tolerance.
fn tooLarge(w: usize, h: usize) bool {
    const f = patch * downsample;
    const area: f64 = @floatFromInt(@max(patch, roundBy(h, f)) * @max(patch, roundBy(w, f)));
    return area > max_tokens * patch * patch * downsample * downsample * tolerance;
}

/// find_closest_aspect_ratio over `grids`: (columns, rows); a tie goes to a later grid the picture fills over half of.
fn closestGrid(w: usize, h: usize) [2]usize {
    const aspect = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    var best: [2]usize = .{ 1, 1 };
    var best_diff = std.math.inf(f64);
    for (grids) |g| {
        const diff = @abs(aspect - @as(f64, @floatFromInt(g[0])) / @as(f64, @floatFromInt(g[1])));
        if (diff < best_diff) {
            best_diff = diff;
            best = .{ g[0], g[1] };
        } else if (diff == best_diff and @as(f64, @floatFromInt(w * h)) > 0.5 * @as(f64, @floatFromInt(tile * tile * @as(usize, g[0]) * g[1]))) {
            best = .{ g[0], g[1] };
        }
    }
    return best;
}

/// The (x0, y0, tw, th) region of an RGB image of width `w` as normalized patches at `patches[at..]`; the next `at`.
fn cut(patches: []f32, at: usize, rgb: []const u8, w: usize, x0: usize, y0: usize, tw: usize, th: usize) usize {
    const normalized = comptime blk: {
        var t: [256]f32 = undefined;
        for (&t, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) - 127.5) / 127.5;
        break :blk t;
    };
    var out = at;
    for (0..th / patch) |py| for (0..tw / patch) |px| {
        for (0..patch) |r| {
            const row = rgb[((y0 + py * patch + r) * w + x0 + px * patch) * 3 ..][0 .. 3 * patch];
            for (patches[out + r * 3 * patch ..][0 .. 3 * patch], row) |*p, v| p.* = normalized[v];
        }
        out += patch_width;
    };
    return out;
}

test {
    _ = @import("image_test.zig");
}

test "vision markers: every LFM2 picture and tile marker must be added" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var added: [104]core.tokenizer.AddedToken = undefined;
    for (&added, 0..) |*token, i| token.* = .{ .id = @intCast(i + 300), .content = "fixture", .special = false };
    var tok = try core.tokenizer.parse(a, "{\"model\":{\"type\":\"WordLevel\",\"unk_token\":\"?\",\"vocab\":{\"?\":0}}}");
    defer tok.deinit();
    tok.added = &added;
    var marks: Marks = .{ .image = 300, .start = 301, .end = 302, .thumbnail = 303, .tiles = undefined };
    var id: u32 = 304;
    for (&marks.tiles) |*row| for (row) |*slot| {
        slot.* = id;
        id += 1;
    };
    try std.testing.expectEqual(@as(?u32, null), marks.firstNonAdded(&tok));
    for (0..added.len) |i| {
        added[i].id += 1000;
        try std.testing.expectEqual(@as(?u32, @intCast(i + 300)), marks.firstNonAdded(&tok));
        added[i].id -= 1000;
    }
    std.debug.print("VISION_LFM2 markers={d}; every omitted entry refused\n", .{added.len});
}
