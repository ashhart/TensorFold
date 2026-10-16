//! An image's bytes to Step-5's tower inputs as vLLM's Step3-VL processor makes them: crops of 504, then an overview.
//! Checked against vLLM dd944845777b (vllm/transformers_utils/processors/step3_vl.py), the 728-pixel boundary included.
const std = @import("std");
const img = @import("../glm/image.zig");

pub const Error = img.Error || error{TooManyTokens};

/// The processor's constants: sides, tokens a piece, the CLIP mean and std, the longest side kept, and token ids.
pub const overview = 728;
pub const crop = 504;
pub const patch = 14;
pub const patch_width = 3 * patch * patch; // a patch's values: channel, row, column (conv1's weight order)
pub const overview_tokens = 169; // 52 x 52 patches, twice downsampled by 2
pub const crop_tokens = 81; // 36 x 36
const max_side = 3024;
const mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const std_dev = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
/// One tower input (engine_api.Piece's fields): patches, grid, rows, and the tokens around its rows.
pub const Piece = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, before: []const u32 = &.{}, after: []const u32 = &.{} };
/// An image as tower inputs (engine_api.PreparedImage's fields): its rows in all, a content hash, its pieces.
pub const Prepared = struct { tokens: u32, hash: u64, pieces: []const Piece };
pub const Ids = struct { im_start: u32 = 128000, im_patch: u32 = 128001, im_end: u32 = 128002, patch_start: u32 = 128003, patch_newline: u32 = 128004, patch_end: u32 = 128005 };

/// The crops' window (ImagePatcher.determine_window_size): 0 for none.
pub fn window(long: u32, short: u32) u32 {
    const r = @as(f64, @floatFromInt(long)) / @as(f64, @floatFromInt(short));
    if (long < 728) return if (r > 1.5) short else 0;
    return if (r > 4) @min(short, crop) else crop;
}

/// get_image_size_for_crop: each side to a whole number of windows when it passes one by more than 0.2 of a window.
pub fn cropSize(side: u32, win: u32) u32 {
    const ratio = @as(f64, @floatFromInt(side)) / @as(f64, @floatFromInt(win));
    if (ratio < 1) return side;
    const whole: u32 = side / win;
    const n = if (ratio - @as(f64, @floatFromInt(whole)) > 0.2) whole + 1 else whole;
    return win * n;
}

/// slide_window's starts along one side (windows a step apart, the last moved back inside).
fn starts(gpa: std.mem.Allocator, side: u32, win: u32) Error![]u32 {
    const n: u32 = if (side <= win) 1 else @intFromFloat(@ceil(@as(f64, @floatFromInt(side - win)) / @as(f64, @floatFromInt(win)) + 1));
    const out = try gpa.alloc(u32, n);
    for (out, 0..) |*s, i| s.* = @intCast(i * win);
    if (n > 1 and out[n - 1] + win > side) out[n - 1] = side - win;
    return out;
}

/// The image's bytes to its pieces (crops in row order, then the overview) with the tokens around each.
pub fn prepare(gpa: std.mem.Allocator, bytes: []const u8, ids: Ids, max_tokens: u32) Error!Prepared {
    var src = try img.decode(gpa, bytes);
    defer src.deinit(gpa);
    var w = src.w;
    var h = src.h;
    const ratio = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    var base = src;
    var padded: ?img.Rgb = null;
    defer if (padded) |p| p.deinit(gpa);
    if (@min(w, h) < 32 and (ratio > 4 or ratio < 0.25)) { // square_pad: black, the image at the top left
        const side = @max(w, h);
        const px = try gpa.alloc(u8, @as(usize, side) * side * 3);
        @memset(px, 0);
        for (0..h) |y| @memcpy(px[y * side * 3 ..][0 .. w * 3], src.px[y * w * 3 ..][0 .. w * 3]);
        padded = .{ .w = side, .h = side, .px = px };
        base = padded.?;
        w = side;
        h = side;
    }
    if (@max(w, h) > max_side) { // int(side * scale), as Python truncates
        const s = @as(f64, max_side) / @as(f64, @floatFromInt(@max(w, h)));
        w = @intFromFloat(@as(f64, @floatFromInt(w)) * s);
        h = @intFromFloat(@as(f64, @floatFromInt(h)) * s);
    }
    var scaled = try img.resizeWith(gpa, base, w, h, img.bilinear_filter);
    defer scaled.deinit(gpa);
    var pieces: std.ArrayList(Piece) = .empty;
    const win = window(@max(w, h), @min(w, h));
    if (win > 0) {
        const cw = cropSize(w, win);
        const ch = cropSize(h, win);
        var grid = if (cw != w or ch != h) try img.resizeWith(gpa, scaled, cw, ch, img.bilinear_filter) else scaled;
        defer if (cw != w or ch != h) grid.deinit(gpa);
        const xs = try starts(gpa, cw, win);
        defer gpa.free(xs);
        const ys = try starts(gpa, ch, win);
        defer gpa.free(ys);
        for (ys, 0..) |y0, r| for (xs, 0..) |x0, c| {
            var cut = try cropRgb(gpa, grid, x0, y0, win);
            defer cut.deinit(gpa);
            const last_of_row = c + 1 == xs.len and r + 1 < ys.len;
            try pieces.append(gpa, .{
                .pixels = try patches(gpa, cut, crop),
                .gh = crop / patch,
                .gw = crop / patch,
                .tokens = crop_tokens,
                .before = try gpa.dupe(u32, &.{ids.patch_start}),
                .after = try gpa.dupe(u32, if (last_of_row) &.{ ids.patch_end, ids.patch_newline } else &.{ids.patch_end}),
            });
        };
    }
    try pieces.append(gpa, .{
        .pixels = try patches(gpa, scaled, overview),
        .gh = overview / patch,
        .gw = overview / patch,
        .tokens = overview_tokens,
        .before = try gpa.dupe(u32, &.{ids.im_start}),
        .after = try gpa.dupe(u32, &.{ids.im_end}),
    });
    var tokens: u32 = 0;
    for (pieces.items) |p| tokens += p.tokens;
    if (tokens > max_tokens) return error.TooManyTokens;
    return .{ .tokens = tokens, .hash = std.hash.Wyhash.hash(0x5735, bytes), .pieces = pieces.items };
}

/// PIL's crop: a side x side window at (x0, y0), black past the image.
fn cropRgb(gpa: std.mem.Allocator, src: img.Rgb, x0: u32, y0: u32, side: u32) Error!img.Rgb {
    const px = try gpa.alloc(u8, @as(usize, side) * side * 3);
    @memset(px, 0);
    for (0..side) |y| {
        if (y0 + y >= src.h) break;
        const n = @min(side, src.w -| x0);
        @memcpy(px[y * side * 3 ..][0 .. n * 3], src.px[((y0 + y) * src.w + x0) * 3 ..][0 .. n * 3]);
    }
    return .{ .w = side, .h = side, .px = px };
}

/// ToTensor, Normalize(mean, std), then Resize((side, side), bilinear, antialias) on the floats, as patches.
fn patches(gpa: std.mem.Allocator, src: img.Rgb, side: u32) Error![]f32 {
    const plane = try gpa.alloc(f32, @as(usize, src.w) * src.h * 3);
    defer gpa.free(plane);
    for (0..@as(usize, src.w) * src.h) |i| for (0..3) |c| {
        plane[c * src.w * src.h + i] = (@as(f32, @floatFromInt(src.px[i * 3 + c])) / 255.0 - mean[c]) / std_dev[c];
    };
    const out = try resizeFloat(gpa, plane, src.w, src.h, side);
    defer gpa.free(out);
    const g = side / patch;
    const p = try gpa.alloc(f32, @as(usize, g) * g * patch_width);
    for (0..g) |py| for (0..g) |px_| for (0..3) |c| for (0..patch) |y| for (0..patch) |x| {
        p[((py * g + px_) * 3 + c) * patch * patch + y * patch + x] = out[(c * side + py * patch + y) * side + px_ * patch + x];
    };
    return p;
}

/// torch's antialiased bilinear resize of 3 float planes to side x side: PIL's weights, unrounded, width then height.
fn resizeFloat(gpa: std.mem.Allocator, plane: []const f32, w: u32, h: u32, side: u32) Error![]f32 {
    const fw = try img.weights(gpa, w, side, img.bilinear_filter);
    defer gpa.free(fw.bounds);
    defer gpa.free(fw.w);
    const fh = try img.weights(gpa, h, side, img.bilinear_filter);
    defer gpa.free(fh.bounds);
    defer gpa.free(fh.w);
    const mid = try gpa.alloc(f32, @as(usize, side) * h * 3);
    defer gpa.free(mid);
    for (0..3) |c| for (0..h) |y| for (0..side) |xx| {
        var s: f64 = 0;
        const x0 = fw.bounds[xx * 2];
        for (0..fw.bounds[xx * 2 + 1]) |k| s += fw.w[xx * fw.ksize + k] * plane[(c * h + y) * w + x0 + k];
        mid[(c * h + y) * side + xx] = @floatCast(s);
    };
    const out = try gpa.alloc(f32, @as(usize, side) * side * 3);
    for (0..3) |c| for (0..side) |yy| for (0..side) |x| {
        var s: f64 = 0;
        const y0 = fh.bounds[yy * 2];
        for (0..fh.bounds[yy * 2 + 1]) |k| s += fh.w[yy * fh.ksize + k] * mid[(c * h + y0 + k) * side + x];
        out[(c * side + yy) * side + x] = @floatCast(s);
    };
    return out;
}

test "the patcher's windows and crop sizes" {
    try std.testing.expectEqual(@as(u32, 0), window(700, 600));
    try std.testing.expectEqual(@as(u32, 300), window(700, 300));
    try std.testing.expectEqual(@as(u32, 504), window(1600, 900));
    try std.testing.expectEqual(@as(u32, 200), window(3000, 200));
    try std.testing.expectEqual(@as(u32, 1512), cropSize(1600, 504)); // 3.17 windows: 3
    try std.testing.expectEqual(@as(u32, 1008), cropSize(900, 504)); // 1.79: 2
    try std.testing.expectEqual(@as(u32, 400), cropSize(400, 504));
}
