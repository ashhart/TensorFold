//! Native macOS ImageIO + CoreGraphics image decoder & VLM patch preprocessor (zero external deps).
//!
//! Decodes PNG, JPEG, WebP, HEIC, GIF, TIFF, BMP from local file paths, `file://` URIs, or
//! base64 `data:image/...;base64,...` strings, resizes with CoreGraphics high-quality interpolation,
//! and packs normalized BF16 patches for Qwen3.5-VL and Gemma 3-VL (SigLIP).
const std = @import("std");
const st = @import("safetensors.zig");

const CFDataRef = *const anyopaque;
const CGImageSourceRef = *const anyopaque;
const CGImageRef = *const anyopaque;
const CGColorSpaceRef = *const anyopaque;
const CGContextRef = *const anyopaque;

const CGRect = extern struct {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
};

const kCGImageAlphaPremultipliedLast: u32 = 1;
const kCGBitmapByteOrder32Big: u32 = 4 << 12;
const kCGInterpolationHigh: c_int = 3;

extern "c" fn CFDataCreate(allocator: ?*const anyopaque, bytes: [*]const u8, length: isize) ?CFDataRef;
extern "c" fn CFRelease(cf: ?*const anyopaque) void;
extern "c" fn CGImageSourceCreateWithData(data: CFDataRef, options: ?*const anyopaque) ?CGImageSourceRef;
extern "c" fn CGImageSourceCreateImageAtIndex(src: CGImageSourceRef, index: usize, options: ?*const anyopaque) ?CGImageRef;
extern "c" fn CGImageGetWidth(img: CGImageRef) usize;
extern "c" fn CGImageGetHeight(img: CGImageRef) usize;
extern "c" fn CGImageRelease(img: ?CGImageRef) void;
extern "c" fn CGColorSpaceCreateDeviceRGB() ?CGColorSpaceRef;
extern "c" fn CGColorSpaceRelease(cs: ?CGColorSpaceRef) void;
extern "c" fn CGBitmapContextCreate(
    data: ?*anyopaque,
    width: usize,
    height: usize,
    bits_per_component: usize,
    bytes_per_row: usize,
    space: CGColorSpaceRef,
    bitmap_info: u32,
) ?CGContextRef;
extern "c" fn CGContextSetInterpolationQuality(ctx: CGContextRef, quality: c_int) void;
extern "c" fn CGContextDrawImage(ctx: CGContextRef, rect: CGRect, img: CGImageRef) void;
extern "c" fn CGContextRelease(ctx: ?CGContextRef) void;

pub const RgbImage = struct {
    width: u32,
    height: u32,
    /// Packed RGB u8 [height * width * 3]
    pixels: []u8,

    pub fn deinit(self: RgbImage, a: std.mem.Allocator) void {
        a.free(self.pixels);
    }
};

inline fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7FFF +% lsb) >> 16);
}

/// Load raw image file bytes from a local path, `file://` URI, or `data:image/...;base64,...` URI.
pub fn loadImageSource(a: std.mem.Allocator, source: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, source, "data:")) {
        const comma = std.mem.indexOfScalar(u8, source, ',') orelse return error.InvalidDataUri;
        const b64 = source[comma + 1 ..];
        // Strip any whitespace/newlines inside base64 payload
        var clean: std.ArrayList(u8) = .empty;
        defer clean.deinit(a);
        try clean.ensureTotalCapacity(a, b64.len);
        for (b64) |ch| {
            if (ch != '\n' and ch != '\r' and ch != ' ' and ch != '\t') try clean.append(a, ch);
        }
        const dec_len = try std.base64.standard.Decoder.calcSizeForSlice(clean.items);
        const out = try a.alloc(u8, dec_len);
        errdefer a.free(out);
        try std.base64.standard.Decoder.decode(out, clean.items);
        return out;
    }
    const path = if (std.mem.startsWith(u8, source, "file://")) source["file://".len..] else source;
    const mapped = try st.mapFile(a, path);
    defer st.unmapFile(mapped);
    return try a.dupe(u8, mapped);
}

/// Query original (width, height) of encoded PNG/JPEG/WebP/HEIC bytes without full resize.
pub fn probeImageSize(bytes: []const u8) ![2]u32 {
    const cf_data = CFDataCreate(null, bytes.ptr, @intCast(bytes.len)) orelse return error.ImageDecodeFailed;
    defer CFRelease(cf_data);
    const src = CGImageSourceCreateWithData(cf_data, null) orelse return error.ImageDecodeFailed;
    defer CFRelease(src);
    const img = CGImageSourceCreateImageAtIndex(src, 0, null) orelse return error.ImageDecodeFailed;
    defer CGImageRelease(img);
    const w: u32 = @intCast(CGImageGetWidth(img));
    const h: u32 = @intCast(CGImageGetHeight(img));
    if (w == 0 or h == 0) return error.ImageDecodeFailed;
    return .{ w, h };
}

/// Decode encoded image bytes into an RGB u8 buffer resized to (target_w, target_h) (0 = keep original).
pub fn decodeAndResize(a: std.mem.Allocator, bytes: []const u8, target_w: u32, target_h: u32) !RgbImage {
    const cf_data = CFDataCreate(null, bytes.ptr, @intCast(bytes.len)) orelse return error.ImageDecodeFailed;
    defer CFRelease(cf_data);
    const src = CGImageSourceCreateWithData(cf_data, null) orelse return error.ImageDecodeFailed;
    defer CFRelease(src);
    const img = CGImageSourceCreateImageAtIndex(src, 0, null) orelse return error.ImageDecodeFailed;
    defer CGImageRelease(img);

    const ow: u32 = @intCast(CGImageGetWidth(img));
    const oh: u32 = @intCast(CGImageGetHeight(img));
    const w: u32 = if (target_w > 0) target_w else ow;
    const h: u32 = if (target_h > 0) target_h else oh;
    if (w == 0 or h == 0) return error.ImageDecodeFailed;

    const rgba = try a.alloc(u8, @as(usize, w) * h * 4);
    defer a.free(rgba);
    @memset(rgba, 255);

    const cs = CGColorSpaceCreateDeviceRGB() orelse return error.ImageDecodeFailed;
    defer CGColorSpaceRelease(cs);

    const ctx = CGBitmapContextCreate(
        rgba.ptr,
        w,
        h,
        8,
        @as(usize, w) * 4,
        cs,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big,
    ) orelse return error.ImageDecodeFailed;
    defer CGContextRelease(ctx);

    CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
    CGContextDrawImage(ctx, .{ .x = 0, .y = 0, .w = @floatFromInt(w), .h = @floatFromInt(h) }, img);

    // Un-premultiply alpha (onto white if transparent) and strip to packed RGB u8
    const rgb = try a.alloc(u8, @as(usize, w) * h * 3);
    for (0..@as(usize, w) * h) |i| {
        const a_val = rgba[i * 4 + 3];
        if (a_val == 255) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        } else if (a_val == 0) {
            rgb[i * 3 + 0] = 255;
            rgb[i * 3 + 1] = 255;
            rgb[i * 3 + 2] = 255;
        } else {
            const inv_a = 255.0 / @as(f32, @floatFromInt(a_val));
            rgb[i * 3 + 0] = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(rgba[i * 4 + 0])) * inv_a));
            rgb[i * 3 + 1] = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(rgba[i * 4 + 1])) * inv_a));
            rgb[i * 3 + 2] = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(rgba[i * 4 + 2])) * inv_a));
        }
    }
    return .{ .width = w, .height = h, .pixels = rgb };
}

/// Qwen2VL / Qwen3.5-VL aspect-ratio-preserving smart resize to multiples of `factor` (= patch_size * merge_size).
pub fn smartResizeQwen(orig_h: u32, orig_w: u32, factor: u32, min_pixels: u64, max_pixels: u64) [2]u32 {
    const f_f: f64 = @floatFromInt(factor);
    var h_bar: f64 = @round(@as(f64, @floatFromInt(orig_h)) / f_f) * f_f;
    var w_bar: f64 = @round(@as(f64, @floatFromInt(orig_w)) / f_f) * f_f;
    if (h_bar < f_f) h_bar = f_f;
    if (w_bar < f_f) w_bar = f_f;

    const cur_px = h_bar * w_bar;
    const max_p: f64 = @floatFromInt(max_pixels);
    const min_p: f64 = @floatFromInt(min_pixels);
    if (cur_px > max_p) {
        const beta = @sqrt(@as(f64, @floatFromInt(@as(u64, orig_h) * orig_w)) / max_p);
        h_bar = @max(f_f, @floor(@as(f64, @floatFromInt(orig_h)) / beta / f_f) * f_f);
        w_bar = @max(f_f, @floor(@as(f64, @floatFromInt(orig_w)) / beta / f_f) * f_f);
    } else if (cur_px < min_p) {
        const beta = @sqrt(min_p / @as(f64, @floatFromInt(@as(u64, orig_h) * orig_w)));
        h_bar = @ceil(@as(f64, @floatFromInt(orig_h)) * beta / f_f) * f_f;
        w_bar = @ceil(@as(f64, @floatFromInt(orig_w)) * beta / f_f) * f_f;
    }
    return .{ @intFromFloat(h_bar), @intFromFloat(w_bar) };
}

/// Pack RGB u8 image [H, W, 3] into Qwen3.5-VL BF16 patches [N, 3 * P * P] in 2x2 merge-block order
/// (bh, bw, ih, iw), channel-first [C, Ph, Pw], normalized with mean=0.5, std=0.5.
pub fn packQwenPatches(a: std.mem.Allocator, rgb: RgbImage, patch_size: u32, merge_size: u32) ![]u16 {
    const gh = rgb.height / patch_size;
    const gw = rgb.width / patch_size;
    const mh = gh / merge_size;
    const mw = gw / merge_size;
    const p2: usize = @as(usize, patch_size) * patch_size;
    const k_dim: usize = 3 * p2;
    const n_patches: usize = @as(usize, gh) * gw;

    const out = try a.alloc(u16, n_patches * k_dim);
    var p_idx: usize = 0;
    for (0..mh) |bh| {
        for (0..mw) |bw| {
            for (0..merge_size) |ih| {
                for (0..merge_size) |iw| {
                    const py = bh * merge_size + ih;
                    const px = bw * merge_size + iw;
                    const dst = out[p_idx * k_dim ..][0..k_dim];
                    for (0..patch_size) |dy| {
                        const y = py * patch_size + dy;
                        for (0..patch_size) |dx| {
                            const x = px * patch_size + dx;
                            const src_off = (y * rgb.width + x) * 3;
                            const sp = dy * patch_size + dx;
                            for (0..3) |c| {
                                const norm = (@as(f32, @floatFromInt(rgb.pixels[src_off + c])) / 255.0 - 0.5) / 0.5;
                                dst[c * p2 + sp] = f32ToBf16(norm);
                            }
                        }
                    }
                    p_idx += 1;
                }
            }
        }
    }
    return out;
}

/// Pack RGB u8 image [S, S, 3] into Gemma 3-VL SigLIP BF16 patches [grid*grid, 3 * P * P] in raster order,
/// channel-first [C, Ph, Pw], normalized with mean=0.5, std=0.5.
pub fn packSiglipPatches(a: std.mem.Allocator, rgb: RgbImage, patch_size: u32) ![]u16 {
    const gh = rgb.height / patch_size;
    const gw = rgb.width / patch_size;
    const p2: usize = @as(usize, patch_size) * patch_size;
    const k_dim: usize = 3 * p2;
    const out = try a.alloc(u16, @as(usize, gh) * gw * k_dim);
    var p_idx: usize = 0;
    for (0..gh) |py| {
        for (0..gw) |px| {
            const dst = out[p_idx * k_dim ..][0..k_dim];
            for (0..patch_size) |dy| {
                const y = py * patch_size + dy;
                for (0..patch_size) |dx| {
                    const x = px * patch_size + dx;
                    const src_off = (y * rgb.width + x) * 3;
                    const sp = dy * patch_size + dx;
                    for (0..3) |c| {
                        const norm = (@as(f32, @floatFromInt(rgb.pixels[src_off + c])) / 255.0 - 0.5) / 0.5;
                        dst[c * p2 + sp] = f32ToBf16(norm);
                    }
                }
            }
            p_idx += 1;
        }
    }
    return out;
}
