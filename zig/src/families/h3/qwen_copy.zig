//! MiniMax-H3-Image's Qwen copy of a reference: its size, the processor's size for it, and its LANCZOS pixels.
const std = @import("std");
const picture = @import("picture");

/// The Qwen copy of a reference: its short edge exactly this, both sides multiples of `qwen_factor`.
pub const qwen_edge = 1024;
/// Patch 16 times merge 2: the processor's side unit, and one vision token per 2 x 2 patches.
pub const qwen_factor = 32;
pub const merge = 2;

pub const Size = struct { width: usize, height: usize };

/// x rounded to the nearest integer, ties to even, as Python's round() does on a float64.
fn roundEven(x: f64) f64 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// The Qwen copy's size: a 1,024 short edge kept, else both sides times 1024 / short edge, snapped to 32 (half even).
pub fn qwenBound(width: usize, height: usize) !Size {
    if (width < qwen_factor or height < qwen_factor) return error.H3ReferenceTooSmall;
    const short = @min(width, height);
    if (short == qwen_edge) return .{ .width = width, .height = height };
    const scale = @as(f64, qwen_edge) / @as(f64, @floatFromInt(short));
    const snap = struct {
        fn side(v: f64) usize {
            return @as(usize, @intFromFloat(@max(1.0, roundEven(v / qwen_factor)))) * qwen_factor;
        }
    };
    return .{ .width = snap.side(@as(f64, @floatFromInt(width)) * scale), .height = snap.side(@as(f64, @floatFromInt(height)) * scale) };
}

/// The processor's size for a Qwen copy: smart resize to multiples of 32 in its pixel bounds, a no-op when snapped.
pub fn qwenProcessed(s: Size, min_pixels: u64, max_pixels: u64) !Size {
    const r = picture.patches.smartResize(@intCast(s.height), @intCast(s.width), qwen_factor, min_pixels, max_pixels) catch return error.H3ReferenceAspect;
    return .{ .width = r.w, .height = r.h };
}

/// The Qwen copy's RGB pixels: the original when its short edge is 1,024, else Pillow's LANCZOS to `qwenBound`.
pub fn qwenCopy(a: std.mem.Allocator, rgb: picture.Rgb) !picture.Rgb {
    const s = try qwenBound(rgb.width, rgb.height);
    if (s.width == rgb.width and s.height == rgb.height) return .{ .width = s.width, .height = s.height, .pixels = try a.dupe(u8, rgb.pixels) };
    const view = picture.resample.View.of(rgb.pixels, rgb.width, rgb.height);
    return .{ .width = s.width, .height = s.height, .pixels = try picture.resample.resizeWith(a, .pillow, .lanczos, view, s.width, s.height) };
}
