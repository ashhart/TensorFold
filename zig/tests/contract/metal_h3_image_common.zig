//! MiniMax-H3-Image's kernel tests' shared parts: the three sources compiled as their launchers compile them, controls.
const std = @import("std");
const mtl = @import("metal");

pub const Part = enum { encoder, prompt, decoder };

/// The .metal source a part's launcher compiles at load.
pub fn source(part: Part) []const u8 {
    return switch (part) {
        .encoder => @import("h3_image_encoder_kernels").source,
        .prompt => @import("h3_prompt_kernels").source,
        .decoder => @import("h3_image_decoder_kernels").source,
    };
}

/// MLX's options, and for the edit encoder precise fp32 functions (image_encoder.zig's Kernels.init).
pub fn options(part: Part) mtl.CompileOptions {
    var o = mtl.CompileOptions.mlx();
    if (part == .encoder) o.math_functions = .precise;
    return o;
}

pub fn library(device: mtl.Device, part: Part, text: []const u8) !mtl.Library {
    return mtl.Library.fromSource(device, text, options(part));
}

/// The record and spec name of a part's kernel: "minimax image <part> <function>".
pub fn name(comptime part: Part, comptime function: []const u8) []const u8 {
    return "minimax image " ++ @tagName(part) ++ " " ++ function;
}

/// A part's source with `old` replaced by `new`, exactly once: a planted control's edit (refused when `old` moved).
pub fn edited(a: std.mem.Allocator, part: Part, old: []const u8, new: []const u8) ![]u8 {
    const text = source(part);
    const at = std.mem.indexOf(u8, text, old) orelse return error.ControlEditMissing;
    if (std.mem.indexOfPos(u8, text, at + 1, old) != null) return error.ControlEditAmbiguous;
    return std.mem.concat(a, u8, &.{ text[0..at], new, text[at + old.len ..] });
}

pub fn keep(only: ?[]const u8, label: []const u8) bool {
    return if (only) |needle| std.mem.indexOf(u8, label, needle) != null else true;
}
