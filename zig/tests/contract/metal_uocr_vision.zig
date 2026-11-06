//! Unlimited-OCR-2's image-encoder kernels (uv_*) under the contract: the shared source, controls and helpers.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const hub = @import("metal_uocr.zig");
const glue = @import("metal_uocr_vision_glue.zig");
const maps = @import("metal_uocr_vision_maps.zig");
const attn = @import("metal_uocr_vision_attn.zig");

const c = contract.spec;
const g = contract.gates;

/// A planted control: vision.metal with one fault, which one gate must report.
pub const Mutation = enum {
    none,
    add_subtracts,
    add_writes_input,
    bias_next_column,
    norm_mean_over_threads,
    gelu_wrong_scale,
    quick_gelu_unrounded,
    concat_left_twice,
    embed_own_row,
    cols_pad_down,
    window_transposed,
    unwindow_no_add,
    rel_mirrored,
    rel_half_sums,
    attention_row_by_column,
    attention_nan_skipped,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .add_subtracts => "control: uocr vision uv_add, the rows are subtracted",
            .add_writes_input => "control: uocr vision uv_add, each sum is also written over its first input",
            .bias_next_column => "control: uocr vision uv_bias, each column takes the next column's bias",
            .norm_mean_over_threads => "control: uocr vision uv_layer_norm, the mean divides by the threads, not the width",
            .gelu_wrong_scale => "control: uocr vision uv_gelu, erf's argument is x times 0.75, not 1 / sqrt 2",
            .quick_gelu_unrounded => "control: uocr vision uv_quick_gelu, the sigmoid is not rounded to bf16",
            .concat_left_twice => "control: uocr vision uv_concat, the right half repeats the left rows",
            .embed_own_row => "control: uocr vision uv_clip_embed, row r adds SAM's row r - 2, not r - 1",
            .cols_pad_down => "control: uocr vision uv_im2col, the rows are offset by the padding the wrong way",
            .window_transposed => "control: uocr vision uv_window, each window's rows and columns are swapped",
            .unwindow_no_add => "control: uocr vision uv_unwindow_add, the window rows replace the map's",
            .rel_mirrored => "control: uocr vision uv_rel_pos, the offsets are mirrored (key minus query)",
            .rel_half_sums => "control: uocr vision uv_rel_pos, the dot summed in half",
            .attention_row_by_column => "control: uocr vision uv_attention, the row term is read at the key's column",
            .attention_nan_skipped => "control: uocr vision uv_attention, a NaN score weighs 0",
        };
    }

    /// The gate that must go red.
    pub fn gate(m: Mutation) ?g.Gate {
        return switch (m) {
            .none => null,
            .add_writes_input => .repeat,
            .rel_half_sums => .sums,
            .attention_nan_skipped => .nonfinite,
            else => .shapes,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .add_subtracts => .{ .from = "y[i] = bfloat(float(a[i]) + float(b[i]));", .to = "y[i] = bfloat(float(a[i]) - float(b[i]));" },
            .add_writes_input => .{ .from = "y[i] = bfloat(float(a[i]) + float(b[i]));", .to = "y[i] = bfloat(float(a[i]) + float(b[i])); const_cast<device bfloat*>(a)[i] = y[i];" },
            .bias_next_column => .{ .from = "y[i] = bfloat(c[i] + float(b[at.x]));", .to = "y[i] = bfloat(c[i] + float(b[(at.x + 1) % width]));" },
            .norm_mean_over_threads => .{ .from = "uv_sum(s, part, n, lane, sg) / float(p.width);", .to = "uv_sum(s, part, n, lane, sg) / float(n);" },
            .gelu_wrong_scale => .{ .from = "uv_erf(v * 0.70710678118654752f)", .to = "uv_erf(v * 0.75f)" },
            .quick_gelu_unrounded => .{ .from = "const float s = uv_bf(1.0f / (1.0f + metal::precise::exp(-z)));", .to = "const float s = 1.0f / (1.0f + metal::precise::exp(-z));" },
            .concat_left_twice => .{ .from = ": b[ulong(at.y) * p.right + at.x - p.left];", .to = ": a[ulong(at.y) * p.left + (at.x - p.left) % p.left];" },
            .embed_own_row => .{ .from = "float(sam[ulong(at.y - 1) * width + at.x]);", .to = "float(sam[ulong(at.y > 1 ? at.y - 2 : 0) * width + at.x]);" },
            .cols_pad_down => .{ .from = "const int iy = int((at.y / p.out_side) * p.stride + t / p.span) - int(p.pad);", .to = "const int iy = int((at.y / p.out_side) * p.stride + t / p.span) + int(p.pad);" },
            .window_transposed => .{ .from = "(w / across) * p.window + i / p.window, c = (w % across) * p.window + i % p.window;", .to = "(w / across) * p.window + i % p.window, c = (w % across) * p.window + i / p.window;" },
            .unwindow_no_add => .{ .from = "x[to] = bfloat(float(x[to]) + float(win[", .to = "x[to] = bfloat(float(win[" },
            .rel_mirrored => .{ .from = "const uint from = (across ? i % p.side : i / p.side) + p.side - 1 - j;", .to = "const uint from = j + p.side - 1 - (across ? i % p.side : i / p.side);" },
            .rel_half_sums => .{ .from = "s = fma(float(q[c]), float(r[c]), s);", .to = "s = float(half(fma(float(q[c]), float(r[c]), s)));" },
            .attention_row_by_column => .{ .from = "s += float(bfloat(float(rel_h[at + key / p.side])", .to = "s += float(bfloat(float(rel_h[at + key % p.side])" },
            .attention_nan_skipped => .{ .from = "const float w = key < p.len ? metal::precise::exp(", .to = "const float w = key < p.len && !isnan(stage[at]) ? metal::precise::exp(" },
        };
    }
};

/// vision.metal with the control's one edit (a copy either way); an edit that finds nothing errors.
pub fn source(a: std.mem.Allocator, m: Mutation) ![]u8 {
    const text = hub.source(.vision);
    if (m == .none) return a.dupe(u8, text);
    const e = m.edit();
    if (std.mem.count(u8, text, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

/// A vision kernel compiled as families/uocr/kernels.zig compiles it (MLX's options), with the control's edit.
pub fn pipeline(a: std.mem.Allocator, device: mtl.Device, name: []const u8, m: Mutation) !mtl.Pipeline {
    if (m == .none) return hub.pipeline(device, .vision, name);
    const src = try source(a, m);
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

/// A kernel's spec renamed and marked for a control built with `m`.
pub fn mark(k: g.Kernel, m: Mutation) g.Kernel {
    var out = k;
    if (m != .none) {
        out.spec.name = m.name();
        out.control = m.gate();
    }
    return out;
}

/// A value's spread through fp32 steps and bf16 rounding points.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// Each end moved by `e` either way.
    pub fn widen(r: Range, e: f64) Range {
        return .{ .lo = r.lo - e, .hi = r.hi + e };
    }
    /// Each end rounded to bf16, as the kernel rounds a value it stores.
    pub fn bf16(r: Range) Range {
        return .{ .lo = hub.bf(r.lo), .hi = hub.bf(r.hi) };
    }
    /// Times an exact value of either sign.
    pub fn times(r: Range, v: f64) Range {
        return .{ .lo = @min(r.lo * v, r.hi * v), .hi = @max(r.lo * v, r.hi * v) };
    }
};

/// The allowance a range gives about `want`: the farther end.
pub fn spread(want: f64, r: Range) f64 {
    return @max(@abs(want - r.lo), @abs(r.hi - want));
}

/// bf16 bits of random values in [lo, hi).
pub fn fill(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    glue: glue.Suite = undefined,
    maps: maps.Suite = undefined,
    attn: attn.Suite = undefined,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        inline for (.{ "glue", "maps", "attn" }) |f| {
            @field(s, f) = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
            try @field(s, f).collect(only);
        }
    }
    pub fn deinit(s: *Suite) void {
        inline for (.{ "glue", "maps", "attn" }) |f| @field(s, f).deinit();
    }
};

/// Gate 5's texts for this group's registrations (`uocr vision ...`): vision.metal as is, on every chip.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?hub.Texts {
    if (!std.mem.startsWith(u8, name, "uocr vision ")) return null;
    return .{ .here = hub.source(.vision), .pre_m5 = hub.source(.vision) };
}

/// Whether `only` (tf-kernel-contract's --only) selects a kernel named `name`.
pub fn wanted(only: ?[]const u8, name: []const u8) bool {
    const o = only orelse return true;
    return std.mem.indexOf(u8, name, o) != null;
}
