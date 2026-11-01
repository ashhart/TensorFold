//! Planted controls for LFM3's base operators: the production text with one fault, which the shapes gate must report.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const base = @import("metal_lfm3.zig");
const math = @import("lfm3_math.zig");
const g = contract.gates;

pub const Kind = math.Kind;

pub const Mutation = enum {
    embed_group_index,
    embed_straddle_shift,
    embed_bias_dropped,
    norm_inverse_dropped,
    post_add_sign,
    add_operand_index,
    conv_tap_reversed,
    qk_rotation_sign,
    qk_frequency_scale,
    gate_logit_index,
    swiglu_operand,
    attention_scale_dropped,
    attention_window_edge,

    fn what(m: Mutation) []const u8 {
        return switch (m) {
            .embed_group_index => "a scale and bias group twice as wide",
            .embed_straddle_shift => "the word-crossing code shifted one bit short",
            .embed_bias_dropped => "the bias dropped",
            .norm_inverse_dropped => "the inverse root not applied",
            .post_add_sign => "the normalized branch subtracted from the residual",
            .add_operand_index => "the addend read at a neighbouring element",
            .conv_tap_reversed => "the taps' weights read in reverse",
            .qk_rotation_sign => "the first rotated half's sine sign flipped",
            .qk_frequency_scale => "the rotary frequency progression over 128, not 64",
            .gate_logit_index => "the gate logit read at the row, not the head",
            .swiglu_operand => "the up operand replaced by the gate",
            .attention_scale_dropped => "the key scores not scaled",
            .attention_window_edge => "the window's oldest key skipped",
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .embed_group_index => .{ .from = "+ at.x / group;", .to = "+ at.x / (2 * group);" },
            .embed_straddle_shift => .{ .from = "words[word + 1] << (32 - shift)", .to = "words[word + 1] << (31 - shift)" },
            .embed_bias_dropped => .{ .from = "bfloat(product + float(biases[table]))", .to = "bfloat(product)" },
            .norm_inverse_dropped => .{ .from = "out[base + j] = bfloat(float(bfloat(v[j] * inv)) * float(gain[8 * t + j]));", .to = "out[base + j] = bfloat(float(bfloat(v[j])) * float(gain[8 * t + j]));" },
            .post_add_sign => .{ .from = "h[base + j] = bfloat(float(h[base + j]) + float(normalized));", .to = "h[base + j] = bfloat(float(h[base + j]) - float(normalized));" },
            .add_operand_index => .{ .from = "h[i] = bfloat(float(h[i]) + float(x[i]));", .to = "h[i] = bfloat(float(h[i]) + float(x[i ^ 1]));" },
            .conv_tap_reversed => .{ .from = "float(weight[ch * 7 + tap])", .to = "float(weight[ch * 7 + 6 - tap])" },
            .qk_rotation_sign => .{ .from = "out[pair] = bfloat(v[pair] * cosine - v[pair + 2] * sine);", .to = "out[pair] = bfloat(v[pair] * cosine + v[pair + 2] * sine);" },
            .qk_frequency_scale => .{ .from = "-float(frequency) / 64.0f", .to = "-float(frequency) / 128.0f" },
            .gate_logit_index => .{ .from = "float(logits[i / L3H])", .to = "float(logits[i / L3D])" },
            .swiglu_operand => .{ .from = "out[i] = bfloat(float(silu) * float(up[i]));", .to = "out[i] = bfloat(float(silu) * float(gate[i]));" },
            .attention_scale_dropped => .{ .from = "lfm3_attention_dot(q, keys + base) * p.scale;", .to = "lfm3_attention_dot(q, keys + base);" },
            .attention_window_edge => .{ .from = "max(chunk * L3PER + sg * L3PART, cutoff);", .to = "max(chunk * L3PER + sg * L3PART, cutoff + (p.window == 0 ? 0u : 1u));" },
        };
    }
};

/// The one fault planted in each base probe's operator; the two qk and the two attention probes carry different ones.
pub fn mutation(kind: Kind) Mutation {
    return switch (kind) {
        .embed4g32, .embed4g64 => .embed_group_index,
        .embed6g32, .embed6g64 => .embed_straddle_shift,
        .embed8g32, .embed8g64 => .embed_bias_dropped,
        .norm => .norm_inverse_dropped,
        .post_add => .post_add_sign,
        .add => .add_operand_index,
        .conv => .conv_tap_reversed,
        .qk_full => .qk_rotation_sign,
        .qk_sliding => .qk_frequency_scale,
        .gate => .gate_logit_index,
        .swiglu => .swiglu_operand,
        .attention_full => .attention_scale_dropped,
        .attention_sliding => .attention_window_edge,
    };
}

pub fn name(kind: Kind) []const u8 {
    return switch (kind) {
        inline else => |k| "control: " ++ comptime base.name(k) ++ ", " ++ mutation(k).what(),
    };
}

/// The production text with the kind's one edit; an edit that is absent, or ambiguous, would prove nothing.
pub fn source(a: std.mem.Allocator, kind: Kind) ![]u8 {
    const e = mutation(kind).edit();
    const text = try base.Probe.source(a);
    defer a.free(text);
    if (std.mem.count(u8, text, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind) !base.Probe {
    const text = try source(a, kind);
    defer a.free(text);
    return base.Probe.initSource(a, device, queue, kind, text);
}

pub fn kernel(p: *base.Probe) g.Kernel {
    var k = p.kernel();
    k.spec.name = name(p.kind);
    k.control = .shapes;
    return k;
}
