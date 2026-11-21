//! Precise production chain sources and test-only faults in their actual input loads and reduction stages.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Fault = enum { clip, input_write, tail, half_sum, half_norm };
pub const Kind = enum { swiglu, out_in, rms, out_rope, rope_in, gate_up, down, qkv };

pub fn entry(kind: Kind) [:0]const u8 {
    return switch (kind) {
        .swiglu => "tf_exl3_swiglu_prerotate",
        .out_in => "tf_exl3_rotate_out_in",
        .rms => "tf_exl3_rms_rotate_in",
        .out_rope => "tf_exl3_rotate_out_rope",
        .rope_in => "tf_exl3_rope_rotate_in",
        .gate_up => "tf_exl3_gateup_swiglu_prerotate",
        .down => "tf_exl3_down_combine",
        .qkv => "tf_exl3_qkv_epilogue",
    };
}

fn replace(a: std.mem.Allocator, text: *[]const u8, old: []const u8, new: []const u8) !void {
    if (std.mem.count(u8, text.*, old) != 1) return error.ChainFaultAnchorChanged;
    const next = try std.mem.replaceOwned(u8, a, text.*, old, new);
    a.free(text.*);
    text.* = next;
}

pub fn source(a: std.mem.Allocator, fault: ?Fault) ![]const u8 {
    var text: []const u8 = try a.dupe(u8, sources.core_exl3_projection);
    errdefer a.free(text);
    switch (fault orelse return text) {
        .clip => try replace(a, &text, "inline void tf_exl3_hadamard(thread float (&v)[4], uint lane) {", "inline void tf_exl3_hadamard(thread float (&v)[4], uint lane) {\n  for (uint j=0;j<4;++j) if (!isfinite(v[j])) v[j]=0;"),
        .input_write => try replace(a, &text, "inline float tf_exl3_scale(const device ushort* p, uint i, uint signs) {", "inline float tf_exl3_scale(const device ushort* p, uint i, uint signs) {\n  const_cast<device ushort*>(p)[0] = 0;"),
        .tail => try replace(a, &text, "inline void tf_exl3_hadamard(thread float (&v)[4], uint lane) {", "inline void tf_exl3_hadamard(thread float (&v)[4], uint lane) {\n  v[3] = 0;"),
        .half_sum => {
            try replace(a, &text, "v[j] = (lane & d) ? peer - v[j] : v[j] + peer;", "v[j] = float(half((lane & d) ? peer - v[j] : v[j] + peer));");
            try replace(a, &text, "v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;", "v[0] = float(half(a+b)); v[1] = float(half(a-b)); v[2] = float(half(c+d)); v[3] = float(half(c-d));");
            try replace(a, &text, "v[0] = e + g; v[1] = f + h; v[2] = e - g; v[3] = f - h;", "v[0] = float(half(e+g)); v[1] = float(half(f+h)); v[2] = float(half(e-g)); v[3] = float(half(f-h));");
            try replace(a, &text, "for (uint rank = 0; rank < d.picks; rank++) s += values[rank][lane + 32u * j];", "for (uint rank = 0; rank < d.picks; rank++) s = float(half(s + values[rank][lane + 32u * j]));");
        },
        .half_norm => {
            try replace(a, &text, "for (uint i = t; i < d.width; i += 256) { float a = tf_exl3_chain_bf(x[ulong(row) * d.x_stride + i]); s += a * a; }", "for (uint i = t; i < d.width; i += 256) { float a = tf_exl3_chain_bf(x[ulong(row) * d.x_stride + i]); s = float(half(s + a * a)); }");
            try replace(a, &text, "for (uint i = t; i < width; i += 256) { float a = tf_exl3_chain_bf(row[i]); s += a * a; }", "for (uint i = t; i < width; i += 256) { float a = tf_exl3_chain_bf(row[i]); s = float(half(s + a * a)); }");
            try replace(a, &text, "s = fma(v1, v1, s);", "s = float(half(fma(v1, v1, s)));");
        },
    }
    return text;
}

pub fn library(device: mtl.Device, text: []const u8) !mtl.Library {
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    return mtl.Library.fromSource(device, text, options);
}
