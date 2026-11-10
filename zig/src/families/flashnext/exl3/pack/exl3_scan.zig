//! EXL3 safetensors header scan.

const std = @import("std");

/// Same tags, in the same order, as `exl3_format.Codebook`.
pub const format = struct {
    pub const Codebook = enum { inst3, mcg, mul1 };
};

pub const Scales = enum { suh, su, svh, sv };

pub const Group = struct {
    halves: u16,
    codebook: format.Codebook,
    k: usize,
    n: usize,
    in_scales: Scales,
    out_scales: Scales,
    bias: bool,
};

pub const Parts = struct {
    trellis_words: usize,
    k_tiles: usize,
    n_tiles: usize,
    suh_len: ?usize = null,
    su_len: ?usize = null,
    svh_len: ?usize = null,
    sv_len: ?usize = null,
    bias: bool = false,
    mcg: bool = false,
    mul1: bool = false,
};

pub const Kind = enum { exl3_part, ngram_row, plain };

const suffixes = [_][]const u8{ "trellis", "suh", "su", "svh", "sv", "bias", "mcg", "mul1" };

/// One EXL3 linear group from its header parts. Half-bit widths need the mul1 marker.
pub fn parseGroup(parts: Parts) error{NotExl3}!Group {
    if (parts.trellis_words == 0 or parts.trellis_words % 8 != 0) return error.NotExl3;
    const halves_usize = parts.trellis_words / 8;
    if (halves_usize < 2 or halves_usize > 16) return error.NotExl3;
    const halves: u16 = @intCast(halves_usize);
    if (parts.mcg and parts.mul1) return error.NotExl3;
    const codebook: format.Codebook = if (parts.mul1) .mul1 else if (parts.mcg) .mcg else .inst3;
    if (halves % 2 == 1 and codebook != .mul1) return error.NotExl3;

    const k = parts.k_tiles * 16;
    const n = parts.n_tiles * 16;
    if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0) return error.NotExl3;

    const in_scales: Scales = if (parts.suh_len != null and parts.su_len == null)
        .suh
    else if (parts.su_len != null and parts.suh_len == null)
        .su
    else
        return error.NotExl3;
    if (in_scales == .suh and parts.suh_len.? != k) return error.NotExl3;
    if (in_scales == .su and parts.su_len.? != k / 16) return error.NotExl3;

    const out_scales: Scales = if (parts.svh_len != null and parts.sv_len == null)
        .svh
    else if (parts.sv_len != null and parts.svh_len == null)
        .sv
    else
        return error.NotExl3;
    if (out_scales == .svh and parts.svh_len.? != n) return error.NotExl3;
    if (out_scales == .sv and parts.sv_len.? != n / 16) return error.NotExl3;

    return .{
        .halves = halves,
        .codebook = codebook,
        .k = k,
        .n = n,
        .in_scales = in_scales,
        .out_scales = out_scales,
        .bias = parts.bias,
    };
}

/// An n-gram shard is a rank-2 `ngram_embedding` `.trellis`, not an EXL3 tile tensor.
pub fn classifyName(name: []const u8, rank: usize) Kind {
    if (rank == 2 and std.mem.indexOf(u8, name, "ngram_embedding") != null and std.mem.endsWith(u8, name, ".trellis")) {
        return .ngram_row;
    }
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return .plain;
    const suffix = name[dot + 1 ..];
    for (suffixes) |part| if (std.mem.eql(u8, suffix, part)) return .exl3_part;
    return .plain;
}

/// The MTP mixer's extra file, the one turboderp keeps outside the index.
pub fn isMtpMixerPatch(filename: []const u8) bool {
    return std.mem.eql(u8, filename, "mtp_hyper_connection_mixer_patch.safetensors");
}

fn codebookName(name: []const u8) error{QuantRefuse}!void {
    if (std.mem.eql(u8, name, "mul1") or std.mem.eql(u8, name, "mcg") or std.mem.eql(u8, name, "3inst") or std.mem.eql(u8, name, "inst3")) return;
    return error.QuantRefuse;
}

fn integerBits(bits: f32) error{QuantRefuse}!void {
    if (!std.math.isFinite(bits) or bits < 1 or bits > 8) return error.QuantRefuse;
    if (bits != @round(bits)) return error.QuantRefuse;
}

/// Admit a pack from config fields alone. Average bits may be fractional. Head and MTP bits may not.
pub fn admit(codebook: []const u8, average_bits: f32, head_bits: ?f32, mtp_bits: ?f32) error{QuantRefuse}!void {
    try codebookName(codebook);
    if (!std.math.isFinite(average_bits) or average_bits < 1 or average_bits > 8) return error.QuantRefuse;
    if (head_bits) |bits| try integerBits(bits);
    if (mtp_bits) |bits| try integerBits(bits);
}

/// 1 when centred norms are stored as gamma-1, 0 when stored as gamma.
pub fn centredOffset(means: []const f32) error{NormAmbiguous}!f32 {
    if (means.len == 0) return 1;
    if (means.len > 64) return error.NormAmbiguous;
    var copy: [64]f32 = undefined;
    @memcpy(copy[0..means.len], means);
    var i: usize = 1;
    while (i < means.len) : (i += 1) {
        const v = copy[i];
        var j = i;
        while (j > 0 and copy[j - 1] > v) : (j -= 1) copy[j] = copy[j - 1];
        copy[j] = v;
    }
    const median = copy[means.len / 2];
    var above: usize = 0;
    for (means) |m| if (m > 0.5) {
        above += 1;
    };
    const frac = @as(f32, @floatFromInt(above)) / @as(f32, @floatFromInt(means.len));
    const around_zero = frac <= 0.1 and median >= -0.5 and median <= 0.25;
    const around_one = frac >= 0.9 and median >= 0.75 and median <= 1.5;
    if (around_zero == around_one) return error.NormAmbiguous;
    return if (around_zero) 1 else 0;
}

test "a 3-bit mul1 gate group parses and a half-bit group needs mul1" {
    const gate = try parseGroup(.{
        .trellis_words = 48,
        .k_tiles = 160,
        .n_tiles = 40,
        .suh_len = 2560,
        .svh_len = 640,
        .mul1 = true,
    });
    try std.testing.expectEqual(@as(u16, 6), gate.halves);
    try std.testing.expectEqual(format.Codebook.mul1, gate.codebook);
    try std.testing.expectEqual(@as(usize, 2560), gate.k);
    try std.testing.expectEqual(@as(usize, 640), gate.n);
    try std.testing.expectEqual(Scales.suh, gate.in_scales);
    try std.testing.expectEqual(Scales.svh, gate.out_scales);

    const half = Parts{
        .trellis_words = 40,
        .k_tiles = 160,
        .n_tiles = 40,
        .suh_len = 2560,
        .svh_len = 640,
        .mul1 = true,
    };
    try std.testing.expectEqual(@as(u16, 5), (try parseGroup(half)).halves);
    var no_marker = half;
    no_marker.mul1 = false;
    try std.testing.expectError(error.NotExl3, parseGroup(no_marker));
    var both = half;
    both.mcg = true;
    try std.testing.expectError(error.NotExl3, parseGroup(both));
}

test "K must be a multiple of 128" {
    try std.testing.expectError(error.NotExl3, parseGroup(.{
        .trellis_words = 48,
        .k_tiles = 20,
        .n_tiles = 40,
        .suh_len = 320,
        .svh_len = 640,
        .mul1 = true,
    }));
}

test "names sort into n-gram rows, EXL3 parts and plain tensors" {
    try std.testing.expectEqual(Kind.exl3_part, classifyName("model.language_model.layers.0.mlp.experts.0.gate_proj.trellis", 3));
    try std.testing.expectEqual(Kind.ngram_row, classifyName("ple.ngram_embedding.shard_3.trellis", 2));
    try std.testing.expectEqual(Kind.exl3_part, classifyName("ple.ngram_embedding.shard_3.trellis", 3));
    try std.testing.expectEqual(Kind.plain, classifyName("model.embed_tokens.weight", 2));
    try std.testing.expect(isMtpMixerPatch("mtp_hyper_connection_mixer_patch.safetensors"));
    try std.testing.expect(!isMtpMixerPatch("model.safetensors"));
}

test "the 3.05 config is admitted and a fractional head width is not" {
    try admit("mul1", 3.05, 5, 3);
    try admit("3inst", 4, null, null);
    try std.testing.expectError(error.QuantRefuse, admit("mul1", 3.05, 3.05, null));
    try std.testing.expectError(error.QuantRefuse, admit("exl2", 3, null, null));
}

test "centred norms are gamma-1 when the means sit near zero" {
    try std.testing.expectEqual(@as(f32, 1), try centredOffset(&.{ -0.02, 0.01, 0.0, 0.04 }));
    try std.testing.expectEqual(@as(f32, 0), try centredOffset(&.{ 1.0, 0.98, 1.02, 1.01 }));
    try std.testing.expectError(error.NormAmbiguous, centredOffset(&.{ 0.4, 0.9 }));
    try std.testing.expectEqual(@as(f32, 1), try centredOffset(&.{}));
}
