//! One MLX affine 4-bit projection on the host: its quantizer at the engine's groups, comparison and f32 decoding.
const std = @import("std");
const host4 = @import("affine4_host.zig");
const st = @import("safetensors.zig");

/// One projection as stored: packed words, then bf16 scales and biases a group of `group` values.
pub const Quant = struct { words: []u32, scales: []u16, biases: []u16 };

pub const View = struct { words: []const u32, scales: []const u16, biases: []const u16 };

pub fn viewOf(q: Quant) View {
    return .{ .words = q.words, .scales = q.scales, .biases = q.biases };
}

/// Room for `n * k` values in groups of `group`; `freeQuant` returns it.
pub fn allocQuant(gpa: std.mem.Allocator, n: usize, k: usize, group: usize) !Quant {
    const count = try std.math.mul(usize, n, k);
    const words = try gpa.alloc(u32, count / 8);
    errdefer gpa.free(words);
    const scales = try gpa.alloc(u16, count / group);
    errdefer gpa.free(scales);
    return .{ .words = words, .scales = scales, .biases = try gpa.alloc(u16, count / group) };
}

pub fn freeQuant(gpa: std.mem.Allocator, q: Quant) void {
    gpa.free(q.words);
    gpa.free(q.scales);
    gpa.free(q.biases);
}

/// MLX's quantizer on rows of k, groups of 32, 64 or 128 (others refused before a write); the first bad group or null.
pub fn quantize(group: usize, values: []const f32, out: Quant) error{BadGroup}!?usize {
    return switch (group) {
        32 => host4.quantizeMlx(32, values, out.words, out.scales, out.biases),
        64 => host4.quantizeMlx(64, values, out.words, out.scales, out.biases),
        128 => host4.quantizeMlx(128, values, out.words, out.scales, out.biases),
        else => error.BadGroup,
    };
}

/// How alike two quantizations of one tensor are: the fraction of equal scales, biases and codes, and of equal groups.
pub const Agreement = struct {
    scales: f64,
    biases: f64,
    codes: f64,
    groups: f64,

    pub fn identical(a: Agreement) bool {
        return a.scales == 1 and a.biases == 1 and a.codes == 1 and a.groups == 1;
    }
};

/// The number of groups whose scale, bias or codes differ between `a` and `b`.
pub fn differingGroups(group: usize, a: View, b: View) usize {
    const gw = group / 8;
    var n: usize = 0;
    for (0..a.scales.len) |g| {
        const same = a.scales[g] == b.scales[g] and a.biases[g] == b.biases[g] and std.mem.eql(u32, a.words[g * gw ..][0..gw], b.words[g * gw ..][0..gw]);
        n += @intFromBool(!same);
    }
    return n;
}

pub fn agreement(group: usize, a: View, b: View) Agreement {
    const gw = group / 8;
    const groups = a.scales.len;
    var s: usize = 0;
    var bi: usize = 0;
    var c: usize = 0;
    var both: usize = 0;
    for (0..groups) |g| {
        const se = a.scales[g] == b.scales[g];
        const be = a.biases[g] == b.biases[g];
        var ce: usize = 0;
        for (0..group) |i| {
            const x = (a.words[g * gw + i / 8] >> @intCast(4 * (i % 8))) & 0xf;
            const y = (b.words[g * gw + i / 8] >> @intCast(4 * (i % 8))) & 0xf;
            if (x == y) ce += 1;
        }
        s += @intFromBool(se);
        bi += @intFromBool(be);
        c += ce;
        both += @intFromBool(se and be and ce == group);
    }
    const gf: f64 = @floatFromInt(groups);
    return .{ .scales = @as(f64, @floatFromInt(s)) / gf, .biases = @as(f64, @floatFromInt(bi)) / gf, .codes = @as(f64, @floatFromInt(c)) / (gf * @as(f64, @floatFromInt(group))), .groups = @as(f64, @floatFromInt(both)) / gf };
}

/// A tensor's values as f32: f32, bf16 and f16 are the dtypes checkpoints and PEFT adapters save.
pub fn toF32(dtype: st.DType, bytes: []const u8, out: []f32) error{ UnsupportedDType, BadLength }!void {
    switch (dtype) {
        .f32, .bf16, .f16 => {},
        else => return error.UnsupportedDType,
    }
    if (bytes.len != std.math.mul(usize, out.len, dtype.size()) catch return error.BadLength) return error.BadLength;
    switch (dtype) {
        .f32 => for (out, 0..) |*o, i| {
            o.* = @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
        },
        .bf16 => for (out, 0..) |*o, i| {
            o.* = host4.f32of(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
        },
        else => for (out, 0..) |*o, i| {
            o.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little))));
        },
    }
}

test {
    _ = @import("lora_quant_test.zig");
    _ = @import("lora_vectors_test.zig");
}
