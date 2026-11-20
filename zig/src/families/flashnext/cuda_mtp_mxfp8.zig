//! TF_FLASHNEXT_MTP_WEIGHTS=mxfp8 re-codes the drafter's BF16 linears to MXFP8 at load; a draft is only a proposal.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

/// E4M3's largest value: a group's scale brings its largest weight to this or under, so no code saturates.
const top_code: f32 = 448.0;
/// The scale's inputs (UE8M0 a 32-input group, 127 is 2^0).
const group = 32;

fn value(bf16: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bf16[i * 2 ..][0..2], .little)) << 16);
}

/// The least power of two that brings `top` to 448 or under, as the exponent the scale byte stores.
fn exponentFor(top: f32) i32 {
    if (!(top > 0) or !std.math.isFinite(top)) return 0;
    var e: i32 = @intFromFloat(@ceil(@log2(top / top_code)));
    // the float logarithm may be one off at a power of two: settle it on the exact scaled values
    while (std.math.ldexp(top, -e) > top_code) e += 1;
    while (e > -126 and std.math.ldexp(top, -(e - 1)) <= top_code) e -= 1;
    return std.math.clamp(e, -126, 127);
}

/// BF16 rows [n, k] as MXFP8: `codes` [n * k] E4M3 and `scales` [n * k / 32] UE8M0, each group at its own scale.
pub fn encode(bf16: []const u8, n: usize, k: usize, codes: []u8, scales: []u8) void {
    std.debug.assert(k % group == 0 and bf16.len == n * k * 2 and codes.len == n * k and scales.len == n * k / group);
    for (0..n * k / group) |g| {
        const base = g * group;
        var top: f32 = 0;
        for (0..group) |i| top = @max(top, @abs(value(bf16, base + i)));
        const e = exponentFor(top);
        scales[g] = @intCast(127 + e);
        for (0..group) |i| codes[base + i] = core.exlr8.codec.e4m3FromF32(std.math.ldexp(value(bf16, base + i), -e));
    }
}

/// A BF16 matrix [n, k] uploaded through the MXFP8 projection the mixers of an MXFP8 checkpoint load through.
pub fn project(gpa: std.mem.Allocator, driver: *const cuda.Driver, bf16: []const u8, n: usize, k: usize) !projection.Projection {
    if (k % 64 != 0 or bf16.len != n * k * 2) return error.UnexpectedTensor;
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, n * k / group);
    defer gpa.free(scales);
    encode(bf16, n, k, codes, scales);
    return projection.mxfp8(gpa, driver, .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = codes }, .{ .dtype = .u8, .rank = 2, .shape = .{ n, k / group, 1, 1, 1 }, .bytes = scales });
}

test {
    _ = @import("mtp_weights.zig");
}

test "a group's scale brings its largest weight under 448 and every code decodes within an eighth of its weight" {
    const k = 64;
    var bf16: [2 * k * 2]u8 = undefined;
    var want: [2 * k]f32 = undefined;
    for (&want, 0..) |*w, i| {
        // two rows, groups of very different size: 3e-4 up to 40, both signs, a zero
        const mag: f32 = if (i % 7 == 0) 0 else std.math.ldexp(@as(f32, 1.0) + @as(f32, @floatFromInt(i % 13)) / 16.0, @as(i32, @intCast(i / group)) * 5 - 12);
        const bits: u32 = @bitCast(if (i % 3 == 0) -mag else mag);
        std.mem.writeInt(u16, bf16[i * 2 ..][0..2], @intCast(bits >> 16), .little);
        w.* = @bitCast(bits & 0xFFFF0000);
    }
    var codes: [2 * k]u8 = undefined;
    var scales: [2 * k / group]u8 = undefined;
    encode(&bf16, 2, k, &codes, &scales);
    for (want, codes, 0..) |w, c, i| {
        const e = @as(i32, scales[i / group]) - 127;
        const got = std.math.ldexp(core.exlr8.codec.e4m3Value(c), e);
        try std.testing.expect(@abs(core.exlr8.codec.e4m3Value(c)) <= top_code);
        try std.testing.expect(@abs(got - w) <= @abs(w) / 8.0 + std.math.ldexp(@as(f32, 1), e - 9));
    }
    // the largest weight of every group sits in E4M3's top binade
    for (0..2 * k / group) |g| {
        var top: f32 = 0;
        for (0..group) |i| top = @max(top, @abs(core.exlr8.codec.e4m3Value(codes[g * group + i])));
        try std.testing.expect(top > top_code / 2 - 32 and top <= top_code);
    }
}

test "under mxfp8 a drafter linear stored as anything but BF16 stops the load, and checkpoint keeps every format" {
    defer projection.recode = .checkpoint;
    projection.recode = .mxfp8;
    try projection.recodable("language_model.mtp.fc_embedding", .bf16);
    try std.testing.expectError(error.InvalidEnvironment, projection.recodable("language_model.mtp.fc_embedding", .f8_e4m3));
    try std.testing.expectError(error.InvalidEnvironment, projection.recodable("language_model.mtp.fc_hidden", .u32));
    projection.recode = .checkpoint;
    try projection.recodable("language_model.mtp.fc_hidden", .u32);
    // every loader the drafter's window reaches asks before it packs a stored format
    const load_src = @embedFile("cuda_projection_load.zig");
    try std.testing.expect(std.mem.count(u8, load_src, "try projection.recodable(prefix, t.dtype);") == 1);
    try std.testing.expect(std.mem.count(u8, load_src, "try projection.recodable(prefix, first.dtype);") == 1);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_prompt_wt.zig"), "try @import(\"cuda_projection.zig\").recodable(prefix, down_tensor.dtype);") != null);
}
