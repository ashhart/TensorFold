//! Split routed experts from the standard Hy4-flash EXL3 checkpoint.
const std = @import("std");
const core = @import("core");
const Tensor = core.safetensors.Tensor;

pub const Projection = struct {
    parts: core.exl3_load.Parts,
    geometry: core.exl3_layout.Group,
};

pub const Expert = struct { gate: Projection, up: Projection, down: Projection };

pub fn readExpert(source: anytype, layer: usize, expert: usize, latent: usize, intermediate: usize) !Expert {
    return .{
        .gate = try read(source, layer, expert, "gate", latent, intermediate),
        .up = try read(source, layer, expert, "up", latent, intermediate),
        .down = try read(source, layer, expert, "down", intermediate, latent),
    };
}

fn read(source: anytype, layer: usize, expert: usize, name: []const u8, k: usize, n: usize) !Projection {
    var buf: [160]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj", .{ layer, expert, name });
    for ([_][]const u8{ "mcg", "bias", "weight", "su", "sv" }) |suffix| {
        var full: [192]u8 = undefined;
        if (source.has(try std.fmt.bufPrint(&full, "{s}.{s}", .{ prefix, suffix })))
            return error.UnsupportedHy4Expert;
    }
    return validate(
        try tensor(source, prefix, "trellis"),
        try tensor(source, prefix, "suh"),
        try tensor(source, prefix, "svh"),
        try tensor(source, prefix, "mul1"),
        k,
        n,
    );
}

fn tensor(source: anytype, prefix: []const u8, suffix: []const u8) !Tensor {
    var buf: [192]u8 = undefined;
    return source.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, suffix }));
}

fn validate(trellis: Tensor, suh: Tensor, svh: Tensor, marker: Tensor, k: usize, n: usize) !Projection {
    if (trellis.dtype != .i16 or trellis.rank != 3 or k % 128 != 0 or n % 128 != 0)
        return error.BadHy4Trellis;
    const g = try core.exl3_layout.Group.fromShape(trellis.shape[0..trellis.rank]);
    if (g.k() != k or g.n() != n or trellis.bytes.len != g.trellisBytes())
        return error.BadHy4Trellis;
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n}) or suh.bytes.len != 2 * k or svh.bytes.len != 2 * n)
        return error.BadHy4ExpertScales;
    if ((!marker.is(.i32, &.{}) and !marker.is(.i32, &.{1})) or marker.bytes.len != 4)
        return error.BadHy4ExpertMarker;
    if (try core.exl3_layout.fromMarker(std.mem.readInt(u32, marker.bytes[0..4], .little)) != .mul1)
        return error.BadHy4ExpertMarker;
    return .{ .parts = .{ .trellis = trellis, .suh = suh, .svh = svh }, .geometry = g };
}

fn fixed(dtype: core.safetensors.DType, shape: []const usize, bytes: []const u8) Tensor {
    var out: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(out.shape[0..shape.len], shape);
    return out;
}

test "Hy4 orders EXL3 axes as input then output and requires fp16 scales and mul1" {
    const t = std.testing;
    const words: [8 * 16 * 64 * 2]u8 = @splat(0);
    const input: [128 * 2]u8 = @splat(0);
    const output: [256 * 2]u8 = @splat(0);
    const magic = &.{ 0x2d, 0xd1, 0xdc, 0x83 };
    const trellis = fixed(.i16, &.{ 8, 16, 64 }, &words);
    const suh = fixed(.f16, &.{128}, &input);
    const svh = fixed(.f16, &.{256}, &output);
    const marker = fixed(.i32, &.{1}, magic);
    const p = try validate(trellis, suh, svh, marker, 128, 256);
    try t.expectEqual(@as(u32, 8), p.geometry.k2());
    _ = try validate(trellis, suh, svh, fixed(.i32, &.{}, magic), 128, 256);
    try t.expectError(error.BadHy4Trellis, validate(trellis, suh, svh, marker, 256, 128));
    try t.expectError(error.BadHy4ExpertScales, validate(trellis, suh, suh, marker, 128, 256));
    try t.expectError(error.BadHy4ExpertMarker, validate(trellis, suh, svh, fixed(.u32, &.{1}, magic), 128, 256));
    try t.expectError(error.BadHy4ExpertMarker, validate(trellis, suh, svh, fixed(.i32, &.{}, &.{ 0xed, 0x1f, 0xac, 0xcb }), 128, 256));
}
