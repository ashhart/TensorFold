//! The common device arithmetic is checked against independent binary64 dots and real source controls.
const std = @import("std");
const wo = @import("core").weight_only;
const q = @import("core").quant;
extern fn tf_weight_only_params_size() c_int;
extern fn tf_weight_only_cpu(x: [*]const u16, w: [*]const u8, s: [*]const u8, b: [*]const u8, out: [*]u16, p: wo.Params, control: c_int, repetition: c_int) void;
const a = std.testing.allocator;
fn fp(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
fn bit(x: f32) u16 {
    return q.convert.bf16(x);
}
fn asBytes(bits: []const u16) []const u8 {
    return std.mem.sliceAsBytes(bits);
}
fn source(p: wo.Params, w: []u8, s: []u8, b: []u8) void {
    switch (p.format) {
        .bf16 => for (0..w.len / 2) |i| std.mem.writeInt(u16, w[2 * i ..][0..2], bit(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16), .little),
        .mlx4, .mlx8 => {
            @memset(w, 0x72);
            for (0..s.len / 2) |i| {
                std.mem.writeInt(u16, s[2 * i ..][0..2], 0x3c80, .little);
                std.mem.writeInt(u16, b[2 * i ..][0..2], 0xbd00, .little);
            }
        },
        .fp8 => {
            for (w, 0..) |*v, i| {
                v.* = @as(u8, @intCast(i % 119)) ^ (if (i % 3 == 0) @as(u8, 128) else 0);
            }
        },
        .nvfp4 => {
            @memset(w, 0xb6);
            @memset(s, 0x38);
        },
    }
}
fn weight(p: wo.Params, w: []const u8, s: []const u8, b: []const u8, col: usize, k: usize) f64 {
    const width: usize = @intCast(p.k);
    const at = col * width + k;
    return switch (p.format) {
        .bf16 => fp(std.mem.readInt(u16, w[2 * at ..][0..2], .little)),
        .fp8 => q.fp8.e4m3(w[at]) * @as(f64, p.global_scale),
        .nvfp4 => q.nvfp4.e2m1(@as(u4, @truncate(w[at / 2] >> @as(u3, @intCast((at % 2) * 4))))) * q.fp8.e4m3(s[col * (width / 16) + k / 16]) * @as(f64, p.global_scale),
        .mlx4, .mlx8 => blk: {
            const bits: usize = if (p.format == .mlx4) 4 else 8;
            const code = (w[at * bits / 8] >> @as(u3, @intCast((at * bits) % 8))) & (if (bits == 4) @as(u8, 15) else 255);
            const group = col * (width / @as(usize, @intCast(p.group))) + k / @as(usize, @intCast(p.group));
            break :blk @as(f64, @floatFromInt(code)) * fp(std.mem.readInt(u16, s[group * 2 ..][0..2], .little)) + fp(std.mem.readInt(u16, b[group * 2 ..][0..2], .little));
        },
    };
}
test "weight-only projection: raw layouts match binary64 on row and column tails" {
    try std.testing.expectEqual(@as(c_int, @sizeOf(wo.Params)), tf_weight_only_params_size());
    for (std.enums.values(wo.Format)) |format| for ([_]usize{ 1, 3, 8 }) |rows| {
        const p: wo.Params = .{ .rows = @intCast(rows), .n = 7, .k = 128, .format = format, .group = 64, .scale_kind = .bf16, .global_scale = 0.5 };
        const sizes = try wo.validate(p);
        const x = try a.alloc(u16, sizes.x / 2);
        defer a.free(x);
        const w = try a.alloc(u8, sizes.weight);
        defer a.free(w);
        const s = try a.alloc(u8, sizes.scales);
        defer a.free(s);
        const b = try a.alloc(u8, sizes.biases);
        defer a.free(b);
        const out = try a.alloc(u16, sizes.out / 2);
        defer a.free(out);
        for (x, 0..) |*v, i| v.* = bit(@as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 8);
        source(p, w, s, b);
        tf_weight_only_cpu(x.ptr, w.ptr, s.ptr, b.ptr, out.ptr, p, 0, 0);
        for (0..rows) |r| for (0..7) |col| {
            var sum: f64 = 0;
            for (0..128) |k| sum += @as(f64, fp(x[r * 128 + k])) * weight(p, w, s, b, col, k);
            try std.testing.expectEqual(bit(@floatCast(sum)), out[r * 7 + col]);
        };
    };
}
test "weight-only projection: actual source NaN repeat tail and BF16 accumulator controls are detected" {
    const p: wo.Params = .{ .rows = 1, .n = 1, .k = 128, .format = .bf16, .group = 64, .scale_kind = .bf16 };
    var x: [128]u16 = @splat(0);
    var w: [128]u16 = @splat(0x3f80);
    var out: [1]u16 = undefined;
    const empty: [0]u8 = .{};
    x[0] = 0x7fc0;
    tf_weight_only_cpu(&x, asBytes(&w).ptr, &empty, &empty, &out, p, 0, 0);
    try std.testing.expect(std.math.isNan(fp(out[0])));
    tf_weight_only_cpu(&x, asBytes(&w).ptr, &empty, &empty, &out, p, 1, 0);
    try std.testing.expect(!std.math.isNan(fp(out[0])));
    @memset(&x, 0);
    x[127] = 0x3f80;
    tf_weight_only_cpu(&x, asBytes(&w).ptr, &empty, &empty, &out, p, 0, 0);
    const correct = out[0];
    tf_weight_only_cpu(&x, asBytes(&w).ptr, &empty, &empty, &out, p, 3, 0);
    try std.testing.expect(correct != out[0]);
    tf_weight_only_cpu(&x, asBytes(&w).ptr, &empty, &empty, &out, p, 2, 0);
    const first = out[0];
    tf_weight_only_cpu(&x, asBytes(&w).ptr, &empty, &empty, &out, p, 2, 1);
    try std.testing.expect(first != out[0]);
    @memset(&x, 0x4780);
    x[0] = bit(65536 + 512);
    for (64..128) |i| x[i] |= 0x8000;
    tf_weight_only_cpu(&x, asBytes(&w).ptr, &empty, &empty, &out, p, 0, 0);
    try std.testing.expectEqual(bit(512), out[0]);
    tf_weight_only_cpu(&x, asBytes(&w).ptr, &empty, &empty, &out, p, 4, 0);
    try std.testing.expect(bit(512) != out[0]);
}
