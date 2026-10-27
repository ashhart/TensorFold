//! File-backed projections and expert planes through the shared quantized loader, without GPU execution.
const std = @import("std");
const checkpoint = @import("../checkpoint.zig");
const fixture = @import("../checkpoint_fixture.zig");
const quant = @import("quant.zig");
const mlx = quant.mlx;
const a = std.testing.allocator;
const io = std.testing.io;

const Table = struct {
    ck: *checkpoint.Checkpoint,
    pub fn get(t: Table, name: []const u8) !quant.Tensor {
        return t.ck.get(name);
    }
    pub fn has(t: Table, name: []const u8) bool {
        return t.ck.has(name);
    }
    pub fn width(_: Table, _: []const u8) error{UnsupportedQuantization}!mlx.Width {
        return .{ .bits = 4, .group = 32 };
    }
};

fn floatBytes(dtype: quant.DType, bytes: []u8, value: f32) void {
    for (0..bytes.len / dtype.size()) |i| switch (dtype) {
        .f32 => std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little),
        .bf16 => std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @truncate(@as(u32, @bitCast(value)) >> 16), .little),
        .f16 => std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(value))), .little),
        // Exact F64 midpoints round to even F32 values 2 and -1; the expected byte tables encode those directly.
        .f64 => std.mem.writeInt(u64, bytes[i * 8 ..][0..8], @bitCast(@as(f64, if (value == 2) 2 + 0x1p-23 else -1 - 0x1p-24)), .little),
        else => unreachable,
    };
}

fn planes(arena: std.mem.Allocator, entries: *std.ArrayList(fixture.Entry), key: []const u8, count: ?usize, code: u8, table_dtype: quant.DType) !void {
    const experts = count orelse 1;
    inline for (.{ "weight", "scales", "biases" }) |suffix| {
        const words = comptime std.mem.eql(u8, suffix, "weight");
        const dtype: quant.DType = if (words) .u32 else table_dtype;
        const columns: usize = if (words) 4 else 1;
        const shape = if (count != null)
            try arena.dupe(usize, &.{ experts, 2, columns })
        else
            try arena.dupe(usize, &.{ 2, columns });
        const bytes = try arena.alloc(u8, experts * 2 * columns * dtype.size());
        for (0..experts) |expert| {
            const row = bytes[expert * bytes.len / experts ..][0 .. bytes.len / experts];
            if (words) {
                const value = code + @as(u8, @intCast(expert));
                @memset(row, value | (value << 4));
            } else floatBytes(dtype, row, if (comptime std.mem.eql(u8, suffix, "scales")) 2 else -1);
        }
        try entries.append(arena, .{
            .name = try std.fmt.allocPrint(arena, "{s}.{s}", .{ key, suffix }),
            .dtype = switch (dtype) {
                .u32 => "U32",
                .bf16 => "BF16",
                .f16 => "F16",
                .f32 => "F32",
                .f64 => "F64",
                else => unreachable,
            },
            .shape = shape,
            .bytes = bytes,
        });
    }
}

fn device(h: mlx.Host) mlx.Device {
    var out: mlx.Device = .{ .words = undefined, .scales = undefined, .biases = undefined, .bits = h.bits, .group = h.group };
    inline for (.{ "words", "scales", "biases" }) |name| {
        const tensor = @field(h, name);
        @field(out, name) = .{ .ptr = @intFromPtr(tensor.bytes.ptr), .len = tensor.bytes.len, .dtype = tensor.dtype, .rank = tensor.rank, .shape = tensor.shape };
    }
    return out;
}

fn checkPair(routed_type: quant.DType, shared_type: quant.DType) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var routed: std.ArrayList(fixture.Entry) = .empty;
    var shared: std.ArrayList(fixture.Entry) = .empty;
    try planes(alloc, &routed, "projection", null, 1, routed_type);
    inline for (.{ "gate_proj", "up_proj", "down_proj" }) |name| {
        try planes(alloc, &routed, "mlp.switch_mlp." ++ name, 2, 1, routed_type);
        try planes(alloc, &shared, "mlp.shared_expert." ++ name, null, 3, shared_type);
    }
    try fixture.write(a, io, tmp.dir, "model-01.safetensors", routed.items);
    try fixture.write(a, io, tmp.dir, "model-02.safetensors", shared.items);
    var ck = try checkpoint.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    const table = Table{ .ck = &ck };
    const projection = try quant.read(alloc, table, "projection");
    try std.testing.expectEqualSlices(u8, routed.items[0].bytes, projection.mlx.words.bytes);
    var projection_values: [64]f64 = undefined;
    try mlx.reference(projection.mlx, &projection_values);
    for (projection_values) |value| try std.testing.expectEqual(@as(f64, 1), value);
    const pv = try mlx.view(device(projection.mlx), false);
    try std.testing.expectEqual(@as(u32, 2), pv.n);
    try std.testing.expectEqual(@as(u32, 32), pv.k);
    const experts = try quant.experts.experts(alloc, table, "mlp.");
    try std.testing.expectEqual(@as(usize, 3), experts.count);
    try std.testing.expect(experts.gated);
    const ev = try mlx.view(device(experts.fused.mlx), true);
    try std.testing.expectEqual(@as(u32, 4), ev.n);
    try std.testing.expectEqual(@as(u32, 32), ev.k);
    const expected_type = if (routed_type == shared_type and quant.convert.tableDtype(routed_type)) routed_type else quant.DType.f32;
    try std.testing.expectEqual(expected_type, experts.fused.mlx.scales.dtype);
    for ([_]mlx.Host{ experts.fused.mlx, experts.down.mlx }) |h| {
        const scales = try alloc.alloc(u8, h.scales.bytes.len);
        const biases = try alloc.alloc(u8, h.biases.bytes.len);
        floatBytes(expected_type, scales, 2);
        floatBytes(expected_type, biases, -1);
        try std.testing.expectEqualSlices(u8, scales, h.scales.bytes);
        try std.testing.expectEqualSlices(u8, biases, h.biases.bytes);
    }
    var values: [3 * 4 * 32]f64 = undefined;
    try mlx.reference(experts.fused.mlx, &values);
    for (values, 0..) |value, i| try std.testing.expectEqual(@as(f64, @floatFromInt(2 * (i / 128) + 1)), value);
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

test "affine geometry file-backed homogeneous BF16 projection and expert planes keep every byte" {
    try checkPair(.bf16, .bf16);
}

test "affine geometry file-backed F32 routed and BF16 shared experts convert their own planes exactly" {
    try checkPair(.f32, .bf16);
}

test "affine geometry file-backed BF16 routed and F32 shared experts convert their own planes exactly" {
    try checkPair(.bf16, .f32);
}

test "affine geometry file-backed F16 planes work in both mixed directions and homogeneous table dtypes" {
    try checkPair(.f16, .bf16);
    try checkPair(.bf16, .f16);
    try checkPair(.f16, .f32);
    try checkPair(.f32, .f16);
    try checkPair(.f16, .f16);
    try checkPair(.f32, .f32);
}

test "affine geometry file-backed F64 planes narrow through the existing conversion with exact midpoint rounding" {
    try checkPair(.f64, .f64);
    try checkPair(.f64, .bf16);
    try checkPair(.bf16, .f64);
    try checkPair(.f64, .f16);
    try checkPair(.f16, .f64);
    try checkPair(.f64, .f32);
    try checkPair(.f32, .f64);
}
