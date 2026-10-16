//! Rank-local NVFP4 bytes against the independent FP64 decoder and real safetensors headers.

const std = @import("std");
const reader = @import("nvfp4_slice.zig");
const st = @import("safetensors.zig");
const ckpt = @import("checkpoint.zig");
const quant = @import("quant/quant.zig");

const Fixture = struct {
    codes: [512]u8 = undefined,
    scales: [64]u8 = undefined,
    fixed: [48]u8 = @splat(0),
    exceptions: [4]u8 = .{ 3, 0, 0, 0x40 },
    global: [4]u8 = undefined,

    fn init() Fixture {
        var f: Fixture = .{};
        std.mem.writeInt(u32, &f.global, @bitCast(@as(f32, 0.25)), .little);
        for (&f.codes, 0..) |*b, i| b.* = @truncate(i * 29 + 3);
        for (0..16) |row| {
            const base: u8 = @intCast(0x30 + row % 8);
            f.fixed[row] = base;
            f.fixed[16 + row * 2] = 0x10;
            f.fixed[17 + row * 2] = 0x32;
            for (0..4) |column| f.scales[row * 4 + column] = base + @as(u8, @intCast(column));
        }
        f.scales[3] = 0x40;
        return f;
    }

    fn source(f: *const Fixture, compressed: bool) reader.Source {
        return .{
            .weight = tensor(.u8, &.{ 16, 32 }, &f.codes),
            .scales = if (compressed) .{ .csf = .{
                .fixed = tensor(.u8, &.{ 1, 48 }, &f.fixed),
                .exceptions = tensor(.u32, &.{1}, &f.exceptions),
            } } else .{ .plain = tensor(.f8_e4m3, &.{ 16, 4 }, &f.scales) },
            .global = tensor(.f32, &.{}, &f.global),
        };
    }
};

fn tensor(dtype: st.DType, shape: []const usize, bytes: []const u8) st.Tensor {
    var out: st.Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(out.shape[0..shape.len], shape);
    return out;
}

fn reference(linear: reader.Linear, out: []f64) !void {
    try quant.nvfp4.reference(.{
        .weight = tensor(.u8, &.{ linear.n, linear.k / 2 }, linear.codes),
        .scale = tensor(.f8_e4m3, &.{ linear.n, linear.k / 16 }, linear.scales),
        .global = &.{linear.global},
    }, out);
}

test "plain and compressed rows/columns preserve every FP64 decoded value" {
    const f = Fixture.init();
    const a = std.testing.allocator;
    var whole = try reader.slice(a, f.source(false), .{ .from = 0, .to = 16 }, .{ .from = 0, .to = 64 });
    defer whole.deinit(a);
    var expected: [1024]f64 = undefined;
    try reference(whole, &expected);
    for ([_]bool{ false, true }) |compressed| {
        for (0..2) |axis| for (0..2) |rank| {
            const r0 = if (axis == 0) rank * 8 else 0;
            const rn: usize = if (axis == 0) 8 else 16;
            const c0 = if (axis == 1) rank * 32 else 0;
            const cn: usize = if (axis == 1) 32 else 64;
            var part = try reader.slice(a, f.source(compressed), .{ .from = r0, .to = r0 + rn }, .{ .from = c0, .to = c0 + cn });
            defer part.deinit(a);
            var out: [512]f64 = undefined;
            try reference(part, &out);
            for (0..rn) |row| try std.testing.expectEqualSlices(f64, expected[(r0 + row) * 64 + c0 ..][0..cn], out[row * cn ..][0..cn]);
        };
    }
}

test "bad compressed tensor metadata and cuts cannot become a rank projection" {
    const f = Fixture.init();
    var source = f.source(true);
    const bad = std.testing.expectError;
    source.scales.csf.exceptions.dtype = .i32;
    try bad(error.BadNvfp4, reader.dimensions(source));
    source = f.source(true);
    source.scales.csf.fixed.shape[1] -= 1;
    try bad(error.BadNvfp4, reader.dimensions(source));
    source = f.source(false);
    source.global.dtype = .bf16;
    try bad(error.BadNvfp4, reader.dimensions(source));
    const a = std.testing.allocator;
    try bad(error.BadNvfp4Slice, reader.slice(a, f.source(true), .{ .from = 0, .to = 17 }, .{ .from = 0, .to = 64 }));
    try bad(error.BadNvfp4Slice, reader.slice(a, f.source(true), .{ .from = 8, .to = 8 }, .{ .from = 0, .to = 64 }));
    try bad(error.BadNvfp4Slice, reader.slice(a, f.source(true), .{ .from = 0, .to = 16 }, .{ .from = 1, .to = 32 }));
}

test "allocation failures free every rank-local buffer" {
    const f = Fixture.init();
    for (0..3) |fail_at| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_at });
        const a = failing.allocator();
        if (reader.slice(a, f.source(true), .{ .from = 0, .to = 16 }, .{ .from = 0, .to = 32 })) |value| {
            var owned = value;
            owned.deinit(a);
        } else |e| try std.testing.expectEqual(error.OutOfMemory, e);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    }
}

test "checkpoint reading consumes CSF streams and preserves optional calibration" {
    const f = Fixture.init();
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header =
        \\{"l.weight":{"dtype":"U8","shape":[16,32],"data_offsets":[0,512]},
        \\ "l.weight_scale.nvfp4_csf_fixed":{"dtype":"U8","shape":[1,48],"data_offsets":[512,560]},
        \\ "l.weight_scale.nvfp4_csf_exceptions":{"dtype":"U32","shape":[1],"data_offsets":[560,564]},
        \\ "l.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[564,568]},
        \\ "l.input_scale":{"dtype":"F32","shape":[1],"data_offsets":[568,572]}}
    ;
    var image: [8 + header.len + 572]u8 = undefined;
    std.mem.writeInt(u64, image[0..8], header.len, .little);
    @memcpy(image[8..][0..header.len], header);
    const data = image[8 + header.len ..];
    @memcpy(data[0..512], &f.codes);
    @memcpy(data[512..560], &f.fixed);
    @memcpy(data[560..564], &f.exceptions);
    @memcpy(data[564..568], &f.global);
    @memcpy(data[568..572], &f.global);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &image });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    var ck = try ckpt.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    var part = try reader.read(&ck, a, "l", .{ .from = 8, .to = 16 }, .{ .from = 32, .to = 64 });
    defer part.deinit(a);
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
    for (0..8) |r| try std.testing.expectEqualSlices(u8, f.scales[(8 + r) * 4 + 2 ..][0..2], part.scales[r * 2 ..][0..2]);
}
