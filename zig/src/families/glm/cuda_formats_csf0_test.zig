//! csf0_v1 as the loader reads it: by its recipe and its tensors' shapes (cuda_formats.zig).

const std = @import("std");
const core = @import("core");
const formats = @import("cuda_formats.zig");

const of = formats.of;
const Format = formats.Format;
const Meta = core.nvfp4.Meta;
const Refusal = formats.Refusal;
const Stored = formats.Stored;
const csf0Linear = formats.csf0Linear;
const testing = std.testing;

fn meta(dtype: core.checkpoint.DType, shape: []const usize) Meta {
    return .{ .dtype = dtype, .shape = shape };
}

test "csf0_v1 is read by its declared recipe and its tensors' shapes, whose inputs come in 64s" {
    const w = meta(.u8, &.{ 2048, 2048 });
    const bases = meta(.u8, &.{2048});
    const offsets = meta(.u8, &.{ 2048, 64 });
    const routed = Stored{ .weight = w, .csf0 = .{ .bases = bases, .offsets = offsets } };
    try testing.expectEqual(Format.nvfp4_csf0, try of(.{ .mixed = .nvfp4_csf0 }, routed));
    // declared otherwise: refused, whatever else the checkpoint holds
    try testing.expectError(error.FormatMismatch, of(.nvfp4, routed));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4 }, routed));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4_csf }, routed));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = null }, routed));
    // declared csf0_v1, stored otherwise
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4_csf0 }, .{ .weight = w, .scale = meta(.f8_e4m3, &.{ 2048, 256 }) }));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4_csf0 }, .{ .weight = w, .csf = true }));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4_csf0 }, .{ .weight = meta(.bf16, &.{ 2048, 4096 }), .csf0 = routed.csf0 }));
    // both encodings at once, or csf0_v1 beside a plain scale
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4_csf0 }, .{ .weight = w, .csf = true, .csf0 = routed.csf0 }));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4_csf0 }, .{ .weight = w, .scale = meta(.f8_e4m3, &.{ 2048, 256 }), .csf0 = routed.csf0 }));
    // the shapes: a base a row, a byte every 64 inputs
    const shaped = struct {
        fn of_(b: Meta, o: Meta) Refusal!Format {
            return of(.{ .mixed = .nvfp4_csf0 }, .{ .weight = meta(.u8, &.{ 2048, 2048 }), .csf0 = .{ .bases = b, .offsets = o } });
        }
    };
    try testing.expectEqual(Format.nvfp4_csf0, try shaped.of_(bases, offsets));
    try testing.expectError(error.FormatMismatch, shaped.of_(meta(.u8, &.{1024}), offsets));
    try testing.expectError(error.FormatMismatch, shaped.of_(meta(.f8_e4m3, &.{2048}), offsets));
    try testing.expectError(error.FormatMismatch, shaped.of_(meta(.u8, &.{ 2048, 1 }), offsets));
    try testing.expectError(error.FormatMismatch, shaped.of_(bases, meta(.u8, &.{ 2048, 128 })));
    try testing.expectError(error.FormatMismatch, shaped.of_(bases, meta(.u8, &.{ 1024, 64 })));
    try testing.expectError(error.FormatMismatch, shaped.of_(bases, meta(.u8, &.{2048 * 64})));
    // k = 96: 3 groups of 32 make no last pair
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4_csf0 }, .{ .weight = meta(.u8, &.{ 64, 48 }), .csf0 = .{ .bases = meta(.u8, &.{64}), .offsets = meta(.u8, &.{ 64, 1 }) } }));
}

test "a csf0_v1 linear reads every part, refuses a width or shape it cannot hold, and leaves a block 16 linear alone" {
    const gpa = testing.allocator;
    const n = 32;
    const k = 128;
    var codes: [n * k / 2]u8 = undefined;
    var prng: std.Random.DefaultPrng = .init(9);
    prng.random().bytes(&codes);
    var bases: [n]u8 = undefined;
    prng.random().bytes(&bases);
    var offsets: [n * k / 64]u8 = undefined;
    prng.random().bytes(&offsets);
    const plain: [n * k / 16]u8 = @splat(0x30);
    const one = std.mem.toBytes(@as(f32, 0.25));
    const wide: [n * 3]u8 = @splat(0);
    const tensors = [_]core.shard_edit.New{
        .{ .name = "a.weight", .dtype = .u8, .shape = &.{ n, k / 2 }, .bytes = &codes },
        .{ .name = "a.weight_scale.csf0_bases", .dtype = .u8, .shape = &.{n}, .bytes = &bases },
        .{ .name = "a.weight_scale.csf0_codes", .dtype = .u8, .shape = &.{ n, k / 64 }, .bytes = &offsets },
        .{ .name = "a.weight_scale_2", .dtype = .f32, .shape = &.{}, .bytes = &one },
        .{ .name = "a.input_scale", .dtype = .f32, .shape = &.{}, .bytes = &one },
        .{ .name = "b.weight", .dtype = .u8, .shape = &.{ n, k / 2 }, .bytes = &codes },
        .{ .name = "b.weight_scale", .dtype = .f8_e4m3, .shape = &.{ n, k / 16 }, .bytes = &plain },
        .{ .name = "b.weight_scale_2", .dtype = .f32, .shape = &.{}, .bytes = &one },
        .{ .name = "b.input_scale", .dtype = .f32, .shape = &.{}, .bytes = &one },
        .{ .name = "c.weight", .dtype = .u8, .shape = &.{ n, 48 }, .bytes = codes[0 .. n * 48] }, // k = 96
        .{ .name = "d.weight", .dtype = .u8, .shape = &.{ n, k / 2 }, .bytes = &codes },
        .{ .name = "d.weight_scale.csf0_bases", .dtype = .u8, .shape = &.{n}, .bytes = &bases },
        .{ .name = "d.weight_scale.csf0_codes", .dtype = .u8, .shape = &.{ n, 3 }, .bytes = &wide },
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir: [64]u8 = undefined;
    const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var path: [96]u8 = undefined;
    try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, &tensors);
    var ck = try core.Checkpoint.openModel(gpa, testing.io, at);
    defer ck.close();
    const got = try csf0Linear(&ck, "a");
    try testing.expectEqual(@as(usize, n), got.n);
    try testing.expectEqual(@as(usize, k), got.k);
    try testing.expectEqual(@as(f32, 0.25), got.global);
    try testing.expectEqualSlices(u8, &codes, got.matrix.codes);
    try testing.expectEqualSlices(u8, &bases, got.matrix.bases);
    try testing.expectEqualSlices(u8, &offsets, got.matrix.offsets);
    const block16 = try core.nvfp4.Linear.read(&ck, "b");
    try testing.expectEqualSlices(u8, &plain, block16.scales);
    try testing.expectError(error.BadNvfp4, csf0Linear(&ck, "c"));
    try testing.expectError(error.BadNvfp4, csf0Linear(&ck, "d"));
    try testing.expectEqual(@as(usize, 0), ck.unused());
}
