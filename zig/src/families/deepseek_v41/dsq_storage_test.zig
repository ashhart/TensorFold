//! dsq-1 metadata cannot bypass the native storage owners; synthetic groups exercise their original checks.
const std = @import("std");
const Config = @import("config.zig").Config;
const pk = @import("pack.zig");
const Plan = @import("plan.zig").CheckpointPlan;
const exl3 = @import("core").exl3_layout;
const t = std.testing;
const public_config = @embedFile("fixtures/flash1017-dsq-config.json");

fn groupImage(words: usize, output_scale: usize, marker: bool) ![]u8 {
    const specs = [_]pk.Spec{
        .{ .name = "p.trellis", .dtype = "I16", .shape = &.{ 8, 8, words }, .fill = 7 },
        .{ .name = "p.suh", .dtype = "F16", .shape = &.{128} },
        .{ .name = "p.svh", .dtype = "F16", .shape = &.{output_scale} },
        .{ .name = "p.mul1", .dtype = "I32", .shape = &.{} },
    };
    return pk.image(t.allocator, specs[0..if (marker) 4 else 3]);
}

test "dsq admission: config does not waive native trellis, scale, header or marker checks" {
    _ = try Config.parse(t.allocator, public_config);
    const bad_shape = try groupImage(8, 128, true);
    defer t.allocator.free(bad_shape);
    var shapes = pk.Pack.init(t.allocator);
    defer shapes.deinit();
    try shapes.addHeaderDump("shape", bad_shape);
    try t.expectError(error.BadTrellis, Plan.init(t.allocator, &shapes, &.{"p"}));
    const bad_scale = try groupImage(32, 64, true);
    defer t.allocator.free(bad_scale);
    var scales = pk.Pack.init(t.allocator);
    defer scales.deinit();
    try scales.addHeaderDump("scale", bad_scale);
    try t.expectError(error.BadScales, Plan.init(t.allocator, &scales, &.{"p"}));
    const odd = try groupImage(24, 128, false);
    defer t.allocator.free(odd);
    var missing_marker = pk.Pack.init(t.allocator);
    defer missing_marker.deinit();
    try missing_marker.addHeaderDump("odd", odd);
    try t.expectError(error.BadMarker, Plan.init(t.allocator, &missing_marker, &.{"p"}));
    const malformed_marker = try pk.image(t.allocator, &.{
        .{ .name = "p.trellis", .dtype = "I16", .shape = &.{ 8, 8, 32 } },
        .{ .name = "p.suh", .dtype = "F16", .shape = &.{128} },
        .{ .name = "p.svh", .dtype = "F16", .shape = &.{128} },
        .{ .name = "p.mul1", .dtype = "I32", .shape = &.{1} },
    });
    defer t.allocator.free(malformed_marker);
    var markers = pk.Pack.init(t.allocator);
    defer markers.deinit();
    try markers.addHeaderDump("marker", malformed_marker);
    try t.expectError(error.BadMarker, Plan.init(t.allocator, &markers, &.{"p"}));
    try t.expectError(error.UnknownCodebookMarker, exl3.fromMarker(0));
    var truncated = pk.Pack.init(t.allocator);
    defer truncated.deinit();
    try t.expectError(error.BadSafetensors, truncated.addHeaderDump("truncated", &.{ 1, 2, 3 }));
}

test "dsq admission: native file bytes select mul1 and invalid magic is refused without changing storage" {
    _ = try Config.parse(t.allocator, public_config);
    const image = try groupImage(32, 128, true);
    defer t.allocator.free(image);
    std.mem.writeInt(u32, image[image.len - 4 ..][0..4], exl3.marker_mul1, .little);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile(t.io, "model.safetensors", .{ .read = true });
    defer file.close(t.io);
    try file.writePositionalAll(t.io, image, 0);
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    var pack = try pk.Pack.open(t.allocator, t.io, dir);
    defer pack.deinit();
    var pl = try Plan.init(t.allocator, &pack, &.{"p"});
    defer pl.deinit();
    try t.expectEqual(exl3.Codebook.mul1, try pack.codebook(t.io, "p"));
    try t.expectEqual(@as(u32, 4), (try pl.projections[0].storage.cudaGroup()).k2());
    const info = pack.get("p.trellis").?;
    const bytes = try t.allocator.alloc(u8, @intCast(info.nbytes));
    defer t.allocator.free(bytes);
    try pack.read(t.io, info, .{ .offset = 0, .len = info.nbytes }, bytes);
    for (bytes) |byte| try t.expectEqual(@as(u8, 7), byte);
    try file.writePositionalAll(t.io, &.{ 0, 0, 0, 0 }, image.len - 4);
    try t.expectError(error.UnknownCodebookMarker, pack.codebook(t.io, "p"));
}
