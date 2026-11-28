//! Reader.exl3Group's codebook markers: one picks the codebook, two make the group ambiguous and refused.
const std = @import("std");
const t = std.testing;
const core = @import("core");
const proj = @import("weights_proj.zig");

const fixture = core.checkpoint_fixture;
const Codebook = @FieldType(proj.Reader.Group, "codebook");
const mcg = std.mem.toBytes(@as(u32, 0xCBAC1FED));
const mul1 = std.mem.toBytes(@as(u32, 0x83DCD12D));

/// Group `m` [128, 128] at 2 bits (trellis [8, 8, 32] int16, fp16 scales) with the markers named, read back.
fn group(markers: []const []const u8) !proj.Reader.Group {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const trellis: [8 * 8 * 32 * 2]u8 = @splat(0);
    const scales: [128 * 2]u8 = @splat(0);
    var entries: std.ArrayList(fixture.Entry) = .empty;
    try entries.appendSlice(a, &.{
        .{ .name = "m.trellis", .dtype = "I16", .shape = &.{ 8, 8, 32 }, .bytes = &trellis },
        .{ .name = "m.suh", .dtype = "F16", .shape = &.{128}, .bytes = &scales },
        .{ .name = "m.svh", .dtype = "F16", .shape = &.{128}, .bytes = &scales },
    });
    for (markers) |name| try entries.append(a, .{
        .name = try std.fmt.allocPrint(a, "m.{s}", .{name}),
        .dtype = "I32",
        .shape = &.{1},
        .bytes = if (std.mem.eql(u8, name, "mcg")) &mcg else &mul1,
    });
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.write(a, t.io, tmp.dir, "model.safetensors", entries.items);
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    return proj.Reader.exl3Group(&ck, "m", 128, 128);
}

test "an EXL3 group takes its codebook from its one marker" {
    try t.expectEqual(Codebook.mcg, (try group(&.{"mcg"})).codebook);
    try t.expectEqual(Codebook.mul1, (try group(&.{"mul1"})).codebook);
    try t.expectEqual(Codebook.inst3, (try group(&.{})).codebook);
}

test "an EXL3 group carrying both the mcg and the mul1 marker is refused" {
    try t.expectError(error.TwoCodebookMarkers, group(&.{ "mcg", "mul1" }));
}
