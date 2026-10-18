//! Shared checkpoint admission and host images. Backend resources belong in adapters.
pub const Pack = @import("pack.zig").Pack;
pub const Info = @import("pack.zig").Info;
pub const plan = @import("plan.zig");
pub const stage = @import("stage.zig");
pub const named = @import("named.zig");
pub const prepared = @import("prepared.zig");
const std = @import("std");
/// Strict production checkpoint admission. Pack itself also supports partial header inspection.
pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Pack {
    var p = try Pack.open(a, io, dir);
    errdefer p.deinit();
    if (p.missing != 0) return error.CheckpointIndexMismatch;
    var owners = p.owner.iterator();
    while (owners.next()) |e| {
        const info = p.get(e.key_ptr.*) orelse return error.CheckpointIndexMismatch;
        if (!std.mem.eql(u8, p.files.items[info.file], e.value_ptr.*)) return error.CheckpointIndexMismatch;
    }
    return p;
}
pub const Projection = Pack.Projection;
pub const projection = @import("projection.zig").read;

test "shared projection requires dense rotation scales and refuses half-bit groups" {
    var p = Pack.init(std.testing.allocator);
    defer p.deinit();
    try p.addHeader("dense", "{\"p.weight\":{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[0,32768]}}", 32768);
    try std.testing.expectError(error.MissingTensor, projection(&p, "p"));
    try std.testing.expectError(error.BadTrellis, @import("core").exl3_layout.Group.fromShape(&.{ 8, 8, 8 }));
}
test "shared projection admits packed sign owners and gives dense siblings precedence" {
    const a = std.testing.allocator;
    const image = try @import("pack.zig").image(a, &.{
        .{ .name = "p.trellis", .dtype = "I16", .shape = &.{ 8, 8, 32 } },
        .{ .name = "p.su", .dtype = "I16", .shape = &.{8} },
        .{ .name = "p.sv", .dtype = "I16", .shape = &.{8} },
    });
    defer a.free(image);
    var p = Pack.init(a);
    defer p.deinit();
    try p.addHeaderDump("packed", image);
    var packed_plan = try @import("model.zig").Plan.init(a, &p, &.{"p"});
    defer packed_plan.deinit();
    const packed_storage = packed_plan.projections[0].storage;
    try std.testing.expect(packed_storage == .exl3);
    try std.testing.expectEqual(p.get("p.su").?, packed_storage.exl3.scales.input);
    try std.testing.expectEqual(p.get("p.sv").?, packed_storage.exl3.scales.output);
    try std.testing.expectError(error.UnsupportedPackedScales, packed_storage.cudaGroup());
    const dense = try @import("pack.zig").image(a, &.{
        .{ .name = "p.weight", .dtype = "BF16", .shape = &.{ 128, 256 } },
        .{ .name = "p.suh", .dtype = "F16", .shape = &.{256} },
        .{ .name = "p.svh", .dtype = "F16", .shape = &.{128} },
    });
    defer a.free(dense);
    var both = Pack.init(a);
    defer both.deinit();
    try both.addHeaderDump("dense", dense);
    const trellis = try @import("pack.zig").image(a, &.{.{ .name = "p.trellis", .dtype = "I16", .shape = &.{ 16, 8, 32 } }});
    defer a.free(trellis);
    try both.addHeaderDump("trellis", trellis);
    var dense_plan = try @import("model.zig").Plan.init(a, &both, &.{"p"});
    defer dense_plan.deinit();
    const dense_storage = dense_plan.projections[0].storage;
    try std.testing.expect(dense_storage == .dense);
    try std.testing.expectEqual(@as(usize, 256), dense_storage.dense.k);
    try std.testing.expectEqual(@as(usize, 128), dense_storage.dense.n);
    try std.testing.expectEqual(both.get("p.weight").?, dense_storage.dense.weight);
    try std.testing.expectEqual(both.get("p.suh").?, dense_storage.dense.scales.input);
    try std.testing.expectEqual(both.get("p.svh").?, dense_storage.dense.scales.output);
    try std.testing.expectError(error.UnsupportedDenseProjection, dense_storage.cudaGroup());
}
test "odd half-widths require a mul1 marker and scale pairs cannot be ambiguous" {
    const a = std.testing.allocator;
    const data = try @import("pack.zig").image(a, &.{
        .{ .name = "p.trellis", .dtype = "I16", .shape = &.{ 8, 8, 24 } },
        .{ .name = "p.suh", .dtype = "F16", .shape = &.{128} },
        .{ .name = "p.svh", .dtype = "F16", .shape = &.{128} },
    });
    defer a.free(data);
    var p = Pack.init(a);
    defer p.deinit();
    try p.addHeaderDump("odd", data);
    try std.testing.expectError(error.BadMarker, p.codebook(std.testing.io, "p"));
    try std.testing.expectError(error.BadMarker, projection(&p, "p"));
    const signs = try @import("pack.zig").image(a, &.{.{ .name = "p.su", .dtype = "I16", .shape = &.{8} }});
    defer a.free(signs);
    try p.addHeaderDump("signs", signs);
    try std.testing.expectError(error.AmbiguousScales, projection(&p, "p"));
}
