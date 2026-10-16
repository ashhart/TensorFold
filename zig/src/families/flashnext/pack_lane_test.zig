//! A fused stack packs one product per bit width, members of equal bits together wherever they sit (`_stacked`).
const std = @import("std");
const affine = @import("affine.zig");
const lane = @import("pack_lane.zig");
const pio = @import("pack_io.zig");
const synthetic = @import("pack_widen_test.zig").synthetic;

fn spec(bits: u8, group: usize) affine.Spec {
    return affine.Spec.init(bits, group) catch unreachable;
}

const att = [_]usize{ 12288, 512, 512, 640 }; // row counts of q, k, v and index_qk
const gdn = [_]usize{ 10240, 6144, 48, 48 }; // row counts of in_proj_qkv, z, b and a

fn expectRuns(specs: []const affine.Spec, rows: []const usize, want: []const lane.Run) !void {
    var buf: [8]lane.Run = undefined;
    const runs = try lane.runsOf(specs, rows[0..specs.len], &buf);
    try std.testing.expectEqual(want.len, runs.len);
    for (want, runs) |w, r| try std.testing.expectEqual(w, r);
}

test "members of equal bits form one run at their smallest group, in first-seen order, adjacent or not" {
    // oQ2/oQ3/oQ4e attention: an 8-bit v between 6-bit q, k and index_qk; 0.6.6 runs q, k and index_qk as one product
    try expectRuns(&.{ spec(6, 64), spec(6, 64), spec(8, 64), spec(6, 64) }, &att, &.{
        .{ .bits = 6, .group = 64, .members = 0b1011, .rows = 12288 + 512 + 640 },
        .{ .bits = 8, .group = 64, .members = 0b0100, .rows = 512 },
    });
    // a 6/128 v among 6/64 members is regrouped into the one 6/64 product
    try expectRuns(&.{ spec(6, 64), spec(6, 64), spec(6, 128), spec(6, 64) }, &att, &.{
        .{ .bits = 6, .group = 64, .members = 0b1111, .rows = 13952 },
    });
    // oQ4e layer 21's GDN inputs: qkv 4/64, z and b 5/128, a 6/64; the 5-bit run is read in groups of 64 on the lane
    try expectRuns(&.{ spec(4, 64), spec(5, 128), spec(5, 128), spec(6, 64) }, &gdn, &.{
        .{ .bits = 4, .group = 64, .members = 0b0001, .rows = 10240 },
        .{ .bits = 5, .group = 64, .members = 0b0110, .rows = 6192 },
        .{ .bits = 6, .group = 64, .members = 0b1000, .rows = 48 },
    });
    // equal bits in groups of 32 and 64 share the group of 32, so the run's group sums are 0.6.6's
    try expectRuns(&.{ spec(4, 64), spec(4, 32) }, &att, &.{.{ .bits = 4, .group = 32, .members = 0b11, .rows = 12800 }});
    try expectRuns(&.{ spec(6, 32), spec(6, 32), spec(6, 32), spec(6, 32) }, &att, &.{
        .{ .bits = 6, .group = 32, .members = 0b1111, .rows = 13952 },
    });
}

fn find(out: *const pio.Out, name: []const u8) ?usize {
    for (out.names.items, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
    return null;
}

test "a stack of several widths writes a pack per run with its rows' places, and a regrouped one keeps the plain names" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var out = pio.Out.init(gpa);
    defer out.deinit();
    const q = try synthetic(a, 1, 32, 256, 6, 64);
    const v = try synthetic(a, 2, 32, 256, 8, 64);
    const idx = try synthetic(a, 3, 32, 256, 6, 64);
    try lane.putLane(&out, gpa, "L1.att.proj", &.{ q, v, idx });
    try std.testing.expect(find(&out, "L1.att.proj.wq") == null);
    try std.testing.expectEqualSlices(usize, &.{ 64, 48 }, out.shapes.items[find(&out, "L1.att.proj.r0.wq").?]);
    try std.testing.expectEqualSlices(usize, &.{ 32, 64 }, out.shapes.items[find(&out, "L1.att.proj.r1.wq").?]);
    // index_qk's first word follows q's 32 rows in the 6-bit run: the second tile's first word
    const r0 = out.blobs.items[find(&out, "L1.att.proj.r0.wq").?];
    try std.testing.expectEqualSlices(u8, idx.w.bytes[0..4], r0[32 * 48 * 4 ..][0..4]);
    const runs = out.blobs.items[find(&out, "L1.att.proj.runs").?]; // (bits, group, rows) per run
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u32{ 6, 64, 64, 8, 64, 32 }), runs);
    const places = out.blobs.items[find(&out, "L1.att.proj.members").?]; // (run, row in run, row in fused output)
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u32{ 0, 0, 0, 1, 0, 32, 0, 32, 64 }), places);
    const one = [_]lane.Q{ try synthetic(a, 4, 32, 256, 6, 64), try synthetic(a, 5, 32, 256, 6, 128) };
    try lane.putLane(&out, gpa, "L3.att.proj", &one);
    try std.testing.expectEqualSlices(usize, &.{ 4, 64, 2 }, out.shapes.items[find(&out, "L3.att.proj.sbt").?]);
    const regrouped = out.blobs.items[find(&out, "L3.att.proj.runs").?]; // one run, its width told
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u32{ 6, 64, 64 }), regrouped);
    try std.testing.expect(find(&out, "L3.att.proj.members") == null);
    const plain = [_]lane.Q{ try synthetic(a, 6, 32, 256, 6, 32), try synthetic(a, 7, 32, 256, 6, 32) };
    try lane.putLane(&out, gpa, "L7.att.proj", &plain);
    try std.testing.expect(find(&out, "L7.att.proj.runs") == null); // 6/32 writes no table: its bytes are pinned
    const four = [_]lane.Q{try synthetic(a, 8, 64, 256, 4, 64)};
    try lane.putLane(&out, gpa, "L9.att.o", &four);
    try std.testing.expectEqualSlices(usize, &.{ 64, 32 }, out.shapes.items[find(&out, "L9.att.o.wq").?]);
    // 64 rows of 4 bits tile 32 columns at a time, as the lane launches read them: the second tile follows the first
    const wq = out.blobs.items[find(&out, "L9.att.o.wq").?];
    try std.testing.expectEqualSlices(u8, four[0].w.bytes[32 * 32 * 4 ..][0..4], wq[4 * 32 * 8 * 4 ..][0..4]);
}

test "runs whose columns aren't a multiple of 32 keep MLX's layout untiled, as decode.py's nt 0, at the GDN rows" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var out = pio.Out.init(gpa);
    defer out.deinit();
    const widths = [_][2]usize{ .{ 4, 64 }, .{ 5, 128 }, .{ 5, 128 }, .{ 6, 64 } };
    var ms: [4]lane.Q = undefined;
    for (&ms, gdn, widths, 0..) |*m, rows, w, i| m.* = try synthetic(a, 20 + i, rows, 256, @intCast(w[0]), w[1]);
    try lane.putLane(&out, gpa, "L21.gdn.in", &ms);
    const runs = out.blobs.items[find(&out, "L21.gdn.in.runs").?];
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u32{ 4, 64, 10240, 5, 64, 6192, 6, 64, 48 }), runs);
    try std.testing.expect(lane.tiled(10240) and !lane.tiled(6192) and !lane.tiled(48));
    // the ragged 5-bit run is z then b row by row, and the 48-row a is its own bytes
    const r1 = out.blobs.items[find(&out, "L21.gdn.in.r1.wq").?];
    try std.testing.expectEqualSlices(u8, ms[1].w.bytes, r1[0..ms[1].w.bytes.len]);
    try std.testing.expectEqualSlices(u8, ms[2].w.bytes, r1[ms[1].w.bytes.len..]);
    try std.testing.expectEqualSlices(u8, ms[3].w.bytes, out.blobs.items[find(&out, "L21.gdn.in.r2.wq").?]);
    const places = out.blobs.items[find(&out, "L21.gdn.in.members").?];
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u32{ 0, 0, 0, 1, 0, 10240, 1, 6144, 16384, 2, 0, 16432 }), places);
    // a 6/32 lane whose rows don't tile carries its table, so it is never read as a tiled uniform lane
    const odd = [_]lane.Q{try synthetic(a, 30, 48, 256, 6, 32)};
    try lane.putLane(&out, gpa, "L0.gdn.out", &odd);
    try std.testing.expect(find(&out, "L0.gdn.out.runs") != null);
    // columns not in fours have no lane layout: refused by name
    const bad = [_]lane.Q{try synthetic(a, 31, 50, 256, 6, 32)};
    try std.testing.expectError(error.LaneRowsNotFours, lane.putLane(&out, gpa, "L0.att.o", &bad));
}

fn packMixed(gpa: std.mem.Allocator, stack: []const lane.Q, proj: []const lane.Q) !void {
    var out = pio.Out.init(gpa);
    defer out.deinit();
    try lane.putLane(&out, gpa, "L21.gdn.in", stack);
    try @import("pack.zig").mlxProjection(&out, gpa, out.arena.allocator(), "L21.gdn.in", proj);
}

test "a lane stack and a prompt projection of several widths free every buffer when any allocation fails" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // oQ4e's layer-21 GDN input in miniature: 4/64, 5/128 twice, 6/64, with a ragged 48-row 6-bit run
    const l21 = [_]lane.Q{ try synthetic(a, 31, 64, 256, 4, 64), try synthetic(a, 32, 64, 256, 5, 128), try synthetic(a, 33, 48, 256, 5, 128), try synthetic(a, 34, 48, 256, 6, 64) };
    // a fixed buffer under the failing allocator: every run grows and resizes the same way at every optimize mode
    const backing = try std.testing.allocator.alloc(u8, 64 << 20);
    defer std.testing.allocator.free(backing);
    var fixed: std.heap.FixedBufferAllocator = .init(backing);
    try std.testing.checkAllAllocationFailures(fixed.allocator(), packMixed, .{ &l21, &l21 });
}
