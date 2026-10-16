//! Hyper-connection packs: a down and inject pair of two widths widens exactly to the wider, as QWeights.of.
const std = @import("std");
const pack = @import("pack.zig");
const mlx = @import("../../core/quant/mlx.zig");
const st = @import("../../core/safetensors.zig");
const affine = @import("affine.zig");
const synthetic = @import("pack_widen_test.zig").synthetic;

fn values(a: std.mem.Allocator, q: pack.Q) ![]f64 {
    const out = try a.alloc(f64, q.rows * q.k);
    try mlx.reference(.{ .words = q.w, .scales = q.s, .biases = q.b, .bits = q.spec.bits, .group = @intCast(q.spec.group) }, out);
    return out;
}

test "oQ4e's 4/64 down with a 5/64 inject packs as one 5/64 matrix holding the same values" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const down = try synthetic(a, 11, 64, 1024, 4, 64);
    const inject = try synthetic(a, 12, 32, 1024, 5, 64);
    const both = try pack.hcDown(a, &.{ down, inject });
    try std.testing.expectEqual(@as(u8, 5), both.spec.bits);
    try std.testing.expectEqual(@as(usize, 64), both.spec.group);
    try std.testing.expectEqual(@as(usize, 96), both.rows);
    const got = try values(a, both);
    try std.testing.expectEqualSlices(f64, try values(a, down), got[0 .. 64 * 1024]);
    try std.testing.expectEqualSlices(f64, try values(a, inject), got[64 * 1024 ..]);
    // one width keeps its bytes: the uniform pair is a plain row concatenation
    const same = try pack.hcDown(a, &.{ down, try synthetic(a, 13, 32, 1024, 4, 64) });
    try std.testing.expectEqualSlices(u8, down.w.bytes, same.w.bytes[0..down.w.bytes.len]);
}

test "the up projection's 320 inputs can't be stored in groups of 128, so it packs at its stored width" {
    const e = struct {
        fn of(dtype: st.DType, rows: usize, cols: usize) st.Entry {
            return .{ .dtype = dtype, .rank = 2, .shape = .{ rows, cols, 0, 0, 0 }, .begin = 0, .end = rows * cols * dtype.size() };
        }
    }.of;
    // 8-bit codes over 320 inputs are 80 words a row; three groups of 128 would need 96
    const at128 = affine.matrix(e(.u32, 64, 80), e(.bf16, 64, 3), e(.bf16, 64, 3), try affine.Spec.init(8, 128));
    try std.testing.expectError(error.InvalidAffineShape, at128);
    try std.testing.expectEqual(@as(usize, 320), (try affine.matrix(e(.u32, 64, 80), e(.bf16, 64, 5), e(.bf16, 64, 5), try affine.Spec.init(8, 64))).k);
}

test "a prompt projection of several widths keeps each member's bytes in order and tells their widths" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var out = pack.Out.init(std.testing.allocator);
    defer out.deinit();
    const q = try synthetic(a, 21, 64, 256, 3, 64);
    const v = try synthetic(a, 22, 32, 256, 8, 128);
    try pack.mlxProjection(&out, std.testing.allocator, a, "L3.att.proj", &.{ q, v });
    const names = out.names.items;
    const at = struct {
        fn find(n: []const []const u8, name: []const u8) usize {
            for (n, 0..) |x, i| if (std.mem.eql(u8, x, name)) return i;
            unreachable;
        }
    }.find;
    const mw = out.blobs.items[at(names, "L3.att.proj.mw")];
    try std.testing.expectEqualSlices(u8, q.w.bytes, mw[0..q.w.bytes.len]);
    try std.testing.expectEqualSlices(u8, v.w.bytes, mw[q.w.bytes.len..]);
    try std.testing.expectEqualSlices(usize, &.{mw.len / 4}, out.shapes.items[at(names, "L3.att.proj.mw")]);
    const parts = out.blobs.items[at(names, "L3.att.proj.parts")];
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u32{ 64, 3, 64, 32, 8, 128 }), parts);
    // 6/32 members write the pinned 2-D shapes and no table
    const six = [_]pack.Q{ try synthetic(a, 23, 32, 256, 6, 32), try synthetic(a, 24, 32, 256, 6, 32) };
    try pack.mlxProjection(&out, std.testing.allocator, a, "L7.att.proj", &six);
    try std.testing.expectEqualSlices(usize, &.{ 64, 48 }, out.shapes.items[at(out.names.items, "L7.att.proj.mw")]);
    for (out.names.items) |n| try std.testing.expect(!std.mem.eql(u8, n, "L7.att.proj.parts"));
}
