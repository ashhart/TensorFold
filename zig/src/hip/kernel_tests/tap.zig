//! A drafter's tap rings: a row's residual words land at (position % ring rows, tap) of its stream's ring.
const std = @import("std");
const rig = @import("rig.zig");
const plan = @import("../launches/plan.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const plan_ops = @import("../ops/plan.zig");
const Tensor = @import("../ops/types.zig").Tensor;
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const hidden = 64;
const taps = 3;
const ring_rows = 8;
const ring_words = ring_rows * taps * hidden / 2;
const untouched: u32 = 0xABABABAB;

/// The ring as the kernel must leave it: `untouched` everywhere but row r's words at (pos[r] % ring_rows, tap).
fn expected(x: []const u32, pos: []const i32, rows: []const usize, tap: usize) ![]u32 {
    const want = try gpa.alloc(u32, ring_words);
    @memset(want, untouched);
    for (rows) |r| {
        const row: usize = @intCast(@mod(pos[r], ring_rows));
        @memcpy(want[(row * taps + tap) * hidden / 2 ..][0 .. hidden / 2], x[r * hidden / 2 ..][0 .. hidden / 2]);
    }
    return want;
}

fn ring(t: *Rig) !DeviceBuffer {
    const fill = try gpa.alloc(u32, ring_words);
    defer gpa.free(fill);
    @memset(fill, untouched);
    return t.upload(fill);
}

test "a round's tap copies each row's residual into its slot's ring at its position, skipping slots with no ring" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x7A9 };
    // slot 0: three rows; slot 1: a ring that wraps (positions 7, 8, 9); slot 2: no ring; slot 3: the scratch slot
    const pos = [_]i32{ 5, 6, 7, 7, 8, 9, 3, 0, 1 };
    const slot = [_]i32{ 0, 0, 0, 1, 1, 1, 2, 3, 3 };
    const n = pos.len;
    const x = try gpa.alloc(u32, n * hidden / 2);
    defer gpa.free(x);
    for (x) |*w| w.* = @truncate(rng.next());
    var dev_x = try t.upload(x);
    defer dev_x.free();
    var rings: [2]DeviceBuffer = .{ try ring(t), try ring(t) };
    defer for (&rings) |*r| r.free();
    // descriptors of a model with one layer: [cached, kept, two layer words, the tap word], no page table read
    const words = 5;
    var descs: [3]DeviceBuffer = undefined;
    for (&descs, 0..) |*d, i| {
        var desc: [words]u64 = @splat(0);
        desc[words - 1] = if (i < 2) at(rings[i]) else 0;
        d.* = try t.upload(&desc);
    }
    defer for (&descs) |*d| d.free();
    const desc_ptrs = [_]u64{ at(descs[0]), at(descs[1]), at(descs[2]), 0 };
    var dev_descs = try t.upload(&desc_ptrs);
    defer dev_descs.free();
    var dev_pos = try t.upload(&pos);
    defer dev_pos.free();
    var dev_slot = try t.upload(&slot);
    defer dev_slot.free();
    const p: plan.PlanRef = .{ .rows = n, .slots = 4, .args = .{ .pos = at(dev_pos), .slot = at(dev_slot), .desc = at(dev_descs), .pages = words } };
    const tap = 1;
    try t.on.planTap(at(dev_x), p, hidden, tap, taps, ring_rows, t.stream.handle);
    try t.stream.synchronize();
    for (rings, [_][]const usize{ &.{ 0, 1, 2 }, &.{ 3, 4, 5 } }) |r, rows| {
        const got = try rig.download(u32, r);
        defer gpa.free(got);
        const want = try expected(x, &pos, rows, tap);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, got);
    }
}

test "a prompt's tap copies its rows into one ring from its first position, wrapping at the ring's rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x7A8 };
    const rows = 6;
    const pos0 = 5;
    const x = try gpa.alloc(u32, rows * hidden / 2);
    defer gpa.free(x);
    for (x) |*w| w.* = @truncate(rng.next());
    var dev_x = try t.upload(x);
    defer dev_x.free();
    var r = try ring(t);
    defer r.free();
    const tap = 2;
    try t.on.tapRows(at(dev_x), at(r), rows, hidden, tap, taps, ring_rows, pos0, t.stream.handle);
    try t.stream.synchronize();
    const got = try rig.download(u32, r);
    defer gpa.free(got);
    var pos: [rows]i32 = undefined;
    for (&pos, 0..) |*q, i| q.* = pos0 + @as(i32, @intCast(i));
    const want = try expected(x, &pos, &.{ 0, 1, 2, 3, 4, 5 }, tap);
    defer gpa.free(want);
    try std.testing.expectEqualSlices(u32, want, got);
}

test "a tap refuses residuals that are not 16-bit before launching" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    const x: Tensor = .{ .ptr = 0, .kind = .f32 };
    const p: plan.PlanRef = .{ .rows = 1, .slots = 1, .args = .{} };
    try std.testing.expectError(error.BadShape, plan_ops.tap(o, x, p, hidden, 0, taps, ring_rows));
    try std.testing.expectError(error.BadShape, plan_ops.tapRows(o, x, 0, 1, hidden, 0, taps, ring_rows, 0));
}
