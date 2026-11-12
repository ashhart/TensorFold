//! The decode tail's exact output bits (the new residual and its norm), pinned by a digest per architecture.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const kernels = @import("../kernels.zig");

/// The digest of every case's residual and norm, in both activation kinds, on an architecture whose bits were recorded.
const pins = [_]struct { arch: []const u8, digest: u64 }{
    .{ .arch = "gfx1151", .digest = 0x325db42cb918dd99 },
};

/// `rows` rows of `width` plus a plain row (`slots` 0) or the slots' weighted sum; `cancel` makes the slot order show.
const Case = struct { rows: usize, width: usize, slots: usize, weight: bool = true, cancel: bool = false };

/// Cancelling slots (past a slot group too), one row and a window at the 35B's and 27B's widths, ragged, unweighted.
const cases = [_]Case{
    .{ .rows = 2, .width = 2048, .slots = 3, .cancel = true },
    .{ .rows = 2, .width = 2048, .slots = 9, .cancel = true },
    .{ .rows = 1, .width = 2048, .slots = 0 },
    .{ .rows = 1, .width = 2048, .slots = 9 },
    .{ .rows = 4, .width = 2048, .slots = 9 },
    .{ .rows = 1, .width = 5120, .slots = 0 },
    .{ .rows = 3, .width = 5120, .slots = 0 },
    .{ .rows = 2, .width = 1000, .slots = 3 },
    .{ .rows = 2, .width = 2048, .slots = 0, .weight = false },
    .{ .rows = 1, .width = 8192, .slots = 2 },
};

/// FNV-1a over `bytes`, continuing from `h`.
fn fnv(h: u64, bytes: []const u8) u64 {
    var x = h;
    for (bytes) |b| {
        x ^= b;
        x *%= 0x100000001b3;
    }
    return x;
}

/// One case's residual and norm through tf_tail at activation kind `kind` (1 fp16, 2 bf16), folded into `h`.
fn run(t: *Rig, rng: *rig.Rng, c: Case, kind: c_int, h: u64) !u64 {
    const bits = if (kind == 1) &rig.f16Bits else &rig.bf16Bits;
    const n = c.rows * c.width;
    const hx = try gpa.alloc(u16, n);
    defer gpa.free(hx);
    for (hx) |*v| v.* = bits(rng.unit());
    const hy16 = try gpa.alloc(u16, if (c.slots > 0) 0 else n);
    defer gpa.free(hy16);
    for (hy16) |*v| v.* = bits(rng.unit() / 8.0);
    const hy32 = try gpa.alloc(f32, c.rows * c.slots * c.width);
    defer gpa.free(hy32);
    for (hy32, 0..) |*v, i| {
        const slot = (i / c.width) % @max(c.slots, 1);
        const big: f32 = if (!c.cancel or slot > 1) 0 else if (slot == 0) 4096 else -4096;
        v.* = big + rng.unit() / 8.0;
    }
    const hw = try gpa.alloc(f32, c.rows * @max(c.slots, 1));
    defer gpa.free(hw);
    for (hw) |*v| v.* = if (c.cancel) 0.5 else (rng.unit() + 1.0) / @as(f32, @floatFromInt(@max(c.slots, 1)));
    const hn = try gpa.alloc(f32, c.width);
    defer gpa.free(hn);
    for (hn) |*v| v.* = 1.0 + rng.unit() / 4.0;
    var x = try t.upload(hx);
    defer x.free();
    var y = if (c.slots > 0) try t.upload(hy32) else try t.upload(hy16);
    defer y.free();
    var wts = try t.upload(hw);
    defer wts.free();
    var weight = try t.upload(hn);
    defer weight.free();
    var normed = try t.alloc(n * 2);
    defer normed.free();
    const wp: ?[*]const f32 = if (c.slots > 0) @ptrFromInt(at(wts)) else null;
    const np: ?[*]const f32 = if (c.weight) @ptrFromInt(at(weight)) else null;
    try t.on.tf_tail(@ptrFromInt(at(x)), @ptrFromInt(at(y)), wp, np, @ptrFromInt(at(normed)), kind, @intCast(c.rows), @intCast(c.slots), @intCast(c.width), 1e-6, t.stream.handle);
    try t.stream.synchronize();
    const gx = try rig.download(u16, x);
    defer gpa.free(gx);
    const gn = try rig.download(u16, normed);
    defer gpa.free(gn);
    return fnv(fnv(h, std.mem.sliceAsBytes(gx)), std.mem.sliceAsBytes(gn));
}

test "the decode tail writes its pinned residual and norm bits in fp16 and bf16, plain, slot-summed, ragged and wide" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    var h: u64 = 0xcbf29ce484222325;
    for ([_]c_int{ 2, 1 }) |kind| {
        for (cases) |c| h = try run(t, &rng, c, kind, h);
    }
    const want = for (pins) |p| {
        if (std.mem.eql(u8, p.arch, kernels.arch)) break p.digest;
    } else {
        std.debug.print("decode tail: no digest pinned for {s}; this run wrote 0x{x:0>16}\n", .{ kernels.arch, h });
        return error.SkipZigTest;
    };
    if (h != want) std.debug.print("decode tail digest on {s}: 0x{x:0>16}\n", .{ kernels.arch, h });
    try std.testing.expectEqual(want, h);
}
