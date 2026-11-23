//! tf_moe_combine_gate against tf_moe_combine: the shared expert as y's last slot, and separate, byte for byte.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

/// A device address as the read-only fp32 pointer the combine launchers take.
fn f(b: anytype) ?[*]const f32 {
    return @ptrFromInt(at(b));
}

/// A device address as the kernel's output pointer.
fn p(b: anytype) ?*anyopaque {
    return @ptrFromInt(at(b));
}

/// `kind`'s element size: fp32 at four bytes, fp16 and bf16 at two.
fn size(kind: c_int) usize {
    return if (kind == 0) 4 else 2;
}

/// The expert id a tensor-parallel rank gives the pairs another rank holds (tf_moe_zero_foreign's skip).
const foreign = 7;

/// Both kernels over `slots` - 1 routed rows and a shared row; with `remap` some pairs pick `foreign` and read zero.
fn equal(t: *Rig, rng: *rig.Rng, rows: usize, slots: usize, d: usize, remap: bool) !void {
    const routed = slots - 1;
    const share = try gpa.alloc(f32, rows * d); // the shared expert's row, `wts`' last slot weighting it
    defer gpa.free(share);
    for (share) |*v| v.* = @floatCast(rng.unit() * 3 - 1.5);
    const hy = try gpa.alloc(f32, rows * slots * d);
    defer gpa.free(hy);
    for (0..rows) |r| for (0..d) |c| {
        for (0..routed) |s| hy[(r * slots + s) * d + c] = @floatCast((rng.unit() + rng.unit()) * 2 - 2);
        // the slot the gate must never read from y: poison, so a gate summing every slot could not match
        hy[(r * slots + slots - 1) * d + c] = 1e30;
    };
    const hw = try gpa.alloc(f32, rows * slots);
    defer gpa.free(hw);
    for (hw) |*v| v.* = @floatCast(0.5 + rng.unit() / 2);
    const picks = try gpa.alloc(i32, rows * slots);
    defer gpa.free(picks);
    for (picks, 0..) |*v, i| {
        const last = i % slots == slots - 1;
        v.* = if (remap and (last or rng.next() % 3 == 0)) foreign else @intCast(rng.next() % foreign);
    }
    // the combine's y: y's routed slots (zero where another rank holds the pair), the shared expert in the last slot
    const stack = try gpa.alloc(f32, rows * slots * d);
    defer gpa.free(stack);
    for (0..rows) |r| for (0..d) |c| {
        for (0..routed) |s| {
            const held = !remap or picks[r * slots + s] != foreign;
            stack[(r * slots + s) * d + c] = if (held) hy[(r * slots + s) * d + c] else 0;
        }
        stack[(r * slots + routed) * d + c] = share[r * d + c];
    };
    if (remap) for (0..rows) |r| for (0..routed) |s| {
        // another rank's pair: poison too, so a gate that read it could not match
        if (picks[r * slots + s] == foreign) for (0..d) |c| {
            hy[(r * slots + s) * d + c] = 1e30;
        };
    };
    var y = try t.upload(hy); // the gate's y: the shared row's slot is poison, so the gate must skip it
    defer y.free();
    var stacked = try t.upload(stack);
    defer stacked.free();
    var shared = try t.upload(share);
    defer shared.free();
    var wts = try t.upload(hw);
    defer wts.free();
    var dev_picks = try t.upload(picks);
    defer dev_picks.free();
    for ([_]c_int{ 0, 1, 2 }) |kind| {
        var a = try t.alloc(rows * d * size(kind));
        defer a.free();
        var b = try t.alloc(rows * d * size(kind));
        defer b.free();
        const r: c_int = @intCast(rows);
        const s: c_int = @intCast(slots);
        const w: c_int = @intCast(d);
        try t.on.tf_moe_combine(f(stacked), f(wts), p(a), kind, r, s, w, t.stream.handle);
        const by: ?[*]const c_int = if (remap) @ptrFromInt(at(dev_picks)) else null;
        try t.on.tf_moe_combine_gate(f(y), f(wts), f(shared), p(b), kind, r, s, w, by, if (remap) foreign else 0, t.stream.handle);
        try t.stream.synchronize();
        const ga = try rig.download(u8, a);
        defer gpa.free(ga);
        const gb = try rig.download(u8, b);
        defer gpa.free(gb);
        try std.testing.expectEqualSlices(u8, ga, gb);
    }
}

test "tf_moe_combine_gate equals tf_moe_combine with the shared expert as y's last slot, another rank's pairs zeroed" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5ECA1E };
    for ([_]bool{ false, true }) |remap| for ([_]usize{ 1, 3, 16 }) |rows| {
        try equal(t, &rng, rows, 4, 2048, remap);
        try equal(t, &rng, 2, 3, 511, remap); // a width that does not fill a block's columns
        try equal(t, &rng, 1, 2, 64, remap); // one routed slot beside the shared expert
        try equal(t, &rng, rows, 11, 2560, remap); // Flash Next: ten routed slots and the shared expert
    };
}
