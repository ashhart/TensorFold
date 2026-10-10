//! A tensor-parallel rank's kernels and MTP's token probability against host references.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const ops = @import("../ops/ops.zig");
const Kind = ops.Kind;

fn act(t: *const Rig) Kind {
    return if (t.bf16) .bf16 else .f16;
}

/// The activation's round-to-nearest-even of an fp32 value, as torch rounds.
fn round(t: *const Rig, f: f32) u16 {
    if (!t.bf16) return rig.f16Bits(f);
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return 0x7fc0;
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

test "a residual plus an fp32 sum rounds once to the activation, and the expert-share remap and masks are exact" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x7B };
    const n = 3000;
    const hx = try gpa.alloc(u16, n);
    defer gpa.free(hx);
    const hy = try gpa.alloc(f32, n);
    defer gpa.free(hy);
    for (hx, hy) |*x, *y| {
        x.* = t.bits(rng.unit() * 6.0 - 3.0);
        y.* = rng.unit() * 0.01 - 0.005;
    }
    var x = try t.upload(hx);
    defer x.free();
    var y = try t.upload(hy);
    defer y.free();
    var out = try t.alloc(n * 2);
    defer out.free();
    try o.addWide(.{ .ptr = at(x), .kind = act(t) }, at(y), .{ .ptr = at(out), .kind = act(t) }, n);
    const experts = 16;
    const skip = 8;
    var remap: [experts]i32 = undefined;
    for (&remap, 0..) |*r, e| r.* = if (e % 2 == 0) @intCast(e / 2) else -1;
    var picks: [64]i32 = undefined;
    for (&picks) |*p| p.* = @intCast(rng.next() % experts);
    var dev_remap = try t.upload(&remap);
    defer dev_remap.free();
    var dev_picks = try t.upload(&picks);
    defer dev_picks.free();
    var local = try t.alloc(picks.len * 4);
    defer local.free();
    try o.moeLocalize(at(dev_picks), at(dev_remap), at(local), picks.len, skip);
    var items = [_]i32{ 0, 0, 5, skip, 5, 7, 3, 12, 2, skip, 17, 1 };
    var dev_items = try t.upload(&items);
    defer dev_items.free();
    try o.moeForeignItems(at(dev_items), items.len / 3, skip);
    const d = 33;
    const ys = try gpa.alloc(f32, picks.len * d);
    defer gpa.free(ys);
    for (ys) |*v| v.* = rng.unit() + 1.0;
    var dev_ys = try t.upload(ys);
    defer dev_ys.free();
    var local_picks: [picks.len]i32 = undefined;
    for (&local_picks, picks) |*l, p| l.* = if (remap[@intCast(p)] >= 0) remap[@intCast(p)] else skip;
    var dev_local = try t.upload(&local_picks);
    defer dev_local.free();
    try o.moeZeroForeign(at(dev_ys), at(dev_local), skip, picks.len, d);
    try t.stream.synchronize();
    const got = try rig.download(u16, out);
    defer gpa.free(got);
    for (got, hx, hy) |g, xb, yv| try std.testing.expectEqual(round(t, @as(f32, @floatCast(t.value(xb))) + yv), g);
    const gl = try rig.download(i32, local);
    defer gpa.free(gl);
    try std.testing.expectEqualSlices(i32, &local_picks, gl);
    const gi = try rig.download(i32, dev_items);
    defer gpa.free(gi);
    for (0..items.len / 3) |k| {
        const want: i32 = if (items[3 * k] == skip) 0 else items[3 * k + 2];
        try std.testing.expectEqual(items[3 * k], gi[3 * k]);
        try std.testing.expectEqual(want, gi[3 * k + 2]);
    }
    const gy = try rig.download(f32, dev_ys);
    defer gpa.free(gy);
    for (gy, ys, 0..) |g, v, k| try std.testing.expectEqual(if (local_picks[k / d] == skip) 0 else v, g);
}

test "a token's probability is its softmax share of the row, or the largest one's with no ids, within 1e-6 of float64" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x9E1 };
    const rows = 6;
    const n = 4099;
    const hl = try gpa.alloc(u16, rows * n);
    defer gpa.free(hl);
    for (hl) |*x| x.* = t.bits(rng.unit() * 16.0 - 8.0);
    var ids: [rows]i32 = undefined;
    for (&ids) |*id| id.* = @intCast(rng.next() % n);
    var logits = try t.upload(hl);
    defer logits.free();
    var dev_ids = try t.upload(&ids);
    defer dev_ids.free();
    var with_ids = try t.alloc(rows * 4);
    defer with_ids.free();
    var largest = try t.alloc(rows * 4);
    defer largest.free();
    const l: ops.Tensor = .{ .ptr = at(logits), .kind = act(t) };
    try o.tokenProb(l, rows, n, at(dev_ids), at(with_ids));
    try o.tokenProb(l, rows, n, 0, at(largest));
    try t.stream.synchronize();
    const gw = try rig.download(f32, with_ids);
    defer gpa.free(gw);
    const gm = try rig.download(f32, largest);
    defer gpa.free(gm);
    for (0..rows) |r| {
        const row = hl[r * n ..][0..n];
        var peak: f64 = -std.math.inf(f64);
        for (row) |x| peak = @max(peak, t.value(x));
        var total: f64 = 0;
        for (row) |x| total += @exp(t.value(x) - peak);
        const want = @exp(t.value(row[@intCast(ids[r])]) - peak) / total;
        try std.testing.expectApproxEqAbs(want, gw[r], 1e-6);
        try std.testing.expectApproxEqAbs(1.0 / total, gm[r], 1e-6);
    }
}
