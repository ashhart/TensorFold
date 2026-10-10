//! The merged decode-step launches against the launches they replace, each against a float64 reference.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const Launcher = @import("../launches.zig").Launcher;

fn p(b: anytype) ?*anyopaque {
    return @ptrFromInt(at(b));
}

/// The router's logits for `rows` rows over (experts, d) fp32 weights: the merged launch no further from float64.
fn router(t: *Rig, rng: *rig.Rng, rows: usize, d: usize, experts: usize) !void {
    const hx = try gpa.alloc(u16, rows * d);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    const hw = try gpa.alloc(f32, experts * d);
    defer gpa.free(hw);
    for (hw) |*v| v.* = rng.unit() / 8.0;
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    var max_err: [2]f64 = .{ 0, 0 };
    for ([2]*const Launcher{ &t.off, &t.on }, 0..) |l, v| {
        var out = try t.alloc(rows * experts * 4);
        defer out.free();
        try l.tf_moe_router(p(x), t.kind(), @ptrFromInt(at(w)), @ptrFromInt(at(out)), @intCast(rows), @intCast(d), @intCast(experts), t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(f32, out);
        defer gpa.free(got);
        for (0..rows) |r| for (0..experts) |e| {
            var y: f64 = 0;
            var norm: f64 = 0;
            for (0..d) |i| {
                const term = t.value(hx[r * d + i]) * @as(f64, hw[e * d + i]);
                y += term;
                norm += @abs(term);
            }
            max_err[v] = @max(max_err[v], @abs(@as(f64, got[r * experts + e]) - y) / norm);
        };
    }
    try std.testing.expect(max_err[1] <= 2 * max_err[0] + 1e-9);
}

/// x = x + y (or the slots' weighted sum) and the next norm: one merged launch against combine, add and norm.
fn tails(t: *Rig, rng: *rig.Rng, rows: usize, width: usize, slots: usize) !void {
    const n = rows * width;
    const eps: f32 = 1e-6;
    const hx = try gpa.alloc(u16, n);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    const hy16 = try gpa.alloc(u16, if (slots > 0) 0 else n);
    defer gpa.free(hy16);
    const hy32 = try gpa.alloc(f32, if (slots > 0) rows * slots * width else 0);
    defer gpa.free(hy32);
    for (hy16) |*v| v.* = t.bits(rng.unit() / 8.0);
    for (hy32) |*v| v.* = rng.unit() / 8.0;
    const hw = try gpa.alloc(f32, rows * @max(slots, 1));
    defer gpa.free(hw);
    for (hw) |*v| v.* = (rng.unit() + 1.0) / @as(f32, @floatFromInt(@max(slots, 1)));
    const hn = try gpa.alloc(f32, width);
    defer gpa.free(hn);
    for (hn) |*v| v.* = 1.0 + rng.unit() / 4.0;
    var y = if (slots > 0) try t.upload(hy32) else try t.upload(hy16);
    defer y.free();
    var wts = try t.upload(hw);
    defer wts.free();
    var weight = try t.upload(hn);
    defer weight.free();
    var tmp = try t.alloc(n * 2);
    defer tmp.free();
    var errs: [2][2]f64 = .{ .{ 0, 0 }, .{ 0, 0 } };
    for (0..2) |v| {
        var x = try t.upload(hx);
        defer x.free();
        var normed = try t.alloc(n * 2);
        defer normed.free();
        const s = t.stream.handle;
        if (v == 0) {
            const sum = if (slots > 0) p(tmp) else p(y);
            if (slots > 0) try t.off.tf_moe_combine(@ptrFromInt(at(y)), @ptrFromInt(at(wts)), sum, t.kind(), @intCast(rows), @intCast(slots), @intCast(width), s);
            try t.off.tf_add(p(x), sum, p(x), t.kind(), @intCast(n), s);
            try t.off.tf_rms(p(x), @ptrFromInt(at(weight)), p(normed), t.kind(), @intCast(rows), @intCast(width), eps, s);
        } else {
            try t.on.tf_tail(p(x), p(y), if (slots > 0) @ptrFromInt(at(wts)) else null, @ptrFromInt(at(weight)), p(normed), t.kind(), @intCast(rows), @intCast(slots), @intCast(width), eps, s);
        }
        try t.stream.synchronize();
        const gx = try rig.download(u16, x);
        defer gpa.free(gx);
        const gn = try rig.download(u16, normed);
        defer gpa.free(gn);
        const want = try gpa.alloc(f64, width);
        defer gpa.free(want);
        for (0..rows) |r| {
            var sq: f64 = 0;
            var top: f64 = 0;
            for (0..width) |c| {
                var add: f64 = 0;
                if (slots > 0) {
                    for (0..slots) |sl| add += @as(f64, hy32[(r * slots + sl) * width + c]) * @as(f64, hw[r * slots + sl]);
                } else add = t.value(hy16[r * width + c]);
                want[c] = t.value(hx[r * width + c]) + add;
                sq += want[c] * want[c];
                top = @max(top, @abs(want[c]));
            }
            const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(width)) + eps);
            for (0..width) |c| {
                errs[v][0] = @max(errs[v][0], @abs(t.value(gx[r * width + c]) - want[c]) / top);
                errs[v][1] = @max(errs[v][1], @abs(t.value(gn[r * width + c]) - want[c] * inv * hn[c]) / (top * inv * 1.25));
            }
        }
    }
    try std.testing.expect(errs[1][0] <= 2 * errs[0][0] + 1e-7);
    try std.testing.expect(errs[1][1] <= 2 * errs[0][1] + 1e-7);
}

test "the merged router launch is no further from float64 than twice the launch it replaces" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4, 8 }) |rows| try router(t, &rng, rows, 2048, 257);
}

test "the merged residual-and-norm tails are no further from float64 than twice the launches they replace" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 4, 16 }) |rows| {
        try tails(t, &rng, rows, 2048, 0);
        try tails(t, &rng, rows, 2048, 9);
    }
    try tails(t, &rng, 1, 4096, 0);
}
