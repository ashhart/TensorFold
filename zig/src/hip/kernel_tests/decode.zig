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

const affine = @import("../launches/affine.zig");
const ref = @import("affine/reference.zig");

/// A random packed product on the host and the device: (n, k) at `bits` and `group`, bf16 tables, `experts` stacked.
const Matrix = struct {
    n: usize,
    k: usize,
    bits: usize,
    group: usize,
    words: []u32,
    scale: []u16,
    bias: []u16,
    dev: [3]@import("../memory.zig").DeviceBuffer,

    fn init(t: *Rig, rng: *rig.Rng, n: usize, k: usize, bits: usize, group_size: usize, experts: usize) !Matrix {
        var m: Matrix = .{ .n = n, .k = k, .bits = bits, .group = group_size, .words = undefined, .scale = undefined, .bias = undefined, .dev = undefined };
        m.words = try gpa.alloc(u32, experts * n * k * bits / 32);
        m.scale = try gpa.alloc(u16, experts * n * k / group_size);
        m.bias = try gpa.alloc(u16, m.scale.len);
        for (m.words) |*w| w.* = @truncate(rng.next() >> 16);
        for (m.scale) |*v| v.* = rig.bf16Bits((rng.unit() + 1.0) / 16.0);
        for (m.bias) |*v| v.* = rig.bf16Bits(rng.unit() / 2.0);
        m.dev = .{ try t.upload(m.words), try t.upload(m.scale), try t.upload(m.bias) };
        return m;
    }

    fn deinit(m: *Matrix) void {
        gpa.free(m.words);
        gpa.free(m.scale);
        gpa.free(m.bias);
        for (&m.dev) |*b| b.free();
    }

    fn problem(m: Matrix, t: *const Rig, x: []const u16) ref.Problem {
        return .{ .fp16 = !t.bf16, .n = m.n, .k = m.k, .bits = m.bits, .group = m.group, .x = x, .words = m.words, .scale = m.scale, .bias = m.bias };
    }

    fn arg(m: Matrix, t: *const Rig, x: u64, rows: usize, out: u64) affine.Arg {
        return .{
            .x = x,
            .words = at(m.dev[0]),
            .scale = .{ .p = at(m.dev[1]), .kind = 1 },
            .bias = .{ .p = at(m.dev[2]), .kind = 1 },
            .out = out,
            .m = @intCast(rows),
            .n = @intCast(m.n),
            .k = @intCast(m.k),
            .bits = @intCast(m.bits),
            .group = @intCast(m.group),
            .fp16 = @intFromBool(!t.bf16),
        };
    }
};

/// Up to four products sharing x in one launch against one launch each: no further from float64 than twice that.
fn group(t: *Rig, rng: *rig.Rng, rows: usize, ns: []const usize, k: usize, bits: usize, group_size: usize) !void {
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    var ms: [4]Matrix = undefined;
    var outs: [2][4]@import("../memory.zig").DeviceBuffer = undefined;
    for (ns, 0..) |n, i| {
        ms[i] = try Matrix.init(t, rng, n, k, bits, group_size, 1);
        for (&outs) |*o| o[i] = try t.alloc(rows * n * 2);
    }
    defer for (ns, 0..) |_, i| {
        ms[i].deinit();
        for (&outs) |*o| o[i].free();
    };
    const kernels = &t.on.affine;
    for (ms[0..ns.len], 0..) |m, i| try kernels.run(&t.r, m.arg(t, at(x), rows, at(outs[0][i])), 4, t.stream.handle, 0, 1, true);
    var sides: [4]affine.Side = undefined;
    for (ms[0..ns.len], 0..) |m, i| sides[i] = .{ .words = at(m.dev[0]), .scale = at(m.dev[1]), .bias = at(m.dev[2]), .n = @intCast(m.n), .out = at(outs[1][i]) };
    try kernels.groupRun(&t.r, ms[0].arg(t, at(x), rows, 0), sides[0..ns.len], true, t.stream.handle);
    try t.stream.synchronize();
    var errs: [2]f64 = .{ 0, 0 };
    for (0..2) |v| for (ns, 0..) |n, i| {
        const got = try rig.download(u16, outs[v][i]);
        defer gpa.free(got);
        const prob = ms[i].problem(t, hx);
        for (0..rows) |r| for (0..@min(n, 12)) |ci| {
            const col = ci * (n - 1) / @max(@min(n, 12) - 1, 1);
            const want = ref.reference(prob, r, 0, col);
            errs[v] = @max(errs[v], @abs(t.value(got[r * n + col]) - want.y) / want.norm);
        };
    };
    try std.testing.expect(errs[1] <= 2 * errs[0] + 1e-9);
}

/// A token's routed gate and up, then the activation, against one launch with the activation as epilogue.
fn pair(t: *Rig, rng: *rig.Rng, rows: usize, width: usize, k: usize, bits: usize, group_size: usize) !void {
    const experts = 64;
    const slots = 9;
    const pairs = rows * slots;
    var m = try Matrix.init(t, rng, 2 * width, k, bits, group_size, experts);
    defer m.deinit();
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    // a token's pairs are its slots; one pair an item
    const pick = try gpa.alloc(u32, pairs);
    defer gpa.free(pick);
    for (pick) |*e| e.* = @intCast(rng.next() % experts);
    const items = try gpa.alloc(i32, pairs * 3);
    defer gpa.free(items);
    const members = try gpa.alloc(i32, pairs);
    defer gpa.free(members);
    for (0..pairs) |i| {
        items[3 * i ..][0..3].* = .{ @intCast(pick[i]), @intCast(i), 1 };
        members[i] = @intCast(i);
    }
    var dev_items = try t.upload(items);
    defer dev_items.free();
    var dev_members = try t.upload(members);
    defer dev_members.free();
    var both = try t.alloc(pairs * 2 * width * 4);
    defer both.free();
    var acts: [2]@import("../memory.zig").DeviceBuffer = .{ try t.alloc(pairs * width * 2), try t.alloc(pairs * width * 2) };
    defer for (&acts) |*a| a.free();
    const kernels = &t.on.affine;
    var a = m.arg(t, at(x), rows, at(both));
    a.route = .{ .items = at(dev_items), .members = at(dev_members), .x_div = slots };
    try kernels.routedWith(&t.r, a, @intCast(pairs), t.stream.handle, .gemm);
    try t.off.tf_moe_act(@ptrFromInt(at(both)), @ptrFromInt(at(acts[0])), t.kind(), @intCast(pairs), @intCast(width), 0, t.stream.handle);
    a.out = 0;
    a.out16 = at(acts[1]);
    try std.testing.expect(try kernels.pairRun(&t.r, a, 0, @intCast(pairs), t.stream.handle));
    try t.stream.synchronize();
    var errs: [2]f64 = .{ 0, 0 };
    const prob = m.problem(t, hx);
    for (0..2) |v| {
        const got = try rig.download(u16, acts[v]);
        defer gpa.free(got);
        for (0..pairs) |pi| for (0..12) |ci| {
            const col = ci * (width - 1) / 11;
            const g = ref.reference(prob, pi / slots, pick[pi], col);
            const u = ref.reference(prob, pi / slots, pick[pi], col + width);
            const want = g.y / (1.0 + @exp(-g.y)) * u.y;
            errs[v] = @max(errs[v], @abs(t.value(got[pi * width + col]) - want) / @max(@abs(want), 0.01));
        };
    }
    try std.testing.expect(errs[1] <= 2 * errs[0] + 1e-9);
}

test "products sharing x in one launch are no further from float64 than twice one launch each" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| try group(t, &rng, rows, &.{ 8192, 4096, 32, 32 }, 2048, 4, 64);
    try group(t, &rng, 1, &.{ 8192, 512, 512 }, 2048, 4, 64);
    try group(t, &rng, 1, &.{ 12288, 12288 }, 4096, 6, 64);
    try group(t, &rng, 4, &.{ 4096, 4096 }, 4096, 8, 128);
    try group(t, &rng, 2, &.{ 1024, 96, 96 }, 2048, 3, 32);
}

test "the routed gate and up with its activation in one launch is no further from float64 than twice the two launches" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4 }) |rows| try pair(t, &rng, rows, 512, 2048, 4, 64);
    try pair(t, &rng, 1, 768, 2048, 6, 64);
    try pair(t, &rng, 1, 512, 2048, 4, 128);
}
