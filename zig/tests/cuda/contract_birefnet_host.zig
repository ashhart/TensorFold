//! The CUDA background remover's contract references on the host: stress sums, ranges and the control registry.
const std = @import("std");
const cm = @import("contract_birefnet_common.zig");
const cv = @import("contract_birefnet_conv.zig");
const nm = @import("contract_birefnet_norm.zig");
const rw = @import("contract_birefnet_rows.zig");
const wn = @import("contract_birefnet_window.zig");

const c = cm.c;
const g = cm.g;
const t_ = std.testing;

/// Gate 4's arithmetic on a filled test `t`: the worst fp32 miss over the three orders and the least fp16 one.
fn sums(t: anytype, spec: c.Spec, shape: c.Shape) ![2]f64 {
    const a = t_.allocator;
    const n = t.count();
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    const allow = try a.alloc(f64, n);
    defer a.free(allow);
    const got = try a.alloc(f64, n);
    defer a.free(got);
    try t.reference(want, mag);
    for (allow, want, mag) |*l, w, m| l.* = if (spec.sums.exact) c.boundExact(spec, w) else c.bound(spec, shape, w, m);
    var worst: f64 = 0;
    var half = std.math.inf(f64);
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |o| {
        try t.simulate(.f32, o, got);
        for (got, want, allow) |x, w, l| worst = @max(worst, if (x == w) 0 else @abs(x - w) / l);
        try t.simulate(.f16, o, got);
        var m: f64 = 0;
        for (got, want, allow) |x, w, l| m = @max(m, if (x == w) 0 else if (std.math.isFinite(x)) @abs(x - w) / l else std.math.inf(f64));
        half = @min(half, m);
    }
    return .{ worst, half };
}

test "tf_br_conv's stress inputs: fp32 sums in any order inside the bound, fp16 sums far outside it" {
    for ([_]usize{ 0, 4 }) |form| {
        var t: cv.Conv = .{ .a = t_.allocator, .rig = undefined, .k = undefined, .form = cv.forms[form], .name = "host" };
        defer t.free();
        const shape: c.Shape = .{ .rows = 4, .n = 147, .k = 3136 };
        try t_.expect(try t.fill(shape, .stress, 7));
        const m = try sums(&t, cv.Conv.spec("host", t.form), shape);
        try t_.expect(m[0] <= 1);
        try t_.expect(m[1] > 100);
    }
}

test "tf_br_norm's stress rows: fp32 sums inside the bound, fp16 sums outside it" {
    var t: nm.Norm = .{ .a = t_.allocator, .rig = undefined, .k = undefined, .name = "host", .form = .rows };
    defer t.free();
    const shape: c.Shape = .{ .rows = 4, .n = 1536, .k = 1536 };
    try t_.expect(try t.fill(shape, .stress, 3));
    const spec = nm.Norm.spec("host", .rows);
    const a = t_.allocator;
    const want = try a.alloc(f64, t.count());
    defer a.free(want);
    const mag = try a.alloc(f64, t.count());
    defer a.free(mag);
    const got = try a.alloc(f64, t.count());
    defer a.free(got);
    try t.reference(want, mag);
    try t_.expectEqual(c.Accum.f32, spec.accum);
    try t.simulate(.f32, .pairwise, got);
    for (got, want, t.allow) |x, w, l| try t_.expect(@abs(x - w) <= l);
    try t.simulate(.f16, .sequential, got);
    var missed = false;
    for (got, want, t.allow) |x, w, l| missed = missed or !(@abs(x - w) <= l);
    try t_.expect(missed);
}

test "tf_br_window's allowance: finite and positive, wider than fp32's own rounding of each output" {
    var t: wn.Window = .{ .a = t_.allocator, .rig = undefined, .k = undefined, .name = "host", .shift = 6 };
    defer {
        t_.allocator.free(t.qkv);
        t_.allocator.free(t.table);
        t_.allocator.free(t.allow);
    }
    try t_.expect(try t.fill(.{ .rows = 1, .n = 192, .k = 144 }, .random, 11));
    const n = t.count();
    const want = try t_.allocator.alloc(f64, n);
    defer t_.allocator.free(want);
    const mag = try t_.allocator.alloc(f64, n);
    defer t_.allocator.free(mag);
    try t.reference(want, mag);
    for (want, t.allow) |w, l| {
        try t_.expect(std.math.isFinite(w) and std.math.isFinite(l));
        try t_.expect(l >= 0x1p-24 * @abs(w));
    }
}

test "tf_br_deform's and tf_br_resize's ranges hold their float64 values" {
    var d: nm.Deform = .{ .a = t_.allocator, .rig = undefined, .k = undefined, .name = "host", .side = 7, .ends = .{ .a = t_.allocator } };
    defer {
        t_.allocator.free(d.x);
        t_.allocator.free(d.offs);
        d.ends.deinit();
    }
    try t_.expect(try d.fill(.{ .rows = 64, .n = 49 * 64, .k = 64 }, .random, 5));
    try holds(&d);
    var r: rw.Resize = .{ .a = t_.allocator, .rig = undefined, .k = undefined, .name = "host", .ends = .{ .a = t_.allocator } };
    defer {
        t_.allocator.free(r.x);
        r.ends.deinit();
    }
    try t_.expect(try r.fill(.{ .rows = 33, .n = 96, .k = 7 }, .random, 9));
    try holds(&r);
}

fn holds(t: anytype) !void {
    const a = t_.allocator;
    const n = t.count();
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    try t.reference(want, mag);
    for (want, t.ends.lo, t.ends.hi) |w, lo, hi| try t_.expect(lo <= w and w <= hi);
}

test "every control breaks a registered kernel through a launcher it swaps, one symbol each" {
    var labels: std.ArrayList([]const u8) = .empty;
    defer labels.deinit(t_.allocator);
    for (cv.forms) |f| try labels.append(t_.allocator, f.label);
    for ([_][]const u8{ "tf_br_conv3", "tf_br_norm", "tf_br_norm map", "tf_br_norm merge", "tf_br_window", "tf_br_window shift 6", "tf_br_resize", "tf_br_tiles", "tf_br_tiles grid", "tf_br_deform 1x1", "tf_br_deform 3x3", "tf_br_deform 7x7", "tf_br_pool", "tf_br_gate", "tf_br_head" }) |l| try labels.append(t_.allocator, l);
    for (cm.controls, 0..) |x, i| {
        var found = false;
        for (labels.items) |l| found = found or std.mem.eql(u8, l, x.label);
        try t_.expect(found);
        var field = false;
        for (cm.ks.names) |n| field = field or std.mem.eql(u8, n, x.field);
        try t_.expect(field);
        for (cm.controls[i + 1 ..]) |y| try t_.expect(!std.mem.eql(u8, x.symbol, y.symbol));
    }
    // 9 conv forms, conv3, 3 norms, 2 windows, resize, 2 tiles, 3 deforms, pool, gate, head
    try t_.expectEqual(@as(usize, 24), labels.items.len);
}

test {
    _ = cm;
}
