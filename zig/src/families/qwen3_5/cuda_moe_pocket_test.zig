//! Gated pocket launch wiring and independent smooth finite differences.
const std = @import("std");
const cuda = @import("cuda");
const to = @import("cuda_train_ops.zig");
const pk = @import("cuda_pocket.zig");
const t = std.testing;
const Rec = struct {
    var count: usize = 0;
    var shared_ptr: u64 = 0;
    var stride: c_int = 0;
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        count += 1;
        shared_ptr = @as(*const u64, @ptrCast(@alignCast(args.?[9].?))).*;
        stride = @as(*const c_int, @ptrCast(@alignCast(args.?[10].?))).*;
        return 0;
    }
};
test "MoE learning: zero pockets launch nothing; dense passes null and MoE passes the shared slot and stride" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Rec.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var slots: [1]?pk.Rows = .{null};
    var p: pk.Pockets = .{ .gpa = t.allocator, .d = &d, .hidden = 64, .layers = &slots };
    Rec.count = 0;
    try p.addGated(stream, 0, 1, 2, 3, 0, 1032, 9);
    try t.expectEqual(@as(usize, 0), Rec.count);
    const b: cuda.DeviceBuffer = .{ .d = &d, .ptr = 64, .len = 256 };
    slots[0] = .{ .gate = b, .up = b, .down = b, .cap = 1, .used = 0 };
    try p.addGated(stream, 0, 1, 2, 3, 0, 1032, 9);
    try t.expectEqual(@as(usize, 0), Rec.count);
    slots[0].?.used = 1;
    p.k = .{ .own = undefined, .rows = .{ .d = &d, .handle = @ptrFromInt(1) } };
    try p.add(stream, 0, 1, 2, 3, 0);
    try t.expectEqual(@as(u64, 0), Rec.shared_ptr);
    try t.expectEqual(@as(c_int, 0), Rec.stride);
    try p.addGated(stream, 0, 1, 2, 3, 0, 1032, 9);
    try t.expectEqual(@as(u64, 1032), Rec.shared_ptr);
    try t.expectEqual(@as(c_int, 9), Rec.stride);
}
fn sig(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}
const gate = [4]f64{ 0.3, -0.4, 0.1, 0.6 };
const up = [4]f64{ -0.2, 0.7, 0.5, -0.3 };
const down = [4]f64{ 0.8, 0.3, -0.1, 0.2 };
const router = [4]f64{ -0.5, 0.1, 0.6, -0.2 };
const push = [4]f64{ 0.2, -0.3, 0.5, 0.4 };
fn dot(a: [4]f64, b: [4]f64) f64 {
    var v: f64 = 0;
    for (a, b) |x, y| v += x * y;
    return v;
}
fn loss(x: [4]f64, scale: f64) f64 {
    const zg = dot(gate, x);
    return scale * sig(dot(router, x)) * zg * sig(zg) * dot(up, x) * dot(down, push);
}
test "MoE learning: pocket input and shared gate adjoints agree with finite differences including sequence scale" {
    var x = [4]f64{ 0.5, -0.7, 0.2, 0.3 };
    const h = 1e-5;
    for ([_]f64{ 0, 0.25, 1 }) |scale| {
        const zg = dot(gate, x);
        const zu = dot(up, x);
        const sg = sig(dot(router, x));
        const sgz = sig(zg);
        const e = dot(down, push);
        const a_shared = scale * e * zg * sgz * zu;
        for (0..4) |i| {
            const direct = scale * sg * e * (sgz * (1 + zg * (1 - sgz)) * zu * gate[i] + zg * sgz * up[i]);
            const route = a_shared * sg * (1 - sg) * router[i];
            const v = x[i];
            x[i] = v + h;
            const hi = loss(x, scale);
            x[i] = v - h;
            const lo = loss(x, scale);
            x[i] = v;
            const fd = (hi - lo) / (2 * h);
            try t.expectApproxEqAbs(fd, direct + route, 1e-10);
            if (scale > 0) try t.expect(@abs(fd - direct) > 1e-7);
        }
    }
}
