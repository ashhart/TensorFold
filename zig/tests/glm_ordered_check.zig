//! Model-free pre-M5 arithmetic gate: each dense class against FP64, strides, padding and every cut.
const std = @import("std");
const mtl = @import("metal");
const glm = @import("tensorfold").glm;
const classes = @import("glm_rows_classes.zig");
const a = std.heap.page_allocator;
const opts = mtl.ResourceOptions.shared;
const rows = 2048;
const Ref = glm.affine.Ref;
pub fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
pub fn rb(b: mtl.Buffer) Ref {
    return .{ .buf = b, .off = 64 };
}
pub fn buffer(d: mtl.Device, bytes: usize) !mtl.Buffer {
    const b = try d.buffer(bytes + 128, opts);
    @memset(b.contents()[0 .. bytes + 128], 0xa5);
    return b;
}
pub fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("{s}\n", .{why});
        return error.GpuFailed;
    }
}
fn projection(k: *const glm.kernels.Kernels, d: mtl.Device, queue: mtl.Queue, cl: classes.Class, bits: u8) !void {
    const K: usize = cl.k;
    const N: usize = cl.n;
    const ldx = K + 32;
    const ldy = N + 8;
    const wn = N * K * bits / 8;
    const sn = N * (K / 64);
    const bs = [_]mtl.Buffer{ try buffer(d, wn), try buffer(d, sn * 2), try buffer(d, sn * 2), try buffer(d, rows * ldx * 2), try buffer(d, rows * ldy * 2), try buffer(d, rows * ldy * 2) };
    defer for (bs) |b| b.deinit();
    var rng = std.Random.DefaultPrng.init(743 + @as(u64, bits));
    const random = rng.random();
    random.bytes(rb(bs[0]).addr()[0..wn]);
    for (@as([*]u16, @ptrCast(@alignCast(rb(bs[1]).addr())))[0..sn]) |*v| v.* = bf(0.001 + random.float(f32) * 0.02);
    for (@as([*]u16, @ptrCast(@alignCast(rb(bs[2]).addr())))[0..sn]) |*v| v.* = bf((random.float(f32) - 0.5) * 0.1);
    const raw = try a.alloc(u16, rows * K);
    defer a.free(raw);
    const xs: [*]u16 = @ptrCast(@alignCast(rb(bs[3]).addr()));
    for (0..rows) |r| for (0..K) |j| {
        const v = bf((random.float(f32) - 0.5) * 2);
        xs[r * ldx + j] = v;
        raw[r * K + j] = v;
    };
    const q: glm.affine.Affine = .{ .w = rb(bs[0]), .s = rb(bs[1]), .b = rb(bs[2]), .n = cl.n, .k = cl.k, .bits = bits, .group = 64 };
    @memset(rb(bs[4]).addr()[0 .. rows * ldy * 2], 0xff);
    var cb = queue.commandBuffer();
    var e = cb.compute(.serial);
    _ = glm.ordered_projection.encode(k, e, q, .{ .in = rb(bs[3]), .out = rb(bs[4]), .ldx = @intCast(ldx), .ldo = @intCast(ldy), .rows = rows });
    e.end();
    try finish(cb);
    const want: [*]const u16 = @ptrCast(@alignCast(rb(bs[4]).addr()));
    var tally: classes.Tally = .{};
    for ([_]usize{ 0, rows - 1 }) |r| try classes.against(q, rb(bs[0]).addr()[0..wn], @as([*]const u16, @ptrCast(@alignCast(rb(bs[1]).addr())))[0..sn], @as([*]const u16, @ptrCast(@alignCast(rb(bs[2]).addr())))[0..sn], raw, want[0 .. rows * ldy], ldy, r, &tally, 1);
    for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 33, 2048 }) |cut| {
        @memset(rb(bs[5]).addr()[0 .. rows * ldy * 2], 0xff);
        cb = queue.commandBuffer();
        e = cb.compute(.serial);
        var r: u32 = 0;
        // Representative early and final windows exercise both scalar offsets without a 2,048-launch head check.
        for ([_]u32{ 0, rows - cut }) |at| {
            r = at;
            _ = glm.ordered_projection.encode(k, e, q, .{ .in = rb(bs[3]).at(@as(usize, r) * ldx * 2), .out = rb(bs[5]).at(@as(usize, r) * ldy * 2), .ldx = @intCast(ldx), .ldo = @intCast(ldy), .rows = cut });
        }
        e.end();
        try finish(cb);
        const got: [*]const u16 = @ptrCast(@alignCast(rb(bs[5]).addr()));
        for ([_]usize{ 0, rows - cut }) |at| for (at..at + cut) |row| {
            if (!std.mem.eql(u16, want[row * ldy ..][0..N], got[row * ldy ..][0..N])) return error.ProjectionCutDiffers;
            for (got[row * ldy + N ..][0..8]) |v| if (v != 0xffff) return error.WrotePadding;
        };
    }
    for (bs, [_]usize{ wn, sn * 2, sn * 2, rows * ldx * 2, rows * ldy * 2, rows * ldy * 2 }) |b, len| {
        for (b.contents()[0..64]) |v| if (v != 0xa5) return error.WroteGuard;
        for (b.contents()[64 + len ..][0..64]) |v| if (v != 0xa5) return error.WroteGuard;
    }
    std.debug.print("{s} q{d}: cuts 1..17/33/2048 exact; FP64 {d} values worst err/bound {d:.6}\n", .{ cl.name, bits, tally.values, tally.worst });
}
pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const d = try mtl.Device.init();
    defer d.deinit();
    if (d.tensorUnits()) return error.RequiresPreM5;
    const queue = try d.queue();
    defer queue.deinit();
    const k = try glm.kernels.load(a, d);
    defer {
        k.deinit();
        a.destroy(k);
    }
    try k.order(a, d);
    for (classes.classes) |cl| {
        try projection(k, d, queue, cl, 4);
        for (cl.bits) |bits| if (bits != 4) {
            try projection(k, d, queue, cl, bits);
        };
    }
    for ([_]u8{ 2, 3, 5, 6, 8 }) |bits| try projection(k, d, queue, .{ .name = "narrow scalar boundary", .k = 4096, .n = 64, .bits = &.{} }, bits);
    try @import("glm_ordered_kda_check.zig").check(k, d, queue);
    std.debug.print("GLM ordered arithmetic gate passed; model qualification unchecked\n", .{});
}
