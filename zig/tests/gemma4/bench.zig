//! Gemma 4 dense step costs: the 60 layers' projections and head alone by rows, against whole forwards.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const g4 = tf.gemma4;
const cfg = g4.config;
const lane = tf.lane_projection;

fn gpuMs(cb: mtl.CommandBuffer) f64 {
    return cb.gpuSeconds() * 1e3;
}

/// Every projection of the 60 layers and the head at `rows` rows, chained in one command buffer: GPU ms.
fn projections(m: *g4.Model, rows: usize, x: mtl.Buffer, y: mtl.Buffer, reg: bool) !f64 {
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (m.weights.layers) |l| for ([_]g4.weights.Linear{ l.qkv, l.o, l.gate_up, l.down }) |lin| {
        const p = try m.projections.get(m.gpa, m.device, lin.n, lin.k);
        try p.encode(e, .{ .buf = x }, .{ .buf = lin.w }, .{ .buf = lin.sb }, .{ .buf = y }, @intCast(rows));
    };
    const head = try m.projections.get(m.gpa, m.device, m.weights.head.n, m.weights.head.k);
    _ = reg;
    try head.encode(e, .{ .buf = x }, .{ .buf = m.weights.head.w }, .{ .buf = m.weights.head.sb }, .{ .buf = y }, @intCast(rows));
    e.end();
    cb.commit();
    cb.wait();
    return gpuMs(cb);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const m = try g4.Model.load(gpa, io, model_dir);
    defer m.deinit();
    var bytes: usize = 0;
    for (m.weights.layers) |l| for ([_]g4.weights.Linear{ l.qkv, l.o, l.gate_up, l.down }) |lin| {
        bytes += lin.n * lin.k / 2 + lin.n * lin.k / 64 * 4;
    };
    bytes += m.weights.head.n * m.weights.head.k / 2 + m.weights.head.n * m.weights.head.k / 64 * 4;
    const x = try m.device.buffer(16 * cfg.intermediate * 2, mtl.ResourceOptions.private);
    defer x.deinit();
    const y = try m.device.buffer(16 * cfg.vocab * 2, mtl.ResourceOptions.private);
    defer y.deinit();
    std.debug.print("weights read a step: {d:.2} GB\n", .{@as(f64, @floatFromInt(bytes)) / 1e9});
    for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| {
        var best: f64 = 1e9;
        for (0..5) |_| best = @min(best, try projections(m, rows, x, y, false));
        std.debug.print("projections, {d} rows: {d:.2} ms GPU ({d:.0} GB/s)\n", .{ rows, best, @as(f64, @floatFromInt(bytes)) / best / 1e6 });
    }
    var s = try g4.state.Scratch.init(m.device, 4096);
    defer s.deinit();
    var c = try g4.state.Cache.init(m.device, 4096);
    defer c.deinit();
    var ids: [16]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    for ([_]usize{ 0, 1000 }) |depth| {
        c.len = depth;
        for ([_]usize{ 1, 16 }) |rows| {
            var best: f64 = 1e9;
            for (0..5) |_| {
                c.len = depth;
                const t0 = std.c.mach_absolute_time();
                try g4.forward.run(m, &s, &.{.{ .cache = &c, .rows = rows }}, ids[0..rows]);
                best = @min(best, @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e3);
            }
            std.debug.print("whole forward, {d} rows at position {d}: {d:.2} ms wall ({d:.0} tok/s)\n", .{ rows, depth, best, @as(f64, @floatFromInt(rows)) * 1e3 / best });
        }
    }
}
