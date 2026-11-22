//! Synthetic serial/parallel range-repair equality and complete-command A-B-B-A timing on Metal.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const repair = @import("contract/metal_lfm3_range_repair.zig");
const g = ct.gates;

fn now() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

fn measure(k: g.Kernel, serial: bool) !f64 {
    const start = now();
    if (serial) _ = try k.vt.replaced.?(k.ctx) else try k.vt.run(k.ctx);
    return @as(f64, @floatFromInt(now() - start)) / 1e9;
}

fn check(k: g.Kernel) !void {
    try k.vt.run(k.ctx);
    const got = try k.vt.output(k.ctx);
    const want = try k.vt.replaced.?(k.ctx);
    if (!std.mem.eql(u8, got, want)) return error.RangeRepairChangedBytes;
}

fn median(values: []f64) f64 {
    for (1..values.len) |i| {
        const value = values[i];
        var at = i;
        while (at > 0 and values[at - 1] > value) : (at -= 1) values[at] = values[at - 1];
        values[at] = value;
    }
    return (values[values.len / 2 - 1] + values[values.len / 2]) / 2;
}

fn samples(values: []const f64) void {
    for (values, 0..) |value, i| std.debug.print("{s}{d:.9}", .{ if (i == 0) "" else ",", value });
}

fn cell(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, history: usize, shared: bool, forced: bool) !void {
    var probe = try repair.Probe.init(a, device, queue, shared, forced, false);
    defer probe.deinit();
    const k = probe.kernel();
    const shape: ct.spec.Shape = .{ .rows = 17, .n = 2048, .k = history };
    if (!try k.vt.prepare(k.ctx, shape, .random, 0x4c464d33 + history)) return error.RepairShapeRefused;
    try check(k);
    try check(k); // one warm A and B before measurement
    var parallel: [8]f64 = undefined;
    var serial: [8]f64 = undefined;
    var pi: usize = 0;
    var si: usize = 0;
    for (0..4) |_| for ([_]bool{ false, true, true, false }) |old| {
        const seconds = try measure(k, old);
        if (old) {
            serial[si] = seconds;
            si += 1;
        } else {
            parallel[pi] = seconds;
            pi += 1;
        }
    };
    try check(k);
    const pm = median(parallel[0..pi]);
    const sm = median(serial[0..si]);
    std.debug.print("{\"kind\":\"LFM3_RANGE_BENCH\",\"history\":{d},\"rows\":17,\"shared\":{},\"forced\":{},\"scope\":\"partial+merge+repair complete-command wall seconds\",\"abba_cycles\":4,\"bf16_bytes_equal\":true,\"parallel_seconds\":[", .{ history, shared, forced });
    samples(parallel[0..pi]);
    std.debug.print("],\"serial_seconds\":[", .{});
    samples(serial[0..si]);
    std.debug.print("],\"parallel_median_seconds\":{d:.9},\"serial_median_seconds\":{d:.9},\"serial_over_parallel\":{d:.6}}\n", .{ pm, sm, sm / pm });
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var forced = true;
    var shared = false;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--unforced")) forced = false else if (std.mem.eql(u8, arg, "--shared")) shared = true else return error.ExpectedOptionalSharedOrUnforced;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]usize{ 2048, 8192, 32768 }) |history| try cell(init.gpa, device, queue, history, shared, forced);
}
