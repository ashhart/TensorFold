//! tf_moe_router against tf_moe_router_tiles on the 35B-A3B router (2,048 inputs, 257 rows), alternating, by row count.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const moe_tests = @import("moe_tests.zig");

const dims = 2048;
const count = 257;
const row_counts = [_]usize{ 32, 33, 128, 255, 256, 257, 258, 512, 1024, 2048, 4096 };
const rounds = 9; // alternating rounds a cell; each times `calls` launches after `calls` warm ones
const calls = 50;

fn launch(f: cuda.Function, tiles: bool, s: cuda.Stream, x: u64, w: u64, out: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_int, dims));
    a.add(w);
    a.add(out);
    for ([_]usize{ rows, dims, count }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(@as(u64, 0));
    const t: usize = if (tiles) 128 else 16;
    const grid: cuda.Dim3 = .{ .x = @intCast((rows + t - 1) / t), .y = @intCast((count + t - 1) / t), .z = 1 };
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 }, .shared = if (tiles) cuda.moe.router_tiles_shared else 0 }, s, &a);
}

fn timeUs(gpu: check.Gpu, f: cuda.Function, tiles: bool, s: cuda.Stream, x: u64, w: u64, out: u64, rows: usize) !f64 {
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (0..calls) |_| try launch(f, tiles, s, x, w, out, rows);
    try start.record(s);
    for (0..calls) |_| try launch(f, tiles, s, x, w, out, rows);
    try end.record(s);
    try end.synchronize();
    return @as(f64, try cuda.Event.elapsedMs(start, end)) * 1000.0 / calls;
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.moe);
    defer mod.unload();
    const f = try cuda.moe.Fns.resolve(mod);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x7e57);
    const r = prng.random();
    const most = row_counts[row_counts.len - 1];
    const xs = try a.alloc(u16, most * dims);
    defer a.free(xs);
    const ws = try a.alloc(u16, count * dims);
    defer a.free(ws);
    for (xs) |*v| v.* = moe_tests.randBf16(r, 1.0);
    for (ws) |*v| v.* = moe_tests.randBf16(r, 0.1);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs));
    defer dx.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ws));
    defer dw.free();
    var dl = try cuda.DeviceBuffer.alloc(gpu.d, most * count * 4);
    defer dl.free();
    std.debug.print("rows,router_us,tiles_us,tiles_over_router,route_takes\n", .{});
    for (row_counts) |rows| {
        var scalar: [rounds]f64 = undefined;
        var tiled: [rounds]f64 = undefined;
        for (0..rounds) |i| {
            scalar[i] = try timeUs(gpu, f.router_fn, false, s, dx.ptr, dw.ptr, dl.ptr, rows);
            tiled[i] = try timeUs(gpu, f.router_tiles_fn, true, s, dx.ptr, dw.ptr, dl.ptr, rows);
        }
        const ms = median(&scalar);
        const mt = median(&tiled);
        const takes = if (cuda.moe.tiled(dx.ptr, dims, dw.ptr, rows, dims)) "tiles" else "router";
        std.debug.print("{d},{d:.2},{d:.2},{d:.3},{s}\n", .{ rows, ms, mt, mt / ms, takes });
    }
}
