//! Gemma 4 E-series per-layer inputs on CUDA (#1222): tf_gemma4_ple_rows bit-equal to layer_input.zig's host steps.
const std = @import("std");
const cuda = @import("cuda");
const g4 = @import("gemma4_cuda");
const check = @import("check.zig");

const li = g4.layer_input;
const parts = 42;
const width = 256;

fn bf16(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(li.rn(v))) >> 16);
}

fn value(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// One draw of `rows` rows: the kernel's bits against the host steps after the projection, every value.
fn draw(gpu: check.Gpu, s: cuda.Stream, ple: *const g4.ple.PleRows, r: std.Random, rows: usize, spread: f32) !void {
    const gpa = gpu.gpa;
    const n = rows * parts * width;
    const p = try gpa.alloc(u16, n);
    defer gpa.free(p);
    const t = try gpa.alloc(u16, n);
    defer gpa.free(t);
    var w: [width]u16 = undefined;
    for (p) |*v| v.* = bf16(r.floatNorm(f32) * spread);
    for (t) |*v| v.* = bf16(r.floatNorm(f32) * 0.05);
    for (&w) |*v| v.* = bf16(1 + 0.5 * r.floatNorm(f32));
    const c: g4.ple.Scalars = .{ .scale = li.scalar(1.0 / @sqrt(2560.0)), .token = li.scalar(@sqrt(256.0)), .mix = li.scalar(std.math.sqrt1_2), .eps = 1e-6 };
    var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(p));
    defer dp.free();
    var dt = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(t));
    defer dt.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&w));
    defer dw.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, n * 2);
    defer out.free();
    try ple.run(s, dp.ptr, dt.ptr, dw.ptr, out.ptr, rows, parts, c);
    try s.synchronize();
    const got = try gpa.alloc(u16, n);
    defer gpa.free(got);
    try out.download(0, std.mem.sliceAsBytes(got));
    var wf: [width]f32 = undefined;
    for (&wf, w) |*f, b| f.* = value(b);
    var row: [width]f32 = undefined;
    var bad: usize = 0;
    for (0..rows * parts) |j| {
        const at = j * width;
        for (&row, p[at..][0..width]) |*x, b| x.* = li.rn(value(b) * c.scale);
        li.rms(&row, &wf, &row);
        for (row, t[at..][0..width], got[at..][0..width]) |y, tb, g| {
            const want = bf16(li.rn(y + li.rn(value(tb) * c.token)) * c.mix);
            bad += @intFromBool(want != g);
        }
    }
    try check.expect(bad == 0, "gemma4 ple rows={d} spread={d}: {d} of {d} values differ from the host steps", .{ rows, spread, bad, n });
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var ple = try g4.ple.PleRows.load(gpu.d);
    defer ple.unload();
    var prng = std.Random.DefaultPrng.init(1222);
    for ([_]usize{ 1, 3, 17, 128, 512 }) |rows| for ([_]f32{ 0.02, 1, 40 }) |spread| try draw(gpu, s, &ple, prng.random(), rows, spread);
    try std.testing.expectError(error.RowShape, ple.run(s, 8, 16, 0, 16, 1, 1, undefined));
    check.pass("gemma4 ple rows: 1 to 512 rows of 42 parts bit-equal to layer_input.zig's steps at three scales", .{});
}
