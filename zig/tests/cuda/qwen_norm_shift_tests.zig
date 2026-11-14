//! qwen35.cu's _shift norms (a plain BF16 checkpoint's): HF's raw w as 1 + w in FP32, within one BF16 ulp of FP64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const D = 4096; // Clef-Flash's hidden width (the vectorized path)
const rows = 5;
const HD = 256; // the attention head width the q and k norms see
const heads = 6;
const eps: f32 = 1e-6;

fn widen(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// FP32 to BF16, round to nearest even (no NaN in these inputs).
fn round(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// A BF16 ulp at |x|: 8 bits of significand.
fn ulp(x: f64) f64 {
    const ax = @abs(x);
    return if (ax == 0) std.math.floatMin(f32) else std.math.pow(f64, 2, @floor(std.math.log2(ax)) - 7);
}

/// How many of `got` lie further than one BF16 ulp from (1 + w) * x * inv computed in FP64 per row of `width`.
fn misses(got: []const u16, x: []const u16, w: []const u16, width: usize) usize {
    var bad: usize = 0;
    var r: usize = 0;
    while (r * width < x.len) : (r += 1) {
        const row = x[r * width ..][0..width];
        var ss: f64 = 0;
        for (row) |v| ss += widen(v) * widen(v);
        const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
        for (row, got[r * width ..][0..width], 0..) |v, g, i| {
            const want = (1.0 + widen(w[i])) * widen(v) * inv;
            // a non-finite output misses (the fixtures are finite): NaN fails no comparison of its own
            if (!std.math.isFinite(widen(g)) or @abs(widen(g) - want) > ulp(want)) bad += 1;
        }
    }
    return bad;
}

fn buffer(gpu: check.Gpu, data: []const u16) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, data.len * 2);
    errdefer b.free();
    try b.upload(0, std.mem.sliceAsBytes(data));
    return b;
}

fn fill(rand: std.Random, out: []u16, scale: f32) void {
    for (out) |*v| v.* = round((rand.float(f32) * 2 - 1) * scale);
}

/// `tf_q35_norm_shift` with a residual, then `tf_q35_head_norm_shift`; the unshifted kernels on raw w: red controls.
pub fn run(gpu: check.Gpu) !void {
    try check.expect(gateSelfCheck(), "qwen norm shift: the gate passes exact rows and misses NaN and inf", .{});
    const a = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x7f4a);
    const rand = prng.random();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer mod.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    // the norm: h = bf16(h + r) written back, x from it
    const h = try a.alloc(u16, rows * D);
    defer a.free(h);
    const r = try a.alloc(u16, rows * D);
    defer a.free(r);
    const w = try a.alloc(u16, D);
    defer a.free(w);
    fill(rand, h, 4);
    fill(rand, r, 4);
    fill(rand, w, 0.5); // HF's raw norm weights lie around 0
    const summed = try a.alloc(u16, rows * D);
    defer a.free(summed);
    for (summed, h, r) |*o, x, y| o.* = round(@floatCast(widen(x) + widen(y)));
    for ([_]bool{ true, false }) |shift| {
        var hb = try buffer(gpu, h);
        defer hb.free();
        var rb = try buffer(gpu, r);
        defer rb.free();
        var wb = try buffer(gpu, w);
        defer wb.free();
        var xb = try cuda.DeviceBuffer.alloc(gpu.d, rows * D * 2);
        defer xb.free();
        var args: cuda.Args = .{};
        inline for (.{ hb.ptr, rb.ptr, wb.ptr, xb.ptr, eps, @as(c_int, D), @as(u64, 0), @as(u64, 0) }) |v| args.add(v);
        const f = try mod.function(if (shift) "tf_q35_norm_shift" else "tf_q35_norm");
        try cuda.launch.launch(f, .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, s, &args);
        try s.synchronize();
        const got = try a.alloc(u16, rows * D);
        defer a.free(got);
        try xb.download(0, std.mem.sliceAsBytes(got));
        const back = try a.alloc(u16, rows * D);
        defer a.free(back);
        try hb.download(0, std.mem.sliceAsBytes(back));
        try check.expect(std.mem.eql(u16, back, summed), "qwen norm shift: h + r written back as bf16", .{});
        const bad = misses(got, summed, w, D);
        if (shift) {
            try check.expect(bad == 0, "tf_q35_norm_shift: {d} of {d} values past one BF16 ulp of FP64 (1 + w) x inv", .{ bad, rows * D });
            check.pass("tf_q35_norm_shift: {d} rows of {d} within one BF16 ulp of FP64", .{ rows, D });
        } else {
            try check.expect(bad > 0, "control: tf_q35_norm on raw w passed the (1 + w) gate", .{});
            check.pass("control: tf_q35_norm on raw w fails the gate ({d} values)", .{bad});
        }
    }
    // the q and k head norm: rows of `heads` heads of HD, packed
    const x = try a.alloc(u16, rows * heads * HD);
    defer a.free(x);
    const hw = try a.alloc(u16, HD);
    defer a.free(hw);
    fill(rand, x, 8);
    fill(rand, hw, 0.5);
    for ([_]bool{ true, false }) |shift| {
        var xb = try buffer(gpu, x);
        defer xb.free();
        var wb = try buffer(gpu, hw);
        defer wb.free();
        var ob = try cuda.DeviceBuffer.alloc(gpu.d, x.len * 2);
        defer ob.free();
        var args: cuda.Args = .{};
        inline for (.{ xb.ptr, wb.ptr, ob.ptr, @as(c_int, heads), @as(c_int, heads * HD), @as(c_int, HD), eps, @as(u64, 0) }) |v| args.add(v);
        const f = try mod.function(if (shift) "tf_q35_head_norm_shift" else "tf_q35_head_norm");
        try cuda.launch.launch(f, .{ .grid = .{ .x = rows * heads }, .block = .{ .x = 64 } }, s, &args);
        try s.synchronize();
        const got = try a.alloc(u16, x.len);
        defer a.free(got);
        try ob.download(0, std.mem.sliceAsBytes(got));
        const bad = misses(got, x, hw, HD);
        if (shift) {
            try check.expect(bad == 0, "tf_q35_head_norm_shift: {d} of {d} values past one BF16 ulp of FP64", .{ bad, x.len });
            check.pass("tf_q35_head_norm_shift: {d} heads within one BF16 ulp of FP64", .{rows * heads});
        } else {
            try check.expect(bad > 0, "control: tf_q35_head_norm on raw w passed the (1 + w) gate", .{});
            check.pass("control: tf_q35_head_norm on raw w fails the gate ({d} values)", .{bad});
        }
    }
}

/// The gate itself: exact outputs pass, and a NaN or infinite one misses (NaN fails no comparison of its own).
fn gateSelfCheck() bool {
    const x = [_]u16{ 0x3f80, 0x4000 };
    const w = [_]u16{ 0, 0 };
    const inv = 1.0 / @sqrt((1.0 + 4.0) / 2.0 + eps);
    var got = [_]u16{ round(@floatCast(1.0 * inv)), round(@floatCast(2.0 * inv)) };
    if (misses(&got, &x, &w, 2) != 0) return false;
    got[1] = 0x7fc0; // NaN
    if (misses(&got, &x, &w, 2) != 1) return false;
    got[1] = 0x7f80; // +inf
    return misses(&got, &x, &w, 2) == 1;
}
