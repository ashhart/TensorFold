//! window_attention's GPU time at Kolibri-1's served shapes (its caller) and Gemma 4's, 1k to 128k keys (synthetic).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const wa = cuda.window_attention;

const warm = 3;
const reps = 25;

/// One timed call: decode rows at a position, or a prompt chunk at p0.
pub const Shape = struct { name: []const u8, h: usize, hk: usize, d: usize, win: usize, ring: usize, rows: usize, at: usize, prompt: bool, p32: bool = false };

/// Kolibri-1's calls: 48 query heads over 4 of 128, window 513, ring 8768 (an 8192-row chunk); full layers unwindowed.
pub const callers = [_]Shape{
    .{ .name = "kolibri decode tf_window_decode_tc, 1 row", .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 8768, .rows = 1, .at = 6000, .prompt = false },
    .{ .name = "kolibri decode tf_window_decode_tc, 4 rows", .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 8768, .rows = 4, .at = 6000, .prompt = false },
    .{ .name = "kolibri decode tf_window_decode_tc, 16 rows", .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 8768, .rows = 16, .at = 6000, .prompt = false },
    .{ .name = "kolibri decode tf_window_decode_tc, 64 rows", .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 8768, .rows = 64, .at = 6000, .prompt = false },
    .{ .name = "kolibri sliding prompt wattn_kernel<128,8,4,8>, 2048 rows at 0", .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 8768, .rows = 2048, .at = 0, .prompt = true },
    .{ .name = "kolibri sliding prompt wattn_kernel<128,8,4,8>, 8192 rows at 8192", .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 8768, .rows = 8192, .at = 8192, .prompt = true },
    .{ .name = "kolibri full prompt wattn_kernel<128,8,4,8>, 2048 rows at 8192", .h = 48, .hk = 4, .d = 128, .win = 0, .ring = 32768, .rows = 2048, .at = 8192, .prompt = true },
    .{ .name = "kolibri full prompt wattn_kernel<128,8,4,8>, 2048 rows at 30720", .h = 48, .hk = 4, .d = 128, .win = 0, .ring = 32768, .rows = 2048, .at = 30720, .prompt = true },
};

const contexts = [_]usize{ 1024, 8192, 32768, 131072 };

/// Gemma 4's sliding (256, window 1024, ring 1536) and global layers (512, linear): decode rows, 512-row prompt chunks.
pub fn sliding(out: []Shape) []Shape {
    var n: usize = 0;
    for (contexts) |ctx| {
        for ([_]usize{ 1, 4 }) |rows| {
            out[n] = .{ .name = "12B sliding decode (16/8 x 256)", .h = 16, .hk = 8, .d = 256, .win = 1024, .ring = 1536, .rows = rows, .at = ctx - 1, .prompt = false };
            n += 1;
        }
        out[n] = .{ .name = "31B sliding decode (32/16 x 256)", .h = 32, .hk = 16, .d = 256, .win = 1024, .ring = 1536, .rows = 1, .at = ctx - 1, .prompt = false };
        n += 1;
        for ([_]bool{ true, false }) |p32| {
            out[n] = .{ .name = if (p32) "12B sliding prompt (16/8 x 256), P32" else "12B sliding prompt (16/8 x 256), P bf16", .h = 16, .hk = 8, .d = 256, .win = 1024, .ring = 1536, .rows = 512, .at = ctx - 512, .prompt = true, .p32 = p32 };
            n += 1;
        }
        if (!wa.wide(512)) continue;
        for ([_]usize{ 1, 4 }) |rows| {
            out[n] = .{ .name = "12B global decode (16/1 x 512)", .h = 16, .hk = 1, .d = 512, .win = 0, .ring = ctx, .rows = rows, .at = ctx - 1, .prompt = false };
            n += 1;
        }
        out[n] = .{ .name = "31B global decode (32/4 x 512)", .h = 32, .hk = 4, .d = 512, .win = 0, .ring = ctx, .rows = 1, .at = ctx - 1, .prompt = false };
        n += 1;
        for ([_]bool{ true, false }) |p32| {
            out[n] = .{ .name = if (p32) "12B global prompt (16/1 x 512), P32" else "12B global prompt (16/1 x 512), P bf16", .h = 16, .hk = 1, .d = 512, .win = 0, .ring = ctx, .rows = 512, .at = ctx - 512, .prompt = true, .p32 = p32 };
            n += 1;
        }
        out[n] = .{ .name = "31B global prompt (32/4 x 512), P32", .h = 32, .hk = 4, .d = 512, .win = 0, .ring = ctx, .rows = 512, .at = ctx - 512, .prompt = true, .p32 = true };
        n += 1;
    }
    return out[0..n];
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

/// Microseconds a call takes on the GPU (median of `reps`); inputs are normal bf16 values, outputs discarded.
pub fn time(gpu: check.Gpu, s: cuda.Stream, g: *const wa.WindowAttention, x: Shape) !f64 {
    const crow = if (x.prompt) x.ring else x.ring * x.rows; // decode: a stream a row
    const q_bytes = x.rows * x.h * x.d * 2;
    const c_bytes = crow * x.hk * x.d * 2;
    var prng = std.Random.DefaultPrng.init(x.rows *% 31 +% x.at);
    const host = try gpu.gpa.alloc(u16, @max(q_bytes, c_bytes) / 2);
    defer gpu.gpa.free(host);
    for (host) |*v| {
        const f: f32 = prng.random().floatNorm(f32) * 0.5;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    var bufs: [7]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    const sizes = [_]usize{ q_bytes, c_bytes, c_bytes, q_bytes, x.rows * 4, x.rows * 4, if (x.prompt) 16 else wa.WindowAttention.scratchBytes(.{ .h = x.h, .hk = x.hk, .d = x.d, .win = x.win, .ring = x.ring, .scale = 1 }, x.rows) };
    for (sizes, &bufs) |n, *b| {
        b.* = try cuda.DeviceBuffer.alloc(gpu.d, @max(n, 16));
        made += 1;
    }
    try bufs[0].upload(0, std.mem.sliceAsBytes(host[0 .. q_bytes / 2]));
    try bufs[1].upload(0, std.mem.sliceAsBytes(host[0 .. c_bytes / 2]));
    try bufs[2].upload(0, std.mem.sliceAsBytes(host[0 .. c_bytes / 2]));
    const pos = try gpu.gpa.alloc(i32, 2 * x.rows);
    defer gpu.gpa.free(pos);
    for (pos[0..x.rows], pos[x.rows..], 0..) |*p, *sl, i| {
        p.* = @intCast(x.at);
        sl.* = @intCast(i);
    }
    try bufs[4].upload(0, std.mem.sliceAsBytes(pos[0..x.rows]));
    try bufs[5].upload(0, std.mem.sliceAsBytes(pos[x.rows..]));
    const w: wa.Window = .{ .h = x.h, .hk = x.hk, .d = x.d, .win = x.win, .ring = x.ring, .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(x.d)))) };
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var us: [reps]f64 = undefined;
    for (0..warm + reps) |i| {
        try start.record(s);
        if (x.prompt) {
            try call(g, s, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, x, w);
        } else try g.decode(s, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, bufs[4].ptr, bufs[5].ptr, w, bufs[6].ptr, x.rows);
        try end.record(s);
        try s.synchronize();
        if (i >= warm) us[i - warm] = @as(f64, try start.elapsedMs(end)) * 1000;
    }
    return median(&us);
}

fn call(g: *const wa.WindowAttention, s: cuda.Stream, q: u64, kc: u64, vc: u64, out: u64, x: Shape, w0: wa.Window) !void {
    if (@hasDecl(wa, "wide") and wa.wide(x.d)) {
        var w = w0;
        w.p32 = x.p32;
        return g.promptRows(s, q, kc, vc, out, x.at, x.rows, w, null);
    }
    return g.prompt(s, q, kc, vc, out, x.at, x.rows, w0);
}

/// `which`: callers (Kolibri-1's shapes) or gemma (the new head dims).
pub fn run(gpu: check.Gpu, which: []const u8) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try wa.WindowAttention.load(gpu.d);
    defer g.unload();
    var buf: [64]Shape = undefined;
    const list: []const Shape = if (std.mem.eql(u8, which, "callers")) &callers else if (@hasDecl(wa, "wide")) sliding(&buf) else return error.UnknownBench;
    for (list) |x| {
        const us = try time(gpu, s, &g, x);
        std.debug.print("RESULT window-bench {s}, rows {d}, {s} {d}: {d:.1} us\n", .{ x.name, x.rows, if (x.prompt) "p0" else "position", x.at, us });
    }
}
