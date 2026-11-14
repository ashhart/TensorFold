//! TF_GEMMA4_V_ONLY's GPU time against the K+V kernels over the served pages (256 positions, shuffled), cold L2.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rows = @import("window_rows_tests.zig");
const pages512 = @import("window_pages512_tests.zig");

const wa = cuda.window_attention;
const vo_ = cuda.window_vonly;

const d = 512;
const warm = 3;
const reps = 25;
const tok = 256;

const Shape = struct { name: []const u8, h: usize, hk: usize };
const shapes = [_]Shape{
    .{ .name = "31B (32/4 x 512)", .h = 32, .hk = 4 },
    .{ .name = "12B (16/1 x 512)", .h = 16, .hk = 1 },
};

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

/// `window-bench vonly`: decode 1 row and a 512-row prompt chunk at 8K, 32K and 64K keys, K+V against V-only.
pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try wa.WindowAttention.load(gpu.d);
    defer g.unload();
    try g.loadPages(gpu.d);
    var v = try vo_.VOnly.load(gpu.d);
    defer v.unload();
    var b: rows.Bufs = .{ .gpu = gpu };
    defer b.deinit();
    var prng = std.Random.DefaultPrng.init(0x51a7);
    const rng = prng.random();
    var inv_h: [d / 2]f32 = undefined;
    core.rope_freq.proportional(1000000.0, d, 64, &inv_h);
    var kn_h: [d]u16 = undefined;
    for (&kn_h) |*x| x.* = core.bf16_round.bits(0.25 + 1.5 * rng.float(f32));
    const vk: vo_.Vk = .{ .w = try b.up(std.mem.sliceAsBytes(&kn_h)), .inv = try b.up(std.mem.sliceAsBytes(&inv_h)) };
    const flush = try b.out(s, 96 << 20); // past the 24 MiB L2: every rep starts cold
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for ([_]usize{ 8192, 32768, 65536 }) |ctx| for (shapes) |x| {
        _ = arena.reset(.retain_capacity);
        const q = try a.alloc(u16, 512 * x.h * d);
        const kc = try a.alloc(u16, ctx * x.hk * d);
        const vc = try a.alloc(u16, kc.len);
        rows.fill(rng, q, kc, vc, d);
        const pool = try pages512.paged(&b, a, rng, kc, vc, 1, ctx, x.hk, tok);
        const dq = try b.up(std.mem.sliceAsBytes(q));
        const dp = try b.up(std.mem.sliceAsBytes(&[1]i32{@intCast(ctx - 1)}));
        const ds = try b.up(std.mem.sliceAsBytes(&[1]i32{0}));
        const w: wa.Window = .{
            .h = x.h,
            .hk = x.hk,
            .d = d,
            .win = 0,
            .ring = ctx,
            .scale = 0.0625,
            .pg = pool.pages,
        };
        var pw = w;
        pw.ring = 0;
        pw.p32 = true;
        const scratch = try b.out(s, wa.WindowAttention.scratchBytes(w, 1));
        const out = try b.out(s, 512 * x.h * d * 2);
        const ks = try b.out(s, (ctx / tok + 3) * tok * x.hk * d * 2); // V-only prompt keys, the pool's layout
        for ([_]bool{ false, true }) |prompt| for ([_]bool{ false, true }) |vonly| {
            const us = try a.alloc(f64, reps);
            for (0..warm + reps) |i| {
                const f: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = flush, .len = 96 << 20, .borrowed = true };
                try f.fill8(@intCast(i), s.handle);
                try start.record(s);
                if (prompt and vonly) { // the chunk's keys made once into the scratch, then the K+V kernel
                    const keys: vo_.VOnly.Keys = .{ .vc = pool.vc, .ks = ks, .pg = pool.pages, .hk = x.hk, .vk = vk };
                    try v.keys(s, keys, 0, ctx);
                    try g.promptRows(s, dq, ks, pool.vc, out, ctx - 512, 512, pw, null);
                } else if (prompt) {
                    try g.promptRows(s, dq, pool.kc, pool.vc, out, ctx - 512, 512, pw, null);
                } else if (vonly) {
                    try v.decode(s, dq, pool.vc, out, dp, ds, w, vk, scratch, 1);
                } else try g.decode(s, dq, pool.kc, pool.vc, out, dp, ds, w, scratch, 1);
                try end.record(s);
                try s.synchronize();
                if (i >= warm) us[i - warm] = @as(f64, try start.elapsedMs(end)) * 1000;
            }
            const what = if (prompt) "prompt 512 rows" else "decode 1 row";
            const arm = if (vonly) "vo" else "kv";
            const line = "RESULT window-bench vonly {s} {s} {s} ctx {d}: {d:.1} us (median of {d}, cold L2)\n";
            std.debug.print(line, .{ x.name, what, arm, ctx, median(us), reps });
        };
    };
    check.pass("window-bench vonly: K+V and V-only at 8K, 32K and 64K keys", .{});
}
