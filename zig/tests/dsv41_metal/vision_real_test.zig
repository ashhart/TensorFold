//! The release's own vision tower (TF_DSV41_REAL_VISION: its shard and the delimiters) against float64.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const vref = @import("vision_ref.zig");
const vt = @import("vision_test.zig");

const testing = std.testing;
const vis = pkg.vision;

/// A picture-like grid of patches on bf16's grid: smooth colour fields with edges and noise, values in [-1, 1].
fn scene(a: std.mem.Allocator, cfg: vis.Config, gh: u32, gw: u32) ![]f32 {
    var prng = std.Random.DefaultPrng.init(404);
    const rng = prng.random();
    const p = cfg.patch;
    const out = try a.alloc(f32, @as(usize, gh) * gw * cfg.patchValues());
    for (0..gh) |by| for (0..gw) |bx| for (0..3) |c| for (0..p) |y| for (0..p) |x| {
        const fy = @as(f32, @floatFromInt(by * p + y)) / @as(f32, @floatFromInt(gh * p));
        const fx = @as(f32, @floatFromInt(bx * p + x)) / @as(f32, @floatFromInt(gw * p));
        const edge: f32 = if (fx > 0.55 and fy > 0.3) 0.6 else -0.2;
        const v = 0.5 * @sin(6.0 * fx + @as(f32, @floatFromInt(c))) * @cos(4.0 * fy) + edge + 0.08 * (rng.float(f32) * 2 - 1);
        const q: f32 = @max(-1, @min(1, v));
        out[((by * gw + bx) * 3 + c) * p * p + y * p + x] = @as(f32, @bitCast(@as(u32, pkg.weights.bf16Round(q)) << 16));
    };
    return out;
}

test "dsv41 metal vision (real weights): the release's ViT stages and aligner against float64 on one image" {
    const dir_c = std.c.getenv("TF_DSV41_REAL_VISION") orelse return error.SkipZigTest;
    const dir = std.mem.span(dir_c);
    const a = testing.allocator;
    const cfg: vis.Config = .{};
    var pack = try host.pack.Pack.open(a, testing.io, dir);
    defer pack.deinit();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var kit = try vis.Kit.load(device);
    defer kit.deinit();
    var w = try pkg.vision_weights.Weights.load(a, testing.io, device, &pack, cfg, 0, 384);
    defer w.deinit();
    var tower = pkg.vision_tower.Tower.init(a, device, queue, &kit, &w);
    defer tower.deinit();
    var m = try vref.Model.load(a, testing.io, &pack, cfg);
    defer m.deinit();
    const gh: u32 = 6;
    const gw: u32 = 9;
    const px = try scene(a, cfg, gh, gw);
    defer a.free(px);
    const pd = try vt.widen(a, px);
    defer a.free(pd);
    const n = gh * gw;
    const x = try vref.embed(a, &m, pd, n);
    defer a.free(x);
    var done: u32 = 0;
    var worst: f64 = 0;
    for ([_]u32{ 0, 1, 4, 16, 32 }) |stop| {
        try vref.advance(a, &m, x, gh, gw, done, stop);
        done = stop;
        try tower.run(px, gh, gw, .{ .blocks = stop, .finish = false });
        const e = vt.compare(vt.bytesOf(tower.stream(), n * cfg.dim), x, cfg.dim);
        vt.say(e.rms < 0.05, "real tower after {d} of 32 blocks: relative RMS error {d:.5}, worst element {d:.3} of its row's RMS", .{ stop, e.rms, e.worst });
        try testing.expect(e.rms < 0.05);
        worst = @max(worst, e.rms);
    }
    const want = try vref.aligner(a, &m, x, gh, gw);
    defer a.free(want);
    try tower.run(px, gh, gw, .{});
    const t = tower.tokensOf(gh, gw);
    const e = vt.compare(vt.bytesOf(tower.output(), t * cfg.out_dim), want, cfg.out_dim);
    vt.say(e.rms < 0.05, "(d) real tower and aligner on a {d}x{d} patch grid: {d} rows of {d}, relative RMS error {d:.5}, worst element {d:.3}", .{ gh, gw, t, cfg.out_dim, e.rms, e.worst });
    try testing.expect(e.rms < 0.05);
    // the delimiters are the checkpoint's three vectors
    const d = vt.bytesOf(w.delims, 3 * cfg.out_dim);
    var nz: usize = 0;
    for (d) |v| nz += @intFromBool(v != 0);
    vt.say(nz > 3 * cfg.out_dim / 2, "(d) image_start, image_end and image_newline loaded: {d} of {d} values non-zero", .{ nz, d.len });
    try testing.expect(nz > 3 * cfg.out_dim / 2);
}

test "dsv41 metal vision (real weights): the tower's time on the sizes an image takes" {
    const dir_c = std.c.getenv("TF_DSV41_REAL_VISION") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const cfg: vis.Config = .{};
    var pack = try host.pack.Pack.open(a, testing.io, std.mem.span(dir_c));
    defer pack.deinit();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var kit = try vis.Kit.load(device);
    defer kit.deinit();
    var w = try pkg.vision_weights.Weights.load(a, testing.io, device, &pack, cfg, 0, 384);
    defer w.deinit();
    var tower = pkg.vision_tower.Tower.init(a, device, queue, &kit, &w);
    defer tower.deinit();
    for ([_][2]u32{ .{ 39, 39 }, .{ 64, 64 } }) |g| {
        const px = try scene(a, cfg, g[0], g[1]);
        defer a.free(px);
        try tower.run(px, g[0], g[1], .{});
        const t0 = std.Io.Clock.awake.now(testing.io);
        try tower.run(px, g[0], g[1], .{});
        const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(testing.io)).toNanoseconds())) / 1e6;
        vt.say(true, "real tower on {d}x{d} patches ({d} aligner rows): {d:.0} ms", .{ g[0], g[1], tower.tokensOf(g[0], g[1]), ms });
    }
}
