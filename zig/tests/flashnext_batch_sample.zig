//! A device gate for sampled windows with distinct seeds and positions, with untouched greedy and spare rows.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const fz = tf.flashnext_replay;
const sample = tf.flashnext_batch_sample;

pub fn check(a: std.mem.Allocator) !void {
    const dev = try mtl.Device.init();
    defer dev.deinit();
    const lib = try mtl.Library.fromSource(dev, @import("kernel_sources").nemotron_sample, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(dev, lib, "tf_sample_full", false);
    defer pipe.deinit();
    const q = try dev.queue();
    defer q.deinit();
    const logits = try dev.buffer(fz.MAXR * fz.VOCAB * 2, fz.opts);
    defer logits.deinit();
    const positions = try dev.buffer(fz.MAXR * 4, fz.opts);
    defer positions.deinit();
    const picks = try dev.buffer(fz.MAXR * 4, fz.opts);
    defer picks.deinit();
    const plan = try tf.flashnext_batch_meta.Plan.init(&.{
        .{ .slot = 3, .pos = 19, .rows = 3 },
        .{ .slot = 0, .pos = 2047, .rows = 2 },
        .{ .slot = 5, .pos = 9000, .rows = 1 },
    }, 10000);
    const expected_positions = [_]u32{ 20, 21, 22, 2048, 2049, 9001 };
    const settings = [_]?tf.lanes.Sampling{
        .{ .seed = 1, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        null,
        .{ .seed = 1 << 63 | 4, .temperature = 1.3, .top_k = 0, .top_p = 0.8, .min_p = 0.05 },
    };
    var rng = std.Random.DefaultPrng.init(11);
    const random = rng.random();
    for (logits.slice(u16, fz.MAXR * fz.VOCAB)) |*word| {
        const x: f32 = random.floatNorm(f32) * 2;
        const bits: u32 = @bitCast(x);
        word.* = @truncate((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }
    const sentinel = 0xDEAD_BEEF;
    @memset(picks.slice(u32, fz.MAXR), sentinel);
    @memset(positions.slice(u32, fz.MAXR), sentinel);
    @memcpy(positions.slice(u32, fz.MAXR)[0..plan.total], plan.keys[0..plan.total]);
    const cb = q.commandBuffer();
    const enc = cb.compute(.serial);
    for (settings, 0..) |setting, i| if (setting) |p| {
        const start = plan.starts[i];
        const end = if (i + 1 < settings.len) plan.starts[i + 1] else plan.total;
        sample.encode(enc, pipe, .{ .b = logits }, .{ .b = positions }, .{ .b = picks }, start, end - start, p);
    };
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.SampleCommandFailed;
    const row = try a.alloc(f32, fz.VOCAB);
    defer a.free(row);
    for (0..fz.MAXR) |r| {
        const setting: ?tf.lanes.Sampling = if (r < 3) settings[0] else if (r == 5) settings[2] else null;
        const expected = if (setting) |p| blk: {
            for (row, logits.slice(u16, fz.MAXR * fz.VOCAB)[r * fz.VOCAB ..][0..fz.VOCAB]) |*x, w|
                x.* = @bitCast(@as(u32, w) << 16);
            break :blk try tf.lanes.gpu_full.sample(a, row, p, expected_positions[r], null);
        } else sentinel;
        if (picks.slice(u32, fz.MAXR)[r] != expected) return error.PackedSampleMismatch;
        if (positions.slice(u32, fz.MAXR)[r] != (if (r < plan.total) expected_positions[r] else sentinel))
            return error.DrawPositionsOverwritten;
    }
    std.debug.print("PASS GPU sampling: two seeds, unequal positions, packed windows and untouched greedy/spare rows\n", .{});
}
