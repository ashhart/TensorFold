//! A window's rows absorbed ahead of its acceptance, on the GPU over the synthetic drafter.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const syn = @import("glm_metal_dflash_synthetic.zig");
const config = syn.config;
const Context = glm.metal_dflash_model.Context;
const Model = glm.metal_dflash_model.Model;
const ahead = glm.metal_dflash_ahead;

/// Keys and values at every position of `x`'s context equal `y`'s, each read where its ring holds it.
pub fn sameCaches(x: *const Context, y: *const Context) !void {
    const ctx = x.session.context;
    try std.testing.expectEqualDeep(ctx, y.session.context);
    const row = @as(usize, config.kvWidth()) * 2;
    const ring = glm.metal_dflash_context.ring(config);
    for (x.caches, y.caches) |a, b| for ([2][2]mtl.Buffer{ .{ a.keys, b.keys }, .{ a.values, b.values } }) |pair| {
        var p = ctx.begin;
        while (p < ctx.end) : (p += 1) {
            const at: usize = @intCast(p % ring);
            try std.testing.expectEqualSlices(u8, pair[1].contents()[at * row ..][0..row], pair[0].contents()[at * row ..][0..row]);
        }
    };
}

/// `rows` of seeded taps from `seed`.
pub fn seeded(device: mtl.Device, rows: usize, seed: u64) !mtl.Buffer {
    const taps = try device.buffer(rows * config.tapWidth() * 2, mtl.ResourceOptions.shared);
    var random: std.Random.DefaultPrng = .init(seed);
    for (taps.slice(u16, rows * config.tapWidth())) |*v| v.* = syn.bf16FromF32((random.random().float(f32) * 2 - 1) * 2);
    return taps;
}

/// A context fed `rows` taps and absorbed, as a stream's is after its prompt.
pub fn primed(a: std.mem.Allocator, m: *Model, taps: mtl.Buffer, rows: u32) !*Context {
    const x = try Context.init(a, m);
    errdefer x.deinit(a);
    try x.reset(config, 0);
    try syn.feedStream(m, x, taps, 0, rows);
    try m.swap(x);
    defer m.swap(x) catch unreachable;
    try m.flush();
    return x;
}

/// The kept rows of a window absorbed after its acceptance, as today: its committed taps flushed.
fn absorbKept(m: *Model, x: *Context, taps: mtl.Buffer, first: u64, kept: u32) !void {
    try syn.feedStream(m, x, taps, @intCast(first), kept + 1);
    try m.swap(x);
    defer m.swap(x) catch unreachable;
    try m.flush();
}

/// Windows of `rows` absorbed ahead and settled at every kept count: K/V equal to absorbing only the kept rows.
fn testSettled(a: std.mem.Allocator, h: *syn.Harness, q4: bool) !void {
    const m = try syn.create(a, h, q4);
    defer m.deinit();
    const total = 400;
    const taps = try seeded(h.device, total, 313);
    defer taps.deinit();
    for ([_]u32{ 8, 3, 16 }) |rows| for (0..rows) |start| {
        const spec = try primed(a, m, taps, 20); // windows absorbed ahead
        defer spec.deinit(a);
        const host = try primed(a, m, taps, 20); // kept rows absorbed after each window
        defer host.deinit(a);
        // windows past the 64-row window, each with its own kept count, so rejected rows sit under later windows
        var first: u64 = 20;
        var round: u32 = 0;
        while (first + rows <= total) : (round += 1) {
            const kept: u32 = @intCast((start + round) % rows);
            spec.committed_end = first;
            try syn.feedStream(m, spec, taps, @intCast(first), rows); // the window's taps land in the ring, every row
            try ahead.absorbNow(m, &.{.{ .context = spec, .first = first, .rows = rows }});
            ahead.settle(config, spec, first, kept);
            try absorbKept(m, host, taps, first, kept);
            try sameCaches(spec, host);
            first += kept + 1;
        }
    };
}

/// Three streams' windows absorbed ahead in one frame: each stream's K/V equal to its own absorbed alone.
fn testStreams(a: std.mem.Allocator, h: *syn.Harness) !void {
    const m = try syn.create(a, h, true);
    defer m.deinit();
    var taps: [3]mtl.Buffer = undefined;
    for (&taps, 0..) |*t, k| t.* = try seeded(h.device, 200, 500 + k);
    defer for (taps) |t| t.deinit();
    const lens = [3]u32{ 30, 61, 9 };
    const rows = [3]u32{ 8, 5, 8 };
    var spec: [3]*Context = undefined;
    var host: [3]*Context = undefined;
    for (0..3) |k| {
        spec[k] = try primed(a, m, taps[k], lens[k]);
        host[k] = try primed(a, m, taps[k], lens[k]);
    }
    defer for (0..3) |k| {
        spec[k].deinit(a);
        host[k].deinit(a);
    };
    var first = lens;
    for (0..12) |round| {
        var aheads: [3]ahead.Ahead = undefined;
        for (0..3) |k| {
            spec[k].committed_end = first[k];
            try syn.feedStream(m, spec[k], taps[k], first[k], rows[k]);
            aheads[k] = .{ .context = spec[k], .first = first[k], .rows = rows[k] };
        }
        try ahead.absorbNow(m, &aheads);
        for (0..3) |k| {
            const kept: u32 = @intCast((round * 3 + k * 5) % rows[k]);
            ahead.settle(config, spec[k], first[k], kept);
            try absorbKept(m, host[k], taps[k], first[k], kept);
            try sameCaches(spec[k], host[k]);
            first[k] += kept + 1;
        }
    }
    // a context with rows it has not absorbed is refused rather than absorbed out of order
    spec[0].committed_end = first[0] + 2;
    const late = ahead.Ahead{ .context = spec[0], .first = first[0] + 1, .rows = 4 };
    try std.testing.expectError(error.BadDraftContext, ahead.absorbNow(m, &.{late}));
}

pub fn run(a: std.mem.Allocator, h: *syn.Harness) !void {
    try testSettled(a, h, true);
    try testSettled(a, h, false);
    try testStreams(a, h);
}
