//! The picks a chain behind the verify makes on the GPU against the host's: greedy, and keyed where the host races.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");

const lanes = host.lanes;
const testing = std.testing;
const none: u32 = 0xffffffff;

extern "c" fn log(x: f64) f64;

/// The host's score gap between its winner and the runner-up of a raced row (ties to the sorted order aside).
fn gap(row: []const f32, s: lanes.Sampling, draw: u64) f64 {
    const t = @max(s.temperature, 1e-6);
    var best = -std.math.inf(f64);
    var next = -std.math.inf(f64);
    for (row, 0..) |v, id| {
        const score = @as(f64, v) / t - log(-log(lanes.sampling.uniform(s.seed, draw, id)));
        if (score > best) {
            next = best;
            best = score;
        } else if (score > next) next = score;
    }
    return best - next;
}

test "dsv41 metal early picks: keyed picks equal the host's race (bar near-ties), greedy picks its argmax, bad rows none" {
    const a = testing.allocator;
    const device = try mtl.Device.init();
    const queue = try device.queue();
    var picks = try pkg.early_picks.Picks.init(device);
    defer picks.deinit();
    const rows = 16;
    var prng = std.Random.DefaultPrng.init(4104);
    const rnd = prng.random();
    const ids = try a.alloc(u64, 129280);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = i;
    const vals = try a.alloc(f64, 129280);
    defer a.free(vals);
    const cases = [_]lanes.Sampling{
        .{ .seed = 1234, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 77, .temperature = 0.7, .top_k = 0, .top_p = 0.0 },
        .{ .seed = 1 << 62 | 5, .temperature = 1.3, .top_k = 129280, .top_p = 1.0 },
    };
    var checked: usize = 0;
    var near: usize = 0;
    var bad: usize = 0;
    for ([_]u32{ 129280, 3001 }) |vocab| {
        const lg = try device.buffer(rows * @as(usize, vocab) * 4, mtl.ResourceOptions.shared);
        defer lg.deinit();
        const out = try device.buffer(rows * 4 + 8 * rows, mtl.ResourceOptions.shared);
        defer out.deinit();
        const logits = lg.slice(f32, rows * vocab);
        for (cases) |s| for (0..3) |round| {
            for (0..rows) |r| for (logits[r * vocab ..][0..vocab]) |*v| {
                // every third row of whole values ties often
                v.* = if (r % 3 == 0) @floatFromInt(rnd.intRangeAtMost(i32, -12, 12)) else rnd.floatNorm(f32) * 3;
            };
            // a row holding a NaN and one an infinity: the host sorts them, the GPU picks none
            logits[5 * vocab + 17] = std.math.nan(f32);
            logits[9 * vocab + 3] = std.math.inf(f32);
            var draws: [rows]u64 = undefined;
            for (&draws, 0..) |*d, i| d.* = 900 + 41 * round + i;
            pkg.early_picks.bases(s, &draws, @alignCast(std.mem.bytesAsSlice(u64, out.contents()[rows * 4 ..][0 .. 8 * rows])));
            {
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = queue.commandBuffer();
                const e = cb.compute(.serial);
                picks.encode(e, .{ .buf = lg }, .{ .buf = out }, .{ .buf = out, .off = rows * 4 }, rows, vocab, s);
                e.end();
                cb.commit();
                cb.wait();
            }
            const got = out.slice(u32, rows);
            for (0..rows) |r| {
                const row = logits[r * vocab ..][0..vocab];
                if (r == 5 or r == 9) {
                    bad += @intFromBool(got[r] != none);
                    continue;
                }
                for (vals[0..vocab], row) |*o, v| o.* = v;
                const want: u32 = @intCast(try lanes.sampling.choose(a, vals[0..vocab], ids[0..vocab], draws[r], s));
                checked += 1;
                if (got[r] == want) continue;
                if (gap(row, s, draws[r]) < 1e-4) near += 1 else bad += 1;
            }
        };
    }
    std.debug.print("{s} dsv41 metal early picks: {d} keyed rows equal the host's race, {d} near-ties apart, {d} wrong\n", .{ if (bad == 0) "PASS" else "FAIL", checked - near, near, bad });
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expect(near * 100 <= checked);
}

test "dsv41 metal early picks: greedy picks are the host's first maximum" {
    const device = try mtl.Device.init();
    const queue = try device.queue();
    var picks = try pkg.early_picks.Picks.init(device);
    defer picks.deinit();
    const rows = 8;
    const vocab = 129280;
    var prng = std.Random.DefaultPrng.init(4105);
    const lg = try device.buffer(rows * vocab * 4, mtl.ResourceOptions.shared);
    defer lg.deinit();
    const out = try device.buffer(rows * 4, mtl.ResourceOptions.shared);
    defer out.deinit();
    const logits = lg.slice(f32, rows * vocab);
    for (logits, 0..) |*v, i| v.* = if ((i / vocab) % 2 == 0) @floatFromInt(prng.random().intRangeAtMost(i32, -9, 9)) else prng.random().floatNorm(f32);
    {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        picks.encode(e, .{ .buf = lg }, .{ .buf = out }, .{ .buf = out }, rows, vocab, null);
        e.end();
        cb.commit();
        cb.wait();
    }
    var bad: usize = 0;
    for (out.slice(u32, rows), 0..) |g, r| bad += @intFromBool(g != pkg.pick.argmax(logits[r * vocab ..][0..vocab]));
    try testing.expectEqual(@as(usize, 0), bad);
}
