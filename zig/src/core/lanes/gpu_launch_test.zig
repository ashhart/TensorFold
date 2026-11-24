//! A Metal sampler launch of several rows binds each row's settings as that row's own one-row launch did.
const std = @import("std");
const rule = @import("gpu_rule.zig");
const Sampling = @import("sampling.zig").Sampling;
const t = std.testing;

const settings = [_]Sampling{
    .{ .seed = 1234, .temperature = 1.0, .top_k = 20, .top_p = 0.95 },
    .{ .seed = 7, .temperature = 0.7, .top_k = 1 },
    .{ .seed = 9, .temperature = 1.3, .top_k = 1024, .top_p = 0.5 },
    .{ .seed = 11, .temperature = 1.0, .top_k = 1025, .top_p = 1.0 },
    .{ .seed = 13, .temperature = 1.0, .top_k = 0, .top_p = 0.95 },
    .{ .seed = 23, .temperature = 0.9, .top_k = 40, .top_p = 0.9, .min_p = 0.05 },
    .{ .seed = 25, .temperature = 1e-7, .top_k = 20, .top_p = 0.95 },
    .{ .seed = 27, .temperature = 100.0, .top_k = 0, .top_p = 0.8, .min_p = 0.5 },
};

/// Main's one-row binding in Flint's pick: row 0 of arrays padded to 8 rows, as Nemotron's launches pad them.
const OneRow = struct { seeds: [16]u32, positions: [8]u32, cfgs: [32]f32, caps: [8]u32 };

fn mainOneRow(s: Sampling, position: u64) OneRow {
    const r = rule.rule(s);
    var o: OneRow = .{ .seeds = @splat(0), .positions = @splat(0), .cfgs = @splat(0), .caps = @splat(0) };
    o.seeds[0..2].* = r.seedHalves();
    o.positions[0] = @intCast(position);
    o.cfgs[0..4].* = r.cfg();
    o.caps[0] = r.top_k;
    return o;
}

test "sampler launch: a one-row launch binds exactly main's one-row bytes, in Flint's 32-row launch type" {
    for (settings, 0..) |s, i| {
        const position: u64 = 4096 + 17 * i;
        const want = mainOneRow(s, position);
        var l: rule.Launch(32) = .{};
        l.put(s, position);
        const n = l.padded();
        try t.expectEqual(@as(usize, 8), n);
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&want.seeds), std.mem.sliceAsBytes(l.seeds[0 .. 2 * n]));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&want.positions), std.mem.sliceAsBytes(l.positions[0..n]));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&want.cfgs), std.mem.sliceAsBytes(l.cfgs[0 .. 4 * n]));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&want.caps), std.mem.sliceAsBytes(l.caps[0..n]));
    }
}

test "sampler launch: many rows bind each row's settings at its own index, zeros past the last" {
    var prng = std.Random.DefaultPrng.init(32);
    const r = prng.random();
    for ([_]usize{ 1, 2, 7, 8, 9, 31, 32 }) |rows| {
        var many: rule.Launch(32) = .{};
        var picks: [32]struct { s: Sampling, position: u64 } = undefined;
        for (picks[0..rows], 0..) |*p, i| {
            p.* = .{ .s = settings[r.uintLessThan(usize, settings.len)], .position = 4096 + 3 * i };
            p.s.seed = r.int(u64);
            many.put(p.s, p.position);
        }
        try t.expectEqual(@max(rows, 8), many.padded());
        for (picks[0..rows], 0..) |p, i| {
            const one = mainOneRow(p.s, p.position);
            try t.expectEqualSlices(u32, one.seeds[0..2], many.seeds[2 * i ..][0..2]);
            try t.expectEqual(one.positions[0], many.positions[i]);
            try t.expectEqualSlices(u8, std.mem.sliceAsBytes(one.cfgs[0..4]), std.mem.sliceAsBytes(many.cfgs[4 * i ..][0..4]));
            try t.expectEqual(one.caps[0], many.caps[i]);
        }
        for (rows..32) |i| try t.expectEqual(@as(u32, 0), many.positions[i] | many.caps[i] | many.seeds[2 * i] | many.seeds[2 * i + 1]);
        for (many.cfgs[4 * rows ..]) |f| try t.expectEqual(@as(u32, 0), @as(u32, @bitCast(f)));
    }
}
