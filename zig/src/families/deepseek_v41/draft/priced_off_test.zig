//! With TF_DSV41_DRAFT_PRICED off the depth rule's choices are the ones it made before the setting existed.
const std = @import("std");
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");

const gpa = std.testing.allocator;

/// Three rounds' ks a case (joint modes, confidences, 1..4 streams, first draft, two tables), hashed in order.
pub fn choicesHash(priced: bool) !struct { hash: u64, ks: usize } {
    var lin = try costs_mod.defaults(gpa, 64);
    defer lin.deinit(gpa);
    var m3buf: [32]f64 = undefined;
    for (&m3buf, 1..) |*v, r| v.* = 30.6 + 10.0 * @as(f64, @floatFromInt(r - 1)) + (if (r > 16) @as(f64, 20.0) else 0.0);
    const m3: costs_mod.Costs = .{ .verify = &m3buf, .draft = 12.0, .slot = 0.5 };
    var h = std.hash.Fnv1a_64.init();
    var count: usize = 0;
    for ([_]costs_mod.Costs{ lin, m3 }) |c| for ([_]u2{ 0, 1, 2 }) |jm| for ([_]f64{ 0.95, 0.8, 0.6, 0.4, 0.2 }) |p| for (1..5) |n| for ([_]?u32{ null, 7 }) |first| {
        var d = try depth.Depth.init(gpa, c, 5, .{ .joint = jm, .priced = priced });
        defer d.deinit();
        const conf = [5]f64{ p, p * 0.98, p * 0.96, p * 0.94, p * 0.92 };
        var asks: [4]depth.Depth.Ask = undefined;
        for (asks[0..n], 0..) |*a, i| a.* = .{ .slot = @intCast(i), .conf = &conf, .most = 5, .first = first, .position = 100 + i };
        for (0..3) |round| {
            var ks: [4]usize = undefined;
            try d.chooseJoint(asks[0..n], 0, 0, null, ks[0..n]);
            for (ks[0..n], 0..) |k, i| {
                h.update(std.mem.asBytes(&@as(u64, k)));
                count += 1;
                // the window keeps its drafts up to a miss that moves with the round and the slot
                const keep = 1 + @min(k, (round + i) % 4);
                try d.record(@intCast(i), 1 + k, keep, null);
                d.bonus(@intCast(i), if ((round + i) % 2 == 0) 7 else 8);
            }
        }
    };
    return .{ .hash = h.final(), .ks = count };
}

test "priced off: the same ks as before the setting, across joint modes, confidences, tables and 1..4 streams" {
    const got = try choicesHash(false);
    try std.testing.expectEqual(@as(usize, 1800), got.ks);
    try std.testing.expectEqual(golden, got.hash);
}

test "priced on: the same rounds choose differently (the hash sees the setting)" {
    const got = try choicesHash(true);
    try std.testing.expect(got.hash != golden);
}

/// choicesHash(false) on the base commit, before TF_DSV41_DRAFT_PRICED existed.
const golden: u64 = 0xec5f995eeeba8c04;
