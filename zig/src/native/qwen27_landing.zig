//! How often the 27B's drafted nodes land, by their own log-probability: greedy rounds' counts kept apart.
const std = @import("std");

pub const Landing = struct {
    counts: [2][16][2]f64 = @splat(@splat(.{ 0, 0 })), // greedy rounds', then every round's: offered, landed

    fn bucket(own: f64) usize {
        return @min(15, @as(usize, @intFromFloat(@max(0, @floor(-own / 0.5)))));
    }

    /// A node offered with its parent row kept; `keyed`: a sampled round, which the greedy counts skip.
    pub fn observe(l: *Landing, keyed: bool, own: f64, landed: bool) void {
        for (l.counts[@intFromBool(keyed)..]) |*table| {
            const x = &table[bucket(own)];
            x[0] += 1;
            if (landed) x[1] += 1;
        }
    }

    /// A node's chance of landing given its parent: its bucket's counts, e^own as the prior.
    pub fn chance(l: *const Landing, keyed: bool, own: f64) f64 {
        const x = l.counts[@intFromBool(keyed)][bucket(own)];
        return (x[1] + 2 * @exp(own)) / (x[0] + 2);
    }
};

test "sampled rounds' landings move sampled streams' chances alone" {
    var l: Landing = .{};
    for (0..40) |_| l.observe(false, -0.2, true);
    const greedy = l.chance(false, -0.2);
    const every = l.chance(true, -0.2);
    try std.testing.expectEqual(greedy, every); // greedy rounds count for both
    for (0..400) |_| l.observe(true, -0.2, false);
    try std.testing.expectEqual(greedy, l.chance(false, -0.2));
    try std.testing.expect(l.chance(true, -0.2) < 0.1 * greedy);
    try std.testing.expectEqual(@as(usize, 15), Landing.bucket(-30.0));
}
