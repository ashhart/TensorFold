//! Recall's reader: priced proposals, the quiet rounds after a miss, bands that back off and probe, graft candidates.
const std = @import("std");
const recall = @import("recall.zig");
const Rates = recall.Rates;
const gpa = std.testing.allocator;

fn ids(comptime n: usize, from: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = from + @as(u32, @intCast(i));
    return out;
}

const Fixture = struct {
    shared: recall.Shared,

    fn init() !Fixture {
        return .{ .shared = .{ .io = std.testing.io, .record = try recall.Record.init(gpa, 14) } };
    }

    fn deinit(f: *Fixture) void {
        f.shared.record.deinit();
    }
};

test "bands, and a copy's expected tokens at a per-id rate" {
    try std.testing.expectEqual(@as(usize, 0), Rates.band(16));
    try std.testing.expectEqual(@as(usize, 0), Rates.band(31));
    try std.testing.expectEqual(@as(usize, 1), Rates.band(32));
    try std.testing.expectEqual(@as(usize, 2), Rates.band(64));
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), recall.expected(0.5, 2), 1e-12);
    try std.testing.expectEqual(@as(f64, 1), recall.expected(0, 15));
}

test "a long enough match is proposed, a short one is not, and a copy that lands nothing quiets the stream" {
    var f = try Fixture.init();
    defer f.deinit();
    const seg = ids(100, 100);
    try f.shared.record.add(&seg);
    var r: recall.Reader = .{ .gpa = gpa, .shared = &f.shared };
    defer r.deinit();
    try std.testing.expectEqualSlices(u32, seg[70..76], r.propose(seg[0..70], 6, 0));
    try std.testing.expectEqual(@as(u64, 1), r.lookups);
    _ = r.propose(seg[0..70], 6, 0); // the same context again in its round: no second lookup
    try std.testing.expectEqual(@as(u64, 1), r.lookups);
    r.observe(6, 0, 0); // the 64-id band still pays after one miss: only the quiet rounds hold it back
    for (1..5) |round| try std.testing.expectEqual(@as(usize, 0), r.propose(seg[0..71], 6, round).len);
    try std.testing.expectEqualSlices(u32, seg[71..77], r.propose(seg[0..71], 6, 5));
    try std.testing.expectEqual(@as(usize, 0), r.propose(seg[4..18], 6, 6).len); // 14 matched ids: under min_match
}

test "a band whose copies miss stops proposing, probes ever more rarely, and a landed probe ends the backoff" {
    var f = try Fixture.init();
    defer f.deinit();
    const seg = ids(200, 1000);
    try f.shared.record.add(&seg);
    var r: recall.Reader = .{ .gpa = gpa, .shared = &f.shared };
    defer r.deinit();
    f.shared.rates.landed[0] = 0.1; // the 16-31 band's copies have landed next to nothing
    f.shared.rates.missed[0] = 10;
    var round: u64 = 0;
    var offered: usize = 0;
    var probes_seen: usize = 0;
    for (0..64) |i| { // a priced-out band still offers its probes: one every probe_every lookups
        round += 10;
        const got = r.propose(seg[i .. i + 20], 8, round);
        if (got.len > 0) {
            offered += 1;
            probes_seen += @intFromBool(r.probing);
            r.observe(@intCast(got.len), 0, round);
        }
    }
    try std.testing.expect(offered >= 1 and offered == probes_seen and offered <= 2);
    try std.testing.expect(f.shared.rates.probe_every > 32);
    f.shared.rates.observe(0, 8, 8, true);
    try std.testing.expectEqual(@as(u32, 32), f.shared.rates.probe_every);
}

test "the stream's own rounds set the bar a copy must beat" {
    var f = try Fixture.init();
    defer f.deinit();
    const seg = ids(60, 500);
    try f.shared.record.add(&seg);
    var r: recall.Reader = .{ .gpa = gpa, .shared = &f.shared };
    defer r.deinit();
    for (0..40) |_| r.noteHead(6); // a head that lands six tokens a round
    try std.testing.expectEqual(@as(usize, 0), r.propose(seg[0..20], 4, 0).len); // 0.6 an id over four ids: under six
    r.memo = null;
    for (0..40) |_| r.noteHead(1);
    try std.testing.expectEqual(@as(usize, 4), r.propose(seg[0..20], 4, 0).len);
}

test "graft candidates are distinct continuations weighted up to the cap" {
    var f = try Fixture.init();
    defer f.deinit();
    const ctx = ids(20, 300);
    try f.shared.record.add(&(ctx ++ [_]u32{ 7, 8, 9 }));
    try f.shared.record.add(&(ctx ++ [_]u32{ 7, 8, 9 }));
    try f.shared.record.add(&(ctx ++ [_]u32{ 4, 5 }));
    var r: recall.Reader = .{ .gpa = gpa, .shared = &f.shared };
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const got = try r.candidates(arena.allocator(), &ctx, 3, 8, 16);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, got[0].tokens);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, got[1].tokens);
    try std.testing.expect(got[0].recalled and got[0].weight == 16);
}
