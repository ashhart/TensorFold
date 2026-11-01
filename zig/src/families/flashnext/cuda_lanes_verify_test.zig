//! Shared verify dispatch with a recording owner: TP=2 groups go to the ranks, a lone window verifies alone.
const std = @import("std");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const verify = @import("cuda_lanes_verify.zig");
const be = lanes.backend;
const t = std.testing;

/// Each call a pass made: 0 a lone verify, else a shared pass of that many windows (100 + n on the ranks).
const Log = struct {
    calls: [16]usize = undefined,
    n: usize = 0,
    fn add(l: *Log, v: usize) void {
        l.calls[l.n] = v;
        l.n += 1;
    }
};

const Owner = struct {
    const Lane = struct { seq: prompt.Seq = undefined, draft: ?mtp.Draft = @as(mtp.Draft, undefined) }; // a set optional: windows() unwraps it
    const Ranks = struct {
        log: *Log,
        pub fn verifyMany(r: *Ranks, comptime Tri: type, _: Tri, seqs: []const *prompt.Seq, ds: []const *mtp.Draft, idss: []const []const u32, _: u64) !void {
            try t.expect(seqs.len == ds.len and seqs.len == idss.len);
            r.log.add(100 + seqs.len);
        }
    };
    const Ops = struct {
        log: *Log,
        pub fn verify(o: Ops, _: *prompt.Seq, _: *mtp.Draft, _: []const u32) !void {
            o.log.add(0);
        }
        pub fn verifyMany(o: Ops, seqs: []const *prompt.Seq, _: []const *mtp.Draft, _: []const []const u32, _: u64) !void {
            o.log.add(seqs.len);
        }
        pub fn draws(_: Ops, _: *mtp.Draft, _: anytype, _: []const u32, _: []u32) !void {}
    };
    log: Log = .{},
    lanes_: [4]Lane = .{ .{}, .{}, .{}, .{} },
    streams: [4]lanes.Stream = undefined,
    ranks: ?*Ranks = null,
    logits: struct { ptr: u64 = 0x1000 } = .{},

    pub fn lane(o: *Owner, s: *lanes.Stream) !*Lane {
        for (&o.streams, 0..) |*x, i| if (x == s) return &o.lanes_[i];
        return error.UnknownStream;
    }
    pub fn tri(_: *Owner) u8 {
        return 0;
    }
    pub fn ops(o: *Owner) Ops {
        return .{ .log = &o.log };
    }
};

fn run(o: *Owner, rows: []const usize, cap: usize) !void {
    var wins: [4]be.Window = undefined;
    var out: [4]be.Verified = undefined;
    var sampled: [4][mtp.max_rows]u32 = undefined;
    var drafted: [4]usize = undefined;
    var row_ids: [4][mtp.max_rows]u32 = undefined;
    for (rows, 0..) |r, i| {
        o.streams[i].sampling = null;
        wins[i] = .{ .stream = &o.streams[i], .pending = 0, .held = @intCast(r - 1), .tokens = &.{}, .parents = null, .positions = &.{} };
        out[i] = .{ .sampled = &sampled[i], .drafts = &.{} };
        drafted[i] = i;
        row_ids[i] = @splat(1);
    }
    o.log = .{};
    try verify.windows(o, wins[0..rows.len], out[0..rows.len], drafted[0..rows.len], row_ids[0..rows.len], cap);
}

test "shared verify dispatch: at TP=2 a group goes to the ranks, a lone window and a split group verify alone" {
    var log: Log = .{};
    var ranks: Owner.Ranks = .{ .log = &log };
    var o: Owner = .{};
    o.ranks = &ranks;
    try run(&o, &.{ 2, 3, 2 }, 64);
    try t.expectEqual(@as(usize, 1), log.n);
    try t.expectEqual(@as(usize, 103), log.calls[0]);
    try t.expectEqual(@as(usize, 0), o.log.n);
    log = .{};
    try run(&o, &.{4}, 64);
    try t.expectEqual(@as(usize, 0), log.n);
    try t.expectEqualSlices(usize, &.{0}, o.log.calls[0..o.log.n]);
    // a cap of 4 rows splits two 3-row windows into two lone verifies
    try run(&o, &.{ 3, 3 }, 4);
    try t.expectEqual(@as(usize, 0), log.n);
    try t.expectEqualSlices(usize, &.{ 0, 0 }, o.log.calls[0..o.log.n]);
    // one rank: the shared pass stays on this engine
    o.ranks = null;
    try run(&o, &.{ 2, 3, 2 }, 64);
    try t.expectEqualSlices(usize, &.{3}, o.log.calls[0..o.log.n]);
}
