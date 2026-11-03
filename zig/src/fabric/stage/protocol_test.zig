//! The stage protocol's messages round-trip, every mismatch is refused by its name, and no input makes a decoder panic.
const std = @import("std");
const payload = @import("payload.zig");
const boot = @import("boot.zig");
const plan = @import("plan.zig");
const frame = @import("frame.zig");
const fixtures = @import("fixtures.zig");
const t = std.testing;

test "stage protocol: a description round-trips, from no taps to sixteen and a 64-byte version" {
    var buf: [512]u8 = undefined;
    for (0..2) |rank| {
        const d = fixtures.description(@intCast(rank));
        const back = try boot.decodeDescription(try boot.encodeDescription(&d, &buf));
        try t.expect(std.meta.eql(d, back));
        try boot.check(fixtures.expects[rank], &back);
    }
    var ids: [boot.most_taps]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(100 + 3 * i);
    var d = fixtures.description(1);
    d.range = .{ .first = 100, .end = 200 };
    d.taps = try boot.Taps.init(&ids);
    const vs: [boot.most_version + 1]u8 = @splat('v');
    d.version = try boot.Text.init(vs[0..boot.most_version]);
    d.kv_rows = std.math.maxInt(u64);
    const p = try boot.encodeDescription(&d, &buf);
    try t.expectEqual(boot.describe_fixed + 4 * 16 + 64, p.len);
    try t.expect(std.meta.eql(d, try boot.decodeDescription(p)));
    d.taps = .{};
    try t.expect(std.meta.eql(d, try boot.decodeDescription(try boot.encodeDescription(&d, &buf))));
    try t.expectError(error.NoRoom, boot.encodeDescription(&d, buf[0 .. d.size() - 1]));
    try t.expectError(error.BadVersionText, boot.Text.init(""));
    try t.expectError(error.BadVersionText, boot.Text.init(&vs));
    try t.expectError(error.BadTaps, boot.Taps.init(&.{ 5, 5 }));
    try t.expectError(error.BadTaps, boot.Taps.init(&(ids ++ [_]u32{999})));
}

test "stage protocol: each mismatch against rank 0's plan is refused by its own name" {
    const Case = struct { err: anyerror, edit: *const fn (*boot.Description) void };
    const cases = [_]Case{
        .{ .err = error.OtherRank, .edit = struct {
            fn f(d: *boot.Description) void {
                d.rank = 0;
            }
        }.f },
        .{ .err = error.OtherVersion, .edit = struct {
            fn f(d: *boot.Description) void {
                d.version = boot.Text.init("tensorfold stage-v0 tes") catch unreachable;
            }
        }.f },
        .{ .err = error.OtherBackend, .edit = struct {
            fn f(d: *boot.Description) void {
                d.backend = .cuda;
            }
        }.f },
        .{ .err = error.OtherRange, .edit = struct {
            fn f(d: *boot.Description) void {
                d.range.end = 7;
            }
        }.f },
        .{ .err = error.OtherShapes, .edit = struct {
            fn f(d: *boot.Description) void {
                d.shapes.prompt_chunk = 8;
            }
        }.f },
        .{ .err = error.OtherShapes, .edit = struct {
            fn f(d: *boot.Description) void {
                d.shapes.parallel = 3;
            }
        }.f },
        .{ .err = error.OtherTaps, .edit = struct {
            fn f(d: *boot.Description) void {
                d.taps = .{};
            }
        }.f },
        .{ .err = error.OtherTaps, .edit = struct {
            fn f(d: *boot.Description) void {
                d.taps = boot.Taps.init(&.{ 5, 6 }) catch unreachable;
            }
        }.f },
        .{ .err = error.ShortCapacity, .edit = struct {
            fn f(d: *boot.Description) void {
                d.states = 3;
            }
        }.f },
    };
    var buf: [512]u8 = undefined;
    for (cases) |c| {
        var d = fixtures.description(1);
        c.edit(&d);
        const back = try boot.decodeDescription(try boot.encodeDescription(&d, &buf));
        try t.expectError(c.err, boot.check(fixtures.expects[1], &back));
    }
}

test "stage protocol: a layout must run from layer 0 to the last, contiguous, none empty or overlapping" {
    const R = boot.Range;
    try boot.checkLayout(8, &.{ R{ .first = 0, .end = 4 }, R{ .first = 4, .end = 8 } });
    try boot.checkLayout(8, &.{ R{ .first = 0, .end = 1 }, R{ .first = 1, .end = 7 }, R{ .first = 7, .end = 8 } });
    try t.expectError(error.BadLayout, boot.checkLayout(8, &.{}));
    const Case = struct { err: anyerror, a: R, b: R };
    const cases = [_]Case{
        .{ .err = error.RangeGap, .a = .{ .first = 1, .end = 4 }, .b = .{ .first = 4, .end = 8 } },
        .{ .err = error.RangeGap, .a = .{ .first = 0, .end = 3 }, .b = .{ .first = 4, .end = 8 } },
        .{ .err = error.RangeOverlap, .a = .{ .first = 0, .end = 5 }, .b = .{ .first = 4, .end = 8 } },
        .{ .err = error.EmptyRange, .a = .{ .first = 0, .end = 4 }, .b = .{ .first = 4, .end = 4 } },
        .{ .err = error.RangeShort, .a = .{ .first = 0, .end = 4 }, .b = .{ .first = 4, .end = 7 } },
        .{ .err = error.RangeBeyond, .a = .{ .first = 0, .end = 4 }, .b = .{ .first = 4, .end = 9 } },
    };
    for (cases) |c| try t.expectError(c.err, boot.checkLayout(8, &.{ c.a, c.b }));
    const descs = [_]boot.Description{ fixtures.description(0), fixtures.description(1) };
    const tight = boot.tightest(&descs);
    try t.expectEqual(@as(u64, 128), tight.kv_rows);
    try t.expectEqual(@as(u32, 4), tight.states);
}

test "stage protocol: a ready report round-trips and is ready only when loaded and warm" {
    var buf: [boot.ready_size]u8 = undefined;
    for ([_]bool{ false, true }) |loaded| for ([_]bool{ false, true }) |warm| {
        const r: boot.Ready = .{ .rank = 7, .loaded = loaded, .warm = warm };
        const back = try boot.decodeReady(try boot.encodeReady(r, &buf));
        try t.expectEqual(r, back);
        try t.expectEqual(loaded and warm, back.ready());
    };
    try t.expectError(error.NoRoom, boot.encodeReady(.{ .rank = 0, .loaded = true, .warm = true }, buf[0..15]));
}

test "stage protocol: a plan of every op round-trips, and an empty plan is the keep-alive" {
    var buf: [plan.planSize(8, 4)]u8 = undefined;
    const lim: plan.Limits = .{ .streams = 8, .rows = 8, .context = 64 };
    const ops = [_]plan.Stream{
        .{ .id = 1, .op = .open, .first = 0, .rows = 3 },
        .{ .id = 2, .op = .@"continue", .first = 9, .rows = 2, .accepted = 1 },
        .{ .id = 4, .op = .close, .first = 11, .rows = 0, .accepted = 3 },
        .{ .id = 8, .op = .cancel, .first = 5, .rows = 0 },
        .{ .id = 0xffff_ffff, .op = .@"resume", .first = 60, .rows = 3, .resume_at = 48 },
    };
    const p = try plan.decode(try plan.encode(&ops, &.{ 3, 5 }, lim, &buf), lim);
    try t.expectEqual(@as(u32, 5), p.streams);
    try t.expectEqual(@as(u32, 8), p.rows);
    for (ops, 0..) |s, i| try t.expectEqual(s, p.stream(i));
    try t.expectEqual(@as(u32, 3), p.cut(0));
    try t.expectEqual(@as(u32, 5), p.cut(1));
    try t.expect(!p.keepAlive());
    const k = try plan.decode(try plan.encode(&.{}, &.{}, lim, &buf), lim);
    try t.expect(k.keepAlive());
    try t.expectEqual(@as(usize, plan.plan_fixed), k.bytes.len);
    const two = [_]plan.Stream{ ops[0], ops[0] };
    try t.expectError(error.UnorderedStreams, plan.encode(&two, &.{}, lim, &buf));
    try t.expectError(error.TooManyStreams, plan.encode(&(ops ++ ops), &.{}, lim, &buf));
    try t.expectError(error.NoRoom, plan.encode(&ops, &.{}, lim, buf[0 .. plan.planSize(5, 0) - 1]));
}

test "stage protocol: a count that claims more than the payload holds is refused before any read" {
    var buf: [plan.planSize(4, 1)]u8 = undefined;
    const p = try plan.write(&fixtures.streams, &fixtures.cuts, &buf);
    for ([_]u32{ 5, 0xffff_ffff }) |n| {
        var bad = buf;
        payload.put32(&bad, 8, n);
        try t.expectError(error.BadLength, plan.decode(bad[0..p.len], fixtures.limits));
        bad = buf;
        payload.put32(&bad, 12, n);
        try t.expectError(error.BadLength, plan.decode(bad[0..p.len], fixtures.limits));
    }
    var big: [plan.planSize(5, 0)]u8 = undefined;
    var five: [5]plan.Stream = undefined;
    for (&five, 0..) |*s, i| s.* = .{ .id = @intCast(i), .op = .@"continue", .first = 0, .rows = 0xffff_ffff };
    const wide: plan.Limits = .{ .streams = 4, .rows = 8, .context = 0xffff_ffff };
    try t.expectError(error.TooManyRows, plan.decode(try plan.write(five[0..4], &.{}, &big), wide));
    try t.expectError(error.PastContext, plan.decode(try plan.write(five[0..4], &.{}, &big), fixtures.limits));
}

test "stage protocol: the rows layout is plane-major, rows in plan order, then the tap sums in fp32" {
    const r = try plan.Rows.of(fixtures.model.boundary, 7);
    try t.expectEqual(@as(u64, 8), r.row_bytes);
    try t.expectEqual(@as(u64, 0), r.rowOffset(0, 0));
    try t.expectEqual(@as(u64, 48), r.rowOffset(0, 6));
    try t.expectEqual(@as(u64, 56), r.rowOffset(1, 0));
    try t.expectEqual(@as(u64, 112), r.tapsOffset());
    try t.expectEqual(@as(u64, 120), r.tapOffset(1));
    try t.expectEqual(@as(u64, 168), r.total());
    var b = fixtures.model.boundary;
    b.draft_width = 0;
    try t.expectEqual(@as(u64, 112), (try plan.Rows.of(b, 7)).total());
    b.row_dtype = .nvfp4;
    try t.expectError(error.BadRowDtype, plan.Rows.of(b, 7));
    const max = 0xffff_ffff;
    const huge: boot.Boundary = .{ .hidden = max, .row_dtype = .f32, .planes = 255, .draft_width = max };
    try t.expectError(error.TooLarge, plan.Rows.of(huge, 0xffff_ffff));
    var buf: [plan.planSize(4, 1)]u8 = undefined;
    const p = try plan.decode(try plan.write(&fixtures.streams, &fixtures.cuts, &buf), fixtures.limits);
    const h: frame.Header = .{ .kind = .rows, .round = 1, .rows = 7, .bytes = 168 };
    _ = try plan.checkRows(h, p, fixtures.model.boundary);
    var bad = h;
    bad.kind = .plan;
    try t.expectError(error.WrongKind, plan.checkRows(bad, p, fixtures.model.boundary));
    bad = h;
    bad.rows = 8;
    try t.expectError(error.OtherRowCount, plan.checkRows(bad, p, fixtures.model.boundary));
    bad = h;
    bad.bytes = 112;
    try t.expectError(error.OtherRowBytes, plan.checkRows(bad, p, fixtures.model.boundary));
}

/// Every prefix of `good` shorter than it, and `n` random edits of it, decoded by `judge` without a panic.
fn hammer(kind: fixtures.Kind, good: []const u8, n: usize) !void {
    for (0..good.len) |len| try t.expect(std.meta.isError(fixtures.judge(kind, good[0..len])));
    var prng = std.Random.DefaultPrng.init(0x5446_5350);
    const rnd = prng.random();
    var buf: [512]u8 = undefined;
    for (0..n) |_| {
        const len = rnd.intRangeAtMost(usize, 0, @min(good.len + 8, buf.len));
        for (buf[0..len], 0..) |*b, i| b.* = if (i < good.len) good[i] else rnd.int(u8);
        for (0..rnd.intRangeAtMost(usize, 1, 4)) |_| if (len > 0) {
            buf[rnd.uintLessThan(usize, len)] = rnd.int(u8);
        };
        fixtures.judge(kind, buf[0..len]) catch {};
    }
}

test "stage protocol: truncated and randomly edited messages are refused, never a panic" {
    inline for (fixtures.all) |f| if (f.want == null) {
        var buf: [512]u8 = undefined;
        try hammer(f.kind, try f.build(&buf), 2000);
    };
}
