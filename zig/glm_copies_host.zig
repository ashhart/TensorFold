//! Actual GLM facts on the shared round loop, with recording target windows and the production peer codec.
const std = @import("std");
const glm = @import("glm");
const lanes = @import("lanes");
const wire = glm.verify_wire;
const be = lanes.backend;
const a = std.testing.allocator;
const t = std.testing;
const reply = 96;

fn facts(head: bool, copies: bool) lanes.Model {
    var engine: glm.engine.Engine = undefined;
    engine.mtp = head;
    engine.drafter = null;
    var back: glm.lanes.Lanes = undefined;
    back.x.e = &engine;
    back.alone = false;
    back.mtp_confidence = null;
    back.copy_drafts = copies;
    return back.facts();
}

const Stats = struct { copies: usize = 0, matched: usize = 0, rejected: usize = 0, absorb: usize = 0, absorbed_rows: usize = 0, neural: usize = 0, peer_windows: usize = 0 };
const Target = struct {
    inner: lanes.fake.Fake = .{ .gpa = a, .cycle_after = 0, .probe_prompt = 120 },
    table: be.Backend.VTable = undefined,
    streams: [4]?*lanes.Stream = @splat(null),
    stats: Stats = .{},

    fn backend(x: *Target) be.Backend {
        x.table = x.inner.backend().vtable.*;
        x.table.verify = verify;
        x.table.draft = draft;
        return .{ .ptr = &x.inner, .vtable = &x.table };
    }
    fn of(ptr: *anyopaque) *Target {
        return @fieldParentPtr("inner", @as(*lanes.fake.Fake, @ptrCast(@alignCast(ptr))));
    }
    fn slot(x: *Target, stream: *lanes.Stream) u32 {
        for (&x.streams, 0..) |*s, i| {
            if (s.* == null) s.* = stream;
            if (s.* == stream) return @intCast(i);
        }
        unreachable;
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const x = of(ptr);
        try x.inner.backend().vtable.verify(ptr, windows, out);
        var ids: [4][glm.lanes.max_rows]u32 = undefined;
        var parts: [4]wire.Part = undefined;
        for (windows, out, 0..) |w, v, i| {
            const words = try wire.window(w.pending, v.drafts[0 .. w.rows() - 1], &ids[i]);
            parts[i] = .{ .slot = x.slot(w.stream), .ids = words };
            if (w.tokens.len > 0) {
                x.stats.copies += 1;
                for (w.tokens, v.sampled[0..w.tokens.len]) |p, expected| {
                    if (p != expected) {
                        x.stats.rejected += 1;
                        break;
                    }
                    x.stats.matched += 1;
                }
            }
        }
        var encoded: [2 * glm.engine.max_shared + glm.engine.shared_rows]u32 = undefined;
        const words = try wire.shared(parts[0..windows.len], &encoded, glm.engine.max_shared, glm.engine.shared_rows);
        for ([_]usize{ 1, 2 }) |world| for (0..world) |_| {
            var replay: [glm.engine.max_shared]wire.Part = undefined;
            const got = try wire.following(words, windows.len, &replay, glm.engine.shared_rows);
            try t.expect(same(parts[0..windows.len], got));
            x.stats.peer_windows += got.len;
        };
    }
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const x = of(ptr);
        for (requests) |r| {
            if (r.depth == 0) {
                x.stats.absorb += 1;
                if (r.rows) |rows| x.stats.absorbed_rows += rows.len;
            } else x.stats.neural += 1;
        }
        try x.inner.backend().vtable.draft(ptr, requests);
    }
};

fn same(left: []const wire.Part, right: []const wire.Part) bool {
    if (left.len != right.len) return false;
    for (left, right) |l, r| if (l.slot != r.slot or !std.mem.eql(u32, l.ids, r.ids)) return false;
    return true;
}

const Wrong = struct {
    ids: [15]u32 = @splat(95),
    fn propose(ptr: *anyopaque, _: []const u32, most: i64) ![]const u32 {
        const x: *Wrong = @ptrCast(@alignCast(ptr));
        return x.ids[0..@intCast(@max(@as(i64, 0), @min(most, @as(i64, x.ids.len))))];
    }
    fn after(ptr: *anyopaque, history: []const u32, _: []const u32, most: i64) ![]const u32 {
        return propose(ptr, history, most);
    }
    fn matched(_: *anyopaque) i64 {
        return 100;
    }
    fn proposer(x: *Wrong) lanes.proposer.Proposer {
        return .{ .ptr = x, .vtable = &.{ .propose = propose, .propose_after = after, .last_match = matched } };
    }
};
const Result = struct { ids: [4][reply]u32 = undefined, stats: Stats = .{} };
fn run(head: bool, copies: bool, drafted: bool, n: usize, wrong: bool, mixed: bool, sampled: bool) !Result {
    var config = try lanes.Config.init(a, facts(head, copies), glm.lanes.max_rows, glm.lanes.max_rows - 1);
    defer config.deinit(a);
    var target: Target = .{};
    if (sampled) target.inner.cycle_after = null;
    defer target.inner.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &config, target.backend(), clock.clock());
    defer core.deinit();
    var prompt: [120]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(11 + i % 3);
    var streams: [4]lanes.Stream = undefined;
    var suffix: [4]lanes.SuffixLookup = undefined;
    var bad: [4]Wrong = @splat(.{});
    var started: usize = 0;
    defer for (streams[0..started], suffix[0..started]) |*s, *p| {
        s.deinit(a);
        p.deinit();
    };
    for (streams[0..n], suffix[0..n], bad[0..n], 0..) |*s, *p, *b, i| {
        p.* = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
        s.* = try lanes.Stream.init(a, .{ .id = "copy", .prompt = &prompt, .max_new = reply, .eos = &.{}, .drafts = drafted and !(mixed and i % 2 == 1), .proposer = if (wrong) b.proposer() else p.proposer(), .sampling = if (sampled) .{ .seed = 73, .temperature = 0.8 } else null });
        started += 1;
        try core.addStream(s);
    }
    var rounds: usize = 0;
    while (core.activeCount() > 0) : (rounds += 1) {
        if (rounds > 1000) return error.NoProgress;
        try core.step();
    }
    var result: Result = .{ .stats = target.stats };
    for (streams[0..n], result.ids[0..n]) |*s, *ids| {
        try t.expectEqual(@as(usize, reply), s.emitted().len);
        @memcpy(ids, s.emitted());
    }
    return result;
}

test "GLM copies: real facts enable root and context copies independently of the head" {
    for ([_]bool{ false, true }) |head| {
        var config = try lanes.Config.init(a, facts(head, true), glm.lanes.max_rows, glm.lanes.max_rows - 1);
        defer config.deinit(a);
        try t.expect(config.root_copies and config.context_copies);
        try t.expectEqual(@as(u32, glm.lanes.max_rows - 1), config.max_copy);
        try t.expectEqual(head, config.family_mtp);
    }
    var off = try lanes.Config.init(a, facts(false, false), glm.lanes.max_rows, glm.lanes.max_rows - 1);
    defer off.deinit(a);
    try t.expectEqual(@as(u32, 0), off.max_copy);
    try t.expect(!off.root_copies and !off.context_copies and !off.family_mtp);
}

test "GLM copies: actual planner verifies suffix copies and matches plain at TP1 and TP2, alone and concurrent" {
    for ([_]bool{ false, true }) |head| for ([_]usize{ 1, 4 }) |n| {
        const plain = try run(head, true, false, n, false, false, false);
        const drafted = try run(head, true, true, n, false, false, false);
        try t.expect(drafted.stats.copies > 0 and drafted.stats.matched > 0);
        try t.expect(drafted.stats.peer_windows > 0);
        try t.expectEqual(@as(usize, 0), plain.stats.copies);
        if (head) try t.expect(drafted.stats.absorb > 0 and drafted.stats.absorbed_rows > 0);
        for (plain.ids[0..n], drafted.ids[0..n]) |p, d| try t.expectEqualSlices(u32, &p, &d);
        const solo = try run(head, true, true, 1, false, false, false);
        for (drafted.ids[0..n]) |d| try t.expectEqualSlices(u32, &solo.ids[0], &d);
    };
}

test "GLM copies: wrong proposals reject, neural drafting resumes, request and global off stay plain" {
    const plain = try run(true, true, false, 4, false, false, false);
    const bad = try run(true, true, true, 4, true, false, false);
    try t.expect(bad.stats.copies > 0 and bad.stats.rejected > 0 and bad.stats.neural > 0);
    const mixed = try run(true, true, true, 4, false, true, false);
    const off = try run(false, false, true, 4, false, false, false);
    try t.expectEqual(@as(usize, 0), off.stats.copies);
    for (plain.ids, bad.ids, mixed.ids, off.ids) |p, b, m, o| {
        try t.expectEqualSlices(u32, &p, &b);
        try t.expectEqualSlices(u32, &p, &m);
        try t.expectEqualSlices(u32, &p, &o);
    }
    const seeded_plain = try run(true, true, false, 4, false, false, true);
    const seeded_draft = try run(true, true, true, 4, false, false, true);
    const seeded_bad = try run(true, true, true, 4, true, false, true);
    for (seeded_plain.ids, seeded_draft.ids, seeded_bad.ids) |p, d, b| {
        try t.expectEqualSlices(u32, &p, &d);
        try t.expectEqualSlices(u32, &p, &b);
    }
}

test "GLM copies: production codec preserves max windows and catches altered proposals or malformed frames" {
    var ids: [16]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i + 1);
    const parts = [_]wire.Part{ .{ .slot = 7, .ids = &ids }, .{ .slot = 2, .ids = &ids } };
    var words: [64]u32 = undefined;
    const encoded = try wire.shared(&parts, &words, glm.engine.max_shared, glm.engine.shared_rows);
    var decoded: [16]wire.Part = undefined;
    try t.expect(same(&parts, try wire.following(encoded, 2, &decoded, glm.engine.shared_rows)));
    words[0] ^= 1;
    try t.expect(!same(&parts, try wire.following(encoded, 2, &decoded, glm.engine.shared_rows)));
    words[0] ^= 1;
    words[4] ^= 1;
    try t.expect(!same(&parts, try wire.following(encoded, 2, &decoded, glm.engine.shared_rows)));
    words[4] ^= 1;
    try t.expectError(error.BadStep, wire.following(encoded[0 .. encoded.len - 1], 2, &decoded, glm.engine.shared_rows));
    try t.expectError(error.BadStep, wire.following(encoded, 0, &decoded, glm.engine.shared_rows));
    try t.expectError(error.WindowTooWide, wire.shared(&parts, &words, glm.engine.max_shared, 31));
    try t.expectError(error.WindowTooWide, wire.window(0, &ids, ids[0..16]));
}

test "GLM copies: the host's four-token suffix match reaches verification instead of the inherited eight gate" {
    var config = try lanes.Config.init(a, facts(false, true), glm.lanes.max_rows, glm.lanes.max_rows - 1);
    defer config.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &config, target.backend(), clock.clock());
    defer core.deinit();
    var suffix = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
    defer suffix.deinit();
    var stream = try lanes.Stream.init(a, .{ .id = "short-match", .prompt = &.{ 1, 2, 3, 4, 5, 6, 7, 90, 1, 2, 3, 4 }, .max_new = 32, .proposer = suffix.proposer() });
    defer stream.deinit(a);
    try t.expectEqualSlices(u32, &.{ 5, 6, 7 }, (try lanes.windows.copyProposal(&core, &stream)).tokens);
    try t.expectEqual(@as(i64, 4), suffix.last_match);
    config.enter_match = 8;
    try t.expectEqual(@as(usize, 0), (try lanes.windows.copyProposal(&core, &stream)).tokens.len);
}
