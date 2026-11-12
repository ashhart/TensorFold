//! Native Nemotron copy policies exercised through the real lane planner and an instrumented target.
const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
const be = lanes.backend;
fn facts(enabled: bool, head: bool) lanes.Model {
    const n = @import("nemotron");
    var e: n.Engine = undefined;
    e.c.latent = 0;
    e.c.bolt = false;
    var b: n.Lanes = undefined;
    b.e = &e;
    b.head = if (head) @ptrFromInt(16) else null;
    b.copy_drafts = enabled;
    b.mtp_drafts = null;
    b.mtp_confidence = null;
    b.cost_count = 0;
    b.mtp_ms = 0;
    return b.facts();
}

fn old(m: lanes.Model) lanes.Model {
    var out = m;
    out.root_copies = false;
    out.context_copies = false;
    out.copy_match = 8;
    return out;
}

test "Nemotron copies: actual family facts gate copies and retain the drafts-off model" {
    for ([_]bool{ false, true }) |on| {
        const m = facts(on, true);
        var cfg = try lanes.Config.init(a, m, m.exact_width, m.exact_width - 1);
        defer cfg.deinit(a);
        try std.testing.expectEqual(on, cfg.root_copies);
        try std.testing.expectEqual(on, cfg.context_copies);
        try std.testing.expectEqual(@as(i64, if (on) 4 else 8), cfg.enter_match);
        if (on) try std.testing.expect(cfg.max_copy > 1) else try std.testing.expectEqualDeep(old(m), m);
    }
    const bolt = facts(true, false);
    try std.testing.expect(bolt.root_copies and bolt.context_copies and !bolt.mtp);
}

const Copy = struct {
    matched: i64,
    fn propose(_: *anyopaque, _: []const u32, most: i64) ![]const u32 {
        const ids = &[_]u32{ 11, 12, 13, 14 };
        return ids[0..@intCast(@min(4, most))];
    }
    fn match(p: *anyopaque) i64 {
        return @as(*Copy, @ptrCast(@alignCast(p))).matched;
    }
};

test "Nemotron copies: three matches refuse, four enter, disabled copies and miss refuse" {
    const m = facts(true, true);
    var cfg = try lanes.Config.init(a, m, m.exact_width, m.exact_width - 1);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = true;
    var copy: Copy = .{ .matched = 3 };
    var s = try lanes.Stream.init(a, .{ .id = "gate", .prompt = &.{ 1, 2, 3, 4 }, .max_new = 64, .proposer = .{ .ptr = &copy, .vtable = &.{ .propose = Copy.propose, .last_match = Copy.match } } });
    defer s.deinit(a);
    try std.testing.expectEqual(@as(usize, 0), (try lanes.windows.copyProposal(&e, &s)).tokens.len);
    copy.matched = 4;
    try std.testing.expect((try lanes.windows.copyProposal(&e, &s)).tokens.len > 0);
    s.root_copy_miss = true;
    try std.testing.expectEqual(@as(usize, 0), (try lanes.windows.copyProposal(&e, &s)).tokens.len);
    s.root_copy_miss = false;
    cfg.max_copy = 0;
    try std.testing.expectEqual(@as(usize, 0), (try lanes.windows.copyProposal(&e, &s)).tokens.len);
}

const Stats = struct { zeros: usize = 0, heads: usize = 0, copied: usize = 0, rejected: usize = 0, trace: u64 = 0 };
threadlocal var current: ?*Stats = null;

fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
    for (requests) |r| {
        if (r.depth == 0) current.?.zeros += 1 else current.?.heads += 1;
        current.?.trace = std.hash.Wyhash.hash(current.?.trace, std.mem.asBytes(&r.depth));
    }
    const t: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
    try t.backend().vtable.draft(ptr, requests);
}

fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
    const t: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
    try t.backend().vtable.verify(ptr, windows, out);
    for (windows, out) |w, o| {
        current.?.trace = std.hash.Wyhash.hash(current.?.trace, std.mem.sliceAsBytes(w.positions));
        current.?.trace = std.hash.Wyhash.hash(current.?.trace, std.mem.sliceAsBytes(w.tokens));
        if (w.held == 0 and w.tokens.len > 0) {
            current.?.copied += 1;
            if (!std.mem.eql(u32, w.tokens, o.sampled[0..w.tokens.len])) current.?.rejected += 1;
        }
    }
}

const Result = struct { tokens: [4][64]u32 = undefined, stats: Stats = .{} };

fn run(m: lanes.Model, count: usize, sampled: bool, drafted: bool, breaks: bool, restored: bool) !Result {
    var out: Result = .{};
    current = &out.stats;
    defer current = null;
    var prompt: [80]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(11 + i % 4);
    const pattern = if (breaks) &[_]u32{ 11, 12, 13, 14, 11, 12, 13, 14, 91 } else &[_]u32{ 11, 12, 13, 14 };
    var target: lanes.fake.Fake = .{ .gpa = a, .probe_prompt = prompt.len, .cycle_after = 0, .pattern = pattern };
    defer target.deinit();
    var backend = target.backend();
    var vt = backend.vtable.*;
    vt.draft = draft;
    vt.verify = verify;
    backend.vtable = &vt;
    var cfg = try lanes.Config.init(a, m, m.exact_width, m.exact_width - 1);
    defer cfg.deinit(a);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    var streams: [4]lanes.Stream = undefined;
    var proposers: [4]lanes.SuffixLookup = undefined;
    var made: usize = 0;
    defer for (streams[0..made], proposers[0..made]) |*s, *p| {
        s.deinit(a);
        p.deinit();
    };
    var saved: ?*anyopaque = null;
    defer if (saved) |p| target.drop(p);
    if (restored) {
        var early = try lanes.Stream.init(a, .{ .id = "prefix", .prompt = prompt[0..32], .max_new = 1 });
        defer early.deinit(a);
        try backend.prefill(&early);
        saved = try target.save(&early, 32);
        backend.release(&early);
    }
    for (streams[0..count], proposers[0..count]) |*s, *p| {
        p.* = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
        s.* = try lanes.Stream.init(a, .{ .id = "copy", .prompt = &prompt, .max_new = 64, .drafts = drafted, .proposer = p.proposer(), .sampling = if (sampled) .{ .seed = 19, .temperature = 0.7 } else null, .reuse = .{ .saved = saved, .at = if (restored) 32 else 0 } });
        made += 1;
        try e.addStream(s);
    }
    var steps: usize = 0;
    while (e.activeCount() > 0 and steps < 512) : (steps += 1) try e.step();
    try std.testing.expectEqual(@as(usize, 0), e.activeCount());
    for (streams[0..count], out.tokens[0..count]) |*s, *ids| {
        try std.testing.expectEqual(@as(usize, 64), s.emitted().len);
        @memcpy(ids, s.emitted());
    }
    return out;
}

test "Nemotron copies: real core copies equal plain tokens solo shared sampled and restored" {
    for ([_]bool{ false, true }) |sampled| for ([_]bool{ false, true }) |restored| for ([_]bool{ false, true }) |breaks| {
        const m = facts(true, true);
        const plain = try run(m, 1, sampled, false, breaks, restored);
        for ([_]usize{ 1, 4 }) |count| {
            const got = try run(m, count, sampled, true, breaks, restored);
            for (got.tokens[0..count]) |ids| try std.testing.expectEqualSlices(u32, &plain.tokens[0], &ids);
            try std.testing.expect(got.stats.copied > 0);
            try std.testing.expect(got.stats.zeros > 0);
            if (breaks) {
                if (count == 1) try std.testing.expect(got.stats.rejected > 0);
                try std.testing.expect(got.stats.heads > 0);
            }
            const before = try run(old(m), count, sampled, false, breaks, restored);
            const after = try run(m, count, sampled, false, breaks, restored);
            try std.testing.expectEqualDeep(before, after);
        }
    };
}

test "Nemotron copies: cancel a copied round and the next request equals fresh plain" {
    const m = facts(true, true);
    const plain = try run(m, 1, false, false, false, false);
    var cfg = try lanes.Config.init(a, m, m.exact_width, m.exact_width - 1);
    defer cfg.deinit(a);
    cfg.timed = false;
    var prompt: [80]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(11 + i % 4);
    var target: lanes.fake.Fake = .{ .gpa = a, .probe_prompt = prompt.len, .cycle_after = 0, .pattern = &.{ 11, 12, 13, 14 } };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    for (0..2) |request| {
        var suffix = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
        defer suffix.deinit();
        var s = try lanes.Stream.init(a, .{ .id = "cancel-next", .prompt = &prompt, .max_new = 64, .proposer = suffix.proposer() });
        defer s.deinit(a);
        try e.addStream(&s);
        try e.step();
        try std.testing.expect(e.drafted > 0);
        if (request == 0) {
            e.discard(&s);
            try std.testing.expectEqual(lanes.stream.Reason.cancelled, s.reason);
        } else {
            while (e.activeCount() > 0) try e.step();
            try std.testing.expectEqualSlices(u32, &plain.tokens[0], s.emitted());
        }
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}
