//! A stream handed to another driver between rounds and back, or opened by one: it replies as its one-token decode.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const rounds = @import("engine_test.zig");
const Case = rounds.Case;
const modelWith = rounds.modelWith;
const p1 = rounds.p1;
const p2 = rounds.p2;

const gpa = std.testing.allocator;

/// The fake target's one-token decode after `c.prompt`, to `c.max_new` tokens or an eos one: `s`'s reply and reason.
fn expectDecode(c: Case, s: *const sm.Stream) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, c.prompt);
    var reason: sm.Reason = .length;
    while (history.items.len - c.prompt.len < c.max_new) {
        const t = fake.next(history.items, c.sampling, history.items.len);
        try history.append(gpa, t);
        if (std.mem.indexOfScalar(u32, c.eos, t) != null) {
            reason = .stop;
            break;
        }
    }
    try std.testing.expectEqualSlices(u32, history.items[c.prompt.len..], s.emitted());
    try std.testing.expectEqual(reason, s.reason);
}

test "a lone stream handed to another driver between rounds and back replies as if kept, pipelined or windowed" {
    // a plain stream left in verify by a shared round, a windowed drafted one, a copy-drafted one left there by a drain
    const Set = struct { cases: []const Case, mtp: bool };
    const sets = [_]Set{
        .{ .cases = &.{ .{ .prompt = &p1, .drafts = false }, .{ .prompt = &p2, .max_new = 6 } }, .mtp = true },
        .{ .cases = &.{.{ .prompt = &p2 }}, .mtp = true },
        .{ .cases = &.{.{ .prompt = &p2, .max_new = 60 }}, .mtp = false },
    };
    for (sets) |set| {
        var cfg = try modelWith(set.mtp);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var proposers: [2]SuffixLookup = undefined;
        var streams: [2]sm.Stream = undefined;
        for (set.cases, proposers[0..set.cases.len], streams[0..set.cases.len]) |c, *p, *s| {
            p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
            s.* = try sm.Stream.init(gpa, .{
                .id = "s",
                .prompt = c.prompt,
                .max_new = c.max_new,
                .eos = c.eos,
                .drafts = c.drafts,
                .proposer = p.proposer(),
            });
        }
        defer for (proposers[0..set.cases.len], streams[0..set.cases.len]) |*p, *s| {
            s.deinit(gpa);
            p.deinit();
        };
        for (streams[0..set.cases.len]) |*s| try engine.addStream(s);
        var handed: usize = 0;
        var settle = false; // a round runs on what the adopted round leaves before the next hand-back
        while (engine.activeCount() > 0) {
            try engine.step();
            if (engine.live.items.len != 1) continue;
            if (settle) {
                settle = false;
                continue;
            }
            const s = engine.live.items[0];
            if (!engine.detach(s)) continue;
            handed += 1;
            try engine.adopt(s);
            settle = true;
        }
        try std.testing.expect(handed > 0);
        for (set.cases, streams[0..set.cases.len]) |c, *s| try expectDecode(c, s);
    }
}

test "a stream another driver opened and hands to the lane core replies as its one-token decode" {
    const c: Case = .{ .prompt = &p2 };
    for ([_]bool{ true, false }) |mtp| {
        var cfg = try modelWith(mtp);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        defer p.deinit();
        var s = try sm.Stream.init(gpa, .{
            .id = "s",
            .prompt = c.prompt,
            .max_new = c.max_new,
            .eos = c.eos,
            .proposer = p.proposer(),
        });
        defer s.deinit(gpa);
        // as a lone driver hands a stream over after its first token (lane_host_lone_test.zig's Lone)
        _ = try target.backend().opening(gpa, &s);
        s.pending = s.context.items[s.context.items.len - 1];
        s.cache_len = s.prompt_len;
        try engine.adopt(&s);
        while (engine.activeCount() > 0) try engine.step();
        try expectDecode(c, &s);
    }
}
