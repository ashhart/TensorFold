//! A --drafter server's logprobs: each token's row, drafting on or off, is the one the plain target gives.
const std = @import("std");
const lanes = @import("lanes");
const dc = @import("dflash2_config.zig");
const fixture = lanes.drafted_fixture;
const Row = lanes.LogprobRow;

const gpa = std.testing.allocator;

const Run = struct { tokens: []u32, rows: []Row, drafted: u64 };

/// A stream asking for 3 logprobs a token: through the host's wrapper with drafting `drafts`, or on the bare target.
fn run(wrapped: bool, drafts: bool, sampling: ?lanes.sampling.Sampling) !Run {
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: fixture.FakeDrafter = .{};
    defer head.deinit();
    var w = try dc.wrap(gpa, target.backend(), head.drafter());
    defer w.deinit();
    var cfg = try lanes.Config.init(gpa, if (wrapped) w.facts(fixture.base()) else fixture.base(), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, if (wrapped) w.backend() else target.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &fixture.p1, .max_new = 40, .eos = &.{96}, .sampling = sampling, .drafts = drafts, .logprobs = 3 });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqual(s.emitted().len, s.rows.items.len);
    return .{ .tokens = try gpa.dupe(u32, s.emitted()), .rows = try gpa.dupe(Row, s.rows.items), .drafted = s.drafted };
}

test "a drafter server answers logprobs with drafting on and off, row for row the plain target's" {
    for ([_]?lanes.sampling.Sampling{ null, fixture.sampled }) |sampling| {
        const plain = try run(false, false, sampling);
        defer gpa.free(plain.tokens);
        defer gpa.free(plain.rows);
        for ([_]bool{ true, false }) |drafts| {
            const got = try run(true, drafts, sampling);
            defer gpa.free(got.tokens);
            defer gpa.free(got.rows);
            try std.testing.expectEqual(drafts, got.drafted > 0);
            try std.testing.expectEqualSlices(u32, plain.tokens, got.tokens);
            for (plain.rows, got.rows) |p, d| {
                try std.testing.expectEqual(p.token, d.token);
                try std.testing.expectEqual(p.logprob, d.logprob);
                try std.testing.expectEqualSlices(u32, p.ids[0..p.count], d.ids[0..d.count]);
                try std.testing.expectEqualSlices(f32, p.logprobs[0..p.count], d.logprobs[0..d.count]);
            }
        }
    }
}
