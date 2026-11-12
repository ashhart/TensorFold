//! A wrapped target's logprob rows: offered only on request, and a drafted stream's rows are the plain stream's.
const std = @import("std");
const fixture = @import("drafted_test.zig");
const fake = @import("fake.zig");
const sm = @import("stream.zig");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const Drafted = @import("drafted.zig").Drafted;
const Row = @import("logprob.zig").Row;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

/// The stream's committed tokens and rows after `max_new` tokens, drafted (a wrapper over the fake target) or plain.
const Run = struct { tokens: []u32, rows: []Row, drafted: u64 };

fn rowsOf(wrapped: bool, sampling: ?Sampling) !Run {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: fixture.FakeDrafter = .{};
    defer head.deinit();
    var w = try Drafted.init(gpa, target.backend(), head.drafter());
    defer w.deinit();
    w.rows = true;
    var cfg = try Config.init(gpa, if (wrapped) w.facts(fixture.base()) else fixture.base(), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, if (wrapped) w.backend() else target.backend(), clock.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(gpa, .{
        .id = "s",
        .prompt = &fixture.p1,
        .max_new = 40,
        .eos = &.{96},
        .sampling = sampling,
        .drafts = wrapped,
        .logprobs = 3,
    });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqual(s.emitted().len, s.rows.items.len);
    const tokens = try gpa.dupe(u32, s.emitted());
    return .{ .tokens = tokens, .rows = try gpa.dupe(Row, s.rows.items), .drafted = s.drafted };
}

test "a wrapper offers the target's first row only when asked" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: fixture.FakeDrafter = .{};
    defer head.deinit();
    var w = try Drafted.init(gpa, target.backend(), head.drafter());
    defer w.deinit();
    try std.testing.expect(target.backend().vtable.first_row != null);
    try std.testing.expect(w.backend().vtable.first_row == null);
    w.rows = true;
    try std.testing.expect(w.backend().vtable.first_row != null);
}

test "drafted rounds give the rows of one-token rounds, greedy and sampled" {
    for ([_]?Sampling{ null, fixture.sampled }) |sampling| {
        const drafted = try rowsOf(true, sampling);
        defer gpa.free(drafted.tokens);
        defer gpa.free(drafted.rows);
        const plain = try rowsOf(false, sampling);
        defer gpa.free(plain.tokens);
        defer gpa.free(plain.rows);
        try std.testing.expect(drafted.drafted > 0);
        try std.testing.expectEqualSlices(u32, plain.tokens, drafted.tokens);
        for (plain.rows, drafted.rows, plain.tokens) |p, d, t| {
            try std.testing.expectEqual(t, d.token);
            try std.testing.expectEqual(p.token, d.token);
            try std.testing.expectEqual(p.logprob, d.logprob);
            try std.testing.expectEqualSlices(u32, p.ids[0..p.count], d.ids[0..d.count]);
            try std.testing.expectEqualSlices(f32, p.logprobs[0..p.count], d.logprobs[0..d.count]);
        }
    }
}
