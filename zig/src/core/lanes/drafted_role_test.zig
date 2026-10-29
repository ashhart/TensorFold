//! The production wrapper distinguishes prompt ingestion from accepted live rows.
const std = @import("std");
const fixture = @import("drafted_test.zig");
const Sampling = @import("sampling.zig").Sampling;
const p1 = fixture.p1;
const sampled = fixture.sampled;
const run = fixture.run;
const freeOut = fixture.freeOut;
const gpa = std.testing.allocator;
const fake = @import("fake.zig");
const sm = @import("stream.zig");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const Drafted = @import("drafted.zig").Drafted;
const FakeDrafter = fixture.FakeDrafter;
const base = fixture.base;

test "an external drafter's rounds commit the one-token decode, greedy and sampled, and its drafts land" {
    for ([_]?Sampling{ null, sampled }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }}, .wrapped, false);
        defer freeOut(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, .plain, false);
        defer freeOut(plain);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], drafted.tokens[0]);
        // four in five of its drafts are the target's, so far more than chance (1 in 97) land
        try std.testing.expect(drafted.drafted > 0 and drafted.accepted * 3 > drafted.drafted);
        try std.testing.expect(drafted.head.rows >= p1.len - 1 + drafted.tokens[0].len - 1);
        try std.testing.expectEqual(@as(u64, p1.len), drafted.head.prompt_rows);
        try std.testing.expect(drafted.head.live_rows > 0);
    }
}

test "the drafter wrapper sees the committed callback before its first hold" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var seen = false;
    var head: FakeDrafter = .{ .observe_first = &seen };
    defer head.deinit();
    var backend = target.backend();
    backend.first_before_draft = true;
    var wrapped = try Drafted.init(gpa, backend, head.drafter());
    defer wrapped.deinit();
    try std.testing.expect(wrapped.backend().first_before_draft);
    var cfg = try Config.init(gpa, wrapped.facts(base()), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var stream = try sm.Stream.init(gpa, .{ .id = "order", .prompt = &p1, .max_new = 8 });
    defer stream.deinit(gpa);
    const Hook = struct {
        fn call(ptr: *anyopaque) void {
            const flag: *bool = @ptrCast(@alignCast(ptr));
            flag.* = true;
        }
    };
    try engine.addStreamCommitted(&stream, .{ .ptr = &seen, .call = Hook.call });
    try std.testing.expect(seen and head.ordered and head.holds > 0);
    engine.discard(&stream);
}
