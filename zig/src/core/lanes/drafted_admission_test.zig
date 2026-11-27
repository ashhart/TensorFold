//! Owner round-2 ask 3 (PR 5102): unsupported concurrent drafting is refused at ADMISSION —
//! before any target launch, before any existing-stream mutation — and the proof test shows the
//! second request failing while the first stays alive and its tokens stay byte-identical.
const std = @import("std");
const fixture = @import("drafted_test.zig");
const fake = @import("fake.zig");
const sm = @import("stream.zig");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Drafted = @import("drafted.zig").Drafted;

const gpa = std.testing.allocator;

test "a second drafted stream is refused at admission; the first stays alive and byte-identical" {
    // the live request's own tokens: the same drafted case runs alone under the wrapper
    const solo = try fixture.run(&.{.{ .prompt = &fixture.p1 }}, .wrapped, false);
    defer fixture.freeOut(solo);

    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: fixture.FakeDrafter = .{ .facts_ = .{ .depth = 6, .step_ms = 0.5, .one_stream = true } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const m = fixture.base();
    var cfg = try Config.init(gpa, wrapped.facts(m), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();

    var pa = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pa.deinit();
    var a = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &fixture.p1, .max_new = 40, .eos = &.{96}, .proposer = pa.proposer() });
    defer a.deinit(gpa);
    try engine.addStream(&a); // the live drafted request: its prompt pass ran, the drafter opened it

    // the second request's whole cost is the refusal: no target launch, no drafter open, and the
    // live stream's lane exactly as it stood (the old shape rejected it inside target forward,
    // after partial work, taking the active request down with it).
    const prefills = target.prefill_count;
    var pb = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pb.deinit();
    var b = try sm.Stream.init(gpa, .{ .id = "b", .prompt = &fixture.p2, .max_new = 40, .eos = &.{96}, .proposer = pb.proposer() });
    defer b.deinit(gpa);
    try std.testing.expectError(error.DrafterBusy, engine.addStream(&b));
    try std.testing.expectEqual(prefills, target.prefill_count); // no launch for b, and none replayed for a
    try std.testing.expectEqual(@as(u32, 1), head.opens);
    try std.testing.expectEqual(@as(usize, 1), wrapped.lanes.count()); // a lives on, untouched

    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(a.finished);
    try std.testing.expectEqualSlices(u32, solo.tokens[0], a.emitted()); // byte-identical

    // the lane freed with the live request: the next request is admitted and drafts fine
    var pc = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pc.deinit();
    var c = try sm.Stream.init(gpa, .{ .id = "c", .prompt = &fixture.p2, .max_new = 40, .eos = &.{96}, .proposer = pc.proposer() });
    defer c.deinit(gpa);
    try engine.addStream(&c);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(c.finished);
    const plain = try fixture.run(&.{.{ .prompt = &fixture.p2, .drafts = false }}, .plain, false);
    defer fixture.freeOut(plain);
    try std.testing.expectEqualSlices(u32, plain.tokens[0], c.emitted()); // drafted equals one-token decode
}

test "admission is inert without the drafter's one-stream fact" {
    for ([_]bool{ false, true }) |batched| {
        const both = try fixture.run(&.{ .{ .prompt = &fixture.p1 }, .{ .prompt = &fixture.p2 } }, .wrapped, batched);
        defer fixture.freeOut(both);
        const one = try fixture.run(&.{.{ .prompt = &fixture.p1, .drafts = false }}, .plain, false);
        defer fixture.freeOut(one);
        const two = try fixture.run(&.{.{ .prompt = &fixture.p2, .drafts = false }}, .plain, false);
        defer fixture.freeOut(two);
        // no one_stream fact: both streams draft side by side, each still reaching its own decode
        try std.testing.expectEqualSlices(u32, one.tokens[0], both.tokens[0]);
        try std.testing.expectEqualSlices(u32, two.tokens[0], both.tokens[1]);
        try std.testing.expectEqual(@as(u32, 2), both.head.opens);
    }
}
// Maintainer mixture ask A2 (PR 5102): the one-stream guard refuses BOTH mixture orders —
// an undrafted newcomer beside a drafted walk, and a drafted newcomer beside an undrafted
// one — before any target work, and the incumbent's completion tokens stay byte-identical.

test "an undrafted newcomer is refused beside a drafted walk; the incumbent byte-identical" {
    const solo = try fixture.run(&.{.{ .prompt = &fixture.p1 }}, .wrapped, false);
    defer fixture.freeOut(solo);

    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: fixture.FakeDrafter = .{ .facts_ = .{ .depth = 6, .step_ms = 0.5, .one_stream = true } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const m = fixture.base();
    var cfg = try Config.init(gpa, wrapped.facts(m), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();

    var pa = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pa.deinit();
    var a = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &fixture.p1, .max_new = 40, .eos = &.{96}, .proposer = pa.proposer() });
    defer a.deinit(gpa);
    try engine.addStream(&a); // the drafted incumbent: the drafter opened it

    const prefills = target.prefill_count;
    var pb = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pb.deinit();
    var b = try sm.Stream.init(gpa, .{ .id = "b", .prompt = &fixture.p2, .max_new = 40, .eos = &.{96}, .drafts = false, .proposer = pb.proposer() });
    defer b.deinit(gpa);
    try std.testing.expectError(error.DrafterBusy, engine.addStream(&b)); // the mixture refused before any target work
    try std.testing.expectEqual(prefills, target.prefill_count); // no launch for b, and none replayed for a
    try std.testing.expectEqual(@as(u32, 1), head.opens);
    try std.testing.expectEqual(@as(usize, 1), wrapped.lanes.count()); // a lives on, untouched

    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(a.finished);
    try std.testing.expectEqualSlices(u32, solo.tokens[0], a.emitted()); // byte-identical

    // it ended: an undrafted newcomer is admitted (recovery unchanged for either kind)
    var pc = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pc.deinit();
    var c = try sm.Stream.init(gpa, .{ .id = "c", .prompt = &fixture.p2, .max_new = 40, .eos = &.{96}, .drafts = false, .proposer = pc.proposer() });
    defer c.deinit(gpa);
    try engine.addStream(&c);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(c.finished);
    const plain = try fixture.run(&.{.{ .prompt = &fixture.p2, .drafts = false }}, .plain, false);
    defer fixture.freeOut(plain);
    try std.testing.expectEqualSlices(u32, plain.tokens[0], c.emitted());
}

test "a drafted newcomer is refused beside a live undrafted stream; the incumbent byte-identical" {
    const solo = try fixture.run(&.{.{ .prompt = &fixture.p1, .drafts = false }}, .plain, false);
    defer fixture.freeOut(solo);

    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: fixture.FakeDrafter = .{ .facts_ = .{ .depth = 6, .step_ms = 0.5, .one_stream = true } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const m = fixture.base();
    var cfg = try Config.init(gpa, wrapped.facts(m), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();

    var pa = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pa.deinit();
    var a = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &fixture.p1, .max_new = 40, .eos = &.{96}, .drafts = false, .proposer = pa.proposer() });
    defer a.deinit(gpa);
    try engine.addStream(&a); // the undrafted incumbent: the drafter never engages

    const prefills = target.prefill_count;
    var pb = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pb.deinit();
    var b = try sm.Stream.init(gpa, .{ .id = "b", .prompt = &fixture.p2, .max_new = 40, .eos = &.{96}, .proposer = pb.proposer() });
    defer b.deinit(gpa);
    try std.testing.expectError(error.DrafterBusy, engine.addStream(&b)); // the same refusal, reversed mixture order
    try std.testing.expectEqual(prefills, target.prefill_count); // no launch for b, and none replayed for a
    try std.testing.expectEqual(@as(u32, 0), head.opens);
    try std.testing.expectEqual(@as(usize, 0), wrapped.lanes.count());

    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(a.finished);
    try std.testing.expectEqualSlices(u32, solo.tokens[0], a.emitted()); // the incumbent byte-identical

    // it ended: a drafted newcomer is admitted and drafts fine (recovery unchanged)
    var pc = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pc.deinit();
    var c = try sm.Stream.init(gpa, .{ .id = "c", .prompt = &fixture.p2, .max_new = 40, .eos = &.{96}, .proposer = pc.proposer() });
    defer c.deinit(gpa);
    try engine.addStream(&c);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(c.finished);
    const plain = try fixture.run(&.{.{ .prompt = &fixture.p2, .drafts = false }}, .plain, false);
    defer fixture.freeOut(plain);
    try std.testing.expectEqualSlices(u32, plain.tokens[0], c.emitted()); // drafted equals one-token decode
}

test "two undrafted streams serve fine under a one_stream drafter: the gate never engages" {
    const one = try fixture.run(&.{.{ .prompt = &fixture.p1, .drafts = false }}, .plain, false);
    defer fixture.freeOut(one);
    const two = try fixture.run(&.{.{ .prompt = &fixture.p2, .drafts = false }}, .plain, false);
    defer fixture.freeOut(two);

    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: fixture.FakeDrafter = .{ .facts_ = .{ .depth = 6, .step_ms = 0.5, .one_stream = true } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const m = fixture.base();
    var cfg = try Config.init(gpa, wrapped.facts(m), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();

    var pa = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pa.deinit();
    var a = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &fixture.p1, .max_new = 40, .eos = &.{96}, .drafts = false, .proposer = pa.proposer() });
    defer a.deinit(gpa);
    var pb = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pb.deinit();
    var b = try sm.Stream.init(gpa, .{ .id = "b", .prompt = &fixture.p2, .max_new = 40, .eos = &.{96}, .drafts = false, .proposer = pb.proposer() });
    defer b.deinit(gpa);
    try engine.addStream(&a);
    try engine.addStream(&b); // no mixture exists: both serve fine, the fast path unbroken
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(a.finished);
    try std.testing.expect(b.finished);
    try std.testing.expectEqualSlices(u32, one.tokens[0], a.emitted());
    try std.testing.expectEqualSlices(u32, two.tokens[0], b.emitted());
    try std.testing.expectEqual(@as(u32, 0), head.opens); // the drafter never engaged
}
