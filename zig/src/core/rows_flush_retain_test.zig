//! A partial flush keeps on disk only the retained states' own disk states; no dropped state's write publishes.
const std = @import("std");
const fixture = @import("rows_async_fixture.zig");
const api = @import("engine_api.zig");

/// The newest of `prompts` kept at `ats` survives a retain-one flush alone, on disk and in the store's folder.
fn retainNewest(x: *fixture.Fixture, prompts: []const *const [1100]u32, ats: []const u32) !void {
    for (prompts, ats) |p, at| try x.keep(p, at);
    try x.idle(prompts.len);
    try std.testing.expectEqual(prompts.len, x.s.entries.items.len);
    try std.testing.expectEqual(prompts.len, x.r.states.items.len);
    const last = ats[ats.len - 1];
    var kept_disk: u64 = 0;
    for (x.r.states.items) |s| if (s.at == last) {
        kept_disk = s.own + s.rows;
    };
    const held_disk = x.r.held();
    var ask: api.ImprintAsk = .{ .op = .flush, .retain = 1 };
    ask.answer(&x.s, .{});
    try std.testing.expectEqual(@as(usize, 1), x.s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
    try std.testing.expectEqual(last, x.r.states.items[0].at);
    try std.testing.expectEqualSlices(u32, x.s.entries.items[0].tokens, x.r.states.items[0].tokens);
    try std.testing.expectEqual(held_disk - kept_disk, ask.freed);
    try std.testing.expectEqual(kept_disk, x.r.held());
    try std.testing.expectEqual(@as(usize, 1), fixture.files(x.r.dir, ".st")); // what a restart reads back
}

test "prompt cache flush: retain one drops a shorter disk state on the retained state's own path" {
    const x = try fixture.Fixture.open("flush-prefix", .off, true, 1 << 20);
    defer x.close();
    const p = fixture.prompt(80);
    try retainNewest(x, &.{ &p, &p }, &.{ 300, 600 });
}

test "prompt cache flush: retain one keeps the retained state's disk state with no lookahead" {
    const x = try fixture.Fixture.keyed("flush-ahead0", .off, true, 1 << 20, 0, 0, false);
    defer x.close();
    const p0 = fixture.prompt(81);
    const p1 = fixture.prompt(82);
    try retainNewest(x, &.{ &p0, &p1 }, &.{ 300, 600 });
}

test "prompt cache flush: retain one keeps the retained state's disk state under its planned chunk starts" {
    const x = try fixture.Fixture.keyed("flush-planned", .off, true, 1 << 20, 0, 1, true);
    defer x.close();
    x.f.starts = &.{ 150, 300, 450, 600 };
    const p0 = fixture.prompt(83);
    const p1 = fixture.prompt(84);
    try retainNewest(x, &.{ &p0, &p1 }, &.{ 300, 600 });
}

test "prompt cache flush: retain one stops a dropped state's staged write from publishing" {
    const x = try fixture.Fixture.open("flush-staged", .off, false, 1 << 30);
    defer x.close();
    const p0 = fixture.prompt(85);
    const p1 = fixture.prompt(86);
    try x.keep(&p0, 300); // staged on the writer; published only when the engine reaps it
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    try x.keep(&p1, 600);
    var ask: api.ImprintAsk = .{ .op = .flush, .retain = 1 };
    ask.answer(&x.s, .{});
    try std.testing.expectEqual(@as(usize, 1), x.s.entries.items.len);
    try std.testing.expectEqual(@as(u32, 600), x.s.entries.items[0].at);
    try x.settle();
    for (x.r.states.items) |s| try std.testing.expect(s.at != 300);
    try std.testing.expect(x.r.best(&p0, &.{}, &.{}, 0) == null);
}
