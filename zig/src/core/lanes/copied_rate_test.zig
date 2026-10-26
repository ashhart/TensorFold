//! Model.own_copy_rates: a shared round's copy windows priced by each stream's own copies, else by copy_rate.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");
const fake = @import("fake.zig");

const gpa = std.testing.allocator;
const copied = [_]u32{ 5, 6, 7, 8, 9, 10, 11 };

/// The copy rows allocation grants each of four streams' seven-row copy windows in one shared round.
fn granted(e: *Engine, streams: []Stream, out: []usize) !void {
    _ = e.arena.reset(.retain_capacity);
    var plans: [4]win.Plan = undefined;
    for (plans[0..streams.len], streams) |*p, *s| p.* = .{ .stream = s, .position = s.context.items.len, .kind = .copy, .tokens = &copied };
    try win.allocate(e, plans[0..streams.len]);
    for (plans[0..streams.len], out) |p, *o| o.* = p.count();
}

test "own_copy_rates: copies that keep breaking are cut, by source; without it every copy is priced at copy_rate" {
    // plain rows 10 ms, 0.6 ms a row past four: seven copy rows a stream pay at the prior, not landing 1 a copy
    var costs: [32]Cost = undefined;
    for (&costs, 1..) |*c, r| c.* = .{ .width = @intCast(r), .ms = 10 + 0.6 * @as(f64, @floatFromInt(@max(r, 4) - 4)) };
    for ([_]bool{ false, true }) |own| {
        var cfg = try Config.init(gpa, .{ .exact_width = 16, .hidden_rows = true, .window_costs = &costs, .batch_rows = 32, .own_copy_rates = own }, 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        var streams: [4]Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(gpa);
        for (&streams) |*s| {
            s.* = try Stream.init(gpa, .{ .id = "copied", .prompt = &.{ 1, 2 }, .max_new = 64 });
            made += 1;
        }
        var rows: [4]usize = undefined;
        try granted(&e, &streams, &rows);
        try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7, 7 }, &rows);
        for (0..8) |_| win.observeCopied(&e, &streams[0], false, 1, 7); // stream 0's context copies land one row of seven
        try granted(&e, &streams, &rows);
        if (own) {
            try std.testing.expect(rows[0] <= 1);
            try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7 }, rows[1..]);
            try std.testing.expect(win.copiedRate(&e, &streams[0], false) < 0.6);
            try std.testing.expectApproxEqAbs(cfg.copy_rate, win.copiedRate(&e, &streams[0], true), 1e-12); // Recall's keep theirs
        } else {
            try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7, 7 }, &rows);
            try std.testing.expectEqual(@as(f64, -1), streams[0].copied_landed[0]); // no history kept
        }
    }
}
