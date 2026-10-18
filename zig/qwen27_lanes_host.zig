//! Shared lane equality gates use the real scheduler and history-dependent fake target.
const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;

fn run(prompt: []const u32, depth: u32, restored: usize, neighbors: usize, seed: ?u64) ![]u32 {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 5 + @as(f64, @floatFromInt(width)) };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .mtp = depth > 0, .speculate = true, .speculate_early = false, .drafts = depth, .window_costs = &costs, .mtp_step_ms = 0.1, .hidden_rows = true, .batch_rows = 128, .max_streams = 128, .draft_streams = true }, 16, depth);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target = lanes.fake.Fake{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var prefix: std.ArrayList(u32) = .empty;
    defer prefix.deinit(a);
    try prefix.appendSlice(a, prompt[0..restored]);
    var streams: [128]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (streams[0 .. neighbors + 1], 0..) |*s, i| {
        s.* = try lanes.Stream.init(a, .{ .id = "speed", .prompt = prompt, .max_new = 40, .drafts = depth > 0, .sampling = if (seed) |v| .{ .seed = v + i, .temperature = 0.8 } else null });
        made += 1;
        if (i == 0 and restored > 0) s.reuse = .{ .saved = &prefix, .at = @intCast(restored) };
        try engine.addStream(s);
    }
    while (engine.activeCount() > 0) try engine.step();
    if (depth > 0 and neighbors == 0) try std.testing.expect(streams[0].drafted > 0);
    if (restored > 0) try std.testing.expectEqual(@as(u32, @intCast(restored)), streams[0].cached);
    return a.dupe(u32, streams[0].emitted());
}

test "qwen27 speed: shared scheduler drafted plain resumed fresh concurrent solo token identities" {
    const prompt = [_]u32{ 7, 1, 2, 3, 4, 5, 6, 8, 9, 7, 1, 2, 3, 4 };
    for ([_]?u64{ null, 17, 91 }) |seed| {
        const plain = try run(&prompt, 0, 0, 0, seed);
        defer a.free(plain);
        for ([_]u32{ 0, 1, 4, 15 }) |depth| for ([_]usize{ 0, 5, 13 }) |restore| {
            for ([_]usize{ 0, 1, 7, 8, 63, 64, 112, 113, 127 }) |neighbors| {
                const got = try run(&prompt, depth, restore, neighbors, seed);
                defer a.free(got);
                try std.testing.expectEqualSlices(u32, plain, got);
            }
        };
    }
}
