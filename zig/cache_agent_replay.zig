//! E24 measurements: real snapshot store rows, not device latency or model output claims.
const std = @import("std");
const traces = @import("cache_agent_trace.zig");
const run = @import("cache_replay_run.zig");

pub fn main(init: std.process.Init) !void {
    var buffer: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    const out = &stdout.interface;
    for (std.enums.values(traces.Shape)) |shape| {
        var trace = try traces.make(init.gpa, shape);
        defer trace.deinit();
        for (traces.budgets) |budget| for ([_]run.policy.Policy{ .main, .frequency_probation }) |selected| {
            var replay: run.Replay = undefined;
            replay.init(init.gpa, selected, budget, .{});
            defer replay.deinit();
            var stats: run.Stats = .{};
            defer stats.deinit(init.gpa);
            for (trace.requests) |request| try stats.add(init.gpa, request, try replay.request(request));
            try out.print("{{\"trace\":\"{s}\",\"digest\":\"{x}\",\"policy\":\"{s}\",\"budget_bytes\":{d},\"requests\":{d},\"hits\":{d},\"prompt_rows\":{d},\"rebuilt_rows\":{d},\"evictions\":{d},\"peak_bytes\":{d}}}\n", .{ @tagName(shape), trace.digest, @tagName(selected), budget, stats.requests, stats.hits, stats.prompt_tokens, stats.recompute_tokens, stats.evicted, replay.peak_bytes });
        };
    }
    try out.flush();
}
