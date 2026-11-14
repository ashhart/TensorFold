//! Fresh-seed confirmation of the frozen eviction selector; each invocation measures one policy.
const std = @import("std");
const traces = @import("cache_agent_trace.zig");
const run = @import("cache_replay_run.zig");
pub const seeds = [_]u64{ 11, 23 };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedPolicyArgument;
    const native = std.mem.eql(u8, args[1], "native_decayed");
    const selected = if (native) .main else std.meta.stringToEnum(run.policy.Policy, args[1]) orelse return error.UnknownPolicy;
    if (selected != .main and selected != .decayed_resume) return error.UnfrozenPolicy;
    var buffer: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    const out = &stdout.interface;
    for (seeds) |seed| for (std.enums.values(traces.Shape)) |shape| {
        var trace = try traces.makeSeeded(init.gpa, shape, seed);
        defer trace.deinit();
        for (traces.budgets) |budget| {
            var replay: run.Replay = undefined;
            replay.init(init.gpa, selected, budget, .{});
            if (native) replay.store.eviction = .decayed_resume;
            defer replay.deinit();
            var stats: run.Stats = .{};
            defer stats.deinit(init.gpa);
            for (trace.requests) |request| try stats.add(init.gpa, request, try replay.request(request));
            try out.print("{{\"seed\":{d},\"trace\":\"{s}\",\"digest\":\"{x}\",\"policy\":\"{s}\",\"budget_bytes\":{d},\"requests\":{d},\"hits\":{d},\"prompt_rows\":{d},\"rebuilt_rows\":{d},\"evictions\":{d},\"peak_bytes\":{d}}}\n", .{ seed, @tagName(shape), trace.digest, if (native) "decayed_resume" else @tagName(selected), budget, stats.requests, stats.hits, stats.prompt_tokens, stats.recompute_tokens, stats.evicted, replay.peak_bytes });
        }
    };
    try out.flush();
}
