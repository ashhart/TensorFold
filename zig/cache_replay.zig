//! Replay synthetic agent prompts on the real cache and two eviction alternatives; all times are modelled.
const std = @import("std");
const traces = @import("cache_replay_trace.zig");
const run = @import("cache_replay_run.zig");
const Model = @import("cache_replay_model.zig").Model;
pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var rounds: u32 = 61;
    var seed: u64 = 7;
    var tps: u32 = 2000;
    var details = false;
    var bytes: Model = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("Usage: tf-cache-replay [--requests] [--raw-bytes] [--rounds 46..61] [--seed N] [--tok-per-second N]\n", .{});
            return;
        } else if (std.mem.eql(u8, arg, "--requests")) {
            details = true;
        } else if (std.mem.eql(u8, arg, "--raw-bytes")) {
            bytes.padded = false;
        } else if (std.mem.eql(u8, arg, "--rounds") or std.mem.eql(u8, arg, "--seed") or std.mem.eql(u8, arg, "--tok-per-second")) {
            i += 1;
            if (i == args.len) return error.MissingArgument;
            if (std.mem.eql(u8, arg, "--rounds")) rounds = try std.fmt.parseInt(u32, args[i], 10) else if (std.mem.eql(u8, arg, "--seed")) seed = try std.fmt.parseInt(u64, args[i], 10) else tps = try std.fmt.parseInt(u32, args[i], 10);
        } else return error.UnknownArgument;
    }
    if (rounds < 46 or rounds > 61 or tps == 0) return error.BadReplayRange;
    var buffer: [16384]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    const out = &stdout.interface;
    try out.print("{{\"record\":\"model\",\"rounds\":{d},\"seed\":{d},\"tok_per_second\":{d},\"recurrent_bytes\":{d},\"kv_bytes_per_token\":{d},\"padded_buffers\":{s},\"live_state_outside_snapshot_budget\":true,\"mandatory_admission\":true}}\n", .{ rounds, seed, tps, Model.recurrent_bytes, Model.kv_bytes_per_token, if (bytes.padded) "true" else "false" });
    for (std.enums.values(traces.Shape)) |shape| {
        var trace = try traces.make(a, shape, rounds, seed);
        defer trace.deinit();
        for ([_]u64{ 4, 8, 16 }) |gib| for (std.enums.values(run.policy.Policy)) |selected| {
            var replay: run.Replay = undefined;
            replay.init(a, selected, gib << 30, bytes);
            defer replay.deinit();
            var stats: [5]run.Stats = @splat(.{});
            defer for (&stats) |*s| s.deinit(a);
            for (trace.requests, 0..) |req, index| {
                const got = try replay.request(req);
                try stats[@backingInt(req.kind)].add(a, req, got);
                try stats[4].add(a, req, got);
                if (details) try out.print("{{\"record\":\"request\",\"trace\":\"{s}\",\"digest\":\"{x}\",\"policy\":\"{s}\",\"budget_gib\":{d},\"index\":{d},\"round\":{d},\"kind\":\"{s}\",\"session\":{d},\"prompt_tokens\":{d},\"from\":{d},\"recompute_tokens\":{d},\"prefill_ttft_s\":{d:.6},\"held_bytes\":{d},\"evicted\":{d},\"refused\":{d}}}\n", .{ @tagName(shape), trace.digest, @tagName(selected), gib, index, req.round, @tagName(req.kind), req.session, req.tokens.len, got.from, got.recompute, seconds(got.recompute, tps), got.held, got.evicted, got.refused });
            }
            for (&stats, 0..) |*s, kind| {
                const label = if (kind == 4) "all" else @tagName(@as(traces.Kind, @fromBackingInt(@intCast(kind))));
                try out.print("{{\"record\":\"summary\",\"trace\":\"{s}\",\"digest\":\"{x}\",\"policy\":\"{s}\",\"budget_gib\":{d},\"kind\":\"{s}\",\"requests\":{d},\"prompt_tokens\":{d},\"recompute_tokens\":{d},\"prefill_total_s\":{d:.6},\"prefill_mean_s\":{d:.6},\"prefill_p50_s\":{d:.6},\"prefill_p95_s\":{d:.6},\"prefill_max_s\":{d:.6},\"hits\":{d},\"evicted\":{d},\"refused\":{d},\"peak_bytes\":{d}}}\n", .{ @tagName(shape), trace.digest, @tagName(selected), gib, label, s.requests, s.prompt_tokens, s.recompute_tokens, seconds(s.recompute_tokens, tps), seconds(s.recompute_tokens, tps) / @as(f64, @floatFromInt(s.requests)), seconds(s.percentile(50), tps), seconds(s.percentile(95), tps), seconds(s.percentile(100), tps), s.hits, s.evicted, s.refused, replay.peak_bytes });
            }
        };
    }
    try out.flush();
}
fn seconds(tokens: u64, tps: u32) f64 {
    return @as(f64, @floatFromInt(tokens)) / @as(f64, @floatFromInt(tps));
}
