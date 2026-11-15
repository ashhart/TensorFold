// Probe decode throughput: prefill, then `arg` (default 128) greedy tokens with readback after 8 warm-up tokens.
const std = @import("std");
const Engine = @import("qwen35_vulkan").engine.Engine;

/// Prints one DECODE-BENCH line and returns the probe's exit code.
pub fn run(io: std.Io, engine: *Engine, prompt: []const u32, arg: []const u8) u8 {
    const n = std.fmt.parseInt(usize, arg, 10) catch 128;
    bench(io, engine, prompt, n) catch |e| {
        std.debug.print("FAIL decode bench: {s}\n", .{@errorName(e)});
        return 1;
    };
    return 0;
}

fn bench(io: std.Io, engine: *Engine, prompt: []const u32, n: usize) !void {
    try engine.prefill(prompt);
    var tok = try engine.draw();
    var tok_hash: u64 = 1469598103934665603;
    var t0 = std.Io.Clock.awake.now(io);
    for (0..8 + n) |i| {
        if (i == 8) t0 = std.Io.Clock.awake.now(io);
        try engine.decode(tok);
        tok = try engine.draw();
        tok_hash = (tok_hash ^ tok) *% 1099511628211;
    }
    const wall_ms: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds());
    const per = wall_ms / @as(f64, @floatFromInt(n));
    std.debug.print("DECODE-BENCH prompt={d} tokens={d} ms_per_token={d:.3} tok_s={d:.2} tok_hash={x}\n", .{ prompt.len, n, per, 1000.0 / per, tok_hash });
}
