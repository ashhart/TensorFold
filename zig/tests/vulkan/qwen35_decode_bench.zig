// Probe decode throughput: prefill the prompt, then `n` greedy tokens with readback after 8 warm-up tokens.
const std = @import("std");
const Engine = @import("qwen35_vulkan").engine.Engine;

pub fn run(io: std.Io, engine: *Engine, prompt: []const u32, n: usize) !void {
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
