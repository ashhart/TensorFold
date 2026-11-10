//! Calibrated prompt activation owns its projections before the ordinary dense path can claim them.
pub const Claim = enum { none, gate_up_ready, act_ready };

pub fn claim(e: anytype, gate: anytype, up: @TypeOf(gate), x: u64, rows: usize, prompt: bool) !Claim {
    return claimWith(@import("cuda_fast_prompt.zig"), e, gate, up, x, rows, prompt);
}

pub fn claimWith(comptime Provider: type, e: anytype, gate: anytype, up: @TypeOf(gate), x: u64, rows: usize, prompt: bool) !Claim {
    if (!prompt) return .none;
    if (@hasField(@TypeOf(e.*), "prompt_many")) {
        const calibrated = if (e.mo) |m| m.native_requested or m.fast8 != null else false;
        if (calibrated) e.prompt_many = false;
    }
    if (e.learning != null) return .none;
    return if (try Provider.swiglu(e, gate, up, x, e.act, rows)) .act_ready else .none;
}

pub fn finishDense(e: anytype, m: anytype, layer: usize, total: usize, rows: usize, prompt: bool, output: u64) !void {
    try e.projAgain(m.down, e.act, output, rows, prompt);
    if (e.mlp) |k| try k.f(k.ctx, layer, e.x, e.r, total, e.taught) else try e.pockets.add(e.stream, layer, e.x, e.r, total, e.taught);
}

pub fn taps(e: anytype, enabled: bool, layer: usize, total: usize, segments: u64, row_segments: u64, parameters: u64) !void {
    if (enabled) for (e.taps[0..e.n_taps], 0..) |t, ti| if (t == layer) {
        try e.run(e.fns.tap, .{ total, (e.g.hidden + 2047) / 2048, 1 }, .{ 256, 1, 1 }, .{
            e.h, e.r, segments, row_segments, @as(c_int, @intCast(ti)), @as(c_int, @intCast(e.g.hidden)), parameters,
        });
    };
}

test {
    _ = @import("cuda_mlp_dispatch_test.zig");
}
