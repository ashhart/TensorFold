//! Gemma 4's decode kernels as dependent launches (TF_GEMMA4_PDL, default off): each opens with gemma4_pdl::enter().
const std = @import("std");
const cuda = @import("cuda");

var state: std.atomic.Value(u8) = .init(0); // 0 unread, 1 off, 2 on
var supported: std.atomic.Value(bool) = .init(true);

fn bindMajor(major: c_int) void {
    supported.store(major >= 9, .release);
}

/// Bind the process-wide launch policy to a context before loading any Gemma 4 modules.
pub fn bind(ctx: *const cuda.Context) !void {
    bindMajor(try ctx.attribute(.compute_capability_major));
}

/// TF_GEMMA4_PDL, read once: exact 1 opts in (the environment refuses malformed values at startup).
pub fn on() bool {
    const v = state.load(.monotonic);
    if (v != 0) return v == 2 and supported.load(.acquire);
    const env = @import("sys").environment;
    const want = env.value(bool, "TF_GEMMA4_PDL", @import("profile").lookup("TF_GEMMA4_PDL"), false) catch false;
    state.store(if (want) 2 else 1, .monotonic);
    return want and supported.load(.acquire);
}

/// `cfg` as a Gemma 4 launch makes it: dependent when `pdl`.
pub fn config(cfg: cuda.launch.Config, pdl: bool) cuda.launch.Config {
    var c = cfg;
    c.pdl = pdl;
    return c;
}

test "a Gemma 4 launch keeps its config and sets only pdl" {
    const base: cuda.launch.Config = .{ .grid = .{ .x = 7, .y = 2 }, .block = .{ .x = 128 }, .shared = 4096 };
    const c = config(base, true);
    try std.testing.expect(c.pdl and c.grid.x == 7 and c.grid.y == 2 and c.block.x == 128 and c.shared == 4096);
    try std.testing.expect(!config(c, false).pdl);
}

test "Gemma 4 PDL policy: default off, sm90 opt-in on, pre-sm90 stays plain" {
    state.store(0, .monotonic);
    supported.store(true, .release);
    defer {
        state.store(0, .monotonic);
        supported.store(true, .release);
    }
    try std.testing.expect(!on()); // TF_GEMMA4_PDL unset: the default keeps plain launches
    state.store(2, .monotonic);
    bindMajor(9);
    try std.testing.expect(on());
    bindMajor(8);
    try std.testing.expect(!on()); // pre-sm90 launches plain even with the flag
    bindMajor(9);
    state.store(1, .monotonic);
    try std.testing.expect(!on()); // TF_GEMMA4_PDL=0
}
