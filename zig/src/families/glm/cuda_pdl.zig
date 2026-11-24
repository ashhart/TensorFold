//! GLM's launches as dependent launches (TF_GLM_PDL, default on): every GLM kernel opens with glm_pdl::enter().
const std = @import("std");
const cuda = @import("cuda");

var state: std.atomic.Value(u8) = .init(0); // 0 unread, 1 off, 2 on
var supported: std.atomic.Value(bool) = .init(true);

fn bindMajor(major: c_int) void {
    supported.store(major >= 9, .release);
}

/// Bind the process-wide launch policy to a context before loading any GLM modules.
pub fn bind(ctx: *const cuda.Context) !void {
    bindMajor(try ctx.attribute(.compute_capability_major));
}

/// TF_GLM_PDL, read once: exact 0 launches GLM's kernels plain (the environment refuses malformed values at startup).
pub fn on() bool {
    const v = state.load(.monotonic);
    if (v != 0) return v == 2 and supported.load(.acquire);
    const want = @import("sys").environment.value(bool, "TF_GLM_PDL", @import("profile").lookup("TF_GLM_PDL"), true) catch true;
    state.store(if (want) 2 else 1, .monotonic);
    return want and supported.load(.acquire);
}

/// `cfg` as a GLM launch makes it: dependent when `pdl`.
pub fn config(cfg: cuda.launch.Config, pdl: bool) cuda.launch.Config {
    var c = cfg;
    c.pdl = pdl;
    return c;
}

/// Launches a GLM image's kernel (test-glm-pdl-scan: each one opens with its wait), dependent under TF_GLM_PDL.
pub fn launch(f: cuda.Function, cfg: cuda.launch.Config, s: cuda.Stream, a: *cuda.Args) !void {
    return cuda.launch.launch(f, config(cfg, on()), s, a);
}

/// Test drivers: a cuLaunchKernelEx handing each launch, its attributes dropped, to a recording cuLaunchKernel.
pub fn forwardEx(comptime plain: @FieldType(cuda.abi.Api, "cuLaunchKernel")) @FieldType(cuda.abi.Api, "cuLaunchKernelEx") {
    const P = ?[*]?*anyopaque;
    return &struct {
        fn ex(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, p: P, e: P) callconv(.c) cuda.abi.Result {
            return plain(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, p, e);
        }
    }.ex;
}

test "a GLM launch keeps its config and sets only pdl" {
    const base: cuda.launch.Config = .{ .grid = .{ .x = 7, .y = 2 }, .block = .{ .x = 128 }, .shared = 4096 };
    const c = config(base, true);
    try std.testing.expect(c.pdl and c.grid.x == 7 and c.grid.y == 2 and c.block.x == 128 and c.shared == 4096);
    try std.testing.expect(!config(c, false).pdl);
}

test "GLM PDL policy: pre-sm90 devices disable the default and sm90 retains it" {
    state.store(2, .monotonic);
    supported.store(true, .release);
    defer {
        state.store(0, .monotonic);
        supported.store(true, .release);
    }
    bindMajor(8);
    try std.testing.expect(!on());
    supported.store(true, .release);
    bindMajor(9);
    try std.testing.expect(on());
    state.store(1, .monotonic);
    try std.testing.expect(!on());
}
