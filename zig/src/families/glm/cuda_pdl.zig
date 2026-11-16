//! GLM's launches as dependent launches (TF_GLM_PDL, default on): every GLM kernel opens with glm_pdl::enter().
const std = @import("std");
const cuda = @import("cuda");

var state: std.atomic.Value(u8) = .init(0); // 0 unread, 1 off, 2 on

/// TF_GLM_PDL, read once: exact 0 launches GLM's kernels plain (the environment refuses malformed values at startup).
pub fn on() bool {
    const v = state.load(.monotonic);
    if (v != 0) return v == 2;
    const want = @import("sys").environment.value(bool, "TF_GLM_PDL", @import("profile").lookup("TF_GLM_PDL"), true) catch true;
    state.store(if (want) 2 else 1, .monotonic);
    return want;
}

/// A GLM image's symbol (tf_glm_*, glm_mm* and glm_vis_*, glm_dflash's df_*), or a name the driver cannot give.
pub fn glmKernel(name: []const u8) bool {
    if (std.mem.eql(u8, name, "?")) return true;
    for ([_][]const u8{ "tf_glm_", "glm_", "df_" }) |p| if (std.mem.startsWith(u8, name, p)) return true;
    return false;
}

/// `cfg` as a GLM launch makes it: dependent when `pdl`.
pub fn config(cfg: cuda.launch.Config, pdl: bool) cuda.launch.Config {
    var c = cfg;
    c.pdl = pdl;
    return c;
}

/// Launches a GLM image's kernel, dependent under TF_GLM_PDL; safe builds refuse any other kernel (it does not wait).
pub fn launch(f: cuda.Function, cfg: cuda.launch.Config, s: cuda.Stream, a: *cuda.Args) !void {
    if (std.debug.runtime_safety) {
        const name = f.d.functionName(f.handle);
        if (!glmKernel(name)) std.debug.panic("cuda_pdl.launch: {s} is not a GLM kernel; launch it with cuda.launch.launch", .{name});
    }
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

test "GLM symbols pass, shared kernels do not" {
    for ([_][]const u8{ "tf_glm_hc_post", "glm_mm_reduce", "glm_vis_rmsnorm", "df_attn", "?" }) |n| try std.testing.expect(glmKernel(n));
    for ([_][]const u8{ "tf_draw", "tf_pack_dense", "roce_gather", "_ZN7tf_qmmf11qmmf_kernel", "" }) |n| try std.testing.expect(!glmKernel(n));
}

test "a GLM launch keeps its config and sets only pdl" {
    const base: cuda.launch.Config = .{ .grid = .{ .x = 7, .y = 2 }, .block = .{ .x = 128 }, .shared = 4096 };
    const c = config(base, true);
    try std.testing.expect(c.pdl and c.grid.x == 7 and c.grid.y == 2 and c.block.x == 128 and c.shared == 4096);
    try std.testing.expect(!config(c, false).pdl);
}
