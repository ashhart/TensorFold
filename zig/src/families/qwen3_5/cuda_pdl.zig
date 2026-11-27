//! Qwen3.8's launches as dependent launches (TF_QWEN_PDL, unset: on for sm_12x): glue opens with qwen_pdl::enter().
const std = @import("std");
const cuda = @import("cuda");

var state: std.atomic.Value(u8) = .init(0); // 0 unread, 1 off, 2 on, 3 unset (the bound GPU's default)
var supported: std.atomic.Value(bool) = .init(true);
var by_default: std.atomic.Value(bool) = .init(false); // sm_12x (GB10, RTX 50): unset launches dependent

fn bindMajor(major: c_int) void {
    supported.store(major >= 9, .release);
    by_default.store(major == 12, .release);
}

/// Bind the process-wide launch policy to a context before loading any Qwen modules.
pub fn bind(ctx: *const cuda.Context) !void {
    bindMajor(try ctx.attribute(.compute_capability_major));
}

/// TF_QWEN_PDL, read once: 0 launches Qwen's kernels plain, 1 dependent from sm_90, unset dependent on sm_12x alone.
pub fn on() bool {
    var v = state.load(.monotonic);
    if (v == 0) {
        const set = @import("sys").environment.value(?bool, "TF_QWEN_PDL", @import("profile").lookup("TF_QWEN_PDL"), null) catch false;
        v = if (set) |want| (if (want) 2 else 1) else 3;
        state.store(v, .monotonic);
    }
    return switch (v) {
        2 => supported.load(.acquire),
        3 => by_default.load(.acquire),
        else => false,
    };
}

/// `cfg` as a Qwen launch makes it: dependent when `pdl`.
pub fn config(cfg: cuda.launch.Config, pdl: bool) cuda.launch.Config {
    var c = cfg;
    c.pdl = pdl;
    return c;
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

test "a Qwen launch keeps its config and sets only pdl" {
    const base: cuda.launch.Config = .{ .grid = .{ .x = 7, .y = 2 }, .block = .{ .x = 128 }, .shared = 4096 };
    const c = config(base, true);
    try std.testing.expect(c.pdl and c.grid.x == 7 and c.grid.y == 2 and c.block.x == 128 and c.shared == 4096);
    try std.testing.expect(!config(c, false).pdl);
}

test "Qwen PDL policy: unset is on for sm_12x alone, 1 from sm_90, 0 always off" {
    defer {
        state.store(0, .monotonic);
        supported.store(true, .release);
        by_default.store(false, .release);
    }
    state.store(3, .monotonic);
    for ([_]c_int{ 7, 8, 9, 10, 11, 13 }) |major| {
        bindMajor(major);
        try std.testing.expect(!on());
    }
    bindMajor(12);
    try std.testing.expect(on());
    state.store(2, .monotonic);
    bindMajor(8);
    try std.testing.expect(!on());
    for ([_]c_int{ 9, 10, 12 }) |major| {
        bindMajor(major);
        try std.testing.expect(on());
    }
    state.store(1, .monotonic);
    for ([_]c_int{ 8, 9, 12 }) |major| {
        bindMajor(major);
        try std.testing.expect(!on());
    }
}

test "Qwen PDL policy: the switch read once, unset taking the bound GPU's default" {
    defer {
        state.store(0, .monotonic);
        supported.store(true, .release);
        by_default.store(false, .release);
    }
    state.store(0, .monotonic);
    if (@import("profile").lookup("TF_QWEN_PDL") != null) return error.SkipZigTest; // the test's own environment set it
    bindMajor(12);
    try std.testing.expect(on());
    try std.testing.expectEqual(@as(u8, 3), state.load(.monotonic));
    bindMajor(9);
    try std.testing.expect(!on());
}

test "TF_QWEN_PDL changes only launches with a dependency prologue" {
    const Seen = struct {
        var plain: usize = 0;
        var dependent: usize = 0;
        fn run(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            plain += 1;
            return cuda.abi.success;
        }
        fn ex(c: *const cuda.abi.LaunchConfig, _: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (c.attrs) |attrs| for (attrs[0..c.num_attrs]) |at| if (at.id == .programmatic_stream_serialization and at.value.int == 1) {
                dependent += 1;
                return cuda.abi.success;
            };
            plain += 1;
            return cuda.abi.success;
        }
    };
    defer {
        state.store(0, .monotonic);
        supported.store(true, .release);
        by_default.store(false, .release);
        Seen.plain = 0;
        Seen.dependent = 0;
    }
    state.store(3, .monotonic);
    bindMajor(12);
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Seen.run;
    d.api.cuLaunchKernelEx = Seen.ex;
    const rope: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(10) };
    const other: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(11) };
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.kv8 = true;
    e.fp8.rope = rope;
    e.stream = .{ .d = &d, .handle = null };
    try e.runPdl(rope, .{ 1, 1, 1 }, .{ 256, 1, 1 }, .{});
    try std.testing.expectEqual(0, Seen.dependent);
    try std.testing.expectEqual(1, Seen.plain);
    e.kv8 = false;
    try e.runPdl(other, .{ 1, 1, 1 }, .{ 256, 1, 1 }, .{});
    try std.testing.expectEqual(1, Seen.dependent);
    try std.testing.expectEqual(1, Seen.plain);
}
