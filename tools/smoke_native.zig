const std = @import("std");
const sync = @import("sync_upstream.zig");

const Capabilities = struct {
    mlx_version: []const u8,
    cpu_arithmetic: bool,
    metal_available: bool,
    metal_arithmetic: bool,
    tensor_units: bool,
};

fn tensorTests(caps: Capabilities) !bool {
    if (!caps.cpu_arithmetic) return error.HostRuntimeFailed;
    if (!caps.metal_available) return error.MetalUnavailable;
    if (!caps.metal_arithmetic) return error.MetalRuntimeFailed;
    return caps.tensor_units;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) return error.InvalidArguments;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(4096));
    const caps = (try std.json.parseFromSlice(Capabilities, a, bytes, .{})).value;
    const tensor = tensorTests(caps) catch |err| {
        std.debug.print("Metal smoke checks cannot run: {s}. This is not a GPU verification pass.\n", .{@errorName(err)});
        return err;
    };
    try sync.command(init.io, &.{ ".zig-toolchain/zig", "build", "test-simd-attention", "test-row-attention", "test-affine", "test-simd-bits", "test-prefill-math", "-Doptimize=safe", "-j1" });
    if (tensor) {
        try sync.command(init.io, &.{ ".zig-toolchain/zig", "build", "test-tensor-quantization", "test-glm-model", "-Doptimize=safe", "-j1" });
    } else std.debug.print("M5 tensor tests not applicable to this GPU; SIMD smoke checks passed.\n", .{});
}

test "smoke selection never reports unavailable Metal as a pass" {
    var caps: Capabilities = .{ .mlx_version = "test", .cpu_arithmetic = true, .metal_available = false, .metal_arithmetic = false, .tensor_units = false };
    try std.testing.expectError(error.MetalUnavailable, tensorTests(caps));
    caps.metal_available = true;
    try std.testing.expectError(error.MetalRuntimeFailed, tensorTests(caps));
    caps.metal_arithmetic = true;
    try std.testing.expect(!try tensorTests(caps));
    caps.tensor_units = true;
    try std.testing.expect(try tensorTests(caps));
}
