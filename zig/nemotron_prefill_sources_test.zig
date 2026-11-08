//! Compile Nemotron's runtime prompt libraries without creating a queue or dispatching a kernel.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("src/core/frags.zig");

fn compile(device: mtl.Device, name: []const u8, text: []const u8) !void {
    const a = std.testing.allocator;
    const full = try frags.sourceMode(device, a, text, false);
    defer a.free(full);
    const lib = mtl.Library.fromSource(device, full, mtl.CompileOptions.mlx()) catch |err| {
        std.log.err("Metal prompt source {s}: {t}", .{ name, err });
        return err;
    };
    lib.deinit();
}

test "Metal source compile gate: Nemotron prompt libraries" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    for (sources.prefill) |file| try compile(device, file.name, file.text);
    try compile(device, "masked-contiguous", sources.nemotron_prompt_masked.contiguous);
    try compile(device, "masked-paged", sources.nemotron_prompt_masked.paged_source);
}
