//! The real TFQ3 GLU helper against a BF16 rounding-point witness.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub fn run(gpu: check.Gpu, kernels: *const cuda.tfq3.Kernels, stream: cuda.Stream) !void {
    const f = try kernels.mod_moe.function("tf_tfq3_glu_contract");
    var out = try cuda.DeviceBuffer.alloc(gpu.d, 8);
    defer out.free();
    try out.fill8(0xa5, stream.handle);
    var args: cuda.Args = .{};
    args.add(out.ptr);
    try cuda.launch.launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &args);
    try stream.synchronize();
    var got: [2]f32 = undefined;
    try out.download(0, std.mem.asBytes(&got));
    const want = [_]f32{ 0.46484375, 0.46484375 };
    try check.sameBytes("TFQ3 BF16(SiLU) before up product", std.mem.asBytes(&got), std.mem.asBytes(&want));
    check.pass("TFQ3 GLU: GLM and Flash Next round SiLU before up; old GLM value 0.466796875 refused", .{});
}
