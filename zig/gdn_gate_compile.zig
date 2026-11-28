//! Analyze the actual CUDA component suite without loading a driver or a device image.
const suite = @import("tests/cuda/gdn_back_tests.zig");
const check = @import("tests/cuda/check.zig");
export fn compileGdnGate(p: *anyopaque) void {
    const gpu: *check.Gpu = @ptrCast(@alignCast(p));
    suite.run(gpu.*) catch {};
    suite.runShape(gpu.*, 5120, 64) catch {};
}
