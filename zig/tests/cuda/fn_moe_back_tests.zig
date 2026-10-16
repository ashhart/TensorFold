//! fn-moe-back: moe-back's layer with Flash Next's pocket neurons beside the shared expert, through flashnext.fn_back.
const flashnext = @import("flashnext");
const check = @import("check.zig");
const moe = @import("moe_back_tests.zig");
const fb = flashnext.fn_back;

pub fn run(gpu: check.Gpu) !void {
    if (!@import("cuda").kernels.available) return moe.runCase(gpu, "fn-moe-back", void, {});
    var k = try fb.Kernels.load(gpu.d);
    defer k.deinit();
    try moe.runCase(gpu, "fn-moe-back", fb.Ops, &k);
}
