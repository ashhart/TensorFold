//! Check the production NVFP4 kernels against independent references, without loading model weights.
const std = @import("std");
const drama = @import("drama3");

pub fn main() !void {
    const gpu = try drama.gpu.Gpu.open(std.heap.page_allocator, @import("drama3_metallib").bytes);
    defer gpu.close();
    try drama.quant_check.run(gpu);
    try drama.guidance.check(gpu);
}
