//! Host test steps of the small CUDA families (Spark 2.5 and LFM2 MoE), registered from cuda.zig's host step.
const std = @import("std");

pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    @import("spark25.zig").host(b, @import("spark25.zig").module(b, host, .debug, cuda, core, lanes), all);
    @import("lfm2_moe_cuda.zig").host(b, @import("lfm2_moe_cuda.zig").module(b, host, .debug, cuda, core, lanes), all);
}
