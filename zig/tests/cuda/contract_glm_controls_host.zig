//! GLM page-case registration and planted-control wiring, without opening or calling a device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Controls = @import("contract_glm_controls.zig").Controls;
const g = @import("core").contract.gates;
const t = std.testing;

fn validate(kernels: []const g.Kernel) !void {
    if (kernels.len != 8) return error.MissingPageControl;
    for (kernels) |k| {
        if (k.control != .shapes or !std.mem.startsWith(u8, k.spec.name, "control:"))
            return error.MissingPageControl;
        if (k.spec.served.len == 0) return error.MissingPageShape;
    }
}

fn registered(fault: bool) !void {
    var d: cuda.Driver = undefined;
    var ctx: cuda.Context = undefined;
    var functions: glm.glue.Kernels = undefined;
    inline for (@typeInfo(glm.glue.Kernels).@"struct".field_names) |name|
        @field(functions, name) = .{ .d = &d, .handle = null };
    const gpu: check.Gpu = .{ .d = &d, .ctx = &ctx, .gpa = t.allocator, .io = t.io };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var controls = Controls.init(gpu, stream, functions, .{ .d = &d, .handle = null });
    // Only descriptors are read: empty rigs own no allocations and no CUDA callback is invoked.
    if (fault) controls.lat.wrong_table = false;
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(t.allocator);
    try controls.append(t.allocator, &kernels);
    try validate(kernels.items);
}

test "GLM page contract: all eight paged numerical cases have a swapped-table or physical-row control" {
    try registered(false);
}

test "GLM page contract: a disabled planted control is rejected by host inventory" {
    try t.expectError(error.MissingPageControl, registered(true));
}
