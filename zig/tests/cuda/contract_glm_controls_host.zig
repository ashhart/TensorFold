//! GLM page-case registration and planted-control wiring, without opening or calling a device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Controls = @import("contract_glm_controls.zig").Controls;
const after = @import("contract_after.zig");
const g = @import("core").contract.gates;
const registry = @import("core").contract.registry;
const t = std.testing;

const fp8_launches = [_]struct { symbol: []const u8, threads: u32, dynamic: u32, static: u32 }{
    .{ .symbol = "tf_glm_lat_write_fp8_paged", .threads = 512, .dynamic = 0, .static = 64 },
    .{ .symbol = "tf_glm_chunks_16_fp8_paged", .threads = 256, .dynamic = glm.glue.chunksSmem(16), .static = 0 },
    .{ .symbol = "tf_glm_chunks_32_fp8_paged", .threads = 256, .dynamic = glm.glue.chunksSmem(32), .static = 0 },
};

fn fp8Inventory(records: []const registry.CudaRecord) !void {
    for (fp8_launches) |want| {
        const r = for (records) |record| {
            if (std.mem.eql(u8, want.symbol, record.launch.symbol)) break record;
        } else return error.MissingFp8PagedLaunch;
        if (!std.mem.eql(u8, r.launch.image, "glm_glue") or r.launch.threads != want.threads or
            r.launch.dynamic != want.dynamic or r.launch.opt_in != (want.dynamic > 48 * 1024))
            return error.BadFp8PagedLaunch;
        for (r.usage) |usage| {
            if (usage.regs != 0) return error.UnmeasuredFp8PagedUsage;
            if (usage.static != 0) return error.UnmeasuredFp8PagedUsage;
        }
        if (!r.later_pending) return error.UnmeasuredFp8PagedUsage;
        _ = want.static; // static bytes are documented here; compiler evidence remains deliberately pending.
    }
}

test "GLM page contract: paged FP8 launches are registered with exact geometry and honest pending usage" {
    try fp8Inventory(&registry.all_cuda);
}

test "GLM page contract: missing and wrong paged FP8 launch controls are rejected" {
    var records: [fp8_launches.len]registry.CudaRecord = undefined;
    for (fp8_launches, &records) |want, *out| out.* = for (registry.all_cuda) |record| {
        if (std.mem.eql(u8, want.symbol, record.launch.symbol)) break record;
    } else return error.MissingFp8PagedLaunch;
    for (0..records.len) |missing| {
        var incomplete: std.ArrayList(registry.CudaRecord) = .empty;
        defer incomplete.deinit(t.allocator);
        for (records, 0..) |record, i| if (i != missing) try incomplete.append(t.allocator, record);
        try t.expectError(error.MissingFp8PagedLaunch, fp8Inventory(incomplete.items));
    }
    var wrong = records;
    wrong[1].launch.dynamic -= 16;
    try t.expectError(error.BadFp8PagedLaunch, fp8Inventory(&wrong));
    wrong = records;
    wrong[0].usage[0].regs = 1;
    try t.expectError(error.UnmeasuredFp8PagedUsage, fp8Inventory(&wrong));
}

fn validate(kernels: []const g.Kernel) !void {
    if (kernels.len != 13) return error.MissingPageControl;
    for (kernels, 0..) |k, i| {
        if (k.control != .shapes or !std.mem.startsWith(u8, k.spec.name, "control:"))
            return error.MissingPageControl;
        if (k.spec.served.len == 0) return error.MissingPageShape;
        for (kernels[0..i]) |prior| if (std.mem.eql(u8, prior.spec.name, k.spec.name))
            return error.DuplicatePageControlName;
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
    var controls = Controls.init(gpu, stream, functions, .{ .d = &d, .handle = null }, null);
    // Only descriptors are read: empty rigs own no allocations and no CUDA callback is invoked.
    if (fault) controls.lat.wrong_table = false;
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(t.allocator);
    try controls.append(t.allocator, &kernels);
    try validate(kernels.items);
    after.keep(&kernels, &.{"_fp8_paged"});
    if (kernels.items.len != 5) return error.MissingFocusedFp8PageControl;
}

test "GLM page contract: all thirteen BF16 and FP8 paged numerical cases have a planted address control" {
    try registered(false);
}

test "GLM page contract: a disabled planted control is rejected by host inventory" {
    try t.expectError(error.MissingPageControl, registered(true));
}
