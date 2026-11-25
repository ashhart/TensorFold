//! Exercise the backend's actual registration constructor and list, with a recording CUDA module.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const contract = @import("contract_qmmf_exact.zig");
const q = cuda.qmmf;
const wide = q.exact_tile;
const abi = cuda.abi;

test "qmmf exact registration: actual constructor appends every candidate and its real order control" {
    const Mock = struct {
        var lookups: usize = 0;
        var attrs: usize = 0;
        var wrong_symbol: bool = false;
        fn function(f: *abi.Function, _: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
            const symbols = [_][:0]const u8{ wide.symbol(.bf16, false), wide.symbol(.bf16, true), wide.symbol(.f32, false), wide.symbol(.f32, true), wide.control_symbol };
            if (lookups >= symbols.len or !std.mem.eql(u8, std.mem.span(name), symbols[lookups])) wrong_symbol = true;
            f.* = @ptrFromInt(20 + lookups);
            lookups += 1;
            return abi.success;
        }
        fn attribute(_: abi.Function, _: abi.FunctionAttribute, n: c_int) callconv(.c) abi.Result {
            if (n != wide.shared) wrong_symbol = true;
            attrs += 1;
            return abi.success;
        }
        fn sync(_: abi.Stream) callconv(.c) abi.Result {
            return abi.success;
        }
    };
    var driver: cuda.Driver = .{ .lib = undefined, .api = undefined };
    driver.api.cuModuleGetFunction = Mock.function;
    driver.api.cuFuncSetAttribute = Mock.attribute;
    driver.api.cuStreamSynchronize = Mock.sync;
    const a = std.testing.allocator;
    const gpu: check.Gpu = .{ .d = &driver, .ctx = undefined, .gpa = a, .io = undefined };
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const old: cuda.Function = .{ .d = &driver, .handle = @ptrFromInt(10) };
    const lane: q.Lane = .{
        .mod = .{ .d = &driver, .handle = null },
        .fns = @splat(@splat(old)),
        .f32_fns = @splat(@splat(old)),
        .reduce = @splat(old),
        .half = @splat(@splat(@splat(old))),
        .clusters = false,
    };
    var kernels: std.ArrayList(core.contract.gates.Kernel) = .empty;
    defer kernels.deinit(a);
    var unsupported = try contract.Registrations.load(gpu, stream, lane, 89);
    defer unsupported.deinit();
    try unsupported.append(a, &kernels);
    try std.testing.expectEqual(@as(usize, 0), kernels.items.len);
    try std.testing.expectEqual(@as(usize, 0), Mock.lookups);
    var registrations = try contract.Registrations.load(gpu, stream, lane, 121);
    defer registrations.deinit();
    try registrations.append(a, &kernels);
    try std.testing.expectEqual(@as(usize, 5), kernels.items.len);
    try std.testing.expectEqual(@as(usize, 5), Mock.lookups);
    try std.testing.expectEqual(@as(usize, 4), Mock.attrs);
    try std.testing.expect(!Mock.wrong_symbol);
    for (kernels.items, 0..) |kernel, i| {
        try std.testing.expect(@intFromPtr(kernel.ctx) == @intFromPtr(&registrations.products[i]));
        try std.testing.expect(kernel.spec.replaces != null and kernel.vt.replaced != null and kernel.vt.simulate != null);
        try std.testing.expectEqual(i == 4, kernel.control != null);
        if (i == 4) try std.testing.expectEqual(core.contract.spec.Gate.shapes, kernel.control.?);
    }
    const records = @import("core").contract.registry.all_cuda;
    const symbols = [_][:0]const u8{ wide.symbol(.bf16, false), wide.symbol(.bf16, true), wide.symbol(.f32, false), wide.symbol(.f32, true), wide.control_symbol };
    for (symbols) |symbol| {
        var count: usize = 0;
        for (records) |record| if (std.mem.eql(u8, record.launch.symbol, symbol)) {
            count += 1;
            try std.testing.expectEqual(@as(u32, 128), record.launch.threads);
            try std.testing.expectEqual(wide.shared, record.launch.dynamic);
            try std.testing.expectEqual(@as(usize, 5), record.usage.len);
            for (record.usage) |usage| try std.testing.expectEqual(@as(u32, 0), usage.regs);
            const missing: usize = if (record.launch.cluster == 8) 2 else 5;
            try std.testing.expectEqual(missing, try core.contract.registry.checkCuda(a, &.{record}, true));
        };
        try std.testing.expectEqual(@as(usize, 1), count);
    }
}
