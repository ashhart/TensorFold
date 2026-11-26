//! The actual MTP picker contract constructor and pending launch records, without a GPU.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const contract = @import("contract_glm_mtp_pick.zig");

test "GLM MTP registration: backend constructor appends the picker and both real controls" {
    const Mock = struct {
        var calls: usize = 0;
        var wrong = false;
        fn function(out: *cuda.abi.Function, mod: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
            if (calls >= contract.entries.len) return 1;
            const e = contract.entries[calls];
            if (!std.mem.eql(u8, std.mem.span(name), e.symbol)) wrong = true;
            if (@intFromPtr(mod) != (if (e.control == null) @as(usize, 10) else 20)) wrong = true;
            out.* = @ptrFromInt(30 + calls);
            calls += 1;
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
    };
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleGetFunction = Mock.function;
    d.api.cuStreamSynchronize = Mock.sync;
    const a = std.testing.allocator;
    const gpu: check.Gpu = .{ .d = &d, .ctx = undefined, .gpa = a, .io = undefined };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var kernels: std.ArrayList(core.contract.gates.Kernel) = .empty;
    defer kernels.deinit(a);
    var set: contract.Set = .{};
    defer set.deinit();
    try set.add(gpu, stream, .{ .d = &d, .handle = @ptrFromInt(10) }, .{ .d = &d, .handle = @ptrFromInt(20) }, &kernels);
    try std.testing.expectEqual(@as(usize, 3), kernels.items.len);
    try std.testing.expectEqual(@as(usize, 3), Mock.calls);
    try std.testing.expect(!Mock.wrong);
    for (kernels.items, contract.entries) |k, e| {
        try std.testing.expectEqualStrings(e.name, k.spec.name);
        try std.testing.expectEqual(e.control, k.control);
        try std.testing.expect(k.vt.replaced != null and k.vt.check != null and k.vt.release != null);
        try std.testing.expectEqual(core.contract.spec.Elem.raw, k.spec.out);
        try std.testing.expectEqual(core.contract.spec.Accum.none, k.spec.accum);
        try std.testing.expect(k.spec.replaces != null);
        var found: usize = 0;
        for (core.contract.registry.all_cuda) |r| if (std.mem.eql(u8, e.symbol, r.launch.symbol)) {
            found += 1;
            try std.testing.expectEqual(@as(u32, 32), r.launch.threads);
            try std.testing.expectEqual(@as(u32, 0), r.launch.dynamic);
            try std.testing.expect(!r.geometry_pending);
            for (r.usage) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
            try std.testing.expectEqual(@as(usize, 5), try core.contract.registry.checkCuda(a, &.{r}, true));
        };
        try std.testing.expectEqual(@as(usize, 1), found);
    }
}
