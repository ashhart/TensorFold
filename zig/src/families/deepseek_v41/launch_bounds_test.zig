//! Actual resolver calls and launches must agree with the measured resource contract.
const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const contract = @import("core").contract;
const catalog = @import("catalog.zig");
const t = std.testing;

test "DeepSeek bounds: grouped resolver raises the dynamic shared-memory limit" {
    try catalog.collect();
    for (catalog.entries[0..catalog.count]) |e| {
        if (!std.mem.eql(u8, e.image, "dsv41_exl3_experts") or !std.mem.startsWith(u8, e.name[0..e.len], "_ZN15tf_exl3_experts12group_kernel")) continue;
        try t.expect(e.opt_in);
        try t.expectEqual(@as(u32, 49152), e.shared);
        return;
    }
    return error.MissingGroupKernel;
}

test "DeepSeek bounds: declarations match actual shared-memory opt-in calls" {
    try catalog.collect();
    for (catalog.entries[0..catalog.count]) |e| {
        if (!std.mem.startsWith(u8, e.image, "dsv41_")) continue;
        const r = for (contract.registry.all_cuda) |r| {
            if (std.mem.eql(u8, r.launch.image, e.image) and std.mem.eql(u8, r.launch.symbol, e.name[0..e.len])) break r;
        } else return error.MissingContract;
        try t.expectEqual(e.opt_in, r.launch.opt_in);
    }
}

test "DeepSeek bounds: supplied 13.2 usage fits every applicable target and planted excesses refuse" {
    const a = t.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var problems: usize = 0;
    var checked: usize = 0;
    for (contract.registry.all_cuda) |r| {
        if (!std.mem.startsWith(u8, r.launch.image, "dsv41_")) continue;
        const usage = contract.registry.usageFor(r, .{ .major = 13, .minor = 2 });
        for (contract.limits.sms, usage) |sm, u| {
            if (contract.registry.targetDecision(r, sm, u) != .applicable) continue;
            problems += try contract.limits.cuda(r.launch, u, sm, &out, a);
            checked += 1;
            if (sm.sm == 121 and std.mem.eql(u8, r.launch.image, "dsv41_exl3_experts") and r.launch.dynamic == 49152) {
                var bad = r.launch;
                bad.opt_in = false;
                try t.expectEqual(@as(usize, 1), try contract.limits.cuda(bad, u, sm, &out, a));
                out.clearRetainingCapacity();
            }
            if (sm.sm == 121 and std.mem.eql(u8, r.launch.image, "dsv41_l2pf")) {
                var bad = r.launch;
                bad.threads = 1024;
                try t.expectEqual(@as(usize, 1), try contract.limits.cuda(bad, u, sm, &out, a));
                out.clearRetainingCapacity();
            }
        }
    }
    if (problems > 0) std.debug.print("DeepSeek bounds: {d} problems across {d} applicable measured pairs\n{s}", .{ problems, checked, out.items });
    try t.expect(checked > 0);
    try t.expectEqual(@as(usize, 0), problems);
}

test "DeepSeek bounds: L2 prefetch refuses excess threads before either driver launch" {
    const abi = cuda.abi;
    const Fake = struct {
        var calls: usize = 0;
        var threads: u32 = 0;
        fn launch(_: abi.Function, _: u32, _: u32, _: u32, x: u32, _: u32, _: u32, _: u32, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            calls += 1;
            threads = x;
            return abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
    var f: dk.ops.Functions = undefined;
    f.segments = .{ .{ .d = &driver, .handle = @ptrFromInt(1) }, .{ .d = &driver, .handle = @ptrFromInt(2) } };
    const ops: dk.ops.Ops = .{ .f = &f, .s = .{ .d = &driver, .handle = null } };
    Fake.calls = 0;
    for ([_]bool{ false, true }) |bulk| {
        for ([_]usize{ 32, 256 }) |threads| {
            try ops.l2Segments(0x1000, 1, 4, threads, 4096, bulk);
            try t.expectEqual(threads, Fake.threads);
        }
        const before = Fake.calls;
        for ([_]usize{ 257, 1024 }) |threads|
            try t.expectError(error.Unsupported, ops.l2Segments(0x1000, 1, 4, threads, 4096, bulk));
        try t.expectEqual(before, Fake.calls);
    }
    try t.expectEqual(@as(usize, 4), Fake.calls);
}
