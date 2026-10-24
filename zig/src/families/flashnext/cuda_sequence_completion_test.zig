//! Real sequence construction/rollback and release under queued destination writes, using only driver fakes.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const t = std.testing;
const Fake = struct {
    var allocations: usize = 0;
    var frees: usize = 0;
    var early: usize = 0;
    var syncs: usize = 0;
    var fills: usize = 0;
    var deny_after: usize = 0;
    var fill_failure: usize = 0;
    var pending: [256]bool = @splat(false);
    var pin: ?[]align(16) u8 = null;
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        pin = t.allocator.alignedAlloc(u8, .@"16", len) catch return cuda.abi.error_not_ready;
        out.* = pin.?.ptr;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        t.allocator.free(pin.?);
        pin = null;
        return 0;
    }
    fn event(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn query(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return cuda.abi.error_not_ready;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn copyAndRefuseNext(ptr: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending[ptr / 256] = true;
        const u = cuda.usage(false);
        _ = cuda.memory.setAllocationLimit(u.device + u.host) catch unreachable;
        return 0;
    }
    fn alloc(out: *u64, len: usize) callconv(.c) cuda.abi.Result {
        allocations += 1;
        out.* = allocations * 256;
        if (allocations == deny_after) {
            const u = cuda.usage(false);
            _ = cuda.memory.setAllocationLimit(u.device + u.host + len) catch unreachable;
        }
        return 0;
    }
    fn fill(ptr: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending[ptr / 256] = true;
        fills += 1;
        return if (fills == fill_failure) cuda.abi.error_not_ready else 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        @memset(&pending, false);
        return 0;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        if (pending[ptr / 256]) early += 1;
        frees += 1;
        return 0;
    }
    fn init(deny: usize, fail_fill: usize) cuda.Driver {
        allocations = 0;
        frees = 0;
        early = 0;
        syncs = 0;
        fills = 0;
        deny_after = deny;
        fill_failure = fail_fill;
        @memset(&pending, false);
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = fill;
        d.api.cuStreamSynchronize = sync;
        return d;
    }
};

test "Flash Next completion: allocation refusals drain fills before every constructor-prefix release" {
    const old_limit = try cuda.memory.setAllocationLimit(null);
    defer _ = cuda.memory.setAllocationLimit(old_limit) catch unreachable;
    const before = cuda.usage(false);
    // Cover failures inside each array and between arrays, including after the last pooled fill.
    for ([_]usize{ 1, 2, 35, 36, 37, 71, 72, 73, 83, 84, 85, 95, 96, 97, 107, 108, 109, 119, 120 }) |deny| {
        var d = Fake.init(deny, 0);
        var stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(325) };
        var e: prompt.Engine = undefined;
        e.gpa = t.allocator;
        e.driver = &d;
        e.stream = &stream;
        e.eos = 1;
        e.round_graph = .{};
        try t.expectError(error.OutOfDeviceMemory, e.newSeq(4));
        _ = try cuda.memory.setAllocationLimit(null);
        try t.expectEqual(deny, Fake.allocations);
        try t.expectEqual(Fake.allocations, Fake.frees);
        if (Fake.early != 0) std.log.warn("completion control: allocs={d}, fills={d}, frees={d}, syncs={d}", .{ Fake.allocations, Fake.fills, Fake.frees, Fake.syncs });
        try t.expectEqual(@as(usize, 0), Fake.early);
        try t.expect(Fake.syncs > 0);
        try t.expectEqual(before.device, cuda.usage(false).device);
    }
}

test "Flash Next completion: failed asynchronous fill and an interrupted prefill both drain before release" {
    for ([_]usize{ 1, 37, 73, 121 }) |fail_fill| {
        var d = Fake.init(0, fail_fill);
        var stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(326) };
        var e: prompt.Engine = undefined;
        e.gpa = t.allocator;
        e.driver = &d;
        e.stream = &stream;
        e.eos = 1;
        e.round_graph = .{};
        try t.expectError(error.NotReady, e.newSeq(4));
        try t.expectEqual(Fake.allocations, Fake.frees);
        if (Fake.early != 0) std.log.warn("completion control: allocs={d}, fills={d}, frees={d}, syncs={d}", .{ Fake.allocations, Fake.fills, Fake.frees, Fake.syncs });
        try t.expectEqual(@as(usize, 0), Fake.early);
    }
    var d = Fake.init(0, 0);
    var stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(327) };
    var e: prompt.Engine = undefined;
    e.gpa = t.allocator;
    e.driver = &d;
    e.stream = &stream;
    e.eos = 1;
    e.round_graph = .{};
    var seq = try e.newSeq(4);
    // Model a queued prefill kernel writing every owned sequence buffer before request refusal.
    for (Fake.pending[1 .. Fake.allocations + 1]) |*pending| pending.* = true;
    seq.deinit(t.allocator);
    try t.expectEqual(Fake.allocations, Fake.frees);
    try t.expectEqual(@as(usize, 0), Fake.early);
}

test "Flash Next completion: draft constructor fill failures and queued window writes drain before release" {
    const Draft = @import("cuda_mtp_draft.zig").Draft;
    for ([_]usize{ 1, 2, 3, 4 }) |fail_fill| {
        var d = Fake.init(0, fail_fill);
        const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(328) };
        try t.expectError(error.NotReady, Draft.init(t.allocator, &d, stream, 4, 8, 1));
        try t.expectEqual(Fake.allocations, Fake.frees);
        try t.expectEqual(@as(usize, 0), Fake.early);
    }
    var d = Fake.init(0, 0);
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(329) };
    var draft = try Draft.init(t.allocator, &d, stream, 4, 8, 1);
    @memset(Fake.pending[1 .. Fake.allocations + 1], true);
    draft.deinit();
    try t.expectEqual(Fake.allocations, Fake.frees);
    try t.expectEqual(@as(usize, 0), Fake.early);
}

test "Flash Next completion: scratch growth refusal after queued inverse-frequency upload drains before rollback" {
    const before = cuda.usage(false);
    const old_limit = try cuda.memory.setAllocationLimit(null);
    defer _ = cuda.memory.setAllocationLimit(old_limit) catch unreachable;
    var d = Fake.init(0, 0);
    d.api.cuMemHostAlloc = Fake.hostAlloc;
    d.api.cuMemFreeHost = Fake.hostFree;
    d.api.cuEventCreate = Fake.event;
    d.api.cuEventRecord = Fake.record;
    d.api.cuEventQuery = Fake.query;
    d.api.cuEventDestroy_v2 = Fake.destroy;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.copyAndRefuseNext;
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(330) };
    try t.expectError(error.OutOfDeviceMemory, prompt.batch.Scratch.alloc(t.allocator, &d, stream, 1));
    const cleanup_syncs = Fake.syncs;
    _ = try cuda.memory.setAllocationLimit(null);
    try stream.synchronize(); // also clean the old-main control's deliberately unfinished copy source
    try t.expectEqual(@as(usize, 0), Fake.early);
    try t.expect(cleanup_syncs > 0);
    try t.expectEqual(Fake.allocations, Fake.frees);
    try t.expect(Fake.pin == null);
    try t.expectEqual(before.device, cuda.usage(false).device);
    try t.expectEqual(before.host, cuda.usage(false).host);
}
