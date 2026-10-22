//! The diagnostic page owners and their admission charge, through real constructors and a fake driver.
const std = @import("std");
const cuda = @import("cuda");
const pages = @import("cuda_pages.zig");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");

const Fake = struct {
    var allocs: usize = 0;
    var fail_alloc: usize = 0;
    var fills: usize = 0;
    var pending = false;
    var early_free = false;
    var fail_copy = false;
    var fail_wait = false;
    var contexts: usize = 0;
    var wrong_stream = false;
    const handle: cuda.abi.Stream = @ptrFromInt(5);

    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        allocs += 1;
        if (allocs == fail_alloc) return cuda.abi.error_not_ready;
        out.* = 0x10000 * allocs;
        return cuda.abi.success;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        early_free = early_free or pending;
        return cuda.abi.success;
    }
    fn fill(_: u64, _: u8, _: usize, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        fills += 1;
        pending = true;
        wrong_stream = wrong_stream or s != handle;
        return cuda.abi.success;
    }
    fn upload(_: u64, _: ?*const anyopaque, _: usize, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending = true;
        wrong_stream = wrong_stream or s != handle;
        return if (fail_copy) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn wait(s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        wrong_stream = wrong_stream or s != handle;
        if (fail_wait) return cuda.abi.error_not_ready;
        pending = false;
        return cuda.abi.success;
    }
    fn context() callconv(.c) cuda.abi.Result {
        contexts += 1;
        pending = false;
        return cuda.abi.success;
    }
    fn driver() cuda.Driver {
        allocs = 0;
        fail_alloc = 0;
        fills = 0;
        pending = false;
        early_free = false;
        fail_copy = false;
        fail_wait = false;
        contexts = 0;
        wrong_stream = false;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = fill;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        return d;
    }
};

test "diagnostic pages: rejected host IDs or device table allocation queues no fill" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = Fake.handle };
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, pages.testPages(fail.allocator(), &d, stream, 17, 16));
    try std.testing.expect(Fake.fills == 0 and !Fake.pending and !Fake.early_free);
    d = Fake.driver();
    Fake.fail_alloc = 2;
    try std.testing.expectError(error.NotReady, pages.testPages(std.testing.allocator, &d, stream, 17, 16));
    try std.testing.expect(Fake.fills == 0 and !Fake.pending and !Fake.early_free);
}

test "diagnostic pages: upload and final wait errors prove completion before freeing any owner" {
    for ([_]bool{ false, true }) |failed_wait| {
        var d = Fake.driver();
        const stream: cuda.Stream = .{ .d = &d, .handle = Fake.handle };
        Fake.fail_copy = !failed_wait;
        Fake.fail_wait = failed_wait;
        try std.testing.expectError(error.NotReady, pages.testPages(std.testing.allocator, &d, stream, 17, 16));
        try std.testing.expect(!Fake.pending and !Fake.early_free and !Fake.wrong_stream);
        try std.testing.expectEqual(@as(usize, @intFromBool(failed_wait)), Fake.contexts);
    }
}

test "diagnostic pages: initialization and deinit complete on the owning stream before returning or freeing" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = Fake.handle };
    var p = try pages.testPages(std.testing.allocator, &d, stream, 49, 48);
    try std.testing.expect(!Fake.pending and !Fake.wrong_stream);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, p.ids);
    Fake.pending = true;
    Fake.fail_wait = true;
    p.deinit(std.testing.allocator);
    try std.testing.expect(!Fake.pending and !Fake.early_free);
    try std.testing.expectEqual(@as(usize, 1), Fake.contexts);
}

test "diagnostic pages: malformed or oversized sizes and capacities refuse before allocation" {
    for ([_]usize{ 0, 6, wt.window + 4, std.math.maxInt(usize) }) |tokens| {
        try std.testing.expect(!pages.valid(tokens));
        try std.testing.expectError(error.BadFlashNextPageTest, pages.checkedLayout(17, tokens));
    }
    for ([_]usize{ 0, wt.window + 1, std.math.maxInt(usize) }) |cap| {
        try std.testing.expectError(error.BadFlashNextPageTest, pages.checkedLayout(cap, 16));
    }
    for ([_]usize{ 4, 8, 16, 48, wt.window }) |tokens| {
        const l = try pages.checkedLayout(wt.window, tokens);
        try std.testing.expect(l.pages > 0 and l.arena / 2 <= std.math.maxInt(i64));
        try std.testing.expectEqual(l.arena + l.table, l.total);
    }
}

test "diagnostic pages: admission includes full final pages and the device table" {
    const cap = 49;
    const l = try pages.checkedLayout(cap, 48);
    try std.testing.expectEqual(@as(usize, 2), l.pages);
    try std.testing.expectEqual(2 * pages.pageBytes(48), l.arena);
    try std.testing.expectEqual(@as(usize, 8), l.table);
    const contiguous = wt.attn_n * (cap * (2 * prompt.kv_row_bytes + wt.index_dim * 2) + prompt.pooledBytes(cap));
    try std.testing.expectEqual(prompt.seqBytes(cap) - contiguous + l.total, try prompt.seqBytesPaged(cap, 48));
    try std.testing.expect(try prompt.seqBytesPaged(cap, 48) > prompt.seqBytes(cap));
    try std.testing.expectEqual(prompt.seqBytes(cap), try prompt.seqBytesPaged(cap, 0));
}
