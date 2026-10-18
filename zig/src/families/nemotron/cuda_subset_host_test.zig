//! The actual MTP constructor over recording CUDA calls, with a sentinel on every free.
const std = @import("std");
const cuda = @import("cuda");
const Head = @import("cuda_mtp.zig").Head;
const Engine = @import("cuda_engine.zig").Engine;
const a = std.testing.allocator;
const Fake = struct {
    var calls: usize = 0;
    var fail: usize = 0;
    var live: usize = 0;
    var pending: bool = false;
    var blocked: bool = false;
    var unsafe: usize = 0;
    var host: [1 << 18]u8 align(16) = undefined;
    fn next() cuda.abi.Result {
        const n = calls;
        calls += 1;
        return if (n == fail) cuda.abi.error_not_ready else 0;
    }
    fn alloc(p: *u64, _: usize) callconv(.c) cuda.abi.Result {
        const rc = next();
        if (rc == 0) {
            p.* = 4096;
            live += 1;
        }
        return rc;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        if (pending) unsafe += 1;
        live -= 1;
        return 0;
    }
    fn hostAlloc(p: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        const rc = next();
        if (rc == 0) {
            p.* = &host;
            live += 1;
        }
        return rc;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        if (pending) unsafe += 1;
        live -= 1;
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const rc = next();
        if (rc == 0) pending = true;
        return rc;
    }
    fn event(p: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        const rc = next();
        if (rc == 0) {
            p.* = @ptrFromInt(1);
            live += 1;
        }
        return rc;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return next();
    }
    fn drop(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        if (pending) unsafe += 1;
        live -= 1;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (blocked) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemsetD8Async = fill;
        d.api.cuStreamSynchronize = sync;
        d.api.cuEventCreate = event;
        d.api.cuEventRecord = record;
        d.api.cuEventDestroy_v2 = drop;
        return d;
    }
    fn engine(ctx: *const cuda.Context, d: *const cuda.Driver) Engine {
        var e: Engine = undefined;
        e.gpa = a;
        e.ctx = ctx;
        e.stream = .{ .d = d, .handle = @ptrFromInt(2) };
        e.c = std.mem.zeroes(@import("config.zig").Config);
        e.c.hidden = 256;
        e.c.kv_heads = 2;
        e.c.head_dim = 64;
        e.c.vocab = 128;
        e.max_len = 64;
        e.w.draft_count = 64;
        e.w.draft_coverage = null;
        e.w.mtp = @as(@import("cuda_weights.zig").Mtp, undefined);
        e.head = null;
        return e;
    }
};
test "subset head: every MTP constructor allocation and event failure drains before freeing its prefix" {
    var d = Fake.driver();
    const ctx: cuda.Context = .{ .d = &d, .handle = null, .device = 0 };
    for (0..38) |at| {
        Fake.calls = 0;
        Fake.fail = at;
        Fake.live = 0;
        Fake.pending = false;
        Fake.blocked = false;
        Fake.unsafe = 0;
        var e = Fake.engine(&ctx, &d);
        if (Head.init(&e)) |h| h.deinit() else |err| try std.testing.expectEqual(error.NotReady, err);
        try std.testing.expectEqual(@as(usize, 0), Fake.unsafe);
        try std.testing.expectEqual(@as(usize, 0), Fake.live);
        try std.testing.expect(e.head == null);
    }
}
test "subset head: a failed MTP drain keeps every live device host and event owner for retry" {
    var d = Fake.driver();
    const ctx: cuda.Context = .{ .d = &d, .handle = null, .device = 0 };
    Fake.calls = 0;
    Fake.fail = std.math.maxInt(usize);
    Fake.live = 0;
    Fake.pending = false;
    Fake.blocked = false;
    Fake.unsafe = 0;
    var e = Fake.engine(&ctx, &d);
    const h = try Head.init(&e);
    const live = Fake.live;
    Fake.blocked = true;
    try std.testing.expect(!h.tryDeinit());
    try std.testing.expectEqual(live, Fake.live);
    try std.testing.expect(e.head == h);
    Fake.blocked = false;
    try std.testing.expect(h.tryDeinit());
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
    try std.testing.expectEqual(@as(usize, 0), Fake.unsafe);
}

const Reader = struct {
    fail: bool = false,
    calls: usize = 0,
    pub fn read(r: *Reader, out: []u8, input: []const u8) !void {
        r.calls += 1;
        if (r.fail) return error.InjectedRead;
        @memcpy(out, input);
    }
};
test "subset head: gather validates all rows before reading and preserves row order and guarded output" {
    const w = @import("cuda_weights.zig");
    var r: Reader = .{};
    var loader = .{ .gpa = a, .src = &r };
    const t = w.made(.u8, 3, 2, &.{ 1, 2, 3, 4, 5, 6 });
    var out: [6]u8 = @splat(0xee);
    try w.gatherRows(&loader, t, &.{ 2, 0 }, &out);
    try std.testing.expectEqualSlices(u8, &.{ 5, 6, 1, 2, 0xee, 0xee }, &out);
    r.calls = 0;
    try std.testing.expectError(error.DraftIdPastVocabulary, w.gatherRows(&loader, t, &.{ 0, 3 }, &out));
    try std.testing.expectError(error.UnexpectedTensor, w.gatherRows(&loader, t, &.{ 0, 1 }, out[0..3]));
    var bad = t;
    bad.shape[0] = 0;
    try std.testing.expectError(error.UnexpectedTensor, w.gatherRows(&loader, bad, &.{0}, &out));
    bad = t;
    bad.shape[0] = 4;
    try std.testing.expectError(error.UnexpectedTensor, w.gatherRows(&loader, bad, &.{0}, &out));
    try std.testing.expectEqual(@as(usize, 0), r.calls);
    r.fail = true;
    try std.testing.expectError(error.InjectedRead, w.gatherRows(&loader, t, &.{0}, &out));
}
const projection = @import("cuda_mtp_projection.zig");
const Span = struct { vocab: projection.Vocab };
const P = struct {
    data: [128]u16 = @splat(0xffff),
    const W = struct { n: usize, bias: u16 };
    pub fn dense(p: *P, input: u64, _: u64, w: W, output: u64, rows: usize) !void {
        const first = (input - 8192) / (64 * 2);
        const off = (output - 4096) / 2;
        for (0..rows) |r| {
            for (0..w.n) |c| p.data[off + r * w.n + c] = @intCast((first + r) * 100 + c + w.bias);
        }
    }
};
test "subset head: mixed and shrinking head batches equal each row alone with guarded logits tails" {
    const heads = [2]P.W{ .{ .n = 3, .bias = 10 }, .{ .n = 7, .bias = 20 } };
    const cases = [_][3]Span{ .{ .{ .vocab = .subset }, .{ .vocab = .full }, .{ .vocab = .subset } }, .{ .{ .vocab = .full }, .{ .vocab = .subset }, .{ .vocab = .full } }, .{ .{ .vocab = .subset }, .{ .vocab = .subset }, .{ .vocab = .subset } }, .{ .{ .vocab = .full }, .{ .vocab = .full }, .{ .vocab = .full } } };
    for (cases) |spans| for (1..4) |live| {
        var batch: P = .{};
        const stride = try projection.project(&batch, spans[0..live], heads, .{ 3, 7 }, 8192, 0, 4096, 64);
        for (spans[0..live], 0..) |s, k| {
            var solo: P = .{};
            const width = try projection.project(&solo, &.{s}, heads, .{ 3, 7 }, 8192 + k * 128, 0, 4096, 64);
            try std.testing.expectEqualSlices(u16, solo.data[0..width], batch.data[k * stride ..][0..width]);
            for (batch.data[k * stride + width ..][0 .. stride - width]) |v| try std.testing.expectEqual(@as(u16, 0xffff), v);
        }
        for (batch.data[live * stride ..]) |v| try std.testing.expectEqual(@as(u16, 0xffff), v);
    };
}
test "subset head: head choice follows restored context and cannot inherit a neighbor's vocabulary" {
    var d = Fake.driver();
    const ctx: cuda.Context = .{ .d = &d, .handle = null, .device = 0 };
    var e = Fake.engine(&ctx, &d);
    e.w.draft_coverage = try @import("draft_ids.zig").Coverage.init(a, 128, &.{1});
    defer e.w.draft_coverage.?.deinit(a);
    var h: Head = undefined;
    h.e = &e;
    var context: [20]u32 = @splat(1);
    try std.testing.expectEqual(projection.Vocab.full, h.vocab(context[0..15]));
    try std.testing.expectEqual(projection.Vocab.subset, h.vocab(&context));
    context[19] = 2;
    try std.testing.expectEqual(projection.Vocab.full, h.vocab(&context));
    var neighbor: [20]u32 = @splat(1);
    try std.testing.expectEqual(projection.Vocab.subset, h.vocab(&neighbor));
    try std.testing.expectEqual(projection.Vocab.full, h.vocab(&context));
    context[19] = 1;
    context[0] = 2;
    try std.testing.expectEqual(projection.Vocab.subset, h.vocab(&context));
}

test {
    _ = @import("cuda_subset_restore_test.zig");
}
