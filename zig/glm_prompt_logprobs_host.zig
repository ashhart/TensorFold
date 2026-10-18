//! Controls at the real prompt-logprob owner, head policy and batching boundaries.
const std = @import("std");
const glm = @import("glm");
const lanes = @import("lanes");
test "GLM prompt rows: reader cannot allocate after ranks enter the pass" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var s: lanes.Stream = undefined;
    s.prompt_rows = .empty;
    s.prompt_len = 3000;
    s.prompt_logprobs = 0;
    defer s.prompt_rows.deinit(std.testing.allocator);
    var sink = glm.prompt_sink.Sink{ .gpa = failing.allocator(), .s = &s };
    try sink.prepare();
    failing.fail_index = failing.alloc_index;
    var words: [glm.prompt_logprobs.row_words * 128]u32 = @splat(0);
    var picks: [128]u32 = @splat(7);
    var left: usize = 2999;
    while (left > 0) {
        const n: usize = @min(left, 128);
        try glm.prompt_sink.Sink.read(&sink, words[0 .. n * glm.prompt_logprobs.row_words], picks[0..n]);
        left -= n;
    }
    try std.testing.expectEqual(@as(usize, 2999), s.prompt_rows.items.len);
}
test "GLM prompt rows: tiled packed heads use the forward policy" {
    var e: glm.engine.Engine = undefined;
    e.m.tiled = true;
    var packs: glm.mm.Packed = .{};
    e.m.packs = &packs;
    const set = glm.prompt_logprobs.projectionSet(&e);
    try std.testing.expect(set.mm == &e.m.mm);
    try std.testing.expect(set.packs == &packs);
    try std.testing.expect(set.prompt);
}
test "GLM prompt rows: requests for rows cannot enter the unscored batch path" {
    const S = struct { images: []const u8 = &.{}, reuse: struct { saved: ?*u8 = null } = .{}, prompt_len: usize = 19, prompt_logprobs: ?u8 = null };
    var s: S = .{};
    try std.testing.expect(glm.lanes.canBatch(&s));
    s.prompt_logprobs = 2;
    try std.testing.expect(!glm.lanes.canBatch(&s));
}

test "GLM prompt rows: reservation failure happens before a prompt step" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var s: lanes.Stream = undefined;
    s.prompt_rows = .empty;
    s.prompt_len = 3000;
    s.prompt_logprobs = 0;
    var sink = glm.prompt_sink.Sink{ .gpa = failing.allocator(), .s = &s };
    try std.testing.expectError(error.OutOfMemory, sink.prepare());
    try std.testing.expectEqual(@as(usize, 0), s.prompt_rows.items.len);
}

const cuda = @import("cuda");
const Output = glm.prompt_logprobs.Output;
const Mapping = struct {
    var data: [2][glm.prompt_logprobs.out_bytes]u8 align(16) = undefined;
    var allocating: usize = 0;
    var refuse: [2]bool = .{ true, false };
    var maps: [2]usize = .{ 0, 0 };
    var frees: [2]usize = .{ 0, 0 };
    fn rank(p: ?*anyopaque) usize {
        return if (p == @as(?*anyopaque, @ptrCast(&data[0]))) 0 else 1;
    }
    fn alloc(p: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        if (n != data[0].len) return 2;
        p.* = &data[allocating];
        return 0;
    }
    fn map(p: *u64, ptr: ?*anyopaque, _: c_uint) callconv(.c) cuda.abi.Result {
        const r = rank(ptr);
        maps[r] += 1;
        if (refuse[r]) return cuda.abi.error_not_ready;
        p.* = 0x10000 + r * 0x10000;
        return 0;
    }
    fn free(p: ?*anyopaque) callconv(.c) cuda.abi.Result {
        frees[rank(p)] += 1;
        return 0;
    }
};
fn driver() cuda.Driver {
    var d: cuda.Driver = undefined;
    d.api.cuMemHostAlloc = Mapping.alloc;
    d.api.cuMemHostGetDevicePointer_v2 = Mapping.map;
    d.api.cuMemFreeHost = Mapping.free;
    return d;
}

const Heads = struct {
    var broadcasts: [2]usize = .{ 0, 0 };
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn copy(_: u64, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn group() callconv(.c) cuda.nccl.Result {
        return 0;
    }
    fn broadcast(_: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: c_int, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        broadcasts[@intFromPtr(comm) - 1] += 1;
        return 0;
    }
    fn score(d: *cuda.Driver, output: Output, rank_: usize) !void {
        d.api.cuLaunchKernel = launch;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        var lib: cuda.nccl.Library = undefined;
        lib.api.ncclGroupStart = group;
        lib.api.ncclGroupEnd = group;
        lib.api.ncclBroadcast = broadcast;
        var e: glm.engine.Engine = undefined;
        e.prompt_tried = true;
        e.prompt_bufs = null;
        e.prompt_out = output;
        e.d = d;
        e.c.hidden = 64;
        e.c.vocab = 16;
        e.share.rank = rank_;
        e.share.vocab = .{ .lo = rank_ * 8, .hi = (rank_ + 1) * 8 };
        var spans = [_]glm.shard.Span{ .{ .lo = 0, .hi = 8 }, .{ .lo = 8, .hi = 16 } };
        e.spans = &spans;
        e.lib = &lib;
        e.comm = @ptrFromInt(rank_ + 1);
        e.group = null;
        e.stream = .{ .d = d, .handle = null };
        e.m.tiled = false;
        e.m.packs = null;
        e.m.glue.bmm_decode = .{ .d = d, .handle = @ptrFromInt(1) };
        e.bp.fnormed = 0x10000;
        e.bp.kda.mm_part = 0x20000;
        e.bd.logits = 0x30000;
        e.w.head = 0x40000;
        e.smp.d = d;
        e.smp.vocab = 16;
        e.smp.full = .{ .d = d, .ptr = 0x50000, .len = 1024 };
        try glm.prompt_logprobs.run(&e, .{}, &.{7});
    }
};

test "GLM prompt rows: rank zero mapping failure is a startup refusal while follower mapping succeeds" {
    var d = [_]cuda.Driver{ driver(), driver() };
    var outputs: [2]?Output = .{ null, null };
    defer for (&outputs) |*o| if (o.*) |*p| p.free();
    var errors: [2]?anyerror = .{ null, null };
    Heads.broadcasts = .{ 0, 0 };
    Mapping.refuse = .{ true, false };
    Mapping.maps = .{ 0, 0 };
    Mapping.frees = .{ 0, 0 };
    const before = cuda.memory.usage(false).host;
    for (0..2) |r| {
        Mapping.allocating = r;
        outputs[r] = Output.init(&d[r]) catch |err| blk: {
            errors[r] = err;
            break :blk null;
        };
    }
    if (errors[0] == null and errors[1] == null) {
        for (0..2) |r| Heads.score(&d[r], outputs[r].?, r) catch |err| {
            errors[r] = err;
        };
    }
    try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, &Heads.broadcasts);
    try std.testing.expectEqual(@as(?anyerror, error.NotReady), errors[0]);
    try std.testing.expect(outputs[0] == null);
    try std.testing.expect(errors[1] == null and outputs[1] != null);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, &Mapping.maps);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, &Mapping.frees);
    try std.testing.expectEqual(before + glm.prompt_logprobs.out_bytes, cuda.memory.usage(false).host);
    outputs[1].?.free();
    outputs[1] = null;
    try std.testing.expectEqual(before, cuda.memory.usage(false).host);
}

test "GLM prompt rows: both cached addresses survive later driver mapping refusal" {
    var d = [_]cuda.Driver{ driver(), driver() };
    var outputs: [2]?Output = .{ null, null };
    defer for (&outputs) |*o| if (o.*) |*p| p.free();
    Mapping.refuse = .{ false, false };
    Mapping.maps = .{ 0, 0 };
    for (0..2) |r| {
        Mapping.allocating = r;
        outputs[r] = try Output.init(&d[r]);
    }
    Mapping.refuse = .{ true, true };
    Heads.broadcasts = .{ 0, 0 };
    for (0..2) |r| {
        try Heads.score(&d[r], outputs[r].?, r);
        try std.testing.expectEqual(@as(u64, 0x10000) + r * 0x10000, outputs[r].?.address);
        outputs[r].?.slice(u32)[0] = @intCast(7 + r);
    }
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, &Mapping.maps);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, &Heads.broadcasts);
}
