//! The real cache restore, sequence bind, lane draft dispatch, captured head selection and held-token readback.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;
const Backend = @import("cuda_lanes.zig").Cuda;
const state = @import("cuda_state.zig");
const reuse = @import("cuda_reuse.zig");
const t = std.testing;
pub const Fake = struct {
    pub var e: *Engine = undefined;
    pub var copies: usize = 0;
    var graph: usize = 0;
    var graph_cache: u64 = 0;
    var pins: [16][4096]u8 align(16) = undefined;
    pub var used: [16]bool = @splat(false);
    fn alloc(out: *?*anyopaque, bytes: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        if (bytes > pins[0].len) return cuda.abi.error_not_ready;
        for (&used, 0..) |*live, i| if (!live.*) {
            live.* = true;
            out.* = &pins[i];
            return 0;
        };
        return cuda.abi.error_not_ready;
    }
    fn free(ptr: ?*anyopaque) callconv(.c) cuda.abi.Result {
        for (&pins, &used) |*bytes, *live| if (@intFromPtr(ptr) == @intFromPtr(bytes)) {
            std.debug.assert(live.*);
            live.* = false;
            return 0;
        };
        unreachable;
    }
    fn region(ptr: u64, len: usize) []u8 {
        return @as([*]u8, @ptrFromInt(ptr))[0..len];
    }
    fn copy(dst: u64, src: u64, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        @memcpy(region(dst, len), region(src, len));
        return 0;
    }
    fn upload(dst: u64, src: ?*const anyopaque, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memcpy(region(dst, len), @as([*]const u8, @ptrCast(src.?))[0..len]);
        return 0;
    }
    fn download(dst: ?*anyopaque, src: u64, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..len], region(src, len));
        return 0;
    }
    fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn event(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn capture(_: cuda.abi.Stream, out: *cuda.abi.CaptureStatus) callconv(.c) cuda.abi.Result {
        out.* = .none;
        return 0;
    }
    fn launch(handle: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        graph = @intFromPtr(handle);
        graph_cache = e.b.k_cache;
        const ids: [*]u32 = @ptrFromInt(e.b.ids);
        ids[1] = @intCast(100 + graph);
        return 0;
    }
    pub fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuMemcpyDtoHAsync_v2 = download;
        d.api.cuMemHostAlloc = alloc;
        d.api.cuMemFreeHost = free;
        d.api.cuEventCreate = create;
        d.api.cuEventSynchronize = event;
        d.api.cuEventQuery = event;
        d.api.cuEventDestroy_v2 = event;
        d.api.cuEventRecord = record;
        d.api.cuStreamSynchronize = sync;
        d.api.cuStreamIsCapturing = capture;
        d.api.cuGraphLaunch = launch;
        return d;
    }
};
pub const Buffers = struct {
    k: [128]u8 align(16) = @splat(0),
    v: [128]u8 align(16) = @splat(0),
    hk: [128]u8 align(16) = @splat(0),
    hv: [128]u8 align(16) = @splat(0),
    probs: [32]u32 = @splat(0),
    ids: [32]u32 = @splat(0),
    sampled: [32]u32 = @splat(0),
    hidden: [64]u16 align(16) = @splat(0),
    rule: [32]u8 align(8) = @splat(0),
    pub fn seq(b: *Buffers) state.Seq {
        var s: state.Seq = .{ .arena = null, .ptr = @splat(0), .head = .{ @intFromPtr(&b.hk), @intFromPtr(&b.hv), @intFromPtr(&b.probs) } };
        s.ptr[0] = @intFromPtr(&b.k);
        s.ptr[1] = @intFromPtr(&b.v);
        inline for (state.seq_fields, 0..) |name, i| {
            if (comptime std.mem.eql(u8, name, "hidden")) s.ptr[i] = @intFromPtr(&b.hidden);
            if (comptime std.mem.eql(u8, name, "rule")) s.ptr[i] = @intFromPtr(&b.rule);
            if (comptime std.mem.eql(u8, name, "ids")) s.ptr[i] = @intFromPtr(&b.ids);
            if (comptime std.mem.eql(u8, name, "sampled")) s.ptr[i] = @intFromPtr(&b.sampled);
        }
        return s;
    }
};
test "restored head: a neighbor cannot leave the restored stream on a stale head or sequence" {
    var d = Fake.driver();
    const ctx: cuda.Context = .{ .d = &d, .handle = null, .device = 0 };
    const e = try t.allocator.create(Engine);
    defer t.allocator.destroy(e);
    Fake.e = e;
    Fake.copies = 0;
    Fake.used = @splat(false);
    e.* = undefined;
    e.gpa = t.allocator;
    e.ctx = &ctx;
    e.stream = .{ .d = &d, .handle = null };
    defer e.stream.synchronize() catch unreachable;
    e.c = std.mem.zeroes(@import("config.zig").Config);
    e.c.hidden = 64;
    e.c.kv_heads = 1;
    e.c.head_dim = 1;
    e.c.layers = 1;
    e.c.kinds[0] = .attention;
    e.c.vocab = 128;
    e.max_len = 64;
    e.k.d = &d;
    e.b = undefined;
    e.b.lane_part = 0;
    e.w.draft_coverage = try @import("draft_ids.zig").Coverage.init(t.allocator, 128, &.{1});
    defer e.w.draft_coverage.?.deinit(t.allocator);
    e.pos = 0;
    e.parity = 0;
    e.prev_keep = 0;
    e.rows = 0;
    e.sampling = null;
    e.slot = 0;
    e.decide = -1;
    e.epoch = 0;
    e.stats = .{};
    var own: Buffers = .{};
    var other: Buffers = .{};
    e.own = own.seq();
    e.bound = &e.own;
    var neighbor = other.seq();
    neighbor.pos = 7;
    neighbor.head_pos = 6;
    defer neighbor.graphs.deinit(t.allocator);
    inline for (0..2) |v| try neighbor.graphs.put(t.allocator, 2001 + v * 33, .{ .d = &d, .handle = @ptrFromInt(v + 1) });
    var h: Head = undefined;
    var pinned: [256]u32 align(16) = @splat(0);
    var meta: [33]u32 = @splat(0);
    h.e = e;
    h.pos = 0;
    h.pinned = .{ .d = &d, .bytes = std.mem.sliceAsBytes(&pinned) };
    h.meta = @intFromPtr(&meta);
    h.copied = .{ .d = &d, .handle = @ptrFromInt(1) };
    h.ready[0] = h.copied;
    h.first_graphs = @splat(@splat(null));
    inline for (0..2) |v| h.first_graphs[v][1] = .{ .d = &d, .handle = @ptrFromInt(v + 1) };
    e.head = &h;
    e.bind(&e.own);
    var b: Backend = undefined;
    b.gpa = t.allocator;
    b.e = e;
    b.head = &h;
    b.lanes = .empty;
    defer b.lanes.deinit(t.allocator);
    b.pinned = .{ .d = &d, .bytes = std.mem.sliceAsBytes(&pinned) };
    const covered = @as([20]u32, @splat(1));
    const uncovered = @as([19]u32, @splat(1)) ++ [_]u32{2};
    var n = try lanes.Stream.init(t.allocator, .{ .id = "neighbor", .prompt = &covered, .max_new = 8 });
    defer n.deinit(t.allocator);
    try b.lanes.put(t.allocator, &n, .{ .seq = &neighbor });
    var saved_bytes: [160]u8 align(16) = @splat(0xa5);
    var saved: reuse.Copy = .{ .mamba = .{ .d = &d, .ptr = 0, .len = 0 }, .kv = .{ .d = &d, .ptr = @intFromPtr(&saved_bytes), .len = saved_bytes.len }, .head = true, .pos = 20, .parity = 1, .rows = 1 };
    var target: reuse.Target = .{ .e = e, .head = &h };
    const cases = [_]struct { prompt: []const u32, graph: usize }{ .{ .prompt = &covered, .graph = 1 }, .{ .prompt = &uncovered, .graph = 2 }, .{ .prompt = &covered, .graph = 1 } };
    for (cases) |case| {
        var subject = try lanes.Stream.init(t.allocator, .{ .id = "restored", .prompt = case.prompt, .max_new = 8, .reuse = .{ .saved = &saved, .at = 20 } });
        defer subject.deinit(t.allocator);
        try b.lanes.put(t.allocator, &subject, .{ .seq = &e.own });
        defer _ = b.lanes.remove(&subject);
        e.bind(&e.own);
        try reuse.restore(&target, null, subject.reuse.saved.?);
        try t.expectEqual(@as(usize, 20), e.pos);
        try t.expectEqual(@as(usize, 20), h.pos);
        try t.expectEqual(@as(u8, 0xa5), own.k[0]);
        try t.expectEqual(@as(u8, 0xa5), own.hk[0]);
        try b.backend().draft(&.{.{ .stream = &n, .follow = &.{}, .first = .{ .value = 1 }, .rows = null, .start = 7, .position = 8, .depth = 1 }});
        try t.expect(e.bound == &neighbor);
        try b.backend().draft(&.{.{ .stream = &subject, .follow = &.{}, .first = .{ .value = 1 }, .rows = null, .start = 20, .position = 21, .depth = 1 }});
        try t.expect(e.bound == &e.own);
        try t.expectEqual(@as(usize, 21), h.pos);
        try t.expectEqual(case.graph, Fake.graph);
        try t.expectEqual(@intFromPtr(&own.k), Fake.graph_cache);
        var held: [1]u32 = undefined;
        try b.backend().vtable.held.?(b.backend().ptr, &subject, &held);
        try t.expectEqual(@as(u32, @intCast(100 + case.graph)), held[0]);
        neighbor.pos = 7;
        neighbor.head_pos = 6;
    }
    try t.expectEqual(@as(usize, 12), Fake.copies);
    try e.stream.synchronize();
    for (Fake.used) |live| try t.expect(!live);
}

test {
    _ = @import("cuda_subset_batch_test.zig");
}
