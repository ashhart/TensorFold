//! Plain-round capture ordering and graph-before-communicator abort on a recording driver.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
const t = std.testing;
const round = @import("cuda_round_graph.zig");
const parts = @import("cuda_batch_state.zig");
const ple = @import("cuda_ple.zig");
const collective = @import("cuda_collective.zig");
const dims = 2560;

const Fake = struct {
    var capturing = false;
    var captures: usize = 0;
    var capture_copies: usize = 0;
    var instantiates: usize = 0;
    var launches: usize = 0;
    var copies: usize = 0;
    var token: i32 = 0;
    var position: i32 = 0;
    var seen: [2]i32 = undefined;
    var order: [4]enum { graph_destroy, comm_abort } = undefined;
    var order_len: usize = 0;
    var allocations: [32]?[]align(16) u8 = @splat(null);

    fn alloc(out: *?*anyopaque, n: usize, _: u32) callconv(.c) abi.Result {
        for (&allocations) |*slot| if (slot.* == null) {
            slot.* = t.allocator.alignedAlloc(u8, .@"16", n) catch return 2;
            out.* = slot.*.?.ptr;
            return 0;
        };
        return 2;
    }
    fn free(ptr: ?*anyopaque) callconv(.c) abi.Result {
        for (&allocations) |*slot| if (slot.*) |bytes| {
            if (@intFromPtr(bytes.ptr) == @intFromPtr(ptr)) {
                t.allocator.free(bytes);
                slot.* = null;
                return 0;
            }
        };
        unreachable;
    }
    fn copy(dst: u64, ptr: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        std.debug.assert(!capturing);
        const bytes = @as([*]const u8, @ptrCast(ptr.?))[0..n];
        if (dst == 1) token = std.mem.bytesToValue(i32, bytes[0..4]);
        if (dst == 2) position = std.mem.bytesToValue(i32, bytes[0..4]);
        copies += 1;
        return 0;
    }
    fn event(out: *abi.Event, _: u32) callconv(.c) abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn eventOk(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn contextSync() callconv(.c) abi.Result {
        return 0;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        capture_copies = copies;
        capturing = true;
        captures += 1;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        capturing = false;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn status(_: abi.Stream, out: *abi.CaptureStatus) callconv(.c) abi.Result {
        out.* = if (capturing) .active else .none;
        return 0;
    }
    fn instantiate(out: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        out.* = @ptrFromInt(2);
        instantiates += 1;
        return 0;
    }
    fn destroyGraph(_: abi.Graph) callconv(.c) abi.Result {
        return 0;
    }
    fn destroyExec(_: abi.GraphExec) callconv(.c) abi.Result {
        order[order_len] = .graph_destroy;
        order_len += 1;
        return 0;
    }
    fn abort(_: cuda.nccl.Comm) callconv(.c) c_int {
        order[order_len] = .comm_abort;
        order_len += 1;
        return 0;
    }
    fn upload(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn launch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        std.debug.assert(!capturing and copies == 4 * (launches + 1));
        seen[launches] = token;
        launches += 1;
        return 0;
    }
    fn driver() cuda.Driver {
        capturing = false;
        captures = 0;
        instantiates = 0;
        launches = 0;
        copies = 0;
        order_len = 0;
        var d: cuda.Driver = undefined;
        d.api.cuMemHostAlloc = alloc;
        d.api.cuMemFreeHost = free;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuEventCreate = event;
        d.api.cuEventRecord = record;
        d.api.cuEventQuery = eventOk;
        d.api.cuEventSynchronize = eventOk;
        d.api.cuEventDestroy_v2 = eventOk;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = contextSync;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuStreamIsCapturing = status;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphDestroy = destroyGraph;
        d.api.cuGraphExecDestroy = destroyExec;
        d.api.cuGraphUpload = upload;
        d.api.cuGraphLaunch = launch;
        return d;
    }
};

const Scratch = struct {
    cap: usize = 1,
    ids: [1]i32 = .{0},
    positions: [4]i32 = @splat(0),
    emb: [dims]u16 = undefined,
    xs: [dims / 32]f32 = undefined,
    ple_emb: u64 = 3,
    ple_xs: u64 = 4,
    waits: usize = 0,
    fences: usize = 0,
    pub fn ready(s: *Scratch) !void {
        s.waits += 1;
    }
    pub fn uploaded(s: *Scratch) !void {
        s.fences += 1;
    }
    pub fn hostIds(s: *Scratch, n: usize) []i32 {
        return s.ids[0..n];
    }
    pub fn hostPos(s: *Scratch, n: usize) []i32 {
        return s.positions[0..n];
    }
    pub fn hostEmb(s: *Scratch, n: usize) []u16 {
        return s.emb[0 .. n * dims];
    }
    pub fn hostXs(s: *Scratch, n: usize) []f32 {
        return s.xs[0 .. n * (dims / 32)];
    }
};
const Pass = struct {
    s: *Scratch,
    stream: cuda.Stream,
    io: std.Io = t.io,
    m: struct { ngram: *const ple.Ngram, pocket: ?*anyopaque = null },
    seq: parts.SeqState,
    lanes: []const parts.Lane,
    prepared: bool = false,
    staged: bool = false,
    pos0: usize = 0,
    chunks: usize = 0,
    pub fn upload(p: *Pass, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(p.stream.d, dst, bytes, p.stream);
    }
    pub fn stage(p: *Pass, ids: []const u32) !void {
        if (!p.prepared) try p.s.ready();
        for (p.s.hostIds(ids.len), ids) |*out, id| out.* = @intCast(id);
        try p.upload(1, std.mem.sliceAsBytes(p.s.hostIds(ids.len)));
        @memset(&p.s.positions, 0);
        p.s.positions[0] = @intCast(p.lanes[0].pos);
        try p.upload(2, std.mem.sliceAsBytes(&p.s.positions));
        if (!p.prepared) try p.s.uploaded();
    }
    pub fn chunk(p: *Pass, ids: []const u32, _: usize, _: ?[]u8) !void {
        if (!p.staged) try p.stage(ids);
        p.chunks += 1;
    }
};

const Buf = struct { ptr: u64 };
const linear_n = @import("cuda_prompt_wt.zig").linear_n;
const attn_n = @import("cuda_prompt_wt.zig").attn_n;

/// A sequence's owner, length and the buffer addresses a round's lane takes from it.
const Seq = struct {
    rank_id: u64,
    fed: usize = 10,
    cap: usize = 1024,
    conv: [linear_n]Buf = undefined,
    state: [linear_n]Buf = undefined,
    kc: [attn_n]Buf = undefined,
    vc: [attn_n]Buf = undefined,
    ikc: [attn_n]Buf = undefined,
    pooled: [attn_n]Buf = undefined,
    tail: Buf = undefined,
    pg: @import("cuda_glue.zig").Pg = .{},

    fn at(owner: u64, base: u64) Seq {
        var q: Seq = .{ .rank_id = owner };
        var a = base;
        inline for (.{ "conv", "state", "kc", "vc", "ikc", "pooled" }) |f| {
            for (&@field(q, f)) |*b| {
                b.ptr = a;
                a += 0x10000;
            }
        }
        q.tail.ptr = a;
        return q;
    }

    fn settle(q: *Seq, spare: *[linear_n]Buf) void {
        for (&q.state, spare) |*a, *b| std.mem.swap(Buf, a, b);
    }
};

fn spares(base: u64) [linear_n]Buf {
    var s: [linear_n]Buf = undefined;
    for (&s, 0..) |*b, i| b.ptr = base + i * 0x10000;
    return s;
}

fn keyOf(seqs: []const *const Seq) !round.Key {
    return round.Key.init(seqs, 1, 2, 3);
}

test "decode graph keys: owner order, attention and score tile seams, each lane's buffers and length" {
    var first: Seq = .at(1, 0x1000_0000);
    var second: Seq = .at(2, 0x2000_0000);
    const base = try keyOf(&.{ &first, &second });
    try t.expectEqual(base, try keyOf(&.{ &first, &second }));
    try t.expect(!std.meta.eql(base, try keyOf(&.{ &second, &first })));
    const alone = try keyOf(&.{&first});
    var spare = spares(0x3000_0000);
    second.settle(&spare);
    try t.expect(!std.meta.eql(base, try keyOf(&.{ &first, &second })));
    try t.expectEqual(alone, try keyOf(&.{&first}));
    second.settle(&spare);
    try t.expectEqual(base, try keyOf(&.{ &first, &second }));
    second.tail.ptr += 0x100;
    try t.expect(!std.meta.eql(base, try keyOf(&.{ &first, &second })));
    second.tail.ptr -= 0x100;
    second.cap = 2048;
    try t.expect(!std.meta.eql(base, try keyOf(&.{ &first, &second })));
    first.fed = 511;
    const before = try keyOf(&.{&first});
    first.fed = 512;
    try t.expect(!std.meta.eql(before, try keyOf(&.{&first})));
    first.fed = 2048;
    const dense = try keyOf(&.{&first});
    first.fed = 2051;
    try t.expect(!std.meta.eql(dense, try keyOf(&.{&first})));
    first.fed = 2303;
    const score = try keyOf(&.{&first});
    first.fed = 2304;
    try t.expect(!std.meta.eql(score, try keyOf(&.{&first})));
}

fn ngram(g: *ple.Ngram, buffers: *[2][3][]u8) !void {
    g.* = .{ .gpa = t.allocator, .scale = 1, .hash = try ple.Hash.init(.{
        .vocab = 100,
        .ngram = 3,
        .per = 1,
        .base = 13,
        .divisor = 2,
        .shards = 2,
        .seed = 1234,
        .eos = 99,
        .ple_dim = dims,
        .ple_index = 0,
    }) };
    for (0..2) |shard| {
        const rows: usize = @intCast(g.hash.starts[shard + 1] - g.hash.starts[shard]);
        for ([_]usize{ rows * g.hash.dh / 2, rows * g.hash.dh / 16, rows * g.hash.dh / 16 }, 0..) |n, i| {
            buffers[shard][i] = try t.allocator.alloc(u8, n);
            @memset(buffers[shard][i], 0);
        }
        for (0..rows * g.hash.dh / 32) |i| {
            const val: f32 = @floatFromInt(shard * 32 + i / (g.hash.dh / 32) + 1);
            std.mem.writeInt(u16, buffers[shard][2][i * 2 ..][0..2], @truncate(@as(u32, @bitCast(val)) >> 16), .little);
        }
        g.w[shard] = buffers[shard][0];
        g.s[shard] = buffers[shard][1];
        g.b[shard] = buffers[shard][2];
    }
}

test "rank graphs: plain rounds stage uploads before capture and before every replay" {
    var d = Fake.driver();
    defer {
        cuda.copy_source.synchronizeContext(&d) catch unreachable;
        std.debug.assert(!cuda.copy_source.keepDriver(&d));
    }
    var cache: round.Cache = .{};
    defer cache.clear();
    var scratch: Scratch = .{};
    var buffers: [2][3][]u8 = @splat(@splat(&.{}));
    defer for (buffers) |row| for (row) |b| t.allocator.free(b);
    var g: ple.Ngram = undefined;
    try ngram(&g, &buffers);
    var history = [2]i64{ 99, 99 };
    var seq: parts.SeqState = undefined;
    seq.history = &history;
    var lanes = [_]parts.Lane{.{ .seq = seq, .pos = 5 }};
    var p: Pass = .{ .s = &scratch, .stream = .{ .d = &d, .handle = null }, .m = .{ .ngram = &g }, .seq = seq, .lanes = &lanes };
    const key: round.Key = .{ .count = 1, .scratch = 1, .scores = 2, .output = 3 };
    try cache.run(key, &p, &.{7});
    const handle = cache.exec.?.handle;
    try t.expectEqual(@as(i32, 5), Fake.position);
    lanes[0].pos = 6;
    p.prepared = false;
    p.staged = false;
    try cache.run(key, &p, &.{12});
    try t.expectEqual(handle, cache.exec.?.handle);
    try t.expectEqualSlices(i32, &.{ 7, 12 }, &Fake.seen);
    try t.expectEqualSlices(i64, &.{ 7, 12 }, &history);
    try t.expectEqual(@as(i32, 6), Fake.position);
    try t.expectEqual(@as(usize, 1), Fake.captures);
    try t.expectEqual(@as(usize, 4), Fake.capture_copies);
    try t.expectEqual(@as(usize, 1), Fake.instantiates);
    try t.expectEqual(@as(usize, 1), p.chunks);
    try t.expectEqual(@as(usize, 2), scratch.waits);
    try t.expectEqual(@as(usize, 2), scratch.fences);
    try t.expect(!Fake.capturing);
    try t.expectEqual([2]u64{ 1, 1 }, [2]u64{ cache.captures, cache.replays });
    cache.clear();
    try t.expectEqual([2]u64{ 1, 1 }, [2]u64{ cache.captures, cache.replays });
}

test "rank graphs: collective abort releases the decode graph before ncclCommAbort" {
    var d = Fake.driver();
    var cache: round.Cache = .{ .exec = .{ .d = &d, .handle = @ptrFromInt(2) }, .key = .{ .count = 1, .scratch = 1, .scores = 2, .output = 3 } };
    defer cache.clear();
    const invalid_socket: std.posix.socket_t = if (@import("builtin").os.tag == .windows) @ptrFromInt(std.math.maxInt(usize)) else -1;
    var c: collective.Collective = .{ .lib = undefined, .api = undefined, .comm = @as(cuda.nccl.Comm, @ptrFromInt(3)), .link = .{ .fd = invalid_socket }, .timeout_ms = 1 };
    c.api.ncclCommAbort = Fake.abort;
    if (@hasField(collective.Collective, "round_graph")) c.round_graph = &cache;
    c.abort();
    try t.expectEqual(@as(usize, 2), Fake.order_len);
    try t.expectEqual(.graph_destroy, Fake.order[0]);
    try t.expectEqual(.comm_abort, Fake.order[1]);
    try t.expect(c.comm == null and c.failed and cache.exec == null and cache.key == null);
    if (@hasField(collective.Collective, "round_graph")) try t.expect(c.round_graph == null);
    c.abort();
    try t.expectEqual(@as(usize, 2), Fake.order_len);
}
