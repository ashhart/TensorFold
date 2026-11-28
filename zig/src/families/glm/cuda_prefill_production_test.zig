//! Production-entry prefill ledger regressions over host recording fixtures.
const std = @import("std");
const cuda = @import("cuda");
const ledger = @import("cuda_prefill_trace.zig");
const prof = @import("cuda_prof.zig");
const t = std.testing;

pub const Fake = struct {
    pub var creates: usize = 0;
    pub var live: usize = 0;
    pub var ready = false;
    pub var failure: enum { none, create, record, elapsed } = .none;
    pub var bad_time = false;
    pub var clock: f32 = 0;
    pub var times: [4096]f32 = undefined;
    pub var calls: std.ArrayList(u64) = .empty;
    pub var pinned: ?[]align(16) u8 = null;
    fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        if (failure == .create) return 2;
        creates += 1;
        live += 1;
        out.* = @ptrFromInt(creates);
        return 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
        live -= 1;
        return 0;
    }
    fn record(event: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
        if (failure == .record) return 1;
        times[@intFromPtr(event)] = clock;
        return 0;
    }
    fn query(_: cuda.abi.Event) callconv(.c) c_int {
        return if (ready) 0 else 600;
    }
    fn elapsed(out: *f32, first: cuda.abi.Event, last: cuda.abi.Event) callconv(.c) c_int {
        if (failure == .elapsed) return 1;
        out.* = if (bad_time) std.math.nan(f32) else times[@intFromPtr(last)] - times[@intFromPtr(first)];
        return 0;
    }
    pub fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuEventCreate = create;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuEventRecord = record;
        d.api.cuEventQuery = query;
        d.api.cuEventElapsedTime = elapsed;
        d.api.cuLaunchKernel = launch;
        d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(launch);
        // Synchronization and wait function pointers remain undefined: any added call fails this test.
        return d;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        calls.appendSlice(t.allocator, &.{ @intFromPtr(f), gx, gy, gz, bx, by, bz, shared, @intFromPtr(stream) }) catch unreachable;
        for ("88888844", 0..) |size, i| {
            const at = args.?[i].?;
            const value: u64 = if (size == '8') @as(*const u64, @ptrCast(@alignCast(at))).* else @as(*const u32, @ptrCast(@alignCast(at))).*;
            calls.append(t.allocator, value) catch unreachable;
        }
        clock += 2;
        return 0;
    }
    pub fn work(stream: cuda.Stream) !void {
        const glue = @import("cuda_glue.zig");
        var kernels: glue.Kernels = undefined;
        kernels.rmsnorm_4096 = .{ .d = stream.d, .handle = @ptrFromInt(0xabc) };
        prof.context(null, 1, 2, .mixer);
        const kda = prof.span(null, stream, .kda);
        defer kda.end();
        try glue.rmsnorm(.{ .glue = &kernels }, stream, 0x100, 4096, 0x200, 0x300, 4096, 0x400, 1e-6, 8192, 4096);
    }
    fn sharedLaunch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        return 0;
    }
    fn copyAsync(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn copyDevice(_: u64, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn download(dst: ?*anyopaque, _: u64, n: usize) callconv(.c) c_int {
        @memset(@as([*]u8, @ptrCast(dst.?))[0..n], 0);
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) c_int {
        pinned = t.allocator.alignedAlloc(u8, .@"16", n) catch return 600;
        out.* = pinned.?.ptr;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) c_int {
        t.allocator.free(pinned.?);
        pinned = null;
        return 0;
    }
    fn contextDevice(out: *cuda.abi.Device) callconv(.c) c_int {
        out.* = 0;
        return 0;
    }
    fn deviceAttribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) c_int {
        out.* = 0;
        return 0;
    }
    fn gather(_: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    pub fn sharedDriver() cuda.Driver {
        var d = driver();
        d.api.cuLaunchKernel = sharedLaunch;
        d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(sharedLaunch);
        d.api.cuMemcpyHtoDAsync_v2 = copyAsync;
        d.api.cuMemcpyDtoDAsync_v2 = copyDevice;
        d.api.cuMemcpyDtoH_v2 = download;
        d.api.cuStreamSynchronize = sync;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuCtxGetDevice = contextDevice;
        d.api.cuDeviceGetAttribute = deviceAttribute;
        return d;
    }
};

const BatchState = struct {
    pos: usize = 0,
    mtp_len: usize = 0,
    pub fn setMtpLen(_: *BatchState, _: cuda.Stream, _: usize) !void {}
};
const BatchSeq = struct { st: BatchState = .{} };
const BatchSeg = struct { st: *BatchState, row0: usize, rows: usize };
const BatchBuffers = struct { logits: u64 = 0x1000, fnormed: u64 = 0x2000 };
const BatchRank = struct { w: *u8, st: *BatchState, b: *BatchBuffers, segs: []const BatchSeg = &.{} };
const BatchDrafter = struct {
    pub fn taps(_: *BatchDrafter, _: bool) ?u8 {
        return null;
    }
    pub fn promptSegment(_: *BatchDrafter, _: anytype, _: u32, _: usize, _: usize, _: usize, _: usize) !void {}
};
const BatchEngine = struct {
    capacity: usize,
    stream: cuda.Stream,
    d: *const cuda.Driver,
    prefill_trace: ?*ledger.Trace,
    states: *[2]BatchSeq,
    cur: *BatchSeq,
    w: u8 = 0,
    bp: BatchBuffers = .{},
    m: u8 = 0,
    pieces: u8 = 0,
    settings: struct { split: bool = false, ring: bool = false, kda_wide: bool = false } = .{},
    drafter: ?*BatchDrafter = null,
    share: struct { vocab: @import("cuda_shard.zig").Span } = .{ .vocab = .{ .lo = 0, .hi = 2 } },
    c: struct { hidden: usize = 4096 } = .{},
    mtp: bool = false,
    pub fn bind(e: *BatchEngine, slot: u32) !void {
        e.cur = &e.states[slot];
    }
    pub fn gate(_: *BatchEngine, _: ?void) !void {}
    pub fn exchange(_: *BatchEngine) u8 {
        return 0;
    }
    pub fn draw(_: *BatchEngine, _: u64, _: usize, _: usize, _: u64, out: []u32) !void {
        out[0] = 0;
    }
    pub fn holdPrompt(_: *BatchEngine, _: u64) !void {}
};
const BatchForward = struct {
    d: *const cuda.Driver,
    m: *u8,
    stream: cuda.Stream,
    ranks: *[1]BatchRank,
    exchange: u8,
    pieces: ?*u8,
    split: bool,
    ring: bool,
    kda_wide: bool,
    taps: ?u8,
    pub fn run(f: *BatchForward, _: []const i32) !void {
        try Fake.work(f.stream);
    }
    pub fn commit(_: *BatchForward, _: usize) !void {}
    pub fn mtpMode(_: *BatchForward, _: []const i32, _: []const u64, _: bool) !void {}
};

fn prefix(tmp: std.testing.TmpDir) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
}

test "GLM prefill ledger: cuda_prompts.pass carries both fresh streams in its own one request" {
    Fake.calls = .empty;
    defer Fake.calls.deinit(t.allocator);
    Fake.failure = .none;
    Fake.clock = 0;
    Fake.creates = 0;
    Fake.live = 0;
    Fake.ready = false;
    Fake.bad_time = false;
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try prefix(tmp);
    defer t.allocator.free(dir);
    const path = try std.fmt.allocPrint(t.allocator, "{s}/batched", .{dir});
    defer t.allocator.free(path);
    const trace = try ledger.Trace.init(&d, t.io, path, 1, 2, &.{ .{ .kind = .dsa, .ffn = .dense }, .{ .kind = .kda, .ffn = .moe } });
    defer trace.deinit();
    var states: [2]BatchSeq = .{ .{}, .{} };
    var e: BatchEngine = .{ .capacity = 4096, .stream = stream, .d = &d, .prefill_trace = trace, .states = &states, .cur = &states[0] };
    const left: [1024]u32 = @splat(1);
    const right: [1024]u32 = @splat(2);
    const prompts = @import("cuda_engine.zig");
    const parts = [_]prompts.Prompt{ .{ .slot = 0, .ids = &left }, .{ .slot = 1, .ids = &right } };
    var firsts: [2]u32 = undefined;
    try @import("cuda_prompts.zig").pass(&e, &parts, &firsts, BatchForward, BatchRank);
    try t.expectEqualSlices(u32, &.{ 0, 0 }, &firsts);
    try t.expect(!trace.written);
    Fake.ready = true;
    ledger.poll(trace);
    try t.expect(trace.written);
    const data = try tmp.dir.readFileAlloc(t.io, "batched.rank1.json", t.allocator, .limited(65536));
    defer t.allocator.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, data, .{});
    defer parsed.deinit();
    try t.expectEqual(@as(i64, 0), parsed.value.object.get("slot").?.integer);
    try t.expectEqual(@as(i64, 1024), parsed.value.object.get("prompt_tokens").?.integer);
    try t.expectEqual(@as(usize, 4), parsed.value.object.get("rows").?.array.items.len);
    try t.expect(parsed.value.object.get("complete").?.bool);
}

test "GLM prefill ledger: Engine.verifyShared polls several complete windows without direct ledger help" {
    Fake.calls = .empty;
    defer Fake.calls.deinit(t.allocator);
    Fake.failure = .none;
    Fake.clock = 0;
    Fake.creates = 0;
    Fake.live = 0;
    Fake.ready = false;
    Fake.bad_time = false;
    defer {
        if (Fake.pinned) |bytes| t.allocator.free(bytes);
        Fake.pinned = null;
    }
    var d = Fake.sharedDriver();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try prefix(tmp);
    defer t.allocator.free(dir);
    const path = try std.fmt.allocPrint(t.allocator, "{s}/windows", .{dir});
    defer t.allocator.free(path);
    const trace = try ledger.Trace.init(&d, t.io, path, 1, 2, &.{ .{ .kind = .dsa, .ffn = .dense }, .{ .kind = .kda, .ffn = .moe } });
    defer trace.deinit();
    const first = ledger.enter(trace, stream, 7, 0, 8192, 16384, false);
    try Fake.work(stream);
    first.end();
    const second = ledger.enter(trace, stream, 7, 8192, 8192, 16384, false);
    try Fake.work(stream);
    second.end();
    try t.expect(!trace.written);

    const engine = @import("cuda_engine.zig");
    const fw = @import("cuda_forward.zig");
    const Seq = @import("cuda_engine_types.zig").Seq;
    var seqs: [2]Seq = undefined;
    var seq_ptrs: [2]?*Seq = undefined;
    for (&seqs, &seq_ptrs) |*q, *p| {
        q.* = .{ .st = undefined };
        q.st.capacity = 64;
        q.st.ring_rows = 64;
        q.st.base = 0;
        q.st.pos = 0;
        p.* = q;
    }
    var e: engine.Engine = undefined;
    e.gpa = t.allocator;
    e.d = &d;
    e.stream = stream;
    e.w = .{ .gpa = t.allocator, .config = .{}, .share = undefined };
    e.w.share = try @import("cuda_shard.zig").Share.of(&e.w.config, 0, 1);
    e.c = e.w.config;
    e.share = e.w.share;
    e.settings = .{};
    e.m = undefined;
    e.m.index = null;
    e.m.tiled = false;
    e.m.packs = null;
    e.m.dsel = false;
    e.m.dscore = null;
    e.m.mla = null;
    e.m.tfq3 = null;
    inline for (@typeInfo(@TypeOf(e.m.glue)).@"struct".field_names) |name| {
        @field(e.m.glue, name).d = &d;
        @field(e.m.glue, name).handle = @ptrFromInt(1);
    }
    e.prefill_trace = trace;
    e.profiling_prompt = null;
    e.drafter = null;
    e.l2pf = null;
    e.group = null;
    e.images = null;
    e.local = &.{};
    e.structured = false;
    e.seqs.items = &seq_ptrs;
    e.seqs.capacity = seq_ptrs.len;
    e.cur = &seqs[0];
    var shared: fw.Buffers = undefined;
    shared.rows = engine.shared_rows;
    shared.prefill = false;
    shared.ids = 0x1000;
    shared.x = 0x2000;
    shared.hidden = 0x3000;
    shared.fnormed = 0x4000;
    shared.fxs = 0x5000;
    shared.logits = 0x6000;
    shared.kda.mm_part = 0x7000;
    e.bs = shared;
    e.pairs = .{ .d = &d, .ptr = 0x9000, .len = 16 * engine.shared_rows, .borrowed = true };
    var lib: cuda.nccl.Library = undefined;
    lib.api.ncclAllGather = Fake.gather;
    e.lib = &lib;
    e.comm = null;
    const ids: [1]u32 = .{1};
    var left: [1]u32 = undefined;
    var right: [1]u32 = undefined;
    const parts = [_]engine.Part{ .{ .slot = 0, .ids = &ids, .out = &left }, .{ .slot = 1, .ids = &ids, .out = &right } };
    Fake.ready = true;
    try e.verifyShared(&parts);
    try t.expect(trace.written);
    const data = try tmp.dir.readFileAlloc(t.io, "windows.rank1.json", t.allocator, .limited(65536));
    defer t.allocator.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, data, .{});
    defer parsed.deinit();
    try t.expect(parsed.value.object.get("complete").?.bool);
    for (parsed.value.object.get("rows").?.array.items) |row| {
        try t.expectEqualStrings("complete", row.object.get("event_status").?.string);
    }
}
