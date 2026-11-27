//! Real folded follower control over recording CUDA/NCCL calls, without devices.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const wire = glm.follow;
const Engine = glm.engine.Engine;
const abi = cuda.abi;
const t = std.testing;
const Params = ?[*]?*anyopaque;

const Recording = struct {
    var frames: [4][256]u8 = undefined;
    var lengths: [4]usize = undefined;
    var sent: usize = 0;
    var received: usize = 0;
    var receiving = false;
    var launches: usize = 0;
    var reductions: usize = 0;
    var outcomes: [3]f64 = @splat(0);
    var pinned: [4]?[]align(16) u8 = @splat(null);
    var fail_launch = false;

    fn reset() void {
        sent = 0;
        received = 0;
        receiving = false;
        launches = 0;
        reductions = 0;
        outcomes = @splat(0);
        fail_launch = false;
    }

    fn get(comptime T: type, params: Params, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(params.?[i].?))).*;
    }

    fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, params: Params, _: Params) callconv(.c) abi.Result {
        launches += 1;
        if (fail_launch) return abi.error_not_ready;
        const dst = get(u64, params, 0);
        const n = get(u32, params, 1);
        const words = get(wire.Words, params, 2);
        const out: [*]u8 = @ptrFromInt(dst);
        @memcpy(out[0 .. n * 4], std.mem.sliceAsBytes(words.w[0..n]));
        return 0;
    }

    fn broadcast(ptr: u64, _: u64, n: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: abi.Stream) callconv(.c) cuda.nccl.Result {
        if (n > frames[0].len) return 1;
        const out: [*]u8 = @ptrFromInt(ptr);
        if (receiving) {
            if (received >= sent or lengths[received] != n) return 1;
            @memcpy(out[0..n], frames[received][0..n]);
            received += 1;
        } else {
            if (sent == frames.len) return 1;
            @memcpy(frames[sent][0..n], out[0..n]);
            lengths[sent] = n;
            sent += 1;
        }
        return 0;
    }

    fn reduce(ptr: u64, _: u64, n: usize, _: cuda.nccl.DataType, _: cuda.nccl.RedOp, _: cuda.nccl.Comm, _: abi.Stream) callconv(.c) cuda.nccl.Result {
        if (n != 1 or reductions == outcomes.len) return 1;
        @as(*f64, @ptrFromInt(ptr)).* = outcomes[reductions];
        reductions += 1;
        return 0;
    }

    fn download(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) abi.Result {
        const out: [*]u8 = @ptrCast(dst.?);
        const in: [*]const u8 = @ptrFromInt(src);
        @memcpy(out[0..n], in[0..n]);
        return 0;
    }

    fn upload(dst: u64, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        const out: [*]u8 = @ptrFromInt(dst);
        const in: [*]const u8 = @ptrCast(src.?);
        @memcpy(out[0..n], in[0..n]);
        return 0;
    }

    fn set(dst: u64, value: c_uint, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        const out: [*]u32 = @ptrFromInt(dst);
        @memset(out[0..n], value);
        return 0;
    }

    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) abi.Result {
        for (&pinned) |*slot| if (slot.* == null) {
            const bytes = t.allocator.alignedAlloc(u8, .@"16", n) catch return 1;
            slot.* = bytes;
            out.* = bytes.ptr;
            return 0;
        };
        return 1;
    }

    fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
        for (&pinned) |*slot| if (slot.*) |bytes| {
            if (@intFromPtr(bytes.ptr) != @intFromPtr(ptr.?)) continue;
            t.allocator.free(bytes);
            slot.* = null;
            return 0;
        };
        return 1;
    }

    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }

    fn eventRecord(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }

    fn event(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }

    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }

    fn device(dev: *abi.Device) callconv(.c) abi.Result {
        dev.* = 0;
        return 0;
    }

    fn attribute(value: *c_int, _: abi.DeviceAttribute, _: abi.Device) callconv(.c) abi.Result {
        value.* = 0;
        return 0;
    }
};

fn driver() cuda.Driver {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Recording.launch;
    d.api.cuLaunchKernelEx = glm.pdl.forwardEx(Recording.launch);
    d.api.cuMemcpyHtoDAsync_v2 = Recording.upload;
    d.api.cuMemcpyDtoH_v2 = Recording.download;
    d.api.cuMemsetD32Async = Recording.set;
    d.api.cuMemHostAlloc = Recording.hostAlloc;
    d.api.cuMemFreeHost = Recording.hostFree;
    d.api.cuEventCreate = Recording.eventCreate;
    d.api.cuEventRecord = Recording.eventRecord;
    d.api.cuEventQuery = Recording.event;
    d.api.cuEventSynchronize = Recording.event;
    d.api.cuEventDestroy_v2 = Recording.event;
    d.api.cuStreamSynchronize = Recording.sync;
    d.api.cuCtxGetDevice = Recording.device;
    d.api.cuDeviceGetAttribute = Recording.attribute;
    d.func_name = null;
    return d;
}

fn partialEngine(e: *Engine, d: *cuda.Driver, lib: *cuda.nccl.Library, word: *f64) void {
    e.gpa = t.allocator;
    e.io = t.io;
    e.d = d;
    e.c = .{};
    e.lib = lib;
    e.comm = null;
    e.stream = .{ .d = d, .handle = null };
    e.settings = .{};
    e.local = &.{};
    e.images = null;
    e.drafter = null;
    e.l2pf = null;
    e.pair = null;
    e.prompt_profs = @splat(null);
    e.profs = @splat(null);
    e.profiling_prompt = null;
    e.prompt_logprobs = null;
    e.heat_gate.bands = null;
    e.dropped = .empty;
    e.announce = 0;
    e.cap_rows = 0;
    e.index_split = false;
    e.share.rank = 1;
    e.share.world = 2;
    e.shared_buf = .{
        .d = d,
        .ptr = @intFromPtr(word),
        .len = @sizeOf(f64),
        .borrowed = true,
    };
    e.seqs = .empty;
}

// These states are intentionally partial: the fixture frees only the sequence pointer list, never State.deinit.
fn partialState(st: anytype, d: *cuda.Driver, arena: *[1]u8, pos: *[1]u32, capacity: usize, pending: usize) void {
    st.arena = .{ .buf = .{ .d = d, .ptr = @intFromPtr(arena), .len = arena.len, .borrowed = true } };
    st.capacity = capacity;
    st.pos = 0;
    st.pending = pending;
    st.prompt = false;
    st.pos_dev = @intFromPtr(pos);
    st.base = 0;
    st.ring_rows = capacity;
    st.kda_layers = 0;
}

fn foldedLink(link: *wire.Link, d: *cuda.Driver, lib: *cuda.nccl.Library, tensor: *[1]u8, control: *[256]u8, ids: []u32) void {
    link.* = .{
        .d = d,
        .lib = lib,
        .comm = null,
        .stream = .{ .d = d, .handle = null },
        .buf = .{ .d = d, .ptr = @intFromPtr(tensor), .len = tensor.len, .borrowed = true },
        .ids = ids,
        .gpa = t.allocator,
        .wait = false,
        .fold = true,
        .control_buf = .{ .d = d, .ptr = @intFromPtr(control), .len = control.len, .borrowed = true },
        .control_stream = .{ .d = d, .handle = @ptrFromInt(8) },
    };
}

fn appendSlots(e: *Engine, kept: anytype, running: anytype) !void {
    try e.seqs.appendNTimes(t.allocator, null, 3);
    try e.seqs.append(t.allocator, kept);
    try e.seqs.append(t.allocator, running);
}

test "GLM folded follow: keep commits before cancelled carried prefill and receiving continues" {
    Recording.reset();
    var d = driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = Recording.broadcast;
    lib.api.ncclAllReduce = Recording.reduce;
    cuda.download_stage.forget();
    defer cuda.download_stage.forget();
    defer {
        cuda.copy_source.synchronizeStream(&d, null) catch {};
        cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    }

    var tensor: [1]u8 = .{0};
    var control: [256]u8 = @splat(0);
    var wire_ids: [80]u32 = undefined;
    var link: wire.Link = undefined;
    foldedLink(&link, &d, &lib, &tensor, &control, &wire_ids);
    var word: f64 = 0;
    var e: Engine = undefined;
    partialEngine(&e, &d, &lib, &word);
    defer e.seqs.deinit(t.allocator);
    var kept: @TypeOf(e.cur.*) = .{ .st = undefined };
    var running: @TypeOf(e.cur.*) = .{ .st = undefined };
    var kept_arena: [1]u8 = .{0};
    var running_arena: [1]u8 = .{0};
    var kept_pos: [1]u32 = .{0};
    var running_pos: [1]u32 = .{0};
    partialState(&kept.st, &d, &kept_arena, &kept_pos, 64, 5);
    partialState(&running.st, &d, &running_arena, &running_pos, 64, 0);
    try appendSlots(&e, &kept, &running);

    const put: cuda.Function = .{ .d = &d, .handle = null };
    link.pending_keep = .{ .slot = 3, .rows = 5 };
    try link.send(put, .{ .kind = @backingInt(wire.Kind.prefill), .slot = 4 }, &.{21});
    try link.send(put, .{ .kind = @backingInt(wire.Kind.end) }, &.{});
    Recording.outcomes = .{ 0, 1, 0 };
    Recording.receiving = true;
    try wire.follow(&e, &link);

    try t.expectEqual(@as(usize, 5), kept.st.pos);
    try t.expectEqual(@as(usize, 0), kept.st.pending);
    try t.expectEqual(@as(u32, 5), kept_pos[0]);
    try t.expectEqual(@as(u32, 4), e.slot);
    try t.expect(e.cur == &running);
    try t.expectEqual(@as(usize, 3), Recording.reductions);
    try t.expectEqual(@as(usize, 2), Recording.sent);
    try t.expectEqual(@as(usize, 2), Recording.received);
    try t.expect(link.received_keep == null);
}

test "GLM folded follow: embedding NotReady escapes after keep and leaves carried verify pending" {
    Recording.reset();
    var d = driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = Recording.broadcast;
    lib.api.ncclAllReduce = Recording.reduce;
    cuda.download_stage.forget();
    defer cuda.download_stage.forget();
    defer {
        cuda.copy_source.synchronizeStream(&d, null) catch {};
        cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    }

    var tensor: [1]u8 = .{0};
    var control: [256]u8 = @splat(0);
    var wire_ids: [80]u32 = undefined;
    var link: wire.Link = undefined;
    foldedLink(&link, &d, &lib, &tensor, &control, &wire_ids);
    var word: f64 = 0;
    var token_ids: [1]u32 = .{0};
    var e: Engine = undefined;
    partialEngine(&e, &d, &lib, &word);
    defer e.seqs.deinit(t.allocator);
    e.w.config = .{};
    e.w.embed = 0;
    e.bd.rows = 1;
    e.bd.prefill = false;
    e.bd.ids = @intFromPtr(&token_ids);
    e.bd.x = 0;
    var kept: @TypeOf(e.cur.*) = .{ .st = undefined };
    var running: @TypeOf(e.cur.*) = .{ .st = undefined };
    var kept_arena: [1]u8 = .{0};
    var running_arena: [1]u8 = .{0};
    var kept_pos: [1]u32 = .{0};
    var running_pos: [1]u32 = .{0};
    partialState(&kept.st, &d, &kept_arena, &kept_pos, 64, 5);
    partialState(&running.st, &d, &running_arena, &running_pos, 64, 0);
    try appendSlots(&e, &kept, &running);

    const put: cuda.Function = .{ .d = &d, .handle = null };
    e.m.glue.put_words = put;
    e.m.glue.embed_b16 = put;
    e.m.index = null;
    e.m.tiled = false;
    e.m.packs = null;
    e.m.copy_rows = put;
    e.m.dsel = false;
    e.m.dscore = null;
    e.m.mla = null;
    e.m.tfq3 = null;
    link.pending_keep = .{ .slot = 3, .rows = 5 };
    try link.send(put, .{ .kind = @backingInt(wire.Kind.verify), .slot = 4 }, &.{21});
    Recording.receiving = true;
    Recording.fail_launch = true;
    try t.expectError(error.NotReady, wire.follow(&e, &link));

    try t.expectEqual(@as(usize, 5), kept.st.pos);
    try t.expectEqual(@as(usize, 0), kept.st.pending);
    try t.expectEqual(@as(u32, 5), kept_pos[0]);
    try t.expectEqual(@as(usize, 1), running.st.pending);
    try t.expectEqual(@as(u32, 4), e.slot);
    try t.expect(e.cur == &running);
    try t.expectEqual(@as(u32, 21), token_ids[0]);
    try t.expectEqual(@as(usize, 2), Recording.launches);
    try t.expectEqual(@as(usize, 1), Recording.sent);
    try t.expectEqual(@as(usize, 1), Recording.received);
    try t.expectEqual(@as(usize, 0), Recording.reductions);
}
