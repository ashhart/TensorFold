//! The real rank gather and its startup owners, driven only by recording CUDA/NCCL functions.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;

const Record = struct {
    const Broadcast = struct { send: u64, recv: u64, count: usize, ty: cuda.nccl.DataType, root: c_int };
    var sent: [64]Broadcast = undefined;
    var posts: usize = 0;
    var copies: [3]cuda.abi.Memcpy2D = undefined;
    var copied: usize = 0;
    var starts: usize = 0;
    var ends: usize = 0;
    var sizes: [8]usize = undefined;
    var allocated: usize = 0;
    var freed: usize = 0;
    var value: f64 = 0;
    var peer_refused = false;
    var pinned: [4096]u8 align(16) = undefined;

    fn broadcast(send: u64, recv: u64, count: usize, ty: cuda.nccl.DataType, root: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        if (posts == sent.len) return 1;
        sent[posts] = .{ .send = send, .recv = recv, .count = count, .ty = ty, .root = root };
        posts += 1;
        return 0;
    }
    fn start() callconv(.c) c_int {
        starts += 1;
        return 0;
    }
    fn end() callconv(.c) c_int {
        ends += 1;
        return 0;
    }
    fn copy2d(p: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (copied == copies.len) return 1;
        copies[copied] = p.*;
        copied += 1;
        return 0;
    }
    fn alloc(out: *u64, bytes: usize) callconv(.c) cuda.abi.Result {
        if (allocated == sizes.len) return 2;
        sizes[allocated] = bytes;
        allocated += 1;
        out.* = allocated * 0x10000000;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        freed += 1;
        return 0;
    }
    fn upload(_: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (n != @sizeOf(f64)) return 1;
        @memcpy(std.mem.asBytes(&value), @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn reduce(_: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: cuda.nccl.RedOp, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        value = @max(value, @as(f64, @floatFromInt(@intFromBool(peer_refused))));
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn download(dst: ?*anyopaque, _: u64, n: usize) callconv(.c) cuda.abi.Result {
        if (n != @sizeOf(f64)) return 1;
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], std.mem.asBytes(&value));
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        if (n != pinned.len) return 2;
        out.* = &pinned;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn errorText(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
        out.* = "recording driver refused the fixture call";
        return 0;
    }
    fn eventCreate(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn eventDone(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    // main's downloads ask whether the GPU is integrated first (cuda/download_stage.zig): this fake is a discrete one
    fn contextDevice(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuCtxGetDevice = contextDevice;
        d.api.cuDeviceGetAttribute = attribute;
        d.api.cuMemcpy2DAsync_v2 = copy2d;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuMemcpyDtoH_v2 = download;
        d.api.cuStreamSynchronize = sync;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventDone;
        d.api.cuEventDestroy_v2 = eventDone;
        d.api.cuGetErrorName = errorText;
        d.api.cuGetErrorString = errorText;
        return d;
    }
    fn library() cuda.nccl.Library {
        var lib: cuda.nccl.Library = undefined;
        lib.api.ncclBroadcast = broadcast;
        lib.api.ncclGroupStart = start;
        lib.api.ncclGroupEnd = end;
        lib.api.ncclAllReduce = reduce;
        return lib;
    }
};

test "GLM logit gather: one broadcast per rank at every row count, TP2 and uneven TP3" {
    const c: glm.config.Config = .{};
    var d = Record.driver();
    var lib = Record.library();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var sampler: glm.sampler.Sampler = undefined;
    sampler.d = &d;
    sampler.vocab = c.vocab;
    for ([_]usize{ 2, 3 }) |world| for (0..world) |mine| {
        var spans: [3]glm.shard.Span = undefined;
        for (spans[0..world], 0..) |*span, rank| span.* = (try glm.shard.Share.of(&c, rank, world)).vocab;
        for ([_]usize{ 1, 8, 127, 128, 129, 2048 }) |rows| {
            Record.posts = 0;
            Record.copied = 0;
            Record.starts = 0;
            Record.ends = 0;
            const stage = 0x10000000;
            const full = 0x40000000;
            const logits = 0x80000000;
            try sampler.gather(stream, .{ .nccl = .{ .lib = &lib, .comm = null } }, spans[0..world], &.{.{ .rank = mine, .logits = logits }}, rows, stage, full);
            try t.expectEqual(world, Record.posts);
            try t.expectEqual(world, Record.copied);
            try t.expectEqual(@as(usize, 1), Record.starts);
            try t.expectEqual(@as(usize, 1), Record.ends);
            for (spans[0..world], 0..) |span, rank| {
                const call = Record.sent[rank];
                const at = stage + rows * span.lo * 2;
                try t.expectEqual(at, call.recv);
                try t.expectEqual(if (rank == mine) logits else at, call.send);
                try t.expectEqual(rows * span.len(), call.count);
                try t.expectEqual(cuda.nccl.DataType.bf16, call.ty);
                try t.expectEqual(@as(c_int, @intCast(rank)), call.root);
                const copy = Record.copies[rank];
                try t.expectEqual(at, copy.src);
                try t.expectEqual(full + span.lo * 2, copy.dst);
                try t.expectEqual(span.len() * 2, copy.width);
                try t.expectEqual(span.len() * 2, copy.src_pitch);
                try t.expectEqual(@as(usize, c.vocab) * 2, copy.dst_pitch);
                try t.expectEqual(rows, copy.height);
            }
        }
    };
}

test "GLM logit gather: startup byte counts cover both stages before admission, with coordinated fallback" {
    const vocab: usize = 154880;
    // one bf16 logit row a window row: the window grew from 8 to 32 rows on main, so the count follows window_rows
    try t.expectEqual(vocab * 2 * glm.forward.window_rows, glm.sampler.stagingBytes(vocab, glm.forward.window_rows));
    try t.expectEqual(@as(usize, 39_649_280), glm.sampler.stagingBytes(vocab, glm.prompt_logprobs.batch_rows));
    var d = Record.driver();
    var lib = Record.library();
    for ([_]usize{ 2, 3 }) |world| for (0..world) |rank| {
        for ([_]bool{ false, true }) |peer_refused| {
            Record.allocated = 0;
            Record.freed = 0;
            Record.peer_refused = peer_refused;
            const before = cuda.usage(false).device;
            {
                var stage = try cuda.DeviceBuffer.alloc(&d, glm.sampler.stagingBytes(vocab, glm.forward.window_rows));
                defer stage.free();
                var e: glm.engine.Engine = undefined;
                e.d = &d;
                e.c = .{};
                e.share = try glm.shard.Share.of(&e.c, rank, world);
                e.prompt_tried = false;
                e.prompt_bufs = null;
                e.shared_buf = .{ .d = &d, .ptr = 0x9000, .len = 8 };
                e.stream = .{ .d = &d, .handle = null };
                e.lib = &lib;
                e.comm = null;
                e.profiling_prompt = null;
                defer if (e.prompt_bufs) |*b| b.free();
                try glm.prompt_logprobs.prepare(&e);
                try t.expect(e.prompt_tried);
                try t.expectEqual(@as(usize, 4), Record.allocated);
                const prompt_stage = Record.sizes[3];
                try t.expectEqual(@as(usize, 39_649_280), prompt_stage);
                const held = if (peer_refused) stage.len else stage.len + Record.sizes[1] + Record.sizes[2] + prompt_stage;
                try t.expectEqual(before + held, cuda.usage(false).device);
                try t.expectEqual(!peer_refused, e.prompt_bufs != null);
                try glm.prompt_logprobs.prepare(&e);
                try t.expectEqual(@as(usize, 4), Record.allocated);
            }
            try t.expectEqual(Record.allocated, Record.freed);
            try t.expectEqual(before, cuda.usage(false).device);
        }
    };
}
