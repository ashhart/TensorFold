//! Actual rank-0 DFlash2 rounds over recording CUDA and NCCL calls, without a GPU.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const abi = cuda.abi;
const Params = ?[*]?*anyopaque;

const M = glm.dflash.BLOCK;
const CAND = 2 * M * glm.dflash.CW;
const HIDDEN = M * glm.dflash.RANK;

const Round = struct {
    syncs: usize,
    event_syncs: usize,
    copies: usize,
    copy_lengths: [2]usize,
    copied_bytes: usize,
    cand: [CAND]i32,
    gh: [HIDDEN]f32,
    out: [glm.dflash.BLOCK - 1]u32,
    probs: [glm.dflash.BLOCK - 1]f64,
};

const Recording = struct {
    var syncs: usize = 0;
    var event_syncs: usize = 0;
    var copies: usize = 0;
    var copy_lengths: [2]usize = @splat(0);
    var copied_bytes: usize = 0;

    fn reset() void {
        syncs = 0;
        event_syncs = 0;
        copies = 0;
        copy_lengths = @splat(0);
        copied_bytes = 0;
    }

    fn wait(_: abi.Stream) callconv(.c) abi.Result {
        syncs += 1;
        return abi.success;
    }

    fn eventWait(_: abi.Event) callconv(.c) abi.Result {
        event_syncs += 1;
        return abi.success;
    }

    fn get(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) abi.Result {
        if (copies == copy_lengths.len) return 1;
        const out: [*]u8 = @ptrCast(dst.?);
        const in: [*]const u8 = @ptrFromInt(src);
        @memcpy(out[0..n], in[0..n]);
        copy_lengths[copies] = n;
        copies += 1;
        copied_bytes += n;
        return abi.success;
    }

    fn put(_: u64, _: ?*const anyopaque, _: usize, _: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }

    fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: Params, _: Params) callconv(.c) abi.Result {
        return abi.success;
    }

    fn gather(_: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: abi.Stream) callconv(.c) cuda.nccl.Result {
        return 0;
    }

    fn device(out: *abi.Device) callconv(.c) abi.Result {
        out.* = 0;
        return abi.success;
    }

    fn attribute(out: *c_int, _: abi.DeviceAttribute, _: abi.Device) callconv(.c) abi.Result {
        out.* = 0;
        return abi.success;
    }
};

fn bf16(n: usize, k: usize) glm.dflash.Lin {
    return .{ .bf16 = .{ .w = 0, .n = n, .k = k } };
}

fn recordRound(one_sync: bool) !Round {
    const g = t.allocator;
    const pred = try g.alloc(u16, 19 * glm.dflash.RANK);
    defer g.free(pred);
    const succ = try g.alloc(u16, 19 * glm.dflash.RANK);
    defer g.free(succ);
    const cand_host = try g.alloc(i32, CAND);
    defer g.free(cand_host);
    const gh_host = try g.alloc(f32, HIDDEN);
    defer g.free(gh_host);
    @memset(pred, 0);
    @memset(succ, 0);
    @memset(gh_host, 0);
    for (0..CAND / glm.dflash.CW) |row| {
        const block = cand_host[row * glm.dflash.CW ..][0..glm.dflash.CW];
        @memset(block[0..glm.dflash.TOPK], 18);
        @memset(block[glm.dflash.TOPK..], 0);
    }

    var cand_source: [CAND]i32 = undefined;
    var gh_source: [HIDDEN]f32 = @splat(0);
    for (0..CAND / glm.dflash.CW) |row| {
        const block = cand_source[row * glm.dflash.CW ..][0..glm.dflash.CW];
        @memset(block[0..glm.dflash.TOPK], 18);
        @memset(block[glm.dflash.TOPK..], 0);
    }

    Recording.reset();
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemcpyDtoH_v2 = Recording.get;
    d.api.cuMemcpyHtoDAsync_v2 = Recording.put;
    d.api.cuStreamSynchronize = Recording.wait;
    d.api.cuEventSynchronize = Recording.eventWait;
    d.api.cuLaunchKernel = Recording.launch;
    d.api.cuLaunchKernelEx = glm.pdl.forwardEx(Recording.launch);
    d.api.cuCtxGetDevice = Recording.device;
    d.api.cuDeviceGetAttribute = Recording.attribute;
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclAllGather = Recording.gather;
    cuda.download_stage.forget();
    defer cuda.download_stage.forget();

    var e: glm.engine.Engine = undefined;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.c = .{};
    e.share = try glm.shard.Share.of(&e.c, 0, 2);
    const other = try glm.shard.Share.of(&e.c, 1, 2);
    var spans = [2]glm.shard.Span{ e.share.vocab, other.vocab };
    e.spans = &spans;
    e.lib = &lib;
    e.comm = null;
    e.group = null;
    e.settings = .{};
    if (comptime @hasField(glm.ranks.Settings, "dflash2_one_sync")) {
        e.settings.dflash2_one_sync = one_sync;
    }
    e.w.embed = 0;
    e.w.mtp = .{
        .enorm = 0,
        .hnorm = 0,
        .eh = 0,
        .norm = 0,
        .layer = undefined,
        .draft = .{ .q4 = .{ .w = 0, .s = 0, .b = 0, .n = 77440, .npad = 77440, .k = 4096 } },
    };
    inline for (@typeInfo(glm.glue.Kernels).@"struct".field_names, 1..) |name, i| {
        @field(e.m.glue, name) = .{ .d = &d, .handle = @ptrFromInt(i) };
    }
    e.m.mm.by_rows[0] = .{ .d = &d, .handle = @ptrFromInt(100) };
    e.m.mm.reduce = .{ .d = &d, .handle = @ptrFromInt(101) };
    e.m.mm.pdl = false;
    e.m.mm.prompt_fixed_k = false;
    e.m.q4.group = .{ .d = &d, .handle = @ptrFromInt(102) };
    e.m.q4.gb10 = false;
    e.m.tfq3 = null;

    var dr: glm.dflash.Drafter = .{
        .gpa = g,
        .d = &d,
        .module = undefined,
        .k = undefined,
        .weights = undefined,
        .split = true,
        .ff = 6144,
        .rank = 0,
        .world = 2,
    };
    defer dr.slots.deinit(g);
    inline for (@typeInfo(glm.dflash.Kernels).@"struct".field_names, 1..) |name, i| {
        @field(dr.k, name) = .{ .d = &d, .handle = @ptrFromInt(200 + i) };
    }
    for (&dr.layers) |*layer| layer.* = .{
        .in_norm = 0,
        .post_norm = 0,
        .qn = 0,
        .kn = 0,
        .a_base = 0,
        .m_base = 0,
        .qkv = bf16(6144, 4096),
        .kv = bf16(2048, 4096),
        .o = bf16(4096, 4096),
        .gu = bf16(12288, 4096),
        .down = .{ .w = 0, .n = 4096, .k = 6144 },
        .a_proj = bf16(1024, 4096),
        .m_proj = bf16(1024, 4096),
    };
    dr.hproj = .{ .w = 0, .n = 256, .k = 4096 };
    dr.norm = 0;
    dr.pred = pred;
    dr.succ = succ;
    dr.cand = cand_host;
    dr.gh = gh_host;
    dr.gtk = @intFromPtr(&cand_source);
    dr.g = @intFromPtr(&gh_source);
    const ring: cuda.DeviceBuffer = .{ .d = &d, .ptr = 4096, .len = 0, .borrowed = true };
    try dr.slots.append(g, .{ .ring = ring });

    var round: Round = undefined;
    const q: glm.dflash.Req = .{
        .slot = 0,
        .pos = 0,
        .n = 0,
        .anchor = 17,
        .out = &round.out,
        .probs = &round.probs,
    };
    try dr.draft(&e, &.{q});
    @memcpy(&round.cand, cand_host);
    @memcpy(&round.gh, gh_host);
    round.syncs = Recording.syncs;
    round.event_syncs = Recording.event_syncs;
    round.copies = Recording.copies;
    round.copy_lengths = Recording.copy_lengths;
    round.copied_bytes = Recording.copied_bytes;
    return round;
}

fn expectCompleteRound(round: *const Round) !void {
    try t.expectEqual(@as(usize, 0), round.event_syncs);
    try t.expectEqual(@as(usize, 2), round.copies);
    try t.expectEqualSlices(usize, &.{ 2048, 8192 }, &round.copy_lengths);
    try t.expectEqual(@as(usize, 10240), round.copied_bytes);
    try t.expectEqualSlices(u32, &@as([7]u32, @splat(18)), &round.out);
    const probabilities: [7]f64 = @splat(1.0 / 16.0);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&probabilities), std.mem.sliceAsBytes(&round.probs));
}

test "glm dflash round: actual TP2 draft keeps bytes and proposals with one opt-in wait" {
    const before = try recordRound(false);
    const after = try recordRound(true);
    try expectCompleteRound(&before);
    try expectCompleteRound(&after);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&before.cand), std.mem.sliceAsBytes(&after.cand));
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&before.gh), std.mem.sliceAsBytes(&after.gh));
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&before.out), std.mem.sliceAsBytes(&after.out));
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&before.probs), std.mem.sliceAsBytes(&after.probs));
    try t.expectEqual(@as(usize, 2), before.syncs);
    try t.expectEqual(@as(usize, 1), after.syncs);
}
