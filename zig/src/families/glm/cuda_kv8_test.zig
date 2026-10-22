//! Real state constructors and CUDA launch descriptors use virtual addresses, with no model or GPU execution.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const state = @import("cuda_state.zig");
const t = std.testing;
const Fake = struct {
    var next: u64 = 0x100000000;
    var live: usize = 0;
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = fill;
        return d;
    }
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = next;
        next += std.mem.alignForward(u64, n, 256);
        live += n;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize) callconv(.c) c_int {
        return 0;
    }
};

test "GLM FP8 KV: requested latent caches use codes and scales and save their actual arena bytes" {
    var d = Fake.driver();
    var layers: [2]weights.Layer = undefined;
    layers[0].mixer = .{ .kda = undefined };
    layers[1].mixer = .{ .mla = undefined };
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 2);
    w.layers = &layers;
    const capacity: usize = 8192;
    var bf = try state.State.init(t.allocator, &d, &w, capacity);
    defer bf.deinit();
    var fp = try state.State.initFormat(t.allocator, &d, &w, capacity, true);
    defer fp.deinit();
    try t.expectEqual(@as(u64, 0), bf.caches[0].ls);
    try t.expectEqual(fp.caches[0].lc + capacity * 512, fp.caches[0].ls);
    try t.expectEqual(fp.mtp_caches.lc + capacity * 512, fp.mtp_caches.ls);
    try t.expectEqual(2 * capacity * 508, bf.arena.buf.len - fp.arena.buf.len);
}

test "GLM FP8 KV: the mode is explicit on native options and all ranks agree before allocating weights" {
    const native = @import("cuda_native.zig");
    const ranks = @import("cuda_ranks.zig");
    const options: native.Options = .{ .context = 8192, .drafts = true };
    try t.expect(!options.kv8);
    const h: [32]u8 = @splat(0);
    var all = [_]ranks.Fingerprint{
        ranks.Fingerprint.of(2, 0, .{}, 8192, true, h, h, h),
        ranks.Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    };
    try t.expect(ranks.mismatch(&all) == null);
    all[1] = ranks.Fingerprint.of(2, 1, .{ .kv8 = true }, 8192, true, h, h, h);
    const different = ranks.mismatch(&all) orelse return error.ExpectedPrecisionMismatch;
    try t.expectEqual(ranks.Fingerprint.Field.kv8, different.field);
    all[0] = ranks.Fingerprint.of(2, 0, .{ .kv8 = true }, 8192, true, h, h, h);
    try t.expect(ranks.mismatch(&all) == null);
}

test "GLM FP8 KV: byte counts reject overflow and unsupported latent widths" {
    const latent = @import("cuda_latent.zig");
    try t.expectEqual(@as(usize, 516), try latent.cacheBytes(1, 512, true));
    try t.expectEqual(@as(usize, 1024), try latent.cacheBytes(1, 512, false));
    try t.expectError(error.LatentShape, latent.cacheBytes(1, 256, true));
    try t.expectError(error.Overflow, latent.cacheBytes(std.math.maxInt(usize), 512, true));
}

const Launch = struct {
    var n: usize = 0;
    var ids: [16]usize = undefined;
    var blocks: [16]u32 = undefined;
    var scales: [16]u64 = undefined;
    fn run(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        ids[n] = id;
        blocks[n] = bx;
        scales[n] = if (id == 2) @as(*const u64, @ptrCast(@alignCast(args.?[3].?))).* else if (id == 4 or id == 6)
            @as(*const u64, @ptrCast(@alignCast(args.?[2].?))).*
        else
            0;
        n += 1;
        return 0;
    }
};

test "GLM FP8 KV: real launchers preserve BF16 ABI and pass FP8 scale pointers for writes dense and sparse reads" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Launch.run;
    var ks: glue.Kernels = undefined;
    ks.lat_write = .{ .d = &d, .handle = @ptrFromInt(1) };
    ks.lat_write_fp8 = .{ .d = &d, .handle = @ptrFromInt(2) };
    ks.chunks_16 = .{ .d = &d, .handle = @ptrFromInt(3) };
    ks.chunks_16_fp8 = .{ .d = &d, .handle = @ptrFromInt(4) };
    ks.chunks_32 = .{ .d = &d, .handle = @ptrFromInt(5) };
    ks.chunks_32_fp8 = .{ .d = &d, .handle = @ptrFromInt(6) };
    ks.merge = .{ .d = &d, .handle = @ptrFromInt(7) };
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Launch.n = 0;
    try latent.latentWriteFormat(set, stream, 100, 512, 200, 0, 300, 3, 512);
    try latent.latentWriteFormat(set, stream, 100, 512, 200, 400, 300, 3, 512);
    try t.expectEqualSlices(usize, &.{ 1, 2 }, Launch.ids[0..2]);
    try t.expectEqualSlices(u32, &.{ 128, 512 }, Launch.blocks[0..2]);
    try t.expectEqual(@as(u64, 400), Launch.scales[1]);
    const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    for ([_]usize{ 16, 32 }) |hb| for ([_]u64{ 0, 400 }) |scale| {
        Launch.n = 0;
        try latent.attentionFormat(set, stream, 100, 200, scale, 300, p, 500, 600, 3, 32, 512, 1, 0.125, hb);
        try t.expectEqual(@as(usize, if (hb == 16) 3 else 5) + @intFromBool(scale != 0), Launch.ids[0]);
        try t.expectEqual(scale, Launch.scales[0]);
        try t.expectEqual(@as(usize, 7), Launch.ids[1]);
    };
    Launch.n = 0;
    try latent.sparseAttentionFormat(set, stream, 100, 200, 400, 500, 600, 2051, p, 700, 3, 32, 512, 0.125);
    try t.expectEqual(@as(usize, 4), Launch.ids[0]);
    try t.expectEqual(@as(u64, 400), Launch.scales[0]);
}

const Ops = struct {
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        out.* = @ptrFromInt(if (std.mem.eql(u8, std.mem.span(name), "tf_glm_dsa_tokens")) @as(usize, 15) else 14);
        return 0;
    }
};
test "GLM FP8 KV: actual DSA dense and sparse passes retain the sequence-owned scale pointer" {
    const glue = @import("cuda_glue.zig");
    const dsa = @import("cuda_dsa_block.zig");
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Launch.run;
    d.api.cuModuleGetFunction = Ops.function;
    var ks: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names) |name| @field(ks, name) = .{ .d = &d, .handle = @ptrFromInt(10) };
    ks.lat_write_fp8.handle = @ptrFromInt(2);
    ks.chunks_16_fp8.handle = @ptrFromInt(4);
    ks.merge.handle = @ptrFromInt(7);
    const k: dsa.Kernels = .{ .set = .{ .glue = &ks }, .ops = .{ .d = &d, .handle = null } };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var l = std.mem.zeroes(dsa.Layer);
    l.lw = 512;
    l.heads = 32;
    l.qk = 256;
    l.index.heads = 32;
    l.index.d = 128;
    var scratch = std.mem.zeroes(dsa.Scratch);
    scratch.tokens = 500;
    scratch.counts = 600;
    const cache: dsa.Caches = .{ .lc = 200, .ls = 400, .ik = 800, .ig = 900, .pk = 1000, .pk_rows = 2048 };
    for ([_]usize{ 2048, 1 }) |limit| {
        l.dense_limit = limit;
        Launch.n = 0;
        try dsa.attend(k, stream, l, cache, scratch, 0, 3, 300, 2, 1, 3);
        try t.expectEqual(@as(usize, 2), Launch.ids[0]);
        var reads: usize = 0;
        for (Launch.ids[0..Launch.n], Launch.scales[0..Launch.n]) |id, scale| if (id == 4) {
            reads += 1;
            try t.expectEqual(@as(u64, 400), scale);
        };
        try t.expectEqual(@as(usize, 1), reads);
    }
}

fn allocateState(gpa: std.mem.Allocator) !void {
    var d = Fake.driver();
    var layers: [2]weights.Layer = undefined;
    layers[0].mixer = .{ .kda = undefined };
    layers[1].mixer = .{ .mla = undefined };
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 2);
    w.layers = &layers;
    const before = cuda.memory.usage(false).device;
    defer std.debug.assert(before == cuda.memory.usage(false).device);
    var st = try state.State.initFormat(gpa, &d, &w, 8192, true);
    defer st.deinit();
}
test "GLM FP8 KV: every state-constructor allocation failure unwinds host and device ownership" {
    try std.testing.checkAllAllocationFailures(t.allocator, allocateState, .{});
}

test "GLM FP8 KV: canonical per-rank state bytes at configured long windows are metadata rather than a fit claim" {
    var d = Fake.driver();
    var layers: [45]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 2);
    var mla: usize = 0;
    for (&layers, 0..) |*l, i| if (w.config.kind(@intCast(i)) == .mla) {
        l.mixer = .{ .mla = undefined };
        mla += 1;
    } else {
        l.mixer = .{ .kda = undefined };
    };
    w.layers = &layers;
    for ([_]usize{ 8192, 65536, 199307, 245760 }) |context| {
        const capacity = @import("cuda_native.zig").capacity(context);
        var bf = try state.State.init(t.allocator, &d, &w, capacity);
        defer bf.deinit();
        var fp = try state.State.initFormat(t.allocator, &d, &w, capacity, true);
        defer fp.deinit();
        try t.expectEqual((mla + 1) * capacity * 508, bf.arena.buf.len - fp.arena.buf.len);
        std.debug.print("GLM_KV_BYTES context={d} capacity={d} main_layers={d} bf16={d} fp8={d}\n", .{ context, capacity, mla, bf.arena.buf.len, fp.arena.buf.len });
    }
}
