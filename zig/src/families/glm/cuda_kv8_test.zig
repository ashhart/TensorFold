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
    var ids: [64]usize = undefined;
    var blocks: [64]u32 = undefined;
    var scales: [64]u64 = undefined;
    var strides: [64]i64 = undefined; // the page stride lat_write (1), pool_keys (8) and scores (9) were given
    fn run(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        ids[n] = id;
        blocks[n] = bx;
        scales[n] = if (id == 2) @as(*const u64, @ptrCast(@alignCast(args.?[3].?))).* else if (id == 4 or id == 6)
            @as(*const u64, @ptrCast(@alignCast(args.?[2].?))).*
        else
            0;
        const at: ?usize = switch (id) {
            1 => 6,
            8 => 7,
            9 => 13,
            else => null,
        };
        strides[n] = if (at) |i| @as(*const i64, @ptrCast(@alignCast(args.?[i].?))).* else 0;
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
    try latent.latentWriteFormat(set, stream, 100, 512, 200, 0, .{}, 300, 3, 512);
    try latent.latentWriteFormat(set, stream, 100, 512, 200, 400, .{}, 300, 3, 512);
    try t.expectEqualSlices(usize, &.{ 1, 2 }, Launch.ids[0..2]);
    try t.expectEqualSlices(u32, &.{ 128, 512 }, Launch.blocks[0..2]);
    try t.expectEqual(@as(u64, 400), Launch.scales[1]);
    const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    for ([_]usize{ 16, 32 }) |hb| for ([_]u64{ 0, 400 }) |scale| {
        Launch.n = 0;
        try latent.attentionFormat(set, stream, 100, 200, scale, .{}, 300, p, 500, 600, 3, 32, 512, 1, 0.125, hb);
        try t.expectEqual(@as(usize, if (hb == 16) 3 else 5) + @intFromBool(scale != 0), Launch.ids[0]);
        try t.expectEqual(scale, Launch.scales[0]);
        try t.expectEqual(@as(usize, 7), Launch.ids[1]);
    };
    Launch.n = 0;
    try latent.sparseAttentionFormat(set, stream, 100, 200, 400, .{}, 500, 600, 2051, p, 700, 3, 32, 512, 0.125);
    try t.expectEqual(@as(usize, 4), Launch.ids[0]);
    try t.expectEqual(@as(u64, 400), Launch.scales[0]);
}

test "GLM FP8 KV: an FP8 cache takes no page table, so its writes and reads refuse one before they launch" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Launch.run;
    var ks: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names) |name| @field(ks, name) = .{ .d = &d, .handle = @ptrFromInt(10) };
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const paged: latent.Table = .{ .pt = 900, .ps = 6_684_672, .page_rows = 6_528 };
    const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    Launch.n = 0;
    try t.expectError(error.Fp8OnPages, latent.latentWriteFormat(set, stream, 100, 512, 200, 400, paged, 300, 3, 512));
    try t.expectError(error.Fp8OnPages, latent.attentionFormat(set, stream, 100, 200, 400, paged, 300, p, 500, 600, 3, 32, 512, 1, 0.125, 16));
    try t.expectError(error.Fp8OnPages, latent.sparseAttentionFormat(set, stream, 100, 200, 400, paged, 500, 600, 2051, p, 700, 3, 32, 512, 0.125));
    try t.expectEqual(@as(usize, 0), Launch.n);
}

const Ops = struct {
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        out.* = @ptrFromInt(if (std.mem.eql(u8, std.mem.span(name), "tf_glm_dsa_tokens")) @as(usize, 15) else 14);
        return 0;
    }
};
/// std.mem.zeroes, a projection as stored (a union) taken as a zero bf16 address.
fn zeroed(comptime T: type) T {
    if (T == @import("cuda_proj.zig").Proj) return .{ .bf16 = 0 };
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            var v: T = undefined;
            inline for (info.field_names, info.field_types) |name, F| @field(v, name) = zeroed(F);
            return v;
        },
        else => return std.mem.zeroes(T),
    }
}
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
    var l = zeroed(dsa.Layer);
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

test "GLM KV host: a prompt MTP chunk's pooled keys take their own page stride, so page 9 of 145 stays in them" {
    const glue = @import("cuda_glue.zig");
    const dsa = @import("cuda_dsa_block.zig");
    var layers: [2]weights.Layer = undefined;
    layers[0].mixer = .{ .kda = undefined };
    layers[1].mixer = .{ .mla = undefined };
    var w: weights.Weights = undefined;
    w.config = .{};
    w.layers = &layers;
    const ps = state.pageBytes(&w);
    const pks = state.pkPageBytes(&w);
    try t.expectEqual(@as(u64, 17), ps / pks);
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Launch.run;
    d.api.cuModuleGetFunction = Ops.function;
    var ks: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names) |name| @field(ks, name) = .{ .d = &d, .handle = @ptrFromInt(10) };
    ks.lat_write.handle = @ptrFromInt(1);
    ks.pool_keys.handle = @ptrFromInt(8);
    ks.scores.handle = @ptrFromInt(9);
    const k: dsa.Kernels = .{ .set = .{ .glue = &ks }, .ops = .{ .d = &d, .handle = null } };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var l = zeroed(dsa.Layer);
    l.lw = 512;
    l.heads = 32;
    l.qk = 256;
    l.dv = 256;
    l.q_lora = 1536;
    l.dims = 4096;
    l.index.heads = 32;
    l.index.d = 128;
    l.dense_limit = 1;
    const pages: u64 = 145;
    for ([_]u64{ pks, 0 }) |own| { // TF_GLM_KV_HOST's MTP cache (the second), then pooled keys inside the pages
        const c: dsa.Caches = .{ .lc = 0x10000000 + ps / 2, .ik = 800, .ig = 900, .pk = 0x20000000 + pks / 2, .pk_rows = 16386, .pt = 700, .ps = ps, .pks = own };
        Launch.n = 0;
        try dsa.blockLastRow(k, stream, l, c, std.mem.zeroes(dsa.Scratch), 112, 4096, 2048, 300, 4096, 1, 600); // rows on 16 bytes, as glue.router needs
        var seen: [3]usize = @splat(0);
        for (Launch.ids[0..Launch.n], Launch.strides[0..Launch.n]) |id, stride| if (id == 1 or id == 8 or id == 9) {
            seen[if (id == 1) 0 else id - 7] += 1;
            try t.expectEqual(@as(i64, @intCast(if (id == 1 or own == 0) ps else own)), stride);
            if (id == 8 and own != 0) try t.expect(pks / 2 + 9 * @as(u64, @intCast(stride)) + pks / 2 <= pages * pks);
        };
        try t.expectEqual([3]usize{ 1, 1, 1 }, seen);
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
    try t.expectEqual(@as(usize, 11), mla);
    const contexts = [_]usize{ 8192, 65536, 199307, 245760 };
    const bytes = [_][3]usize{
        .{ 8256, 350153216, 299824640 },
        .{ 65600, 1098836480, 698938880 },
        .{ 199360, 2845207040, 1629908480 },
        .{ 245824, 3451841024, 1953297920 },
    };
    for (contexts, bytes) |context, want| {
        const capacity = @import("cuda_native.zig").capacity(context);
        var bf = try state.State.init(t.allocator, &d, &w, capacity);
        defer bf.deinit();
        var fp = try state.State.initFormat(t.allocator, &d, &w, capacity, true);
        defer fp.deinit();
        try t.expectEqual((mla + 1) * capacity * 508, bf.arena.buf.len - fp.arena.buf.len);
        try t.expectEqual(want, [3]usize{ capacity, bf.arena.buf.len, fp.arena.buf.len });
    }
}

test "GLM FP8 KV: unset, empty or whitespace-only TF_KV_PAGES is auto at every rank count, fp8 or not; other text reaches the start" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const start = @import("cuda_start.zig");
    const sys_env = @import("sys").env;
    const name = "TF_KV_PAGES";
    const before = if (std.c.getenv(name)) |v| try t.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (before) |v| {
            _ = sys_env.setenv(name, v, 1);
            t.allocator.free(v);
        } else _ = sys_env.unsetenv(name);
    }
    _ = sys_env.unsetenv(name);
    try t.expect(start.pagesText() == null);
    for ([_][:0]const u8{ "", " ", "\t", " \t\r\n" }) |v| {
        try t.expectEqual(@as(c_int, 0), sys_env.setenv(name, v, 1));
        for ([_]usize{ 1, 2, 3 }) |w| {
            for ([_]bool{ false, true }) |kv8| try t.expectEqual(try start.paging(null, w, kv8), try start.paging(start.pagesText(), w, kv8));
        }
    }
    for ([_][:0]const u8{ "auto", "on", "off", "0", "1", "all", " on" }) |v| {
        try t.expectEqual(@as(c_int, 0), sys_env.setenv(name, v, 1));
        try t.expectEqualStrings(v, start.pagesText().?);
    }
}

test "GLM shared window: its KDA projection rows have their own room, even when one stream may take every row" {
    var d = Fake.driver();
    var layers: [45]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 4);
    for (&layers, 0..) |*l, i| if (w.config.kind(@intCast(i)) == .mla) {
        l.mixer = .{ .mla = undefined };
    } else {
        l.mixer = .{ .kda = undefined };
    };
    w.layers = &layers;
    const rows = @import("cuda_engine.zig").shared_rows;
    try t.expectEqual(rows, state.window_rows);
    const row_bytes = @import("cuda_kda_block.zig").rowWidth(w.share.kda.len()) * 2;
    var shared = try state.Buffers.initShared(&d, &w, rows, 8256);
    defer shared.deinit();
    try t.expect(shared.kda.ka - shared.kproj >= rows * row_bytes);
    var window = try state.Buffers.init(&d, &w, rows, false, 8256);
    defer window.deinit();
    try t.expect(window.kda.ka - window.kproj < rows * row_bytes); // a lone window's rows stay in the state
}
