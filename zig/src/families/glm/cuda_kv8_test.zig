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

const Rec = struct {
    var n: usize = 0;
    var arity: [128]u8 = @splat(0);
    var ids: [16]usize = undefined;
    var grid: [16][3]u32 = undefined;
    var blocks: [16]u32 = undefined;
    var shared: [16]u32 = undefined;
    var args: [16][16]u64 = undefined;
    fn run(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, sh: c_uint, _: cuda.abi.Stream, a: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const k = @intFromPtr(f);
        ids[n] = k;
        grid[n] = .{ gx, gy, gz };
        blocks[n] = bx;
        shared[n] = sh;
        for (0..arity[k]) |i| args[n][i] = std.mem.readInt(u64, @as(*const [8]u8, @ptrCast(a.?[i].?)), .little);
        n += 1;
        return 0;
    }
    fn word(launch: usize, i: usize) u32 {
        return @truncate(args[launch][i]);
    }
};
fn recording(d: *cuda.Driver, ks: anytype) void {
    d.api.cuLaunchKernel = Rec.run;
    d.api.cuModuleGetFunction = Ops.function;
    inline for (@typeInfo(@TypeOf(ks.*)).@"struct".field_names, 0..) |name, i| @field(ks, name) = .{ .d = d, .handle = @ptrFromInt(16 + i) };
    Rec.n = 0;
    Rec.arity = @splat(0);
}
fn kid(f: cuda.Function) usize {
    return @intFromPtr(f.handle);
}

test "GLM FP8 KV: on pages an FP8 cache's write takes lat_write_fp8_paged, the window's arguments then the table" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    var ks: glue.Kernels = undefined;
    recording(&d, &ks);
    Rec.arity[kid(ks.lat_write_fp8)] = 5;
    Rec.arity[kid(ks.lat_write_fp8_paged)] = 7;
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    // 11 DSA parts of 296,960 bytes a page; this cache's codes in page 0, its scales 262,144 bytes on
    const paged: latent.Table = .{ .pt = 900, .ps = 11 * 296_960, .page_rows = 11 * 296_960 / 1024, .logical = true };
    try latent.latentWriteFormat(set, stream, 100, 576, 200, 200 + 262_144, paged, 300, 7, 512);
    try latent.latentWriteFormat(set, stream, 100, 576, 200, 200 + 262_144, .{}, 300, 7, 512);
    try t.expectEqual(@as(usize, 2), Rec.n);
    try t.expectEqualSlices(usize, &.{ kid(ks.lat_write_fp8_paged), kid(ks.lat_write_fp8) }, Rec.ids[0..2]);
    try t.expectEqualSlices(u64, &.{ 100, 576, 200, 200 + 262_144, 300, 900, 11 * 296_960 }, Rec.args[0][0..7]);
    try t.expectEqualSlices(u64, &.{ 100, 576, 200, 200 + 262_144, 300 }, Rec.args[1][0..5]);
    for (0..2) |i| {
        try t.expectEqual([3]u32{ 7, 1, 1 }, Rec.grid[i]);
        try t.expectEqual(@as(u32, 512), Rec.blocks[i]);
        try t.expectEqual(@as(u32, 0), Rec.shared[i]);
    }
    try t.expectError(error.LatentShape, latent.latentWriteFormat(set, stream, 100, 576, 200, 400, paged, 300, 7, 256));
    try t.expectEqual(@as(usize, 2), Rec.n);
}

test "GLM FP8 KV: on pages an FP8 cache's dense and sparse chunks take the _paged kernels, the window's arguments then the table" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    var ks: glue.Kernels = undefined;
    recording(&d, &ks);
    Rec.arity[kid(ks.chunks_16)] = 15;
    Rec.arity[kid(ks.chunks_32)] = 15;
    Rec.arity[kid(ks.chunks_16_fp8)] = 14;
    Rec.arity[kid(ks.chunks_32_fp8)] = 14;
    Rec.arity[kid(ks.chunks_16_fp8_paged)] = 16;
    Rec.arity[kid(ks.chunks_32_fp8_paged)] = 16;
    Rec.arity[kid(ks.merge)] = 9;
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const ps: u64 = 11 * 296_960;
    const paged: latent.Table = .{ .pt = 900, .ps = ps, .page_rows = ps / 1024, .logical = true };
    const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    const scale: f32 = 0.0625;
    // dense: rows 3 at hb 16 (2 head blocks of 32 heads), rows 64 at hb 32; 5 chunks
    for ([_]usize{ 16, 32 }) |hb| {
        const rows: usize = if (hb == 16) 3 else 64;
        Rec.n = 0;
        try latent.attentionFormat(set, stream, 100, 200, 200 + 262_144, paged, 300, p, 500, 600, rows, 32, 512, 5, scale, hb);
        try t.expectEqual(@as(usize, 2), Rec.n);
        try t.expectEqual(kid(if (hb == 16) ks.chunks_16_fp8_paged else ks.chunks_32_fp8_paged), Rec.ids[0]);
        try t.expectEqualSlices(u64, &.{ 100, 200, 200 + 262_144, 300, 0, 0, 1000, 2000, 3000, rows, 32, 0 }, Rec.args[0][0..12]);
        try t.expectEqual(@as(u32, @bitCast(scale)), Rec.word(0, 12));
        try t.expectEqual(@as(u32, 0), Rec.word(0, 13));
        try t.expectEqualSlices(u64, &.{ 900, ps }, Rec.args[0][14..16]);
        try t.expectEqual([3]u32{ @intCast(rows), @intCast(32 / hb), 5 }, Rec.grid[0]);
        try t.expectEqual(@as(u32, 256), Rec.blocks[0]);
        try t.expectEqual(if (hb == 16) glue.chunksSmem(16) else glue.chunksSmem(32), Rec.shared[0]);
        try t.expectEqual(kid(ks.merge), Rec.ids[1]);
    }
    // sparse: logical lists of 2,051 tokens (5 chunks), rows 3 (hb 16) and 64 (hb 32)
    for ([_]usize{ 3, 64 }) |rows| {
        Rec.n = 0;
        try latent.sparseAttentionFormat(set, stream, 100, 200, 200 + 262_144, paged, 500, 600, 2051, p, 700, rows, 32, 512, scale);
        try t.expectEqual(@as(usize, 2), Rec.n);
        try t.expectEqual(kid(if (rows < 64) ks.chunks_16_fp8_paged else ks.chunks_32_fp8_paged), Rec.ids[0]);
        try t.expectEqualSlices(u64, &.{ 100, 200, 200 + 262_144, 0, 500, 600, 1000, 2000, 3000, rows, 32, 2051 }, Rec.args[0][0..12]);
        try t.expectEqual(@as(u32, 1), Rec.word(0, 13));
        try t.expectEqualSlices(u64, &.{ 900, ps }, Rec.args[0][14..16]);
        try t.expectEqual([3]u32{ @intCast(rows), if (rows < 64) 2 else 1, 5 }, Rec.grid[0]);
        try t.expectEqual(kid(ks.merge), Rec.ids[1]);
        try t.expectEqual(@as(u64, 600), Rec.args[1][4]);
    }
    // the window's FP8 kernels keep their 14 arguments, BF16 its table, on pages or not
    Rec.n = 0;
    try latent.attentionFormat(set, stream, 100, 200, 400, .{}, 300, p, 500, 600, 3, 32, 512, 1, scale, 16);
    try latent.sparseAttentionFormat(set, stream, 100, 200, 400, .{}, 500, 600, 2051, p, 700, 64, 32, 512, scale);
    try latent.attentionFormat(set, stream, 100, 200, 0, .{ .pt = 900, .ps = 6_684_672, .page_rows = 6_528 }, 300, p, 500, 600, 3, 32, 512, 1, scale, 32);
    try latent.sparseAttentionFormat(set, stream, 100, 200, 0, .{ .pt = 900, .ps = 6_684_672, .page_rows = 6_528 }, 500, 600, 2051, p, 700, 3, 32, 512, scale);
    try t.expectEqual(@as(usize, 8), Rec.n);
    try t.expectEqualSlices(usize, &.{ kid(ks.chunks_16_fp8), kid(ks.chunks_32_fp8), kid(ks.chunks_32), kid(ks.chunks_16) }, &.{ Rec.ids[0], Rec.ids[2], Rec.ids[4], Rec.ids[6] });
    try t.expectEqualSlices(u64, &.{ 100, 200, 400, 300 }, Rec.args[0][0..4]);
    try t.expectEqual(@as(u32, 0), Rec.word(0, 13));
    try t.expectEqual(@as(u32, 1), Rec.word(2, 13));
    try t.expectEqualSlices(u64, &.{ 900, 6_684_672 }, Rec.args[4][13..15]);
    try t.expectEqualSlices(u64, &.{ 900, 6_684_672 }, Rec.args[6][13..15]);
}

test "GLM FP8 KV: a paged sparse list in the other cache's units refuses before any launch" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    var ks: glue.Kernels = undefined;
    recording(&d, &ks);
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    // E4M3 on pages reads positions: a list of BF16 rows (logical false) would be read as positions
    try t.expectError(error.TokenUnits, latent.sparseAttentionFormat(set, stream, 100, 200, 400, .{ .pt = 900, .ps = 3_266_560, .page_rows = 3_190 }, 500, 600, 2051, p, 700, 3, 32, 512, 0.125));
    // BF16 on pages reads rows
    try t.expectError(error.TokenUnits, latent.sparseAttentionFormat(set, stream, 100, 200, 0, .{ .pt = 900, .ps = 6_684_672, .page_rows = 6_528, .logical = true }, 500, 600, 2051, p, 700, 3, 32, 512, 0.125));
    try t.expectEqual(@as(usize, 0), Rec.n);
    // dense chunks read no list, and a window takes no table: neither refuses
    try latent.attentionFormat(set, stream, 100, 200, 400, .{ .pt = 900, .ps = 3_266_560, .page_rows = 3_190 }, 300, p, 500, 600, 3, 32, 512, 1, 0.125, 16);
    try latent.sparseAttentionFormat(set, stream, 100, 200, 400, .{ .logical = true }, 500, 600, 2051, p, 700, 3, 32, 512, 0.125);
    try t.expectEqual(@as(usize, 4), Rec.n);
    try t.expectEqual(kid(ks.chunks_16_fp8_paged), Rec.ids[0]);
    try t.expectEqual(kid(ks.chunks_16_fp8), Rec.ids[2]);
}

test "GLM FP8 KV: a logical table keeps the assembled token lists positions, while the pooled keys still read the pages" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    const sparse = @import("cuda_sparse.zig");
    var d: cuda.Driver = undefined;
    var ks: glue.Kernels = undefined;
    recording(&d, &ks);
    Rec.arity[kid(ks.scores)] = 14;
    Rec.arity[15] = 7; // tf_glm_dsa_tokens (Ops)
    const set: glue.Set = .{ .glue = &ks };
    const ops: cuda.Module = .{ .d = &d, .handle = null };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const s: sparse.Scratch = .{ .scores = 5000, .ls = 5100, .li = 5200, .pools = 5300, .at = 5400 };
    for ([_]bool{ true, false }) |logical| {
        Rec.n = 0;
        const tab: latent.Table = .{ .pt = 900, .ps = 3_266_560, .page_rows = 3_190, .logical = logical };
        try sparse.selectTokens(set, ops, stream, 100, 200, 576, 1000, tab, 300, 5000, 3, 32, 128, 2048, s, 6000, 7000);
        try t.expectEqual(@as(usize, 3), Rec.n);
        try t.expectEqual(kid(ks.scores), Rec.ids[0]);
        try t.expectEqualSlices(u64, &.{ 900, 3_266_560 }, Rec.args[0][12..14]);
        try t.expectEqual(@as(usize, 15), Rec.ids[2]);
        try t.expectEqualSlices(u64, &.{ 5300, 300 }, Rec.args[2][0..2]);
        try t.expectEqual(@as(u32, sparse.topk_pools), Rec.word(2, 2));
        try t.expectEqualSlices(u64, &.{ 6000, 7000, if (logical) 0 else 900, 3_190 }, Rec.args[2][3..7]);
    }
}

test "GLM FP8 KV: the glue resolves the paged symbols and opts their chunks into the window's shared memory" {
    const glue = @import("cuda_glue.zig");
    const Res = struct {
        // room for every glue kernel and opt-in (the integrated stack: 66 kernels, 17 opt-ins)
        var names: [128][]const u8 = undefined;
        var n: usize = 0;
        var set_on: [32]usize = undefined;
        var set_bytes: [32]c_int = undefined;
        var sets: usize = 0;
        fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
            names[n] = std.mem.span(name);
            n += 1;
            out.* = @ptrFromInt(16 + n - 1);
            return 0;
        }
        fn attr(f: cuda.abi.Function, _: cuda.abi.FunctionAttribute, v: c_int) callconv(.c) c_int {
            set_on[sets] = @intFromPtr(f);
            set_bytes[sets] = v;
            sets += 1;
            return 0;
        }
        fn handleOf(name: []const u8) ?usize {
            for (names[0..n], 0..) |x, i| if (std.mem.eql(u8, x, name)) return 16 + i;
            return null;
        }
        fn bytesOn(h: usize) ?c_int {
            for (set_on[0..sets], set_bytes[0..sets]) |x, b| if (x == h) return b;
            return null;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Res.function;
    d.api.cuFuncSetAttribute = Res.attr;
    const ks = try glue.Kernels.load(.{ .d = &d, .handle = null });
    inline for (.{ .{ "tf_glm_lat_write_fp8_paged", 0 }, .{ "tf_glm_chunks_16_fp8_paged", glue.chunksSmem(16) }, .{ "tf_glm_chunks_32_fp8_paged", glue.chunksSmem(32) } }) |e| {
        const h = Res.handleOf(e[0]) orelse return error.SymbolNotResolved;
        if (e[1] == 0) try t.expect(Res.bytesOn(h) == null) else try t.expectEqual(@as(?c_int, @intCast(e[1])), Res.bytesOn(h));
    }
    try t.expectEqual(Res.handleOf("tf_glm_lat_write_fp8_paged").?, kid(ks.lat_write_fp8_paged));
    try t.expectEqual(Res.handleOf("tf_glm_chunks_16_fp8_paged").?, kid(ks.chunks_16_fp8_paged));
    try t.expectEqual(Res.handleOf("tf_glm_chunks_32_fp8_paged").?, kid(ks.chunks_32_fp8_paged));
    try t.expectEqual(Res.bytesOn(kid(ks.chunks_16_fp8)), Res.bytesOn(kid(ks.chunks_16_fp8_paged)));
    try t.expectEqual(Res.bytesOn(kid(ks.chunks_32_fp8)), Res.bytesOn(kid(ks.chunks_32_fp8_paged)));
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

test "GLM FP8 KV: paged snapshots copy codes and scales through shuffled pages and skip shared prefixes" {
    const layout = @import("cuda_snapshot_layout.zig");
    var d = Fake.driver();
    var layers: [1]weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 2);
    w.layers = &layers;
    const page = state.pageBytesFormat(&w, true);
    var st = try state.State.initPagedFormat(t.allocator, &d, &w, 1024, .{ .kv = 0x80000000, .pages = 2, .page_bytes = page }, true);
    defer st.deinit();
    st.capacity = 1024;
    st.pages = &.{ 3, 1 };
    const x = st.caches[0];
    try t.expectEqual(x.lc + 512 * 512, x.ls);
    try t.expectEqual(x.ls + 512 * 4, x.pk);
    const full = try layout.Plan.withHalf(&st, &w.config, 600, null, 0, null);
    var found = false;
    for (full.parts[0..full.count]) |part| if (part.live == x.ls) {
        found = true;
        try t.expectEqual(@as(usize, 600 * 4), part.len);
        try t.expectEqual(@as(usize, 512 * 4), part.span);
        try t.expectEqual(page, part.page);
        const second = full.piece(part, 512 * 4);
        try t.expectEqual(x.ls + page, second.live);
    };
    try t.expect(found);
    const shared = try layout.Plan.withHalf(&st, &w.config, 600, null, 1, null);
    found = false;
    for (shared.parts[0..shared.count]) |part| if (part.live == x.ls) {
        found = true;
        try t.expectEqual(@as(usize, 88 * 4), part.len);
        try t.expectEqual(@as(usize, 512 * 4), part.from);
        try t.expectEqual(x.ls + page, shared.piece(part, 0).live);
    };
    try t.expect(found);
}
