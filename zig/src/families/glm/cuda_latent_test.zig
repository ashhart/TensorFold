//! GLM's latent path launches at GLM-5.3-Flash's 512-wide rows and full GLM-5.3's 576 (cuda.mla's roped image).

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const dsa = @import("cuda_dsa_block.zig");
const state = @import("cuda_state.zig");
const t = std.testing;

/// One launch as the driver saw it: function, grid, block, dynamic shared bytes and the arguments' values.
const Seen = struct { id: usize, grid: [3]u32, block: u32, shared: u32, args: [16]u64 };

/// Argument sizes (8 or 4 bytes) of the launches whose values the tests read, by fake handle.
fn sizes(id: usize) []const u8 {
    return switch (id) {
        ids.chunks_16, ids.chunks_32, mla_ids.chunks => "888888888884488",
        ids.merge => "888888844",
        ids.lat_write => "8888888",
        ids.absorb_q, ids.expand_v => "888888",
        mla_ids.absorb_q_192, mla_ids.absorb_q => "88888888",
        mla_ids.rope => "888888884",
        ops_ids.dsa_tokens => "88488884",
        else => "",
    };
}

const Launch = struct {
    var n: usize = 0;
    var seen: [128]Seen = undefined;

    fn run(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        var s: Seen = .{ .id = id, .grid = .{ gx, gy, gz }, .block = bx, .shared = shared, .args = @splat(0) };
        for (sizes(id), 0..) |size, i| {
            const at = args.?[i].?;
            s.args[i] = if (size == '8') @as(*const u64, @ptrCast(@alignCast(at))).* else @as(*const u32, @ptrCast(@alignCast(at))).*;
        }
        seen[n] = s;
        n += 1;
        return 0;
    }

    /// The launches of function `id`, in order.
    fn of(id: usize, out: []Seen) []Seen {
        var k: usize = 0;
        for (seen[0..n]) |s| if (s.id == id) {
            out[k] = s;
            k += 1;
        };
        return out[0..k];
    }
};

/// Fake handles: glue's kernels 1 .. in field order, cuda.mla's from 201, glm_ops' from 301.
const ids = struct {
    const absorb_q = fieldId("absorb_q");
    const expand_v = fieldId("expand_v");
    const lat_write = fieldId("lat_write");
    const chunks_16 = fieldId("chunks_16");
    const chunks_32 = fieldId("chunks_32");
    const merge = fieldId("merge");
};
const mla_ids = struct {
    const chunks = 201;
    const absorb_q = 204;
    const absorb_q_192 = 205;
    const rope = 207;
};
const ops_ids = struct {
    const dsa_tokens = 301;
};
const mla_fields = .{ "chunks_fn", "chunks_split_fn", "merge_part_fn", "absorb_q_fn", "absorb_q_192_fn", "expand_v_fn", "rope_fn", "gate_fn" };

fn fieldId(comptime name: []const u8) usize {
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |f, i| if (comptime std.mem.eql(u8, f, name)) return i;
    unreachable;
}

const Ops = struct {
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        out.* = @ptrFromInt(if (std.mem.eql(u8, std.mem.span(name), "tf_glm_dsa_tokens")) @as(usize, ops_ids.dsa_tokens) else 302);
        return 0;
    }
};

const Fakes = struct {
    d: cuda.Driver,
    ks: glue.Kernels,
    m: cuda.mla.Mla,

    fn init(f: *Fakes) void {
        f.d = undefined;
        f.d.api.cuLaunchKernel = Launch.run;
        f.d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(Launch.run);
        f.d.api.cuModuleGetFunction = Ops.function;
        inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |name, i| @field(f.ks, name) = .{ .d = &f.d, .handle = @ptrFromInt(i) };
        f.m = undefined;
        inline for (mla_fields, 201..) |name, i| @field(f.m, name) = .{ .d = &f.d, .handle = @ptrFromInt(i) };
        Launch.n = 0;
    }

    fn stream(f: *Fakes) cuda.Stream {
        return .{ .d = &f.d, .handle = null };
    }
};

fn args16(vals: []const u64) [16]u64 {
    var a: [16]u64 = @splat(0);
    @memcpy(a[0..vals.len], vals);
    return a;
}

const p: latent.Partials = .{ .po = 0x1000, .pm = 0x2000, .pl = 0x3000 };
const flash_ps: u64 = 13056 * 512; // GLM-5.3-Flash's page: 12 caches
const full_ps: u64 = 79 * 1408 * 512 + 4096; // full GLM-5.3's: 79 caches, padded to whole 576-wide rows

test "GLM latent: GLM-5.3-Flash's absorb, chunks and merge keep their kernels, grids, smem and arguments" {
    var f: Fakes = undefined;
    f.init();
    const set: glue.Set = .{ .glue = &f.ks };
    try latent.absorbQ(set, f.stream(), 0x10, 0x20, 0x30, 100, 32, 256, 0, 512);
    try t.expectEqual(Seen{ .id = ids.absorb_q, .grid = .{ 32, 16, 2 }, .block = 128, .shared = 0, .args = args16(&.{ 0x10, 0x20, 0x30, 100, 32, 64 }) }, Launch.seen[0]);
    try t.expectError(error.LatentShape, latent.absorbQ(set, f.stream(), 0x10, 0x20, 0x30, 3, 32, 192, 0, 512));
    for ([_]usize{ 16, 32 }, [_]usize{ ids.chunks_16, ids.chunks_32 }, [_]u32{ glue.chunksSmem(16), glue.chunksSmem(32) }) |hb, id, smem| {
        Launch.n = 0;
        try latent.attentionFormat(set, f.stream(), 0x40, 0x50, 0, .{ .pt = 0x60, .ps = flash_ps }, 0x70, p, 0x80, 0x90, 3, 32, 512, 2, 0.0625, hb);
        try t.expectEqual(@as(usize, 2), Launch.n);
        const c = Launch.seen[0];
        try t.expectEqual(id, c.id);
        try t.expectEqual([3]u32{ 3, @intCast(32 / hb), 2 }, c.grid);
        try t.expectEqual(smem, c.shared);
        try t.expectEqualSlices(u64, &.{ 0x40, 0x50, 0x70, 0, 0, 0x1000, 0x2000, 0x3000, 3, 32, 0, @as(u32, @bitCast(@as(f32, 0.0625))), 0, 0x60, flash_ps }, c.args[0..15]);
        try t.expectEqual(Seen{ .id = ids.merge, .grid = .{ 3, 32, 1 }, .block = 128, .shared = 0, .args = args16(&.{ 0x1000, 0x2000, 0x3000, 0x80, 0x90, 3, 32, 2, 0 }) }, Launch.seen[1]);
    }
    Launch.n = 0;
    try latent.sparseAttention(set, f.stream(), 0x40, 0x50, .{}, 0xa0, 0xb0, 2051, p, 0x80, 64, 32, 512, 0.0625);
    try t.expectEqual(ids.chunks_32, Launch.seen[0].id); // 64 rows: 32 heads a block
    try t.expectEqual([3]u32{ 64, 1, 5 }, Launch.seen[0].grid);
    try t.expectEqual(@as(u64, 1), Launch.seen[0].args[12]);
}

test "GLM latent: full GLM-5.3's 576-wide rows take cuda.mla's absorb_q_192 and 16-head roped tiles, the same merge" {
    var f: Fakes = undefined;
    f.init();
    const set: glue.Set = .{ .glue = &f.ks, .mla = &f.m };
    try latent.absorbQ(set, f.stream(), 0x10, 0x20, 0x30, 3, 22, 192, 64, 512);
    try t.expectEqual(Seen{ .id = mla_ids.absorb_q_192, .grid = .{ 22, 16, 1 }, .block = 128, .shared = 0, .args = args16(&.{ 0x10, 0x20, 0x30, 3, 22, 3, 256, 576 }) }, Launch.seen[0]);
    try t.expectEqual(@as(usize, 1), Launch.n);
    Launch.n = 0;
    try latent.attentionFormat(set, f.stream(), 0x40, 0x50, 0, .{ .pt = 0x60, .ps = full_ps }, 0x70, p, 0x80, 0x90, 3, 22, latent.roped_row, 2, 0.0625, 16);
    try t.expectEqual(@as(usize, 2), Launch.n);
    const c = Launch.seen[0];
    try t.expectEqual(@as(usize, mla_ids.chunks), c.id);
    try t.expectEqual([3]u32{ 3, 2, 2 }, c.grid); // 22 heads: two blocks of 16, the second guarded past 22
    try t.expectEqual(@as(u32, 256), c.block);
    try t.expectEqual(cuda.mla.chunksSmem(16), c.shared);
    try t.expectEqualSlices(u64, &.{ 0x40, 0x50, 0x70, 0, 0, 0x1000, 0x2000, 0x3000, 3, 22, 0, @as(u32, @bitCast(@as(f32, 0.0625))), 0, 0x60, full_ps }, c.args[0..15]);
    try t.expectEqual(Seen{ .id = ids.merge, .grid = .{ 3, 22, 1 }, .block = 128, .shared = 0, .args = args16(&.{ 0x1000, 0x2000, 0x3000, 0x80, 0x90, 3, 22, 2, 0 }) }, Launch.seen[1]);
    Launch.n = 0;
    try latent.sparseAttention(set, f.stream(), 0x40, 0x50, .{ .pt = 0x60, .ps = full_ps, .page_rows = full_ps / 1152 }, 0xa0, 0xb0, 2048, p, 0x80, 64, 22, latent.roped_row, 0.0625);
    const s = Launch.seen[0];
    try t.expectEqual(@as(usize, mla_ids.chunks), s.id); // 64 rows still 16 heads a block
    try t.expectEqual([3]u32{ 64, 2, 4 }, s.grid);
    try t.expectEqualSlices(u64, &.{ 0, 0xa0, 0xb0, 0x1000, 0x2000, 0x3000, 64, 22, 2048, @as(u32, @bitCast(@as(f32, 0.0625))), 1, 0x60, full_ps }, s.args[2..15]);
    try t.expectEqual(@as(u64, 1), Launch.seen[1].args[8]); // the merge skips count-0 rows
    Launch.n = 0;
    try t.expectError(error.HeadBlock, latent.attentionFormat(set, f.stream(), 0x40, 0x50, 0, .{}, 0x70, p, 0x80, 0x90, 64, 22, latent.roped_row, 1, 0.0625, 32));
    try t.expectError(error.LatentShape, latent.attentionFormat(set, f.stream(), 0x40, 0x50, 0x55, .{}, 0x70, p, 0x80, 0x90, 3, 22, latent.roped_row, 1, 0.0625, 16));
    try t.expectError(error.LatentShape, latent.attentionFormat(set, f.stream(), 0x40, 0x50, 0, .{}, 0x70, p, 0x80, 0x90, 3, 22, 544, 1, 0.0625, 16));
    const flash_set: glue.Set = .{ .glue = &f.ks };
    try t.expectError(error.NoRopedKernels, latent.absorbQ(flash_set, f.stream(), 0x10, 0x20, 0x30, 3, 22, 192, 64, 512));
    try t.expectError(error.NoRopedKernels, latent.sparseAttention(flash_set, f.stream(), 0x40, 0x50, .{}, 0xa0, 0xb0, 2048, p, 0x80, 3, 22, latent.roped_row, 0.0625));
    try t.expectError(error.LatentShape, latent.absorbQ(set, f.stream(), 0x10, 0x20, 0x30, 3, 22, 256, 64, 512));
    try t.expectEqual(@as(usize, 0), Launch.n);
}

test "GLM latent: roped rows keep 16 heads a block, as 32 would take 113,408 bytes of smem, past sm_86..121's 99 KiB" {
    try t.expectEqual(@as(usize, 32), latent.headBlockFor(64, 512));
    try t.expectEqual(@as(usize, 16), latent.headBlockFor(8, 512));
    try t.expectEqual(@as(usize, 16), latent.headBlockFor(64, latent.roped_row));
    const wide32: u32 = (32 * 576 + 2 * 32 * 576 + 32 * 32) * 2 + (32 * 4 + 2 * 32) * 4; // common/mla.cuh's wide_smem<32, 64>()
    try t.expectEqual(@as(u32, 113_408), wide32);
    try t.expect(wide32 > 101_376 and cuda.mla.chunksSmem(16) <= 101_376);
}

/// A rank's DSA layer at either shape, its indexer at GLM-5.3-Flash's pools of 4 or full GLM-5.3's every key.
fn layer(full: bool) dsa.Layer {
    return .{ .proj = .{ .bf16 = 0 }, .q_norm = 0, .kv_norm = 0, .q_b = .{ .bf16 = 0 }, .wk = 0x11_0000, .wv = 0x12_0000, .o = .{ .bf16 = 0 }, .index = .{ .kw = .{ .bf16 = 0 }, .gate = 0, .ln_w = 0, .ln_b = 0, .ape = 0, .qb = .{ .bf16 = 0 }, .heads = 32, .d = 128, .geom = if (full) .{ .pool = 1, .topk = 2048 } else .{} }, .heads = if (full) 22 else 32, .qk = if (full) 192 else 256, .dv = 256, .lw = 512, .q_lora = if (full) 2048 else 1536, .dims = 6144, .eps = 1e-5, .dense_limit = 1 << 20, .rope = if (full) 64 else 0, .rope_inv = if (full) 0x13_0000 else 0 };
}

fn scratch() dsa.Scratch {
    var s = std.mem.zeroes(dsa.Scratch);
    s.dp = 0x100_0000;
    s.lat = 0x200_0000;
    s.q = 0x300_0000;
    s.qa = 0x400_0000;
    s.ol = 0x500_0000;
    s.tokens = 0x600_0000;
    s.counts = 0x610_0000;
    s.partials = .{ .po = 0x700_0000, .pm = 0x710_0000, .pl = 0x720_0000 };
    s.dummy = 0x730_0000;
    return s;
}

test "GLM DSA block: a window's rows from r0 rope full GLM-5.3's keys and queries at 576 wide, GLM-5.3-Flash's none" {
    var f: Fakes = undefined;
    const r0 = 2;
    const s = scratch();
    var got: [8]Seen = undefined;
    for ([_]bool{ false, true }) |full| {
        f.init();
        const k: dsa.Kernels = .{ .set = .{ .glue = &f.ks, .mla = if (full) &f.m else null }, .ops = .{ .d = &f.d, .handle = null } };
        const l = layer(full);
        const c: dsa.Caches = .{ .lc = 0x800_0000, .ik = 0x810_0000, .ig = 0x820_0000, .pk = 0x830_0000, .pk_rows = 4098, .pt = 0x840_0000, .ps = if (full) full_ps else flash_ps };
        try dsa.attend(k, f.stream(), l, c, s, r0, 3, 0x850_0000, 5, 1, 3);
        const row: u64 = if (full) 576 else 512;
        const heads: u64 = l.heads;
        try t.expectEqualSlices(u64, &.{ s.lat + r0 * row * 2, row, c.lc, 0x850_0000, row, c.pt, c.ps }, Launch.of(ids.lat_write, &got)[0].args[0..7]);
        const ropes = Launch.of(mla_ids.rope, &got);
        try t.expectEqual(@as(usize, if (full) 2 else 0), ropes.len);
        if (full) {
            const one: u64 = @as(u32, @bitCast(@as(f32, 1)));
            const width: u64 = 2048 + 576; // dp: [q_a | kv_a] after the norms' reads, k_pe at 2048 + 512
            try t.expectEqual([3]u32{ 3, 1, 1 }, ropes[0].grid);
            try t.expectEqualSlices(u64, &.{ s.dp + r0 * width * 2 + (2048 + 512) * 2, width, 0, s.lat + r0 * 576 * 2 + 512 * 2, 576, 0, 0x850_0000, 0x13_0000, one }, ropes[0].args[0..9]);
            try t.expectEqual([3]u32{ 3, 22, 1 }, ropes[1].grid);
            try t.expectEqual(@as(u32, 32), ropes[1].block);
            try t.expectEqualSlices(u64, &.{ s.q + r0 * 22 * 256 * 2 + 192 * 2, 22 * 256, 256, s.qa + r0 * 22 * 576 * 2 + 512 * 2, 22 * 576, 576, 0x850_0000, 0x13_0000, one }, ropes[1].args[0..9]);
            try t.expect(Launch.seen[0].id == mla_ids.rope); // the key's rope before its write
        }
        const chunk = Launch.of(if (full) mla_ids.chunks else ids.chunks_16, &got)[0];
        try t.expectEqual([3]u32{ 3, @intCast((heads + 15) / 16), 1 }, chunk.grid);
        try t.expectEqualSlices(u64, &.{ s.qa + r0 * heads * row * 2, c.lc }, chunk.args[0..2]);
        try t.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 0.0625)))), chunk.args[11]); // 1 / sqrt(256) both
        try t.expectEqual(s.ol + r0 * heads * 512 * 2, Launch.of(ids.merge, &got)[0].args[3]);
    }
    var l = layer(true);
    l.rope_inv = 0;
    f.init();
    const k: dsa.Kernels = .{ .set = .{ .glue = &f.ks, .mla = &f.m }, .ops = .{ .d = &f.d, .handle = null } };
    try t.expectError(error.NoRopeFreqs, dsa.attend(k, f.stream(), l, .{ .lc = 0, .ik = 0, .ig = 0, .pk = 0, .pk_rows = 0 }, s, 0, 3, 0x850_0000, 5, 1, 3));
}

test "GLM DSA block: token lists resolve pages to 576-wide rows on full GLM-5.3's padded page, Flash's 6,528 rows" {
    for ([_]bool{ false, true }) |full| {
        var f: Fakes = undefined;
        f.init();
        var got: [8]Seen = undefined;
        const k: dsa.Kernels = .{ .set = .{ .glue = &f.ks, .mla = if (full) &f.m else null }, .ops = .{ .d = &f.d, .handle = null } };
        var l = layer(full);
        l.dense_limit = 1; // every row sparse
        const c: dsa.Caches = .{ .lc = 0x800_0000, .ik = 0x810_0000, .ig = 0x820_0000, .pk = 0x830_0000, .pk_rows = 4098, .pt = 0x840_0000, .ps = if (full) full_ps else flash_ps };
        try dsa.attend(k, f.stream(), l, c, scratch(), 0, 3, 0x850_0000, 4096, 1, 3);
        try t.expectEqual(@as(u64, if (full) 49_440 else 6_528), Launch.of(ops_ids.dsa_tokens, &got)[0].args[6]);
        const chunk = Launch.of(if (full) mla_ids.chunks else ids.chunks_16, &got)[0];
        try t.expectEqual(@as(u32, if (full) 4 else 5), chunk.grid[2]); // 2,048 or 2,051 tokens a row
        try t.expectEqual(@as(u64, 1), chunk.args[12]);
    }
    const flash = state.Shape.of(&@import("config.zig").Config{}, 12);
    const full = state.Shape.of(&@import("config.zig").Config{ .rope = 64, .kpool = 1 }, 79);
    try t.expectEqual(flash_ps, flash.pageBytes());
    try t.expectEqual(full_ps, full.pageBytes());
}
