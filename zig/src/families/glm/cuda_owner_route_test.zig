//! TF_GLM_KV_SPLIT's owner passes through dsa_block on a recording driver: which launches each rank makes, and on what.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const dsa = @import("cuda_dsa_block.zig");
const t = std.testing;

/// Each launch's kernel, grid, and the arguments a test reads (a chunk call's queries, a merge's outputs).
const Rec = struct {
    var n: usize = 0;
    var ids: [64]usize = undefined;
    var gx: [64]c_uint = undefined;
    var gy: [64]c_uint = undefined;
    var a0: [64]u64 = undefined; // chunks: the queries; lat_write: the latent rows written
    var a3: [64]u64 = undefined; // merge: the outputs
    var heads: [64]i64 = undefined; // chunks: argument 9, merge: argument 6
    fn run(f: cuda.abi.Function, x: c_uint, y: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        ids[n] = id;
        gx[n] = x;
        gy[n] = y;
        const arg = struct {
            fn u(p: ?[*]?*anyopaque, i: usize) u64 {
                return @as(*const u64, @ptrCast(@alignCast(p.?[i].?))).*;
            }
        }.u;
        a0[n] = arg(args, 0);
        a3[n] = if (id == merge) arg(args, 3) else 0;
        heads[n] = if (id == chunks16 or id == chunks32) @bitCast(arg(args, 9)) else if (id == merge) @bitCast(arg(args, 6)) else 0;
        n += 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        out.* = @ptrFromInt(if (std.mem.eql(u8, std.mem.span(name), "tf_glm_dsa_tokens")) @as(usize, tokens) else 14);
        return 0;
    }
    fn count(id: usize) usize {
        var k: usize = 0;
        for (ids[0..n]) |x| k += @intFromBool(x == id);
        return k;
    }
    const lat_write = 1;
    const chunks16 = 3;
    const chunks32 = 4;
    const merge = 5;
    const pool_keys = 8;
    const scores = 9;
    const tokens = 15;
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

const Rig = struct {
    d: cuda.Driver,
    ks: glue.Kernels,
    l: dsa.Layer,

    fn init(r: *Rig, dense: bool) void {
        r.d = undefined;
        r.d.api.cuLaunchKernel = Rec.run;
        r.d.api.cuModuleGetFunction = Rec.function;
        inline for (@typeInfo(glue.Kernels).@"struct".field_names) |name| @field(r.ks, name) = .{ .d = &r.d, .handle = @ptrFromInt(10) };
        r.ks.lat_write.handle = @ptrFromInt(Rec.lat_write);
        r.ks.chunks_16.handle = @ptrFromInt(Rec.chunks16);
        r.ks.chunks_32.handle = @ptrFromInt(Rec.chunks32);
        r.ks.merge.handle = @ptrFromInt(Rec.merge);
        r.ks.pool_keys.handle = @ptrFromInt(Rec.pool_keys);
        r.ks.scores.handle = @ptrFromInt(Rec.scores);
        r.l = zeroed(dsa.Layer);
        r.l.lw = 512;
        r.l.heads = 21; // TP=3's rank 1 or 2
        r.l.qk = 256;
        r.l.index.heads = 32;
        r.l.index.d = 128;
        r.l.dense_limit = if (dense) 2048 else 1;
    }

    fn k(r: *Rig) dsa.Kernels {
        return .{ .set = .{ .glue = &r.ks }, .ops = .{ .d = &r.d, .handle = null } };
    }

    fn scratch() dsa.Scratch {
        var s = std.mem.zeroes(dsa.Scratch);
        s.qa = 0x10_0000;
        s.ol = 0x20_0000;
        s.lat = 0x30_0000;
        s.tokens = 0x40_0000;
        s.counts = 0x50_0000;
        return s;
    }
};

// as cuda_state.onPages places cache 0 on two regions: the latent table and stride, the pooled ones
const caches: dsa.Caches = .{ .lc = 0x4000_0000, .ik = 800, .ig = 900, .pk = 0x1000_0000, .pk_rows = 2048, .pt = 0x7100, .ps = 12 * 512 * 1024, .pkt = 0x7000, .pks = 12 * 128 * 256 };

test "GLM KV split owner passes: a rank without the latent rows writes them (to its trash page) and attends nothing" {
    var r: Rig = undefined;
    for ([_]bool{ true, false }) |dense| {
        r.init(dense);
        Rec.n = 0;
        try dsa.attendNone(r.k(), .{ .d = &r.d, .handle = null }, r.l, caches, Rig.scratch(), 2, 3, 300, 2);
        try t.expectEqual(@as(usize, 1), Rec.count(Rec.lat_write));
        try t.expectEqual(@as(u64, 0x30_0000 + 2 * 512 * 2), Rec.a0[0]); // its rows from r0, as attend writes them
        try t.expectEqual(@as(usize, 1), Rec.count(Rec.pool_keys));
        try t.expectEqual(@intFromBool(!dense), Rec.count(Rec.tokens)); // the token lists past the dense limit
        try t.expectEqual(@as(usize, 0), Rec.count(Rec.chunks16) + Rec.count(Rec.chunks32) + Rec.count(Rec.merge));
    }
}

test "GLM KV split owner passes: the owner's pass for a peer reads its queries and fills its outputs alone" {
    var r: Rig = undefined;
    for ([_]bool{ true, false }) |dense| {
        r.init(dense);
        Rec.n = 0;
        try dsa.attendPeer(r.k(), .{ .d = &r.d, .handle = null }, r.l, caches, Rig.scratch(), 3, 300, 2, 1, 3, 0xA000_0000, 0xB000_0000, 21);
        try t.expectEqual(@as(usize, 0), Rec.count(Rec.lat_write) + Rec.count(Rec.pool_keys) + Rec.count(Rec.tokens)); // no writes, no lists
        try t.expectEqual(@as(usize, 1), Rec.count(Rec.chunks16));
        try t.expectEqual(@as(usize, 1), Rec.count(Rec.merge));
        for (Rec.ids[0..Rec.n], 0..) |id, i| switch (id) {
            Rec.chunks16 => try t.expect(Rec.a0[i] == 0xA000_0000 and Rec.heads[i] == 21 and Rec.gx[i] == 3),
            Rec.merge => try t.expect(Rec.a3[i] == 0xB000_0000 and Rec.heads[i] == 21),
            else => {},
        };
    }
}

test "GLM KV split owner passes: a wider peer (TP=3's 22 heads on a 21-head rank) takes fewer rows a call, same blocks" {
    var r: Rig = undefined;
    for ([_]bool{ true, false }) |dense| {
        r.init(dense);
        Rec.n = 0;
        const rows = 64; // the wide head block: every call keeps it, though each takes fewer rows
        try dsa.attendPeer(r.k(), .{ .d = &r.d, .handle = null }, r.l, caches, Rig.scratch(), rows, 300, 2, 1, rows, 0xA000_0000, 0xB000_0000, 22);
        var covered: usize = 0;
        var next_q: u64 = 0xA000_0000;
        for (Rec.ids[0..Rec.n], 0..) |id, i| switch (id) {
            Rec.chunks32 => {
                try t.expect(Rec.heads[i] == 22 and Rec.gx[i] * 22 <= rows * 21);
                if (!dense) {
                    try t.expectEqual(next_q, Rec.a0[i]); // row blocks in order, each its rows' queries
                    next_q += Rec.gx[i] * 22 * 512 * 2;
                }
                covered += Rec.gx[i];
            },
            Rec.chunks16 => return error.HeadBlockChanged,
            else => {},
        };
        try t.expectEqual(@as(usize, rows), covered);
    }
}

test "GLM KV split owner passes: the last row's pass for a peer is one row at the chunk's head block" {
    var r: Rig = undefined;
    for ([_]bool{ true, false }) |dense| {
        r.init(dense);
        Rec.n = 0;
        try dsa.lastRowPeer(r.k(), .{ .d = &r.d, .handle = null }, r.l, caches, Rig.scratch(), 100, 300, 2, 1, 0xA000_0000, 0xB000_0000, 22);
        const id: usize = if (dense) Rec.chunks32 else Rec.chunks16; // dense: the chunk's 100 rows' block; sparse: one row's
        try t.expectEqual(@as(usize, 1), Rec.count(id));
        try t.expectEqual(@as(usize, 1), Rec.count(Rec.merge));
        for (Rec.ids[0..Rec.n], 0..) |x, i| if (x == id) try t.expect(Rec.a0[i] == 0xA000_0000 and Rec.gx[i] == 1 and Rec.heads[i] == 22);
    }
}
