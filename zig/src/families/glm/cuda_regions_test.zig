//! TF_GLM_KV_SPLIT's two regions through dsa.attend on a recording driver: which table and stride each launch takes.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const dsa = @import("cuda_dsa_block.zig");
const t = std.testing;

/// Each launch's kernel and the page table and step (bytes, or latent rows a page for the token lists) it was given.
const Rec = struct {
    var n: usize = 0;
    var ids: [64]usize = undefined;
    var pt: [64]u64 = undefined;
    var step: [64]i64 = undefined;
    fn run(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        const at: ?usize = switch (id) { // where each launch's (table, step) pair sits among its arguments
            lat_write, dsa_tokens => 5,
            pool_keys => 6,
            scores => 12,
            chunks => 13,
            else => null,
        };
        ids[n] = id;
        pt[n] = if (at) |i| @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).* else 0;
        step[n] = if (at) |i| @as(*const i64, @ptrCast(@alignCast(args.?[i + 1].?))).* else 0;
        n += 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        out.* = @ptrFromInt(if (std.mem.eql(u8, std.mem.span(name), "tf_glm_dsa_tokens")) @as(usize, dsa_tokens) else 14);
        return 0;
    }
    const lat_write = 1;
    const chunks = 3;
    const pool_keys = 8;
    const scores = 9;
    const dsa_tokens = 15;
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

    fn init(r: *Rig) void {
        r.d = undefined;
        r.d.api.cuLaunchKernel = Rec.run;
        r.d.api.cuModuleGetFunction = Rec.function;
        inline for (@typeInfo(glue.Kernels).@"struct".field_names) |name| @field(r.ks, name) = .{ .d = &r.d, .handle = @ptrFromInt(10) };
        r.ks.lat_write.handle = @ptrFromInt(Rec.lat_write);
        r.ks.chunks_16.handle = @ptrFromInt(Rec.chunks);
        r.ks.chunks_32.handle = @ptrFromInt(Rec.chunks);
        r.ks.pool_keys.handle = @ptrFromInt(Rec.pool_keys);
        r.ks.scores.handle = @ptrFromInt(Rec.scores);
        r.l = zeroed(dsa.Layer);
        r.l.lw = 512;
        r.l.heads = 32;
        r.l.qk = 256;
        r.l.index.heads = 32;
        r.l.index.d = 128;
    }

    /// One dense pass (limit past the rows) and one sparse pass (limit 1) of three rows over `c`.
    fn attend(r: *Rig, c: dsa.Caches, dense: bool) !void {
        const k: dsa.Kernels = .{ .set = .{ .glue = &r.ks }, .ops = .{ .d = &r.d, .handle = null } };
        r.l.dense_limit = if (dense) 2048 else 1;
        var s = std.mem.zeroes(dsa.Scratch);
        s.tokens = 500;
        s.counts = 600;
        Rec.n = 0;
        try dsa.attend(k, .{ .d = &r.d, .handle = null }, r.l, c, s, 0, 3, 300, 2, 1, 3);
    }
};

// GLM-5.3-Flash's page at one rank: 12 caches (11 DSA layers and the head) of 512 latent rows and 128 pooled keys
const latent_page: u64 = 12 * 512 * 1024;
const pooled_page: u64 = 12 * 128 * 256;

test "GLM KV split: latent launches take the latent table and stride, pooled-key launches the pooled table and stride" {
    var r: Rig = undefined;
    r.init();
    const table: u64 = 0x7000;
    const ltable: u64 = 0x7100;
    // as cuda_state.onPages places cache 1: pooled keys in the first region, latent rows in the second
    const split: dsa.Caches = .{ .lc = 0x4000_0000 + 512 * 1024, .ik = 800, .ig = 900, .pk = 0x1000_0000 + 128 * 256, .pk_rows = 2048, .pt = ltable, .ps = latent_page, .pkt = table, .pks = pooled_page };
    for ([_]bool{ true, false }) |dense| {
        try r.attend(split, dense);
        var seen: [5]usize = @splat(0);
        for (Rec.ids[0..Rec.n], Rec.pt[0..Rec.n], Rec.step[0..Rec.n]) |id, pt, step| switch (id) {
            Rec.lat_write, Rec.chunks => {
                seen[@intFromBool(id == Rec.chunks)] += 1;
                try t.expectEqual(ltable, pt);
                try t.expectEqual(@as(i64, @intCast(latent_page)), step);
            },
            Rec.dsa_tokens => { // the token lists name latent rows: 6,144 a page here, not the interleaved 6,528
                seen[2] += 1;
                try t.expectEqual(ltable, pt);
                try t.expectEqual(@as(i64, 6144), step);
            },
            Rec.pool_keys, Rec.scores => {
                seen[@as(usize, 3) + @intFromBool(id == Rec.scores)] += 1;
                try t.expectEqual(table, pt);
                try t.expectEqual(@as(i64, @intCast(pooled_page)), step);
            },
            else => {},
        };
        try t.expectEqual([5]usize{ 1, 1, @intFromBool(!dense), 1, @intFromBool(!dense) }, seen);
    }
}

test "GLM KV split: off, every page launch takes the one table and the interleaved stride, as before the split" {
    var r: Rig = undefined;
    r.init();
    const page = latent_page + pooled_page;
    const one: dsa.Caches = .{ .lc = 0x4000_0000, .ik = 800, .ig = 900, .pk = 0x4000_0000 + 512 * 1024, .pk_rows = 2048, .pt = 0x7000, .ps = page };
    for ([_]bool{ true, false }) |dense| {
        try r.attend(one, dense);
        for (Rec.ids[0..Rec.n], Rec.pt[0..Rec.n], Rec.step[0..Rec.n]) |id, pt, step| switch (id) {
            Rec.lat_write, Rec.chunks, Rec.pool_keys, Rec.scores, Rec.dsa_tokens => {
                try t.expectEqual(@as(u64, 0x7000), pt);
                try t.expectEqual(@as(i64, @intCast(if (id == Rec.dsa_tokens) page / 1024 else page)), step);
            },
            else => {},
        };
    }
}

test "GLM KV split: a pooled-key table without its own stride is refused before any launch" {
    var r: Rig = undefined;
    r.init();
    const bad: dsa.Caches = .{ .lc = 0x4000_0000, .ik = 800, .ig = 900, .pk = 0x1000_0000, .pk_rows = 2048, .pt = 0x7100, .ps = latent_page, .pkt = 0x7000 };
    try t.expectError(error.LatentShape, r.attend(bad, false));
    try t.expectEqual(@as(usize, 0), Rec.n);
}
