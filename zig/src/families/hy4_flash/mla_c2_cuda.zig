//! Tensor-core MLA query absorb (`W_UK_T`), value expand + sigmoid gate (`W_UV`), and `kv_b` tile repack.
const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Function = cuda.Function;

const kl = 512;
const nd = 192;
const vd = 256;
const kvb_rows = nd + vd;

const Cfg = struct { f: Function, bm: u32, bn: u32, threads: u32, smem: u32 };

pub fn cfgSmem(k: u32, mt: u32, nt: u32, warps: u32, kc: u32, stages: u32, planes: u32) u32 {
    const bn = 8 * nt;
    const cw = 16 * mt;
    const slots = @min(stages, k / kc);
    return slots * planes * bn * kc * 2 + warps * 8 * (cw + 4) * 4;
}

fn cfg(m: *cuda.Module, name: [:0]const u8, k: u32, mt: u32, nt: u32, warps: u32, kc: u32, stages: u32, planes: u32) !Cfg {
    const bn = 8 * nt;
    const cw = 16 * mt;
    const smem = cfgSmem(k, mt, nt, warps, kc, stages, planes);
    const f = try m.function(name);
    try f.allowDynamicShared(smem);
    return .{ .f = f, .bm = cw * warps, .bn = bn, .threads = 32 * warps, .smem = smem };
}

fn dependentLaunches(d: *const cuda.Driver) bool {
    var dev: cuda.abi.Device = 0;
    var major: c_int = 0;
    if (d.api.cuCtxGetDevice(&dev) != cuda.abi.success) return false;
    return d.api.cuDeviceGetAttribute(&major, .compute_capability_major, dev) == cuda.abi.success and major >= 9;
}

pub const C2 = struct {
    mod: cuda.Module,
    ab: [3]Cfg,
    ex: [3]Cfg,
    ex_d4: Cfg,
    pdl: bool = false,

    pub fn load(d: *const cuda.Driver) !C2 {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_mla_c2);
        errdefer mod.unload();
        var c: C2 = undefined;
        c.mod = mod;
        c.ab = .{
            try cfg(&c.mod, "hy4c2_absorb_d1_x", 192, 1, 1, 4, 192, 2, 1),
            try cfg(&c.mod, "hy4c2_absorb_d2_x", 192, 1, 2, 4, 192, 2, 1),
            try cfg(&c.mod, "hy4c2_absorb_p_x", 192, 2, 8, 8, 64, 3, 1),
        };
        c.ex = .{
            try cfg(&c.mod, "hy4c2_expand_d1_f32lat", 512, 1, 1, 4, 512, 2, 3),
            try cfg(&c.mod, "hy4c2_expand_d2_f32lat", 512, 1, 2, 4, 512, 2, 3),
            try cfg(&c.mod, "hy4c2_expand_p_f32lat", 512, 2, 4, 8, 64, 3, 3),
        };
        c.ex_d4 = try cfg(&c.mod, "hy4c2_expand_d4k64s2_f32lat", 512, 1, 4, 4, 64, 2, 3);
        c.pdl = dependentLaunches(d);
        return c;
    }

    pub fn unload(c: *C2) void {
        c.mod.unload();
        c.* = undefined;
    }
};

fn grid(c: Cfg, cols: usize, heads: usize, rows: usize) u32 {
    return @intCast((cols / c.bm) * heads * ((rows + c.bn - 1) / c.bn));
}

pub fn absorb(c: *const C2, s: Stream, q: u64, ldq: usize, rows: usize, heads: usize, wukt: u64, qa: u64, qlo: usize) !void {
    if (rows == 0) return;
    const k = if (rows <= 8) c.ab[0] else if (rows <= 16) c.ab[1] else c.ab[2];
    var a: cuda.Args = .{};
    a.add(q);
    a.add(@as(c_int, @intCast(ldq)));
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(heads)));
    a.add(wukt);
    a.add(qa);
    a.add(@as(i64, @intCast(qlo)));
    try cuda.launch.launch(k.f, .{ .grid = .{ .x = grid(k, kl, heads, rows) }, .block = .{ .x = k.threads }, .shared = k.smem, .pdl = c.pdl }, s, &a);
}

pub fn expand(c: *const C2, s: Stream, lat: u64, rows: usize, heads: usize, wuv: u64, gate: u64, ldg: usize, out: u64) !void {
    if (rows == 0) return;
    const k = if (rows <= 8) c.ex[0] else if (rows <= 16) c.ex[1] else if (rows <= 32) c.ex_d4 else c.ex[2];
    var a: cuda.Args = .{};
    a.add(lat);
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(heads)));
    a.add(wuv);
    a.add(gate);
    a.add(@as(c_int, @intCast(ldg)));
    a.add(out);
    try cuda.launch.launch(k.f, .{ .grid = .{ .x = grid(k, vd, heads, rows) }, .block = .{ .x = k.threads }, .shared = k.smem, .pdl = c.pdl }, s, &a);
}

fn tile(out: []u16, n: usize, k: usize, ctx: anytype, comptime at: fn (@TypeOf(ctx), usize, usize) u16) void {
    const ch = k / 32;
    for (0..n / 16) |t| for (0..ch) |b| for (0..2) |j| for (0..32) |l| {
        const u = ((t * ch + b) * 2 + j) * 32 + l;
        const r = 16 * t + l / 4 + 8 * j;
        const k0 = 32 * b + 8 * (l % 4);
        for (0..8) |i| out[u * 8 + i] = at(ctx, r, k0 + i);
    };
}

const Head = struct { kv: []const u16 };
fn ukAt(h: Head, c: usize, d: usize) u16 {
    return h.kv[d * kl + c];
}
fn uvAt(h: Head, v: usize, c: usize) u16 {
    return h.kv[(nd + v) * kl + c];
}

/// Repack `kv_b` `[heads * 448][512]` bf16 into tiled `W_UK_T` and `W_UV` on host (caller frees both).
pub fn repack(gpa: std.mem.Allocator, kvb: []const u8, heads: usize) !struct { uk: []u16, uv: []u16 } {
    if (kvb.len != heads * kvb_rows * kl * 2) return error.KvbShape;
    const src: []align(1) const u16 = std.mem.bytesAsSlice(u16, kvb);
    const all = try gpa.alloc(u16, heads * kvb_rows * kl);
    defer gpa.free(all);
    for (all, src) |*o, v| o.* = v;
    const uk = try gpa.alloc(u16, heads * kl * nd);
    errdefer gpa.free(uk);
    const uv = try gpa.alloc(u16, heads * vd * kl);
    errdefer gpa.free(uv);
    for (0..heads) |h| {
        const hd: Head = .{ .kv = all[h * kvb_rows * kl ..][0 .. kvb_rows * kl] };
        tile(uk[h * kl * nd ..][0 .. kl * nd], kl, nd, hd, ukAt);
        tile(uv[h * vd * kl ..][0 .. vd * kl], vd, kl, hd, uvAt);
    }
    return .{ .uk = uk, .uv = uv };
}

test "mla_c2_cuda: repack places every kv_b value in tiled unit order and smem matches records" {
    const gpa = std.testing.allocator;
    const kv = try gpa.alloc(u16, kvb_rows * kl);
    defer gpa.free(kv);
    for (kv, 0..) |*v, i| v.* = @truncate(i *% 2654435761 >> 7);
    const r = try repack(gpa, std.mem.sliceAsBytes(kv), 1);
    defer gpa.free(r.uk);
    defer gpa.free(r.uv);
    for (0..8) |i| try std.testing.expectEqual(kv[(8 + i) * kl + 1], r.uk[5 * 8 + i]);
    for (0..8) |i| try std.testing.expectEqual(kv[(nd + 1) * kl + 8 + i], r.uv[5 * 8 + i]);
    try std.testing.expectEqual(@as(u32, 5632), cfgSmem(192, 1, 1, 4, 192, 2, 1));
    try std.testing.expectEqual(@as(u32, 8704), cfgSmem(192, 1, 2, 4, 192, 2, 1));
    try std.testing.expectEqual(@as(u32, 33792), cfgSmem(192, 2, 8, 8, 64, 3, 1));
    try std.testing.expectEqual(@as(u32, 27136), cfgSmem(512, 1, 1, 4, 512, 2, 3));
    try std.testing.expectEqual(@as(u32, 51712), cfgSmem(512, 1, 2, 4, 512, 2, 3));
    try std.testing.expectEqual(@as(u32, 27136), cfgSmem(512, 1, 4, 4, 64, 2, 3));
    try std.testing.expectEqual(@as(u32, 46080), cfgSmem(512, 2, 4, 8, 64, 3, 3));
}
