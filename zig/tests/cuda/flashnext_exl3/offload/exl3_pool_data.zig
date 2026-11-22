//! exl3-pool: the pack reader, one layer's experts and the synthetic routing the pool gates share.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const st = core.safetensors;
const exl3e = cuda.exl3_experts;
const fpool = flash.exl3_pool;
const policy = fpool.offload;
const Gpu = check.Gpu;

pub const dims = 2560;
pub const width = 640;
pub const routed_n = 512;
pub const table_n = 513;
pub const top_k = 10;
pub const slots = 11;
pub const max_rows = 16;
pub const shard_count = 7;
pub const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };

pub const Pack = struct {
    files: [shard_count]st.File = undefined,
    open_n: usize = 0,

    pub fn open(p: *Pack, gpu: Gpu, dir: []const u8) !void {
        var buf: [512]u8 = undefined;
        for (0..shard_count) |i| {
            const path = try std.fmt.bufPrint(&buf, "{s}/model-{d:0>5}-of-{d:0>5}.safetensors", .{ dir, i + 1, shard_count });
            p.files[i] = try st.File.open(gpu.gpa, gpu.io, path);
            p.open_n += 1;
        }
    }

    pub fn close(p: *Pack, io: std.Io) void {
        for (p.files[0..p.open_n]) |*f| f.close(io);
    }

    pub fn get(p: *const Pack, comptime fmt: []const u8, args: anytype) !st.Tensor {
        var buf: [200]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        for (p.files[0..p.open_n]) |*f| if (f.get(name)) |t| return t;
        std.debug.print("missing tensor {s}\n", .{name});
        return error.MissingTensor;
    }
};

// ---------------------------------------------------------------------------------------------------- one layer

/// One layer's experts: host trellis slices (mmap), the stacked scales and the fully resident reference on the device.
pub const LayerData = struct {
    layer: usize,
    routed: [routed_n][3][]const u8,
    shared: [3][]const u8,
    scale_bufs: [6]cuda.DeviceBuffer,
    res_mem: cuda.DeviceBuffer,
    res_tab: cuda.DeviceBuffer,
    res: exl3e.Layer,

    pub fn load(gpu: Gpu, pack: *const Pack, layer: usize) !*LayerData {
        const gpa = gpu.gpa;
        const ld = try gpa.create(LayerData);
        errdefer gpa.destroy(ld);
        ld.layer = layer;
        // stacked scales [E, ·]: gate suh [D] svh [I], up suh [D] svh [I], down suh [I] svh [D]
        const sizes = [6]usize{ dims, dims, width, width, width, dims }; // suh_g suh_u svh_g svh_u suh_d svh_d
        var host: [6][]u8 = undefined;
        for (&host, sizes) |*h, n| h.* = try gpa.alloc(u8, table_n * n * 2);
        defer for (host) |h| gpa.free(h);
        var total: usize = 0;
        for (0..table_n) |e| {
            for (projs, 0..) |pn, p| {
                const tr = if (e < routed_n)
                    try pack.get("model.language_model.layers.{d}.mlp.experts.{d}.{s}.trellis", .{ layer, e, pn })
                else
                    try pack.get("model.language_model.layers.{d}.mlp.shared_expert.{s}.trellis", .{ layer, pn });
                const su = if (e < routed_n)
                    try pack.get("model.language_model.layers.{d}.mlp.experts.{d}.{s}.suh", .{ layer, e, pn })
                else
                    try pack.get("model.language_model.layers.{d}.mlp.shared_expert.{s}.suh", .{ layer, pn });
                const sv = if (e < routed_n)
                    try pack.get("model.language_model.layers.{d}.mlp.experts.{d}.{s}.svh", .{ layer, e, pn })
                else
                    try pack.get("model.language_model.layers.{d}.mlp.shared_expert.{s}.svh", .{ layer, pn });
                const k: usize = if (p == 2) width else dims;
                const n: usize = if (p == 2) dims else width;
                if (tr.rank != 3 or tr.dim(0) * 16 != k or tr.dim(1) * 16 != n) return error.BadShape;
                if (su.bytes.len != k * 2 or sv.bytes.len != n * 2) return error.BadShape;
                if (e < routed_n) ld.routed[e][p] = tr.bytes else ld.shared[p] = tr.bytes;
                total += (tr.bytes.len + 255) / 256 * 256;
                const su_i: usize = switch (p) {
                    0 => 0,
                    1 => 1,
                    else => 4,
                };
                const sv_i: usize = switch (p) {
                    0 => 2,
                    1 => 3,
                    else => 5,
                };
                @memcpy(host[su_i][e * k * 2 ..][0 .. k * 2], su.bytes);
                @memcpy(host[sv_i][e * n * 2 ..][0 .. n * 2], sv.bytes);
            }
        }
        var made: usize = 0;
        errdefer for (ld.scale_bufs[0..made]) |*b| b.free();
        for (&ld.scale_bufs, host) |*b, h| {
            b.* = try cuda.DeviceBuffer.fromHost(gpu.d, h);
            made += 1;
        }
        // the fully resident reference: every expert's trellis back to back, its own tables
        ld.res_mem = try cuda.DeviceBuffer.alloc(gpu.d, total);
        errdefer ld.res_mem.free();
        var ptrs: [3][table_n]u64 = undefined;
        var k2s: [3][table_n]i32 = undefined;
        var at: usize = 0;
        const tiles = (dims / 16) * (width / 16);
        for (0..table_n) |e| for (0..3) |p| {
            const bytes = if (e < routed_n) ld.routed[e][p] else ld.shared[p];
            try ld.res_mem.upload(at, bytes);
            ptrs[p][e] = ld.res_mem.ptr + at;
            k2s[p][e] = @intCast(bytes.len / (tiles * 16));
            at += (bytes.len + 255) / 256 * 256;
        };
        const tl = cuda.exl3_pool.Tables.init(table_n);
        const tab = try gpa.alloc(u8, tl.bytes);
        defer gpa.free(tab);
        @memset(tab, 0);
        for (0..3) |p| {
            @memcpy(tab[tl.ptr[p]..][0 .. table_n * 8], std.mem.sliceAsBytes(&ptrs[p]));
            @memcpy(tab[tl.k2[p]..][0 .. table_n * 4], std.mem.sliceAsBytes(&k2s[p]));
        }
        ld.res_tab = try cuda.DeviceBuffer.fromHost(gpu.d, tab);
        ld.res = .{
            .gate = ld.res_tab.ptr + tl.ptr[0],
            .up = ld.res_tab.ptr + tl.ptr[1],
            .down = ld.res_tab.ptr + tl.ptr[2],
            .gate_k2 = ld.res_tab.ptr + tl.k2[0],
            .up_k2 = ld.res_tab.ptr + tl.k2[1],
            .down_k2 = ld.res_tab.ptr + tl.k2[2],
            .suh_g = ld.scale_bufs[0].ptr,
            .suh_u = ld.scale_bufs[1].ptr,
            .svh_g = ld.scale_bufs[2].ptr,
            .svh_u = ld.scale_bufs[3].ptr,
            .suh_d = ld.scale_bufs[4].ptr,
            .svh_d = ld.scale_bufs[5].ptr,
            .experts = table_n,
            .dims = dims,
            .width = width,
            .codebook = .mul1,
            .act_mode = 1,
        };
        std.debug.print("  layer {d}: 513 experts mapped, fully resident reference {d:.0} MiB; routed k2 gate/up/down {d}/{d}/{d}, shared {d}/{d}/{d}\n", .{
            layer, @as(f64, @floatFromInt(total)) / 1048576.0, k2s[0][0], k2s[1][0], k2s[2][0], k2s[0][512], k2s[1][512], k2s[2][512],
        });
        return ld;
    }

    pub fn free(ld: *LayerData, gpa: std.mem.Allocator) void {
        ld.res_tab.free();
        ld.res_mem.free();
        for (&ld.scale_bufs) |*b| b.free();
        gpa.destroy(ld);
    }

    pub fn scales(ld: *const LayerData) fpool.Scales {
        return .{ .suh_g = ld.scale_bufs[0].ptr, .suh_u = ld.scale_bufs[1].ptr, .svh_g = ld.scale_bufs[2].ptr, .svh_u = ld.scale_bufs[3].ptr, .suh_d = ld.scale_bufs[4].ptr, .svh_d = ld.scale_bufs[5].ptr };
    }

    pub fn source(ld: *const LayerData) fpool.Source {
        return .{ .routed = &ld.routed, .shared = ld.shared };
    }
};

// ---------------------------------------------------------------------------------------------------- inputs

/// Skewed expert choice: Zipf(s) over a fixed random permutation of the 512 ids.
pub const Zipf = struct {
    perm: [routed_n]u16,
    cdf: [routed_n]f64,

    pub fn init(seed: u64, s: f64) Zipf {
        var z: Zipf = undefined;
        var prng = std.Random.DefaultPrng.init(seed);
        for (&z.perm, 0..) |*p, i| p.* = @intCast(i);
        prng.random().shuffle(u16, &z.perm);
        var sum: f64 = 0;
        for (&z.cdf, 0..) |*c, i| {
            sum += 1.0 / std.math.pow(f64, @floatFromInt(i + 1), s);
            c.* = sum;
        }
        for (&z.cdf) |*c| c.* /= sum;
        return z;
    }

    /// One row: 10 distinct routed ids then the shared expert.
    pub fn row(z: *const Zipf, r: std.Random, out: []i32) void {
        var n: usize = 0;
        while (n < top_k) {
            const u = r.float(f64);
            var i: usize = 0;
            while (i + 1 < routed_n and z.cdf[i] < u) i += 1;
            const id: i32 = z.perm[i];
            if (std.mem.indexOfScalar(i32, out[0..n], id) != null) continue;
            out[n] = id;
            n += 1;
        }
        out[top_k] = 512;
    }
};

pub fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// The device buffers of one engine side (pool or resident): x, picks, weights, scratch, out.
pub const Io = struct {
    x: cuda.DeviceBuffer,
    picks: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    sbuf: cuda.DeviceBuffer,
    sc: exl3e.Scratch,

    pub fn init(gpu: Gpu, l: exl3e.Layer) !Io {
        var v: Io = undefined;
        v.x = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * dims * 2);
        v.picks = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * slots * 4);
        v.wts = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * slots * 4);
        v.out = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * dims * 4);
        v.sbuf = try cuda.DeviceBuffer.alloc(gpu.d, try exl3e.scratchBytes(l, max_rows, slots));
        v.sc = try exl3e.scratch(v.sbuf.ptr, l, max_rows, slots);
        return v;
    }

    pub fn free(v: *Io) void {
        v.x.free();
        v.picks.free();
        v.wts.free();
        v.out.free();
        v.sbuf.free();
    }

    pub fn set(v: *Io, x: []const u16, picks: []const i32, wts: []const f32) !void {
        try v.x.upload(0, std.mem.sliceAsBytes(x));
        try v.picks.upload(0, std.mem.sliceAsBytes(picks));
        try v.wts.upload(0, std.mem.sliceAsBytes(wts));
    }

    pub fn y(v: Io, gpa: std.mem.Allocator, rows: usize) ![]u8 {
        const out = try gpa.alloc(u8, rows * slots * dims * 4);
        try v.sbuf.d.check(v.sbuf.d.api.cuMemcpyDtoH_v2(out.ptr, v.sc.y, out.len), "cuMemcpyDtoH");
        return out;
    }
};

pub fn call(ex: *const exl3e.Experts, s: cuda.Stream, l: exl3e.Layer, v: Io, picks: u64, rows: usize, combine: bool) !void {
    if (combine) {
        try ex.run(s, l, v.x.ptr, dims, .bf16, picks, v.wts.ptr, rows, v.out.ptr, v.sc);
    } else {
        try ex.runSlots(s, l, v.x.ptr, dims, .bf16, picks, rows, v.sc);
    }
}

/// One round through the pool: plan, then each pass staged, computed and released.
pub fn poolRound(pool: *fpool.Pool, ex: *const exl3e.Experts, s: cuda.Stream, v: Io, picks: []const i32, rows: usize, combine: bool) !*const fpool.Round {
    const r = try pool.plan(picks, rows);
    const l = pool.tables();
    for (0..r.n_passes) |j| {
        try pool.stage(r, j, s);
        try call(ex, s, l, v, pool.picksFor(r, j, v.picks.ptr), rows, combine);
        try pool.release(r, j, s);
    }
    return r;
}

pub fn genRound(z: *const Zipf, rng: std.Random, rows: usize, x: []u16, picks: []i32, wts: []f32) void {
    for (x[0 .. rows * dims]) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    for (0..rows) |i| z.row(rng, picks[i * slots ..][0..slots]);
    for (wts[0 .. rows * slots]) |*w| w.* = rng.float(f32) * 0.2;
}
