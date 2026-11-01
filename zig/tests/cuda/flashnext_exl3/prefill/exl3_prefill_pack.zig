//! exl3-prefill fixtures: the checkpoint pack, a timer, one layer's experts on the device and random picks.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("../../check.zig");
const st = core.safetensors;
const exl3e = cuda.exl3_experts;
const Gpu = check.Gpu;

const dims = 2560;
const width = 640;
const routed_n = 512;
const table_n = 513;
const top_k = 10;
const slots = 11;
const shard_count = 7;
const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };

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

pub fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub fn bfValue(bits: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)));
}

pub const Timer = struct {
    a: cuda.Event,
    b: cuda.Event,

    pub fn init(d: *const cuda.Driver) !Timer {
        return .{ .a = try cuda.Event.init(d, true), .b = try cuda.Event.init(d, true) };
    }
    pub fn deinit(t: *Timer) void {
        t.a.deinit();
        t.b.deinit();
    }
    /// Best of 5 batches of `reps` calls, ms a call (after 2 warm-up calls).
    pub fn best(t: *Timer, s: cuda.Stream, reps: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
        for (0..2) |_| try f(ctx);
        try s.synchronize();
        var b: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try t.a.record(s);
            for (0..reps) |_| try f(ctx);
            try t.b.record(s);
            try t.b.synchronize();
            b = @min(b, @as(f64, try t.a.elapsedMs(t.b)) / @as(f64, @floatFromInt(reps)));
        }
        return b;
    }
};

/// One layer's 513 experts: the mmapped trellis, the stacked scales and every expert resident on the device.
pub const LayerData = struct {
    layer: usize,
    routed: [routed_n][3][]const u8,
    shared: [3][]const u8,
    scale_bufs: [6]cuda.DeviceBuffer,
    res_mem: cuda.DeviceBuffer,
    res_tab: cuda.DeviceBuffer,
    res: exl3e.Layer,
    line_bytes: usize,

    pub fn load(gpu: Gpu, pack: *const Pack, layer: usize) !*LayerData {
        const gpa = gpu.gpa;
        const ld = try gpa.create(LayerData);
        errdefer gpa.destroy(ld);
        ld.layer = layer;
        const sizes = [6]usize{ dims, dims, width, width, width, dims }; // suh_g suh_u svh_g svh_u suh_d svh_d
        var host: [6][]u8 = undefined;
        for (&host, sizes) |*h, n| h.* = try gpa.alloc(u8, table_n * n * 2);
        defer for (host) |h| gpa.free(h);
        var total: usize = 0;
        ld.line_bytes = 0;
        for (0..table_n) |e| {
            var line: usize = 0;
            for (projs, 0..) |pn, p| {
                const base = "model.language_model.layers.{d}.mlp.";
                const tr = if (e < routed_n) try pack.get(base ++ "experts.{d}.{s}.trellis", .{ layer, e, pn }) else try pack.get(base ++ "shared_expert.{s}.trellis", .{ layer, pn });
                const su = if (e < routed_n) try pack.get(base ++ "experts.{d}.{s}.suh", .{ layer, e, pn }) else try pack.get(base ++ "shared_expert.{s}.suh", .{ layer, pn });
                const sv = if (e < routed_n) try pack.get(base ++ "experts.{d}.{s}.svh", .{ layer, e, pn }) else try pack.get(base ++ "shared_expert.{s}.svh", .{ layer, pn });
                const k: usize = if (p == 2) width else dims;
                const n: usize = if (p == 2) dims else width;
                if (tr.rank != 3 or tr.dim(0) * 16 != k or tr.dim(1) * 16 != n) return error.BadShape;
                if (e < routed_n) ld.routed[e][p] = tr.bytes else ld.shared[p] = tr.bytes;
                total += (tr.bytes.len + 255) / 256 * 256;
                line += (tr.bytes.len + 255) / 256 * 256;
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
            if (e < routed_n) ld.line_bytes = @max(ld.line_bytes, line);
        }
        var made: usize = 0;
        errdefer for (ld.scale_bufs[0..made]) |*b| b.free();
        for (&ld.scale_bufs, host) |*b, h| {
            b.* = try cuda.DeviceBuffer.fromHost(gpu.d, h);
            made += 1;
        }
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
        std.debug.print("  layer {d}: 513 experts resident ({d:.0} MiB), routed k2 gate/up/down {d}/{d}/{d}, a routed line {d:.3} MiB\n", .{ layer, @as(f64, @floatFromInt(total)) / 1048576.0, k2s[0][0], k2s[1][0], k2s[2][0], @as(f64, @floatFromInt(ld.line_bytes)) / 1048576.0 });
        return ld;
    }

    pub fn free(ld: *LayerData, gpa: std.mem.Allocator) void {
        ld.res_tab.free();
        ld.res_mem.free();
        for (&ld.scale_bufs) |*b| b.free();
        gpa.destroy(ld);
    }
};

/// Picks for `rows` rows: ten distinct routed ids then the shared expert;
pub fn genPicks(rng: std.Random, s: f64, rows: usize, out: []i32) void {
    var perm: [routed_n]u16 = undefined;
    for (&perm, 0..) |*p, i| p.* = @intCast(i);
    var prng = std.Random.DefaultPrng.init(0x9e3779b9);
    prng.random().shuffle(u16, &perm);
    var cdf: [routed_n]f64 = undefined;
    var sum: f64 = 0;
    for (&cdf, 0..) |*c, i| {
        sum += if (s == 0) 1.0 else 1.0 / std.math.pow(f64, @floatFromInt(i + 1), s);
        c.* = sum;
    }
    for (&cdf) |*c| c.* /= sum;
    for (0..rows) |r| {
        const row = out[r * slots ..][0..slots];
        var n: usize = 0;
        while (n < top_k) {
            const u = rng.float(f64);
            var i: usize = 0;
            while (i + 1 < routed_n and cdf[i] < u) i += 1;
            const id: i32 = perm[i];
            if (std.mem.indexOfScalar(i32, row[0..n], id) != null) continue;
            row[n] = id;
            n += 1;
        }
        row[top_k] = 512;
    }
}
