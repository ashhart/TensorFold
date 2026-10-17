//! ModelOpt checkpoints in their own math: FP8 x FP8 projections (ck_lane.cu), NVFP4 x NVFP4 experts (ck_experts.cu).
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const qmmf = @import("qmmf.zig");
const grouped = @import("grouped.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Plan = grouped.Plan;

/// One fp32 factor: the checkpoint's runtimes multiply the input scale by the weight's in fp32.
pub fn alpha(act: f32, scale: f32) f32 {
    return act * scale;
}

/// 1 / the input scale in fp32, as the quantizers take it.
pub fn inverse(act: f32) f32 {
    return 1.0 / act;
}

/// Bytes an NVFP4 row quantization of `rows` rows of `k` inputs takes: codes, then an e4m3 scale a 16.
pub fn rowsBytes(rows: usize, k: usize) usize {
    return rows * k / 2 + rows * k / 16;
}

/// One layer's experts in the FP4 mma's layout: words and scale tiles a direction, alpha a direction and expert.
pub const Layer = struct {
    up: u64, // [E][width/64][dims/64][8][32][2] u32
    up_s: u64, // [E][width/64][dims/64][64][4] e4m3
    down: u64, // [E][dims/64][width/64][8][32][2]
    down_s: u64,
    up_alpha: u64, // fp32 [E]: up's input scale times each expert's weight scale
    down_alpha: u64,
    up_act: f32, // up's input scale (every expert's the same)
    down_act: f32,
    experts: usize,
    width: usize,
    dims: usize,
};

pub const Ck = struct {
    lane_mod: Module,
    experts_mod: Module,
    lane: [8]Function, // tiles 16, 32, 64, 64 skipping, each plain then meeting its K slices in a cluster
    reduce: Function,
    quant8_fn: Function,
    quant4_fn: Function,
    pack4_fn: Function,
    pack4_scales_fn: Function,
    up_fn: Function,
    down_fn: Function,
    resident: [2]usize, // blocks the up and down kernels fill: per SM times SMs
    clusters: bool,

    pub fn load(d: *const Driver, capability_major: u32, sms: usize) !Ck {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var lane_mod = try Module.load(d, kernels.ck_lane);
        errdefer lane_mod.unload();
        var experts_mod = try Module.load(d, kernels.ck_experts);
        errdefer experts_mod.unload();
        var c: Ck = undefined;
        c.lane_mod = lane_mod;
        c.experts_mod = experts_mod;
        for ([_][:0]const u8{ "tf_ck8_16", "tf_ck8_16c", "tf_ck8_32", "tf_ck8_32c", "tf_ck8_64", "tf_ck8_64c", "tf_ck8_64s", "tf_ck8_64sc" }, &c.lane) |name, *f| f.* = try lane_mod.function(name);
        c.reduce = try lane_mod.function("tf_ck8_reduce");
        c.quant8_fn = try lane_mod.function("tf_ck_quant8");
        c.quant4_fn = try experts_mod.function("tf_ck_quant4");
        c.pack4_fn = try experts_mod.function("tf_ck_pack4");
        c.pack4_scales_fn = try experts_mod.function("tf_ck_pack4_scales");
        c.up_fn = try experts_mod.function("tf_ck4_up_relu2");
        c.down_fn = try experts_mod.function("tf_ck4_down_f32");
        c.resident = .{ @max(1, try c.up_fn.occupancy(128, 0)) * sms, @max(1, try c.down_fn.occupancy(128, 0)) * sms };
        c.clusters = capability_major >= 9;
        return c;
    }

    pub fn unload(c: *Ck) void {
        c.experts_mod.unload();
        c.lane_mod.unload();
    }

    fn go(f: Function, s: Stream, grid: [3]usize, block: u32, shared: u32, a: *launch_.Args) !void {
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared }, s, a);
    }

    /// bf16 rows [m, k] (`ldx` apart) to e4m3(x / act) at `out` [m, k], in the FP8 weights' fragment order.
    pub fn quant8(c: *const Ck, s: Stream, x: u64, ldx: usize, m: usize, k: usize, act: f32, out: u64) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(ldx)));
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(k)));
        a.add(inverse(act));
        a.add(out);
        try go(c.quant8_fn, s, .{ (m * (k / 16) + 127) / 128, 1, 1 }, 128, 0, &a);
    }

    /// e4m3 rows [m, k] at `x8` times the `.fp8` weight `w` under input scale `act`, bf16 at `y` [m, n].
    pub fn lane8(c: *const Ck, s: Stream, x8: u64, m: usize, w: qmmf.Weight, act: f32, y: u64, part: u64) !void {
        if (w.mode != .fp8) return error.FormatMismatch;
        const sk = qmmf.splitK(w.n, w.k);
        const cluster = sk > 1 and sk <= 8 and c.clusters;
        if (sk > 1 and !cluster and part == 0) return error.NeedsPartials;
        const bm: usize = if (m <= 16) 16 else if (m <= 32) 32 else 64;
        const ti: usize = if (bm == 16) 0 else if (bm == 32) 1 else if (m % 64 == 0) 2 else 3;
        const f = c.lane[ti * 2 + @intFromBool(cluster)];
        const stage = (bm * 64 + 4096 + 127) / 128 * 128;
        const smem: u32 = @intCast(@max(4 * stage, if (cluster) bm * 64 * 4 else 0));
        const rows_t = (m + bm - 1) / bm;
        const group = @max(1, @min(rows_t, (12 << 20) / (bm * w.k)));
        const al = alpha(act, w.scale);
        var a: launch_.Args = .{};
        a.add(x8);
        a.add(w.codes);
        a.add(al);
        a.add(y);
        a.add(part);
        for ([_]usize{ m, w.n, w.k, sk, group }) |v| a.add(@as(i32, @intCast(v)));
        try launch_.launch(f, .{
            .grid = .{ .x = @intCast(rows_t * ((w.n + 63) / 64)), .y = 1, .z = sk },
            .block = .{ .x = 128, .y = 1, .z = 1 },
            .shared = smem,
            .cluster = if (cluster) .{ .x = 1, .y = 1, .z = sk } else null,
        }, s, &a);
        if (sk > 1 and !cluster) {
            const total: usize = m * w.n;
            var r: launch_.Args = .{};
            r.add(part);
            r.add(y);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(sk)));
            r.add(al);
            try go(c.reduce, s, .{ (total + 255) / 256, 1, 1 }, 256, 0, &r);
        }
    }

    /// bf16 rows [m, k] (`ldx` apart) to NVFP4 under input scale `act`: codes [m][k/2] then scales [m][k/16] at `out`.
    pub fn quant4(c: *const Ck, s: Stream, x: u64, ldx: usize, m: usize, k: usize, act: f32, out: u64) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(ldx)));
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(k)));
        a.add(inverse(act));
        a.add(out);
        a.add(out + m * k / 2);
        try go(c.quant4_fn, s, .{ (m * (k / 16) + 127) / 128, 1, 1 }, 128, 0, &a);
    }

    /// `count` experts' NVFP4 bytes [count][n][ks/2] and scales [count][n][ks/16], inputs k0 .. k0 + k, packed.
    pub fn pack(c: *const Ck, s: Stream, codes: u64, scales: u64, count: usize, n: usize, ks: usize, k0: usize, k: usize, words: u64, tiles: u64) !void {
        if (n % 64 != 0 or k % 64 != 0 or k0 % 64 != 0 or k0 + k > ks) return error.UnexpectedTensor;
        var a: launch_.Args = .{};
        a.add(codes);
        for ([_]usize{ n, ks, k0, k }) |v| a.add(@as(i32, @intCast(v)));
        a.add(words);
        try go(c.pack4_fn, s, .{ n / 64, k / 64, count }, 256, 0, &a);
        const total: usize = count * n * (k / 16);
        var b: launch_.Args = .{};
        b.add(scales);
        for ([_]usize{ n, ks, k0, k }) |v| b.add(@as(i32, @intCast(v)));
        b.add(tiles);
        b.add(@as(i64, @intCast(total)));
        try go(c.pack4_scales_fn, s, .{ (total + 255) / 256, 1, 1 }, 256, 0, &b);
    }

    /// Bytes a layer's words and scale tiles take a direction: n outputs from k inputs for `count` experts.
    pub fn packedBytes(count: usize, n: usize, k: usize) [2]usize {
        return .{ count * n * k / 2, count * n * (k / 16) };
    }

    /// relu^2 up over a plan: NVFP4 rows (quant4's, at `rows`, `k` wide; row = pair / slots) to bf16 [pairs][width].
    pub fn up(c: *const Ck, s: Stream, l: Layer, rows: u64, nrows: usize, slots: usize, p: Plan, out: u64, items: usize) !void {
        return c.run(s, c.up_fn, c.resident[0], rows, nrows, slots, l.up, l.up_s, l.up_alpha, l.dims, l.width, p, out, items);
    }

    /// down over a plan: NVFP4 pair rows (quant4's, at `rows`) to fp32 [pairs][dims].
    pub fn down(c: *const Ck, s: Stream, l: Layer, rows: u64, nrows: usize, p: Plan, out: u64, items: usize) !void {
        return c.run(s, c.down_fn, c.resident[1], rows, nrows, 0, l.down, l.down_s, l.down_alpha, l.width, l.dims, p, out, items);
    }

    fn run(c: *const Ck, s: Stream, f: Function, resident: usize, rows: u64, nrows: usize, slots: usize, w: u64, ws: u64, al: u64, k: usize, n: usize, p: Plan, out: u64, items: usize) !void {
        _ = c;
        const nb = n / 64;
        const grid = @min((items * nb + 3) / 4, resident);
        if (grid < 1) return;
        var a: launch_.Args = .{};
        a.add(rows);
        a.add(rows + nrows * k / 2);
        a.add(@as(i32, @intCast(slots)));
        a.add(w);
        a.add(ws);
        a.add(al);
        a.add(@as(i32, @intCast(k)));
        a.add(@as(i32, @intCast(n)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        try go(f, s, .{ grid, 1, 1 }, 128, 0, &a);
    }
};

test "the factors and the row bytes" {
    try std.testing.expectEqual(@as(f32, 0.5) * @as(f32, 0.25), alpha(0.5, 0.25));
    try std.testing.expectEqual(@as(usize, 4 * 512 + 4 * 64), rowsBytes(4, 1024));
    try std.testing.expectEqual([2]usize{ 2 * 64 * 1024 / 2, 2 * 64 * 64 }, Ck.packedBytes(2, 64, 1024));
}
