//! Shared vision and prefix-decoder operators through the HIP contract: independent FP64 formulas and buffers.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const Rig = @import("contract_rig.zig").Rig;
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

pub const Which = enum { linear, norm, gelu, add, scale, heads, four, embed };
fn depth(s: c.Shape) usize {
    return 4 * s.k + 257;
}
fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn spec(w: Which) c.Spec {
    var s: c.Spec = .{ .name = @tagName(w), .accum = .none, .depth = noSums, .out = .bf16, .served = &.{.{ .rows = 17, .n = 1, .k = 1024 }}, .tiles = .{ .{ .size = 256 }, .{}, .{ .size = 256 } } };
    s.name = switch (w) {
        .linear => "hip tf_vl_linear BF16 biased FP32 dense",
        .norm => "hip tf_vl_layernorm BF16 biased two-pass norm",
        .gelu => "hip tf_vl_pointwise exact GELU",
        .add => "hip tf_vl_pointwise residual",
        .scale => "hip tf_vl_pointwise scale",
        .heads => "hip tf_vl_layout heads to rows",
        .four => "hip tf_vl_layout horizontal four",
        .embed => "hip tf_vl_embed scaled BF16 table",
    };
    switch (w) {
        .linear => {
            s.accum = .f32;
            s.depth = depth;
            s.tiles = .{ .{ .size = 16 }, .{ .size = 64 }, .{ .size = 16 } };
            s.served = &.{ .{ .rows = 1, .n = 72512, .k = 1024 }, .{ .rows = 17, .n = 1152, .k = 1152 }, .{ .rows = 1, .n = 4304, .k = 1152 }, .{ .rows = 1, .n = 1024, .k = 4304 } };
        },
        .norm => {
            s.accum = .f32;
            s.depth = depth;
            s.ops = 5;
            s.served = &.{ .{ .rows = 17, .n = 1, .k = 1024 }, .{ .rows = 1, .n = 1, .k = 1152 }, .{ .rows = 1, .n = 1, .k = 2304 } };
        },
        .gelu => s.ops = 5,
        .heads => {
            s.served = &.{ .{ .rows = 17, .n = 16, .k = 72 }, .{ .rows = 1, .n = 16, .k = 64 } };
            s.tiles[1] = .{ .size = 16 };
        },
        .four => {
            s.served = &.{.{ .rows = 416, .n = 1, .k = 1024 }};
            s.tiles[0] = .{ .size = 256, .min = 4, .step = 4 };
        },
        .embed, .scale => s.ops = 1,
        .add => {},
    }
    return s;
}

pub const Vision = struct {
    rig: Rig,
    kernels: hip.vision.Kernels,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    dense: ?Dense = null,
    x: []u16 = &.{},
    y: []u16 = &.{},
    weight: []u16 = &.{},
    bias: []u16 = &.{},
    ids: []i32 = &.{},
    ptrs: [5]u64 = @splat(0),
    stress: bool = false,

    pub fn init(a: std.mem.Allocator, r: *const hip.Runtime, stream: hip.Stream, kernels: hip.vision.Kernels, which: Which) Vision {
        return .{ .rig = .{ .a = a, .r = r, .s = stream }, .kernels = kernels, .which = which, .arena = .init(a) };
    }
    pub fn deinit(v: *Vision) void {
        v.rig.deinit();
        v.arena.deinit();
    }
    pub fn kernel(v: *Vision) g.Kernel {
        return .{ .spec = spec(v.which), .ctx = v, .vt = &vt };
    }
    const vt = g.hooks(Vision);

    pub fn count(v: *Vision) usize {
        const s = v.shape;
        return s.rows * switch (v.which) {
            .linear => s.n,
            .heads => s.n * s.k,
            else => s.k,
        };
    }

    pub fn prepare(v: *Vision, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.k > 16384 or s.n == 0 or s.rows > 16384) return false;
        if (v.which == .four and s.rows % 4 != 0) return false;
        v.rig.reset();
        _ = v.arena.reset(.retain_capacity);
        const a = v.arena.allocator();
        v.shape = s;
        v.stress = fill == .stress;
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        v.dense = null;
        const xn = if (v.which == .embed) 8 * s.k else if (v.which == .heads) s.rows * s.n * s.k else s.rows * s.k;
        v.x = try a.alloc(u16, xn);
        v.y = try a.alloc(u16, xn);
        v.weight = try a.alloc(u16, if (v.which == .linear) s.n * s.k else s.k);
        v.bias = try a.alloc(u16, if (v.which == .linear) s.n else s.k);
        for (v.x) |*x| x.* = c.toBf16(random.float(f32) * 4 - 2);
        for (v.y) |*x| x.* = c.toBf16(random.float(f32) * 4 - 2);
        for (v.weight) |*x| x.* = c.toBf16(0.5 + random.float(f32));
        for (v.bias) |*x| x.* = c.toBf16(random.float(f32) * 2 - 1);
        if (v.which == .linear) {
            const d = try Dense.init(a, s, 1);
            d.fillX(fill, seed, 2);
            v.x = d.x;
            for (v.weight, d.w, 0..) |*bits, *w, i| {
                const j = i % s.k;
                w.* = if (fill == .stress)
                    (if (j < s.k / 2) @as(f64, 1) else -1) * @as(f64, @floatFromInt(1 + ((j % (s.k / 2)) * 7 + i / s.k) % 6))
                else
                    c.fromBf16(bits.*);
                bits.* = c.toBf16(@floatCast(w.*));
            }
            if (fill == .stress) @memset(v.bias, 0);
            v.dense = d;
        } else if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |x, i| {
                const at = if (v.which == .norm) i * s.k + s.k / 2 else i;
                if (at < v.x.len) v.x[at] = c.toBf16(x);
            }
        } else if (fill == .stress and v.which == .norm) {
            for (v.x, 0..) |*x, i| x.* = c.toBf16(256 + @as(f32, @floatFromInt(i % s.k % 7)) * 2);
            @memset(v.weight, c.toBf16(1));
            @memset(v.bias, 0);
        }
        v.ids = try a.alloc(i32, s.rows);
        for (v.ids, 0..) |*id, i| id.* = @intCast(i % 8);
        v.ptrs = .{
            try v.rig.input(std.mem.sliceAsBytes(v.x)),
            try v.rig.input(std.mem.sliceAsBytes(v.weight)),
            try v.rig.input(std.mem.sliceAsBytes(v.bias)),
            try v.rig.input(if (v.which == .embed) std.mem.sliceAsBytes(v.ids) else std.mem.sliceAsBytes(v.y)),
            try v.rig.output(v.count() * 2),
        };
        return true;
    }

    pub fn run(v: *Vision) !void {
        const s = v.shape;
        const x: hip.ops.Tensor = .{ .ptr = v.ptrs[0], .kind = .bf16 };
        const y: hip.ops.Tensor = .{ .ptr = v.ptrs[3], .kind = .bf16 };
        const out: hip.ops.Tensor = .{ .ptr = v.ptrs[4], .kind = .bf16 };
        const k = v.kernels;
        const stream = v.rig.s;
        switch (v.which) {
            .linear => try k.linear(stream, x, v.ptrs[1], v.ptrs[2], out, s.rows, s.n, s.k),
            .norm => try k.norm(stream, x, v.ptrs[1], v.ptrs[2], out, s.rows, s.k, 1e-6),
            .gelu, .add, .scale => try k.point(stream, x, y, out, v.count(), switch (v.which) {
                .gelu => .gelu,
                .add => .add,
                else => .scale,
            }, 32),
            .heads => try k.layout(stream, x, out, s.rows, s.n, s.k, .heads_to_rows),
            .four => try k.layout(stream, x, out, s.rows, 1, s.k, .horizontal_four),
            .embed => try k.embed(stream, x, v.ptrs[3], out, s.rows, s.k, 32),
        }
        try stream.synchronize();
    }

    pub fn reference(v: *Vision, want: []f64, mag: []f64) !void {
        const s = v.shape;
        @memset(mag, 0);
        if (v.dense) |d| {
            try d.reference(want, mag);
            for (want, mag, 0..) |*w, *m, i| {
                const bias = c.fromBf16(v.bias[i % s.n]);
                w.* += bias;
                m.* += @abs(bias);
            }
            return;
        }
        for (want, mag, 0..) |*w, *m, i| {
            const row = i / s.k;
            const col = i % s.k;
            const x: f64 = c.fromBf16(v.x[@min(i, v.x.len - 1)]);
            w.* = switch (v.which) {
                .norm => blk: {
                    const values = v.x[row * s.k ..][0..s.k];
                    var mean: f64 = 0;
                    var absolute: f64 = 0;
                    for (values) |z| {
                        mean += c.fromBf16(z);
                        absolute += @abs(c.fromBf16(z));
                    }
                    mean /= @floatFromInt(s.k);
                    var variance: f64 = 0;
                    for (values) |z| variance += (c.fromBf16(z) - mean) * (c.fromBf16(z) - mean);
                    variance /= @floatFromInt(s.k);
                    const inv = 1 / @sqrt(variance + 1e-6);
                    const weight: f64 = c.fromBf16(v.weight[col]);
                    m.* = @abs(weight) * inv * (absolute / @as(f64, @floatFromInt(s.k)) + @abs(mean));
                    break :blk (x - mean) * inv * weight + c.fromBf16(v.bias[col]);
                },
                .gelu => (0.5 * x) * (1 + erf(x / @sqrt(@as(f64, 2)))),
                .add => x + c.fromBf16(v.y[i]),
                .scale => x * 32,
                .heads => c.fromBf16(v.x[((i / s.k % s.n) * s.rows + i / (s.n * s.k)) * s.k + i % s.k]),
                .four => c.fromBf16(v.x[(i / (4 * s.k) * 4 + i % 4) * s.k + i / 4 % s.k]),
                .embed => @as(f64, c.fromBf16(v.x[@as(usize, @intCast(v.ids[row])) * s.k + col])) * 32,
                .linear => unreachable,
            };
        }
    }

    pub fn simulate(v: *Vision, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (v.dense) |d| return d.simulate(acc, order, out);
        const s = v.shape;
        const a = v.arena.allocator();
        const terms = try a.alloc(f64, s.k);
        for (0..s.rows) |row| {
            const x = v.x[row * s.k ..][0..s.k];
            for (terms, x) |*t, bits| t.* = c.fromBf16(bits);
            const mean = contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(s.k));
            for (terms, x) |*t, bits| t.* = (c.fromBf16(bits) - mean) * (c.fromBf16(bits) - mean);
            const variance = contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(s.k));
            for (0..s.k) |j| out[row * s.k + j] = (c.fromBf16(x[j]) - mean) / @sqrt(variance + 1e-6) * c.fromBf16(v.weight[j]) + c.fromBf16(v.bias[j]);
        }
    }
};

extern "c" fn erf(f64) f64;
