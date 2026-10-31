//! LFM2's DSpark bf16 linears under gates 1-4: x W^T with bf16 inputs and fp32 sums, few rows or 64x64 tiles.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_lfm2_controls.zig");

const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

pub const Which = enum { rows, tile };
const names = [_][]const u8{ "metal lfm2 bf16 rows", "metal lfm2 bf16 tile" };

/// The drafter's matrices: q and o (2,048 from 2,048), k and v (512), gate and up (6,144), down (2,048 from 6,144).
const rows_served = [_]c.Shape{
    .{ .rows = 1, .n = 2048, .k = 2048 },  .{ .rows = 9, .n = 512, .k = 2048 },   .{ .rows = 16, .n = 6144, .k = 2048 },
    .{ .rows = 32, .n = 2048, .k = 6144 }, .{ .rows = 4, .n = 2048, .k = 10240 },
};
/// Shared holds past 16 rows, and a cold prompt's absorb up to 2,048 rows a pass.
const tile_served = [_]c.Shape{
    .{ .rows = 17, .n = 2048, .k = 2048 },    .{ .rows = 18, .n = 512, .k = 2048 },
    .{ .rows = 24, .n = 6144, .k = 2048 },    .{ .rows = 32, .n = 2048, .k = 6144 },
    .{ .rows = 33, .n = 2048, .k = 2048 },    .{ .rows = 128, .n = 512, .k = 2048 },
    .{ .rows = 2048, .n = 2048, .k = 10240 }, .{ .rows = 70, .n = 6144, .k = 2048 },
};

/// A lane's k / 32 fp32 fmas in turn, then the simdgroup's five-level sum.
fn rowsDepth(s: c.Shape) usize {
    return s.k / 32 + 8;
}

/// One 8-wide MMA step a k step of 8, accumulating in fp32.
fn tileDepth(s: c.Shape) usize {
    return s.k / 8 + 8;
}

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        // rows in groups of 16, a column a simdgroup (any n), k in 256-wide lane steps of 8
        .rows => .{ .name = names[0], .accum = .f32, .depth = rowsDepth, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{ .size = 8 }, .{ .size = 256, .step = 8, .min = 8 } }, .served = &rows_served },
        .tile => .{ .name = names[1], .accum = .f32, .depth = tileDepth, .out = .bf16, .tiles = .{ .{ .size = 64 }, .{ .size = 64 }, .{ .size = 32, .step = 32, .min = 32 } }, .served = &tile_served },
    };
}

const Args = extern struct { rows: u32, n: u32, k: u32 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: controls.Mutation,
    pipe: mtl.Pipeline,
    d: ?Dense = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, mutation: controls.Mutation) !Probe {
        const text = try controls.source(a, mutation);
        defer if (mutation != .none) a.free(text);
        const key: []const u8 = if (w == .rows) "lfm2_bf16_rows" else "lfm2_bf16_mma";
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .mutation = mutation, .pipe = try rig_.pipeline(device, text, key) };
    }

    pub fn deinit(p: *Probe) void {
        if (p.d) |d| d.deinit();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec(p.which);
        if (p.mutation != .none) s.name = p.mutation.name();
        return .{ .spec = s, .ctx = p, .vt = &vt, .control = p.mutation.gate() };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        if (p.which == .rows and (shape.k % 8 != 0 or shape.rows > 32)) return false;
        if (p.which == .tile and shape.k % 32 != 0) return false;
        if (p.d) |d| d.deinit();
        p.d = null;
        p.rig.reset();
        const d = try Dense.init(p.a, shape, 1);
        p.d = d;
        d.fillX(fill, seed, 2);
        // W as bf16 bits in [-1, 1), or gate 4's mirrored integers whose halves cancel (partials under 2^24)
        const bits = try p.a.alloc(u16, shape.n * shape.k);
        defer p.a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6d);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*wv, b| wv.* = c.fromBf16(b);
        p.x = try p.rig.input(std.mem.sliceAsBytes(d.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(bits));
        p.out = try p.rig.output(d.outputs() * c.size(spec(p.which).out));
        return true;
    }

    pub fn run(p: *Probe) !void {
        const d = p.d.?;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.x.buf, p.x.off, 0);
        e.setBuffer(p.w.buf, p.w.off, 1);
        const args = Args{ .rows = @intCast(d.shape.rows), .n = @intCast(d.shape.n), .k = @intCast(d.shape.k) };
        e.setBytes(std.mem.asBytes(&args), 2);
        e.setBuffer(p.out.buf, p.out.off, 3);
        if (p.which == .rows) {
            e.dispatchGroups(mtl.Size.of((d.shape.n + 7) / 8, (d.shape.rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
        } else e.dispatchGroups(mtl.Size.of((d.shape.n + 63) / 64, (d.shape.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.d.?.outputs();
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        return p.d.?.reference(want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.d.?.simulate(acc, order, out);
    }
};
