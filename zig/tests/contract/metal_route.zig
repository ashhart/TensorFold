//! The MoE routers' logits kernel (core/moe_route.metal) under the contract on Metal, at each shape a family compiles.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const route = tf.moe_route;

/// GLM's router and indexer gate (4,096 wide), and the 35B-A3B's router with its shared gate column (2,048 wide).
pub const Which = enum { glm_router, glm_igate, qwen35b_router };

pub fn shapeOf(w: Which) route.Shape {
    return switch (w) {
        .glm_router => .{ .hidden = 4096, .experts = 288, .topk = 8 },
        .glm_igate => .{ .hidden = 4096, .experts = 128, .topk = 8, .out_bf16 = true },
        .qwen35b_router => .{ .hidden = 2048, .experts = 260, .topk = 8 },
    };
}

/// A lane's chain of four 4-wide steps (16 terms), one simd_sum tree, then the eight simdgroups' partials in order.
fn depth(_: c.Shape) usize {
    return 16 + 5 + 7;
}

/// The rows a family calls it at: GLM's decode and prompt chunks to 16,384 (tails any height), the 35B's to 128.
fn served(comptime w: Which) []const c.Shape {
    const s = comptime shapeOf(w);
    const n = s.experts;
    const k = s.hidden;
    if (w == .qwen35b_router) return comptime &.{ .{ .rows = 1, .n = n, .k = k }, .{ .rows = 16, .n = n, .k = k }, .{ .rows = 128, .n = n, .k = k } };
    return comptime &.{ .{ .rows = 1, .n = n, .k = k }, .{ .rows = 16, .n = n, .k = k }, .{ .rows = 2047, .n = n, .k = k }, .{ .rows = 16384, .n = n, .k = k } };
}

/// Which router a registered launch compiles: its label names the family.
pub fn named(label: []const u8) Which {
    if (std.mem.indexOf(u8, label, "35B") != null) return .qwen35b_router;
    return if (std.mem.indexOf(u8, label, "indexer") != null) .glm_igate else .glm_router;
}

pub fn spec(w: Which) c.Spec {
    const tiles: [3]c.Tile = .{ .{ .size = 4 }, .{}, .{} };
    return switch (w) {
        .glm_router => .{ .name = "metal moe_route logits glm router tf_route_logits", .accum = .f32, .depth = depth, .out = .f32, .tiles = tiles, .served = served(.glm_router) },
        .glm_igate => .{ .name = "metal moe_route logits glm indexer gate tf_route_logits", .accum = .f32, .depth = depth, .out = .bf16, .tiles = tiles, .served = served(.glm_igate) },
        .qwen35b_router => .{ .name = "metal moe_route logits qwen27 35B-A3B router tf_route_logits", .accum = .f32, .depth = depth, .out = .f32, .tiles = tiles, .served = served(.qwen35b_router) },
    };
}

pub const Route = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    pipe: mtl.Pipeline,
    d: ?contract.dense.Dense = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which) !Route {
        const text = try route.source(a, shapeOf(which));
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .pipe = try rig_.pipeline(device, text, route.names[0]) };
    }

    pub fn deinit(t: *Route) void {
        if (t.d) |d| d.deinit();
        t.pipe.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *Route) g.Kernel {
        return .{ .spec = spec(t.which), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Route);

    /// A router of bf16 weights in [-1, 1), or gate 4's mirrored integers 1-30: big lane partials, halves cancelling.
    pub fn prepare(t: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const s = shapeOf(t.which);
        if (shape.k != s.hidden or shape.n != s.experts or shape.rows == 0) return false;
        if (t.d) |d| d.deinit();
        t.d = null;
        t.rig.reset();
        const d = try contract.dense.Dense.init(t.a, shape, 1);
        t.d = d;
        d.fillX(fill, seed, 1);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5eed);
        const rng = prng.random();
        const k = shape.k;
        const bits = try t.a.alloc(u16, shape.n * k);
        defer t.a.free(bits);
        for (0..shape.n) |j| for (0..k) |i| {
            const v: f32 = if (fill != .stress) rng.float(f32) * 2 - 1 else @floatFromInt(1 + ((i % (k / 2)) * 7 + j) % 30);
            bits[j * k + i] = c.toBf16(if (fill == .stress and i >= k / 2) -v else v);
            d.w[j * k + i] = c.fromBf16(bits[j * k + i]);
        };
        const packed_w = try t.a.alloc(u16, bits.len);
        defer t.a.free(packed_w);
        route.pack(bits, packed_w, shape.n, k);
        t.x = try t.rig.input(std.mem.sliceAsBytes(d.x));
        t.w = try t.rig.input(std.mem.sliceAsBytes(packed_w));
        t.out = try t.rig.output(shape.rows * shape.n * c.size(spec(t.which).out));
        return true;
    }

    pub fn run(t: *Route) !void {
        const d = t.d.?;
        const p = t.rig.begin();
        route.logits(p.e, t.pipe, shapeOf(t.which), t.x, t.w, t.out, @intCast(d.shape.rows));
        try t.rig.finish(p);
    }

    pub fn count(t: *Route) usize {
        return t.d.?.outputs();
    }

    pub fn reference(t: *Route, want: []f64, mag: []f64) !void {
        return t.d.?.reference(want, mag);
    }

    pub fn simulate(t: *Route, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.d.?.simulate(acc, order, out);
    }
};
