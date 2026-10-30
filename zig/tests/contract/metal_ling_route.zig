//! Ling's group-limited route under the contract: picks and weights against route.zig's host formula.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const route = @import("ling_route");

const c = contract.spec;
const g = contract.gates;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "NaN never wins a group or a pick; +Inf beats every finite; ties go to the lower group then the lower expert id";

const source = @import("ling_metal").route;

/// rows: tokens; n: top-k; k: experts. Served: tiny (128, 8 groups of 16) and flash (512, 8 groups of 64).
pub fn spec() c.Spec {
    return .{
        .name = "metal ling route tf_ling_route",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .served = &.{ .{ .rows = 8, .n = 8, .k = 128 }, .{ .rows = 1, .n = 8, .k = 128 }, .{ .rows = 4, .n = 8, .k = 512 } },
    };
}

fn shapeOf(shape: c.Shape) route.Shape {
    const ng: u32 = if (shape.k == 512 or shape.k == 128) 8 else 1;
    return .{ .experts = @intCast(shape.k), .topk = @intCast(shape.n), .n_group = ng, .topk_group = if (ng > 1) 4 else 1, .scale = 2.5 };
}

const Args = extern struct { rows: i32, ne: i32, topk: i32, ng: i32, keep: i32, scale: f32 };

pub const Route = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    s: route.Shape = undefined,
    logits: []f32 = &.{},
    bias: []f32 = &.{},
    x: rig_.Ref = undefined,
    b: rig_.Ref = undefined,
    pick: rig_.Ref = undefined,
    wts: rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Route {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "tf_ling_route") };
    }

    pub fn deinit(t: *Route) void {
        t.a.free(t.logits);
        t.a.free(t.bias);
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *Route) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Route);

    pub fn prepare(t: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != 8 or (shape.k != 128 and shape.k != 512)) return false;
        t.rig.reset();
        t.shape = shape;
        t.s = shapeOf(shape);
        t.a.free(t.logits);
        t.logits = &.{};
        t.a.free(t.bias);
        t.bias = &.{};
        const n = shape.rows * shape.k;
        t.logits = try t.a.alloc(f32, n);
        t.bias = try t.a.alloc(f32, shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.logits) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 33))) / 8 - 2;
        for (t.bias) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 9))) / 8 - 0.5;
        if (fill == .nonfinite) {
            t.logits[1] = std.math.nan(f32);
            if (shape.rows > 1) t.logits[shape.k + 3] = std.math.inf(f32);
            if (shape.rows > 2) t.logits[2 * shape.k + 5] = -std.math.inf(f32);
            if (shape.rows > 3) {
                t.logits[3 * shape.k] = std.math.inf(f32);
                t.logits[3 * shape.k + 1] = -std.math.inf(f32);
            }
        }
        t.x = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.b = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        t.pick = try t.rig.output(shape.rows * shape.n * 4);
        t.wts = try t.rig.output(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(t: *Route) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.x.buf, t.x.off, 0);
        p.e.setBuffer(t.b.buf, t.b.off, 1);
        p.e.setValue(Args{
            .rows = @intCast(t.shape.rows),
            .ne = @intCast(t.s.experts),
            .topk = @intCast(t.s.topk),
            .ng = @intCast(t.s.n_group),
            .keep = @intCast(t.s.topk_group),
            .scale = t.s.scale,
        }, 2);
        p.e.setBuffer(t.pick.buf, t.pick.off, 3);
        p.e.setBuffer(t.wts.buf, t.wts.off, 4);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Route) usize {
        return t.shape.rows * t.shape.n * 8;
    }

    pub fn reference(_: *Route, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(t: *Route, at: g.Gate) !?[]const u8 {
        const topk = t.s.topk;
        const ids = std.mem.bytesAsSlice(i32, t.rig.bytes(t.pick, t.shape.rows * topk * 4));
        const wts = std.mem.bytesAsSlice(f32, t.rig.bytes(t.wts, t.shape.rows * topk * 4));
        var planted: usize = 0;
        for (0..t.shape.rows) |r| {
            const row = t.logits[r * t.shape.k ..][0..t.shape.k];
            var want: [route.max_topk]route.Pick = undefined;
            route.selectF32(row, t.bias, want[0..topk], t.s);
            for (0..topk) |k| {
                const got_id: u32 = @intCast(ids[r * topk + k]);
                if (got_id != want[k].id)
                    return try std.fmt.bufPrint(&t.msg, "row {d} pick {d}: id {d}, host {d}", .{ r, k, got_id, want[k].id });
                const got_w = wts[r * topk + k];
                if (@abs(got_w - want[k].wt) > 1e-5)
                    return try std.fmt.bufPrint(&t.msg, "row {d} pick {d}: wt {e:.6}, host {e:.6}", .{ r, k, got_w, want[k].wt });
            }
            var nf: bool = false;
            for (row) |v| nf = nf or !std.math.isFinite(v);
            if (nf) planted += 1;
        }
        if (at == .nonfinite and planted == 0) return try std.fmt.bufPrint(&t.msg, "no non-finite row planted", .{});
        return null;
    }
};
