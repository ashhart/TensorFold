//! Compact routing: original-ID ordering, masking before softmax and compact slot translation.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const source = @import("metal_drama3.zig").source;
const c = contract.spec;
const g = contract.gates;
pub const Variant = enum { pruned8, pruned73, all256 };
const offset = 19;
const stride = 291;

fn depth(_: c.Shape) usize {
    return 21;
}
pub fn spec(v: Variant) c.Spec {
    return .{
        .name = switch (v) {
            .pruned8 => "metal drama3 d3_router_compact retained=8",
            .pruned73 => "metal drama3 d3_router_compact retained=73",
            .all256 => "metal drama3 d3_router_compact retained=256",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 12,
        .out = .f32,
        .nan = .{ .stated = "dropped nonfinites ignored; invalid softmax gives NaN weights and first8 retained original IDs" },
        .served = &.{ .{ .rows = 8, .n = 8, .k = 256 }, .{ .rows = 1, .n = 8, .k = 256 }, .{ .rows = 168, .n = 8, .k = 256 } },
        // Seven exp(-10) terms survive FP32 but both reduced-precision denominators round to 1 in every order.
        .sums = .{ .exact = false },
    };
}

pub const Router = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    variant: Variant,
    shape: c.Shape = undefined,
    logits: []u16 = &.{},
    slots: [256]i32 = undefined,
    previous: []u8 = &.{},
    runs: usize = 0,
    x: rig_.Ref = undefined,
    map: rig_.Ref = undefined,
    picks: rig_.Ref = undefined,
    weights: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, variant: Variant) !Router {
        const src = try source(a, "drama3 nvfp4 d3_router_compact");
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .variant = variant, .pipe = try rig_.pipeline(device, src, "d3_router_compact") };
    }
    pub fn deinit(t: *Router) void {
        t.a.free(t.logits);
        t.a.free(t.previous);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Router) g.Kernel {
        return .{ .spec = spec(t.variant), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Router);

    pub fn prepare(t: *Router, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != 8 or shape.k != 256) return false;
        t.rig.reset();
        t.shape = shape;
        t.runs = 0;
        t.a.free(t.logits);
        t.logits = &.{};
        t.a.free(t.previous);
        t.previous = &.{};
        t.logits = try t.a.alloc(u16, shape.rows * stride);
        t.previous = try t.a.alloc(u8, t.count() * 4);
        @memset(&t.slots, -1);
        const kept: usize = switch (t.variant) {
            .pruned8 => 8,
            .pruned73 => 73,
            .all256 => 256,
        };
        // Permuted compact slots prove that ties are resolved in original ID space, not slot space.
        for (0..kept) |slot| t.slots[(slot * 37 + 7) % 256] = @intCast(kept - 1 - slot);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(t.logits, c.toBf16(91));
        for (0..shape.rows) |row| for (0..256) |id| {
            const score: f32 = if (t.slots[id] < 0) 100 else if (row == 0 and fill != .stress) -2 else @as(f32, @floatFromInt(rng.uintLessThan(u32, 33))) / 8 - 2;
            t.logits[row * stride + offset + id] = c.toBf16(score);
        };
        if (fill == .stress) for (0..shape.rows) |row| {
            var rank: usize = 0;
            for (t.slots, 0..) |slot, id| if (slot >= 0) {
                t.logits[row * stride + offset + id] = c.toBf16(if (rank == 0) 0 else if (rank < 8) -10 else -80);
                rank += 1;
            };
        };
        if (fill == .nonfinite) {
            var ids: [256]usize = undefined;
            var n: usize = 0;
            for (t.slots, 0..) |slot, id| if (slot >= 0) {
                ids[n] = id;
                n += 1;
            };
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (values, 0..) |plant, row| t.logits[row * stride + offset + ids[1]] = c.toBf16(plant);
            t.logits[3 * stride + offset + ids[0]] = c.toBf16(std.math.inf(f32));
            t.logits[3 * stride + offset + ids[1]] = c.toBf16(-std.math.inf(f32));
            if (kept < 256) for (t.slots, 0..) |slot, id| if (slot < 0) {
                t.logits[4 * stride + offset + id] = c.toBf16(values[id % 3]);
            };
            for (t.slots, 0..) |slot, id| if (slot >= 0) {
                t.logits[5 * stride + offset + id] = c.toBf16(-std.math.inf(f32));
            };
        }
        t.x = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.map = try t.rig.input(std.mem.sliceAsBytes(&t.slots));
        t.picks = try t.rig.scratch(t.count() * 4);
        t.weights = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Router) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.x.buf, t.x.off, 0);
        p.e.setBuffer(t.picks.buf, t.picks.off, 1);
        p.e.setBuffer(t.weights.buf, t.weights.off, 2);
        p.e.setBuffer(t.map.buf, t.map.off, 3);
        p.e.setValue([2]u32{ stride, offset }, 4);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
        const picked = t.rig.bytes(t.picks, t.count() * 4);
        if (t.runs > 0 and !std.mem.eql(u8, t.previous, picked)) return error.RouterPicksNotRepeatable;
        @memcpy(t.previous, picked);
        t.runs += 1;
    }
    pub fn count(t: *Router) usize {
        return t.shape.rows * 8;
    }

    fn selected(t: *Router, row: usize, ids: *[8]usize) bool {
        var used: [256]bool = @splat(false);
        var poisoned = false;
        var finite = false;
        for (t.slots, 0..) |slot, id| if (slot >= 0) {
            const v = c.fromBf16(t.logits[row * stride + offset + id]);
            poisoned = poisoned or std.math.isNan(v) or v == std.math.inf(f32);
            finite = finite or std.math.isFinite(v);
        };
        poisoned = poisoned or !finite;
        for (ids) |*out| {
            var best: ?usize = null;
            for (t.slots, 0..) |slot, id| {
                if (slot < 0 or used[id]) continue;
                if (best == null or (!poisoned and t.value(row, id) > t.value(row, best.?))) best = id;
            }
            out.* = best.?;
            used[out.*] = true;
        }
        return poisoned;
    }
    fn value(t: *Router, row: usize, id: usize) f64 {
        return c.fromBf16(t.logits[row * stride + offset + id]);
    }

    pub fn reference(t: *Router, want: []f64, mag: []f64) !void {
        for (0..t.shape.rows) |row| {
            var ids: [8]usize = undefined;
            const poisoned = t.selected(row, &ids);
            var terms: [8]f64 = undefined;
            var sum: f64 = 0;
            for (ids, &terms) |id, *term| {
                term.* = @exp(t.value(row, id) - t.value(row, ids[0]));
                sum += term.*;
            }
            for (terms, 0..) |v, j| {
                want[row * 8 + j] = if (poisoned) std.math.nan(f64) else v / sum;
                mag[row * 8 + j] = @abs(want[row * 8 + j]);
            }
        }
    }

    pub fn simulate(t: *Router, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.shape.rows) |row| {
            var ids: [8]usize = undefined;
            _ = t.selected(row, &ids);
            var all: [256]f64 = @splat(0);
            for (t.slots, 0..) |slot, id| if (slot >= 0) {
                all[id] = @exp(t.value(row, id) - t.value(row, ids[0]));
            };
            const total = contract.rows.sumIn(acc, order, &all);
            var chosen: [8]f64 = undefined;
            for (ids, &chosen) |id, *v| v.* = all[id] / total;
            const denom = contract.rows.sumIn(acc, order, &chosen);
            for (chosen, 0..) |v, j| out[row * 8 + j] = v / denom;
        }
    }

    pub fn check(t: *Router, at: g.Gate) !?[]const u8 {
        const picked = t.rig.bytes(t.picks, t.count() * 4);
        const weights = t.rig.bytes(t.weights, t.count() * 4);
        const values = try t.a.alloc(f64, t.count() * 2);
        defer t.a.free(values);
        const want = values[0..t.count()];
        const mag = values[t.count()..];
        try t.reference(want, mag);
        for (0..t.shape.rows) |row| {
            var ids: [8]usize = undefined;
            _ = t.selected(row, &ids);
            for (ids, 0..) |id, j| {
                const i = row * 8 + j;
                const slot = std.mem.readInt(i32, picked[i * 4 ..][0..4], .little);
                if (slot != t.slots[id]) return try std.fmt.bufPrint(&t.msg, "row {d} rank {d}: compact slot {d}, expected {d} (original ID {d})", .{ row, j, slot, t.slots[id], id });
                const got = c.read(.f32, weights, i);
                if (c.classOf(got) != c.classOf(want[i])) return "router weight class differs from masked FP64 softmax";
                if (at != .nonfinite and @abs(got - want[i]) > c.bound(spec(t.variant), t.shape, want[i], mag[i]))
                    return "router weight exceeds FP64 reduction bound";
            }
        }
        return null;
    }
};
