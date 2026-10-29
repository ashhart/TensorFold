//! Production semantic CFG: independent rows, conditional-only aliasing and untouched codec tails.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const source = @import("metal_drama3.zig").source;
const c = contract.spec;
const g = contract.gates;
const tail = 17;
pub const gammas = [_]f32{ 1, 1.5, 2, 3, 4, 5 };

fn depth(_: c.Shape) usize {
    return 0;
}

pub fn spec(gamma: f32) c.Spec {
    return .{
        .name = if (gamma == 1) "metal drama3 d3_cfg_mix gamma=1 identity" else if (gamma == 1.5) "metal drama3 d3_cfg_mix gamma=1.5" else if (gamma == 2) "metal drama3 d3_cfg_mix gamma=2" else if (gamma == 3) "metal drama3 d3_cfg_mix gamma=3" else if (gamma == 4) "metal drama3 d3_cfg_mix gamma=4" else "metal drama3 d3_cfg_mix gamma=5",
        // Pointwise subtraction, multiply and addition; no reduction or accumulated partial sums.
        .accum = .none,
        .depth = depth,
        .ops = 3,
        .out = .f32,
        .in_place = true,
        .nan = .{ .stated = "gamma=1 preserves all conditional bits without reading unconditional; otherwise IEEE formula" },
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = &.{ .{ .rows = 1, .n = 4097, .k = 1 }, .{ .rows = 5, .n = 4097, .k = 1 } },
    };
}

pub const Mix = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    gamma: f32,
    shape: c.Shape = undefined,
    conditional: []f32 = &.{},
    unconditional: []f32 = &.{},
    u: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, gamma: f32) !Mix {
        if (!std.math.isFinite(gamma) or gamma < 1 or gamma > 5) return error.InvalidGamma;
        const src = try source(a, "drama3 guidance d3_cfg_mix");
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .gamma = gamma, .pipe = try rig_.pipeline(device, src, "d3_cfg_mix") };
    }

    pub fn deinit(t: *Mix) void {
        t.a.free(t.conditional);
        t.a.free(t.unconditional);
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *Mix) g.Kernel {
        return .{ .spec = spec(t.gamma), .ctx = t, .vt = &vt };
    }
    const vt = blk: {
        var hooks = g.hooks(Mix);
        hooks.overlap = overlap;
        break :blk hooks;
    };
    fn overlap(_: *anyopaque) bool {
        return true;
    }

    pub fn prepare(t: *Mix, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != 1 or shape.n == 0 or shape.n > 4097) return false;
        t.rig.reset();
        t.shape = shape;
        t.a.free(t.conditional);
        t.conditional = &.{};
        t.a.free(t.unconditional);
        t.unconditional = &.{};
        t.conditional = try t.a.alloc(f32, t.count());
        t.unconditional = try t.a.alloc(f32, t.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.conditional, t.unconditional) |*cv, *uv| {
            cv.* = rng.float(f32) * 16 - 8;
            uv.* = rng.float(f32) * 16 - 8;
        }
        t.conditional[0] = -0.0;
        if (t.gamma == 1) t.unconditional[0] = std.math.nan(f32);
        if (fill == .nonfinite) {
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (values, 0..) |v, row| {
                t.conditional[row * (shape.n + tail) + 1] = v;
                t.unconditional[row * (shape.n + tail) + 2] = v;
            }
            t.conditional[3 * (shape.n + tail) + 3] = std.math.inf(f32);
            t.unconditional[3 * (shape.n + tail) + 3] = std.math.inf(f32);
        }
        t.u = try t.rig.input(std.mem.sliceAsBytes(t.unconditional));
        // Only conditional logits are mutable. Their initial bytes live on the host, not in an immutable rig slot.
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Mix) !void {
        @memcpy(t.rig.bytes(t.out, t.count() * 4), std.mem.sliceAsBytes(t.conditional));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        const Args = extern struct { count: u32, gamma: f32 };
        p.e.setValue(Args{ .count = @intCast(t.shape.n), .gamma = t.gamma }, 2);
        for (0..t.shape.rows) |row| {
            const off = row * (t.shape.n + tail) * 4;
            p.e.setBuffer(t.out.buf, t.out.off + off, 0);
            p.e.setBuffer(t.u.buf, t.u.off + off, 1);
            p.e.dispatchGroups(mtl.Size.of((t.shape.n + 255) / 256, 1, 1), mtl.Size.of(256, 1, 1));
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Mix) usize {
        return t.shape.rows * (t.shape.n + tail);
    }

    pub fn reference(t: *Mix, want: []f64, mag: []f64) !void {
        for (want, mag, t.conditional, t.unconditional, 0..) |*w, *m, cv, uv, i| {
            const active = t.gamma != 1 and i % (t.shape.n + tail) < t.shape.n;
            w.* = if (active) @as(f64, cv) + (@as(f64, t.gamma) - 1) * (@as(f64, cv) - uv) else cv;
            m.* = if (active) @abs(cv) + (@as(f64, t.gamma) - 1) * (@abs(cv) + @abs(uv)) else 0;
        }
    }

    pub fn check(t: *Mix, at: g.Gate) !?[]const u8 {
        const out = t.rig.bytes(t.out, t.count() * 4);
        var nonfinite: usize = 0;
        for (t.conditional, t.unconditional, 0..) |cv, uv, i| {
            const got = c.read(.f32, out, i);
            if (t.gamma == 1 or i % (t.shape.n + tail) >= t.shape.n) {
                const bits = std.mem.readInt(u32, out[i * 4 ..][0..4], .little);
                if (bits != @as(u32, @bitCast(cv)))
                    return try std.fmt.bufPrint(&t.msg, "conditional identity/codec tail changed at {d}", .{i});
                nonfinite += @intFromBool(!std.math.isFinite(cv));
                continue;
            }
            const alpha = @as(f64, t.gamma) - 1;
            const want = @as(f64, cv) + alpha * (@as(f64, cv) - uv);
            nonfinite += @intFromBool(!std.math.isFinite(want));
            if (c.classOf(got) != c.classOf(want))
                return try std.fmt.bufPrint(&t.msg, "CFG class mismatch at {d}", .{i});
            // Three rounded FP32 operations (or their contraction), bounded by the term magnitudes.
            const bound = c.gamma(3, c.unit(.f32)) * (@abs(cv) + alpha * (@abs(cv) + @abs(uv)));
            if (at != .nonfinite and std.math.isFinite(want) and @abs(got - want) > bound)
                return try std.fmt.bufPrint(&t.msg, "CFG FP64 bound exceeded at {d}", .{i});
        }
        if (at == .nonfinite and (nonfinite == 0 or nonfinite == t.count())) return "nonfinite fixture proves nothing";
        return null;
    }
};
