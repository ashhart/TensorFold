//! The image tower's LayerNorm and attention softmax rows.
const std = @import("std");
const mtl = @import("metal");
const rows = @import("contract").rows;
const u = @import("metal_inkling2_vision_util.zig");
const r = @import("rig.zig");
const c = u.c;
const g = u.g;

pub const Kind = enum { layer_norm, softmax };

pub fn kernelName(comptime k: Kind) []const u8 {
    return "ik2v_" ++ @tagName(k);
}

pub fn name(comptime k: Kind) []const u8 {
    return "metal inkling2 vision " ++ kernelName(k);
}

const layer_shapes = [_]c.Shape{ .{ .rows = 9, .n = 1024, .k = 1024 }, .{ .rows = 5, .n = 1024, .k = 1024 } };
const softmax_shapes = [_]c.Shape{ .{ .rows = 9, .n = 1025, .k = 1025 }, .{ .rows = 5, .n = 5, .k = 5 }, .{ .rows = 9, .n = 16385, .k = 16385 } };

fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 12;
}

pub const Probe = struct {
    rig: r.Rig,
    pipe: mtl.Pipeline,
    arena: std.heap.ArenaAllocator,
    kind: Kind,
    label: []const u8,
    s: c.Shape = undefined,
    words: []const u16 = &.{},
    weights: []const u16 = &.{},
    bias: []const u16 = &.{},
    scores: []const f32 = &.{},
    refs: [3]r.Ref = undefined,
    out: r.Ref = undefined,
    stress: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, k: Kind, label: []const u8) !Probe {
        const pipe = switch (k) {
            inline else => |tag| try mtl.Pipeline.init(device, lib, kernelName(tag), false),
        };
        return .{ .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .arena = .init(a), .kind = k, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe, control_gate: ?c.Gate) g.Kernel {
        const soft = p.kind == .softmax;
        return .{ .ctx = p, .vt = &vt, .control = control_gate, .spec = .{
            .name = p.label,
            .accum = .f32,
            .depth = depth,
            .ops = 12,
            .in_place = soft,
            .out = if (soft) .f32 else .bf16,
            .store = if (soft) .bf16 else null,
            .tiles = .{ .{}, .{}, .{ .size = 256 } },
            .served = if (soft) &softmax_shapes else &layer_shapes,
            .sums = .{ .f16 = if (soft) 0 else 100, .bf16 = 0, .exact = !soft, .why = "BF16 store limits discrimination in normalized outputs; exp terms are not dyadic" },
        } };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = u.reset(&p.arena, &p.rig);
        p.s = s;
        p.stress = fill == .stress;
        if (p.kind == .softmax) {
            const x = try a.alloc(f32, s.rows * s.k);
            var prng = std.Random.DefaultPrng.init(seed);
            for (x) |*v| v.* = prng.random().float(f32) * 16 - 8;
            if (fill == .nonfinite) {
                x[0] = std.math.nan(f32);
                x[s.k + s.k / 2] = std.math.inf(f32);
                x[2 * s.k] = -std.math.inf(f32);
                x[3 * s.k] = std.math.inf(f32);
                x[4 * s.k - 1] = -std.math.inf(f32);
            }
            p.scores = x;
            p.out = try p.rig.output(x.len * 4);
            return true;
        }
        const base = try rows.Rows.init(a, .layer_norm, s);
        base.fill(fill, seed);
        p.words = base.x;
        p.weights = base.w;
        p.bias = base.b;
        p.out = try p.rig.output(s.rows * s.k * 2);
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.words));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const w: u32 = @intCast(p.s.k);
        const n: u32 = @intCast(p.s.rows);
        e.setPipeline(p.pipe);
        if (p.kind == .softmax) {
            const data = std.mem.sliceAsBytes(p.scores);
            @memcpy(p.rig.bytes(p.out, data.len), data);
            e.setBuffer(p.out.buf, p.out.off, 0);
            e.setValue(w, 1);
            e.setValue(@as(f32, 0.125), 2);
        } else {
            for (p.refs, 0..) |x, i| e.setBuffer(x.buf, x.off, i);
            e.setBuffer(p.out.buf, p.out.off, 3);
            e.setValue([3]u32{ n, w, @bitCast(@as(f32, 1e-5)) }, 4);
        }
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.s.rows * p.s.k;
    }

    fn roundSum(v: []const f64, acc: c.Accum, order: g.Order) f64 {
        return rows.sumIn(acc, order, v);
    }

    fn calculate(p: *Probe, acc: c.Accum, order: g.Order, want: []f64, mags: ?[]f64) !void {
        const k = p.s.k;
        const kf: f64 = @floatFromInt(k);
        const a = p.rig.a;
        const terms = try a.alloc(f64, k);
        defer a.free(terms);
        for (0..p.s.rows) |i| {
            const out = want[i * k ..][0..k];
            const mag = if (mags) |m| m[i * k ..][0..k] else null;
            if (p.kind == .softmax) {
                var top = -std.math.inf(f64);
                const src = p.scores[i * k ..][0..k];
                for (terms, src) |*x, v| {
                    x.* = u.val(u.word(u.val(u.word(v)) / 8));
                    top = @max(top, x.*);
                }
                for (terms) |*x| x.* = @exp(x.* - top);
                const z = roundSum(terms, acc, order);
                for (out, terms) |*y, x| y.* = x / z;
                if (mag) |m| for (m, out) |*x, y| {
                    x.* = @abs(y);
                };
                continue;
            }
            var abs: f64 = 0;
            for (terms, 0..) |*x, j| {
                x.* = u.val(p.words[i * k + j]);
                abs += @abs(x.*);
            }
            const mean = roundSum(terms, acc, order) / kf;
            for (terms) |*x| x.* = (x.* - mean) * (x.* - mean);
            const inv = 1 / @sqrt(roundSum(terms, acc, order) / kf + 1e-5);
            for (out, 0..) |*y, j| {
                const x = u.val(p.words[i * k + j]) - mean;
                const w = u.val(p.weights[j]);
                const b = u.val(p.bias[j]);
                y.* = x * inv * w + b;
                if (mag) |m| m[j] = @abs(w) * inv * (@abs(x) + abs / kf) + @abs(b);
            }
        }
    }

    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        // Affine cancellation can make |y| much smaller than the terms whose rounding influences it.
        const work = if (p.stress and p.kind != .softmax) 12 else depth(p.s) + 16;
        const delta = c.gamma(@floatFromInt(work), c.unit(.f32));
        for (want, mag, allow) |y, m, *b| {
            const error_f32 = m * delta;
            b.* = error_f32 + (error_f32 + @abs(y)) * c.unit(.bf16) + 0x1p-133;
        }
    }

    pub fn reference(p: *Probe, out: []f64, mag: []f64) !void {
        try p.calculate(.none, .sequential, out, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.calculate(acc, order, out, null);
    }
};
