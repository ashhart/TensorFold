//! ud_moe_gate under the contract: softmax top 6; near-ties may swap, so the kernel's own picks are checked.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_uocr_decoder_common.zig");

const c = contract.spec;
const g = contract.gates;
const H = k.hidden;
const E = k.experts;
const P = k.topk;
const Mutation = k.Mutation;
const u = c.unit(.f32);

/// A lane's two 4-wide dots and the add each 256 columns, then the simd_sum tree.
fn depth(s: c.Shape) usize {
    return 6 * ((s.k + 255) / 256) + 5;
}

pub const Gate = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    w: [E * H]u16 = undefined,
    ins: [2]rig_.Ref = undefined,
    wts: rig_.Ref = undefined,
    ids: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    /// Rows: tokens; n: experts (64); k: width (1280). Outputs: weights [rows, 6] (against float64), picks [rows, 6].
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_moe_gate softmax top 6",
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .nan = .{ .stated = "a row with a NaN or +Inf logit gives NaN probabilities and picks experts 0 to 5; a -Inf logit's expert weighs 0" },
            .served = &.{ .{ .rows = k.prompt_rows, .n = E, .k = H }, .{ .rows = 1, .n = E, .k = H } },
            // the logits sum exactly in fp32 at gate 4; the softmax's mass does not
            .sums = .{ .exact = false },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Gate {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_moe_gate"), .m = m };
    }
    pub fn deinit(t: *Gate) void {
        t.a.free(t.x);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Gate) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Gate);

    pub fn prepare(t: *Gate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != E or shape.k != H) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, shape.rows * H);
        const h = H / 2;
        if (fill == .stress) {
            // rows' halves mirror (8 nudged) against gate rows mirrored with the sign turned: huge partials
            for (0..shape.rows) |r| for (0..h) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                t.x[r * H + i] = c.toBf16(v);
                t.x[r * H + h + i] = c.toBf16(v + @as(f32, if (i < 8) 1 else 0));
            };
            for (0..E) |e| for (0..h) |i| {
                const q: f32 = @floatFromInt(if (i < 8) rng.uintLessThan(u32, 2) else 1 + rng.uintLessThan(u32, 7));
                t.w[e * H + i] = c.toBf16(q);
                t.w[e * H + h + i] = c.toBf16(-q);
            };
        } else {
            k.fillBf16(rng, t.x, 2);
            k.fillBf16(rng, &t.w, 0.05);
        }
        if (fill == .nonfinite) {
            // row 0 NaN, row 1 +Inf at column 3, expert 7 -Inf at column 5 (later rows hold 1 there)
            t.x[0] = c.toBf16(k.nan32);
            t.x[H + 3] = c.toBf16(k.inf32);
            t.w[3] = c.toBf16(0.5);
            t.w[7 * H + 5] = c.toBf16(-k.inf32);
            for (2..shape.rows) |r| t.x[r * H + 5] = c.toBf16(1);
            for (0..E) |e| if (e != 7) {
                t.w[e * H + 5] = c.toBf16(0.25);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(&t.w));
        t.wts = try t.rig.output(shape.rows * P * 4);
        t.ids = try t.rig.output(shape.rows * P * 4);
        return true;
    }

    pub fn run(t: *Gate) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.ids.buf, t.ids.off, 2);
        p.e.setBuffer(t.wts.buf, t.wts.off, 3);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Gate) usize {
        return t.shape.rows * P;
    }

    /// Row r's probabilities (logits and mass in `acc` and `order`) and each one's error bound in the kernel.
    fn probs(t: *Gate, r: usize, acc: c.Accum, order: g.Order, p: *[E]f64, err: *[E]f64) void {
        var terms: [H]f64 = undefined;
        var logit: [E]f64 = undefined;
        var worst: f64 = 0;
        for (&logit, 0..) |*l, e| {
            var s: f64 = 0;
            var m: f64 = 0;
            for (&terms, 0..) |*q, i| {
                q.* = @as(f64, c.fromBf16(t.x[r * H + i])) * c.fromBf16(t.w[e * H + i]);
                s += q.*;
                m += @abs(q.*);
            }
            if (acc != .none) s = contract.rows.sumIn(acc, order, &terms);
            l.* = s;
            if (t.fill != .stress and std.math.isFinite(m)) worst = @max(worst, c.gamma(@floatFromInt(depth(t.shape)), u) * m);
        }
        var top = -std.math.inf(f64);
        for (logit) |l| top = @max(top, l);
        var mass: f64 = 0;
        var far: f64 = 0;
        for (p, logit) |*q, l| {
            q.* = @exp(l - top);
            mass += q.*;
            if (std.math.isFinite(l - top)) far = @max(far, @abs(l - top));
        }
        if (acc != .none) mass = contract.rows.sumIn(acc, order, p);
        for (p) |*q| q.* /= mass;
        // logit errors move a probability by twice the worst; exp, the mass's chain, the division
        const rel = @exp(2 * worst + u * far) - 1 + 4 * u + c.gamma(E, u) + u;
        for (err, p) |*e, q| e.* = rel * q + 0x1p-148;
    }

    /// The 6 largest, ties to the lower expert; a row of NaN picks experts 0 to 5.
    fn pick(p: *const [E]f64, out: *[P]usize) void {
        var taken: [E]bool = @splat(false);
        for (out) |*o| {
            var best: usize = E;
            for (p, 0..) |q, e| {
                if (taken[e]) continue;
                if (best == E or rankOf(q) > rankOf(p[best])) best = e;
            }
            o.* = best;
            taken[best] = true;
        }
    }

    fn rankOf(q: f64) f64 {
        return if (std.math.isNan(q)) -1 else q;
    }

    fn outputs(t: *Gate, acc: c.Accum, order: g.Order, out: []f64) void {
        var p: [E]f64 = undefined;
        var err: [E]f64 = undefined;
        var ids: [P]usize = undefined;
        for (0..t.shape.rows) |r| {
            t.probs(r, acc, order, &p, &err);
            pick(&p, &ids);
            for (ids, 0..) |e, j| out[r * P + j] = p[e];
        }
    }

    pub fn reference(t: *Gate, want: []f64, mag: []f64) !void {
        t.outputs(.none, .sequential, want);
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }

    pub fn simulate(t: *Gate, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.outputs(acc, order, out);
    }

    pub fn bounds(t: *Gate, _: []const f64, _: []const f64, allow: []f64) !void {
        var p: [E]f64 = undefined;
        var err: [E]f64 = undefined;
        var ids: [P]usize = undefined;
        for (0..t.shape.rows) |r| {
            t.probs(r, .none, .sequential, &p, &err);
            pick(&p, &ids);
            for (ids, 0..) |e, j| allow[r * P + j] = err[e];
        }
    }

    fn say(t: *Gate, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&t.msg, fmt, args) catch "check failed (message too long)";
    }

    /// The kernel's picks and weights, row by row (gates 1 and 3).
    pub fn check(t: *Gate, _: g.Gate) !?[]const u8 {
        const wts = t.rig.bytes(t.wts, t.count() * 4);
        const ids = std.mem.bytesAsSlice(u32, t.rig.bytes(t.ids, t.count() * 4));
        var p: [E]f64 = undefined;
        var err: [E]f64 = undefined;
        for (0..t.shape.rows) |r| {
            t.probs(r, .none, .sequential, &p, &err);
            const sel: []const u32 = @alignCast(ids[r * P ..][0..P]);
            if (std.math.isNan(p[0])) {
                for (sel, 0..) |e, j| if (e != j or !std.math.isNan(c.read(.f32, wts, r * P + j))) return t.say("row {d}: a NaN row picks expert {d} at {d}, or a number", .{ r, e, j });
                continue;
            }
            for (sel, 0..) |e, j| {
                if (e >= E) return t.say("row {d} pick {d}: expert {d} out of range", .{ r, j, e });
                if (std.mem.indexOfScalar(u32, sel[0..j], e) != null) return t.say("row {d} pick {d}: expert {d} picked twice", .{ r, j, e });
                if (j > 0 and p[e] > p[sel[j - 1]] + err[e] + err[sel[j - 1]]) return t.say("row {d} pick {d}: expert {d} ranks above the pick before it", .{ r, j, e });
                const w = c.read(.f32, wts, r * P + j);
                if (!(@abs(w - p[e]) <= err[e])) return t.say("row {d} pick {d} (expert {d}): weight {e:.6} where float64 gives {e:.6}", .{ r, j, e, w, p[e] });
            }
            const last = sel[P - 1];
            for (0..E) |e| if (std.mem.indexOfScalar(u32, sel, @intCast(e)) == null and p[e] > p[last] + err[e] + err[last])
                return t.say("row {d}: expert {d} beats the last pick {d} but is not picked", .{ r, e, last });
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Gate) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Mutation{ .none, .gate_unnormalized }) |m| {
            const x = try s.a.create(Gate);
            x.* = try Gate.init(s.a, s.device, s.queue, m);
            const kern = x.kernel();
            if (!k.wanted(only, kern.spec.name)) {
                x.deinit();
                s.a.destroy(x);
                continue;
            }
            try s.probes.append(s.a, x);
            try s.kernels.append(s.a, kern);
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.probes.deinit(s.a);
    }
};

/// Gate 5's texts for this group's launch.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?k.Texts {
    return k.limitsText(name, &.{"ud_moe_gate"});
}
