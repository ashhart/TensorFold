//! Mini's loop gates under the contract: tf_ink2_loop_msg, tf_ink2_loop_down8/16/64 and tf_ink2_loop_blend.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");
const lp = @import("inkling2").loop_ops;

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const W = lp.width;
const R = 64; // gate_rank
const P = 2 * W; // the down projection's input
const sources = 2; // Mini's bank: the previous pass's blocks after its first, then its output
const split = 9;

fn launch(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// Served launches: decode rows, a 16-row window, prompt passes of 37, 128 and 2048 rows; 3 gates (q, k, v) or 1.
const served = [_]c.Shape{ .{ .rows = 1, .n = W, .k = 3 }, .{ .rows = 16, .n = W, .k = 1 }, .{ .rows = 37, .n = W, .k = 3 }, .{ .rows = 128, .n = W, .k = 1 }, .{ .rows = 2048, .n = W, .k = 1 } };

/// Each gate's weights (down [R][2W], forget and gain [W][R]), as the launch's Gates take them.
const Weights = struct {
    down: []u16 = &.{},
    forget: []u16 = &.{},
    gain: []u16 = &.{},
    dev: lp.Gates = .{ .down = @splat(0), .forget = @splat(0), .gain = @splat(0) },

    fn free(w: *Weights, a: std.mem.Allocator) void {
        inline for (.{ "down", "forget", "gain" }) |f| {
            a.free(@field(w, f));
            @field(w, f) = &.{};
        }
    }

    /// Weights in [-0.25, 0.25) for `gates` gates, uploaded; the fourth place repeats the last gate.
    fn make(w: *Weights, a: std.mem.Allocator, rig: *rig_.Rig, rng: std.Random, gates: usize) !void {
        w.free(a);
        w.down = try ref.bfs(a, rng, gates * R * P, -0.25, 0.25);
        w.forget = try ref.bfs(a, rng, gates * W * R, -0.25, 0.25);
        w.gain = try ref.bfs(a, rng, gates * W * R, -0.25, 0.25);
        try w.upload(rig, gates);
    }

    fn upload(w: *Weights, rig: *rig_.Rig, gates: usize) !void {
        for (0..4) |i| {
            const j = @min(i, gates - 1);
            if (i < gates) {
                w.dev.down[i] = try ref.input(rig, w.down[j * R * P ..][0 .. R * P]);
                w.dev.forget[i] = try ref.input(rig, w.forget[j * W * R ..][0 .. W * R]);
                w.dev.gain[i] = try ref.input(rig, w.gain[j * W * R ..][0 .. W * R]);
            } else {
                w.dev.down[i] = w.dev.down[j];
                w.dev.forget[i] = w.dev.forget[j];
                w.dev.gain[i] = w.dev.gain[j];
            }
        }
    }
};

/// A thread's 9 values in order, its warp's 5 shuffles, the 8 warp sums in order.
fn msgDepth(_: c.Shape) usize {
    return 9 + 5 + 8;
}

/// The message kernel's two outputs, one registration each: msg (fp32) and the down input pair (bf16).
pub const View = enum { msg, pair };

pub fn msgSpec(v: View, wrong: bool) c.Spec {
    const names = [2][]const u8{ "inkling2 loop tf_ink2_loop_msg messages (attention over the bank, fp32)", "inkling2 loop tf_ink2_loop_msg down inputs ([norm(h); norm(msg)], bf16)" };
    if (wrong) return .{ .name = "control: inkling2 loop tf_ink2_loop_msg told one bank source", .accum = .f32, .depth = msgDepth, .out = .f32, .served = served[0..1] };
    return .{ .name = names[@backingInt(v)], .accum = .f32, .depth = msgDepth, .out = if (v == .msg) .f32 else .bf16, .served = &served };
}

pub const Msg = struct {
    kern: *const lp.Kernels,
    view: View,
    wrong: bool = false,
    shape: c.Shape = undefined,
    bank: []f32 = &.{},
    q: []f32 = &.{},
    h: []f32 = &.{}, // [gates][rows][W] (one gate: [rows][W])
    w: Weights = .{},
    in: [3]u64 = @splat(0),
    msg: u64 = 0,
    pair: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Msg) c.Spec {
        return msgSpec(self.view, self.wrong);
    }

    pub fn control(self: *const Msg) ?g.Gate {
        return if (self.wrong) .shapes else null;
    }

    pub fn free(self: *Msg, a: std.mem.Allocator) void {
        a.free(self.bank);
        a.free(self.q);
        a.free(self.h);
        self.bank = &.{};
        self.q = &.{};
        self.h = &.{};
        self.w.free(a);
    }

    /// States in [-3, 3) with one in 41 eight times larger, queries in [-1, 1); stress: integer states.
    pub fn prepare(self: *Msg, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != W or shape.k < 1 or shape.k > 3) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const stress = fill == .stress;
        self.bank = try a.alloc(f32, rows * sources * W);
        self.h = try a.alloc(f32, shape.k * rows * W);
        for ([_][]f32{ self.bank, self.h }) |xs| for (xs, 0..) |*v, i| {
            v.* = if (stress) ref.int(rng, -48, 48) else (rng.float(f32) * 6 - 3) * (if (i % 41 == 7) @as(f32, 8) else 1);
        };
        // stress: zero queries, so the softmax is exactly even and gate 4 sees the norms alone
        self.q = try a.alloc(f32, shape.k * W);
        for (self.q) |*v| v.* = if (stress) 0 else rng.float(f32) * 2 - 1;
        if (fill == .nonfinite) {
            ref.plant(f32, self.bank, @min(rows * sources, 4), W, ref.asF32);
            if (rows > 1) ref.plant(f32, self.h[W..], @min(rows - 1, 3), W, ref.asF32);
        }
        try self.w.make(a, rig, rng, shape.k);
        self.in = .{ try ref.input(rig, self.bank), try ref.input(rig, self.q), try ref.input(rig, self.h) };
        const n = rows * shape.k * W;
        if (self.view == .msg or self.wrong) {
            self.msg = try rig.output(n * 4);
            self.pair = try rig.scratch(n * 4);
        } else {
            self.msg = try rig.scratch(n * 4);
            self.pair = try rig.output(n * 4);
        }
        return true;
    }

    pub fn run(self: *Msg, rig: *rig_.Rig) !void {
        const rows = self.shape.rows;
        const gates = self.shape.k;
        var a: cuda.Args = .{};
        a.add(self.in[0]);
        a.add(@as(i32, if (self.wrong) sources - 1 else sources));
        a.add(self.in[1]);
        a.add(self.in[2]);
        a.add(@as(i64, W));
        a.add(@as(i64, if (gates == 1) 0 else @as(i64, @intCast(rows * W))));
        a.add(@as(i32, @intCast(gates)));
        a.add(@as(f32, @floatCast(std.math.pow(f64, W, -0.5))));
        a.add(ref.eps);
        a.add(self.w.dev);
        a.add(self.msg);
        a.add(self.pair);
        try launch(self.kern.msg, rig.s, .{ .x = @intCast(rows * gates) }, 256, &a);
    }

    pub fn count(self: *const Msg) usize {
        return self.shape.rows * self.shape.k * @as(usize, if (self.view == .pair and !self.wrong) P else W);
    }

    /// Per (row, gate): RMS-normed bank states as keys and values, a softmax of q . k / sqrt(W), the mix, both norms.
    pub fn solve(self: *const Msg, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const rows = self.shape.rows;
        const gates = self.shape.k;
        const terms = try a.alloc(Iv, W);
        defer a.free(terms);
        const scratch = try a.alloc(f64, W);
        defer a.free(scratch);
        const e = try a.alloc(Iv, sources * W);
        defer a.free(e);
        const mv = try a.alloc(Iv, W);
        defer a.free(mv);
        const d = msgDepth(self.shape);
        const scale = Iv.of(@as(f32, @floatCast(std.math.pow(f64, W, -0.5))));
        const st = ref.later(s);
        for (0..rows) |r| for (0..gates) |gi| {
            const hv = self.h[(gi * rows + r) * W ..][0..W];
            const qv = self.q[gi * W ..][0..W];
            for (terms, hv) |*t, v| t.* = Iv.of(v).mul(Iv.of(v));
            const rh = ref.rms(s.sum(terms, d, scratch), W, ref.eps);
            var p: [sources]Iv = undefined;
            for (0..sources) |src| {
                const b = self.bank[(r * sources + src) * W ..][0..W];
                for (terms, b) |*t, v| t.* = Iv.of(v).mul(Iv.of(v));
                const rs = ref.rms(s.sum(terms, d, scratch), W, ref.eps);
                for (e[src * W ..][0..W], b) |*x, v| x.* = Iv.of(v).mul(rs);
                for (terms, qv, e[src * W ..][0..W]) |*t, q, x| t.* = Iv.of(q).mul(x);
                p[src] = st.sum(terms, d, scratch).mul(scale);
            }
            var m = p[0];
            for (p[1..]) |v| m = .{ .m = @max(m.m, v.m), .lo = @max(m.lo, v.lo), .hi = @max(m.hi, v.hi) };
            var den = Iv.of(0);
            for (&p) |*v| {
                v.* = v.sub(m).exp();
                den = den.add(v.*);
            }
            for (&p) |*v| v.* = v.div(den);
            for (mv, 0..) |*x, j| {
                var acc = Iv.of(0);
                for (0..sources) |src| acc = acc.add(p[src].mul(e[src * W + j]));
                x.* = acc;
            }
            const o = r * gates + gi;
            if (self.view == .msg or self.wrong) {
                @memcpy(out[o * W ..][0..W], mv);
                continue;
            }
            for (terms, mv) |*t, x| t.* = ref.sq(x).mul(Iv.of(1));
            const rm = ref.rms(st.sum(terms, d, scratch), W, ref.eps);
            for (out[o * P ..][0..W], hv) |*y, v| y.* = Iv.of(v).mul(rh).bf();
            for (out[o * P + W ..][0..W], mv) |*y, x| y.* = x.mul(rm).bf();
        };
    }
};

/// A warp's 8 mma k-steps of 16 (a tensor-core add may truncate: twice its chain), the 4 quarters, the 9 slices.
fn downDepth(_: c.Shape) usize {
    return 2 * 128 + 3 + split;
}

/// Outputs a block: down8 (8), down16 (16), down64 (64); loop_ops.zig picks by rows, none moving a bit.
pub const Width = enum { o8, o16, o64 };

pub fn downSpec(o: Width, swapped: bool) c.Spec {
    const names = [3][]const u8{ "inkling2 loop tf_ink2_loop_down8 (g = down . pair, fixed K cut, 8 outputs a block)", "inkling2 loop tf_ink2_loop_down16 (g = down . pair, fixed K cut, 16 outputs a block)", "inkling2 loop tf_ink2_loop_down64 (g = down . pair, fixed K cut, 64 outputs a block)" };
    const shapes = [3][]const c.Shape{ &.{ .{ .rows = 1, .n = R, .k = 3 }, .{ .rows = 16, .n = R, .k = 1 }, .{ .rows = 9, .n = R, .k = 3 } }, &.{ .{ .rows = 17, .n = R, .k = 3 }, .{ .rows = 128, .n = R, .k = 1 }, .{ .rows = 37, .n = R, .k = 3 } }, &.{ .{ .rows = 129, .n = R, .k = 3 }, .{ .rows = 2048, .n = R, .k = 1 } } };
    if (swapped) return .{ .name = "control: inkling2 loop tf_ink2_loop_down8 with gate 0's weights in gate 1's place", .accum = .f32, .depth = downDepth, .out = .bf16, .served = shapes[0][0..1] };
    return .{ .name = names[@backingInt(o)], .accum = .f32, .depth = downDepth, .out = .bf16, .served = shapes[@backingInt(o)] };
}

pub const Down = struct {
    kern: *const lp.Kernels,
    width: Width,
    swapped: bool = false,
    shape: c.Shape = undefined,
    pair: []u16 = &.{},
    w: Weights = .{},
    in: [2]u64 = @splat(0), // pair, the zeroed counts
    part: u64 = 0,
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Down) c.Spec {
        return downSpec(self.width, self.swapped);
    }

    pub fn control(self: *const Down) ?g.Gate {
        return if (self.swapped) .shapes else null;
    }

    pub fn free(self: *Down, a: std.mem.Allocator) void {
        a.free(self.pair);
        self.pair = &.{};
        self.w.free(a);
    }

    /// Inputs in [-2, 2), weights in [-0.25, 0.25); stress: integers (inputs to 48, weights to 8), exact in fp32.
    pub fn prepare(self: *Down, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != R or shape.k < 1 or shape.k > 3) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        self.pair = try ref.bfs(a, rng, rows * shape.k * P, -2, 2);
        try self.w.make(a, rig, rng, shape.k);
        if (fill == .stress) {
            for (self.pair) |*v| v.* = c.toBf16(ref.int(rng, -48, 48));
            for (self.w.down) |*v| v.* = c.toBf16(ref.int(rng, -8, 8));
            try self.w.upload(rig, shape.k);
        }
        if (fill == .nonfinite) ref.plant(u16, self.pair, @min(rows * shape.k, 4), P, ref.asBf);
        const counts = try a.alloc(u32, (rows + 15) / 16 * shape.k * 8);
        defer a.free(counts);
        @memset(counts, 0);
        self.in = .{ try ref.input(rig, self.pair), try ref.input(rig, counts) };
        self.part = try rig.scratch(rows * shape.k * split * R * 4);
        self.out = try rig.output(rows * shape.k * R * 2);
        return true;
    }

    pub fn run(self: *Down, rig: *rig_.Rig) !void {
        const rows = self.shape.rows;
        const gates = self.shape.k;
        var w = self.w.dev;
        if (self.swapped) w.down[1] = w.down[0];
        const outs: usize = switch (self.width) {
            .o8 => 8,
            .o16 => 16,
            .o64 => 64,
        };
        var a: cuda.Args = .{};
        a.add(self.in[0]);
        a.add(w);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(gates)));
        a.add(self.part);
        a.add(self.in[1]);
        a.add(self.out);
        const f = switch (self.width) {
            .o8 => self.kern.down8,
            .o16 => self.kern.down16,
            .o64 => self.kern.down64,
        };
        try launch(f, rig.s, .{ .x = @intCast((rows + 15) / 16), .y = split, .z = @intCast(gates * (R / outs)) }, 128, &a);
    }

    pub fn count(self: *const Down) usize {
        return self.shape.rows * self.shape.k * R;
    }

    /// g[row][gate][n] = bf16(sum over 2W of pair * down[gate][n]), bf16 products exact in fp32.
    pub fn solve(self: *const Down, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const gates = self.shape.k;
        const terms = try a.alloc(Iv, P);
        defer a.free(terms);
        const scratch = try a.alloc(f64, P);
        defer a.free(scratch);
        for (0..self.shape.rows) |r| for (0..gates) |gi| for (0..R) |n| {
            const x = self.pair[(r * gates + gi) * P ..][0..P];
            const w = self.w.down[(gi * R + n) * P ..][0..P];
            for (terms, x, w) |*t, u, v| t.* = Iv.of(ref.bf(u) * ref.bf(v));
            out[(r * gates + gi) * R + n] = s.sum(terms, downDepth(self.shape), scratch).bf();
        };
    }
};

/// A lane's 8 index-ordered products, the 3 xor shuffles over an output's 8 lanes.
fn blendDepth(_: c.Shape) usize {
    return 8 + 3;
}

pub fn blendSpec(swapped: bool) c.Spec {
    if (swapped) return .{ .name = "control: inkling2 loop tf_ink2_loop_blend with forget and gain swapped", .accum = .f32, .depth = blendDepth, .ops = 6, .out = .f32, .served = served[0..1] };
    return .{ .name = "inkling2 loop tf_ink2_loop_blend ((1 - tanh(F g)) h + tanh(M g) msg)", .accum = .f32, .depth = blendDepth, .ops = 6, .out = .f32, .served = &served };
}

pub const Blend = struct {
    kern: *const lp.Kernels,
    swapped: bool = false,
    shape: c.Shape = undefined,
    h: []f32 = &.{},
    msg: []f32 = &.{},
    gv: []u16 = &.{},
    w: Weights = .{},
    in: [3]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Blend) c.Spec {
        return blendSpec(self.swapped);
    }

    pub fn control(self: *const Blend) ?g.Gate {
        return if (self.swapped) .shapes else null;
    }

    pub fn free(self: *Blend, a: std.mem.Allocator) void {
        a.free(self.h);
        a.free(self.msg);
        a.free(self.gv);
        self.h = &.{};
        self.msg = &.{};
        self.gv = &.{};
        self.w.free(a);
    }

    /// h, msg in [-3, 3), g in [-2, 2); stress: g and weights mirrored a step apart (dots cancel through big sums).
    pub fn prepare(self: *Blend, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != W or shape.k < 1 or shape.k > 3) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const n = rows * shape.k * W;
        self.h = try ref.f32s(a, rng, n, -3, 3);
        self.msg = try ref.f32s(a, rng, n, -3, 3);
        self.gv = try ref.bfs(a, rng, rows * shape.k * R, -2, 2);
        try self.w.make(a, rig, rng, shape.k);
        if (fill == .stress) {
            for (0..rows * shape.k) |o| for (0..R / 2) |k| {
                const v = ref.int(rng, 1, 64);
                self.gv[o * R + k] = c.toBf16(v);
                self.gv[o * R + R / 2 + (k + 1) % (R / 2)] = c.toBf16(-v + @as(f32, if (k == 0) 1 else 0));
            };
            for ([_][]u16{ self.w.forget, self.w.gain }) |ws| for (0..shape.k * W) |col| for (0..R / 2) |k| {
                const x = if (k == 0) ref.int(rng, 1, 2) else ref.int(rng, 1, 64);
                ws[col * R + k] = c.toBf16(x);
                ws[col * R + R / 2 + (k + 1) % (R / 2)] = c.toBf16(x);
            };
            try self.w.upload(rig, shape.k);
        }
        if (fill == .nonfinite) {
            ref.plant(f32, self.h, @min(rows * shape.k, 4), W, ref.asF32);
            if (rows * shape.k > 4) ref.plant(f32, self.msg[4 * W ..], @min(rows * shape.k - 4, 4), W, ref.asF32);
        }
        self.in = .{ try ref.input(rig, self.h), try ref.input(rig, self.msg), try ref.input(rig, self.gv) };
        self.out = try rig.output(n * 4);
        return true;
    }

    pub fn run(self: *Blend, rig: *rig_.Rig) !void {
        const rows = self.shape.rows;
        const gates = self.shape.k;
        var w = self.w.dev;
        if (self.swapped) std.mem.swap([4]u64, &w.forget, &w.gain);
        const per: usize = if (rows <= 16) 16 else 32;
        var a: cuda.Args = .{};
        a.add(self.in[0]);
        a.add(@as(i64, W));
        a.add(@as(i64, if (gates == 1) 0 else @as(i64, @intCast(rows * W))));
        a.add(self.in[1]);
        a.add(self.in[2]);
        a.add(w);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(gates)));
        a.add(@as(i32, @intCast(per)));
        a.add(self.out);
        try launch(self.kern.blend, rig.s, .{ .x = W / 32, .y = @intCast(gates), .z = @intCast((rows + per - 1) / per) }, 256, &a);
    }

    pub fn count(self: *const Blend) usize {
        return self.shape.rows * self.shape.k * W;
    }

    /// out = (1 - tanh(F g)) h + tanh(M g) msg; h [gates][rows][W], msg and out [rows][gates][W].
    pub fn solve(self: *const Blend, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const rows = self.shape.rows;
        const gates = self.shape.k;
        var tf: [R]Iv = undefined;
        var tm: [R]Iv = undefined;
        var scratch: [R]f64 = undefined;
        for (0..rows) |r| for (0..gates) |gi| for (0..W) |col| {
            const gr = self.gv[(r * gates + gi) * R ..][0..R];
            const fw = self.w.forget[(gi * W + col) * R ..][0..R];
            const mw = self.w.gain[(gi * W + col) * R ..][0..R];
            for (&tf, &tm, gr, fw, mw) |*a, *b, x, f, m| {
                a.* = Iv.of(ref.bf(f) * ref.bf(x));
                b.* = Iv.of(ref.bf(m) * ref.bf(x));
            }
            const fd = s.sum(&tf, blendDepth(self.shape), &scratch);
            const md = s.sum(&tm, blendDepth(self.shape), &scratch);
            const o = r * gates + gi;
            const hv = Iv.of(self.h[(gi * rows + r) * W + col]);
            const keep = Iv.of(1).sub(ref.tanh(fd));
            out[o * W + col] = keep.mul(hv).add(ref.tanh(md).mul(Iv.of(self.msg[o * W + col])));
        };
    }
};
