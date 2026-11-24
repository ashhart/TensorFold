//! Mini loop math under the contract, with independent host arithmetic and planted single-fault controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const lm = ink.loop_math_cuda;
pub const eps = ref.eps;
pub const mini_h = 2304;

/// Exact raw-output checks, with interval references and accumulator simulations for gate 4.
pub fn Checked(comptime T: type) type {
    return struct {
        const Self = @This();
        inner: ref.Gated(T),
        rig: *rig_.Rig = undefined,
        msg: [200]u8 = undefined,

        pub fn deinit(self: *Self) void {
            self.inner.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            self.rig = &self.inner.rig;
            return .{ .spec = self.inner.k.spec(), .ctx = self, .vt = &vt, .control = self.inner.k.control() };
        }

        const vt = g.hooks(Self);

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            return self.inner.prepare(shape, fill, seed);
        }

        pub fn run(self: *Self) !void {
            try self.inner.run();
        }

        pub fn count(self: *Self) usize {
            return self.inner.count();
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            try self.inner.reference(want, mag);
        }

        pub fn bounds(self: *Self, want: []const f64, mag: []const f64, allow: []f64) !void {
            try self.inner.bounds(want, mag, allow);
        }

        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            try self.inner.simulate(acc, order, out);
        }

        pub fn replaced(self: *Self) ![]const u8 {
            return self.inner.replaced();
        }

        pub fn check(self: *Self, _: g.Gate) !?[]const u8 {
            const a = self.rig.gpu.gpa;
            const got = try a.dupe(u8, try self.rig.outputBytes());
            defer a.free(got);
            return self.inner.k.check(a, got, &self.msg);
        }
    };
}

pub fn queryScale(cols: usize) ref.Iv {
    return if (cols == mini_h) ref.Iv.of(scale(cols)) else ref.Iv.of(@floatFromInt(cols)).rsqrt();
}

pub fn softmax(logits: []ref.Iv) void {
    var m = logits[0];
    for (logits[1..]) |v| m = .{ .m = @max(m.m, v.m), .lo = @max(m.lo, v.lo), .hi = @max(m.hi, v.hi) };
    var den = ref.Iv.of(0);
    for (logits) |*v| {
        v.* = v.sub(m).exp();
        den = den.add(v.*);
    }
    for (logits) |*v| v.* = v.div(den);
}

/// Irregular integer squares stay exact in FP32 for gate 4, but overflow FP16 and round in every BF16 order.
fn stressBank(i: usize) f32 {
    return if ((i * 37 + i / 7) % 5 == 0) 384 else 256;
}

/// Messages nearly cancel two normalized banks; the unmatched first tap keeps FP16 normalization overflow visible.
fn messageStressBank(bank: usize, col: usize, cols: usize) f32 {
    if (bank == 0) return stressBank(col);
    if (col == 0) return 0;
    return -stressBank((col + 1) % cols);
}

/// AR cyclically assigns [x, -x, 0] over its sources so zero-query outputs expose RMS-sum differences.
pub fn arStressSource(source: usize, col: usize) f32 {
    if (source == 0 and col == 0) return 0;
    const value = stressBank(col - col % 3);
    return switch ((col + 2 * source) % 3) {
        0 => value,
        1 => -value,
        2 => 0,
        else => unreachable,
    };
}

pub fn close(got: f32, want: f32) bool {
    if (std.math.isNan(want)) return std.math.isNan(got);
    if (!std.math.isFinite(want)) return got == want;
    if (!std.math.isFinite(got)) return false;
    const tolerance = 1e-4 * (1.0 + @abs(want));
    return @abs(got - want) <= tolerance;
}

pub fn bf(x: u16) f32 {
    return @floatCast(ref.bf(x));
}

pub fn bfRound(x: f32) f32 {
    return bf(c.toBf16(x));
}
pub fn cudaMax(a: f32, b: f32) f32 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @max(a, b);
}

pub fn sameBf(got: u16, want: u16) bool {
    if ((want & 0x7fff) > 0x7f80) return (got & 0x7fff) > 0x7f80; // NaN payload is not part of the math rule
    return got == want;
}

pub fn scale(cols: usize) f32 {
    return if (cols == mini_h) 1.0 / 48.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(cols)));
}

fn warp(v: *[32]f32) f32 {
    var offset: usize = 16;
    while (offset != 0) : (offset >>= 1) {
        const old = v.*;
        for (0..32) |lane| v.*[lane] = old[lane] + old[lane ^ offset];
    }
    return v.*[0];
}

/// The exact lane chains and butterfly tree of loop_math.cu. Terms are fused into each lane's accumulator.
pub fn reduce(a: []const f32, b: []const f32, threads: usize) f32 {
    var partial: [32]f32 = @splat(0);
    for (0..threads / 32) |w| {
        var lanes: [32]f32 = undefined;
        for (0..32) |lane| {
            const tid = w * 32 + lane;
            var sum: f32 = 0;
            var i = tid;
            while (i < a.len) : (i += threads) sum = @mulAdd(f32, a[i], b[i], sum);
            lanes[lane] = sum;
        }
        partial[w] = warp(&lanes);
    }
    return warp(&partial);
}

pub fn fail(msg: []u8, comptime fmt: []const u8, args: anytype) !?[]const u8 {
    return try std.fmt.bufPrint(msg, fmt, args);
}

const message_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 4 * 12 * mini_h, .k = mini_h },
    .{ .rows = 16, .n = 4 * 128, .k = 128 },
    .{ .rows = 2048, .n = 4 * 128, .k = 128 },
};

/// Two normalized banks and the checkpoint's even/odd q,mlp and k,v queries, compared element by element.
pub const Messages = struct {
    kern: *const lm.Kernels,
    aliased: bool = false,
    shape: c.Shape = undefined,
    banks: [2][]u16 = .{ &.{}, &.{} },
    queries: [2][]f32 = .{ &.{}, &.{} },
    dev: [5]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const Messages) c.Spec {
        return .{ .name = if (self.aliased) "control: inkling2 loop messages with both previous-pass inputs wired to bank zero" else "inkling2 loop tf_ink2_loop_messages (two normalized banks, post-dot 1/48, four roles)", .accum = .f32, .depth = depth, .out = .f32, .served = if (self.aliased) message_shapes[0..2] else &message_shapes };
    }

    fn depth(s: c.Shape) usize {
        return 2 * ((s.k + 255) / 256 + 10);
    }

    pub fn control(self: *const Messages) ?g.Gate {
        return if (self.aliased) .shapes else null;
    }

    pub fn free(self: *Messages, a: std.mem.Allocator) void {
        for (&self.banks) |*x| a.free(x.*);
        for (&self.queries) |*x| a.free(x.*);
        self.banks = .{ &.{}, &.{} };
        self.queries = .{ &.{}, &.{} };
    }

    pub fn prepare(self: *Messages, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n % (4 * shape.k) != 0) return false;
        self.free(a);
        self.shape = shape;
        const cols = shape.k;
        const layers = shape.n / (4 * cols);
        for (&self.banks, 0..) |*x, bank| {
            x.* = try a.alloc(u16, shape.rows * cols);
            if (fill == .stress) {
                for (x.*, 0..) |*v, i| v.* = c.toBf16(messageStressBank(bank, i % cols, cols));
            } else for (x.*) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        }
        for (&self.queries) |*x| {
            x.* = try ref.f32s(a, rng, 2 * layers * cols, -0.25, 0.25);
            if (fill == .stress) @memset(x.*, 0);
        }
        if (fill == .nonfinite) {
            ref.plant(u16, self.banks[0], shape.rows, cols, ref.asBf);
            ref.plant(u16, self.banks[1], shape.rows, cols, ref.asBf);
        }
        for (self.banks, 0..) |x, i| self.dev[i] = try ref.input(rig, x);
        for (self.queries, 0..) |x, i| self.dev[2 + i] = try ref.input(rig, x);
        self.out = try rig.output(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(self: *Messages, rig: *rig_.Rig) !void {
        const b1 = if (self.aliased) self.dev[0] else self.dev[1];
        const layers = self.shape.n / (4 * self.shape.k);
        try lm.messages(self.kern, rig.s, self.dev[0], b1, self.dev[2], self.dev[3], self.out, self.shape.rows, layers, self.shape.k, eps);
    }

    pub fn count(self: *const Messages) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Messages, a: std.mem.Allocator, s: ref.Sums, out: []ref.Iv) !void {
        const cols = self.shape.k;
        const layers = self.shape.n / (4 * cols);
        const terms = try a.alloc(ref.Iv, cols);
        defer a.free(terms);
        const scratch = try a.alloc(f64, cols);
        defer a.free(scratch);
        const later = ref.later(s);
        for (0..self.shape.rows) |row| {
            var inv: [2]ref.Iv = undefined;
            for (0..2) |b| {
                for (terms, self.banks[b][row * cols ..][0..cols]) |*t, v| {
                    const x = ref.Iv.of(ref.bf(v));
                    t.* = x.mul(x);
                }
                inv[b] = ref.rms(s.sum(terms, depth(self.shape), scratch), cols, eps);
            }
            for (0..layers) |j| for (0..4) |role| {
                const query = self.queries[if (role == 0 or role == 3) 0 else 1][(2 * j + @intFromBool(role >= 2)) * cols ..][0..cols];
                var logits: [2]ref.Iv = undefined;
                for (0..2) |b| {
                    for (terms, query, self.banks[b][row * cols ..][0..cols]) |*t, q, v| t.* = ref.Iv.of(q).mul(ref.Iv.of(ref.bf(v)).mul(inv[b]));
                    logits[b] = later.sum(terms, depth(self.shape), scratch).mul(queryScale(cols));
                }
                softmax(&logits);
                for (0..cols) |i| {
                    var value = ref.Iv.of(0);
                    for (0..2) |b| {
                        const beta = ref.Iv.of(ref.bf(self.banks[b][row * cols + i])).mul(inv[b]);
                        value = value.add(logits[b].mul(beta));
                    }
                    const at = ((j * 4 + role) * self.shape.rows + row) * cols + i;
                    out[at] = value;
                }
            };
        }
    }

    pub fn check(self: *const Messages, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const cols = self.shape.k;
        const layers = self.shape.n / (4 * cols);
        const raw0 = try a.alloc(f32, cols);
        defer a.free(raw0);
        const raw1 = try a.alloc(f32, cols);
        defer a.free(raw1);
        const beta0 = try a.alloc(f32, cols);
        defer a.free(beta0);
        const beta1 = try a.alloc(f32, cols);
        defer a.free(beta1);
        for (0..self.shape.rows) |row| {
            for (raw0, self.banks[0][row * cols ..][0..cols]) |*x, v| x.* = bf(v);
            for (raw1, self.banks[1][row * cols ..][0..cols]) |*x, v| x.* = bf(v);
            const inv0 = 1.0 / @sqrt(reduce(raw0, raw0, 256) / @as(f32, @floatFromInt(cols)) + eps);
            const inv1 = 1.0 / @sqrt(reduce(raw1, raw1, 256) / @as(f32, @floatFromInt(cols)) + eps);
            for (beta0, raw0) |*x, v| x.* = v * inv0;
            for (beta1, raw1) |*x, v| x.* = v * inv1;
            for (0..layers) |j| for (0..4) |role| {
                const query = self.queries[if (role == 0 or role == 3) 0 else 1][(2 * j + @intFromBool(role >= 2)) * cols ..][0..cols];
                const l0 = reduce(query, beta0, 256) * scale(cols);
                const l1 = reduce(query, beta1, 256) * scale(cols);
                const m = cudaMax(l0, l1);
                const e0 = @exp(l0 - m);
                const e1 = @exp(l1 - m);
                const den = e0 + e1;
                const w0 = e0 / den;
                const w1 = e1 / den;
                for (0..cols) |i| {
                    const want = w0 * beta0[i] + w1 * beta1[i];
                    const at = ((j * 4 + role) * self.shape.rows + row) * cols + i;
                    const value = ref.readF32(got, at);
                    if (!close(value, want)) return fail(msg, "row {d} core {d} role {d} col {d}: {e} where {e} is due", .{ row, j, role, i, value, want });
                }
            };
        }
        return null;
    }
};

const gate_shapes = [_]c.Shape{
    .{ .rows = 1, .n = mini_h, .k = 2 * mini_h },
    .{ .rows = 16, .n = mini_h, .k = 2 * mini_h },
    .{ .rows = 2048, .n = 128, .k = 256 },
};

/// The served gate, or a control: forget weights again for gain (up half), or zeta from half the rank pairs (down).
pub const GateFault = enum { none, wrong_gain, half_ranks };

/// Host reference preserves BF16 n/zeta and all four FP32 output steps through both halves' launches.
pub const Gate = struct {
    kern: *const lm.Kernels,
    fault: GateFault = .none,
    shape: c.Shape = undefined,
    h: []f32 = &.{},
    m: []f32 = &.{},
    d: []u16 = &.{},
    f: []u16 = &.{},
    gain: []u16 = &.{},
    dev: [5]u64 = @splat(0),
    zeta: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn spec(self: *const Gate) c.Spec {
        const name = switch (self.fault) {
            .none => "inkling2 loop tf_ink2_loop_gate_down + _up (BF16 down input/zeta, FP32 F/G tanh and four output steps)",
            .wrong_gain => "control: inkling2 loop rank-64 gate up half with forget weights again instead of gain",
            .half_ranks => "control: inkling2 loop rank-64 gate down half launched for 16 of its 32 rank pairs",
        };
        const served = if (self.fault != .none) gate_shapes[0..2] else &gate_shapes;
        return .{ .name = name, .accum = .f32, .depth = depth, .out = .f32, .served = served, .replaces = "tf_ink2_loop_gate" };
    }

    fn depth(s: c.Shape) usize {
        return (s.k + 255) / 256 + 10 + lm.rank;
    }

    pub fn control(self: *const Gate) ?g.Gate {
        return if (self.fault != .none) .shapes else null;
    }

    pub fn free(self: *Gate, a: std.mem.Allocator) void {
        a.free(self.h);
        a.free(self.m);
        a.free(self.d);
        a.free(self.f);
        a.free(self.gain);
        self.h = &.{};
        self.m = &.{};
        self.d = &.{};
        self.f = &.{};
        self.gain = &.{};
    }

    pub fn prepare(self: *Gate, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != 2 * shape.n) return false;
        self.free(a);
        self.shape = shape;
        const cols = shape.n;
        self.h = try ref.f32s(a, rng, shape.rows * cols, -1, 1);
        self.m = try ref.f32s(a, rng, shape.rows * cols, -1, 1);
        self.d = try ref.bfs(a, rng, lm.rank * 2 * cols, -0.1, 0.1);
        self.f = try ref.bfs(a, rng, cols * lm.rank, -0.2, 0.2);
        self.gain = try ref.bfs(a, rng, cols * lm.rank, -0.2, 0.2);
        if (fill == .stress) {
            for (self.h, 0..) |*v, i| v.* = stressBank(i % cols);
            for (self.m, 0..) |*v, i| v.* = stressBank(i % cols);
            for (self.d) |*v| v.* = c.toBf16(ref.int(rng, -8, 8));
            for (self.f) |*v| v.* = c.toBf16(ref.int(rng, -8, 8));
            for (self.gain) |*v| v.* = c.toBf16(ref.int(rng, -8, 8));
        }
        if (fill == .nonfinite) {
            ref.plant(f32, self.h, shape.rows, cols, ref.asF32);
            ref.plant(f32, self.m, shape.rows, cols, ref.asF32);
        }
        self.dev = .{ try ref.input(rig, self.h), try ref.input(rig, self.m), try ref.input(rig, self.d), try ref.input(rig, self.f), try ref.input(rig, self.gain) };
        self.zeta = try rig.scratch(shape.rows * lm.rank * 2);
        self.base = try rig.scratch(shape.rows * cols * 4);
        self.out = try rig.output(shape.rows * cols * 4);
        return true;
    }

    pub fn run(self: *Gate, rig: *rig_.Rig) !void {
        const d = self.dev;
        const rows = self.shape.rows;
        const pairs: usize = if (self.fault == .half_ranks) lm.rank / 4 else lm.rank / 2;
        try lm.gateDown(self.kern, rig.s, d[0], d[1], d[2], self.zeta, rows, pairs, self.shape.n, eps);
        try lm.gateUp(self.kern, rig.s, d[0], d[1], self.zeta, d[3], if (self.fault == .wrong_gain) d[3] else d[4], self.out, rows, self.shape.n);
    }

    /// The one-block tf_ink2_loop_gate the served halves replace, on the same inputs: gate 3 wants its bytes.
    pub fn replaced(self: *Gate, rig: *rig_.Rig) ![]const u8 {
        const f = try self.kern.mod.function("tf_ink2_loop_gate");
        var a: cuda.Args = .{};
        for ([_]u64{ self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.dev[4], self.base }) |v| a.add(v);
        for ([_]i32{ @intCast(self.shape.rows), @intCast(self.shape.n) }) |v| a.add(v);
        a.add(@as(f32, eps));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 256 } }, rig.s, &a);
        return rig.bytes(self.base, self.shape.rows * self.shape.n * 4);
    }

    pub fn count(self: *const Gate) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Gate, a: std.mem.Allocator, s: ref.Sums, out: []ref.Iv) !void {
        const cols = self.shape.n;
        const n = try a.alloc(ref.Iv, 2 * cols);
        defer a.free(n);
        const terms = try a.alloc(ref.Iv, 2 * cols);
        defer a.free(terms);
        const scratch = try a.alloc(f64, 2 * cols);
        defer a.free(scratch);
        const zeta = try a.alloc(ref.Iv, lm.rank);
        defer a.free(zeta);
        const later = ref.later(s);
        const zeta_depth = (2 * cols + 31) / 32 + 5;
        for (0..self.shape.rows) |row| {
            const h = self.h[row * cols ..][0..cols];
            const m = self.m[row * cols ..][0..cols];
            var inv: [2]ref.Iv = undefined;
            for (0..2) |b| {
                const x = if (b == 0) h else m;
                for (terms[0..cols], x) |*t, v| {
                    const e = ref.Iv.of(v);
                    t.* = e.mul(e);
                }
                inv[b] = ref.rms(s.sum(terms[0..cols], depth(self.shape), scratch), cols, eps);
            }
            for (n[0..cols], h) |*v, x| v.* = ref.Iv.of(x).mul(inv[0]).bf();
            for (n[cols..], m) |*v, x| v.* = ref.Iv.of(x).mul(inv[1]).bf();
            for (0..lm.rank) |r| {
                for (terms, n, self.d[r * 2 * cols ..][0 .. 2 * cols]) |*t, x, w| t.* = ref.Iv.of(ref.bf(w)).mul(x);
                zeta[r] = later.sum(terms, zeta_depth, scratch).bf();
            }
            for (0..cols) |i| {
                for (terms[0..lm.rank], zeta, self.f[i * lm.rank ..][0..lm.rank]) |*t, z, w| t.* = ref.Iv.of(ref.bf(w)).mul(z);
                const forget = ref.tanh(later.sum(terms[0..lm.rank], lm.rank, scratch));
                for (terms[0..lm.rank], zeta, self.gain[i * lm.rank ..][0..lm.rank]) |*t, z, w| t.* = ref.Iv.of(ref.bf(w)).mul(z);
                const gain = ref.tanh(later.sum(terms[0..lm.rank], lm.rank, scratch));
                const keep = ref.Iv.of(1).sub(forget);
                const old = keep.mul(ref.Iv.of(h[i]));
                const added = gain.mul(ref.Iv.of(m[i]));
                out[row * cols + i] = old.add(added);
            }
        }
    }

    pub fn check(self: *const Gate, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const cols = self.shape.n;
        const n = try a.alloc(f32, 2 * cols);
        defer a.free(n);
        const weights = try a.alloc(f32, 2 * cols);
        defer a.free(weights);
        const zeta = try a.alloc(f32, lm.rank);
        defer a.free(zeta);
        const zf = try a.alloc(f32, lm.rank);
        defer a.free(zf);
        const zg = try a.alloc(f32, lm.rank);
        defer a.free(zg);
        for (0..self.shape.rows) |row| {
            const h = self.h[row * cols ..][0..cols];
            const m = self.m[row * cols ..][0..cols];
            const ih = 1.0 / @sqrt(reduce(h, h, 256) / @as(f32, @floatFromInt(cols)) + eps);
            const im = 1.0 / @sqrt(reduce(m, m, 256) / @as(f32, @floatFromInt(cols)) + eps);
            for (n[0..cols], h) |*v, x| v.* = bfRound(x * ih);
            for (n[cols..], m) |*v, x| v.* = bfRound(x * im);
            for (0..lm.rank) |r| {
                for (weights, self.d[r * 2 * cols ..][0 .. 2 * cols]) |*v, w| v.* = bf(w);
                zeta[r] = bfRound(reduce(weights, n, 256));
            }
            for (0..cols) |i| {
                for (zf, self.f[i * lm.rank ..][0..lm.rank]) |*v, w| v.* = bf(w);
                for (zg, self.gain[i * lm.rank ..][0..lm.rank]) |*v, w| v.* = bf(w);
                var sf: f32 = 0;
                var sg: f32 = 0;
                for (0..lm.rank) |r| {
                    sf = @mulAdd(f32, zf[r], zeta[r], sf);
                    sg = @mulAdd(f32, zg[r], zeta[r], sg);
                }
                const forget = std.math.tanh(sf);
                const gain = std.math.tanh(sg);
                const keep: f32 = 1.0 - forget;
                const old: f32 = keep * h[i];
                const added: f32 = gain * m[i];
                const want: f32 = old + added;
                const value = ref.readF32(got, row * cols + i);
                if (!close(value, want)) return fail(msg, "row {d} col {d}: {e} where {e} is due", .{ row, i, value, want });
            }
        }
        return null;
    }
};
