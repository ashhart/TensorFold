//! ZONOS3's GDN layer on Metal under the contract: norms, gates, recurrence, gated norm (contract_zonos3_gdn's twin).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ref = @import("metal_zonos3_ref.zig");

const c = contract.spec;
const g = contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

const conv_width = 1536;

fn name(comptime plain: []const u8, comptime broken: []const u8, fault: bool) []const u8 {
    return if (fault) "control: metal zonos3 " ++ broken else "metal zonos3 " ++ plain;
}

const heads = 4;
const head = 128;
const inner = heads * head;

/// A simdgroup's butterfly, then the four simdgroups' (the second butterfly's other lanes add exact zeros).
fn headDeep(_: c.Shape) usize {
    return 7;
}

/// Rows x the q | k | v channels x a head's 128.
pub fn prepSpec(fault: bool) c.Spec {
    return .{ .name = name("z3_gdn_prep (q and k in place)", "z3_gdn_prep reading its gates' inputs one value late", fault), .accum = .f32, .depth = headDeep, .out = .f32, .in_place = true, .served = &.{ .{ .rows = 64, .n = conv_width, .k = head }, .{ .rows = 1, .n = conv_width, .k = head }, .{ .rows = 7, .n = conv_width, .k = head } } };
}

/// The GDN's inputs: q and k l2-normed per head in place (q times 128^-1/2), and each (row, head)'s decay and beta.
pub const GdnPrep = struct {
    pipe: mtl.Pipeline,
    /// control: the device reads a and b one value late in the projection row
    fault: bool = false,
    T: usize = 0,
    conv: []f32 = &.{},
    proj: []f32 = &.{},
    a_log: [heads]u16 = undefined,
    dt_bias: [heads]u16 = undefined,
    dev: [3]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub const sums = {};
    const ps = 2 * heads + 4;
    const a_off = 3;
    const q_scale: f32 = 0.088388347648318447;

    pub fn spec(self: *const GdnPrep) c.Spec {
        return prepSpec(self.fault);
    }

    pub fn control(self: *const GdnPrep) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *GdnPrep, a: std.mem.Allocator) void {
        a.free(self.conv);
        a.free(self.proj);
        self.conv = &.{};
        self.proj = &.{};
    }

    pub fn prepare(self: *GdnPrep, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != conv_width or shape.k != head) return false;
        self.free(a);
        self.T = shape.rows;
        self.conv = try ref.values(a, rng, self.T * conv_width, fill, 2);
        self.proj = try ref.values(a, rng, self.T * ps, .random, 3);
        for (&self.a_log, &self.dt_bias) |*x, *y| {
            x.* = c.toBf16(rng.float(f32) * 2 - 2);
            y.* = c.toBf16(rng.float(f32) * 2 - 1);
        }
        if (fill == .nonfinite) ref.plant(self.conv, self.T, conv_width);
        self.dev = .{ try ref.input(rig, self.proj), try ref.input(rig, &self.a_log), try ref.input(rig, &self.dt_bias) };
        self.out = .{ try rig.output(self.conv.len * 4), try rig.output(self.T * heads * 2 * 4) };
        return true;
    }

    pub fn run(self: *GdnPrep, rig: *ref.Rig) !void {
        try ref.restore(rig, self.out[0], std.mem.sliceAsBytes(self.conv));
        const A = extern struct { ps: i32, a_off: i32 };
        const a: A = .{ .ps = ref.int(ps), .a_off = ref.int(a_off + ref.one(self.fault)) };
        try rig.go(self.pipe, .{ self.T, heads, 1 }, head, &.{ self.out[0], self.dev[0], self.dev[1], self.dev[2], self.out[1] }, a, 0);
    }

    pub fn count(self: *const GdnPrep) usize {
        return self.T * conv_width + self.T * heads * 2;
    }

    pub fn solve(self: *const GdnPrep, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        var scratch: [head]f64 = undefined;
        var sq: [head]Iv = undefined;
        const gates = out[self.T * conv_width ..];
        for (0..self.T) |t| {
            const row = self.conv[t * conv_width ..][0..conv_width];
            for (0..2 * heads) |part| {
                const x = row[part * head ..][0..head];
                for (&sq, x) |*q, v| q.* = ref.square(Iv.of(v));
                const inv = s.sum(&sq, 7, &scratch).add(Iv.of(@as(f32, 1e-6))).rsqrt();
                for (x, 0..) |v, i| {
                    const y = Iv.of(v).mul(inv);
                    out[t * conv_width + part * head + i] = if (part < heads) y.mul(Iv.of(q_scale)) else y;
                }
            }
            for (row[2 * inner ..], 0..) |v, i| out[t * conv_width + 2 * inner + i] = Iv.of(v);
            const p = self.proj[t * ps + a_off ..][0 .. 2 * heads];
            for (0..heads) |h| {
                const rate = Iv.of(ref.bf(self.a_log[h])).exp().neg();
                gates[(t * heads + h) * 2] = rate.mul(ref.softplus(Iv.of(p[h]).add(Iv.of(ref.bf(self.dt_bias[h]))))).exp();
                gates[(t * heads + h) * 2 + 1] = Iv.of(p[heads + h]).sigmoid();
            }
        }
    }
};

fn chainDeep(_: c.Shape) usize {
    return head;
}

/// Rows x the heads' outputs x a state row's 128.
pub fn gdnSpec(fault: bool) c.Spec {
    return .{ .name = name("z3_gdn (state in place)", "z3_gdn launched one row short", fault), .accum = .f32, .depth = chainDeep, .out = .f32, .in_place = true, .served = &.{ .{ .rows = 8, .n = inner, .k = head }, .{ .rows = 1, .n = inner, .k = head }, .{ .rows = 64, .n = inner, .k = head } } };
}

/// The GDN recurrence by token: a state row decays, predicts, takes the gated error along k, reads out along q.
pub const Gdn = struct {
    pipe: mtl.Pipeline,
    /// control: the device runs rows - 1 rows
    fault: bool = false,
    T: usize = 0,
    conv: []f32 = &.{},
    gates: []f32 = &.{},
    state: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub const sums = {};
    const state_len = heads * head * head;

    pub fn spec(self: *const Gdn) c.Spec {
        return gdnSpec(self.fault);
    }

    pub fn control(self: *const Gdn) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Gdn, a: std.mem.Allocator) void {
        a.free(self.conv);
        a.free(self.gates);
        a.free(self.state);
        self.conv = &.{};
        self.gates = &.{};
        self.state = &.{};
    }

    pub fn prepare(self: *Gdn, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != inner or shape.k != head) return false;
        self.free(a);
        const T = shape.rows;
        self.T = T;
        // keys far below unit length keep the recurrence's float64 ranges from widening a step
        self.conv = try ref.values(a, rng, T * conv_width, .random, 0.02);
        self.gates = try a.alloc(f32, T * heads * 2);
        for (self.gates, 0..) |*v, i| v.* = if (i % 2 == 0) 0.9 + 0.1 * rng.float(f32) else rng.float(f32);
        self.state = try ref.values(a, rng, state_len, .random, 1);
        for (0..T) |t| for (self.conv[t * conv_width + 2 * inner ..][0..inner]) |*v| {
            v.* = rng.float(f32) * 2 - 1;
        };
        if (fill == .stress) {
            // q and k all ones, no decay, values zero; the first row takes its whole error, the later rows none
            for (0..T) |t| {
                @memset(self.conv[t * conv_width ..][0 .. 2 * inner], 1);
                @memset(self.conv[t * conv_width + 2 * inner ..][0..inner], 0);
            }
            for (self.gates, 0..) |*v, i| v.* = if (i % 2 == 0 or i < heads * 2) 1 else 0;
            for (0..heads * head) |row| ref.stress(self.state[row * head ..][0..head], rng);
        }
        if (fill == .nonfinite) {
            self.state[5 * head + 3] = ref.nan; // head 0, value row 5
            if (T > 1) self.conv[conv_width + head + 7] = ref.nan; // row 1, head 1's q
            if (T > 2) self.conv[2 * conv_width + 2 * head + 9] = ref.inf; // row 2, head 2's q
        }
        self.dev = .{ try ref.input(rig, self.conv), try ref.input(rig, self.gates) };
        self.out = .{ try rig.output(T * inner * 4), try rig.output(state_len * 4) };
        return true;
    }

    pub fn run(self: *Gdn, rig: *ref.Rig) !void {
        try ref.restore(rig, self.out[1], std.mem.sliceAsBytes(self.state));
        const A = extern struct { T: i32 };
        try rig.go(self.pipe, .{ heads, 1, 1 }, head, &.{ self.dev[0], self.dev[1], self.out[1], self.out[0] }, A{ .T = ref.int(self.T - ref.one(self.fault)) }, 0);
    }

    pub fn count(self: *const Gdn) usize {
        return self.T * inner + state_len;
    }

    pub fn solve(self: *const Gdn, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        var scratch: [head]f64 = undefined;
        var row: [head]Iv = undefined;
        var terms: [head]Iv = undefined;
        for (0..heads * head) |at| {
            const h = at / head;
            for (&row, self.state[at * head ..][0..head]) |*r, v| r.* = Iv.of(v);
            for (0..self.T) |t| {
                const q = self.conv[t * conv_width + h * head ..][0..head];
                const k = self.conv[t * conv_width + inner + h * head ..][0..head];
                const val = Iv.of(self.conv[t * conv_width + 2 * inner + at]);
                const decay = Iv.of(self.gates[(t * heads + h) * 2]);
                const beta = Iv.of(self.gates[(t * heads + h) * 2 + 1]);
                for (&row, &terms, k) |*r, *term, kv| {
                    r.* = r.mul(decay);
                    term.* = r.mul(Iv.of(kv));
                }
                const delta = beta.mul(val.sub(s.sum(&terms, head, &scratch)));
                for (&row, &terms, k, q) |*r, *term, kv, qv| {
                    r.* = r.add(delta.mul(Iv.of(kv)));
                    term.* = r.mul(Iv.of(qv));
                }
                out[t * inner + at] = s.sum(&terms, head, &scratch);
            }
            for (row, 0..) |r, j| out[self.T * inner + at * head + j] = r;
        }
    }
};

/// Rows x the heads' outputs x a head's 128.
pub fn gatedSpec(fault: bool) c.Spec {
    return .{ .name = name("z3_gated_norm", "z3_gated_norm reading its gate one value late", fault), .accum = .f32, .depth = headDeep, .out = .f32, .served = &.{ .{ .rows = 64, .n = inner, .k = head }, .{ .rows = 1, .n = inner, .k = head }, .{ .rows = 7, .n = inner, .k = head } } };
}

/// y = rms_norm(o, w, 1e-5) silu(z) per head of 128.
pub const GatedNorm = struct {
    pipe: mtl.Pipeline,
    /// control: the device reads z one value late in the projection row
    fault: bool = false,
    T: usize = 0,
    o: []f32 = &.{},
    proj: []f32 = &.{},
    w: []u16 = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};
    const z_off = 5;
    const ps = inner + z_off + 1;

    pub fn spec(self: *const GatedNorm) c.Spec {
        return gatedSpec(self.fault);
    }

    pub fn control(self: *const GatedNorm) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *GatedNorm, a: std.mem.Allocator) void {
        a.free(self.o);
        a.free(self.proj);
        a.free(self.w);
        self.o = &.{};
        self.proj = &.{};
        self.w = &.{};
    }

    pub fn prepare(self: *GatedNorm, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != inner or shape.k != head) return false;
        self.free(a);
        self.T = shape.rows;
        self.o = try ref.values(a, rng, self.T * inner, fill, 2);
        self.proj = try ref.values(a, rng, self.T * ps, .random, 3);
        self.w = try ref.bfs(a, rng, head, -1, 3);
        if (fill == .nonfinite) ref.plant(self.o, self.T, inner);
        self.dev = .{ try ref.input(rig, self.o), try ref.input(rig, self.proj), try ref.input(rig, self.w) };
        self.out = try rig.output(self.T * inner * 4);
        return true;
    }

    pub fn run(self: *GatedNorm, rig: *ref.Rig) !void {
        const A = extern struct { ps: i32, z_off: i32 };
        const a: A = .{ .ps = ref.int(ps), .z_off = ref.int(z_off + ref.one(self.fault)) };
        try rig.go(self.pipe, .{ self.T, heads, 1 }, head, &.{ self.dev[0], self.dev[1], self.dev[2], self.out }, a, 0);
    }

    pub fn count(self: *const GatedNorm) usize {
        return self.T * inner;
    }

    pub fn solve(self: *const GatedNorm, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        var scratch: [head]f64 = undefined;
        var sq: [head]Iv = undefined;
        for (0..self.T * heads) |at| {
            const x = self.o[at * head ..][0..head];
            for (&sq, x) |*q, v| q.* = ref.square(Iv.of(v));
            const inv = ref.rmsScale(s.sum(&sq, 7, &scratch), head, 1e-5);
            const z = self.proj[at / heads * ps + z_off + at % heads * head ..][0..head];
            for (x, z, self.w, 0..) |v, gate, w, i| out[at * head + i] = Iv.of(v).mul(inv).mul(Iv.of(ref.bf(w))).mul(ref.silu(Iv.of(gate)));
        }
    }
};

fn Pair(comptime K: type) type {
    return [2]ref.Gated(K);
}

/// The GDN layer's kernels, each followed by its control.
pub const Set = struct {
    prep: Pair(GdnPrep),
    gdn: Pair(Gdn),
    gated_norm: Pair(GatedNorm),

    const fields = .{ "prep", "gdn", "gated_norm" };

    pub fn init(set: *Set, libs: *ref.Libs, rig: ref.Rig, list: ref.List) !void {
        for (0..2) |j| {
            const fault = j == 1;
            set.prep[j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.backbone, "z3_gdn_prep"), .fault = fault } };
            set.gdn[j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.backbone, "z3_gdn"), .fault = fault } };
            set.gated_norm[j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.backbone, "z3_gated_norm"), .fault = fault } };
        }
        inline for (fields) |f| for (&@field(set, f)) |*x| try list.add(x.kernel());
    }

    pub fn deinit(set: *Set) void {
        inline for (fields) |f| for (&@field(set, f)) |*x| x.deinit();
    }
};
