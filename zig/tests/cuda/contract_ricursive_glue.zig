//! Ripple's CUDA glue in the contract: tf_rc_embed, tf_rc_add_norm (4 forms), tf_rc_gate, tf_rc_silu_clamp, tf_rc_tap.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig"); // the family-neutral Gated wrapper and fills

const c = core.contract.spec;
const g = core.contract.gates;
const kern = rc.kernels;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Kernels = kern.Kernels;
const Function = cuda.Function;
const Tap = @import("contract_ricursive_tap.zig").Tap; // tf_rc_tap's check

pub const hidden = 5120;
const query = 80 * 128; // the output gate's rows: query heads x head dim
const intermediate = 40960;
const vocab = 100352;
const eps: f32 = 1e-6;
const embed_scale: f32 = 71.55417527999327;
const limit: f32 = 20; // the FFN's clamp

/// The forward's row counts at width n: decode 1 to 16, odd tails and prompt chunks up to 2,048.
fn rowsAt(n: usize, k: usize) [7]c.Shape {
    var out: [7]c.Shape = undefined;
    for ([_]usize{ 16, 1, 3, 7, 333, 1000, 2048 }, &out) |r, *s| s.* = .{ .rows = r, .n = n, .k = k };
    return out;
}

const embed_rows = rowsAt(hidden, 1);
const norm_rows = rowsAt(hidden, hidden);
const gate_rows = rowsAt(query, 1);
const silu_rows = rowsAt(intermediate, 2 * intermediate);
const elementwise: c.Tile = .{ .size = 256, .step = 1, .min = 1 };

pub fn embedSpec(control: bool) c.Spec {
    const name = if (control) "control: cuda ricursive tf_rc_embed, the scale rounded to bf16 (71.5)" else "cuda ricursive tf_rc_embed (fp32 scale, rounded once; R and V0)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .tiles = .{ .{}, elementwise, .{} }, .served = &embed_rows };
}

/// R and V0 = bf16(E[ids] * scale) over the whole vocabulary's table: row 0 reads its last row.
pub const Embed = struct {
    kern: *const Kernels,
    f: Function,
    /// control: the scale rounded to bf16 before the product
    ctl: bool = false,
    shape: c.Shape = undefined,
    table: []u16 = &.{},
    ids: []u32 = &.{},
    dev: [2]u64 = @splat(0), // the table, the ids
    out: [2]u64 = @splat(0), // R, V0

    pub fn spec(self: *const Embed) c.Spec {
        return embedSpec(self.ctl);
    }

    pub fn control(self: *const Embed) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Embed, a: std.mem.Allocator) void {
        a.free(self.table);
        a.free(self.ids);
        self.table = &.{};
        self.ids = &.{};
    }

    /// The rows the ids read in [-1, 1), the rest zero (never read); non-finite: NaN, +Inf, -Inf in the first rows.
    pub fn prepare(self: *Embed, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.ids = try a.alloc(u32, shape.rows);
        for (self.ids, 0..) |*id, r| id.* = if (r == 0) vocab - 1 else rng.uintLessThan(u32, vocab);
        self.table = try a.alloc(u16, vocab * n);
        @memset(self.table, 0);
        for (self.ids) |id| for (self.table[@as(usize, id) * n ..][0..n]) |*v| {
            v.* = c.toBf16(rng.float(f32) * 2 - 1);
        };
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |x, r| if (r < shape.rows) {
            self.table[@as(usize, self.ids[r]) * n + (7 * r + 1) % n] = c.toBf16(x);
        };
        self.dev = .{ try ref.input(rig, self.table), try ref.input(rig, self.ids) };
        self.out = .{ try rig.output(shape.rows * n * 2), try rig.output(shape.rows * n * 2) };
        return true;
    }

    pub fn run(self: *Embed, rig: *rig_.Rig) !void {
        try kern.embedBy(self.f, rig.s, self.dev[1], self.dev[0], self.out[0], self.out[1], self.shape.rows, self.shape.n, embed_scale);
    }

    pub fn count(self: *const Embed) usize {
        return 2 * self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Embed, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const half = self.shape.rows * n;
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const v = Iv.of(ref.bf(self.table[@as(usize, self.ids[r]) * n + i])).mul(Iv.of(embed_scale)).bf();
            out[r * n + i] = v;
            out[half + r * n + i] = v;
        };
    }
};

/// tf_rc_add_norm's forms in the forward: a first norm (plain or with v's lerp), an add, an add into the next lerp.
pub const Form = enum { plain, lerp, add, add_lerp };

/// A control's dropped rounding point (ricursive.cuh's add_norm bits).
pub const Point = enum { y_row, y_out, resid, x_row, x_lerp };

fn adds(f: Form) bool {
    return f == .add or f == .add_lerp;
}

fn lerps(f: Form) bool {
    return f == .lerp or f == .add_lerp;
}

/// The blend each lerp form runs at: below one half and above it, torch.lerp's two branches.
fn blendOf(f: Form) f32 {
    return if (f == .add_lerp) 0.69921875 else 0.30078125;
}

/// A thread's fma chain (8 values a chunk, a chunk per 2,048 of the row), the warp's 5 shuffles and the block's 5.
fn normDepth(s: c.Shape) usize {
    return 8 * ((s.n + 2047) / 2048) + 10;
}

pub fn normSpec(f: Form, drop: ?Point) c.Spec {
    const name = if (drop) |p| switch (p) {
        .y_row => "control: cuda ricursive tf_rc_add_norm without bf16(y / rms)",
        .y_out => "control: cuda ricursive tf_rc_add_norm without bf16(w_post y / rms)",
        .resid => "control: cuda ricursive tf_rc_add_norm, the next norm reads the residual before its bf16",
        .x_row => "control: cuda ricursive tf_rc_add_norm without bf16(r / rms)",
        .x_lerp => "control: cuda ricursive tf_rc_add_norm, the lerp reads X before its bf16",
    } else switch (f) {
        .plain => "cuda ricursive tf_rc_add_norm (X = norm(R))",
        .lerp => "cuda ricursive tf_rc_add_norm lerp (X and lerp(V0, X), blend below one half)",
        .add => "cuda ricursive tf_rc_add_norm add (R += norm(Y), in place, then X)",
        .add_lerp => "cuda ricursive tf_rc_add_norm add lerp (R += norm(Y), X, lerp(V0, X) above one half)",
    };
    const chunks: c.Tile = .{ .size = 2048, .step = 8, .min = 8 }; // a thread's chunk per 2,048 of the row
    return .{ .name = name, .accum = .f32, .depth = normDepth, .out = .bf16, .in_place = adds(f), .tiles = .{ .{}, chunks, .{} }, .served = &norm_rows, .sums = .{} };
}

/// Residual rows as bf16: [-1, 1), one in 64 twenty times larger; stress: integers in [-32, 32].
fn residual(a: std.mem.Allocator, rng: std.Random, n: usize, fill: g.Fill) ![]u16 {
    const x = try a.alloc(u16, n);
    for (x, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress) ref.int(rng, -32, 32) else (rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1));
    return x;
}

/// NaN, +Inf and -Inf at known places: a row each, row 3 holding both infinities.
fn plant(x: []u16, rows: usize, n: usize) void {
    const at = [_][2]usize{ .{ 0, n / 3 }, .{ 1, n / 2 }, .{ 2, n - 1 }, .{ 3, 0 }, .{ 3, n - 1 } };
    const put = [_]f32{ ref.nan, ref.inf, -ref.inf, ref.inf, -ref.inf };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * n + p[1]] = c.toBf16(v);
    };
}

/// 1 / rms of a row's values (ranges): an fp32 sum of squares over `depth` roundings, then 1 / sqrt(mean + eps).
fn invRms(s: Sums, x: []const Iv, depth: usize, scratch: []f64, sq: []Iv) Iv {
    for (sq, x) |*q, v| {
        const lo = @min(@abs(v.lo), @abs(v.hi));
        const hi = @max(@abs(v.lo), @abs(v.hi));
        q.* = .{ .m = v.m * v.m, .lo = if (v.lo <= 0 and v.hi >= 0) 0 else lo * lo, .hi = hi * hi };
    }
    const total = s.sum(sq, depth, scratch);
    return total.div(Iv.of(@floatFromInt(x.len))).add(Iv.of(eps)).rsqrt();
}

/// torch.lerp as the kernel's fma: w (e - s) + s below one half, (w - 1)(e - s) + e above it; rounded to bf16 once.
fn lerp(s: Iv, e: Iv, w: f32) Iv {
    const d = e.sub(s);
    return (if (@abs(w) < 0.5) Iv.of(w).mul(d).add(s) else Iv.of(w - 1).mul(d).add(e)).bf();
}

/// tf_rc_add_norm on hidden-wide rows: R, w_next; Y and w_post with the add; V0 and the blend with the lerp.
pub const AddNorm = struct {
    kern: *const Kernels,
    f: Function,
    form: Form,
    drop: ?Point = null,
    shape: c.Shape = undefined,
    r: []u16 = &.{}, // R before the launch
    y: []u16 = &.{},
    post: []u16 = &.{},
    next: []u16 = &.{},
    v0: []u16 = &.{},
    p: kern.NormRows = undefined,

    pub const sums = {};

    pub fn spec(self: *const AddNorm) c.Spec {
        return normSpec(self.form, self.drop);
    }

    pub fn control(self: *const AddNorm) ?g.Gate {
        return if (self.drop != null) .shapes else null;
    }

    pub fn free(self: *AddNorm, a: std.mem.Allocator) void {
        for ([_]*[]u16{ &self.r, &self.y, &self.post, &self.next, &self.v0 }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
    }

    /// Y in [-0.5, 0.5), weights [0.5, 1.5), V0 [-2, 2); stress: Y zero; non-finite: in Y (add) or R, V0 rows 4-6.
    pub fn prepare(self: *AddNorm, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.n > kern.most_width) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        const add = adds(self.form);
        const mix = lerps(self.form);
        self.r = try residual(a, rng, rows * n, fill);
        self.y = try ref.bfs(a, rng, if (add) rows * n else 0, -0.5, 0.5);
        if (fill == .stress) @memset(self.y, 0);
        self.post = try ref.bfs(a, rng, if (add) n else 0, 0.5, 1.5);
        self.next = try ref.bfs(a, rng, n, 0.5, 1.5);
        self.v0 = try ref.bfs(a, rng, if (mix) rows * n else 0, -2, 2);
        if (fill == .nonfinite) {
            plant(if (add) self.y else self.r, rows, n);
            if (mix) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 4..) |x, row| if (row < rows) {
                self.v0[row * n + n / 4] = c.toBf16(x);
            };
        }
        const bytes = rows * n * 2;
        self.p = .{ .resid = 0, .next = try ref.input(rig, self.next), .x = 0, .rows = rows, .width = n, .eps = eps, .blend = if (mix) blendOf(self.form) else 0 };
        if (add) {
            self.p.branch = try ref.input(rig, self.y);
            self.p.post = try ref.input(rig, self.post);
            self.p.resid = try rig.output(bytes); // in place: restored before each launch
        } else self.p.resid = try ref.input(rig, self.r);
        if (mix) self.p.v0 = try ref.input(rig, self.v0);
        self.p.x = try rig.output(bytes);
        if (mix) self.p.vx = try rig.output(bytes);
        return true;
    }

    pub fn run(self: *AddNorm, rig: *rig_.Rig) !void {
        if (adds(self.form)) try ref.restore(rig, self.p.resid, std.mem.sliceAsBytes(self.r)); // read and written in place: every run the same pass
        try kern.addNormBy(self.f, rig.s, self.p);
    }

    pub fn count(self: *const AddNorm) usize {
        const outs = 1 + @as(usize, @intFromBool(adds(self.form))) + @intFromBool(lerps(self.form));
        return outs * self.shape.rows * self.shape.n;
    }

    /// The outputs in the order prepare allocates them: R (with the add), X, VX (with the lerp).
    pub fn solve(self: *const AddNorm, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const total = self.shape.rows * n;
        const buf = try a.alloc(Iv, 3 * n);
        defer a.free(buf);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        const depth = normDepth(self.shape);
        const r = buf[0..n];
        const y = buf[n .. 2 * n];
        const sq = buf[2 * n ..];
        const lead: usize = if (adds(self.form)) total else 0;
        const out_r = out[0..lead];
        const out_x = out[lead..][0..total];
        const out_vx = out[lead + total ..];
        for (0..self.shape.rows) |row| {
            const o = row * n;
            for (r, self.r[o..][0..n]) |*v, h| v.* = Iv.of(ref.bf(h));
            if (adds(self.form)) {
                for (y, self.y[o..][0..n]) |*v, h| v.* = Iv.of(ref.bf(h));
                const inv_y = invRms(s, y, depth, scratch, sq);
                for (r, y, self.post) |*v, yv, w| v.* = v.add(Iv.of(ref.bf(w)).mul(yv.mul(inv_y).bf()).bf()).bf();
                @memcpy(out_r[o..][0..n], r);
            }
            const inv = invRms(s, r, depth, scratch, sq);
            for (out_x[o..][0..n], r, self.next) |*x, v, w| x.* = Iv.of(ref.bf(w)).mul(v.mul(inv).bf()).bf();
            if (lerps(self.form)) for (out_vx[o..][0..n], out_x[o..][0..n], self.v0[o..][0..n]) |*vx, x, h| {
                vx.* = lerp(Iv.of(ref.bf(h)), x, blendOf(self.form));
            };
        }
    }
};

/// The control kernel without point `p` (ricursive.cu's tf_rc_ctl_add_norm_*).
fn dropped(k: *const Kernels, p: Point) !Function {
    return switch (p) {
        .y_row => k.mod.function("tf_rc_ctl_add_norm_y_row"),
        .y_out => k.mod.function("tf_rc_ctl_add_norm_y_out"),
        .resid => k.mod.function("tf_rc_ctl_add_norm_resid"),
        .x_row => k.mod.function("tf_rc_ctl_add_norm_x_row"),
        .x_lerp => k.mod.function("tf_rc_ctl_add_norm_x_lerp"),
    };
}

pub fn gateSpec(control: bool) c.Spec {
    const name = if (control) "control: cuda ricursive tf_rc_gate without bf16(sigmoid(g))" else "cuda ricursive tf_rc_gate (fp32 sigmoid then bf16, in place)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .in_place = true, .tiles = .{ .{}, elementwise, .{} }, .served = &gate_rows };
}

/// A = bf16(A * bf16(sigmoid(G))) in place: A in [-4, 4), G in [-12, 12); non-finite: G's NaN and infinities, then A's.
pub const Gate = struct {
    kern: *const Kernels,
    f: Function,
    /// control: the sigmoid left in fp32
    ctl: bool = false,
    shape: c.Shape = undefined,
    act: []u16 = &.{},
    gv: []u16 = &.{},
    dev_a: u64 = 0,
    dev_g: u64 = 0,

    pub fn spec(self: *const Gate) c.Spec {
        return gateSpec(self.ctl);
    }

    pub fn control(self: *const Gate) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Gate, a: std.mem.Allocator) void {
        a.free(self.act);
        a.free(self.gv);
        self.act = &.{};
        self.gv = &.{};
    }

    pub fn prepare(self: *Gate, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.act = try ref.bfs(a, rng, shape.rows * n, -4, 4);
        self.gv = try ref.bfs(a, rng, shape.rows * n, -12, 12);
        if (fill == .nonfinite) {
            const put = [_]struct { usize, bool, f32 }{ .{ 0, true, ref.nan }, .{ 1, true, ref.inf }, .{ 2, true, -ref.inf }, .{ 3, false, ref.nan }, .{ 4, false, ref.inf }, .{ 5, false, -ref.inf } };
            for (put) |p| if (p[0] < shape.rows) {
                (if (p[1]) self.gv else self.act)[p[0] * n + p[0] + 1] = c.toBf16(p[2]);
            };
        }
        self.dev_g = try ref.input(rig, self.gv);
        self.dev_a = try rig.output(shape.rows * n * 2); // in place: restored before each launch
        return true;
    }

    pub fn run(self: *Gate, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.dev_a, std.mem.sliceAsBytes(self.act));
        try kern.gateBy(self.f, rig.s, self.dev_a, self.dev_g, self.shape.rows, self.shape.n);
    }

    pub fn count(self: *const Gate) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Gate, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.act, self.gv) |*o, x, gh| o.* = Iv.of(ref.bf(x)).mul(Iv.of(ref.bf(gh)).sigmoid().bf()).bf();
    }
};

/// A SiLU-clamp control: the silu left in fp32, or up clamped above only.
pub const SiluDrop = enum { round, lower };

const silu_nan = "torch.clamp's bounds: a NaN gate or up gives NaN, +Inf reads as the limit, a -Inf up as -limit; a -Inf gate gives NaN (-Inf / Inf)";

pub fn siluSpec(drop: ?SiluDrop) c.Spec {
    const name = if (drop) |d| switch (d) {
        .round => "control: cuda ricursive tf_rc_silu_clamp without bf16(silu(g))",
        .lower => "control: cuda ricursive tf_rc_silu_clamp, up clamped above only",
    } else "cuda ricursive tf_rc_silu_clamp (stacked gate | up, clamp 20)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .nan = .{ .stated = silu_nan }, .tiles = .{ .{}, elementwise, .{} }, .served = &silu_rows };
}

/// bf16(bf16(silu(min(g, 20))) * clamp(u, -20, 20)) from stacked gate | up rows, as the up-gate projection writes them.
pub const SiluClamp = struct {
    kern: *const Kernels,
    f: Function,
    drop: ?SiluDrop = null,
    shape: c.Shape = undefined,
    gate: []u16 = &.{},
    up: []u16 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    pub fn spec(self: *const SiluClamp) c.Spec {
        return siluSpec(self.drop);
    }

    pub fn control(self: *const SiluClamp) ?g.Gate {
        return if (self.drop != null) .shapes else null;
    }

    pub fn free(self: *SiluClamp, a: std.mem.Allocator) void {
        a.free(self.gate);
        a.free(self.up);
        self.gate = &.{};
        self.up = &.{};
    }

    /// Gate and up in [-30, 30) (past the clamp both ways); non-finite: NaN, +Inf and -Inf in each of gate and up.
    pub fn prepare(self: *SiluClamp, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        self.gate = try ref.bfs(a, rng, rows * n, -30, 30);
        self.up = try ref.bfs(a, rng, rows * n, -30, 30);
        if (fill == .nonfinite) {
            const put = [_]struct { usize, bool, f32 }{ .{ 0, true, ref.nan }, .{ 1, true, ref.inf }, .{ 2, true, -ref.inf }, .{ 3, false, ref.nan }, .{ 4, false, ref.inf }, .{ 5, false, -ref.inf } };
            for (put) |p| if (p[0] < rows) {
                (if (p[1]) self.gate else self.up)[p[0] * n + p[0] + 1] = c.toBf16(p[2]);
            };
        }
        const both = try a.alloc(u16, 2 * rows * n);
        defer a.free(both);
        for (0..rows) |r| {
            @memcpy(both[2 * r * n ..][0..n], self.gate[r * n ..][0..n]);
            @memcpy(both[(2 * r + 1) * n ..][0..n], self.up[r * n ..][0..n]);
        }
        self.dev = try ref.input(rig, both);
        self.out = try rig.output(rows * n * 2);
        return true;
    }

    pub fn run(self: *SiluClamp, rig: *rig_.Rig) !void {
        try kern.siluClampBy(self.f, rig.s, self.dev, self.out, self.shape.rows, self.shape.n, limit);
    }

    pub fn count(self: *const SiluClamp) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const SiluClamp, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.gate, self.up) |*o, gh, uh| {
            const gv = Iv.of(ref.bf(gh)).atMost(limit);
            const u = Iv.of(ref.bf(uh)).atMost(limit).atLeast(-limit);
            const silu = gv.div(Iv.of(1).add(gv.neg().exp())).bf();
            o.* = silu.mul(u).bf();
        }
    }
};

/// The glue's registrations, held where the gates point at them (the caller's heap set).
pub const Set = struct {
    embeds: [2]ref.Gated(Embed) = undefined,
    norms: [4]ref.Gated(AddNorm) = undefined,
    norm_ctls: [5]ref.Gated(AddNorm) = undefined,
    gates: [2]ref.Gated(Gate) = undefined,
    silus: [3]ref.Gated(SiluClamp) = undefined,
    taps: [5]ref.Checked(Tap) = undefined,

    /// Every registration over `k`, appended to `list` (each control after the kernel it breaks).
    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, k: *const Kernels, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        set.embeds = .{
            .{ .rig = rig, .k = .{ .kern = k, .f = k.embed } },
            .{ .rig = rig, .k = .{ .kern = k, .f = try k.mod.function("tf_rc_ctl_embed_bf16_scale"), .ctl = true } },
        };
        for (&set.norms, std.enums.values(Form)) |*x, f| x.* = .{ .rig = rig, .k = .{ .kern = k, .f = k.add_norm, .form = f } };
        for (&set.norm_ctls, std.enums.values(Point)) |*x, p| x.* = .{ .rig = rig, .k = .{ .kern = k, .f = try dropped(k, p), .form = .add_lerp, .drop = p } };
        set.gates = .{
            .{ .rig = rig, .k = .{ .kern = k, .f = k.gate } },
            .{ .rig = rig, .k = .{ .kern = k, .f = try k.mod.function("tf_rc_ctl_gate_sigmoid"), .ctl = true } },
        };
        set.silus = .{
            .{ .rig = rig, .k = .{ .kern = k, .f = k.silu_clamp } },
            .{ .rig = rig, .k = .{ .kern = k, .f = try k.mod.function("tf_rc_ctl_silu_round"), .drop = .round } },
            .{ .rig = rig, .k = .{ .kern = k, .f = try k.mod.function("tf_rc_ctl_silu_no_lower"), .drop = .lower } },
        };
        set.taps = .{
            .{ .rig = rig, .k = .{ .kern = k, .f = k.tap, .form = .chunk } },
            .{ .rig = rig, .k = .{ .kern = k, .f = k.tap, .form = .round } },
            .{ .rig = rig, .k = .{ .kern = k, .f = try k.mod.function("tf_rc_ctl_tap_column"), .form = .round, .fault = .column } },
            .{ .rig = rig, .k = .{ .kern = k, .f = try k.mod.function("tf_rc_ctl_tap_ring"), .form = .round, .fault = .ring } },
            .{ .rig = rig, .k = .{ .kern = k, .f = try k.mod.function("tf_rc_ctl_tap_swap"), .form = .round, .fault = .swap } },
        };
        for (&set.embeds) |*x| try list.append(a, x.kernel());
        for (&set.norms) |*x| try list.append(a, x.kernel());
        for (&set.norm_ctls) |*x| try list.append(a, x.kernel());
        for (&set.gates) |*x| try list.append(a, x.kernel());
        for (&set.silus) |*x| try list.append(a, x.kernel());
        for (&set.taps) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        for (&set.embeds) |*x| x.deinit();
        for (&set.norms) |*x| x.deinit();
        for (&set.norm_ctls) |*x| x.deinit();
        for (&set.gates) |*x| x.deinit();
        for (&set.silus) |*x| x.deinit();
        for (&set.taps) |*x| x.deinit();
    }
};
