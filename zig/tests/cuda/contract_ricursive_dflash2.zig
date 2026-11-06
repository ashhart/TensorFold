//! Ripple's DFlash2 drafter glue on CUDA under the contract, with controls: embed, add-norm, prep, conv and silu.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig"); // the family-neutral Gated wrapper and fills

const c = core.contract.spec;
const g = core.contract.gates;
const df = rc.dflash2_kernels;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Kernels = df.Kernels;
const Function = cuda.Function;
const prep_ = @import("contract_ricursive_dflash2_prep.zig"); // the prep kernel: entries, streams' plan, check
const Prep = prep_.Prep;
const PrepCtl = prep_.PrepCtl;
const prep_entries = prep_.prep_entries;

const hidden = 5120;
const ffn = 40960;
const vocab = 100352;
const block = 16; // rows a stream's block
const taps = 4;
const group = 16; // channels a dynamic tap weight covers
pub const eps: f32 = 1e-6;
const elementwise: c.Tile = .{ .size = 256, .step = 1, .min = 1 };

/// The drafter's rows a launch at width n: a block of 16 a stream, a row, 8 streams' blocks, odd tails.
fn rowsAt(comptime counts: []const usize, n: usize, k: usize) [counts.len]c.Shape {
    var out: [counts.len]c.Shape = undefined;
    for (counts, &out) |r, *s| s.* = .{ .rows = r, .n = n, .k = k };
    return out;
}

const block_rows = [_]usize{ 16, 1, 128, 7, 48 };
const embed_rows = rowsAt(&block_rows, hidden, 1);
const add_rows = rowsAt(&block_rows, hidden, hidden);
const norm16_rows = rowsAt(&.{ 2048, 7, 16, 1 }, hidden, hidden); // the context's rows, a prompt chunk at most
const silu_rows = rowsAt(&block_rows, ffn, 2 * ffn);

pub fn embedSpec(control: bool) c.Spec {
    const name = if (control) "control: cuda ricursive_dflash2 tf_rcd_ctl_embed_own_row, row r reads table row r, not ids[r]" else "cuda ricursive tf_rc_embed at scale 1 (the drafter's unscaled embed; R and its copy)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .tiles = .{ .{}, elementwise, .{} }, .served = &embed_rows };
}

/// R and its copy = E[ids], bits unchanged (x * 1.0f is x, and bf16 of a bf16 is itself), over the whole table.
pub const Embed = struct {
    f: Function,
    ctl: bool = false,
    shape: c.Shape = undefined,
    table: []u16 = &.{},
    ids: []u32 = &.{},
    dev: [2]u64 = @splat(0), // the table, the ids
    out: [2]u64 = @splat(0), // R, the copy

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

    /// Rows the ids read in [-1, 1), the rest zero; row 0 reads the last row. Non-finite: NaN, +Inf, -Inf in rows 0-2.
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
        try df.embedBy(self.f, rig.s, self.dev[1], self.dev[0], self.out[0], self.out[1], self.shape.rows, self.shape.n);
    }

    pub fn count(self: *const Embed) usize {
        return 2 * self.shape.rows * self.shape.n;
    }

    /// Exact: each output is its table entry.
    pub fn solve(self: *const Embed, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const half = self.shape.rows * n;
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const v = Iv.of(ref.bf(self.table[@as(usize, self.ids[r]) * n + i]));
            out[r * n + i] = v;
            out[half + r * n + i] = v;
        };
    }
};

/// tf_rcd_add_norm's forms: the add (R = bf16(R + Y) in place, then X), and norm16 (X = norm(R), R only read).
pub const NormForm = enum { add, norm16 };

/// A control's dropped rounding point (ricursive_dflash2.cuh's add_norm bits).
pub const NormPoint = enum { resid, x_row };

/// A thread's fma chain (8 values a chunk, a chunk per 2,048 of the row), the warp's 5 shuffles and the block's 5.
fn normDepth(s: c.Shape) usize {
    return 8 * ((s.n + 2047) / 2048) + 10;
}

pub fn normSpec(f: NormForm, drop: ?NormPoint) c.Spec {
    const name = if (drop) |p| switch (p) {
        .resid => "control: cuda ricursive_dflash2 tf_rcd_add_norm, the norm reads R + Y before its bf16",
        .x_row => "control: cuda ricursive_dflash2 tf_rcd_add_norm without bf16(r / rms)",
    } else switch (f) {
        .add => "cuda ricursive_dflash2 tf_rcd_add_norm (R = bf16(R + Y) in place, then X = norm(R))",
        .norm16 => "cuda ricursive_dflash2 tf_rcd_add_norm as norm16 (X = norm(R), R only read)",
    };
    const chunks: c.Tile = .{ .size = 2048, .step = 8, .min = 8 }; // a thread's chunk per 2,048 of the row
    const served: []const c.Shape = if (f == .add) &add_rows else &norm16_rows;
    return .{ .name = name, .accum = .f32, .depth = normDepth, .out = .bf16, .in_place = f == .add, .tiles = .{ .{}, chunks, .{} }, .served = served, .sums = .{} };
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

/// tf_rcd_add_norm on hidden-wide rows: R and w; Y with the add.
pub const AddNorm = struct {
    f: Function,
    form: NormForm,
    drop: ?NormPoint = null,
    shape: c.Shape = undefined,
    r: []u16 = &.{}, // R before the launch
    y: []u16 = &.{},
    w: []u16 = &.{},
    p: df.Norm = undefined,

    pub const sums = {};

    pub fn spec(self: *const AddNorm) c.Spec {
        return normSpec(self.form, self.drop);
    }

    pub fn control(self: *const AddNorm) ?g.Gate {
        return if (self.drop != null) .shapes else null;
    }

    pub fn free(self: *AddNorm, a: std.mem.Allocator) void {
        for ([_]*[]u16{ &self.r, &self.y, &self.w }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
    }

    /// Y in [-2, 2), w in [0.5, 1.5); stress: Y zero (R's integers stay whole); non-finite: in Y with the add, else R.
    pub fn prepare(self: *AddNorm, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.n > df.most_width) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        const add = self.form == .add;
        self.r = try residual(a, rng, rows * n, fill);
        self.y = try ref.bfs(a, rng, if (add) rows * n else 0, -2, 2);
        if (fill == .stress) @memset(self.y, 0);
        self.w = try ref.bfs(a, rng, n, 0.5, 1.5);
        if (fill == .nonfinite) plant(if (add) self.y else self.r, rows, n);
        const bytes = rows * n * 2;
        self.p = .{ .resid = 0, .w = try ref.input(rig, self.w), .x = 0, .rows = rows, .width = n, .eps = eps };
        if (add) {
            self.p.branch = try ref.input(rig, self.y);
            self.p.resid = try rig.output(bytes); // in place: restored before each launch
        } else self.p.resid = try ref.input(rig, self.r);
        self.p.x = try rig.output(bytes);
        return true;
    }

    pub fn run(self: *AddNorm, rig: *rig_.Rig) !void {
        if (self.form == .add) try ref.restore(rig, self.p.resid, std.mem.sliceAsBytes(self.r)); // every run the same pass
        try df.normBy(self.f, rig.s, self.p);
    }

    pub fn count(self: *const AddNorm) usize {
        const outs: usize = if (self.form == .add) 2 else 1;
        return outs * self.shape.rows * self.shape.n;
    }

    /// The outputs in the order prepare allocates them: R (with the add), X.
    pub fn solve(self: *const AddNorm, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const total = self.shape.rows * n;
        const buf = try a.alloc(Iv, 2 * n);
        defer a.free(buf);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        const depth = normDepth(self.shape);
        const r = buf[0..n];
        const sq = buf[n..];
        const add = self.form == .add;
        const lead: usize = if (add) total else 0;
        const out_x = out[lead..][0..total];
        for (0..self.shape.rows) |row| {
            const o = row * n;
            for (r, self.r[o..][0..n]) |*v, h| v.* = Iv.of(ref.bf(h));
            if (add) {
                for (r, self.y[o..][0..n]) |*v, h| v.* = v.add(Iv.of(ref.bf(h))).bf();
                @memcpy(out[o..][0..n], r);
            }
            const inv = invRms(s, r, depth, scratch, sq);
            for (out_x[o..][0..n], r, self.w) |*x, v, w| x.* = Iv.of(ref.bf(w)).mul(v.mul(inv).bf()).bf();
        }
    }
};

/// The control kernel without point `p` (ricursive_dflash2.cu's tf_rcd_ctl_add_norm_*).
fn normControl(k: *const Kernels, p: NormPoint) !Function {
    return switch (p) {
        .resid => k.mod.function("tf_rcd_ctl_add_norm_resid"),
        .x_row => k.mod.function("tf_rcd_ctl_add_norm_x_row"),
    };
}

fn prepControl(k: *const Kernels, x: PrepCtl) !Function {
    return switch (x) {
        .norm_row => k.mod.function("tf_rcd_ctl_prep_norm_row"),
        .norm_out => k.mod.function("tf_rcd_ctl_prep_norm_out"),
        .cos => k.mod.function("tf_rcd_ctl_prep_cos"),
        .sin => k.mod.function("tf_rcd_ctl_prep_sin"),
        .rot_c => k.mod.function("tf_rcd_ctl_prep_rot_c"),
        .rot_s => k.mod.function("tf_rcd_ctl_prep_rot_s"),
        .out => k.mod.function("tf_rcd_ctl_prep_out"),
        .nope_ropes => k.mod.function("tf_rcd_ctl_prep_nope_ropes"),
        .mask => k.mod.function("tf_rcd_ctl_prep_mask"),
    };
}

/// A conv control: a rounding point dropped, or an edit (Qwen's two roundings, no block-edge guard, base/dyn swapped).
pub const ConvCtl = enum { base, add, dyn, two_round, no_edge, swap };

fn convControl(k: *const Kernels, x: ConvCtl) !Function {
    return switch (x) {
        .base => k.mod.function("tf_rcd_ctl_conv_base"),
        .add => k.mod.function("tf_rcd_ctl_conv_add"),
        .dyn => k.mod.function("tf_rcd_ctl_conv_dyn"),
        .two_round => k.mod.function("tf_rcd_ctl_conv_two_round"),
        .no_edge => k.mod.function("tf_rcd_ctl_conv_no_edge"),
        .swap => k.mod.function("tf_rcd_ctl_conv_swap"),
    };
}

pub const ConvEntry = struct { name: []const u8, side: usize, ctl: ?ConvCtl = null };

pub const conv_entries = [_]ConvEntry{
    .{ .name = "cuda ricursive_dflash2 tf_rcd_conv side 0 (4 taps inside each block of 16)", .side = 0 },
    .{ .name = "cuda ricursive_dflash2 tf_rcd_conv side 1", .side = 1 },
    .{ .name = "control: ricursive_dflash2 tf_rcd_conv drops bf16(base x)", .side = 1, .ctl = .base },
    .{ .name = "control: ricursive_dflash2 tf_rcd_conv drops bf16(y + bf16(base x))", .side = 1, .ctl = .add },
    .{ .name = "control: ricursive_dflash2 tf_rcd_conv drops bf16(fma(dyn, x, y))", .side = 1, .ctl = .dyn },
    .{ .name = "control: ricursive_dflash2 tf_rcd_conv rounds dyn x before its add (Qwen's form, not the fma)", .side = 1, .ctl = .two_round },
    .{ .name = "control: ricursive_dflash2 tf_rcd_conv reads across its block's edge", .side = 1, .ctl = .no_edge },
    .{ .name = "control: ricursive_dflash2 tf_rcd_conv puts base in the fma and dyn in the rounded product", .side = 1, .ctl = .swap },
};

/// Rows x width x taps: a block, 8 streams' blocks, two, and a partial block.
const conv_served = [_]c.Shape{
    .{ .rows = 16, .n = hidden, .k = taps },
    .{ .rows = 128, .n = hidden, .k = taps },
    .{ .rows = 32, .n = hidden, .k = taps },
    .{ .rows = 7, .n = hidden, .k = taps },
};

pub fn convSpec(e: ConvEntry) c.Spec {
    const rows: c.Tile = .{ .size = block, .step = 1, .min = 1 }; // a block's edge
    const chans: c.Tile = .{ .size = 256, .step = group, .min = group }; // a launch block's channels, a group at a time
    return .{ .name = e.name, .accum = .none, .depth = ref.noSums, .out = .bf16, .tiles = .{ rows, chans, .{} }, .served = &conv_served };
}

/// tf_rcd_conv: x in [-4, 4), dyn in [-0.5, 0.5), base in [-1.5, 1.5); non-finite: x's rows 1-3, row 4's first dyn.
pub const Conv = struct {
    f: Function,
    entry: ConvEntry,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    dyn: []u16 = &.{},
    base: []u16 = &.{},
    p: df.Conv = undefined,

    pub fn spec(self: *const Conv) c.Spec {
        return convSpec(self.entry);
    }

    pub fn control(self: *const Conv) ?g.Gate {
        return if (self.entry.ctl != null) .shapes else null;
    }

    pub fn free(self: *Conv, a: std.mem.Allocator) void {
        for ([_]*[]u16{ &self.x, &self.dyn, &self.base }) |x| {
            a.free(x.*);
            x.* = &.{};
        }
    }

    pub fn prepare(self: *Conv, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % group != 0 or shape.k != taps) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        const groups = n / group;
        self.x = try ref.bfs(a, rng, rows * n, -4, 4);
        self.dyn = try ref.bfs(a, rng, rows * 2 * taps * groups, -0.5, 0.5);
        self.base = try ref.bfs(a, rng, 2 * taps * n, -1.5, 1.5);
        if (fill == .nonfinite) {
            for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 1..) |v, r| if (r < rows) {
                self.x[r * n + (4 * r + 1) % n] = c.toBf16(v);
            };
            if (rows > 4) self.dyn[4 * 2 * taps * groups + self.entry.side * taps * groups] = c.toBf16(ref.nan);
        }
        self.p = .{ .x = try ref.input(rig, self.x), .dyn = try ref.input(rig, self.dyn), .base = try ref.input(rig, self.base), .y = try rig.output(rows * n * 2), .rows = rows, .width = n, .block = block, .groups = groups, .side = self.entry.side, .taps = taps };
        return true;
    }

    pub fn run(self: *Conv, rig: *rig_.Rig) !void {
        try df.convBy(self.f, rig.s, self.p);
    }

    pub fn count(self: *const Conv) usize {
        return self.shape.rows * self.shape.n;
    }

    /// bf16 products are exact in fp32: base x's one rounding is its bf16, and the fma's is the add's.
    pub fn solve(self: *const Conv, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const groups = n / group;
        const side = self.entry.side;
        for (0..self.shape.rows) |r| for (0..n) |ch| {
            const dyn = self.dyn[r * 2 * taps * groups + side * taps * groups ..];
            var y = Iv.of(0);
            for (0..taps) |t| {
                if (r % block < t) break;
                const x = ref.bf(self.x[(r - t) * n + ch]);
                const b = ref.bf(self.base[(side * taps + t) * n + ch]);
                const d = ref.bf(dyn[t * groups + ch / group]);
                y = y.add(Iv.of(b * x).bf()).bf();
                y = Iv.of(d * x).add(y).bf();
            }
            out[r * n + ch] = y;
        };
    }
};

const silu_nan = "a NaN gate or up gives NaN; a -Inf gate gives NaN (-Inf x sigmoid 0); +Inf x up 0 gives NaN";

pub fn siluSpec(control: bool) c.Spec {
    const name = if (control) "control: cuda ricursive_dflash2 tf_rcd_silu without bf16(g sigmoid(g))" else "cuda ricursive_dflash2 tf_rcd_silu (stacked gate | up, no clamp)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .nan = .{ .stated = silu_nan }, .tiles = .{ .{}, elementwise, .{} }, .served = &silu_rows };
}

/// bf16(bf16(g s) u) from stacked gate | up rows, s from e = exp(-|g|): 1 / (1 + e), times e below 0.
pub const Silu = struct {
    f: Function,
    ctl: bool = false,
    shape: c.Shape = undefined,
    gate: []u16 = &.{},
    up: []u16 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    pub fn spec(self: *const Silu) c.Spec {
        return siluSpec(self.ctl);
    }

    pub fn control(self: *const Silu) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Silu, a: std.mem.Allocator) void {
        a.free(self.gate);
        a.free(self.up);
        self.gate = &.{};
        self.up = &.{};
    }

    /// Gate in [-24, 24), up in [-4, 4); non-finite: NaN, +Inf and -Inf in gate's rows 0-2, then in up's rows 3-5.
    pub fn prepare(self: *Silu, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        self.gate = try ref.bfs(a, rng, rows * n, -24, 24);
        self.up = try ref.bfs(a, rng, rows * n, -4, 4);
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

    pub fn run(self: *Silu, rig: *rig_.Rig) !void {
        try df.siluBy(self.f, rig.s, self.dev, self.out, self.shape.rows, self.shape.n);
    }

    pub fn count(self: *const Silu) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Silu, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.gate, self.up) |*o, gh, uh| {
            const gv = ref.bf(gh);
            const e = Iv.of(-@abs(gv)).exp();
            const y = Iv.of(1).div(Iv.of(1).add(e));
            const s = if (gv < 0) e.mul(y) else y;
            o.* = Iv.of(gv).mul(s).bf().mul(Iv.of(ref.bf(uh))).bf();
        }
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    rk: rc.kernels.Kernels, // tf_rc_embed's image
    k: Kernels,
    embeds: [2]ref.Gated(Embed) = undefined,
    norms: [2]ref.Gated(AddNorm) = undefined,
    norm_ctls: [2]ref.Gated(AddNorm) = undefined,
    preps: [prep_entries.len]ref.Gated(Prep) = undefined,
    convs: [conv_entries.len]ref.Gated(Conv) = undefined,
    silus: [2]ref.Gated(Silu) = undefined,

    /// The set, its kernels appended to `list` (the controls after the kernels they break).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var rk = try rc.kernels.Kernels.load(gpu.d);
        errdefer rk.unload();
        var k = try Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .rk = rk, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const kk = &set.k;
        set.embeds = .{
            .{ .rig = rig, .k = .{ .f = set.rk.embed } },
            .{ .rig = rig, .k = .{ .f = try kk.mod.function("tf_rcd_ctl_embed_own_row"), .ctl = true } },
        };
        for (&set.norms, std.enums.values(NormForm)) |*x, f| x.* = .{ .rig = rig, .k = .{ .f = kk.add_norm, .form = f } };
        for (&set.norm_ctls, std.enums.values(NormPoint)) |*x, p| x.* = .{ .rig = rig, .k = .{ .f = try normControl(kk, p), .form = .add, .drop = p } };
        for (&set.preps, prep_entries) |*x, e| x.* = .{ .rig = rig, .k = .{ .f = if (e.ctl) |ct| try prepControl(kk, ct) else kk.prep, .entry = e } };
        for (&set.convs, conv_entries) |*x, e| x.* = .{ .rig = rig, .k = .{ .f = if (e.ctl) |ct| try convControl(kk, ct) else kk.conv, .entry = e } };
        set.silus = .{
            .{ .rig = rig, .k = .{ .f = kk.silu } },
            .{ .rig = rig, .k = .{ .f = try kk.mod.function("tf_rcd_ctl_silu_gs"), .ctl = true } },
        };
        for (&set.embeds) |*x| try list.append(a, x.kernel());
        for (&set.norms) |*x| try list.append(a, x.kernel());
        for (&set.norm_ctls) |*x| try list.append(a, x.kernel());
        for (&set.preps) |*x| try list.append(a, x.kernel());
        for (&set.convs) |*x| try list.append(a, x.kernel());
        for (&set.silus) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.embeds) |*x| x.deinit();
        for (&set.norms) |*x| x.deinit();
        for (&set.norm_ctls) |*x| x.deinit();
        for (&set.preps) |*x| x.deinit();
        for (&set.convs) |*x| x.deinit();
        for (&set.silus) |*x| x.deinit();
        set.k.unload();
        set.rk.unload();
        set.gpa.destroy(set);
    }
};

/// The drafter glue's gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x726364, &t);
    std.debug.print("SUMMARY ricursive dflash2: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "ricursive-dflash2-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "dflash2 controls: each prep and conv control registered once" {
    var prep = std.EnumSet(PrepCtl).initEmpty();
    for (prep_entries) |e| if (e.ctl) |x| {
        try std.testing.expect(!prep.contains(x));
        prep.insert(x);
    };
    try std.testing.expectEqual(std.enums.values(PrepCtl).len, prep.count());
    var conv = std.EnumSet(ConvCtl).initEmpty();
    for (conv_entries) |e| if (e.ctl) |x| {
        try std.testing.expect(!conv.contains(x));
        conv.insert(x);
    };
    try std.testing.expectEqual(std.enums.values(ConvCtl).len, conv.count());
}
