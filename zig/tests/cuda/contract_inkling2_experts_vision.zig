//! Inkling 2's tower glue under the contract: patch rows to the sequence, axial rotary (vision.cu), layer-scaled adds.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Kernels = ink.vision.Kernels;

const width = ink.vision.width;
const qkv_width = 3 * width;
const pairs = ink.vision.head_dim / 2;
const grid = 1024; // the position table's rows (32 x 32)

/// Sequence rows R = 4 T + 1 (the class row, then four patches a token): 1, 4, 256, 1,024 and 4,096 tokens.
fn tower(comptime n: usize, comptime k: usize) [5]c.Shape {
    var out: [5]c.Shape = undefined;
    for (&out, [_]usize{ 17, 5, 1025, 4097, 16385 }) |*s, r| s.* = .{ .rows = r, .n = n, .k = k };
    return out;
}

const embed_served = tower(width, 4);
const rope_served = tower(qkv_width, 1);
const add_served = tower(width, 1);

fn four(_: c.Shape) usize {
    return 4;
}

fn launch(f: cuda.Function, s: cuda.Stream, blocks: usize, args: []const u64, tail: ?i64) !void {
    var a: cuda.Args = .{};
    for (args) |v| a.add(v);
    if (tail) |t| a.add(t);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 256 } }, s, &a);
}

/// Row 0 bf16(cls + cls_pos); row r bf16(pe + bf16(sum of four table rows times their bilinear weights, in fp32)).
pub const Embed = struct {
    kern: *const Kernels,
    late: bool = false, // control: each weight read one tap late
    shape: c.Shape = undefined,
    pe: []u16 = &.{},
    table: []u16 = &.{},
    taps: []u32 = &.{},
    wts: []f32 = &.{},
    cls: []u16 = &.{},
    pos: []u16 = &.{},
    dev: [6]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Embed) c.Spec {
        const name = if (self.late) "inkling2 experts control: tf_ink2v_embed reading each weight one tap late" else "inkling2 experts vision tf_ink2v_embed (class row, patches plus resampled positions)";
        return .{ .name = name, .accum = .f32, .depth = four, .out = .bf16, .served = &embed_served, .sums = .{ .f16 = 0, .bf16 = 0, .why = "four taps a value, rounded to bf16: no sum long enough to tell fp32 from half" } };
    }

    pub fn control(self: *const Embed) ?g.Gate {
        return if (self.late) .shapes else null;
    }

    pub fn free(self: *Embed, a: std.mem.Allocator) void {
        inline for (.{ "pe", "table", "taps", "wts", "cls", "pos" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
    }

    pub fn prepare(self: *Embed, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows < 2 or shape.n != width or shape.k != 4) return false;
        self.free(a);
        self.shape = shape;
        const p = shape.rows - 1;
        const stress = fill == .stress;
        self.pe = try ref.bfs(a, rng, p * width, -2, 2);
        self.table = try ref.bfs(a, rng, grid * width, -1, 1);
        self.cls = try ref.bfs(a, rng, width, -1, 1);
        self.pos = try ref.bfs(a, rng, width, -1, 1);
        self.taps = try a.alloc(u32, p * 4);
        self.wts = try a.alloc(f32, p * 4);
        for (0..p) |i| {
            const x = rng.float(f32);
            const y = rng.float(f32);
            self.wts[4 * i ..][0..4].* = if (stress) .{ ref.int(rng, 0, 3), ref.int(rng, 0, 3), ref.int(rng, 0, 3), ref.int(rng, 0, 3) } else .{ (1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y };
            for (self.taps[4 * i ..][0..4]) |*t| t.* = rng.uintLessThan(u32, grid);
        }
        if (stress) for ([_][]u16{ self.pe, self.table, self.cls, self.pos }) |v| {
            for (v) |*e| e.* = c.toBf16(ref.int(rng, -8, 8));
        };
        if (fill == .nonfinite) {
            self.pe[3] = c.toBf16(ref.nan); // patch 0
            self.table[self.taps[4] * width + 5] = c.toBf16(ref.inf); // every patch that taps patch 1's first row
            self.cls[7] = c.toBf16(-ref.inf);
            if (p > 2) self.pe[2 * width + 9] = c.toBf16(-ref.inf);
        }
        self.dev = .{ try ref.input(rig, self.pe), try ref.input(rig, self.table), try ref.input(rig, self.taps), try ref.input(rig, self.wts), try ref.input(rig, self.cls), try ref.input(rig, self.pos) };
        self.out = try rig.output(shape.rows * width * 2);
        return true;
    }

    pub fn run(self: *Embed, rig: *rig_.Rig) !void {
        const d = self.dev;
        try launch(self.kern.embed, rig.s, self.shape.rows, &.{ d[0], d[1], d[2], d[3] + @as(u64, if (self.late) 4 else 0), d[4], d[5], self.out }, null);
    }

    pub fn count(self: *const Embed) usize {
        return self.shape.rows * width;
    }

    pub fn solve(self: *const Embed, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        var terms: [4]Iv = undefined;
        var scratch: [4]f64 = undefined;
        for (0..width) |e| out[e] = Iv.of(ref.bf(self.cls[e])).add(Iv.of(ref.bf(self.pos[e]))).bf();
        for (1..self.shape.rows) |r| for (0..width) |e| {
            const p = r - 1;
            for (&terms, 0..) |*t, i| t.* = Iv.of(ref.bf(self.table[self.taps[4 * p + i] * width + e])).mul(Iv.of(self.wts[4 * p + i]));
            const pos = s.sum(&terms, 4, &scratch).bf();
            out[r * width + e] = Iv.of(ref.bf(self.pe[p * width + e])).add(pos).bf();
        };
    }
};

/// q and k heads of qkv [R, 3 x 1024] turned in place, pair (2j, 2j + 1) of each head by fp32 cos and sin of row r.
pub const Rope = struct {
    kern: *const Kernels,
    swapped: bool = false, // control: cos and sin swapped
    shape: c.Shape = undefined,
    qkv: []u16 = &.{},
    cs: []f32 = &.{},
    sn: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const Rope) c.Spec {
        const name = if (self.swapped) "inkling2 experts control: tf_ink2v_rope with cos and sin swapped" else "inkling2 experts vision tf_ink2v_rope (2D rotary of q and k, in place)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .in_place = true, .served = &rope_served };
    }

    pub fn control(self: *const Rope) ?g.Gate {
        return if (self.swapped) .shapes else null;
    }

    pub fn free(self: *Rope, a: std.mem.Allocator) void {
        a.free(self.qkv);
        a.free(self.cs);
        a.free(self.sn);
        self.qkv = &.{};
        self.cs = &.{};
        self.sn = &.{};
    }

    pub fn prepare(self: *Rope, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != qkv_width) return false;
        self.free(a);
        self.shape = shape;
        const r = shape.rows;
        self.qkv = try ref.bfs(a, rng, r * qkv_width, -3, 3);
        self.cs = try a.alloc(f32, r * pairs);
        self.sn = try a.alloc(f32, r * pairs);
        for (self.cs, self.sn) |*x, *y| {
            const angle = rng.float(f32) * 64;
            x.* = @cos(angle);
            y.* = @sin(angle);
        }
        if (fill == .nonfinite) {
            self.qkv[3] = c.toBf16(ref.nan); // row 0, q head 0, pair 1's second value
            self.qkv[qkv_width + width + 64 + 10] = c.toBf16(ref.inf); // row 1, k head 1
            self.qkv[2 * qkv_width + 2 * width + 5] = c.toBf16(-ref.inf); // row 2, v: as it is
            if (r > 3) self.qkv[3 * qkv_width + 63] = c.toBf16(-ref.inf);
        }
        self.dev = .{ try ref.input(rig, self.cs), try ref.input(rig, self.sn) };
        self.out = try rig.output(self.qkv.len * 2);
        return true;
    }

    pub fn run(self: *Rope, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.qkv)); // in place: every run from the same rows
        const cs = if (self.swapped) self.dev[1] else self.dev[0];
        const sn = if (self.swapped) self.dev[0] else self.dev[1];
        try launch(self.kern.rope, rig.s, self.shape.rows, &.{ self.out, cs, sn }, null);
    }

    pub fn count(self: *const Rope) usize {
        return self.qkv.len;
    }

    pub fn solve(self: *const Rope, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.qkv, 0..) |*o, x, at| {
            const col = at % qkv_width;
            if (col >= 2 * width) {
                o.* = Iv.of(ref.bf(x));
                continue;
            }
            const d = col % 64;
            const first = at - d % 2;
            const r = at / qkv_width;
            const cv = Iv.of(self.cs[r * pairs + d / 2]);
            const sv = Iv.of(self.sn[r * pairs + d / 2]);
            const av = Iv.of(ref.bf(self.qkv[first]));
            const bv = Iv.of(ref.bf(self.qkv[first + 1]));
            o.* = (if (d % 2 == 0) av.mul(cv).add(bv.neg().mul(sv)) else bv.mul(cv).add(av.mul(sv))).bf();
        }
    }
};

/// tf_vit_pointwise_gamma as the tower runs it: x = bf16(x + bf16(gamma[i % 1024] y)) over [R, 1024] in place.
pub const ScaleAdd = struct {
    kern: *const cuda.vit.Kernels,
    late: bool = false, // control: gamma read one column late
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    y: []u16 = &.{},
    gamma: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const ScaleAdd) c.Spec {
        const name = if (self.late) "inkling2 experts control: tf_vit_pointwise_gamma reading gamma one column late" else "inkling2 vit tf_vit_pointwise_gamma (layer-scaled residual add, in place)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .in_place = true, .served = &add_served };
    }

    pub fn control(self: *const ScaleAdd) ?g.Gate {
        return if (self.late) .shapes else null;
    }

    pub fn free(self: *ScaleAdd, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.y);
        a.free(self.gamma);
        self.x = &.{};
        self.y = &.{};
        self.gamma = &.{};
    }

    pub fn prepare(self: *ScaleAdd, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != width) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * width;
        self.x = try ref.bfs(a, rng, n, -4, 4);
        self.y = try ref.bfs(a, rng, n, -4, 4);
        self.gamma = try ref.bfs(a, rng, width, -0.5, 0.5);
        if (fill == .nonfinite) {
            self.x[0] = c.toBf16(ref.nan);
            self.y[1] = c.toBf16(ref.inf);
            self.gamma[2] = c.toBf16(-ref.inf); // column 2 of every row
            self.x[width + 3] = c.toBf16(ref.inf);
        }
        self.dev = .{ try ref.input(rig, self.y), try ref.input(rig, self.gamma) };
        self.out = try rig.output(n * 2);
        return true;
    }

    pub fn run(self: *ScaleAdd, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.x));
        const n = self.count();
        try self.kern.pointwiseGamma(rig.s, self.dev[0], self.out, self.out, n, .none, self.dev[1] + @as(u64, if (self.late) 2 else 0), width);
    }

    pub fn count(self: *const ScaleAdd) usize {
        return self.shape.rows * width;
    }

    pub fn solve(self: *const ScaleAdd, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.x, self.y, 0..) |*o, x, y, i| {
            const scaled = Iv.of(ref.bf(self.gamma[i % width])).mul(Iv.of(ref.bf(y))).bf();
            o.* = Iv.of(ref.bf(x)).add(scaled).bf();
        }
    }
};
