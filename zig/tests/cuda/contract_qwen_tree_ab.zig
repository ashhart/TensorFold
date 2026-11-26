//! Raw identity A/B coverage for Qwen3.6 MTP tree attention at its actual CUDA ABI.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const heads = 16;
const kv_heads = 2;
const hd = 256;
const width = kv_heads * hd;
const gstride = heads * 2 * hd;
const lanes = 64;
const anc_stride = lanes + 1;
const nan_bits: u16 = 0x7fc0;

/// k is live ring slots and n is heads times head_dim; cases cover root/no-ring, prefetch and long context.
const served = [_]c.Shape{
    .{ .rows = 1, .n = heads * hd, .k = 0 },
    .{ .rows = 1, .n = heads * hd, .k = 6 },
    .{ .rows = 15, .n = heads * hd, .k = 6 },
    .{ .rows = 16, .n = heads * hd, .k = 6 },
    .{ .rows = 17, .n = heads * hd, .k = 6 },
    .{ .rows = 64, .n = heads * hd, .k = 6 },
    .{ .rows = 4, .n = heads * hd, .k = 9 },
    .{ .rows = 4, .n = heads * hd, .k = 4200 },
};

const Form = struct { wrapped: bool, control: bool = false, name: []const u8 };
const forms = [_]Form{
    .{ .wrapped = false, .name = "cuda qwen3.6 mtp tree attention tf_q36m_attn_tree, contiguous ring" },
    .{ .wrapped = true, .name = "cuda qwen3.6 mtp tree attention tf_q36m_attn_tree, wrapping ring" },
    .{ .wrapped = false, .control = true, .name = "control: qwen tree attention dropping row zero's last ancestor" },
    .{ .wrapped = true, .control = true, .name = "control: wrapping qwen tree attention dropping row zero's last ancestor" },
};

fn depth(s: c.Shape) usize {
    return 2 * (s.k + lanes + 1) + 16;
}

fn spec(form: Form) c.Spec {
    return .{
        .name = form.name,
        .accum = .f32,
        .depth = depth,
        .ops = 2,
        .mids = 1, // o / l is rounded to bf16 before the gate product
        .out = .bf16,
        .served = &served,
        .sums = .{ .why = attn.why },
    };
}

const Tree = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    form: Form,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    cap: usize = 0,
    begin: usize = 0,
    q: []u16 = &.{},
    gate: []u16 = &.{},
    kb: []u16 = &.{},
    vb: []u16 = &.{},
    lk: []u16 = &.{},
    lv: []u16 = &.{},
    rk: []u16 = &.{},
    rv: []u16 = &.{},
    anc: []i32 = &.{},
    dev: [9]u64 = @splat(0),
    out: u64 = 0,

    fn kernel(self: *Tree) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }
    const vt = g.hooks(Tree);

    fn deinit(self: *Tree) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    /// Controls drop one ancestor while row zero still expects two, so the shapes gate genuinely fails.
    fn ancestors(self: *const Tree, row: usize) usize {
        const n: usize = @intCast(self.anc[row * anc_stride]);
        return if (self.form.control and row == 0) 2 else n;
    }

    fn terms(self: *const Tree, row: usize) usize {
        return self.shape.k + self.ancestors(row) + 1;
    }

    /// Ring terms first, then root-to-parent ancestors from LK/LV, then this row's KB/VB.
    fn term(self: *Tree, row: usize, j: usize, kvh: usize, value_side: bool) []u16 {
        if (j < self.shape.k) {
            const at = ((self.begin + j) % self.cap) * width + kvh * hd;
            return (if (value_side) self.rv else self.rk)[at..][0..hd];
        }
        const path = j - self.shape.k;
        if (path < self.ancestors(row)) {
            const lane: usize = @intCast(self.anc[row * anc_stride + 1 + path]);
            return (if (value_side) self.lv else self.lk)[lane * width + kvh * hd ..][0..hd];
        }
        return (if (value_side) self.vb else self.kb)[row * width + kvh * hd ..][0..hd];
    }

    fn prepare(self: *Tree, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > lanes or shape.n != heads * hd or shape.k > 4200) return false;
        self.rig.reset();
        _ = self.arena.reset(.free_all);
        const a = self.arena.allocator();
        self.shape = shape;
        self.fill = fill;
        self.cap = if (self.form.wrapped) 32768 else @max(7, shape.k + 1);
        self.begin = if (self.form.wrapped) self.cap - 3 else 0;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        self.q = try a.alloc(u16, rows * heads * hd);
        self.gate = try a.alloc(u16, rows * gstride);
        self.kb = try a.alloc(u16, rows * width);
        self.vb = try a.alloc(u16, rows * width);
        self.lk = try a.alloc(u16, lanes * width);
        self.lv = try a.alloc(u16, lanes * width);
        self.rk = try a.alloc(u16, self.cap * width);
        self.rv = try a.alloc(u16, self.cap * width);
        self.anc = try a.alloc(i32, rows * anc_stride);
        @memset(self.anc, -1);

        // Paths cover roots, variable depths and a two-ancestor row zero; unnamed lanes stay NaN.
        for (0..rows) |row| {
            const n = if (row == 0) @as(usize, 2) else @min(row, row % 4);
            self.anc[row * anc_stride] = @intCast(n);
            const paths = [_][3]i32{ .{ 3, 1, -1 }, .{ 2, -1, -1 }, .{ 3, 1, -1 }, .{ 4, 3, 1 } };
            const selected = paths[if (row == 0) 0 else row % 4];
            for (0..n) |j| self.anc[row * anc_stride + 1 + j] = selected[j];
        }
        if (self.form.control) self.anc[0] = 1;

        for (self.q) |*x| x.* = if (fill == .stress) 0 else number(rng, 0.5);
        for (self.gate) |*x| x.* = if (fill == .stress) 0 else number(rng, 4);
        for (self.kb) |*x| x.* = if (fill == .stress) 0 else number(rng, 0.5);
        @memset(self.lk, nan_bits);
        @memset(self.lv, nan_bits);
        @memset(self.rk, nan_bits);
        @memset(self.rv, nan_bits);
        for (0..shape.k) |j| {
            const at = ((self.begin + j) % self.cap) * width;
            for (0..width) |i| self.rk[at + i] = if (fill == .stress) 0 else number(rng, 0.5);
            self.values(self.rv[at..][0..width], j, rng);
        }
        for (0..rows) |row| for (0..self.ancestors(row)) |j| {
            const lane: usize = @intCast(self.anc[row * anc_stride + 1 + j]);
            for (0..width) |i| self.lk[lane * width + i] = if (fill == .stress) 0 else number(rng, 0.5);
            self.values(self.lv[lane * width ..][0..width], shape.k + j, rng);
        };
        // Own keys are the final term, after all ancestors.
        for (0..rows) |row| self.values(self.vb[row * width ..][0..width], shape.k + self.ancestors(row), rng);

        if (fill == .nonfinite) self.plantNonfinite();
        if (fill == .random) self.plantAdversarial();

        for ([_][]const u8{
            std.mem.sliceAsBytes(self.q),   std.mem.sliceAsBytes(self.gate),
            std.mem.sliceAsBytes(self.kb),  std.mem.sliceAsBytes(self.vb),
            std.mem.sliceAsBytes(self.lk),  std.mem.sliceAsBytes(self.lv),
            std.mem.sliceAsBytes(self.anc), std.mem.sliceAsBytes(self.rk),
            std.mem.sliceAsBytes(self.rv),
        }, &self.dev) |bytes, *p| p.* = try self.rig.input(bytes);
        self.out = try self.rig.output(rows * heads * hd * 2);
        return true;
    }

    fn number(rng: std.Random, scale: f32) u16 {
        return c.toBf16((rng.float(f32) - 0.5) * scale);
    }

    /// Stress values make fp32 exact under every reduction order but lower-precision sums red.
    fn values(self: *const Tree, out: []u16, j: usize, rng: std.Random) void {
        const stress = self.fill == .stress;
        for (out, 0..) |*x, i| {
            x.* = if (stress) c.toBf16(([3]f32{ 4096, 1, -4096 })[(j + i / hd) % 3]) else number(rng, 0.5);
        }
    }

    fn plantNonfinite(self: *Tree) void {
        self.q[hd + 3] = nan_bits; // row zero, query head one
        if (self.shape.k > 0) {
            const at = (self.begin % self.cap) * width;
            self.rk[at] = c.toBf16(std.math.inf(f32));
            self.rv[at + 7] = c.toBf16(std.math.inf(f32));
        }
        self.lk[3 * width + hd + 5] = nan_bits;
        self.lv[3 * width + hd + 9] = c.toBf16(-std.math.inf(f32));
        self.vb[hd + 11] = c.toBf16(std.math.inf(f32));
    }

    /// A score/value pair with exact signed-zero and representable-subnormal inputs.
    fn pair(self: *Tree, j: usize, score: f32, value: f32, odd: bool) void {
        const key = self.term(0, j, 0, false);
        const val = self.term(0, j, 0, true);
        key[0] = c.toBf16(score);
        key[1] = if (odd) 0 else 0x8000;
        key[2] = if (odd) 0x8001 else 0x0001;
        key[3] = if (odd) 0x007f else 0x807f;
        val[0] = c.toBf16(value);
        val[1] = if (odd) 0 else 0x8000;
        val[2] = if (odd) 0x8001 else 0x0001;
        val[3] = if (odd) 0x8001 else 0x007f;
        val[4] = if (odd) 0x807f else 0x007f;
    }

    /// Ring maxima tie across the prefetch seam; ancestors change then tie the maximum, and the own key always runs.
    fn plantAdversarial(self: *Tree) void {
        const q = self.q[0..hd];
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        for (0..self.terms(0)) |j| self.term(0, j, 0, false)[0] = 0;
        if (self.shape.k > 0) {
            self.pair(0, 16, 1, false);
            self.pair(if (self.shape.k > 8) 8 else self.shape.k - 1, 16, 3, true);
        }
        const own = self.terms(0) - 1;
        if (self.ancestors(0) >= 2) {
            self.pair(self.shape.k, 16, 5, false);
            self.pair(self.shape.k + 1, 32, 7, true);
            self.pair(own, 32, 9, false);
        } else self.pair(own, 32, 5, false);
        self.gate[hd] = 0x8000;
        self.gate[hd + 1] = 0x0001;
        self.gate[hd + 2] = 0x8001;
        self.gate[hd + 3] = 0x007f;
        self.gate[hd + 4] = 0x807f;
    }

    fn run(self: *Tree) !void {
        var args: cuda.Args = .{};
        args.add(self.dev[0]);
        args.add(self.dev[1]);
        args.add(@as(c_int, gstride));
        for (self.dev[2..7]) |p| args.add(p);
        args.add(@as(c_int, anc_stride));
        args.add(self.dev[7]);
        args.add(self.dev[8]);
        args.add(@as(c_int, heads));
        args.add(@as(c_int, kv_heads));
        args.add(@as(c_int, @intCast(self.cap)));
        args.add(@as(c_longlong, @intCast(self.begin)));
        args.add(@as(c_longlong, @intCast(self.begin + self.shape.k)));
        args.add(self.out);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = heads }, .block = .{ .x = 32 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    fn count(self: *const Tree) usize {
        return self.shape.rows * heads * hd;
    }

    fn reference(self: *Tree, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        for (0..self.shape.rows) |row| for (0..heads) |h| {
            const kvh = h / (heads / kv_heads);
            const scores = try a.alloc(f64, self.terms(row));
            defer a.free(scores);
            for (scores, 0..) |*score, j| {
                var dot: f64 = 0;
                for (self.q[(row * heads + h) * hd ..][0..hd], self.term(row, j, kvh, false)) |x, k| dot += c.fromBf16(x) * c.fromBf16(k);
                score.* = dot / 16;
            }
            const top = std.mem.max(f64, scores);
            var total: f64 = 0;
            for (scores) |*score| {
                score.* = @exp(score.* - top);
                total += score.*;
            }
            for (0..hd) |d| {
                var sum: f64 = 0;
                var most: f64 = 0;
                for (scores, 0..) |weight, j| {
                    const v = c.fromBf16(self.term(row, j, kvh, true)[d]);
                    sum += weight * v;
                    most = @max(most, @abs(v));
                }
                const gate = 1 / (1 + @exp(-c.fromBf16(self.gate[row * gstride + h * 2 * hd + hd + d])));
                const at = (row * heads + h) * hd + d;
                want[at] = sum / total * gate;
                mag[at] = most * gate;
            }
        };
    }

    fn simulate(self: *Tree, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        for (0..self.shape.rows) |row| {
            const weights = try a.alloc(f64, self.terms(row));
            defer a.free(weights);
            for (0..heads) |h| for (0..hd) |d| {
                for (weights, 0..) |*x, j| x.* = c.fromBf16(self.term(row, j, h / (heads / kv_heads), true)[d]);
                const sum = core.contract.rows.sumIn(acc, order, weights);
                const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(weights.len)));
                out[(row * heads + h) * hd + d] = c.fromBf16(c.toBf16(c.fromBf16(c.toBf16(mean)) * 0.5));
            };
        }
    }
};

/// Both actual ring forms and their planted mask controls, at stable heap addresses.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    items: [forms.len]Tree = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen36_mtp) };
        errdefer self.mod.unload();
        errdefer for (self.items[0..self.made]) |*item| item.deinit();
        for (forms) |form| {
            self.items[self.made] = .{
                .rig = .{ .gpu = gpu, .s = s },
                .f = try self.mod.function("tf_q36m_attn_tree"),
                .form = form,
                .arena = .init(a),
            };
            self.made += 1;
            try kernels.append(a, self.items[self.made - 1].kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*item| item.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};
