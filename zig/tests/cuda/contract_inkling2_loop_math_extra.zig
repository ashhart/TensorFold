//! Mini QKV projection, raw AR and BF16 tap rounding, with the loop suites' shared registration.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");
const base = @import("contract_inkling2_loop_math.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const lm = ink.loop_math_cuda;
const K = base.Checked;
const eps = base.eps;
const mini_h = base.mini_h;
const queryScale = base.queryScale;
const softmax = base.softmax;
const arStressSource = base.arStressSource;
const close = base.close;
const bf = base.bf;
const bfRound = base.bfRound;
const cudaMax = base.cudaMax;
const sameBf = base.sameBf;
const scale = base.scale;
const reduce = base.reduce;
const fail = base.fail;
const Messages = base.Messages;
const Gate = base.Gate;
const qkv_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 3072, .k = mini_h },
    .{ .rows = 16, .n = 384, .k = 128 },
    .{ .rows = 2048, .n = 384, .k = 128 },
};

/// Distinct q/k/v rows, with the Small single-input aliasing bug planted as the control.
pub const ProjectQkv = struct {
    kern: *const lm.Kernels,
    shared_input: bool = false,
    shape: c.Shape = undefined,
    u: [3][]u16 = .{ &.{}, &.{}, &.{} },
    w: [3][]u16 = .{ &.{}, &.{}, &.{} },
    dev: [6]u64 = @splat(0),
    out: [3]u64 = @splat(0),

    pub fn spec(self: *const ProjectQkv) c.Spec {
        return .{ .name = if (self.shared_input) "control: Mini split QKV projection using Small's one normalized input for all roles" else "inkling2 loop tf_ink2_loop_project_qkv (distinct normalized q/k/v inputs, unchanged raw layouts)", .accum = .f32, .depth = depth, .out = .bf16, .served = if (self.shared_input) qkv_shapes[0..2] else &qkv_shapes };
    }

    fn depth(s: c.Shape) usize {
        return s.k;
    }

    pub fn control(self: *const ProjectQkv) ?g.Gate {
        return if (self.shared_input) .shapes else null;
    }

    pub fn free(self: *ProjectQkv, a: std.mem.Allocator) void {
        for (&self.u) |*x| a.free(x.*);
        for (&self.w) |*x| a.free(x.*);
        self.u = .{ &.{}, &.{}, &.{} };
        self.w = .{ &.{}, &.{}, &.{} };
    }

    pub fn prepare(self: *ProjectQkv, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 3 != 0 or shape.k == 0) return false;
        self.free(a);
        self.shape = shape;
        for (&self.u) |*x| x.* = try ref.bfs(a, rng, shape.rows * shape.k, -1, 1);
        for (&self.w, 0..) |*x, role| x.* = try ref.bfs(a, rng, (if (role == 0) shape.n else shape.n / 3) * shape.k, -0.1, 0.1);
        if (fill == .stress) {
            // Mirrored products cancel to eight small FP32 terms; lower precisions lose their large partials.
            const half = shape.k / 2;
            for (self.u) |bank| for (0..shape.rows) |row| {
                const values = bank[row * shape.k ..][0..shape.k];
                for (0..half) |i| {
                    values[i] = c.toBf16(ref.int(rng, 8, 16));
                    values[half + i] = values[i];
                }
                if (shape.k % 2 != 0) values[shape.k - 1] = 0;
            };
            for (self.w) |bank| for (0..bank.len / shape.k) |row| {
                const values = bank[row * shape.k ..][0..shape.k];
                for (0..half) |i| {
                    const v = ref.int(rng, 8, 16);
                    values[i] = c.toBf16(v);
                    values[half + i] = c.toBf16(-v);
                }
                for (0..@min(8, half)) |i| values[half + i] = c.toBf16(1 - bf(values[i]));
                if (shape.k % 2 != 0) values[shape.k - 1] = 0;
            };
        }
        if (fill == .nonfinite) for (&self.u) |*x| ref.plant(u16, x.*, shape.rows, shape.k, ref.asBf);
        for (self.u, 0..) |x, i| self.dev[i] = try ref.input(rig, x);
        for (self.w, 0..) |x, i| self.dev[3 + i] = try ref.input(rig, x);
        for (0..3) |role| self.out[role] = try rig.output(shape.rows * (if (role == 0) shape.n else shape.n / 3) * 2);
        return true;
    }

    pub fn run(self: *ProjectQkv, rig: *rig_.Rig) !void {
        const k = if (self.shared_input) self.dev[0] else self.dev[1];
        const v = if (self.shared_input) self.dev[0] else self.dev[2];
        try lm.projectQkv(self.kern, rig.s, self.dev[0], k, v, self.dev[3], self.dev[4], self.dev[5], self.out[0], self.out[1], self.out[2], self.shape.rows, self.shape.k, self.shape.n, self.shape.n / 3);
    }

    pub fn count(self: *const ProjectQkv) usize {
        return self.shape.rows * self.shape.n * 5 / 3;
    }

    pub fn solve(self: *const ProjectQkv, a: std.mem.Allocator, s: ref.Sums, out: []ref.Iv) !void {
        const cols = self.shape.k;
        const terms = try a.alloc(ref.Iv, cols);
        defer a.free(terms);
        const scratch = try a.alloc(f64, cols);
        defer a.free(scratch);
        var offset: usize = 0;
        for (0..3) |role| {
            const width = if (role == 0) self.shape.n else self.shape.n / 3;
            for (0..self.shape.rows) |row| for (0..width) |j| {
                for (terms, self.u[role][row * cols ..][0..cols], self.w[role][j * cols ..][0..cols]) |*t, x, w| t.* = ref.Iv.of(ref.bf(x) * ref.bf(w));
                out[offset + row * width + j] = s.sum(terms, depth(self.shape), scratch).bf();
            };
            offset += self.shape.rows * width;
        }
    }

    pub fn check(self: *const ProjectQkv, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const cols = self.shape.k;
        const x = try a.alloc(f32, cols);
        defer a.free(x);
        const w = try a.alloc(f32, cols);
        defer a.free(w);
        var byte_at: usize = 0;
        for (0..3) |role| {
            const width = if (role == 0) self.shape.n else self.shape.n / 3;
            for (0..self.shape.rows) |row| for (0..width) |j| {
                for (x, self.u[role][row * cols ..][0..cols]) |*v, u| v.* = bf(u);
                for (w, self.w[role][j * cols ..][0..cols]) |*v, u| v.* = bf(u);
                var sum: f32 = 0;
                for (0..cols) |i| sum = @mulAdd(f32, x[i], w[i], sum);
                const want = c.toBf16(sum);
                const at = byte_at + (row * width + j) * 2;
                const value = std.mem.readInt(u16, got[at..][0..2], .little);
                if (!sameBf(value, want)) return fail(msg, "role {d} row {d} output {d}: {x} where {x} is due", .{ role, row, j, value, want });
            };
            byte_at += self.shape.rows * width * 2;
        }
        return null;
    }
};

const ar_shapes = [_]c.Shape{
    .{ .rows = 1, .n = mini_h, .k = 1 },
    .{ .rows = 16, .n = 128, .k = 3 },
    .{ .rows = 2048, .n = 128, .k = 3 },
};

/// Raw AR before controllers. Two completed blocks plus a partial; control omits the second completed block.
pub const Ar = struct {
    kern: *const lm.Kernels,
    partial_bf16: bool,
    short: bool = false,
    shape: c.Shape = undefined,
    blocks: [2][]u16 = .{ &.{}, &.{} },
    partial32: []f32 = &.{},
    partial16: []u16 = &.{},
    query: []f32 = &.{},
    dev: [4]u64 = @splat(0), // two blocks, partial, query
    out: u64 = 0,

    pub fn spec(self: *const Ar) c.Spec {
        return .{ .name = if (self.short) "control: inkling2 loop raw AR omitting its second completed block" else if (self.partial_bf16) "inkling2 loop tf_ink2_loop_ar (BF16 partial, FP32 unnormalized output)" else "inkling2 loop tf_ink2_loop_ar (FP32 partial, FP32 unnormalized output)", .accum = .f32, .depth = depth, .out = .f32, .served = if (self.short) ar_shapes[1..2] else &ar_shapes };
    }

    fn depth(s: c.Shape) usize {
        return (s.n + 511) / 512 + 10 + 3;
    }

    pub fn control(self: *const Ar) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *Ar, a: std.mem.Allocator) void {
        for (&self.blocks) |*x| a.free(x.*);
        a.free(self.partial32);
        a.free(self.partial16);
        a.free(self.query);
        self.blocks = .{ &.{}, &.{} };
        self.partial32 = &.{};
        self.partial16 = &.{};
        self.query = &.{};
    }

    pub fn prepare(self: *Ar, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k > lm.max_queries) return false;
        self.free(a);
        self.shape = shape;
        for (&self.blocks, 0..) |*x, source| {
            x.* = try ref.bfs(a, rng, shape.rows * shape.n, -1, 1);
            if (fill == .stress) {
                for (x.*, 0..) |*v, i| v.* = c.toBf16(arStressSource(source, i % shape.n));
            }
        }
        if (self.partial_bf16) {
            self.partial16 = try ref.bfs(a, rng, shape.rows * shape.n, -1, 1);
            if (fill == .stress) {
                for (self.partial16, 0..) |*v, i| v.* = c.toBf16(arStressSource(2, i % shape.n));
            }
            self.dev[2] = try ref.input(rig, self.partial16);
        } else {
            self.partial32 = try ref.f32s(a, rng, shape.rows * shape.n, -1, 1);
            if (fill == .stress) {
                for (self.partial32, 0..) |*v, i| v.* = arStressSource(2, i % shape.n);
            }
            self.dev[2] = try ref.input(rig, self.partial32);
        }
        self.query = try ref.f32s(a, rng, shape.k * shape.n, -0.25, 0.25);
        if (fill == .stress) @memset(self.query, 0);
        if (fill == .nonfinite) for (&self.blocks) |*x| ref.plant(u16, x.*, shape.rows, shape.n, ref.asBf);
        for (self.blocks, 0..) |x, i| self.dev[i] = try ref.input(rig, x);
        self.dev[3] = try ref.input(rig, self.query);
        self.out = try rig.output(shape.rows * shape.n * shape.k * 4);
        return true;
    }

    pub fn run(self: *Ar, rig: *rig_.Rig) !void {
        var args: lm.ArArgs = .{ .partial = self.dev[2], .query = self.dev[3], .out = self.out, .n_blocks = if (self.short) 1 else 2, .partial_bf16 = @intFromBool(self.partial_bf16), .nq = @intCast(self.shape.k), .rows = @intCast(self.shape.rows), .c = @intCast(self.shape.n), .eps = eps };
        args.blocks[0] = self.dev[0];
        args.blocks[1] = self.dev[1];
        try lm.ar(self.kern, rig.s, args);
    }

    pub fn count(self: *const Ar) usize {
        return self.shape.rows * self.shape.n * self.shape.k;
    }

    fn entry(self: *const Ar, source: usize, row: usize, col: usize) f32 {
        const at = row * self.shape.n + col;
        return if (source < 2) bf(self.blocks[source][at]) else if (self.partial_bf16) bf(self.partial16[at]) else self.partial32[at];
    }

    pub fn solve(self: *const Ar, a: std.mem.Allocator, s: ref.Sums, out: []ref.Iv) !void {
        const cols = self.shape.n;
        const terms = try a.alloc(ref.Iv, cols);
        defer a.free(terms);
        const scratch = try a.alloc(f64, cols);
        defer a.free(scratch);
        for (0..self.shape.rows) |row| {
            var inv: [3]ref.Iv = undefined;
            for (0..3) |j| {
                for (terms, 0..) |*t, i| {
                    const x = ref.Iv.of(self.entry(j, row, i));
                    t.* = x.mul(x);
                }
                inv[j] = ref.rms(s.sum(terms, depth(self.shape), scratch), cols, eps);
            }
            for (0..self.shape.k) |q| {
                var logits: [3]ref.Iv = undefined;
                for (0..3) |j| {
                    for (terms, 0..) |*t, i| t.* = ref.Iv.of(self.entry(j, row, i)).mul(inv[j]).mul(ref.Iv.of(self.query[q * cols + i]));
                    logits[j] = s.sum(terms, depth(self.shape), scratch).mul(queryScale(cols));
                }
                softmax(&logits);
                for (0..cols) |i| {
                    var value = ref.Iv.of(0);
                    for (0..3) |j| {
                        const beta = ref.Iv.of(self.entry(j, row, i)).mul(inv[j]);
                        value = value.add(logits[j].mul(beta));
                    }
                    out[(q * self.shape.rows + row) * cols + i] = value;
                }
            }
        }
    }

    pub fn check(self: *const Ar, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const cols = self.shape.n;
        const sources = 3;
        const raw = try a.alloc(f32, cols);
        defer a.free(raw);
        const beta = try a.alloc(f32, sources * cols);
        defer a.free(beta);
        const weights = try a.alloc(f32, sources);
        defer a.free(weights);
        for (0..self.shape.rows) |row| {
            var invs: [3]f32 = undefined;
            for (0..sources) |j| {
                for (raw, 0..) |*v, i| v.* = self.entry(j, row, i);
                invs[j] = 1.0 / @sqrt(reduce(raw, raw, 512) / @as(f32, @floatFromInt(cols)) + eps);
                for (beta[j * cols ..][0..cols], raw) |*v, x| v.* = x * invs[j];
            }
            for (0..self.shape.k) |q| {
                var logits: [3]f32 = undefined;
                var m = -std.math.inf(f32);
                for (0..sources) |j| {
                    const query = self.query[q * cols ..][0..cols];
                    logits[j] = reduce(query, beta[j * cols ..][0..cols], 512) * scale(cols);
                    m = cudaMax(m, logits[j]);
                }
                var den: f32 = 0;
                for (0..sources) |j| {
                    weights[j] = @exp(logits[j] - m);
                    den += weights[j];
                }
                for (0..sources) |j| weights[j] /= den;
                for (0..cols) |i| {
                    var want: f32 = 0;
                    for (0..sources) |j| want = @mulAdd(f32, weights[j], beta[j * cols + i], want);
                    const at = (q * self.shape.rows + row) * cols + i;
                    const value = ref.readF32(got, at);
                    if (!close(value, want)) return fail(msg, "query {d} row {d} col {d}: {e} where {e} is due", .{ q, row, i, value, want });
                }
            }
        }
        return null;
    }
};

test "AR normalizes each FP32 query term before reduction and scales after its dot" {
    const source = [_]f32{
        @bitCast(@as(u32, 0x3f800008)), @bitCast(@as(u32, 0x3f800001)),
        @bitCast(@as(u32, 0x40400001)), 0,
    };
    const query = [_]f32{
        @bitCast(@as(u32, 0x3f800010)), @bitCast(@as(u32, 0x3fffffff)),
        @bitCast(@as(u32, 0xc0000001)), 0,
    };
    const inv: f32 = @bitCast(@as(u32, 0x3f800001));
    var normalized: [4]f32 = undefined;
    for (&normalized, source) |*v, x| v.* = x * inv;
    const dot = reduce(&query, &normalized, 512);
    const old = reduce(&query, &source, 512) * inv * scale(source.len);
    const scaled = dot * scale(source.len);
    try std.testing.expectEqual(@as(u32, 0xbfbffffb), @as(u32, @bitCast(scaled)));
    try std.testing.expectEqual(@as(u32, 0xbfbffffa), @as(u32, @bitCast(old)));
}

const round_shapes = [_]c.Shape{
    .{ .rows = 1, .n = mini_h, .k = 1 },
    .{ .rows = 2048, .n = 32, .k = 1 },
};

/// Dynamic taps' required BF16 storage rounding, followed by exact widening for the FP32 convolution.
pub const RoundBf16 = struct {
    kern: *const lm.Kernels,
    short: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    pub fn spec(self: *const RoundBf16) c.Spec {
        return .{ .name = if (self.short) "control: inkling2 loop dynamic taps leave the final value unrounded" else "inkling2 loop tf_ink2_loop_round_bf16 (global dynamic taps, BF16 then FP32)", .accum = .none, .depth = ref.noSums, .out = .f32, .served = if (self.short) round_shapes[0..1] else &round_shapes };
    }

    pub fn control(self: *const RoundBf16) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *RoundBf16, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }

    pub fn prepare(self: *RoundBf16, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        self.x = try ref.f32s(a, rng, shape.rows * shape.n, -32, 32);
        if (fill == .nonfinite) ref.plant(f32, self.x, shape.rows, shape.n, ref.asF32);
        self.dev = try ref.input(rig, self.x);
        self.out = try rig.output(self.x.len * 4);
        return true;
    }

    pub fn run(self: *RoundBf16, rig: *rig_.Rig) !void {
        try lm.roundBf16(self.kern, rig.s, self.dev, self.out, self.x.len - @intFromBool(self.short));
    }

    pub fn count(self: *const RoundBf16) usize {
        return self.x.len;
    }

    pub fn solve(self: *const RoundBf16, _: std.mem.Allocator, _: ref.Sums, out: []ref.Iv) !void {
        for (out, self.x) |*o, x| o.* = ref.Iv.of(x).bf();
    }

    pub fn check(self: *const RoundBf16, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        for (self.x, 0..) |x, i| {
            const want = bfRound(x);
            const value = ref.readF32(got, i);
            if (!ref.sameF32(value, want)) return fail(msg, "tap {d}: {e} where {e} is due", .{ i, value, want });
        }
        return null;
    }
};

pub const Set = struct {
    k: lm.Kernels = undefined,
    loaded: bool = false,
    messages: [2]K(Messages) = undefined,
    gates: [2]K(Gate) = undefined,
    qkvs: [2]K(ProjectQkv) = undefined,
    ars: [3]K(Ar) = undefined,
    rounds: [2]K(RoundBf16) = undefined,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try lm.Kernels.load(gpu.d);
        set.loaded = true;
        const k = &set.k;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&set.messages, 0..) |*x, i| x.* = .{ .inner = .{ .rig = rig, .k = .{ .kern = k, .aliased = i == 1 } } };
        for (&set.gates, 0..) |*x, i| x.* = .{ .inner = .{ .rig = rig, .k = .{ .kern = k, .wrong_gain = i == 1 } } };
        for (&set.qkvs, 0..) |*x, i| x.* = .{ .inner = .{ .rig = rig, .k = .{ .kern = k, .shared_input = i == 1 } } };
        for (&set.ars, 0..) |*x, i| x.* = .{ .inner = .{ .rig = rig, .k = .{ .kern = k, .partial_bf16 = i == 1, .short = i == 2 } } };
        for (&set.rounds, 0..) |*x, i| x.* = .{ .inner = .{ .rig = rig, .k = .{ .kern = k, .short = i == 1 } } };
        inline for (.{ "messages", "gates", "qkvs", "ars", "rounds" }) |f| for (&@field(set, f)) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        inline for (.{ "messages", "gates", "qkvs", "ars", "rounds" }) |f| for (&@field(set, f)) |*x| x.deinit();
        set.k.unload();
        set.loaded = false;
    }
};
