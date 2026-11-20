//! ZONOS3 Mini's Metal kernels under the contract, CUDA's tests with Metal launches: norms, linears here, rest grouped.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ref = @import("metal_zonos3_ref.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const Dense = contract.dense.Dense;
const Rig = ref.Rig;

/// bf16 bit patterns widened to the fp32 values a kernel's rows hold.
fn widen(a: std.mem.Allocator, bits: []const u16) ![]f32 {
    const out = try a.alloc(f32, bits.len);
    for (out, bits) |*o, b| o.* = c.fromBf16(b);
    return out;
}

/// ceil(k / 256) squares a thread, then a simdgroup's butterfly and the threadgroup's butterfly over 8 simdgroups.
fn normDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

pub const Which = enum { rmsnorm, layernorm };

pub fn normSpec(which: Which) c.Spec {
    return switch (which) {
        // the backbone's norms at its width, the prompt's rows a pass
        .rmsnorm => .{ .name = "metal zonos3 z3_rmsnorm (fp32 rows, bf16 gamma)", .accum = .f32, .depth = normDepth, .ops = 8, .out = .f32, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 1, .n = 1024, .k = 1024 }, .{ .rows = 64, .n = 1024, .k = 1024 } } },
        // the codec's stages (768, 1280), the teacher (1024) and its features (160)
        .layernorm => .{ .name = "metal zonos3 zc_layernorm (fp32 rows)", .accum = .f32, .depth = normDepth, .ops = 10, .out = .f32, .tiles = .{ .{}, .{}, .{ .size = 256, .step = 32, .min = 32 } }, .served = &.{ .{ .rows = 300, .n = 1280, .k = 1280 }, .{ .rows = 999, .n = 1024, .k = 1024 }, .{ .rows = 999, .n = 160, .k = 160 }, .{ .rows = 1024, .n = 768, .k = 768 } } },
    };
}

/// A norm over fp32 rows: z3_rmsnorm with its bf16 weight, or zc_layernorm with fp32 weight and bias.
pub const Norm = struct {
    rig: Rig,
    which: Which,
    pipe: mtl.Pipeline,
    /// control: the launch's epsilon is one more than the reference's
    fault: bool = false,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    b: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        var spec = normSpec(self.which);
        if (self.fault) spec.name = if (self.which == .rmsnorm) "control: metal zonos3 z3_rmsnorm with its epsilon one too large" else "control: metal zonos3 zc_layernorm with its epsilon one too large";
        return .{ .spec = spec, .ctx = self, .vt = &vt, .control = if (self.fault) .shapes else null };
    }

    const vt = g.hooks(Norm);

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != shape.k or shape.k % 32 != 0) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const a = self.rig.gpa();
        const r = try rows.Rows.init(a, if (self.which == .rmsnorm) .rms else .layer_norm, shape);
        self.r = r;
        r.fill(fill, seed);
        const x = try widen(a, r.x);
        defer a.free(x);
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        if (self.which == .rmsnorm) {
            self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        } else {
            const w = try widen(a, r.w);
            defer a.free(w);
            const b = try widen(a, r.b);
            defer a.free(b);
            self.w = try self.rig.input(std.mem.sliceAsBytes(w));
            self.b = try self.rig.input(std.mem.sliceAsBytes(b));
        }
        self.out = try self.rig.output(r.outputs() * 4);
        return true;
    }

    pub fn run(self: *Norm) !void {
        const r = self.r.?;
        const k = ref.int(r.shape.k);
        const eps = if (self.fault) r.eps + 1 else r.eps;
        const groups = [3]usize{ r.shape.rows, 1, 1 };
        if (self.which == .rmsnorm) {
            const A = extern struct { xs: i32, ys: i32, D: i32, has_w: i32, eps: f32 };
            try self.rig.go(self.pipe, groups, 256, &.{ self.x, self.w, self.out }, A{ .xs = k, .ys = k, .D = k, .has_w = 1, .eps = eps }, 0);
        } else {
            const A = extern struct { D: i32, eps: f32 };
            try self.rig.go(self.pipe, groups, 256, &.{ self.x, self.w, self.b, self.out }, A{ .D = k, .eps = eps }, 0);
        }
    }

    pub fn count(self: *Norm) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }

    /// The layer norm's scaled value rounds in fp32 before the bias joins it: those roundings are of it, not the sum.
    pub fn bounds(self: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        if (self.which != .layernorm) return;
        const r = self.r.?;
        const k = r.shape.k;
        for (allow, want, 0..) |*l, w, i| l.* += normSpec(.layernorm).ops * 0x1p-24 * @abs(w - c.fromBf16(r.b[i % k]));
    }
};

/// z3_linear: a lane's 8-fma chains over every 32nd group of 8 inputs; zc_linear: every 32nd input; then a butterfly.
fn bf16Depth(s: c.Shape) usize {
    return 8 * ((s.k + 255) / 256) + 5;
}

fn f32Depth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

pub const Weights = enum { bf16, f32 };

pub fn linearSpec(weights: Weights) c.Spec {
    return switch (weights) {
        // the backbone's projections (GDN, dense feed-forward, expert, head) and the SSL teacher's feed-forward
        .bf16 => .{ .name = "metal zonos3 z3_linear (fp32 rows, bf16 weights)", .accum = .f32, .depth = bf16Depth, .out = .f32, .tiles = .{ .{}, .{ .size = 8 }, .{ .size = 256, .step = 8, .min = 8 } }, .served = &.{ .{ .rows = 1, .n = 2056, .k = 1024 }, .{ .rows = 64, .n = 1024, .k = 2048 }, .{ .rows = 2, .n = 3072, .k = 256 }, .{ .rows = 1, .n = 40980, .k = 1024 }, .{ .rows = 16, .n = 1024, .k = 4096 }, .{ .rows = 16, .n = 4096, .k = 1024 } } },
        // the codec's: a stage's attention and feed-forward, the encoder's first projection, an iSTFT branch
        .f32 => .{ .name = "metal zonos3 zc_linear (fp32 rows and weights)", .accum = .f32, .depth = f32Depth, .out = .f32, .tiles = .{ .{}, .{ .size = 8 }, .{ .size = 32 } }, .served = &.{ .{ .rows = 16, .n = 2304, .k = 768 }, .{ .rows = 8, .n = 1280, .k = 5120 }, .{ .rows = 8, .n = 768, .k = 5888 }, .{ .rows = 8, .n = 2208, .k = 2210 } } },
    };
}

/// The rows a linear stages in threadgroup memory: at most 8,192 floats (32 KiB).
const max_k = 8192;

/// y [rows, n] = x [rows, k] W^T with fp32 sums: z3_linear over bf16 weights, or zc_linear over fp32 ones.
pub const Linear = struct {
    rig: Rig,
    weights: Weights,
    pipe: mtl.Pipeline,
    /// control: the launch is one threadgroup of eight outputs short
    fault: bool = false,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Linear) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Linear) g.Kernel {
        var spec = linearSpec(self.weights);
        if (self.fault) spec.name = if (self.weights == .bf16) "control: metal zonos3 z3_linear launched one threadgroup short" else "control: metal zonos3 zc_linear launched one threadgroup short";
        return .{ .spec = spec, .ctx = self, .vt = &vt, .control = if (self.fault) .shapes else null };
    }

    const vt = g.hooks(Linear);

    pub fn prepare(self: *Linear, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // z3_linear reads eight weights a load
        if (shape.k == 0 or shape.n == 0 or shape.rows == 0 or shape.k > max_k or shape.k % 2 != 0) return false;
        if (self.weights == .bf16 and shape.k % 8 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpa();
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5a);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        const x = try widen(a, d.x);
        defer a.free(x);
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        if (self.weights == .bf16) {
            self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        } else {
            const w = try widen(a, bits);
            defer a.free(w);
            self.w = try self.rig.input(std.mem.sliceAsBytes(w));
        }
        self.out = try self.rig.output(d.outputs() * 4);
        return true;
    }

    pub fn run(self: *Linear) !void {
        const s = self.d.?.shape;
        const k = ref.int(s.k);
        const n = ref.int(s.n);
        const groups = [3]usize{ (s.n + 7) / 8 - ref.one(self.fault), s.rows, 1 };
        if (self.weights == .bf16) {
            const A = extern struct { xs: i32, xrep: i32, K: i32, N: i32, ys: i32, has_bias: i32, has_picks: i32, pad: i32 = 0, pick_stride: i64 };
            try self.rig.go(self.pipe, groups, 256, &.{ self.w, self.x, self.out, self.w, self.x }, A{ .xs = k, .xrep = 1, .K = k, .N = n, .ys = n, .has_bias = 0, .has_picks = 0, .pick_stride = 0 }, s.k * 4);
        } else {
            const A = extern struct { xs: i32, K: i32, N: i32, ys: i32, has_bias: i32 };
            try self.rig.go(self.pipe, groups, 256, &.{ self.w, self.x, self.out, self.w }, A{ .xs = k, .K = k, .N = n, .ys = n, .has_bias = 0 }, s.k * 4);
        }
    }

    pub fn count(self: *Linear) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Linear, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Linear, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};

const parts = .{ @import("metal_zonos3_point.zig"), @import("metal_zonos3_backbone.zig"), @import("metal_zonos3_attn.zig"), @import("metal_zonos3_windows.zig") };

/// The family's registrations: each kernel, then the control planted for it.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    libs: ref.Libs = undefined,
    norms: [4]Norm = undefined,
    linears: [4]Linear = undefined,
    point: parts[0].Set = undefined,
    backbone: parts[1].Set = undefined,
    attn: parts[2].Set = undefined,
    windows: parts[3].Set = undefined,
    made: bool = false,

    /// Every registration whose name holds `only` (all when null), compiled and appended to `kernels`.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        s.libs = try ref.Libs.init(s.a, s.device);
        s.made = true;
        const rig = Rig.init(s.a, s.device, s.queue);
        for (0..4) |i| {
            const codec = i >= 2;
            const fault = i % 2 == 1;
            s.norms[i] = .{ .rig = rig, .which = if (codec) .layernorm else .rmsnorm, .pipe = try s.libs.pipe(if (codec) .codec else .backbone, if (codec) "zc_layernorm" else "z3_rmsnorm"), .fault = fault };
            s.linears[i] = .{ .rig = rig, .weights = if (codec) .f32 else .bf16, .pipe = try s.libs.pipe(if (codec) .codec else .backbone, if (codec) "zc_linear" else "z3_linear"), .fault = fault };
        }
        const list: ref.List = .{ .a = s.a, .kernels = s.kernels, .only = only };
        for (&s.norms) |*x| try list.add(x.kernel());
        for (&s.linears) |*x| try list.add(x.kernel());
        try s.point.init(&s.libs, rig, list);
        try s.backbone.init(&s.libs, rig, list);
        try s.attn.init(&s.libs, rig, list);
        try s.windows.init(&s.libs, rig, list);
    }

    pub fn deinit(s: *Suite) void {
        if (!s.made) return;
        s.windows.deinit();
        s.attn.deinit();
        s.backbone.deinit();
        s.point.deinit();
        for (&s.linears) |*x| x.deinit();
        for (&s.norms) |*x| x.deinit();
        s.libs.deinit();
    }
};

/// Gate 5's text for "zonos3 <file> <function>": the file's library with z3_common.h inlined, the same before the M5.
pub fn limitsText(a: std.mem.Allocator, name: []const u8) ![]u8 {
    var it = std.mem.tokenizeScalar(u8, name, ' ');
    _ = it.next();
    const file = std.meta.stringToEnum(ref.File, it.next() orelse return error.UnknownLaunch) orelse return error.UnknownLaunch;
    return ref.source(a, file);
}
