//! ZONOS3 Mini's summing kernels under the contract on CUDA: the two norms and the two linears, all fp32 rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const Dense = contract.dense.Dense;

/// bf16 bit patterns widened to the fp32 values a kernel's rows hold.
fn widen(a: std.mem.Allocator, bits: []const u16) ![]f32 {
    const out = try a.alloc(f32, bits.len);
    for (out, bits) |*o, b| o.* = c.fromBf16(b);
    return out;
}

/// ceil(k / 256) squares a thread, then a warp's butterfly and the block's butterfly over 8 warps.
fn normDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

pub const Which = enum { rmsnorm, layernorm };

pub fn normSpec(which: Which) c.Spec {
    return switch (which) {
        // the backbone's norms at its width, the prompt's rows a pass
        .rmsnorm => .{ .name = "cuda zonos3 z3_rmsnorm (fp32 rows, bf16 gamma)", .accum = .f32, .depth = normDepth, .ops = 8, .out = .f32, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 1, .n = 1024, .k = 1024 }, .{ .rows = 64, .n = 1024, .k = 1024 }, .{ .rows = 300, .n = 1024, .k = 1024 } } },
        // the codec's stages (768, 1280), the teacher (1024) and its features (160)
        .layernorm => .{ .name = "cuda zonos3_codec zc_layernorm (fp32 rows)", .accum = .f32, .depth = normDepth, .ops = 10, .out = .f32, .tiles = .{ .{}, .{}, .{ .size = 256, .step = 32, .min = 32 } }, .served = &.{ .{ .rows = 300, .n = 1280, .k = 1280 }, .{ .rows = 999, .n = 1024, .k = 1024 }, .{ .rows = 999, .n = 160, .k = 160 }, .{ .rows = 1024, .n = 768, .k = 768 } } },
    };
}

/// A norm over fp32 rows: z3_rmsnorm with its bf16 weight, or zc_layernorm with fp32 weight and bias.
pub const Norm = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    /// control: the launch's epsilon is one more than the reference's
    fault: bool = false,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    b: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module, which: Which, fault: bool) !Norm {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .which = which, .f = try mod.function(if (which == .rmsnorm) "z3_rmsnorm" else "zc_layernorm"), .fault = fault };
    }

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        var spec = normSpec(self.which);
        if (self.fault) spec.name = if (self.which == .rmsnorm) "control: zonos3 z3_rmsnorm with its epsilon one too large" else "control: zonos3 zc_layernorm with its epsilon one too large";
        return .{ .spec = spec, .ctx = self, .vt = &vt, .control = if (self.fault) .shapes else null };
    }

    const vt = g.hooks(Norm);

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != shape.k or shape.k % 32 != 0) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
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
        const k: c_int = @intCast(r.shape.k);
        var a: cuda.Args = .{};
        a.add(self.x);
        if (self.which == .rmsnorm) {
            a.add(k);
            a.add(self.w);
            a.add(self.out);
            a.add(k);
        } else {
            a.add(self.w);
            a.add(self.b);
            a.add(self.out);
        }
        a.add(k);
        a.add(if (self.fault) r.eps + 1 else r.eps);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(r.shape.rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
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

/// A lane's chain of fused multiply-adds over every 32nd input, then the warp's butterfly.
fn linearDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

pub const Weights = enum { bf16, f32 };

pub fn linearSpec(weights: Weights) c.Spec {
    return switch (weights) {
        // the backbone's projections: GDN, dense feed-forward, an expert's, the head's, the edit adapter's factors
        .bf16 => .{ .name = "cuda zonos3 z3_linear (fp32 rows, bf16 weights)", .accum = .f32, .depth = linearDepth, .out = .f32, .tiles = .{ .{}, .{ .size = 8 }, .{ .size = 256, .step = 8, .min = 8 } }, .served = &.{ .{ .rows = 1, .n = 2056, .k = 1024 }, .{ .rows = 64, .n = 1024, .k = 2048 }, .{ .rows = 2, .n = 3072, .k = 256 }, .{ .rows = 1, .n = 40980, .k = 1024 }, .{ .rows = 16, .n = 1024, .k = 4096 }, .{ .rows = 1, .n = 64, .k = 1024 }, .{ .rows = 1, .n = 40980, .k = 64 }, .{ .rows = 2, .n = 3072, .k = 16 }, .{ .rows = 2, .n = 16, .k = 256 } } },
        // the codec's: a stage's attention and feed-forward, the encoder's first, an iSTFT branch, the aligner's
        .f32 => .{ .name = "cuda zonos3_codec zc_linear (fp32 rows and weights)", .accum = .f32, .depth = linearDepth, .out = .f32, .tiles = .{ .{}, .{ .size = 8 }, .{ .size = 32 } }, .served = &.{ .{ .rows = 16, .n = 2304, .k = 768 }, .{ .rows = 8, .n = 1280, .k = 5120 }, .{ .rows = 8, .n = 768, .k = 5888 }, .{ .rows = 8, .n = 2208, .k = 2210 }, .{ .rows = 300, .n = 1024, .k = 512 }, .{ .rows = 600, .n = 256, .k = 1280 } } },
    };
}

/// y [rows, n] = x [rows, k] W^T with fp32 sums: z3_linear over bf16 weights, or zc_linear over fp32 ones.
pub const Linear = struct {
    rig: rig_.Rig,
    weights: Weights,
    f: cuda.Function,
    /// control: the launch is one block of eight outputs short
    fault: bool = false,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module, weights: Weights, fault: bool) !Linear {
        const f = try mod.function(if (weights == .bf16) "z3_linear" else "zc_linear");
        return .{ .rig = .{ .gpu = gpu, .s = s }, .weights = weights, .f = f, .fault = fault };
    }

    pub fn deinit(self: *Linear) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Linear) g.Kernel {
        var spec = linearSpec(self.weights);
        if (self.fault) spec.name = if (self.weights == .bf16) "control: zonos3 z3_linear launched one block short" else "control: zonos3 zc_linear launched one block short";
        return .{ .spec = spec, .ctx = self, .vt = &vt, .control = if (self.fault) .shapes else null };
    }

    const vt = g.hooks(Linear);

    pub fn prepare(self: *Linear, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // the bf16 kernel reads eight weights a load; both stage a row in at most 6 KiB of floats
        if (shape.k == 0 or shape.n == 0 or shape.rows == 0 or shape.k > 6144 or shape.k % 2 != 0) return false;
        if (self.weights == .bf16 and shape.k % 8 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
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
        const d = self.d.?;
        const k: c_int = @intCast(d.shape.k);
        const n: c_int = @intCast(d.shape.n);
        var a: cuda.Args = .{};
        a.add(self.w);
        a.add(self.x);
        a.add(k);
        if (self.weights == .bf16) a.add(@as(c_int, 1));
        a.add(k);
        a.add(n);
        a.add(self.out);
        a.add(n);
        a.add(@as(u64, 0));
        if (self.weights == .bf16) {
            a.add(@as(u64, 0));
            a.add(@as(i64, 0));
        }
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((d.shape.n + 7) / 8 - ref.one(self.fault)), .y = @intCast(d.shape.rows), .z = 1 }, .block = .{ .x = 256 }, .shared = @intCast(d.shape.k * 4) }, self.rig.s, &a);
        try self.rig.s.synchronize();
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

const ref = @import("contract_zonos3_ref.zig");
const point = @import("contract_zonos3_point.zig");
const pick = @import("contract_zonos3_pick.zig");
const backbone = @import("contract_zonos3_backbone.zig");
const gdn = @import("contract_zonos3_gdn.zig");
const attn = @import("contract_zonos3_attn.zig");
const windows = @import("contract_zonos3_windows.zig");
const conv = @import("contract_zonos3_conv.zig");

fn Pair(comptime K: type) type {
    return [2]ref.Gated(K);
}

/// The family's registered kernels over its four images: each kernel, then the control planted for it.
pub const Set = struct {
    gpa: std.mem.Allocator,
    mods: ref.Mods,
    norms: [4]Norm,
    linears: [4]Linear,
    points: [2 * point.kinds.len]ref.Gated(point.Point),
    route: [2]ref.Checked(pick.Route),
    nearest: [2]ref.Checked(pick.Nearest),
    embed: Pair(backbone.Embed),
    attnres: Pair(backbone.AttnRes),
    conv_silu: Pair(backbone.ConvSilu),
    gdn_prep: Pair(gdn.GdnPrep),
    gdn: Pair(gdn.Gdn),
    gated_norm: Pair(gdn.GatedNorm),
    attn_prep: Pair(attn.Prep),
    attention: [3]Pair(attn.Attention),
    codec_prep: Pair(attn.CodecPrep),
    dwconv: [2]Pair(windows.DwConv),
    latent: Pair(windows.Latent),
    ola: Pair(windows.Ola),
    fir: [3]Pair(windows.Fir),
    conv: Pair(conv.Conv),

    /// The fields that hold a kernel and its control side by side, in the order they register.
    const pairs = .{ "embed", "attnres", "conv_silu", "gdn_prep", "gdn", "gated_norm", "attn_prep", "codec_prep", "latent", "ola", "conv" };
    const groups = .{ "attention", "dwconv", "fir" };

    /// The set, its kernels appended to `list`.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var m: ref.Mods = undefined;
        m.backbone = try cuda.Module.load(gpu.d, cuda.kernels.zonos3);
        errdefer m.backbone.unload();
        m.codec = try cuda.Module.load(gpu.d, cuda.kernels.zonos3_codec);
        errdefer m.codec.unload();
        m.encoder = try cuda.Module.load(gpu.d, cuda.kernels.zonos3_encoder);
        errdefer m.encoder.unload();
        m.teacher = try cuda.Module.load(gpu.d, cuda.kernels.zonos3_teacher);
        errdefer m.teacher.unload();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        set.gpa = gpu.gpa;
        set.mods = m;
        for (0..4) |i| {
            const codec = i >= 2;
            const fault = i % 2 == 1;
            set.norms[i] = try Norm.init(gpu, s, if (codec) m.codec else m.backbone, if (codec) .layernorm else .rmsnorm, fault);
            set.linears[i] = try Linear.init(gpu, s, if (codec) m.codec else m.backbone, if (codec) .f32 else .bf16, fault);
        }
        for (point.kinds, 0..) |kind, i| for (0..2) |j| {
            set.points[2 * i + j] = .{ .rig = rig, .k = try point.Point.init(m, kind, j == 1) };
        };
        const f = struct {
            fn of(mod: cuda.Module, comptime symbol: [:0]const u8) !cuda.Function {
                return mod.function(symbol);
            }
        }.of;
        for (0..2) |j| {
            const fault = j == 1;
            set.route[j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_route"), .fault = fault } };
            set.nearest[j] = .{ .rig = rig, .k = .{ .f = try f(m.encoder, "ze_nearest"), .fault = fault } };
            set.embed[j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_embed"), .fault = fault } };
            set.attnres[j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_attnres"), .fault = fault } };
            set.conv_silu[j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_conv_silu"), .fault = fault } };
            set.gdn_prep[j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_gdn_prep"), .fault = fault } };
            set.gdn[j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_gdn"), .fault = fault } };
            set.gated_norm[j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_gated_norm"), .fault = fault } };
            set.attn_prep[j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_attn_prep"), .fault = fault } };
            set.attention[0][j] = .{ .rig = rig, .k = .{ .f = try f(m.backbone, "z3_attention"), .which = .backbone, .fault = fault } };
            set.attention[1][j] = .{ .rig = rig, .k = .{ .f = try f(m.codec, "zc_attention"), .which = .codec, .fault = fault } };
            set.attention[2][j] = .{ .rig = rig, .k = .{ .f = try f(m.teacher, "zt_attention"), .which = .teacher, .fault = fault } };
            set.codec_prep[j] = .{ .rig = rig, .k = .{ .f = try f(m.codec, "zc_attn_prep"), .fault = fault } };
            set.dwconv[0][j] = .{ .rig = rig, .k = .{ .f = try f(m.codec, "zc_dwconv"), .which = .codec, .fault = fault } };
            set.dwconv[1][j] = .{ .rig = rig, .k = .{ .f = try f(m.teacher, "zt_dwconv"), .which = .teacher, .fault = fault } };
            set.latent[j] = .{ .rig = rig, .k = .{ .f = try f(m.codec, "zc_latent"), .fault = fault } };
            set.ola[j] = .{ .rig = rig, .k = .{ .f = try f(m.codec, "zc_ola"), .fault = fault } };
            set.fir[0][j] = .{ .rig = rig, .k = .{ .f = try f(m.codec, "zc_mix"), .which = .mix, .fault = fault } };
            set.fir[1][j] = .{ .rig = rig, .k = .{ .f = try f(m.codec, "zc_resample"), .which = .resample, .fault = fault } };
            set.fir[2][j] = .{ .rig = rig, .k = .{ .f = try f(m.encoder, "ze_resample_in"), .which = .resample_in, .fault = fault } };
            set.conv[j] = .{ .rig = rig, .k = .{ .f = try f(m.encoder, "ze_conv"), .fault = fault } };
        }
        for (&set.norms) |*x| try list.append(a, x.kernel());
        for (&set.linears) |*x| try list.append(a, x.kernel());
        for (&set.points) |*x| try list.append(a, x.kernel());
        for (&set.route) |*x| try list.append(a, x.kernel());
        for (&set.nearest) |*x| try list.append(a, x.kernel());
        inline for (pairs) |field| for (&@field(set, field)) |*x| try list.append(a, x.kernel());
        inline for (groups) |field| for (&@field(set, field)) |*pair| for (pair) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        inline for (groups) |field| for (&@field(set, field)) |*pair| for (pair) |*x| x.deinit();
        inline for (pairs) |field| for (&@field(set, field)) |*x| x.deinit();
        for (&set.nearest) |*x| x.deinit();
        for (&set.route) |*x| x.deinit();
        for (&set.points) |*x| x.deinit();
        for (&set.linears) |*x| x.deinit();
        for (&set.norms) |*x| x.deinit();
        set.mods.teacher.unload();
        set.mods.encoder.unload();
        set.mods.codec.unload();
        set.mods.backbone.unload();
        set.gpa.destroy(set);
    }
};
