//! What Qwen3-ASR's XPU contract tests share: the card, its images and each control's, buffers, bf16 values.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const images = @import("qwen3_asr_contract_images");
const guarded = @import("contract_guarded.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const records = contract.qwen3_asr_xpu_records;

/// The binding's image a kernel lives in.
pub const Image = enum { qwen3_asr, lfm3_gq2 };

/// The image a test builds: the binding's, or a control's (an edit of the same source, xpu_qwen3_asr_contract.zig).
pub const Mutation = enum {
    none,
    gemm_no_residual,
    gemm_bf16_sums,
    conv_taps_transposed,
    flatten_f_major,
    layernorm_no_bias,
    gelu_is_silu,
    attn_causal,
    rope_turns_back,
    rms_no_weight,
    attn_reads_ahead,
    swiglu_swapped,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .gemm_no_residual => "control: xpu qwen3-asr asr_gemm, the residual is not added",
            .gemm_bf16_sums => "control: xpu qwen3-asr asr_gemm, each 16-k step added in bf16",
            .conv_taps_transposed => "control: xpu qwen3-asr asr_conv, kh and kw swapped",
            .flatten_f_major => "control: xpu qwen3-asr asr_flatten, rows written f-major",
            .layernorm_no_bias => "control: xpu qwen3-asr asr_layernorm, the bias is dropped",
            .gelu_is_silu => "control: xpu qwen3-asr asr_gelu, SiLU in place of GELU",
            .attn_causal => "control: xpu qwen3-asr asr_attn_full, a causal mask",
            .rope_turns_back => "control: xpu qwen3-asr asr_qk_rope, the rotation turns the other way",
            .rms_no_weight => "control: xpu qwen3-asr lfm3_gq2 lfm3_rmsnorm, the weight is dropped",
            .attn_reads_ahead => "control: xpu qwen3-asr lfm3_gq2 lfm3_attn_partial, keys past the row's position",
            .swiglu_swapped => "control: xpu qwen3-asr lfm3_gq2 lfm3_swiglu, gate and up swapped",
        };
    }

    /// The gate a control must fail.
    pub fn gate(m: Mutation) c.Gate {
        return if (m == .gemm_bf16_sums) .sums else .shapes;
    }
};

fn blob(m: Mutation) []const u8 {
    return switch (m) {
        .none => unreachable,
        inline else => |t| &@field(images, @tagName(t)),
    };
}

/// One card, its context and stream, the binding's two images and each control's, built from SPIR-V by the driver.
pub const Dev = struct {
    drv: xpu.Driver,
    ctx: xpu.Context,
    s: xpu.Stream,
    asr: xpu.Module,
    lf: xpu.Module,
    controls: [std.enums.values(Mutation).len]?xpu.Module = @splat(null),

    /// Opens the card `ordinal` names (else the first), loads the binding's images; `self` stays where it is.
    pub fn init(self: *Dev, ordinal: ?u32) !void {
        if (!xpu.kernels.available) return error.BuiltWithoutKernels;
        self.drv = try xpu.Driver.open();
        self.ctx = try xpu.Context.init(&self.drv, ordinal);
        self.s = try xpu.Stream.init(&self.ctx);
        self.asr = try xpu.Module.load(&self.ctx, xpu.kernels.qwen3_asr);
        self.lf = try xpu.Module.load(&self.ctx, xpu.kernels.lfm3_gq2);
        self.controls = @splat(null);
    }

    pub fn deinit(self: *Dev) void {
        for (&self.controls) |*m| if (m.*) |*x| x.unload();
        self.lf.unload();
        self.asr.unload();
        self.s.deinit();
        self.ctx.deinit();
        self.drv.close();
    }

    /// The image `m` builds for `img`: the binding's for none, else the control's (loaded once).
    pub fn module(self: *Dev, img: Image, m: Mutation) !*xpu.Module {
        if (m == .none) return if (img == .qwen3_asr) &self.asr else &self.lf;
        const slot = &self.controls[@backingInt(m)];
        if (slot.* == null) slot.* = try xpu.Module.load(&self.ctx, blob(m));
        return &slot.*.?;
    }

    /// `symbol` at the work-group its record holds, from the binding's image or the control's.
    pub fn kernel(self: *Dev, img: Image, m: Mutation, symbol: [:0]const u8) !xpu.Kernel {
        const rec = recordOf(symbol) orelse return error.NoRecord;
        return (try self.module(img, m)).kernel(symbol, rec.launch.threads);
    }
};

pub const Rig = guarded.Rig(Dev);

pub fn recordOf(symbol: []const u8) ?records.Record {
    for (records.records) |r| if (std.mem.eql(u8, r.launch.symbol, symbol)) return r;
    return null;
}

/// A test's kernel under its control's name and gate.
pub fn mark(k: g.Kernel, m: Mutation) g.Kernel {
    var out = k;
    if (m != .none) {
        out.spec.name = m.name();
        out.control = m.gate();
    }
    return out;
}

pub fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn u(x: usize) u32 {
    return @intCast(x);
}

/// A value rounded to bf16 (to nearest even), as the kernels' to_bf.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

pub fn val(bits: u16) f64 {
    return c.fromBf16(bits);
}

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

/// Random bf16 values in (-s, s).
pub fn fill(rng: std.Random, out: []u16, s: f32) void {
    for (out) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * s);
}

/// NaN, +Inf and -Inf at three spread places of `x` (and both infinities once more when it is long enough).
pub fn plant(x: []u16) void {
    for (planted, 0..) |v, i| x[(x.len / 3 * i + 1) % x.len] = c.toBf16(v);
    if (x.len > 8) x[x.len - 2] = c.toBf16(planted[1]);
}

/// fp32's unit roundoff.
pub const u32r = 0x1p-24;
/// bf16's: half an ulp at 1.
pub const ubf = 0x1p-8;
/// The least step a bf16 output takes near zero.
pub const floor_bf = 0x1p-133;

/// One query row's attention over `keys` rows into `out` (max subtracted); `allow`: its error bound, as derived.
pub fn attention(a: std.mem.Allocator, q: []const f64, keys: usize, kv: anytype, scale: f64, chunks: usize, out: []f64, allow: ?[]f64) !void {
    const d = q.len;
    const s = try a.alloc(f64, keys);
    defer a.free(s);
    const ds = try a.alloc(f64, keys);
    defer a.free(ds);
    var mx: f64 = -std.math.inf(f64);
    for (0..keys) |j| {
        var dot: f64 = 0;
        var abs: f64 = 0;
        for (0..d) |e| {
            const t = q[e] * kv.k(j, e);
            dot += t;
            abs += @abs(t);
        }
        s[j] = dot * scale;
        ds[j] = c.gamma(@floatFromInt(d + 2), u32r) * abs * scale;
        if (std.math.isNan(s[j]) or s[j] > mx) mx = s[j];
    }
    var l: f64 = 0;
    for (s) |*x| {
        x.* = @exp(x.* - mx);
        l += x.*;
    }
    var worst: f64 = 0;
    for (ds) |e| worst = @max(worst, e);
    // each weight's relative error: its score's and the max's dot errors, exp, the rescales (a key a step) and merges
    const rel_w = 2 * worst + @as(f64, @floatFromInt(2 * keys + 4 * chunks + 8)) * u32r;
    for (0..d) |e| {
        var acc: f64 = 0;
        for (0..keys) |j| acc += s[j] * kv.v(j, e);
        out[e] = acc / l;
        if (allow) |al| {
            var spread: f64 = 0;
            var mag: f64 = 0;
            for (0..keys) |j| {
                spread += s[j] * @abs(kv.v(j, e) - out[e]);
                mag += s[j] * @abs(kv.v(j, e));
            }
            const work = rel_w * spread / l + c.gamma(@floatFromInt(keys + chunks + 4), u32r) * mag / l + 4 * u32r * @abs(out[e]);
            al[e] = work + ubf * (@abs(out[e]) + work) + floor_bf;
        }
    }
}
