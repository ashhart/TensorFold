//! The row kernels under the contract on XPU: asr_layernorm, asr_gelu, lfm3_gq2's RMSNorm and SwiGLU.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const kit = @import("qwen3_asr_contract_kit.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const Mutation = kit.Mutation;
const val = kit.val;
const u = kit.u;

pub const Which = enum { layernorm, rms, gelu };

/// A 256-item row: each item's chain over every 256th value, a 16-lane sub-group tree, then the 16 parts in order.
fn rowDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 4 + 16;
}

const ln_why = "a bf16 output follows the variance as var^-1/2: a bf16 sum's error is a few output ulps";
const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn rowSpec(which: Which) c.Spec {
    return switch (which) {
        // the 0.6B tower (896), the 1.7B's (1024), a short last window
        .layernorm => .{ .name = "xpu qwen3-asr asr_layernorm (bias, eps 1e-5)", .accum = .f32, .depth = rowDepth, .ops = 8, .out = .bf16, .served = &.{ .{ .rows = 5, .n = 896, .k = 896 }, .{ .rows = 104, .n = 1024, .k = 1024 }, .{ .rows = 13, .n = 896, .k = 896 } }, .sums = .{ .bf16 = 0, .why = ln_why } },
        // the 0.6B LM (1024), the 1.7B's (2048), a prompt's rows
        .rms => .{ .name = "xpu qwen3-asr lfm3_gq2 lfm3_rmsnorm (eps 1e-6)", .accum = .f32, .depth = rowDepth, .ops = 8, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 4, .n = 1024, .k = 1024 }, .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 150, .n = 1024, .k = 1024 } }, .sums = .{ .bf16 = 0, .why = rms_why } },
        // the convs' channels, fc1's rows, proj1's
        .gelu => .{ .name = "xpu qwen3-asr asr_gelu (erf)", .accum = .none, .depth = kit.noSums, .ops = 8, .out = .bf16, .served = &.{ .{ .rows = 4, .n = 1000, .k = 1000 }, .{ .rows = 104, .n = 3584, .k = 3584 }, .{ .rows = 13, .n = 896, .k = 896 } } },
    };
}

pub const Row = struct {
    a: std.mem.Allocator,
    rig: kit.Rig,
    k: xpu.Kernel,
    m: Mutation,
    which: Which,
    r: ?rows.Rows = null,
    x: ?*anyopaque = null,
    w: ?*anyopaque = null,
    b: ?*anyopaque = null,
    out: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *kit.Dev, m: Mutation, which: Which) !Row {
        const k = switch (which) {
            .layernorm => try dev.kernel(.qwen3_asr, m, "asr_layernorm"),
            .gelu => try dev.kernel(.qwen3_asr, m, "asr_gelu"),
            .rms => try dev.kernel(.lfm3_gq2, m, "lfm3_rmsnorm"),
        };
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .k = k, .m = m, .which = which };
    }

    pub fn deinit(t: *Row) void {
        if (t.r) |r| r.deinit();
        t.k.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *Row) g.Kernel {
        return kit.mark(.{ .spec = rowSpec(t.which), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Row);

    fn op(w: Which) rows.Op {
        return switch (w) {
            .layernorm => .layer_norm,
            .rms => .rms,
            .gelu => .gelu,
        };
    }

    pub fn prepare(t: *Row, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n != shape.k) return false;
        // lfm3_rmsnorm holds 16 values an item: whole 256-item passes up to 4,096
        if (t.which == .rms and (shape.k % 256 != 0 or shape.k > 4096)) return false;
        if (t.r) |r| r.deinit();
        t.r = null;
        t.rig.reset();
        const r = try rows.Rows.init(t.a, op(t.which), shape);
        t.r = r;
        r.fill(fill, seed);
        t.x = try t.rig.input(std.mem.sliceAsBytes(r.x));
        if (t.which != .gelu) t.w = try t.rig.input(std.mem.sliceAsBytes(r.w));
        if (t.which == .layernorm) t.b = try t.rig.input(std.mem.sliceAsBytes(r.b));
        t.out = try t.rig.output(r.outputs() * 2);
        return true;
    }

    pub fn run(t: *Row) !void {
        const r = t.r.?;
        const k = r.shape.k;
        switch (t.which) {
            .layernorm => try t.rig.launch(t.k, .{ u(r.shape.rows), 1, 1 }, .{ t.x, t.w, t.b, t.out, u(k), r.eps }),
            .rms => try t.rig.launch(t.k, .{ u(r.shape.rows), 1, 1 }, .{ t.x, t.w, t.out, u(k), r.eps }),
            .gelu => try t.rig.launch(t.k, .{ u((r.outputs() + 255) / 256), 1, 1 }, .{ t.x, t.out, u(r.outputs()) }),
        }
    }

    pub fn count(t: *Row) usize {
        return t.r.?.outputs();
    }

    pub fn reference(t: *Row, want: []f64, mag: []f64) !void {
        return t.r.?.reference(want, mag);
    }

    /// GELU only: what an fp32 erf of 16 ulps (OpenCL's allowance) adds, x / 2 times 2^-20 near erf = +-1.
    pub fn bounds(t: *Row, _: []const f64, _: []const f64, allow: []f64) !void {
        if (t.which != .gelu) return;
        for (allow, t.r.?.x) |*l, x| l.* += 0.5 * @abs(val(x)) * 0x1p-20;
    }

    pub fn simulate(t: *Row, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.r.?.simulate(acc, order, out);
    }
};

/// lfm3_swiglu over [gate | up] rows: bf16(bf16(silu(gate)) up), a value an item. rows x n values, k = n.
pub const Swiglu = struct {
    a: std.mem.Allocator,
    rig: kit.Rig,
    k: xpu.Kernel,
    m: Mutation,
    shape: c.Shape = undefined,
    gate: []u16 = &.{},
    up: []u16 = &.{},
    ins: [2]?*anyopaque = undefined,
    out: ?*anyopaque = null,

    pub fn spec() c.Spec {
        // the 0.6B's intermediate (3072), the 1.7B's (6144), an odd count
        return .{ .name = "xpu qwen3-asr lfm3_gq2 lfm3_swiglu", .accum = .none, .depth = kit.noSums, .ops = 10, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 3, .n = 300, .k = 300 }, .{ .rows = 1, .n = 3072, .k = 3072 }, .{ .rows = 1, .n = 6144, .k = 6144 }, .{ .rows = 37, .n = 3072, .k = 3072 } } };
    }

    pub fn init(a: std.mem.Allocator, dev: *kit.Dev, m: Mutation) !Swiglu {
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .k = try dev.kernel(.lfm3_gq2, m, "lfm3_swiglu"), .m = m };
    }

    pub fn deinit(t: *Swiglu) void {
        t.a.free(t.gate);
        t.a.free(t.up);
        t.k.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *Swiglu) g.Kernel {
        return kit.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Swiglu);

    pub fn prepare(t: *Swiglu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.gate);
        t.a.free(t.up);
        t.shape = shape;
        const n = shape.rows * shape.n;
        t.gate = try t.a.alloc(u16, n);
        t.up = try t.a.alloc(u16, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        kit.fill(rng, t.gate, 6);
        kit.fill(rng, t.up, 3);
        if (fill == .nonfinite) {
            kit.plant(t.gate);
            t.up[n / 2] = c.toBf16(std.math.nan(f32));
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.gate));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.up));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Swiglu) !void {
        const n = t.shape.rows * t.shape.n;
        try t.rig.launch(t.k, .{ u((n + 63) / 64), 1, 1 }, .{ t.ins[0], t.ins[1], t.out, u(n) });
    }

    pub fn count(t: *Swiglu) usize {
        return t.shape.rows * t.shape.n;
    }

    /// silu(gate) up in float64, as the kernel's formula (NaN where -inf meets inf).
    pub fn reference(t: *Swiglu, want: []f64, mag: []f64) !void {
        for (want, mag, t.gate, t.up) |*w, *m, a, b| {
            const x = val(a);
            w.* = x / (1 + @exp(-x)) * val(b);
            m.* = 0;
        }
    }
};
