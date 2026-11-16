//! Flint's glue under the contract: the embedding, SwiGLU (split and stacked), the sorted rows' gather and argmax.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const fl = @import("metal_flint.zig");

const c = contract.spec;
const g = contract.gates;
const bf = fl.bf;
const Range = fl.Range;

// --- fl_embed: an 8-bit affine row, bf16(scale * q + bias) held in fp32 ----------------------------------------------

/// Table rows the test's ids index (the gates need distinct rows, not a 150k vocabulary).
const table = 48;
const ebits = 8;
const egroup = 64;

pub const Embed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: ?g.Gate = null,
    label: [160]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    ids: []u32 = &.{},
    words: []u32 = &.{},
    s: []u16 = &.{},
    b: []u16 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    const base = "metal flint glue fl_embed q8 g64";
    const drop: fl.Edit = .{ .kernel = "fl_embed", .from = "float(bfloat(float(S[g]) * q + float(B[g])))", .to = "(float(S[g]) * q + float(B[g]))" };

    /// Rows: tokens; n: the hidden width (k unused). The embedding is the checkpoint's 8-bit groups of 64.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .f32,
            .tiles = .{ .{}, .{ .size = 256, .step = 64, .min = 64 }, .{} },
            .served = &.{ .{ .rows = fl.window_rows, .n = fl.hidden, .k = 1 }, .{ .rows = fl.prompt_rows, .n = fl.hidden, .k = 1 }, .{ .rows = 1, .n = fl.hidden, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Embed {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try fl.gluePipe(a, device, "fl_embed", if (control) drop else null), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Embed) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Embed) void {
        inline for (.{ "ids", "words", "s", "b" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Embed) g.Kernel {
        t.name = fl.named(base, if (t.control != null) "the bf16 rounding dropped" else null, &t.label);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Embed);

    pub fn prepare(t: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % egroup != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.ids = try t.a.alloc(u32, shape.rows);
        for (t.ids, 0..) |*v, r| v.* = if (r < 4) @intCast(r) else rng.uintLessThan(u32, table);
        t.words = try t.a.alloc(u32, table * n / 4);
        for (t.words) |*v| v.* = rng.int(u32);
        t.s = try t.a.alloc(u16, table * n / egroup);
        t.b = try t.a.alloc(u16, table * n / egroup);
        // MLX's affine groups: a positive scale, the bias the group's least value
        for (t.s, t.b) |*s, *b| {
            const sc = 0.001 + rng.float(f32) * 0.02;
            s.* = c.toBf16(sc);
            b.* = c.toBf16(-sc * (rng.float(f32) * 255));
        }
        if (fill == .nonfinite) {
            const gpr = n / egroup;
            t.s[0 * gpr] = c.toBf16(fl.planted[0]);
            t.b[1 * gpr] = c.toBf16(fl.planted[1]);
            t.b[2 * gpr] = c.toBf16(fl.planted[2]);
            t.s[3 * gpr] = c.toBf16(fl.planted[1]);
            if (gpr > 1) t.b[3 * gpr + 1] = c.toBf16(fl.planted[2]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.words));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.s));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.out = try t.rig.output(shape.rows * n * 4);
        return true;
    }

    pub fn run(t: *Embed) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue(fl.Args{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.n), .bits = ebits, .group = egroup }, 4);
        p.e.setBuffer(t.out.buf, t.out.off, 5);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Embed) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Element (r, j)'s scale times its code, and its bias.
    fn parts(t: *Embed, r: usize, j: usize) [2]f64 {
        const n = t.shape.n;
        const row = t.ids[r];
        const q: f64 = @floatFromInt((t.words[(row * n + j) / 4] >> @intCast((j % 4) * 8)) & 0xff);
        const gi = row * (n / egroup) + j / egroup;
        return .{ @as(f64, c.fromBf16(t.s[gi])) * q, c.fromBf16(t.b[gi]) };
    }

    pub fn reference(t: *Embed, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const p = t.parts(i / t.shape.n, i % t.shape.n);
            w.* = bf(p[0] + p[1]);
            m.* = 0;
        }
    }

    /// The scale-and-add in fp32 (fused or not, the compiler's choice): two roundings at most, then the bf16 point.
    pub fn intervals(t: *Embed, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *h, i| {
            const p = t.parts(i / t.shape.n, i % t.shape.n);
            const e = fl.u32_ulp * (@abs(p[0]) + 2 * @abs(p[0] + p[1])) + fl.tiny;
            const r = Range.point(p[0] + p[1]).widen(e).bf16();
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    pub fn bounds(t: *Embed, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }
};

// --- fl_swiglu and fl_swiglu_gu: bf16(bf16(g * sigmoid(g)) * u), MLX's silu(gate) * up, its sigmoid in bf16 steps -

/// MLX's Sigmoid of bf16, each step bf16: e = exp(|x|), y = 1 / (1 + e), y below 0, 1 - y above; else fp32's range.
fn sigmoidBf16(x: f64, exact: bool) Range {
    const e0 = @exp(@abs(x));
    const e = (if (exact) Range.point(e0) else Range.point(e0).widen(2 * fl.u32_ulp * e0)).bf16();
    const d = e.plus(Range.point(1));
    const den = (if (exact) d else d.fp32()).bf16();
    const inv = Range{ .lo = 1 / den.hi, .hi = 1 / den.lo };
    const y = (if (exact) inv else inv.fp32()).bf16();
    if (x < 0) return y;
    const one = Range{ .lo = 1 - y.hi, .hi = 1 - y.lo };
    return (if (exact) one else one.fp32()).bf16();
}

pub const Swiglu = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    stacked: bool,
    what: Control = .none,
    label: [160]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    gate: []u16 = &.{},
    up: []u16 = &.{},
    gu: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn base(stacked: bool) []const u8 {
        return if (stacked) "metal flint glue fl_swiglu_gu (stacked gate|up, dense and shared MLPs)" else "metal flint glue fl_swiglu (routed experts' slots)";
    }
    /// The controls: the product's bf16 point dropped, or the sigmoid's 1 + e point (MLX rounds every step).
    const Control = enum { none, product, sigmoid };
    fn drop(stacked: bool, what: Control) fl.Edit {
        if (what == .sigmoid) return .{ .kernel = "fl_sigmoid_bf16", .from = "float(bfloat(1.0f / float(bfloat(1.0f + e))))", .to = "float(bfloat(1.0f / (1.0f + e)))" };
        return .{ .kernel = if (stacked) "fl_swiglu_gu" else "fl_swiglu", .from = "bfloat(float(bfloat(g * s)) * u)", .to = "bfloat((g * s) * u)" };
    }

    /// Rows: slots or tokens; n: the SwiGLU width.
    pub fn spec(name: []const u8, stacked: bool) c.Spec {
        // the routed experts' slots: a window's 16 x 6, a shared forward's 32 x 6, a prompt chunk's 2,048 x 6
        const routed = [_]c.Shape{ .{ .rows = fl.window_rows * fl.topk, .n = fl.moe_inter, .k = 1 }, .{ .rows = fl.batch_rows * fl.topk, .n = fl.moe_inter, .k = 1 }, .{ .rows = fl.prompt_rows * fl.topk, .n = fl.moe_inter, .k = 1 } };
        // the dense layers' and the shared expert's rows
        const dense = [_]c.Shape{ .{ .rows = fl.batch_rows, .n = fl.inter, .k = 1 }, .{ .rows = fl.prompt_rows, .n = fl.shared_inter, .k = 1 }, .{ .rows = 1, .n = fl.inter, .k = 1 }, .{ .rows = fl.window_rows, .n = fl.shared_inter, .k = 1 } };
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = if (stacked) &dense else &routed,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, stacked: bool, what: Control) !Swiglu {
        const pipe = try fl.gluePipe(a, device, if (stacked) "fl_swiglu_gu" else "fl_swiglu", if (what != .none) drop(stacked, what) else null);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .stacked = stacked, .what = what };
    }
    pub fn deinit(t: *Swiglu) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Swiglu) void {
        inline for (.{ "gate", "up", "gu" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Swiglu) g.Kernel {
        const what: ?[]const u8 = switch (t.what) {
            .none => null,
            .product => "the product's bf16 rounding dropped",
            .sigmoid => "the sigmoid's 1 + e bf16 rounding dropped",
        };
        t.name = fl.named(base(t.stacked), what, &t.label);
        return .{ .spec = spec(t.name, t.stacked), .ctx = t, .vt = &vt, .control = if (t.what != .none) .shapes else null };
    }
    const vt = g.hooks(Swiglu);

    pub fn prepare(t: *Swiglu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const total = shape.rows * shape.n;
        t.gate = try t.a.alloc(u16, total);
        t.up = try t.a.alloc(u16, total);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gates deep into the sigmoid's tails on both sides
        for (t.gate) |*v| v.* = c.toBf16(rng.float(f32) * 48 - 32);
        for (t.up) |*v| v.* = c.toBf16(rng.float(f32) * 20 - 10);
        if (fill == .nonfinite) for (0..@min(shape.rows, 6)) |r| {
            const at = r * shape.n + (r * 7) % shape.n;
            if (r < 3) t.gate[at] = c.toBf16(fl.planted[r]) else t.up[at] = c.toBf16(fl.planted[r - 3]);
        };
        if (t.stacked) {
            const n = shape.n;
            t.gu = try t.a.alloc(u16, 2 * total);
            for (0..shape.rows) |r| {
                @memcpy(t.gu[r * 2 * n ..][0..n], t.gate[r * n ..][0..n]);
                @memcpy(t.gu[r * 2 * n + n ..][0..n], t.up[r * n ..][0..n]);
            }
            t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.gu));
        } else {
            t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.gate));
            t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.up));
        }
        t.out = try t.rig.output(total * 2);
        return true;
    }

    pub fn run(t: *Swiglu) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        const args = fl.Args{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.n) };
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        if (t.stacked) {
            p.e.setValue(args, 1);
            p.e.setBuffer(t.out.buf, t.out.off, 2);
        } else {
            p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
            p.e.setValue(args, 2);
            p.e.setBuffer(t.out.buf, t.out.off, 3);
        }
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Swiglu) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Swiglu, want: []f64, mag: []f64) !void {
        for (want, mag, t.gate, t.up) |*w, *m, gb, ub| {
            const x: f64 = c.fromBf16(gb);
            w.* = bf(bf(x * sigmoidBf16(x, true).lo) * @as(f64, c.fromBf16(ub)));
            m.* = 0;
        }
    }

    pub fn intervals(t: *Swiglu, low: []f64, high: []f64) !void {
        for (low, high, t.gate, t.up) |*l, *h, gb, ub| {
            const x: f64 = c.fromBf16(gb);
            const r = sigmoidBf16(x, false).times(x).bf16().times(c.fromBf16(ub)).bf16();
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    pub fn bounds(t: *Swiglu, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }
};

// --- fl_rows_by_order: each sorted pair's input row, a copy ---------------------------------------------------------

pub const Gather = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: ?g.Gate = null,
    label: [160]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    order: []i32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    const base = "metal flint glue fl_rows_by_order top 6";
    const wrong: fl.Edit = .{ .kernel = "fl_rows_by_order", .from = "(order[p.y] / int(a.heads))", .to = "(order[p.y] % int(a.heads))" };

    /// Rows: a prompt chunk's slots (tokens x 6; the gates' small calls end mid-token); n: the hidden width.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = fl.topk, .step = fl.topk, .min = fl.topk }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            // the shortest prompt chunk (33 tokens) and the longest
            .served = &.{ .{ .rows = (fl.batch_rows + 1) * fl.topk, .n = fl.hidden, .k = 1 }, .{ .rows = fl.prompt_rows * fl.topk, .n = fl.hidden, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Gather {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try fl.gluePipe(a, device, "fl_rows_by_order", if (control) wrong else null), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Gather) void {
        t.a.free(t.x);
        t.a.free(t.order);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Gather) g.Kernel {
        t.name = fl.named(base, if (t.control != null) "a pair reads the wrong token's row" else null, &t.label);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Gather);

    pub fn prepare(t: *Gather, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.a.free(t.order);
        t.shape = shape;
        const tokens = (shape.rows + fl.topk - 1) / fl.topk;
        t.x = try t.a.alloc(u16, tokens * shape.n);
        t.order = try t.a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        if (fill == .nonfinite) for (fl.planted, 0..) |v, i| {
            t.x[(i % tokens) * shape.n + i] = c.toBf16(v);
        };
        // fl_sort's order: a permutation of the slots
        for (t.order, 0..) |*v, i| v.* = @intCast(i);
        rng.shuffle(i32, t.order);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.order));
        t.out = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(t: *Gather) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setValue(fl.Args{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.n), .heads = fl.topk }, 2);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Gather) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Gather, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*w, *m, i| {
            const tok: usize = @intCast(@divTrunc(t.order[i / n], fl.topk));
            w.* = c.fromBf16(t.x[tok * n + i % n]);
            m.* = 0;
        }
    }
};

// --- fl_argmax: the greedy draw over a bf16 logits row, ties to the lower id ----------------------------------------

/// fl_better: NaN beats any number (the lower id among NaNs), else the larger value, ties to the lower id.
fn better(v: f32, i: usize, best: f32, at: usize) bool {
    if (std.math.isNan(v)) return !std.math.isNan(best) or i < at;
    if (std.math.isNan(best)) return false;
    return v > best or (v == best and i < at);
}

pub const Argmax = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: ?g.Gate = null,
    label: [160]u8 = undefined,
    name: []const u8 = "",
    logits: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    const base = "metal flint glue fl_argmax";
    const last_tie: fl.Edit = .{ .kernel = "fl_better", .from = "(a == b && ia < ib)", .to = "(a == b && ia > ib)" };

    /// n: the vocabulary (rows and k unused: one row a launch).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .raw,
            .nan = .{ .stated = "a NaN logit wins (the lowest id among NaNs), as torch.argmax takes NaN as the largest; else ties to the lower id" },
            .tiles = .{ .{}, .{ .size = 1024, .step = 1, .min = 1 }, .{} },
            .served = &.{.{ .rows = 1, .n = fl.vocab, .k = 1 }},
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Argmax {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try fl.gluePipe(a, device, "fl_argmax", if (control) last_tie else null), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Argmax) void {
        t.a.free(t.logits);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Argmax) g.Kernel {
        t.name = fl.named(base, if (t.control != null) "a tie goes to the higher id" else null, &t.label);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Argmax);

    pub fn prepare(t: *Argmax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.logits);
        t.logits = try t.a.alloc(u16, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // few distinct values, so the largest is tied many times over
        for (t.logits) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 4 - 8);
        if (fill == .nonfinite) {
            const n = shape.n;
            t.logits[n / 3] = c.toBf16(fl.planted[1]);
            t.logits[n / 2] = c.toBf16(fl.planted[0]);
            t.logits[n - 1] = c.toBf16(fl.planted[0]);
            t.logits[0] = c.toBf16(fl.planted[2]);
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.out = try t.rig.output(4);
        return true;
    }

    pub fn run(t: *Argmax) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setBuffer(t.out.buf, t.out.off, 1);
        p.e.setValue(@as(u32, @intCast(t.logits.len)), 2);
        p.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(_: *Argmax) usize {
        return 1;
    }
    pub fn reference(_: *Argmax, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Argmax, _: g.Gate) !?[]const u8 {
        var best: f32 = -std.math.inf(f32);
        var at: usize = std.math.maxInt(u32);
        for (t.logits, 0..) |v, i| if (better(c.fromBf16(v), i, best, at)) {
            best = c.fromBf16(v);
            at = i;
        };
        const got = std.mem.bytesToValue(u32, t.rig.bytes(t.out, 4));
        if (got != at) return try std.fmt.bufPrint(&t.msg, "drew id {d} where the rule picks {d} of {d}", .{ got, at, t.logits.len });
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    embeds: fl.List(Embed) = .{},
    swiglus: fl.List(Swiglu) = .{},
    gathers: fl.List(Gather) = .{},
    argmaxes: fl.List(Argmax) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |control| {
            if (fl.wanted(only, Embed.base)) try s.embeds.add(s.a, s.kernels, try Embed.init(s.a, s.device, s.queue, control));
            for ([_]bool{ false, true }) |stacked| if (fl.wanted(only, Swiglu.base(stacked))) {
                try s.swiglus.add(s.a, s.kernels, try Swiglu.init(s.a, s.device, s.queue, stacked, if (control) .product else .none));
            };
            if (control and fl.wanted(only, Swiglu.base(false))) try s.swiglus.add(s.a, s.kernels, try Swiglu.init(s.a, s.device, s.queue, false, .sigmoid));
            if (fl.wanted(only, Gather.base)) try s.gathers.add(s.a, s.kernels, try Gather.init(s.a, s.device, s.queue, control));
            if (fl.wanted(only, Argmax.base)) try s.argmaxes.add(s.a, s.kernels, try Argmax.init(s.a, s.device, s.queue, control));
        }
    }
    pub fn deinit(s: *Suite) void {
        s.embeds.deinit(s.a);
        s.swiglus.deinit(s.a);
        s.gathers.deinit(s.a);
        s.argmaxes.deinit(s.a);
    }
};

/// Gate 5's texts for `flint glue ...`: glue.metal as is, on every chip.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?fl.Texts {
    if (!std.mem.startsWith(u8, name, "flint glue ")) return null;
    return .{ .here = sources.flint_glue, .pre_m5 = sources.flint_glue };
}
