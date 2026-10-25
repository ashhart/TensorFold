//! Cosmos3's Metal glue under the contract: RMS and layer norms, relu^2, bias and GELUs, adds, 8-bit rows, image rows.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const ref = @import("cosmos3_ref.zig");

const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

pub const Which = enum { norm, layer_norm, relu2, bias, gelu_tanh, gelu_erf, add, embed, place, ctl_norm, ctl_layer_norm, ctl_relu2, ctl_gelu_erf, ctl_embed };

/// The kernel a registration launches (a control launches its kernel's planted mutation).
fn base(w: Which) Which {
    return switch (w) {
        .ctl_norm => .norm,
        .ctl_layer_norm => .layer_norm,
        .ctl_relu2 => .relu2,
        .ctl_gelu_erf => .gelu_erf,
        .ctl_embed => .embed,
        else => w,
    };
}

fn function(w: Which) [:0]const u8 {
    return switch (base(w)) {
        .norm => "cos3_norm",
        .layer_norm => "cos3_layer_norm",
        .relu2 => "cos3_relu2",
        .bias, .gelu_tanh, .gelu_erf => "cos3_bias",
        .add => "cos3_add",
        .embed => "cos3_embed_b8",
        else => "cos3_place",
    };
}

/// A control's one planted fault: the production line, and what it becomes.
pub fn mutation(w: Which) ?[2][]const u8 {
    return switch (w) {
        .ctl_norm => .{ "x[base + j] = w[4 * t + j] * static_cast<bfloat>(v[j] * inv[0]);", "x[base + j] = static_cast<bfloat>(float(w[4 * t + j]) * v[j] * inv[0]);" },
        .ctl_layer_norm => .{ "for (uint i = t; i < p.width; i += threads)\n    x[", "for (uint i = t; i + 1 < p.width; i += threads)\n    x[" },
        .ctl_relu2 => .{ "const float v = isnan(f) ? f : max(f, 0.0f);", "const float v = max(f, 0.0f);" },
        .ctl_gelu_erf => .{ "const float v = float(static_cast<bfloat>(float(y[i]) + float(bias[i % p.width])));", "const float v = float(y[i]) + float(bias[i % p.width]);" },
        .ctl_embed => .{ "scales[id * (D / 64) + at.x / 64]", "scales[id * (D / 64) + at.x / 128]" },
        else => null,
    };
}

pub fn control(w: Which) ?g.Gate {
    return switch (w) {
        .ctl_relu2 => .nonfinite,
        .ctl_norm, .ctl_layer_norm, .ctl_gelu_erf, .ctl_embed => .shapes,
        else => null,
    };
}

fn fourteen(_: c.Shape) usize {
    return 14;
}
fn lnDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 13;
}
fn none(_: c.Shape) usize {
    return 0;
}

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const ln_why = "the mean and variance reach bf16 outputs through rsqrt: a bf16 sum's error is a few output ulps";
const width_tile = [3]c.Tile{ .{}, .{ .size = 256 }, .{} };

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .norm => .{ .name = "metal cosmos3 cos3_norm (bf16 residual add, RMS norm, w after the bf16 point)", .accum = .f32, .depth = fourteen, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 64, .n = 2048, .k = 2048 } }, .sums = .{ .bf16 = 0, .why = norm_why } },
        .layer_norm => .{ .name = "metal cosmos3 cos3_layer_norm (bf16 residual add, fp32 mean and variance)", .accum = .f32, .depth = lnDepth, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 1, .n = 1152, .k = 1152 }, .{ .rows = 196, .n = 1152, .k = 1152 } }, .sums = .{ .bf16 = 0, .why = ln_why } },
        .relu2 => .{ .name = "metal cosmos3 cos3_relu2 (in place, NaN kept)", .accum = .none, .depth = none, .out = .bf16, .in_place = true, .tiles = width_tile, .served = &.{ .{ .rows = 1, .n = 9216, .k = 1 }, .{ .rows = 8, .n = 9216, .k = 1 } } },
        .bias => .{ .name = "metal cosmos3 cos3_bias (bf16 y + bias, no activation)", .accum = .none, .depth = none, .out = .bf16, .in_place = true, .tiles = width_tile, .served = &.{ .{ .rows = 1, .n = 1152, .k = 1 }, .{ .rows = 64, .n = 2048, .k = 1 } } },
        .gelu_tanh => .{ .name = "metal cosmos3 cos3_bias GELU (tanh) at bf16(y + bias)", .accum = .none, .depth = none, .out = .bf16, .in_place = true, .tiles = width_tile, .served = &.{.{ .rows = 64, .n = 4352, .k = 1 }} },
        .gelu_erf => .{ .name = "metal cosmos3 cos3_bias GELU (erf, A&S 7.1.26) at bf16(y + bias)", .accum = .none, .depth = none, .out = .bf16, .in_place = true, .tiles = width_tile, .served = &.{.{ .rows = 16, .n = 11520, .k = 1 }} },
        .add => .{ .name = "metal cosmos3 cos3_add (bf16 a + b in place)", .accum = .none, .depth = none, .out = .bf16, .in_place = true, .tiles = width_tile, .served = &.{.{ .rows = 196, .n = 1152, .k = 1 }} },
        .embed => .{ .name = "metal cosmos3 cos3_embed_b8 (8-bit rows, bf16(s q + b) from fp32)", .accum = .none, .depth = none, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 2048, .k = 1 }, .{ .rows = 40, .n = 2048, .k = 1 } } },
        .place => .{ .name = "metal cosmos3 cos3_place (image rows over their placeholders)", .accum = .none, .depth = none, .out = .bf16, .in_place = true, .served = &.{.{ .rows = 64, .n = 2048, .k = 1 }} },
        .ctl_norm => withName(.norm, "control: cosmos3 RMS norm taking w before its bf16 rounding point"),
        .ctl_layer_norm => withName(.layer_norm, "control: cosmos3 layer norm never writing a row's last output"),
        .ctl_relu2 => withName(.relu2, "control: cosmos3 relu^2 by max alone, which drops a NaN"),
        .ctl_gelu_erf => withName(.gelu_erf, "control: cosmos3 GELU (erf) without the bf16 point at y + bias"),
        .ctl_embed => withName(.embed, "control: cosmos3 8-bit rows reading a scale of the wrong group"),
    };
}

fn withName(w: Which, name: []const u8) c.Spec {
    var s = spec(w);
    s.name = name;
    return s;
}

/// The kernel's source, with a control's fault planted.
fn source(a: std.mem.Allocator, w: Which) ![]u8 {
    const m = mutation(w) orelse return a.dupe(u8, sources.cosmos3);
    if (std.mem.indexOf(u8, sources.cosmos3, m[0]) == null) return error.ControlLineMissing;
    return std.mem.replaceOwned(u8, a, sources.cosmos3, m[0], m[1]);
}

const LnArgs = extern struct { width: u32, eps: f32, add: u32 };
const BiasArgs = extern struct { width: u32, count: u32, act: u32 };
const vocab_rows = 64;
const image_rows = 16;

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    which: Which,
    shape: c.Shape = undefined,
    /// Host bf16 bits: the in-place buffer's start (h, x, y or a) and the other inputs.
    start: []u16 = &.{},
    other: []u16 = &.{},
    w: []u16 = &.{},
    b: []u16 = &.{},
    codes: []u8 = &.{},
    ids: []u32 = &.{},
    map: []i32 = &.{},
    point: []f64 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    refs: [6]rig_.Ref = undefined,
    live: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which) !Probe {
        const src = try source(a, which);
        defer a.free(src);
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, function(which)), .which = which };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.pipe.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.which), .ctx = p, .vt = &vt, .control = control(p.which) };
    }

    const vt = g.hooks(Probe);

    fn width(p: *const Probe) usize {
        return switch (base(p.which)) {
            .norm, .layer_norm => p.shape.k,
            else => p.shape.n,
        };
    }

    fn act(p: *const Probe) u32 {
        return switch (base(p.which)) {
            .gelu_tanh => 1,
            .gelu_erf => 2,
            else => 0,
        };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = base(p.which);
        if ((k == .norm and shape.k != 2048) or ((k == .embed or k == .place) and shape.n != 2048)) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const ar = p.arena.allocator();
        p.shape = shape;
        const wd = p.width();
        const total = shape.rows * wd;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const span: f32 = if (k == .gelu_tanh or k == .gelu_erf) 6 else if (k == .norm or k == .layer_norm) 1 else 3;
        p.start = try ar.alloc(u16, if (k == .embed) 0 else total);
        for (p.start, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress) @as(f32, @floatFromInt(rng.uintLessThan(u32, 65))) - 32 else (rng.float(f32) * 2 - 1) * span * (if (i % 64 == 7) @as(f32, 8) else 1));
        const other_len: usize = switch (k) {
            .norm, .layer_norm, .add => total,
            .bias, .gelu_tanh, .gelu_erf => wd,
            .place => image_rows * wd,
            else => 0,
        };
        p.other = try ar.alloc(u16, other_len);
        for (p.other) |*v| v.* = c.toBf16(if (fill == .stress) 0 else (rng.float(f32) * 2 - 1) * (if (k == .bias or k == .gelu_tanh or k == .gelu_erf) @as(f32, 1) else span));
        p.w = try ar.alloc(u16, if (k == .norm or k == .layer_norm) wd else 0);
        for (p.w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        p.b = try ar.alloc(u16, if (k == .layer_norm) wd else 0);
        for (p.b) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.25);
        if (k == .embed) {
            p.codes = try ar.alloc(u8, vocab_rows * wd);
            rng.bytes(p.codes);
            p.w = try ar.alloc(u16, vocab_rows * wd / 64); // scales
            for (p.w) |*v| v.* = c.toBf16(0.001 + rng.float(f32) * 0.02);
            p.b = try ar.alloc(u16, vocab_rows * wd / 64); // biases
            for (p.b) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.1);
            p.ids = try ar.alloc(u32, shape.rows);
            for (p.ids, 0..) |*v, r| v.* = if (r < 4) @intCast(r) else rng.uintLessThan(u32, vocab_rows);
        }
        if (k == .place) {
            p.map = try ar.alloc(i32, shape.rows);
            for (p.map, 0..) |*v, r| v.* = if (r % 3 == 0) -1 else @intCast((r * 7) % image_rows);
        }
        if (fill == .nonfinite) p.plant();
        try p.bind();
        try p.compute();
        return true;
    }

    /// NaN, +Inf and -Inf at known inputs: a row each, one row holding both infinities.
    fn plant(p: *Probe) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const wd = p.width();
        const at = [_][2]usize{ .{ 0, wd / 3 }, .{ 1, wd / 2 }, .{ 2, wd - 1 }, .{ 3, 0 }, .{ 3, wd - 1 } };
        const put = [_]u16{ nan, inf, ninf, inf, ninf };
        switch (base(p.which)) {
            .embed => for (put, 0..) |v, i| { // scales of rows 0 and 1, biases of rows 2 and 3
                if (i < 2) p.w[i * wd / 64 + i] = v else p.b[(i - 2) * wd / 64 + 3 + i] = v;
            },
            .place => for (at, put) |q, v| {
                p.other[(q[0] % image_rows) * wd + q[1]] = v;
                p.start[(q[0] + 3) * wd + q[1]] = v; // row 3 keeps its own h (map -1); later rows take image rows
            },
            else => for (at, put) |q, v| if (q[0] < p.shape.rows) {
                p.start[q[0] * wd + q[1]] = v;
            },
        }
    }

    fn bind(p: *Probe) !void {
        const k = base(p.which);
        const bytes = std.mem.sliceAsBytes;
        switch (k) {
            .norm, .layer_norm => {
                p.refs[0] = try p.rig.input(bytes(p.other));
                p.refs[1] = try p.rig.input(bytes(p.w));
                if (k == .layer_norm) p.refs[2] = try p.rig.input(bytes(p.b));
                p.refs[3] = try p.rig.output(p.start.len * 2); // x
                p.live = try p.rig.output(p.start.len * 2); // h, in place
            },
            .embed => {
                p.refs[0] = try p.rig.input(bytes(p.ids));
                p.refs[1] = try p.rig.input(p.codes);
                p.refs[2] = try p.rig.input(bytes(p.w));
                p.refs[3] = try p.rig.input(bytes(p.b));
                p.live = try p.rig.output(p.shape.rows * 2048 * 2);
            },
            .place => {
                p.refs[0] = try p.rig.input(bytes(p.other));
                p.refs[1] = try p.rig.input(bytes(p.map));
                p.live = try p.rig.output(p.start.len * 2);
            },
            else => {
                if (p.other.len > 0) p.refs[0] = try p.rig.input(bytes(p.other));
                p.live = try p.rig.output(p.start.len * 2);
            },
        }
    }

    pub fn run(p: *Probe) !void {
        const k = base(p.which);
        const wd: u32 = @intCast(p.width());
        const total: u32 = @intCast(p.shape.rows * wd);
        if (k != .embed) @memcpy(p.rig.bytes(p.live, p.start.len * 2), std.mem.sliceAsBytes(p.start));
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        const one = mtl.Size.of(256, 1, 1);
        switch (k) {
            .norm => {
                e.setBuffer(p.live.buf, p.live.off, 0);
                e.setBuffer(p.refs[0].buf, p.refs[0].off, 1);
                e.setBuffer(p.refs[1].buf, p.refs[1].off, 2);
                e.setValue(@as(f32, 1e-5), 3);
                e.setValue(@as(u32, 1), 4);
                e.setBuffer(p.refs[3].buf, p.refs[3].off, 5);
                e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(512, 1, 1));
            },
            .layer_norm => {
                e.setBuffer(p.live.buf, p.live.off, 0);
                for (0..3) |i| e.setBuffer(p.refs[i].buf, p.refs[i].off, i + 1);
                e.setValue(LnArgs{ .width = wd, .eps = 1e-6, .add = 1 }, 4);
                e.setBuffer(p.refs[3].buf, p.refs[3].off, 5);
                e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), one);
            },
            .relu2 => {
                e.setBuffer(p.live.buf, p.live.off, 0);
                e.setValue(total, 1);
                e.dispatchThreads(mtl.Size.of(total, 1, 1), one);
            },
            .bias, .gelu_tanh, .gelu_erf => {
                e.setBuffer(p.live.buf, p.live.off, 0);
                e.setBuffer(p.refs[0].buf, p.refs[0].off, 1);
                e.setValue(BiasArgs{ .width = wd, .count = total, .act = p.act() }, 2);
                e.dispatchThreads(mtl.Size.of(total, 1, 1), one);
            },
            .add => {
                e.setBuffer(p.live.buf, p.live.off, 0);
                e.setBuffer(p.refs[0].buf, p.refs[0].off, 1);
                e.setValue(total, 2);
                e.dispatchThreads(mtl.Size.of(total, 1, 1), one);
            },
            .embed => {
                for (0..4) |i| e.setBuffer(p.refs[i].buf, p.refs[i].off, i);
                e.setBuffer(p.live.buf, p.live.off, 4);
                e.dispatchThreads(mtl.Size.of(2048, p.shape.rows, 1), one);
            },
            else => {
                e.setBuffer(p.live.buf, p.live.off, 0);
                e.setBuffer(p.refs[0].buf, p.refs[0].off, 1);
                e.setBuffer(p.refs[1].buf, p.refs[1].off, 2);
                e.dispatchThreads(mtl.Size.of(2048, p.shape.rows, 1), one);
            },
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        const n = p.shape.rows * p.width();
        return switch (base(p.which)) {
            .norm, .layer_norm => 2 * n,
            else => n,
        };
    }

    fn val(bits: u16) f64 {
        return c.fromBf16(bits);
    }

    /// Every output's float64 value at the kernel's rounding points and its bf16 ends.
    fn compute(p: *Probe) !void {
        const ar = p.arena.allocator();
        const n = p.count();
        p.point = try ar.alloc(f64, n);
        p.lo = try ar.alloc(f64, n);
        p.hi = try ar.alloc(f64, n);
        const wd = p.width();
        const k = base(p.which);
        switch (k) {
            .norm, .layer_norm => {
                const rows = p.shape.rows;
                const v = try ar.alloc(f64, wd);
                const w = try ar.alloc(f64, wd);
                const b = try ar.alloc(f64, wd);
                for (w, 0..) |*x, i| x.* = val(p.w[i]);
                for (b, 0..) |*x, i| x.* = if (k == .layer_norm) val(p.b[i]) else 0;
                for (0..rows) |r| {
                    for (v, 0..) |*x, i| x.* = ref.bf(val(p.start[r * wd + i]) + val(p.other[r * wd + i]));
                    const at = r * wd;
                    if (k == .norm) ref.rmsRow(v, w, 1e-5, p.point[at..][0..wd], p.lo[at..][0..wd], p.hi[at..][0..wd]) else ref.layerRow(v, w, b, 1e-6, p.point[at..][0..wd], p.lo[at..][0..wd], p.hi[at..][0..wd]);
                    const h = rows * wd + at;
                    @memcpy(p.point[h..][0..wd], v);
                    @memcpy(p.lo[h..][0..wd], v);
                    @memcpy(p.hi[h..][0..wd], v);
                }
            },
            .embed => for (0..p.shape.rows) |r| for (0..wd) |i| {
                const id = p.ids[r];
                const s = val(p.w[id * (wd / 64) + i / 64]);
                const b = val(p.b[id * (wd / 64) + i / 64]);
                const y = ref.bf(s * @as(f64, @floatFromInt(p.codes[id * wd + i])) + b);
                p.exact(r * wd + i, y);
            },
            .place => for (0..p.shape.rows) |r| for (0..wd) |i| {
                const m = p.map[r];
                p.exact(r * wd + i, if (m >= 0) val(p.other[@as(usize, @intCast(m)) * wd + i]) else val(p.start[r * wd + i]));
            },
            .relu2 => for (p.start, 0..) |bits, i| {
                const x = val(bits);
                const f = if (std.math.isNan(x)) x else @max(x, 0);
                p.point[i] = f * f;
                p.lo[i] = ref.bf(f * f);
                p.hi[i] = p.lo[i];
            },
            .add => for (p.start, p.other, 0..) |x, y, i| p.exact(i, ref.bf(val(x) + val(y))),
            else => for (p.start, 0..) |x, i| {
                const v = ref.bf(val(x) + val(p.other[i % wd]));
                var half: f64 = 0;
                const y = ref.act(p.act(), v, &half);
                p.point[i] = y;
                if (half == 0) {
                    p.lo[i] = ref.bf(y);
                    p.hi[i] = p.lo[i];
                } else ref.ends(y, half, &p.lo[i], &p.hi[i]);
            },
        }
    }

    fn exact(p: *Probe, i: usize, y: f64) void {
        p.point[i] = y;
        p.lo[i] = y;
        p.hi[i] = y;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        @memcpy(want, p.point);
        @memset(mag, 0);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.lo, p.hi) |*l, w, lo, hi| l.* = @max(@abs(lo - w), @abs(hi - w));
    }

    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        @memcpy(low, p.lo);
        @memcpy(high, p.hi);
    }

    /// Gate 4: the norms' sums in `acc` and `order`, the rest of their arithmetic exact; h's adds as they are.
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const k = base(p.which);
        if (k != .norm and k != .layer_norm) return error.NoSums;
        const wd = p.width();
        const n: f64 = @floatFromInt(wd);
        const t = try p.a.alloc(f64, wd);
        defer p.a.free(t);
        const v = try p.a.alloc(f64, wd);
        defer p.a.free(v);
        for (0..p.shape.rows) |r| {
            for (v, 0..) |*x, i| x.* = ref.bf(val(p.start[r * wd + i]) + val(p.other[r * wd + i]));
            var mean: f64 = 0;
            if (k == .layer_norm) mean = rows_.sumIn(acc, order, v) / n;
            for (t, v) |*o, x| o.* = (x - mean) * (x - mean);
            const inv = 1 / @sqrt(rows_.sumIn(acc, order, t) / n + @as(f64, @as(f32, if (k == .norm) 1e-5 else 1e-6)));
            for (0..wd) |i| out[r * wd + i] = (v[i] - mean) * inv * val(p.w[i]) + (if (k == .layer_norm) val(p.b[i]) else 0);
            @memcpy(out[(p.shape.rows + r) * wd ..][0..wd], v);
        }
    }
};
