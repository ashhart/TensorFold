//! Unlimited-OCR-2's CUDA moves contract, part one: shared helpers and the im2col and window cases.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const uocr = @import("contract_uocr.zig");

pub const c = core.contract.spec;
pub const g = core.contract.gates;
pub const Kernels = uocr.kernels.Kernels;

pub const sam_window = 14;
pub const sam_heads = 12;
pub const sam_head_dim = 64;
const sam_width = uocr.sam_width;
/// A prompt chunk's rows at most, and its expert slots (top-6 a row).
pub const chunk = 2048;
pub const slots = chunk * uocr.topk;

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

/// A planted control: its name, the control kernel standing in for the family's, and the gate that must go red.
pub const Control = struct { name: []const u8, symbol: [:0]const u8, gate: c.Gate = .shapes };

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 round it.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// Any finite bf16 (every exponent, subnormals and both zeros): a copy must keep its bits.
pub fn finiteBits(rng: std.Random) u16 {
    while (true) {
        const v = rng.int(u16);
        if (std.math.isFinite(c.fromBf16(v))) return v;
    }
}

/// A bf16 value in (-scale, scale): two of them sum exactly in float64, so bf() of the sum is fp32's add then bf16.
pub fn value(rng: std.Random, scale: f32) u16 {
    return c.toBf16((rng.float(f32) * 2 - 1) * scale);
}

pub fn bits(v: f32) u16 {
    return c.toBf16(v);
}

pub fn val(h: u16) f64 {
    return c.fromBf16(h);
}

/// Distinct ids from [0, k) in random order (a shuffled prefix): an expert's token rows or slots.
pub fn distinct(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize) ![]u32 {
    const all = try a.alloc(u32, k);
    defer a.free(all);
    for (all, 0..) |*v, i| v.* = @intCast(i);
    for (0..n) |i| std.mem.swap(u32, &all[i], &all[i + rng.uintLessThan(usize, k - i)]);
    return a.dupe(u32, all[0..n]);
}

/// Copies `len` bytes of device input `src` over output `dst` before a launch that writes only part of it.
pub fn preset(rig: *rig_.Rig, dst: u64, src: u64, len: usize) !void {
    try (cuda.DeviceBuffer{ .d = rig.gpu.d, .ptr = dst, .len = len }).copyFrom(0, src, len, rig.s.handle);
}

/// The family's kernels, or with `name` swapped for the control's kernel.
pub fn kernelsFor(ctx: uocr.Ctx, comptime name: []const u8, ctl: ?Control) !Kernels {
    const x = ctl orelse return ctx.k.*;
    return ctx.with(name, x.symbol);
}

fn blocks(n: usize) usize {
    return (n + 255) / 256;
}

// --- tf_uocr_im2col: convolution patches, row (oy, ox) of [out^2, C k k] in (c, ky, kx) order, zero outside ---

pub const Geo = struct {
    name: []const u8,
    k: u32,
    stride: u32,
    pad: u32,
    /// The input as [C, h, w] (pixels) or [h, w, C] (feature maps).
    chw: bool,
    served: []const c.Shape,
    tiles: [3]c.Tile,
};

/// Rows: output side; n: input channels; k: input side. The engine's 4 launches a view, at the 1024 view and 640 tile.
pub const geos = [_]Geo{
    .{
        .name = "cuda uocr moves tf_uocr_im2col patch 16x16 stride 16 (pixels [3, h, w])",
        .k = 16,
        .stride = 16,
        .pad = 0,
        .chw = true,
        .served = &.{ .{ .rows = 64, .n = 3, .k = 1024 }, .{ .rows = 40, .n = 3, .k = 640 } },
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 1008, .min = 1000 } },
    },
    .{
        .name = "cuda uocr moves tf_uocr_im2col neck 3x3 stride 1 pad 1 (256 channels)",
        .k = 3,
        .stride = 1,
        .pad = 1,
        .chw = false,
        .served = &.{ .{ .rows = 40, .n = 256, .k = 40 }, .{ .rows = 64, .n = 256, .k = 64 } },
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 32 } },
    },
    .{
        .name = "cuda uocr moves tf_uocr_im2col net2 3x3 stride 2 pad 1 (256 channels)",
        .k = 3,
        .stride = 2,
        .pad = 1,
        .chw = false,
        .served = &.{ .{ .rows = 20, .n = 256, .k = 40 }, .{ .rows = 32, .n = 256, .k = 64 } },
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 32 } },
    },
    .{
        .name = "cuda uocr moves tf_uocr_im2col net3 3x3 stride 2 pad 1 (512 channels)",
        .k = 3,
        .stride = 2,
        .pad = 1,
        .chw = false,
        .served = &.{ .{ .rows = 10, .n = 512, .k = 20 }, .{ .rows = 16, .n = 512, .k = 32 } },
        .tiles = .{ .{ .size = 8 }, .{ .size = 32 }, .{ .size = 16 } },
    },
};

pub const Im2col = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: Kernels,
    geo: Geo,
    ctl: ?Control,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    pub fn spec(geo: Geo, ctl: ?Control) c.Spec {
        return .{ .name = if (ctl) |x| x.name else geo.name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = geo.tiles, .served = geo.served };
    }
    pub fn init(ctx: uocr.Ctx, geo: Geo, ctl: ?Control) !Im2col {
        return .{ .a = ctx.gpu.gpa, .rig = ctx.rig(), .k = try kernelsFor(ctx, "im2col", ctl), .geo = geo, .ctl = ctl };
    }
    pub fn deinit(t: *Im2col) void {
        t.a.free(t.x);
        t.rig.deinit();
    }
    pub fn kernel(t: *Im2col) g.Kernel {
        return .{ .spec = spec(t.geo, t.ctl), .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(Im2col);

    fn index(t: *const Im2col, y: usize, x: usize, ch: usize) usize {
        const side = t.shape.k;
        return if (t.geo.chw) (ch * side + y) * side + x else (y * side + x) * t.shape.n + ch;
    }

    pub fn prepare(t: *Im2col, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.x = &.{};
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, shape.k * shape.k * shape.n);
        for (t.x) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite) {
            // a pixel each, one pixel row holding both infinities; each read by the first outputs' patches
            const at = [_][3]usize{ .{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ 2, 2, 2 }, .{ 3, 3, 0 }, .{ 3, 4, 1 } };
            const v = [_]f32{ planted[0], planted[1], planted[2], planted[1], planted[2] };
            for (at, v) |p, x| if (p[0] < shape.k and p[1] < shape.k) {
                t.x[t.index(p[0], p[1], p[2] % shape.n)] = bits(x);
            };
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    fn width(t: *const Im2col) usize {
        return t.shape.n * t.geo.k * t.geo.k;
    }

    pub fn run(t: *Im2col) !void {
        const side = t.shape.k;
        const out = t.shape.rows;
        try t.k.run("im2col", t.rig.s, .{ out * out, blocks(t.width()), 1 }, 256, .{ t.in, t.out, side, side, t.shape.n, @as(usize, t.geo.k), @as(usize, t.geo.stride), @as(usize, t.geo.pad), out, out, @as(usize, @intFromBool(t.geo.chw)) });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Im2col) usize {
        return t.shape.rows * t.shape.rows * t.width();
    }

    pub fn reference(t: *Im2col, want: []f64, mag: []f64) !void {
        const w = t.width();
        const k = t.geo.k;
        const out = t.shape.rows;
        const side: isize = @intCast(t.shape.k);
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / w;
            const col = i % w;
            const ch = col / (k * k);
            const ky = (col / k) % k;
            const kx = col % k;
            const iy = @as(isize, @intCast((row / out) * t.geo.stride + ky)) - @as(isize, @intCast(t.geo.pad));
            const ix = @as(isize, @intCast((row % out) * t.geo.stride + kx)) - @as(isize, @intCast(t.geo.pad));
            const inside = iy >= 0 and ix >= 0 and iy < side and ix < side;
            o.* = if (inside) val(t.x[t.index(@intCast(iy), @intCast(ix), ch)]) else 0;
            m.* = 0;
        }
    }
};

// --- tf_uocr_window and tf_uocr_unwindow_add: SAM's 14x14 windows over a zero-padded grid, and back with the add ---

pub const Window = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: Kernels,
    back: bool,
    ctl: ?Control,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    win: []u16 = &.{},
    in: u64 = 0,
    /// tf_uocr_unwindow_add's residual as uploaded: copied over the output before each launch (the step is in place).
    resid: u64 = 0,
    out: u64 = 0,

    /// Rows: the patch grid's side (64 or 40); n: SAM's channels. Sides off the window multiple pad with zeros.
    pub fn spec(back: bool) c.Spec {
        return .{
            .name = if (back) "cuda uocr moves tf_uocr_unwindow_add 14 (crop, bf16 residual add, in place)" else "cuda uocr moves tf_uocr_window 14 (zero padding)",
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .in_place = back,
            .tiles = .{ .{ .size = sam_window }, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 64, .n = sam_width, .k = 1 }, .{ .rows = 40, .n = sam_width, .k = 1 } },
        };
    }
    pub fn init(ctx: uocr.Ctx, back: bool, ctl: ?Control) !Window {
        const k = if (back) try kernelsFor(ctx, "unwindow_add", ctl) else try kernelsFor(ctx, "window", ctl);
        return .{ .a = ctx.gpu.gpa, .rig = ctx.rig(), .k = k, .back = back, .ctl = ctl };
    }
    pub fn deinit(t: *Window) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Window) void {
        inline for (.{ "x", "win" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Window) g.Kernel {
        var s = spec(t.back);
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(Window);

    fn padded(t: *const Window) usize {
        return (t.shape.rows + sam_window - 1) / sam_window * sam_window;
    }
    /// The windows' row holding grid position (y, x).
    fn winRow(t: *const Window, y: usize, x: usize) usize {
        const per = t.padded() / sam_window;
        return ((y / sam_window) * per + x / sam_window) * sam_window * sam_window + (y % sam_window) * sam_window + x % sam_window;
    }

    pub fn prepare(t: *Window, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.n;
        const side = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, side * side * n);
        t.win = try t.a.alloc(u16, if (t.back) t.padded() * t.padded() * n else 0);
        if (t.back) {
            for (t.x) |*v| v.* = value(rng, 4);
            for (t.win) |*v| v.* = value(rng, 4);
        } else for (t.x) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite) {
            // grid positions (0, 0), (0, 1), (1, 0), (1, 1): NaN, +Inf, -Inf, both infinities (the add's two terms)
            const pos = [_][2]usize{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
            for (pos, 0..) |p, j| if (p[0] < side and p[1] < side) {
                const col = (j * 7) % n;
                const tok = (p[0] * side + p[1]) * n + col;
                if (j < 3) {
                    if (t.back and j == 2) t.win[t.winRow(p[0], p[1]) * n + col] = bits(planted[2]) else t.x[tok] = bits(planted[j]);
                } else {
                    t.x[tok] = bits(planted[1]);
                    if (t.back) t.win[t.winRow(p[0], p[1]) * n + col] = bits(planted[2]) else t.x[tok + @min(1, n - 1 - col)] = bits(planted[2]);
                }
            };
            // a NaN in the padding past the grid's bottom edge: the crop must leave it out
            if (t.back and t.padded() > side) t.win[t.winRow(side, 0) * n] = bits(planted[0]);
        }
        if (t.back) {
            t.resid = try t.rig.input(std.mem.sliceAsBytes(t.x));
            t.in = try t.rig.input(std.mem.sliceAsBytes(t.win));
            t.out = try t.rig.output(t.x.len * 2);
        } else {
            t.in = try t.rig.input(std.mem.sliceAsBytes(t.x));
            t.out = try t.rig.output(t.padded() * t.padded() * n * 2);
        }
        return true;
    }

    pub fn run(t: *Window) !void {
        const n = t.shape.n;
        const side = t.shape.rows;
        if (t.back) {
            try preset(&t.rig, t.out, t.resid, t.x.len * 2);
            try t.k.run("unwindow_add", t.rig.s, .{ side * side, blocks(n), 1 }, 256, .{ t.out, t.in, side, t.padded(), @as(usize, sam_window), n });
        } else {
            try t.k.run("window", t.rig.s, .{ t.padded() * t.padded(), blocks(n), 1 }, 256, .{ t.in, t.out, side, t.padded(), @as(usize, sam_window), n });
        }
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Window) usize {
        return if (t.back) t.x.len else t.padded() * t.padded() * t.shape.n;
    }

    pub fn reference(t: *Window, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        const side = t.shape.rows;
        const per = t.padded() / sam_window;
        const w2 = sam_window * sam_window;
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / n;
            const col = i % n;
            m.* = 0;
            if (t.back) {
                o.* = bf(val(t.x[i]) + val(t.win[t.winRow(row / side, row % side) * n + col]));
                continue;
            }
            const win = row / w2;
            const in = row % w2;
            const gy = (win / per) * sam_window + in / sam_window;
            const gx = (win % per) * sam_window + in % sam_window;
            o.* = if (gy < side and gx < side) val(t.x[(gy * side + gx) * n + col]) else 0;
        }
    }
};
