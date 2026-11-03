//! Unlimited-OCR-2's moves contract, part one: shared helpers and the im2col, window and rel_pos cases.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");

const c = contract.spec;
const g = contract.gates;
const bf = uocr.bf;
const planted = uocr.planted;

pub const sam_window = 14;
pub const sam_heads = 12;
pub const sam_head_dim = 64;
const sam_width = uocr.sam_width;
const clip_width = uocr.clip_width;
const hidden = uocr.hidden;
/// A prompt chunk's rows at most, and its expert slots (top-6 a row).
pub const chunk = 2048;
pub const slots = chunk * uocr.topk;

/// A source edit that makes a planted control: `from` must occur once in the family's source.
pub const Edit = struct { from: []const u8, to: []const u8 };
/// A control: its name, the edit, and the gate that must go red.
pub const Control = struct { name: []const u8, edit: Edit, gate: c.Gate = .shapes };

pub fn build(a: std.mem.Allocator, device: mtl.Device, which: uocr.Library, name: []const u8, edit: ?Edit) !mtl.Pipeline {
    const e = edit orelse return uocr.pipeline(device, which, name);
    const src = uocr.source(which);
    if (std.mem.count(u8, src, e.from) != 1) return error.ControlEditNotFound;
    const text = try std.mem.replaceOwned(u8, a, src, e.from, e.to);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
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

// --- uv_im2col: convolution patches, row (oy, ox) of [out^2, C k k] in (c, ky, kx) order, zero outside the input ---

const Geo = struct {
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
        .name = "metal uocr moves uv_im2col patch 16x16 stride 16 (pixels [3, h, w])",
        .k = 16,
        .stride = 16,
        .pad = 0,
        .chw = true,
        .served = &.{ .{ .rows = 64, .n = 3, .k = 1024 }, .{ .rows = 40, .n = 3, .k = 640 } },
        // input sides short of the 64-patch grid: the last patches' outside pixels are zero
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 1008, .min = 1000 } },
    },
    .{
        .name = "metal uocr moves uv_im2col neck 3x3 stride 1 pad 1 (256 channels)",
        .k = 3,
        .stride = 1,
        .pad = 1,
        .chw = false,
        .served = &.{ .{ .rows = 40, .n = 256, .k = 40 }, .{ .rows = 64, .n = 256, .k = 64 } },
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 32 } },
    },
    .{
        .name = "metal uocr moves uv_im2col net2 3x3 stride 2 pad 1 (256 channels)",
        .k = 3,
        .stride = 2,
        .pad = 1,
        .chw = false,
        .served = &.{ .{ .rows = 20, .n = 256, .k = 40 }, .{ .rows = 32, .n = 256, .k = 64 } },
        // odd input sides reach the bottom and right padding
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 32 } },
    },
    .{
        .name = "metal uocr moves uv_im2col net3 3x3 stride 2 pad 1 (512 channels)",
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
    pipe: mtl.Pipeline,
    geo: Geo,
    ctl: ?Control,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn spec(geo: Geo, ctl: ?Control) c.Spec {
        return .{
            .name = if (ctl) |x| x.name else geo.name,
            .accum = .none,
            .depth = uocr.noSums,
            .out = .bf16,
            .tiles = geo.tiles,
            .served = geo.served,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, geo: Geo, ctl: ?Control) !Im2col {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .vision, "uv_im2col", if (ctl) |x| x.edit else null), .geo = geo, .ctl = ctl };
    }
    pub fn deinit(t: *Im2col) void {
        t.a.free(t.x);
        t.rig.deinit();
        t.pipe.deinit();
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
        const side: u32 = @intCast(t.shape.k);
        const out: u32 = @intCast(t.shape.rows);
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setBuffer(t.out.buf, t.out.off, 1);
        p.e.setValue([9]u32{ side, side, @intCast(t.shape.n), t.geo.k, t.geo.stride, t.geo.pad, out, out, @intFromBool(t.geo.chw) }, 2);
        p.e.dispatchThreads(mtl.Size.of(t.width(), t.shape.rows * t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
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

// --- uv_window and uv_unwindow_add: SAM's 14x14 windows over a zero-padded grid, and back (cropped) with the add ---

const UvWindow = extern struct { side: u32, padded: u32, window: u32, channels: u32 };

pub const Window = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    back: bool,
    ctl: ?Control,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    win: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: the patch grid's side (64 or 40); n: SAM's channels. Sides off the window multiple pad with zeros.
    pub fn spec(back: bool) c.Spec {
        return .{
            .name = if (back) "metal uocr moves uv_unwindow_add 14 (crop, bf16 residual add, in place)" else "metal uocr moves uv_window 14 (zero padding)",
            .accum = .none,
            .depth = uocr.noSums,
            .out = .bf16,
            .in_place = back,
            .tiles = .{ .{ .size = sam_window }, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 64, .n = sam_width, .k = 1 }, .{ .rows = 40, .n = sam_width, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, back: bool, ctl: ?Control) !Window {
        const name = if (back) "uv_unwindow_add" else "uv_window";
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .vision, name, if (ctl) |x| x.edit else null), .back = back, .ctl = ctl };
    }
    pub fn deinit(t: *Window) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
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
    const vt = blk: {
        var h = g.hooks(Window);
        h.overlap = overlap;
        break :blk h;
    };
    /// uv_unwindow_add reads and writes the residual in place.
    fn overlap(ctx: *anyopaque) bool {
        const t: *Window = @ptrCast(@alignCast(ctx));
        return t.back;
    }

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
                    // the residual for the first two, the windows' row for the third
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
            t.out = try t.rig.output(t.x.len * 2);
            t.in = try t.rig.input(std.mem.sliceAsBytes(t.win));
        } else {
            t.in = try t.rig.input(std.mem.sliceAsBytes(t.x));
            t.out = try t.rig.output(t.padded() * t.padded() * n * 2);
        }
        return true;
    }

    pub fn run(t: *Window) !void {
        if (t.back) @memcpy(t.rig.bytes(t.out, t.x.len * 2), std.mem.sliceAsBytes(t.x));
        const n = t.shape.n;
        const side = t.shape.rows;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        if (t.back) {
            p.e.setBuffer(t.out.buf, t.out.off, 0);
            p.e.setBuffer(t.in.buf, t.in.off, 1);
        } else {
            p.e.setBuffer(t.in.buf, t.in.off, 0);
            p.e.setBuffer(t.out.buf, t.out.off, 1);
        }
        p.e.setValue(UvWindow{ .side = @intCast(side), .padded = @intCast(t.padded()), .window = sam_window, .channels = @intCast(n) }, 2);
        const rows = if (t.back) side * side else t.padded() * t.padded();
        p.e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
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

// --- uv_rel_pos: rel_h[h, g, q, kh] = bf16(q . Rh[qh - kh + side - 1]), rel_w likewise over kw; 64 fp32 products ----

const UvRel = extern struct { groups: u32, side: u32, heads: u32, width: u32 };

fn sixtyFour(_: c.Shape) usize {
    return sam_head_dim;
}

pub const RelPos = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: ?Control,
    shape: c.Shape = undefined,
    qkv: []u16 = &.{},
    th: []u16 = &.{},
    tw: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    outs: [2]rig_.Ref = undefined,

    /// Rows: windows (1 global, 25 or 9); n: side (14, or 64/40 global); k: dots' head width. Checks rel_h, rel_w.
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr moves uv_rel_pos (12 heads of 64, fp32 sums)",
            .accum = .f32,
            .depth = sixtyFour,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 16 }, .{} },
            .served = &.{ .{ .rows = 1, .n = 40, .k = sam_head_dim }, .{ .rows = 1, .n = 64, .k = sam_head_dim }, .{ .rows = 9, .n = sam_window, .k = sam_head_dim }, .{ .rows = 25, .n = sam_window, .k = sam_head_dim } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: ?Control) !RelPos {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .vision, "uv_rel_pos", if (ctl) |x| x.edit else null), .ctl = ctl };
    }
    pub fn deinit(t: *RelPos) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *RelPos) void {
        inline for (.{ "qkv", "th", "tw" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *RelPos) g.Kernel {
        var s = spec();
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(RelPos);

    fn len(t: *const RelPos) usize {
        return t.shape.n * t.shape.n;
    }

    pub fn prepare(t: *RelPos, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != sam_head_dim) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const side = shape.n;
        const qrows = shape.rows * t.len();
        const trows = 2 * side - 1;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.qkv = try t.a.alloc(u16, qrows * 3 * sam_width);
        t.th = try t.a.alloc(u16, trows * sam_head_dim);
        t.tw = try t.a.alloc(u16, trows * sam_head_dim);
        if (fill == .stress) {
            // integer products, exact fp32 sums; second half = first rotated by 7, q negated: every dot is 0
            const half = sam_head_dim / 2;
            @memset(t.qkv, 0);
            for (0..qrows) |r| for (0..sam_heads) |h| {
                const q = t.qkv[r * 3 * sam_width + h * sam_head_dim ..][0..sam_head_dim];
                for (0..half) |d| q[d] = bits(@floatFromInt(128 + rng.uintLessThan(u32, 129)));
                for (0..half) |d| q[half + d] = bits(-c.fromBf16(q[(d + 7) % half]));
            };
            for ([_][]u16{ t.th, t.tw }) |tab| for (0..trows) |r| {
                const row = tab[r * sam_head_dim ..][0..sam_head_dim];
                for (0..half) |d| row[d] = bits(@floatFromInt(128 + rng.uintLessThan(u32, 129)));
                for (0..half) |d| row[half + d] = row[(d + 7) % half];
            };
        } else {
            for (t.qkv) |*v| v.* = value(rng, 1);
            for (t.th) |*v| v.* = value(rng, 1);
            for (t.tw) |*v| v.* = value(rng, 1);
        }
        if (fill == .nonfinite) {
            // q rows 0, 1 and 2 (heads 0, 1, 2), the last holding both infinities, and one row of Rh
            const row = 3 * sam_width;
            if (qrows > 2) {
                t.qkv[0 * row + 0 * sam_head_dim + 5] = bits(planted[0]);
                t.qkv[1 * row + 1 * sam_head_dim + 7] = bits(planted[1]);
                t.qkv[2 * row + 2 * sam_head_dim + 3] = bits(planted[1]);
                t.qkv[2 * row + 2 * sam_head_dim + 4] = bits(planted[2]);
            }
            t.th[(trows - 1) * sam_head_dim + 9] = bits(planted[2]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.th));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.tw));
        t.outs[0] = try t.rig.output(t.each() * 2);
        t.outs[1] = try t.rig.output(t.each() * 2);
        return true;
    }

    pub fn run(t: *RelPos) !void {
        const side = t.shape.n;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        for (t.outs, 0..) |r, i| p.e.setBuffer(r.buf, r.off, 3 + i);
        p.e.setValue(UvRel{ .groups = @intCast(t.shape.rows), .side = @intCast(side), .heads = sam_heads, .width = sam_width }, 5);
        p.e.dispatchThreads(mtl.Size.of(2 * side, t.shape.rows * t.len(), sam_heads), mtl.Size.of(32, 8, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *RelPos) usize {
        return 2 * t.each();
    }
    /// Outputs in each of rel_h and rel_w.
    fn each(t: *const RelPos) usize {
        return sam_heads * t.shape.rows * t.len() * t.shape.n;
    }

    /// Output i's 64 products (exact in fp32: bf16 times bf16).
    fn terms(t: *const RelPos, i: usize, out: *[sam_head_dim]f64) void {
        const side = t.shape.n;
        const groups = t.shape.rows;
        const w = i >= t.each();
        var o = i % t.each();
        const kk = o % side;
        o /= side;
        const qi = o % t.len();
        o /= t.len();
        const grp = o % groups;
        const h = o / groups;
        const pos = if (w) qi % side else qi / side;
        const tab = if (w) t.tw else t.th;
        const q = t.qkv[(grp * t.len() + qi) * 3 * sam_width + h * sam_head_dim ..][0..sam_head_dim];
        const r = tab[(pos + side - 1 - kk) * sam_head_dim ..][0..sam_head_dim];
        for (out, q, r) |*x, a, b| x.* = val(a) * val(b);
    }

    pub fn reference(t: *RelPos, want: []f64, mag: []f64) !void {
        var buf: [sam_head_dim]f64 = undefined;
        for (want, mag, 0..) |*o, *m, i| {
            t.terms(i, &buf);
            var s: f64 = 0;
            var a: f64 = 0;
            for (buf) |x| {
                s += x;
                a += @abs(x);
            }
            o.* = s;
            m.* = if (std.math.isFinite(a)) a else 0;
        }
    }

    pub fn simulate(t: *RelPos, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [sam_head_dim]f64 = undefined;
        for (out, 0..) |*o, i| {
            t.terms(i, &buf);
            o.* = bf(contract.rows.sumIn(acc, order, &buf));
        }
    }
};
