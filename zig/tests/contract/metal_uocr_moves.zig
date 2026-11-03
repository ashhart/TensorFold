//! Unlimited-OCR-2's moves under the contract: the tower's convolution patches, SAM's windows (out, and back with the
//! residual add), its decomposed relative positions, CLIP's embeddings and the fused rows; the decoder's MoE row
//! gather and scatter. Copies are exact (each output is the input value its index formula names, padding 0); the adds
//! round once; uv_rel_pos sums 64 products in fp32. Each index or padding rule has a planted control (a source edit).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");

const c = contract.spec;
const g = contract.gates;
const bf = uocr.bf;
const planted = uocr.planted;

const sam_window = 14;
const sam_heads = 12;
const sam_head_dim = 64;
const sam_width = uocr.sam_width;
const clip_width = uocr.clip_width;
const hidden = uocr.hidden;
/// A prompt chunk's rows at most, and its expert slots (top-6 a row).
const chunk = 2048;
const slots = chunk * uocr.topk;

/// A source edit that makes a planted control: `from` must occur once in the family's source.
const Edit = struct { from: []const u8, to: []const u8 };
/// A control: its name, the edit, and the gate that must go red.
const Control = struct { name: []const u8, edit: Edit, gate: c.Gate = .shapes };

fn build(a: std.mem.Allocator, device: mtl.Device, which: uocr.Library, name: []const u8, edit: ?Edit) !mtl.Pipeline {
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
fn finiteBits(rng: std.Random) u16 {
    while (true) {
        const v = rng.int(u16);
        if (std.math.isFinite(c.fromBf16(v))) return v;
    }
}

/// A bf16 value in (-scale, scale): two of them sum exactly in float64, so bf() of the sum is fp32's add then bf16.
fn value(rng: std.Random, scale: f32) u16 {
    return c.toBf16((rng.float(f32) * 2 - 1) * scale);
}

fn bits(v: f32) u16 {
    return c.toBf16(v);
}

fn val(h: u16) f64 {
    return c.fromBf16(h);
}

/// Distinct ids from [0, k) in random order (a shuffled prefix): an expert's token rows or slots.
fn distinct(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize) ![]u32 {
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

/// Rows: the output side; n: the input's channels; k: the input side. The engine's four launches a view
/// (vision.zig): the 16x16 patches, the neck's 3x3, and net2's and net3's stride-2 3x3, at the 1024 view and the 640 tile.
const geos = [_]Geo{
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

    /// Rows: the patch grid's side (64 for the 1024 view, 40 for a tile); n: SAM's channels. Sides off the window
    /// multiple pad the last windows (1 to 13 rows and columns of zeros).
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

    /// Rows: the windows (1 for a global block, 25 for the 1024 view's windows, 9 for a tile's); n: the side (14 in a
    /// window, the grid's 64 or 40 in a global block); k: the head width the dots sum over. Both outputs are checked
    /// (rel_h, then rel_w).
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
            // integer products of 2^14 to 2^16: each dot's partial sums reach about 2^21, exact in fp32 in any order;
            // the second half mirrors the first rotated by 7 (q negated), so every dot is 0, yet no order's halves or
            // lanes mirror each other (a reversed mirror would cancel bf16's roundings exactly)
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

// --- uv_concat: y [rows, left + right] = [a row | b row] ------------------------------------------------------------

pub const Concat = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: ?Control,
    shape: c.Shape = undefined,
    l: []u16 = &.{},
    r: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: the view's tokens (16^2 or 10^2); n: CLIP's width (left); k: SAM's net3 width (right).
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr moves uv_concat (CLIP | SAM rows)",
            .accum = .none,
            .depth = uocr.noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256 }, .{ .size = 256 } },
            .served = &.{ .{ .rows = 256, .n = clip_width, .k = 1024 }, .{ .rows = 100, .n = clip_width, .k = 1024 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: ?Control) !Concat {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .vision, "uv_concat", if (ctl) |x| x.edit else null), .ctl = ctl };
    }
    pub fn deinit(t: *Concat) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Concat) void {
        inline for (.{ "l", "r" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Concat) g.Kernel {
        var s = spec();
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(Concat);

    pub fn prepare(t: *Concat, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.l = try t.a.alloc(u16, shape.rows * shape.n);
        t.r = try t.a.alloc(u16, shape.rows * shape.k);
        for (t.l) |*v| v.* = finiteBits(rng);
        for (t.r) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite and shape.rows > 3) {
            t.l[0 * shape.n + 1] = bits(planted[0]);
            t.r[1 * shape.k + 2 % shape.k] = bits(planted[1]);
            t.l[2 * shape.n + 3 % shape.n] = bits(planted[2]);
            t.l[3 * shape.n + shape.n - 1] = bits(planted[1]);
            t.r[3 * shape.k] = bits(planted[2]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.l));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.r));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Concat) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue([3]u32{ @intCast(t.shape.rows), @intCast(t.shape.n), @intCast(t.shape.k) }, 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n + t.shape.k, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Concat) usize {
        return t.shape.rows * (t.shape.n + t.shape.k);
    }

    pub fn reference(t: *Concat, want: []f64, mag: []f64) !void {
        const w = t.shape.n + t.shape.k;
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / w;
            const col = i % w;
            o.* = if (col < t.shape.n) val(t.l[row * t.shape.n + col]) else val(t.r[row * t.shape.k + col - t.shape.n]);
            m.* = 0;
        }
    }
};

// --- uv_clip_embed: y[0] = bf16(cls + pos[0]), y[r] = bf16(sam[r - 1] + pos[r]) -----------------------------------

pub const ClipEmbed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: ?Control,
    shape: c.Shape = undefined,
    cls: []u16 = &.{},
    sam: []u16 = &.{},
    pos: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: CLIP's tokens (the class row and 16^2 or 10^2 of SAM's); n: CLIP's width.
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr moves uv_clip_embed (class row, positions, one bf16 add)",
            .accum = .none,
            .depth = uocr.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 8 }, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 257, .n = clip_width, .k = 1 }, .{ .rows = 101, .n = clip_width, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: ?Control) !ClipEmbed {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .vision, "uv_clip_embed", if (ctl) |x| x.edit else null), .ctl = ctl };
    }
    pub fn deinit(t: *ClipEmbed) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *ClipEmbed) void {
        inline for (.{ "cls", "sam", "pos" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *ClipEmbed) g.Kernel {
        var s = spec();
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(ClipEmbed);

    pub fn prepare(t: *ClipEmbed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.cls = try t.a.alloc(u16, n);
        t.sam = try t.a.alloc(u16, (shape.rows - 1) * n);
        t.pos = try t.a.alloc(u16, shape.rows * n);
        for (t.cls) |*v| v.* = value(rng, 4);
        for (t.sam) |*v| v.* = value(rng, 4);
        for (t.pos) |*v| v.* = value(rng, 1);
        if (fill == .nonfinite and shape.rows > 3 and n > 9) {
            t.cls[2] = bits(planted[0]); // row 0
            t.sam[0 * n + 5] = bits(planted[1]); // row 1
            t.pos[2 * n + 7] = bits(planted[2]); // row 2
            t.sam[2 * n + 9] = bits(planted[1]); // row 3, both infinities
            t.pos[3 * n + 9] = bits(planted[2]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.cls));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.sam));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.pos));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *ClipEmbed) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.setValue(@as(u32, @intCast(t.shape.n)), 4);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *ClipEmbed) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *ClipEmbed, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / n;
            const v = if (row == 0) val(t.cls[i]) else val(t.sam[i - n]);
            o.* = bf(v + val(t.pos[i]));
            m.* = 0;
        }
    }
};

// --- ud_gather and ud_scatter: an expert's rows by its index list, y[i] = x[rows[i]] and slots[dest[i]] = y[i] -------

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    scatter: bool,
    ctl: ?Control,
    shape: c.Shape = undefined,
    src: []u16 = &.{},
    ids: []u32 = &.{},
    /// ud_scatter's slot rows before the launch: the rows no index names keep them.
    bg: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: the expert's rows (1 to a chunk's 2,048); n: the hidden width (the kernels' 1280); k: the table the
    /// indices reach (gather: the chunk's token rows; scatter: its slots, top-6 a row). Indices are distinct (each
    /// expert takes a token once, each slot one expert) and in random order.
    const gather_served = [_]c.Shape{ .{ .rows = 192, .n = hidden, .k = chunk }, .{ .rows = chunk, .n = hidden, .k = chunk }, .{ .rows = 1, .n = hidden, .k = 1 } };
    const scatter_served = [_]c.Shape{ .{ .rows = 192, .n = hidden, .k = slots }, .{ .rows = chunk, .n = hidden, .k = slots }, .{ .rows = 1, .n = hidden, .k = uocr.topk } };

    pub fn spec(scatter: bool) c.Spec {
        return .{
            .name = if (scatter) "metal uocr moves ud_scatter 1280 (slots[dest[i]] = y[i])" else "metal uocr moves ud_gather 1280 (y[i] = x[rows[i]])",
            .accum = .none,
            .depth = uocr.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 64 }, .{}, .{} },
            .served = if (scatter) &scatter_served else &gather_served,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, scatter: bool, ctl: ?Control) !Rows {
        const name = if (scatter) "ud_scatter" else "ud_gather";
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .decoder, name, if (ctl) |x| x.edit else null), .scatter = scatter, .ctl = ctl };
    }
    pub fn deinit(t: *Rows) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Rows) void {
        inline for (.{ "src", "ids", "bg" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Rows) g.Kernel {
        var s = spec(t.scatter);
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(Rows);

    pub fn prepare(t: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != hidden or shape.rows > shape.k) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = hidden;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gather reads table rows by index; scatter reads its own rows in order
        const src_rows = if (t.scatter) shape.rows else shape.k;
        t.src = try t.a.alloc(u16, src_rows * n);
        for (t.src) |*v| v.* = finiteBits(rng);
        t.ids = try distinct(t.a, rng, shape.rows, shape.k);
        t.bg = try t.a.alloc(u16, if (t.scatter) shape.k * n else 0);
        for (t.bg) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite and shape.rows > 3) {
            // the rows the first four indices move: NaN, +Inf, -Inf, both infinities
            const moved = [_]usize{ 0, 1, 2, 3 };
            for (moved, 0..) |j, x| {
                const row = if (t.scatter) j else t.ids[j];
                if (x < 3) t.src[row * n + 11 * x] = bits(planted[x]) else {
                    t.src[row * n + 40] = bits(planted[1]);
                    t.src[row * n + n - 1] = bits(planted[2]);
                }
            }
            // gather: a NaN in a row no index names must stay out
            if (!t.scatter and shape.k > shape.rows) {
                const seen = try t.a.alloc(bool, shape.k);
                defer t.a.free(seen);
                @memset(seen, false);
                for (t.ids) |i| seen[i] = true;
                const free_row = std.mem.indexOfScalar(bool, seen, false).?;
                t.src[free_row * n] = bits(planted[0]);
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.src));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Rows) !void {
        if (t.scatter) @memcpy(t.rig.bytes(t.out, t.bg.len * 2), std.mem.sliceAsBytes(t.bg));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.dispatchThreads(mtl.Size.of(hidden, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Rows) usize {
        return (if (t.scatter) t.shape.k else t.shape.rows) * hidden;
    }

    pub fn reference(t: *Rows, want: []f64, mag: []f64) !void {
        const n = hidden;
        @memset(mag, 0);
        if (t.scatter) {
            for (want, t.bg) |*o, v| o.* = val(v);
            for (t.ids, 0..) |d, i| for (0..n) |j| {
                want[d * n + j] = val(t.src[i * n + j]);
            };
            return;
        }
        for (want, 0..) |*o, i| o.* = val(t.src[t.ids[i / n] * n + i % n]);
    }
};

// --- the group -------------------------------------------------------------------------------------------------------

/// The planted controls: each edits one index or padding rule (or the sums' type) and must go red at its gate.
const im2col_controls = [_]Control{
    .{ .name = "metal uocr moves control uv_im2col padding 1 not 0", .edit = .{ .from = "bfloat v = bfloat(0.0f);", .to = "bfloat v = bfloat(1.0f);" } },
    .{ .name = "metal uocr moves control uv_im2col ky and kx swapped", .edit = .{ .from = "ky = (at.x / p.k) % p.k, kx = at.x % p.k;", .to = "kx = (at.x / p.k) % p.k, ky = at.x % p.k;" } },
};
const window_control = Control{ .name = "metal uocr moves control uv_window rows and columns swapped in a window", .edit = .{
    .from = "gy = (win / per) * p.window + in / p.window, gx = (win % per) * p.window + in % p.window;",
    .to = "gy = (win / per) * p.window + in % p.window, gx = (win % per) * p.window + in / p.window;",
} };
const window_pad_control = Control{ .name = "metal uocr moves control uv_window padding 1 not 0", .edit = .{
    .from = "p.channels + at.x] : bfloat(0.0f);",
    .to = "p.channels + at.x] : bfloat(1.0f);",
} };
const unwindow_control = Control{ .name = "metal uocr moves control uv_unwindow_add rows and columns swapped in a window", .edit = .{
    .from = "in = (gy % p.window) * p.window + gx % p.window;",
    .to = "in = (gx % p.window) * p.window + gy % p.window;",
} };
const rel_controls = [_]Control{
    .{ .name = "metal uocr moves control uv_rel_pos rel_h and rel_w positions swapped", .edit = .{
        .from = "((rows ? qh : qw) - kk + p.side - 1)",
        .to = "((rows ? qw : qh) - kk + p.side - 1)",
    } },
    .{ .name = "metal uocr moves control uv_rel_pos summed in half", .gate = .sums, .edit = .{
        .from = "float s = 0.0f;\n  for (uint d = 0; d < 64; d++) s += float(q[d]) * float(r[d]);",
        .to = "half s = 0.0h;\n  for (uint d = 0; d < 64; d++) s += half(float(q[d]) * float(r[d]));",
    } },
};
const concat_control = Control{ .name = "metal uocr moves control uv_concat left row one column wider", .edit = .{ .from = "at.x < p.left ?", .to = "at.x <= p.left ?" } };
const clip_control = Control{ .name = "metal uocr moves control uv_clip_embed class row takes position 1", .edit = .{
    .from = "float(pos[ulong(at.y) * width + at.x])",
    .to = "float(pos[ulong(max(at.y, 1u)) * width + at.x])",
} };
const gather_control = Control{ .name = "metal uocr moves control ud_gather odd rows read the even row's index", .edit = .{ .from = "x[ulong(rows[at.y]) * 1280", .to = "x[ulong(rows[at.y & ~1u]) * 1280" } };
const scatter_control = Control{ .name = "metal uocr moves control ud_scatter odd rows written to the even row's slot", .edit = .{ .from = "slots[ulong(dest[at.y]) * 1280", .to = "slots[ulong(dest[at.y & ~1u]) * 1280" } };

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    im2col: std.ArrayList(*Im2col) = .empty,
    windows: std.ArrayList(*Window) = .empty,
    rel: std.ArrayList(*RelPos) = .empty,
    concat: std.ArrayList(*Concat) = .empty,
    clip: std.ArrayList(*ClipEmbed) = .empty,
    rows: std.ArrayList(*Rows) = .empty,

    fn wanted(only: ?[]const u8, name: []const u8) bool {
        const o = only orelse return true;
        return std.mem.indexOf(u8, name, o) != null;
    }

    /// One test of T (T.init with these args) when its name is wanted.
    fn add(s: *Suite, comptime T: type, list: *std.ArrayList(*T), only: ?[]const u8, name: []const u8, args: anytype) !void {
        if (!wanted(only, name)) return;
        const x = try s.a.create(T);
        errdefer s.a.destroy(x);
        x.* = try @call(.auto, T.init, .{ s.a, s.device, s.queue } ++ args);
        try list.append(s.a, x);
        try s.kernels.append(s.a, x.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (geos) |geo| try s.add(Im2col, &s.im2col, only, geo.name, .{ geo, @as(?Control, null) });
        // the controls run at the neck's geometry (padding on every side) and the patches' (no padding: the swap)
        for (im2col_controls, 0..) |ctl, i| try s.add(Im2col, &s.im2col, only, ctl.name, .{ geos[if (i == 0) 1 else 0], @as(?Control, ctl) });
        for ([_]bool{ false, true }) |back| try s.add(Window, &s.windows, only, Window.spec(back).name, .{ back, @as(?Control, null) });
        try s.add(Window, &s.windows, only, window_control.name, .{ false, @as(?Control, window_control) });
        try s.add(Window, &s.windows, only, window_pad_control.name, .{ false, @as(?Control, window_pad_control) });
        try s.add(Window, &s.windows, only, unwindow_control.name, .{ true, @as(?Control, unwindow_control) });
        try s.add(RelPos, &s.rel, only, RelPos.spec().name, .{@as(?Control, null)});
        for (rel_controls) |ctl| try s.add(RelPos, &s.rel, only, ctl.name, .{@as(?Control, ctl)});
        try s.add(Concat, &s.concat, only, Concat.spec().name, .{@as(?Control, null)});
        try s.add(Concat, &s.concat, only, concat_control.name, .{@as(?Control, concat_control)});
        try s.add(ClipEmbed, &s.clip, only, ClipEmbed.spec().name, .{@as(?Control, null)});
        try s.add(ClipEmbed, &s.clip, only, clip_control.name, .{@as(?Control, clip_control)});
        for ([_]bool{ false, true }) |scatter| try s.add(Rows, &s.rows, only, Rows.spec(scatter).name, .{ scatter, @as(?Control, null) });
        try s.add(Rows, &s.rows, only, gather_control.name, .{ false, @as(?Control, gather_control) });
        try s.add(Rows, &s.rows, only, scatter_control.name, .{ true, @as(?Control, scatter_control) });
    }

    pub fn deinit(s: *Suite) void {
        inline for (.{ "im2col", "windows", "rel", "concat", "clip", "rows" }) |f| {
            for (@field(s, f).items) |x| {
                x.deinit();
                s.a.destroy(x);
            }
            @field(s, f).deinit(s.a);
        }
    }
};

/// Gate 5's texts for this group's launches (`uocr moves <case> <function>`): the tower's library for uv_*, the
/// decoder's for ud_*.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?uocr.Texts {
    if (!std.mem.startsWith(u8, name, "uocr moves ")) return null;
    const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    const which: uocr.Library = if (std.mem.startsWith(u8, fun, "ud_")) .decoder else if (std.mem.startsWith(u8, fun, "uv_")) .vision else return error.UnknownLaunch;
    const text = uocr.source(which);
    return .{ .here = text, .pre_m5 = text };
}
