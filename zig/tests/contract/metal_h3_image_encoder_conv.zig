//! hie_conv under the contract: the encoder's convolutions (3x3, 3x3 adding in place, 3x3 down, 1x1) against float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("metal_h3_image_encoder_ref.zig");

const c = ref.c;
const g = ref.g;
const threads = contract.reference_threads;

/// A convolution the encoder launches: its kernel side, stride 2 or not, the residual added in place or not.
pub const Form = struct { label: []const u8, side: usize, down: bool = false, add: bool = false, tiles: [3]c.Tile, served: []const c.Shape };

/// Rows: the tile's side (square, as the release cuts every tile at 256); n: output channels; k: input channels.
fn sq(side: usize, cout: usize, cin: usize) c.Shape {
    return .{ .rows = side, .n = cout, .k = cin };
}

/// 8 x 8 output pixels fill the kernel's 64-pixel tile; outputs in 16s around its 64 channels; inputs in steps of 32.
const pixel_tile: c.Tile = .{ .size = 8, .min = 2 };
const channel_tile: c.Tile = .{ .size = 64, .step = 16, .min = 16 };
const input_tile: c.Tile = .{ .size = 32 };

/// Every layer geometry of the released encoder on its 256-pixel tiles, each form's first the edges' base.
pub const forms = [_]Form{
    // conv_out, conv_in, then each level's conv1
    .{ .label = "3x3", .side = 3, .tiles = .{ pixel_tile, channel_tile, input_tile }, .served = &.{ sq(16, 48, 1024), sq(256, 128, 3), sq(256, 128, 128), sq(128, 256, 128), sq(128, 256, 256), sq(64, 256, 256), sq(32, 512, 256), sq(32, 512, 512), sq(16, 512, 512), sq(16, 1024, 512), sq(16, 1024, 1024) } },
    // each level's conv2, adding into the block's input
    .{ .label = "3x3 + residual", .side = 3, .add = true, .tiles = .{ pixel_tile, .{}, .{} }, .served = &.{ sq(256, 128, 128), sq(128, 256, 256), sq(64, 256, 256), sq(32, 512, 512), sq(16, 512, 512), sq(16, 1024, 1024) } },
    // levels 0-3's downsample: two input rows a pixel row, so 16 rows fill the tile
    .{ .label = "3x3 down", .side = 3, .down = true, .tiles = .{ .{ .size = 16, .step = 2, .min = 2 }, .{}, .{} }, .served = &.{ sq(256, 128, 128), sq(128, 256, 256), sq(64, 256, 256), sq(32, 512, 512) } },
    // quant_conv, then the shortcuts
    .{ .label = "1x1", .side = 1, .tiles = .{ .{ .size = 8 }, channel_tile, input_tile }, .served = &.{ sq(16, 48, 48), sq(128, 256, 128), sq(32, 512, 256), sq(16, 1024, 512) } },
};

/// Each output's taps x cin products in one chain at worst, then its bias and residual.
fn depth9(s: c.Shape) usize {
    return 9 * s.k + 2;
}
fn depth1(s: c.Shape) usize {
    return s.k + 2;
}

/// The kernel's HieConv constants.
const HieConv = extern struct { h: u32, w: u32, oh: u32, ow: u32, cin: u32, cout: u32, ldw: u32, k: u32, stride: u32, pad: u32, add: u32 };

/// Index i of an axis of n read with reflect padding (-1 reads 1, n reads n - 2).
fn reflect(i: isize, n: usize) usize {
    const last: isize = @intCast(n - 1);
    return @intCast(if (i < 0) -i else if (i > last) 2 * last - i else i);
}

pub const Conv = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    form: *const Form,
    name: []const u8,
    shape: c.Shape = undefined,
    /// x [side * side, cin], w [taps, cin, ldw], b [cout], the residual [pixels, cout] y starts from
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    r: []f32 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn spec(name: []const u8, f: *const Form) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = if (f.side == 3) depth9 else depth1, .out = .f32, .in_place = f.add, .tiles = f.tiles, .served = f.served };
    }

    pub fn kernel(t: *Conv, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Conv);

    pub fn deinit(t: *Conv) void {
        t.free();
        t.rig.deinit();
    }

    fn free(t: *Conv) void {
        inline for (.{ &t.x, &t.w, &t.b, &t.r }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }

    const Geo = struct { side: usize, out: usize, stride: usize, pad: usize, taps: usize, ldw: usize };

    /// The launcher's sizes: stride 2 halves the side, a stride-1 3x3 pads by one, weight rows padded to 64.
    fn geo(t: *const Conv) Geo {
        const s = t.shape.rows;
        return .{
            .side = s,
            .out = if (t.form.down) s / 2 else s,
            .stride = if (t.form.down) 2 else 1,
            .pad = if (t.form.side == 3 and !t.form.down) 1 else 0,
            .taps = t.form.side * t.form.side,
            .ldw = std.mem.alignForward(usize, t.shape.n, 64),
        };
    }

    pub fn prepare(t: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const least: usize = if (t.form.side == 3) 2 else 1;
        if (shape.rows < least or shape.n == 0 or shape.k == 0 or (t.form.down and shape.rows % 2 != 0)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const e = t.geo();
        t.x = try t.a.alloc(f32, e.side * e.side * shape.k);
        t.w = try t.a.alloc(f32, e.taps * shape.k * e.ldw);
        t.b = try t.a.alloc(f32, shape.n);
        t.r = try t.a.alloc(f32, if (t.form.add) e.out * e.out * shape.n else 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the weights' padded columns stay zero, as packLayers leaves them
        @memset(t.w, 0);
        if (fill == .stress) t.stress(rng) else {
            ref.uniform(rng, t.x, -2, 2);
            for (0..e.taps * shape.k) |row| ref.uniform(rng, t.w[row * e.ldw ..][0..shape.n], -0.1, 0.1);
            ref.uniform(rng, t.b, -1, 1);
            ref.uniform(rng, t.r, -3, 3);
            if (fill == .nonfinite) t.plant();
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    /// NaN, +Inf and -Inf at three pixels, both infinities at the last one, a +Inf bias and a NaN residual.
    fn plant(t: *Conv) void {
        const side = t.shape.rows;
        const cin = t.shape.k;
        const px = side * side;
        const at = [_]usize{ 0, 2 * side + 4, 3 * side + 1 };
        for (at, [_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |p, v, i| if (p < px) {
            t.x[p * cin + i % cin] = v;
        };
        t.x[(px - 1) * cin] = ref.inf;
        t.x[(px - 1) * cin + cin - 1] = -ref.inf;
        if (t.shape.n > 1) t.b[t.shape.n - 1] = ref.inf;
        if (t.r.len > 0) t.r[t.r.len / 2] = ref.nan;
    }

    /// Gate 4: integers, each tap's second half of channels its first (4 nudged by one) under negated weights.
    fn stress(t: *Conv, rng: std.Random) void {
        const cin = t.shape.k;
        const e = t.geo();
        const half = cin / 2;
        // terms reach 512 x 2 and number taps x cin at most 9 x 1024: every partial sum stays below 2^24
        for (0..e.side * e.side) |p| {
            const x = t.x[p * cin ..][0..cin];
            @memset(x, 0);
            for (0..half) |i| {
                x[i] = ref.int(rng, 100, 511);
                x[half + i] = x[i] + @as(f32, if (i < 4) 1 else 0);
            }
        }
        for (0..e.taps) |tap| for (0..half) |i| for (0..t.shape.n) |o| {
            const v = ref.int(rng, 1, 2) * @as(f32, if (o % 3 == 1) -1 else 1);
            t.w[(tap * cin + i) * e.ldw + o] = v;
            t.w[(tap * cin + half + i) * e.ldw + o] = -v;
        };
        for (t.b) |*v| v.* = ref.int(rng, -3, 3);
        for (t.r) |*v| v.* = ref.int(rng, -3, 3);
    }

    /// One launch as image_encoder.zig's `conv` binds and dispatches it (y at r's index when no residual is added).
    pub fn run(t: *Conv) !void {
        if (t.form.add) @memcpy(t.rig.bytes(t.out, t.r.len * 4), std.mem.sliceAsBytes(t.r));
        const e = t.geo();
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| ref.bind(p.e, r, i);
        ref.bind(p.e, t.out, 3);
        ref.bind(p.e, t.out, 4);
        const s: HieConv = .{
            .h = @intCast(e.side),
            .w = @intCast(e.side),
            .oh = @intCast(e.out),
            .ow = @intCast(e.out),
            .cin = @intCast(t.shape.k),
            .cout = @intCast(t.shape.n),
            .ldw = @intCast(e.ldw),
            .k = @intCast(t.form.side),
            .stride = @intCast(e.stride),
            .pad = @intCast(e.pad),
            .add = @intFromBool(t.form.add),
        };
        p.e.setValue(s, 5);
        p.e.dispatchGroups(mtl.Size.of((e.out * e.out + 63) / 64, e.ldw / 64, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Conv) usize {
        const e = t.geo();
        return e.out * e.out * t.shape.n;
    }

    /// The input channels tap `tap` of output pixel p reads, through reflect padding.
    fn source(t: *const Conv, e: Geo, p: usize, tap: usize) []const f32 {
        const oy = p / e.out;
        const ox = p % e.out;
        const pad: isize = @intCast(e.pad);
        const iy = reflect(@as(isize, @intCast(oy * e.stride + tap / t.form.side)) - pad, e.side);
        const ix = reflect(@as(isize, @intCast(ox * e.stride + tap % t.form.side)) - pad, e.side);
        return t.x[(iy * e.side + ix) * t.shape.k ..][0..t.shape.k];
    }

    /// Output pixel p's every channel in float64: the sums and their terms' magnitude.
    fn exact(t: *const Conv, p: usize, want: []f64, mag: []f64, buf: []f64) void {
        const e = t.geo();
        const cout = t.shape.n;
        const sum = buf[0..cout];
        const abs = buf[cout..][0..cout];
        @memset(sum, 0);
        @memset(abs, 0);
        for (0..e.taps) |tap| for (t.source(e, p, tap), 0..) |xv, ci| {
            const xf: f64 = xv;
            for (t.w[(tap * t.shape.k + ci) * e.ldw ..][0..cout], sum, abs) |wv, *s, *m| {
                const prod = xf * wv;
                s.* += prod;
                m.* += @abs(prod);
            }
        };
        for (0..cout) |o| {
            const i = p * cout + o;
            const rv: f64 = if (t.form.add) t.r[i] else 0;
            want[i] = sum[o] + t.b[o] + rv;
            const m = abs[o] + @abs(@as(f64, t.b[o])) + @abs(rv);
            mag[i] = if (std.math.isFinite(m)) m else 0;
        }
    }

    /// Output pixel p's every channel with its products summed in `acc`, in `order` over (tap, input channel).
    fn simulated(t: *const Conv, p: usize, acc: c.Accum, order: g.Order, out: []f64, buf: []f64) void {
        const e = t.geo();
        const cout = t.shape.n;
        for (0..cout) |o| {
            var n: usize = 0;
            for (0..e.taps) |tap| for (t.source(e, p, tap), 0..) |xv, ci| {
                buf[n] = @as(f64, xv) * t.w[(tap * t.shape.k + ci) * e.ldw + o];
                n += 1;
            };
            const i = p * cout + o;
            const rv: f64 = if (t.form.add) t.r[i] else 0;
            out[i] = contract.rows.sumIn(acc, order, buf[0..n]) + t.b[o] + rv;
        }
    }

    /// A share of the output pixels, computed on a thread of its own.
    const Job = struct {
        t: *const Conv,
        lo: usize,
        hi: usize,
        acc: ?c.Accum,
        order: g.Order,
        want: []f64,
        mag: []f64,
        buf: []f64,

        pub fn run(j: *Job) void {
            for (j.lo..j.hi) |p| if (j.acc) |acc| j.t.simulated(p, acc, j.order, j.want, j.buf) else j.t.exact(p, j.want, j.mag, j.buf);
        }
    };

    fn each(t: *Conv, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const e = t.geo();
        const pixels = e.out * e.out;
        const n = threads.count(threads.maximum, pixels);
        const per = if (acc == null) 2 * t.shape.n else e.taps * t.shape.k;
        const buf = try t.a.alloc(f64, n * per);
        defer t.a.free(buf);
        var jobs: [threads.maximum]Job = undefined;
        for (jobs[0..n], 0..) |*j, i| j.* = .{ .t = t, .lo = pixels * i / n, .hi = pixels * (i + 1) / n, .acc = acc, .order = order, .want = want, .mag = mag, .buf = buf[i * per ..][0..per] };
        threads.run(Job, jobs[0..n], .normal);
    }

    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        return t.each(null, .sequential, want, mag);
    }

    pub fn simulate(t: *Conv, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.each(acc, order, out, out);
    }
};
