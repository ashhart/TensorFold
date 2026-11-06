//! moss_image_decoder.cu's tf_md_conv3x3 under the contract: a 3 x 3 convolution's taps, padded at the picture's edges.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const md = moss.decoder;
const c = mi.c;
const g = mi.g;

/// One of the decoder's launches: the output picture, its channels, x read nearest from half size, a residual.
const Call = struct {
    rows: usize,
    cols: usize,
    cin: usize,
    cout: usize,
    up: bool = false,
    add: bool = false,
};

/// A picture's (the latent grid's 8 x) launches in `run`'s order: conv_in, the 384 blocks, each level's up and blocks.
fn picture(rows: usize, cols: usize) [15]Call {
    return .{
        .{ .rows = rows, .cols = cols, .cin = 32, .cout = 384 },
        // a block's residual-adding call comes first: a shape carries no flags, and callFor takes the first match
        .{ .rows = rows, .cols = cols, .cin = 384, .cout = 384, .add = true },
        .{ .rows = rows, .cols = cols, .cin = 384, .cout = 384 },
        .{ .rows = 2 * rows, .cols = 2 * cols, .cin = 384, .cout = 384, .up = true },
        .{ .rows = 2 * rows, .cols = 2 * cols, .cin = 384, .cout = 384, .add = true },
        .{ .rows = 2 * rows, .cols = 2 * cols, .cin = 384, .cout = 384 },
        .{ .rows = 4 * rows, .cols = 4 * cols, .cin = 384, .cout = 384, .up = true },
        .{ .rows = 4 * rows, .cols = 4 * cols, .cin = 384, .cout = 192 },
        .{ .rows = 4 * rows, .cols = 4 * cols, .cin = 192, .cout = 192, .add = true },
        .{ .rows = 4 * rows, .cols = 4 * cols, .cin = 192, .cout = 192 },
        .{ .rows = 8 * rows, .cols = 8 * cols, .cin = 192, .cout = 192, .up = true },
        .{ .rows = 8 * rows, .cols = 8 * cols, .cin = 192, .cout = 96 },
        .{ .rows = 8 * rows, .cols = 8 * cols, .cin = 96, .cout = 96, .add = true },
        .{ .rows = 8 * rows, .cols = 8 * cols, .cin = 96, .cout = 96 },
        // conv_out's three outputs, the same shape: with one call a shape gate 3 runs the block's 96 above, not this
        .{ .rows = 8 * rows, .cols = 8 * cols, .cin = 96, .cout = 3 },
    };
}

/// The decoder's launches at its three served pictures: 416 x 224, 512 x 640 and 1024 x 1024.
const calls = picture(52, 28) ++ picture(64, 80) ++ picture(128, 128);

fn servedShapes() [calls.len]c.Shape {
    var out: [calls.len]c.Shape = undefined;
    for (&out, calls) |*o, call| o.* = .{ .rows = call.rows, .n = call.cols, .k = call.cin };
    return out;
}

/// The served shapes, in the order the decoder launches them: the channels the input brings are their k.
const served = servedShapes();

/// The launch of a prepared shape: the decoder's own call for a served one, else the square form the edges probe.
fn callFor(shape: c.Shape) Call {
    for (calls) |call| if (call.rows == shape.rows and call.cols == shape.n and call.cin == shape.k) return call;
    return .{ .rows = shape.rows, .cols = shape.n, .cin = shape.k, .cout = shape.k };
}

/// The products and the sum they feed, rounded once each, then the bias and the residual.
fn depth(s: c.Shape) usize {
    return 9 * s.k + 2;
}

pub const Conv = struct {
    rig: rig_.Rig,
    k: md.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    call: Call = undefined,
    /// x [the input's pixels, cin] and the taps [9, cin, cout]
    x: []f32 = &.{},
    u: []f32 = &.{},
    /// bias [cout] and the residual [rows * cols, cout] the call adds
    v: []f32 = &.{},
    ins: [4]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: the output's rows; n: its columns; k: the input's channels.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            // the bias's and the residual's adds (a call without the residual takes one fewer)
            .ops = 2,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{ .size = 16 }, .{ .size = 16, .step = 16, .min = 16 } },
            .served = &served,
        };
    }

    pub fn deinit(t: *Conv) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Conv) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.u);
        a.free(t.v);
        t.x = &.{};
        t.u = &.{};
        t.v = &.{};
    }
    pub fn kernel(t: *Conv) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Conv);

    /// The input's columns: half the picture's when the taps read it nearest from half size.
    fn inCols(t: *const Conv) usize {
        return if (t.call.up) t.shape.n / 2 else t.shape.n;
    }

    pub fn prepare(t: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        const call = callFor(shape);
        if (call.up and (shape.n % 2 != 0 or shape.rows % 2 != 0)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.call = call;
        const a = t.rig.gpu.gpa;
        t.x = try a.alloc(f32, t.inCols() * (if (call.up) shape.rows / 2 else shape.rows) * shape.k);
        t.u = try a.alloc(f32, 9 * shape.k * call.cout);
        t.v = try a.alloc(f32, call.cout + shape.rows * shape.n * call.cout);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gate 4 needs whole values: whole taps of whole values sum exactly in fp32, in any order
        for (t.x) |*e| e.* = switch (fill) {
            .stress => @floatFromInt(rng.intRangeAtMost(i32, -32, 32)),
            else => 2 * rng.floatNorm(f32),
        };
        for (t.u) |*e| e.* = switch (fill) {
            .stress => @floatFromInt(rng.intRangeAtMost(i32, -32, 32)),
            else => 0.5 * rng.floatNorm(f32),
        };
        for (t.v[0..call.cout]) |*e| e.* = if (fill == .stress) @floatFromInt(rng.intRangeAtMost(i32, -8, 8)) else 0.5 * rng.floatNorm(f32);
        for (t.v[call.cout..]) |*e| e.* = if (fill == .stress) @floatFromInt(rng.intRangeAtMost(i32, -8, 8)) else rng.floatNorm(f32);
        if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| {
                t.x[(i * t.x.len / 3 + 5) % t.x.len] = v;
                t.u[(i * 11 + 3) % t.u.len] = v;
            }
            if (call.cout > 2) t.v[2] = std.math.nan(f32);
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.u);
        t.ins[2] = try mi.input(&t.rig, t.v[0..call.cout]);
        t.ins[3] = try mi.input(&t.rig, t.v[call.cout..]);
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Conv) !void {
        const shape: md.ConvShape = .{ .rows = t.shape.rows, .cols = t.shape.n, .cin = t.shape.k, .cout = t.call.cout, .up = t.call.up };
        try md.conv3x3(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.out, shape, t.call.add);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Conv) usize {
        return t.shape.rows * t.shape.n * t.call.cout;
    }

    /// The input pixel the tap (dy, dx) of output (oy, ox) reads, or null where the window is past the picture.
    fn source(t: *const Conv, oy: usize, ox: usize, tap: usize) ?usize {
        const dy = @as(isize, @intCast(tap / 3)) - 1;
        const dx = @as(isize, @intCast(tap % 3)) - 1;
        const y = @as(isize, @intCast(oy)) + dy;
        const x = @as(isize, @intCast(ox)) + dx;
        if (y < 0 or x < 0 or y >= @as(isize, @intCast(t.shape.rows)) or x >= @as(isize, @intCast(t.shape.n))) return null;
        const uy: usize = @intCast(y);
        const ux: usize = @intCast(x);
        const sy = if (t.call.up) uy / 2 else uy;
        const sx = if (t.call.up) ux / 2 else ux;
        return (sy * t.inCols() + sx) * t.shape.k;
    }

    /// Tap `tap`'s `ch` of (oy, ox) times its weight: the kernel's zero patch is multiplied too, so 0 * inf is NaN.
    fn product(t: *const Conv, oy: usize, ox: usize, tap: usize, ch: usize, col: usize) f64 {
        const from = t.source(oy, ox, tap);
        const x: f64 = if (from) |f| t.x[f + ch] else 0;
        return x * @as(f64, t.u[(tap * t.shape.k + ch) * t.call.cout + col]);
    }

    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        const cols = t.shape.n;
        const cin = t.shape.k;
        const cout = t.call.cout;
        // a pixel's taps once then a sweep over the outputs: each output sums in tap, then channel order, as before
        for (0..t.shape.rows) |oy| for (0..cols) |ox| {
            const o = (oy * cols + ox) * cout;
            const sum = want[o..][0..cout];
            const terms = mag[o..][0..cout];
            @memset(sum, 0);
            @memset(terms, 0);
            for (0..9) |tap| {
                const from = t.source(oy, ox, tap);
                for (0..cin) |ch| {
                    const x: f64 = if (from) |f| t.x[f + ch] else 0;
                    for (sum, terms, t.u[(tap * cin + ch) * cout ..][0..cout]) |*s, *m, w| {
                        const p = x * @as(f64, w);
                        s.* += p;
                        m.* += @abs(p);
                    }
                }
            }
            for (sum, 0..) |s, col| sum[col] = s + t.v[col] + (if (t.call.add) t.v[cout + o + col] else 0);
        };
    }

    pub fn simulate(t: *Conv, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const cols = t.shape.n;
        const cin = t.shape.k;
        const cout = t.call.cout;
        const terms = try a.alloc(f64, 9 * cin);
        defer a.free(terms);
        for (0..t.shape.rows) |oy| for (0..cols) |ox| for (0..cout) |col| {
            var at: usize = 0;
            for (0..9) |tap| for (0..cin) |ch| {
                terms[at] = t.product(oy, ox, tap, ch, col);
                at += 1;
            };
            var v: f32 = @floatCast(mi.rows.sumIn(acc, order, terms[0..at]));
            v += t.v[col];
            if (t.call.add) v += t.v[cout + (oy * cols + ox) * cout + col];
            out[(oy * cols + ox) * cout + col] = v;
        };
    }
};

/// The decoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: md.Kernels,
    t: [2]Conv = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try md.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        var with = k;
        with.conv3x3 = try ctl.function("tf_ctl_md_conv3x3_bias");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image decoder tf_md_conv3x3" };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image decoder tf_md_conv3x3 drops its bias", .control = .shapes };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
