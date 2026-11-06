//! moss_image_decoder.cu's GroupNorm statistics: tf_md_group_part's chunk sums, tf_md_group_finish's mean or rstd.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const md = moss.decoder;
const c = mi.c;
const g = mi.g;

/// The decoder's two statistics passes: the chunks' sums of the values (the mean's), and of (x - mean)^2.
pub const plain = 1;
pub const centered = 2;

fn isCentered(shape: c.Shape) ?bool {
    return switch (shape.k) {
        plain => false,
        centered => true,
        else => null,
    };
}

/// The (pixels, channels) pairs its GroupNorm runs at: each picture's four levels at 384, 384, then 192 and 96.
const pairs = [_][2]usize{
    .{ 1456, 384 }, .{ 5824, 384 }, .{ 23296, 384 }, .{ 23296, 192 }, .{ 93184, 192 }, .{ 93184, 96 }, // 416 x 224
    .{ 5120, 384 },  .{ 20480, 384 }, .{ 81920, 384 },  .{ 81920, 192 },  .{ 327680, 192 },  .{ 327680, 96 }, // 512 x 640
    .{ 16384, 384 }, .{ 65536, 384 }, .{ 262144, 384 }, .{ 262144, 192 }, .{ 1048576, 192 }, .{ 1048576, 96 },
};

/// The GroupNorm launches: both passes of every pair, the mean's before the rstd's as the decoder makes them.
const served = servedShapes();

fn servedShapes() [2 * pairs.len]c.Shape {
    var out: [2 * pairs.len]c.Shape = undefined;
    for (pairs, 0..) |p, i| {
        out[2 * i] = .{ .rows = p[0], .n = p[1], .k = plain };
        out[2 * i + 1] = .{ .rows = p[0], .n = p[1], .k = centered };
    }
    return out;
}

/// A chunk's pixels of a group, summed a term at a time in fp32: the chunk's own values then its channels.
fn partDepth(s: c.Shape) usize {
    return md.chunk * (s.n / 32) + 3;
}

// --- tf_md_group_part: a chunk's sums of a group's values, or of their squares about the mean ---------------------

pub const Part = struct {
    rig: rig_.Rig,
    k: md.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    centered: bool = false,
    /// x [pixels, channels] and the statistics' mean | rstd [64], as fp32 bits
    x: []f32 = &.{},
    s: []f32 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: pixels; n: channels (a multiple of 384's 12 a group); k: the pass, plain or centered.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = partDepth,
            .out = .f32,
            .tiles = .{ .{ .size = md.chunk, .step = md.chunk, .min = md.chunk }, .{ .size = 32, .step = 32, .min = 32 }, .{} },
            .served = &served,
        };
    }

    pub fn deinit(t: *Part) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Part) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.s);
        t.x = &.{};
        t.s = &.{};
    }
    pub fn kernel(t: *Part) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Part);

    pub fn prepare(t: *Part, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const want = isCentered(shape) orelse return false;
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.centered = want;
        const a = t.rig.gpu.gpa;
        t.x = try a.alloc(f32, shape.rows * shape.n);
        t.s = try a.alloc(f32, 64);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // GroupNorm's inputs about N(0, 1); gate 4 needs whole values, whose squares sum exactly in fp32
        for (t.x) |*e| e.* = switch (fill) {
            .stress => @floatFromInt(rng.intRangeAtMost(i32, -32, 32)),
            else => rng.floatNorm(f32),
        };
        // a mean of 0 under the stress fill: (x - 0)^2 is a whole value too
        for (t.s[0..32]) |*e| e.* = if (fill == .stress) 0 else 0.1 * rng.floatNorm(f32);
        for (t.s[32..]) |*e| e.* = 0.5 + rng.float(f32) * 1.5;
        if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| {
                t.x[(i * 37 + 5) % t.x.len] = v;
            }
            t.s[3] = std.math.nan(f32);
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.s);
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Part) !void {
        try md.groupPart(&t.k, t.rig.s, t.ins[0], t.ins[1], t.out, t.shape.rows, t.shape.n, t.centered);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Part) usize {
        return 32 * md.chunks(t.shape.rows);
    }

    /// Group `grp`'s chunk `tg` of the prepared inputs: its float64 sum, and the terms' magnitude the bound needs.
    fn chunk(t: *const Part, grp: usize, tg: usize) struct { sum: f64, mag: f64 } {
        const channels = t.shape.n;
        const size = channels / 32;
        const first = tg * md.chunk;
        const last = @min(first + md.chunk, t.shape.rows);
        const mean: f64 = if (t.centered) t.s[grp] else 0;
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..size) |i| {
            const ch = grp * size + i;
            for (first..last) |px| {
                const v: f64 = t.x[px * channels + ch];
                const term = if (t.centered) (v - mean) * (v - mean) else v;
                sum += term;
                mag += @abs(term);
            }
        }
        return .{ .sum = sum, .mag = mag };
    }

    pub fn reference(t: *Part, want: []f64, mag: []f64) !void {
        for (0..32) |grp| for (0..md.chunks(t.shape.rows)) |tg| {
            const s = t.chunk(grp, tg);
            want[grp * md.chunks(t.shape.rows) + tg] = s.sum;
            mag[grp * md.chunks(t.shape.rows) + tg] = s.mag;
        };
    }

    pub fn simulate(t: *Part, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const channels = t.shape.n;
        const size = channels / 32;
        const chunks = md.chunks(t.shape.rows);
        const terms = try a.alloc(f64, md.chunk * size);
        defer a.free(terms);
        for (0..32) |grp| for (0..chunks) |tg| {
            const mean: f64 = t.s[grp];
            const first = tg * md.chunk;
            const last = @min(first + md.chunk, t.shape.rows);
            var at: usize = 0;
            for (0..size) |i| {
                const ch = grp * size + i;
                for (first..last) |px| {
                    const v: f64 = t.x[px * channels + ch];
                    terms[at] = if (t.centered) (v - mean) * (v - mean) else v;
                    at += 1;
                }
            }
            out[grp * chunks + tg] = mi.rows.sumIn(acc, order, terms[0..at]);
        };
    }
};

// --- tf_md_group_finish: stats' 32 means from part, or its 32 rstd ------------------------------------------------

/// A lane's chain of the chunk sums, then warp_sum's five steps.
fn finishDepth(s: c.Shape) usize {
    return (md.chunks(s.rows) + 31) / 32 + 6;
}

pub const Finish = struct {
    rig: rig_.Rig,
    k: md.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    centered: bool = false,
    /// part [32, chunks] as fp32 bits
    part: []f32 = &.{},
    part_in: u64 = 0,
    out: u64 = 0,

    /// Rows: pixels (its chunk count follows); n: channels; k: the pass, plain (the mean) or centered (rstd).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = finishDepth,
            // the rstd's division, its add of eps, sqrtf and reciprocal (the mean's pass divides once)
            .ops = 2,
            .out = .f32,
            .tiles = .{ .{ .size = md.chunk, .step = md.chunk, .min = md.chunk }, .{ .size = 32, .step = 32, .min = 32 }, .{} },
            .served = &served,
        };
    }

    pub fn deinit(t: *Finish) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Finish) void {
        const a = t.rig.gpu.gpa;
        a.free(t.part);
        t.part = &.{};
    }
    pub fn kernel(t: *Finish) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Finish);

    pub fn prepare(t: *Finish, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const want = isCentered(shape) orelse return false;
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.centered = want;
        const a = t.rig.gpu.gpa;
        const chunks = md.chunks(shape.rows);
        t.part = try a.alloc(f32, 32 * chunks);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // one half is written and one stays zero (gate 2's); a centered pass sums squares, so s / n + eps is no NaN
        for (t.part) |*e| e.* = switch (fill) {
            // gate 4's shape is 4 pixels, one chunk a group: the sum itself, an integer 16 past 2^15, half an fp16 ulp
            .stress => @floatFromInt(32784 + 32 * rng.uintLessThan(u32, 8)),
            else => if (t.centered) 1 + 8 * rng.float(f32) else 4 * rng.floatNorm(f32),
        };
        if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| t.part[(i * 17 + 3) % t.part.len] = v;
        }
        t.part_in = try mi.input(&t.rig, t.part);
        // stats: the 32 means, then the 32 rstd; a pass writes its own half, the other stays zero
        t.out = try t.rig.output(64 * 4);
        return true;
    }

    pub fn run(t: *Finish) !void {
        // both halves are the launch's output: the one its pass does not write is zeroed before each run
        try t.rig.zero(t.out, 64 * 4);
        try md.groupFinish(&t.k, t.rig.s, t.part_in, t.out, t.shape.rows, t.shape.n, t.centered, mi.eps);
        try t.rig.s.synchronize();
    }

    /// The 32 means, or the 32 rstd: the pass's half of the statistics, the other zero.
    pub fn count(_: *Finish) usize {
        return 64;
    }

    /// Group `grp`'s chunk sums: their float64 sum and their magnitude, in the units of the output they make.
    fn group(t: *const Finish, grp: usize) struct { sum: f64, mag: f64 } {
        const chunks = md.chunks(t.shape.rows);
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..chunks) |i| {
            const v: f64 = t.part[grp * chunks + i];
            sum += v;
            mag += @abs(v);
        }
        return .{ .sum = sum, .mag = mag };
    }

    pub fn reference(t: *Finish, want: []f64, mag: []f64) !void {
        const n = @as(f64, @floatFromInt(t.shape.rows)) * @as(f64, @floatFromInt(t.shape.n / 32));
        @memset(want, 0);
        @memset(mag, 0);
        for (0..32) |grp| {
            const s = t.group(grp);
            if (t.centered) {
                want[32 + grp] = 1 / @sqrt(s.sum / n + @as(f64, mi.eps));
                // the sum enters through 1 / sqrt(s / n): half its relative error reaches the rstd's own magnitude
                mag[32 + grp] = if (s.sum == 0) @abs(want[32 + grp]) else 0.5 * @abs(want[32 + grp]) * s.mag / @abs(s.sum);
            } else {
                want[grp] = s.sum / n;
                mag[grp] = s.mag / n;
            }
        }
    }

    pub fn simulate(t: *Finish, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const chunks = md.chunks(t.shape.rows);
        const n = @as(f64, @floatFromInt(t.shape.rows)) * @as(f64, @floatFromInt(t.shape.n / 32));
        const terms = try a.alloc(f64, chunks);
        defer a.free(terms);
        @memset(out, 0);
        const n32: f32 = @floatCast(n);
        for (0..32) |grp| {
            for (terms, t.part[grp * chunks ..][0..chunks]) |*e, v| e.* = v;
            const s: f32 = @floatCast(mi.rows.sumIn(acc, order, terms));
            if (t.centered) out[32 + grp] = @as(f32, 1.0) / @sqrt(s / n32 + mi.eps) else out[grp] = s / n32;
        }
    }
};

/// The decoder's kernels and their planted controls, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: md.Kernels,
    parts: [2]Part = undefined,
    finishes: [2]Finish = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try md.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        var with = k;
        with.group_part = try ctl.function("tf_ctl_md_group_part_centered");
        set.parts[0] = .{ .rig = rig, .k = k, .name = "cuda moss image decoder tf_md_group_part" };
        set.parts[1] = .{ .rig = rig, .k = with, .name = "control: moss image decoder tf_md_group_part sums squares uncentred", .control = .shapes };
        with = k;
        with.group_finish = try ctl.function("tf_ctl_md_group_finish_pixels");
        set.finishes[0] = .{ .rig = rig, .k = k, .name = "cuda moss image decoder tf_md_group_finish" };
        set.finishes[1] = .{ .rig = rig, .k = with, .name = "control: moss image decoder tf_md_group_finish drops the group's channels", .control = .shapes };
        for (&set.parts) |*x| try list.append(a, x.kernel());
        for (&set.finishes) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.parts) |*x| x.deinit();
        for (&set.finishes) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
