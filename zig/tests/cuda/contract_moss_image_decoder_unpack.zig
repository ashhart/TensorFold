//! moss_image_decoder.cu's tf_md_unpack under the contract: the latents' batch norm undone, each 2 x 2 patch unpacked.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const md = moss.decoder;
const c = mi.c;
const g = mi.g;
const Range = mi.Range;
const Ends = mi.Ends;

/// The latents' channels: 32 lanes of four, the 2 x 2 patch's four positions.
const channels = 128;

/// The engine's latent grids: a 1024 x 1024 picture's 64 x 64, the 1200 x 880 case's 55 rows of 75 columns.
const served = [_]c.Shape{
    .{ .rows = 64, .n = 64, .k = channels },
    .{ .rows = 55, .n = 75, .k = channels },
};

pub const Unpack = struct {
    rig: rig_.Rig,
    k: md.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    /// the latents [rows * columns, 128] and the batch norm's scale and mean [128], as fp32 bits
    x: []f32 = &.{},
    s: []f32 = &.{},
    m: []f32 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows and columns: the latent grid; k: its channels (128, never tiled).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            // the product and the sum, each rounded once: nvcc's glue flags contract neither
            .ops = 2,
            .out = .f32,
            .tiles = .{ .{ .size = 1 }, .{ .size = 2 }, .{} },
            .served = &served,
        };
    }

    pub fn deinit(t: *Unpack) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Unpack) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.s);
        a.free(t.m);
        t.x = &.{};
        t.s = &.{};
        t.m = &.{};
    }
    pub fn kernel(t: *Unpack) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Unpack);

    pub fn prepare(t: *Unpack, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != channels) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * shape.n;
        t.x = try a.alloc(f32, n * channels);
        t.s = try a.alloc(f32, channels);
        t.m = try a.alloc(f32, channels);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the latents four times a unit normal (the transformer's widths), the batch norm's as the checkpoint's
        for (t.x) |*v| v.* = 4 * rng.floatNorm(f32);
        for (t.s) |*v| v.* = 0.25 + rng.float(f32) * 1.75;
        for (t.m) |*v| v.* = 2 * rng.float(f32) - 1;
        if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| {
                t.x[(i * 37 + 5) % t.x.len] = v;
                t.s[(i * 11 + 3) % channels] = v;
                t.m[(i * 7 + 1) % channels] = v;
            }
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.s);
        t.ins[2] = try mi.input(&t.rig, t.m);
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Unpack) !void {
        try md.unpack(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.out, t.shape.rows, t.shape.n);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Unpack) usize {
        return t.shape.rows * t.shape.n * 4 * 32;
    }

    /// Latent pixel (row, column) channel k: the output row its patch position lands on, and its lane.
    fn at(t: *const Unpack, row: usize, col: usize, k: usize) usize {
        const py = 2 * row + (k / 2) % 2;
        const px = 2 * col + k % 2;
        return (py * 2 * t.shape.n + px) * 32 + k / 4;
    }

    pub fn reference(t: *Unpack, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..t.shape.rows) |row| for (0..t.shape.n) |col| for (0..channels) |k| {
            const x: f64 = t.x[(row * t.shape.n + col) * channels + k];
            const s: f64 = t.s[k];
            const m: f64 = t.m[k];
            const r = Range.of(x).mul(Range.of(s)).fp32(0.5).add(Range.of(m)).fp32(0.5);
            t.ends.put(want, mag, t.at(row, col, k), x * s + m, r);
        };
    }
    /// The product's and the sum's roundings' ends about the float64 value: the allowance every output must meet.
    pub fn bounds(t: *Unpack, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
};

/// The decoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: md.Kernels,
    t: [2]Unpack = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try md.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.unpack = try ctl.function("tf_ctl_md_unpack_mean");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image decoder tf_md_unpack", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image decoder tf_md_unpack drops its mean", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
