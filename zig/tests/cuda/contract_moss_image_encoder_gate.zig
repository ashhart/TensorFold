//! moss_image_encoder.cu's tf_me_gate: the attention rows gated by a bf16 sigmoid, at the publisher's bf16 points.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const me = moss.encoder;
const c = mi.c;
const g = mi.g;
const Range = mi.Range;
const Ends = mi.Ends;
const bf = mi.bf;

/// The encoder's hidden width: a gate's row, and a quarter of the q projection's row it reads.
const hidden = 4096;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// The gate's launches: every served token count at the encoder's hidden width.
const served = servedShapes();

fn servedShapes() [tokens.len]c.Shape {
    var out: [tokens.len]c.Shape = undefined;
    for (&out, tokens) |*o, t| o.* = .{ .rows = t, .n = hidden, .k = 1 };
    return out;
}

pub const Gate = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    /// the attention rows [rows, width] and the q projection's rows [rows, 2 x width], as bf16 bits
    a: []u16 = &.{},
    qg: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: tokens; n: the hidden width (k unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            // expf, its add, the sigmoid's division and the product, each rounded once
            .ops = mi.exp_ulps + 0.5 + mi.div_ulps + 0.5,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256, .step = 256, .min = 256 }, .{} },
            .served = &served,
        };
    }

    pub fn deinit(t: *Gate) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Gate) void {
        const a = t.rig.gpu.gpa;
        a.free(t.a);
        a.free(t.qg);
        t.a = &.{};
        t.qg = &.{};
    }
    pub fn kernel(t: *Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Gate);

    pub fn prepare(t: *Gate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 256 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const w = shape.n;
        const n = shape.rows * w;
        t.a = try a.alloc(u16, n);
        t.qg = try a.alloc(u16, 2 * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the attention rows about N(0, 2), the gates a projection's logits
        mi.randomBf16(rng, t.a, -4, 4);
        mi.randomBf16(rng, t.qg, -8, 8);
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| t.a[(i * 37 + 5) % n] = v;
            // a NaN q logit at column 5: the sigmoid, and so its row's output, is NaN
            if (shape.rows > 4) {
                t.qg[2 * w + 256] = mi.nan16;
                t.qg[4 * w + 261] = mi.nan16;
            }
        }
        t.ins[0] = try mi.input(&t.rig, t.a);
        t.ins[1] = try mi.input(&t.rig, t.qg);
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Gate) !void {
        try me.gate(&t.k, t.rig.s, t.ins[0], t.ins[1], t.out, t.shape.rows, t.shape.n);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Gate) usize {
        return t.shape.rows * t.shape.n;
    }

    /// The gate the q projection's second half holds for output (row, col): four columns of a 512-wide head.
    fn at(t: *const Gate, row: usize, col: usize) usize {
        return row * 2 * t.shape.n + (col / 256) * 512 + 256 + col % 256;
    }

    pub fn reference(t: *Gate, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const w = t.shape.n;
        for (0..t.shape.rows) |row| for (0..w) |col| {
            const x: f64 = c.fromBf16(t.qg[t.at(row, col)]);
            const a: f64 = c.fromBf16(t.a[row * w + col]);
            const sig = bf(1 / (1 + @exp(-x)));
            const sr = Range.of(@exp(-x)).fp32(mi.exp_ulps).add(Range.of(1)).fp32(0.5).recip().fp32(mi.div_ulps).bf16();
            const yr = Range.of(a).mul(sr).fp32(0.5).bf16();
            t.ends.put(want, mag, row * w + col, bf(a * sig), yr);
        };
    }
    pub fn bounds(t: *Gate, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Gate, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Gate = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.gate = try ctl.function("tf_ctl_me_gate_sigmoid");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_gate", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_gate keeps its sigmoid in fp32", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
