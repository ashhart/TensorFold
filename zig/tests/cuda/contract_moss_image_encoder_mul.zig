//! moss_image_encoder.cu's tf_me_silu_mul: the MLP's bf16(silu(g)) u, checked at the publisher's bf16 points.
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

/// The encoder's feed-forward width: every MLP's gate, up and product row.
const ffn = 12288;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// The product's launches: every served token count at the encoder's feed-forward width.
const served = servedShapes();

fn servedShapes() [tokens.len]c.Shape {
    var out: [tokens.len]c.Shape = undefined;
    for (&out, tokens) |*o, t| o.* = .{ .rows = t, .n = ffn, .k = 1 };
    return out;
}

pub const Mul = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    /// the MLP's gate and up rows [rows, width], as bf16 bits
    gate: []u16 = &.{},
    up: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: tokens; n: the feed-forward width (k unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            // no sums: the silu's fp32 roundings, then the bf16 gate the product reads
            .accum = .none,
            .depth = mi.noSums,
            // expf, its add and the silu's division, each rounded once
            .ops = mi.exp_ulps + 0.5 + mi.div_ulps,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = &served,
        };
    }

    pub fn deinit(t: *Mul) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Mul) void {
        const a = t.rig.gpu.gpa;
        a.free(t.gate);
        a.free(t.up);
        t.gate = &.{};
        t.up = &.{};
    }
    pub fn kernel(t: *Mul) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Mul);

    pub fn prepare(t: *Mul, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * shape.n;
        t.gate = try a.alloc(u16, n);
        t.up = try a.alloc(u16, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the gate's logits wide enough for both silu's halves, the up projection about N(0, 2)
        mi.randomBf16(rng, t.gate, -8, 8);
        mi.randomBf16(rng, t.up, -4, 4);
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
                t.gate[(i * 37 + 5) % n] = v;
                t.up[(i * 53 + 7) % n] = v;
            }
        }
        t.ins[0] = try mi.input(&t.rig, t.gate);
        t.ins[1] = try mi.input(&t.rig, t.up);
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Mul) !void {
        try me.siluMul(&t.k, t.rig.s, t.ins[0], t.ins[1], t.out, t.shape.rows, t.shape.n);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Mul) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Mul, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const w = t.shape.n;
        for (0..t.shape.rows) |row| for (0..w) |col| {
            const i = row * w + col;
            const gv: f64 = c.fromBf16(t.gate[i]);
            const uv: f64 = c.fromBf16(t.up[i]);
            // the silu's fp32 roundings, its bf16 gate, then the one rounding of bf16 x bf16 to bf16
            const sr = mi.siluRange(Range.of(gv)).fp32(mi.exp_ulps + 0.5 + mi.div_ulps).bf16();
            t.ends.put(want, mag, i, bf(bf(mi.silu(gv)) * uv), sr.mul(Range.of(uv)).bf16());
        };
    }
    pub fn bounds(t: *Mul, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Mul, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Mul = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.silu_mul = try ctl.function("tf_ctl_me_silu_mul_up");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_silu_mul", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_silu_mul drops its up projection", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
