//! moss_image_encoder.cu's tf_me_head: 256-wide heads normed and turned, checked at the publisher's bf16 points.
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

/// The head width: the head norm's reduction and the channels every output head lands on.
const D = 256;

/// The rotated channels of each head and the table's stride: a quarter of the head, halves 32 channels apart.
const rotary = 32;

/// The encoder's head counts: q's 16 heads carry the gate half, k's and v's 4 are 256 wide.
const heads = 16;
const kv_heads = 4;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// One launch's geometry, told apart by the shape's k: q's heads, k's normed ones, then v's copied ones.
const Form = struct { heads: usize, stride: usize, normed: bool };

fn form(k: usize) ?Form {
    return switch (k) {
        2 => .{ .heads = heads, .stride = 2 * D, .normed = true },
        1 => .{ .heads = kv_heads, .stride = D, .normed = true },
        0 => .{ .heads = kv_heads, .stride = D, .normed = false },
        else => null,
    };
}

/// The three launches each token count makes, q's first: q normed, k normed, v copied.
const served = servedShapes();

fn servedShapes() [3 * tokens.len]c.Shape {
    var out: [3 * tokens.len]c.Shape = undefined;
    for (tokens, 0..) |t, i| {
        out[3 * i] = .{ .rows = t, .n = heads, .k = 2 };
        out[3 * i + 1] = .{ .rows = t, .n = kv_heads, .k = 1 };
        out[3 * i + 2] = .{ .rows = t, .n = kv_heads, .k = 0 };
    }
    return out;
}

const why = "a bf16 output follows the head's sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

/// Eight squares a lane in turn, then the warp's five shuffles; the copied heads sum nothing.
fn depth(s: c.Shape) usize {
    const f = form(s.k) orelse return 0;
    return if (f.normed) 13 else 0;
}

pub const Head = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// x [rows, heads x stride] and the head norm's weight [256], as bf16 bits
    x: []u16 = &.{},
    w: []u16 = &.{},
    /// the rotary tables' cos and sin [rows, 32], fp32 as the launch reads them
    cs: []f32 = &.{},
    sn: []f32 = &.{},
    ins: [4]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: tokens; n: heads; k: the launch's form (2 q, 1 k, 0 v).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 2 * (mi.div_ulps + mi.sqrt_ulps) + 1,
            .mids = 2,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{}, .{} },
            .served = &served,
            .sums = .{ .bf16 = 0, .why = why },
        };
    }

    pub fn deinit(t: *Head) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Head) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.w);
        a.free(t.cs);
        a.free(t.sn);
        t.x = &.{};
        t.w = &.{};
        t.cs = &.{};
        t.sn = &.{};
    }
    pub fn kernel(t: *Head) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Head);

    fn inStride(t: *const Head) usize {
        return t.shape.n * form(t.shape.k).?.stride;
    }

    pub fn prepare(t: *Head, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const f = form(shape.k) orelse return false;
        if (shape.rows == 0 or shape.n == 0 or shape.n != f.heads) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        const rows = shape.rows;
        const stride = t.inStride();
        t.x = try a.alloc(u16, rows * stride);
        t.w = try a.alloc(u16, D);
        t.cs = try a.alloc(f32, rows * rotary);
        t.sn = try a.alloc(f32, rows * rotary);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the rows about N(0, 2), the head norm's weight a norm's own, the tables a full turn's angles
        if (fill == .stress) mi.integers(rng, t.x, 128) else mi.randomBf16(rng, t.x, -4, 4);
        mi.randomBf16(rng, t.w, 0.25, 2);
        for (t.cs, t.sn) |*cv, *sv| {
            const angle = rng.float(f32) * 2 * std.math.pi;
            cv.* = @cos(angle);
            sv.* = @sin(angle);
        }
        if (fill == .nonfinite) {
            const at = [_][3]usize{ .{ 0, 0, 0 }, .{ 1, 1, 70 }, .{ 0, 2, 255 }, .{ 2, 3, 9 }, .{ 1, 0, 127 } };
            for (at, [_]u16{ mi.nan16, mi.inf16, mi.ninf16, mi.nan16, mi.ninf16 }) |p, v| {
                if (p[1] < rows and p[0] < shape.n) t.x[p[1] * stride + p[0] * f.stride + p[2]] = v;
            }
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.w);
        t.ins[2] = try mi.input(&t.rig, t.cs);
        t.ins[3] = try mi.input(&t.rig, t.sn);
        t.out = try t.rig.output(shape.n * rows * D * 2);
        return true;
    }

    pub fn run(t: *Head) !void {
        const f = form(t.shape.k).?;
        const shape = t.shape;
        try me.head(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.out, shape.rows, shape.n, t.inStride(), f.stride, mi.eps, f.normed);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Head) usize {
        return t.shape.n * t.shape.rows * D;
    }

    /// Head `h` of row `r` from its sum of squares `ss` (range `ssr`): normed by (1 + w), the first 64 turned.
    fn head(t: *Head, ends: *Ends, r: usize, h: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const f = form(t.shape.k).?;
        const from = r * t.inStride() + h * f.stride;
        const to = (h * t.shape.rows + r) * D;
        if (!f.normed) {
            for (0..D) |ch| {
                const x: f64 = c.fromBf16(t.x[from + ch]);
                ends.put(want, mag, to + ch, x, Range.of(x));
            }
            return;
        }
        const n = mi.invNorm(ss, ssr, D);
        var tv: [D]f64 = undefined;
        var tr: [D]Range = undefined;
        for (0..D) |ch| {
            const x: f64 = c.fromBf16(t.x[from + ch]);
            const w: f64 = c.fromBf16(t.w[ch]);
            const gain = Range.of(1.0).add(Range.of(w)).fp32(0.5);
            tv[ch] = bf(x * n.inv * (1 + w));
            tr[ch] = Range.of(x).mul(n.r).fp32(0.5).mul(gain).fp32(0.5).bf16();
        }
        // the first 64 channels turn by cos and sin, a channel's partner 32 away and the sign of its half
        for (0..D) |ch| {
            if (ch >= 2 * rotary) {
                ends.put(want, mag, to + ch, tv[ch], tr[ch]);
                continue;
            }
            const at = r * rotary + ch % rotary;
            const cs: f64 = t.cs[at];
            const sn: f64 = t.sn[at];
            const sign: f64 = if (ch < rotary) -1 else 1;
            const partner = ch ^ rotary;
            const o = bf(tv[ch] * cs) + bf(sign * tv[partner] * sn);
            const yr = tr[ch].mul(Range.of(cs)).fp32(0.5).bf16().add(tr[partner].mul(Range.of(sign * sn)).fp32(0.5).bf16()).fp32(0.5).bf16();
            ends.put(want, mag, to + ch, bf(o), yr);
        }
    }

    pub fn reference(t: *Head, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const f = form(t.shape.k).?;
        const stride = t.inStride();
        for (0..t.shape.rows) |r| for (0..t.shape.n) |h| {
            const s = mi.squares(t.x[r * stride + h * f.stride ..][0..D], depth(t.shape), t.fill == .stress);
            t.head(&t.ends, r, h, s.ss, s.r, want, mag);
        };
    }
    pub fn bounds(t: *Head, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Head, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Head, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const f = form(t.shape.k).?;
        const stride = t.inStride();
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var terms: [D]f64 = undefined;
        for (0..t.shape.rows) |r| for (0..t.shape.n) |h| {
            for (&terms, t.x[r * stride + h * f.stride ..][0..D]) |*v, x| {
                const xv: f64 = c.fromBf16(x);
                v.* = xv * xv;
            }
            const ss = mi.rows.sumIn(acc, order, &terms);
            t.head(&scratch, r, h, ss, Range.of(ss), out, mag);
        };
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Head = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.head = try ctl.function("tf_ctl_me_head_weight");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_head q k v", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_head drops (1 + w)", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
