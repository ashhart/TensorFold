//! sam3_attention_rows (rows_attention.metal): a simdgroup a (row, head), keys in order, optional additive bias.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const tbl = @import("metal_sam3_rotary_table.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const vt = g.hooks(Probe);

const RowsAttention = extern struct { rows: u32, keys: u32, heads: u32, head_width: u32, has_bias: u32 };
const bias_edit = [2][]const u8{ "if (p.has_bias != 0)", "if (false)" };

pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    return tbl.fileText(a, "// SAM3 attention one simdgroup per (row, head)", null, if (control) bias_edit else null);
}

pub fn limitText(a: std.mem.Allocator, _: []const u8, control: bool) ![]u8 {
    return text(a, control);
}

fn depth(s: c.Shape) usize {
    return s.n + 2 * s.k;
}

/// rows, n = heads x d, k = keys: the box-biased cross attention, 200 queries and presence, 8 heads of 32.
const biased_served = [_]c.Shape{ .{ .rows = 201, .n = 256, .k = 5184 }, .{ .rows = 7, .n = 256, .k = 13 } };
/// Route 7 without a bias: a head width the core kernels lack (no SAM checkpoint has one), one head.
const plain_served = [_]c.Shape{ .{ .rows = 7, .n = 48, .k = 13 }, .{ .rows = 3, .n = 256, .k = 9 }, .{ .rows = 3, .n = 48, .k = 512 } };

fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    const at = if (j < half) j else j - half;
    if (j >= half * 2 or at >= 256) return 0;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    return if (j < half) big + @as(f32, if (at < 8 and at % 2 == 0) 1 else 0) else -big;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    biased: bool,
    control: bool,
    shape: c.Shape = undefined,
    stress: bool = false,
    q: []f32 = &.{},
    k: []f32 = &.{},
    v: []f32 = &.{},
    bias: []f32 = &.{},
    half: []f64 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn label(biased: bool, control: bool) []const u8 {
        if (control) return "control: sam3 rows attention ignores its bias";
        return if (biased) "metal sam3 rows attention box bias" else "metal sam3 rows attention no bias";
    }

    fn spec(p: *const Probe) c.Spec {
        return .{
            .name = label(p.biased, p.control),
            .accum = .f32,
            .depth = depth,
            .ops = 5,
            .out = .f32,
            .served = if (p.biased) &biased_served else &plain_served,
            .tiles = .{ .{}, if (p.biased) .{} else .{ .size = 32 }, .{ .size = 32 } },
        };
    }

    fn heads(p: *const Probe) usize {
        return if (p.biased) 8 else 1;
    }

    fn reset(p: *Probe) void {
        for ([_][]f32{ p.q, p.k, p.v, p.bias }) |x| p.a.free(x);
        p.a.free(p.half);
        p.q = &.{};
        p.k = &.{};
        p.v = &.{};
        p.bias = &.{};
        p.half = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.n == 0 or s.n % p.heads() != 0 or s.n / p.heads() > 256) return false;
        p.reset();
        p.shape = s;
        p.stress = fill == .stress;
        const h = p.heads();
        const d = s.n / h;
        p.q = try p.a.alloc(f32, s.rows * s.n);
        p.k = try p.a.alloc(f32, s.k * s.n);
        p.v = try p.a.alloc(f32, s.k * s.n);
        p.bias = try p.a.alloc(f32, if (p.biased) h * s.rows * s.k else 0);
        p.half = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.q) |*x| x.* = if (p.stress) 0 else rng.float(f32) * 2 - 1;
        for (p.k) |*x| x.* = if (p.stress) 0 else rng.float(f32) * 2 - 1;
        for (p.v, 0..) |*x, i| x.* = if (p.stress) stressValue(i / s.n, s.k) else rng.float(f32) * 2 - 1;
        for (p.bias) |*x| x.* = if (p.stress) 0 else rng.float(f32) * 4 - 2;
        if (fill == .nonfinite) {
            p.q[0] = std.math.nan(f32);
            p.q[s.n] = std.math.inf(f32);
            p.q[2 * s.n] = -std.math.inf(f32);
            p.q[3 * s.n] = std.math.inf(f32);
            if (d > 1) p.q[3 * s.n + 1] = -std.math.inf(f32);
            p.v[0] = std.math.inf(f32);
            if (d > 1) p.v[1] = -std.math.inf(f32);
            if (s.k > 1 and d > 2) p.v[s.n + 2] = std.math.nan(f32);
            if (p.biased and s.rows > 5 and s.k > 3) p.bias[(s.rows + 5) * s.k + 3] = -std.math.inf(f32);
        }
        for ([_][]const f32{ p.q, p.k, p.v }, p.ins[0..3]) |values, *input| input.* = try p.rig.input(std.mem.sliceAsBytes(values));
        if (p.biased) p.ins[3] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.ins[0..3], 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        const bias = if (p.biased) p.ins[3] else p.ins[0];
        pass.e.setBuffer(bias.buf, bias.off, 3);
        pass.e.setBuffer(p.out.buf, p.out.off, 4);
        pass.e.setValue(RowsAttention{ .rows = @intCast(s.rows), .keys = @intCast(s.k), .heads = @intCast(p.heads()), .head_width = @intCast(s.n / p.heads()), .has_bias = @intFromBool(p.biased) }, 5);
        pass.e.dispatchThreads(mtl.Size.of(s.rows * 32, p.heads(), 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    /// The kernel's arithmetic as attention_range's form; the bias enters as one more score column (q 1, k b / scale).
    fn form(d: usize, keys: usize, lifts: usize) ar.Form {
        const n: f64 = @floatFromInt(keys);
        return .{
            // a lane's d / 32 chain, simd_sum (order undocumented: up to 31 deep), fast rsqrt (2 ulps), its product
            .score = @as(f64, @floatFromInt((d + 31) / 32)) + 31 + 4 + 1,
            .score_round = 1,
            .fast_exp = false,
            .chunk_exp = false,
            .p_rel = 0,
            // acc * o + e * v and total * o + e each key: a product, a product and an add
            .num = 2 * n + 2,
            .den = 2 * n + 2,
            .div = 5,
            // fast::exp's 40 + 2|x| ulps for a weight's own exp and each rescale its key can pass through
            .exp_ulps = .{ 40 * @as(f64, @floatFromInt(1 + lifts)), 2 },
        };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const h_n = p.heads();
        const d = s.n / h_n;
        const e = d + 1;
        const scale = 1 / @sqrt(@as(f64, @floatFromInt(d)));
        const buf = try p.a.alloc(f64, 2 * s.k * e + 2 * s.k + 3 * e);
        defer p.a.free(buf);
        const keys = buf[0 .. s.k * e];
        const values = buf[s.k * e ..][0 .. s.k * e];
        const w = buf[2 * s.k * e ..][0..s.k];
        const tau = buf[2 * s.k * e + s.k ..][0..s.k];
        const query = buf[2 * s.k * e + 2 * s.k ..][0..e];
        const point = buf[2 * s.k * e + 2 * s.k + e ..][0..e];
        const half = buf[2 * s.k * e + 2 * s.k + 2 * e ..][0..e];
        for (0..h_n) |h| {
            for (0..s.k) |j| {
                for (0..d) |ch| {
                    keys[j * e + ch] = p.k[(j * h_n + h) * d + ch];
                    values[j * e + ch] = p.v[(j * h_n + h) * d + ch];
                }
                keys[j * e + d] = 0;
                values[j * e + d] = 0;
            }
            for (0..s.rows) |r| {
                for (0..d) |ch| query[ch] = p.q[(r * h_n + h) * d + ch];
                query[d] = 1;
                if (p.biased) for (0..s.k) |j| {
                    keys[j * e + d] = p.bias[(h * s.rows + r) * s.k + j] / scale;
                };
                const f = form(d, s.k, p.increases(query, keys, e, scale, d));
                ar.head(query, keys, values, s.k, e, scale, f, w, tau, point, half);
                const at = (r * h_n + h) * d;
                @memcpy(want[at..][0..d], point[0..d]);
                @memcpy(p.half[at..][0..d], half[0..d]);
            }
        }
        @memcpy(mag, p.half);
    }

    /// Keys whose score can lift the kernel's running maximum: above the float64 maximum so far less twice the spread.
    fn increases(p: *const Probe, q: []const f64, keys: []const f64, e: usize, scale: f64, d: usize) usize {
        const spread = form(d, p.shape.k, 0).score;
        var top: f64 = -std.math.inf(f64);
        var widest: f64 = 0;
        for (0..p.shape.k) |j| {
            var s: f64 = 0;
            var a: f64 = 0;
            for (0..e) |ch| {
                s += q[ch] * keys[j * e + ch];
                a += @abs(q[ch] * keys[j * e + ch]);
            }
            const t = spread * 0x1p-24 * a * scale + 0x1p-24 * @abs(s * scale);
            if (std.math.isFinite(t)) widest = @max(widest, t);
        }
        var n: usize = 0;
        for (0..p.shape.k) |j| {
            var s: f64 = 0;
            for (0..e) |ch| s += q[ch] * keys[j * e + ch];
            const w = s * scale;
            if (!std.math.isFinite(w)) continue;
            if (w > top - 2 * widest) n += 1;
            top = @max(top, w);
        }
        return n;
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (want, p.half, allow) |w, half, *v| v.* = if (p.stress) c.boundExact(p.spec(), w) else half;
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.shape.k);
        defer p.a.free(terms);
        for (terms, 0..) |*x, j| x.* = stressValue(j, terms.len);
        @memset(out, contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(terms.len)));
    }

    fn deinit(p: *Probe) void {
        p.reset();
        p.rig.deinit();
        p.pipe.deinit();
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [3]?*Probe = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (0..3) |i| {
            const biased = i != 1;
            const control = i == 2;
            if (only) |needle| if (std.mem.indexOf(u8, Probe.label(biased, control), needle) == null) continue;
            const source = try text(s.a, control);
            defer s.a.free(source);
            const pipe = try rig_.pipeline(device, source, "sam3_attention_rows");
            errdefer pipe.deinit();
            try kernels.ensureUnusedCapacity(s.a, 1);
            const p = try s.a.create(Probe);
            p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .biased = biased, .control = control };
            s.probes[i] = p;
            kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes) |maybe| if (maybe) |p| {
            p.deinit();
            s.a.destroy(p);
        };
    }
};
