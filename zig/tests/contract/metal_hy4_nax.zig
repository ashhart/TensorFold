//! Hy4-flash's tensor-unit MLA products (mla_nax.metal, Nax in mla_metal.zig); attention in metal_hy4_nax_attend.zig.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const at = @import("metal_hy4_nax_attend.zig");
const fl = h.fl;
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const threads = contract.reference_threads;
const mw = h.hy.mla_weights;

/// mla_nax.metal as Nax.load builds it (frags.source, then MLX's options), a control's edit applied to the text first.
pub fn pipeline(a: std.mem.Allocator, device: mtl.Device, e: ?fl.Edit, function: []const u8) !mtl.Pipeline {
    const text = try fl.edited(a, sources.hy4_flash_mla_nax, e);
    defer a.free(text);
    const body = try tf.frags.source(device, a, text);
    defer a.free(body);
    return rig_.pipeline(device, body, function);
}

/// mla_metal.zig's Args for the two products.
const Args = extern struct { rows: u32, heads: u32 };

pub const Which = enum { absorb, expand };

/// rows: the pass's rows; n: its heads (sparse 32, sliding 64); k: the sum's length. Decode, DFly2's window, a chunk.
fn served(comptime k: usize) [5]c.Shape {
    return .{
        .{ .rows = h.window_rows, .n = 32, .k = k },
        .{ .rows = h.decode_rows, .n = 64, .k = k },
        .{ .rows = h.window_rows, .n = 64, .k = k },
        .{ .rows = h.prompt_rows, .n = 32, .k = k },
        .{ .rows = h.prompt_rows, .n = 64, .k = k },
    };
}
const absorb_served = served(mw.nope);
const expand_served = served(mw.latent);

/// A product's fp32 chain: one mma step (attention_range's 36 roundings) for each 16 of its sum's terms.
fn absorbDepth(_: c.Shape) usize {
    return mw.nope / 16 * ar.mma_step;
}
fn expandDepth(_: c.Shape) usize {
    return mw.latent / 16 * ar.mma_step;
}

/// absorb: bf16 of the sum; expand: bf16(bf16(sum) * bf16(sigmoid)), the sigmoid by precise exp, an add, fast division.
pub fn spec(w: Which, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = if (w == .absorb) absorbDepth else expandDepth,
        .ops = if (w == .expand) 14 else 0,
        .mids = if (w == .expand) 2 else 0,
        .out = .bf16,
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = if (w == .absorb) &absorb_served else &expand_served,
    };
}

/// The sigmoid's ends through precise exp (4 ulps), the add, fast division (2.5 ulps), then its bf16 point.
fn sigmoid(x: f64) fl.Range {
    const e = @exp(-x);
    const d = (fl.Range{ .lo = e * (1 - 4 * 0x1p-23), .hi = e * (1 + 4 * 0x1p-23) }).plus(fl.Range.point(1)).fp32();
    return (fl.Range{ .lo = 1 / d.hi * (1 - 2.5 * 0x1p-23), .hi = 1 / d.lo * (1 + 2.5 * 0x1p-23) }).bf16();
}

/// What absorb leaves in a head's rope columns (512 to 575, the rotation's): a finite value no launch may change.
fn placeholder(i: usize) u16 {
    return c.toBf16(@floatFromInt(1 + i % 61));
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    which: Which,
    control: bool,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    heads: usize = 0,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    w: []u16 = &.{},
    gl: []u16 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    refs: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn base(w: Which) []const u8 {
        return switch (w) {
            .absorb => "metal hy4 mla_nax tf_hy4_absorb_nax (192 query columns into the 512-wide latent, 16-row tiles)",
            .expand => "metal hy4 mla_nax tf_hy4_expand_gated_nax (the 512-wide latent to 256 values, then the gate)",
        };
    }
    fn edit(w: Which) fl.Edit {
        return switch (w) {
            .absorb => .{ .kernel = "tf_hy4_absorb_nax", .from = "k < 192;", .to = "k < 176;" },
            .expand => .{ .kernel = "tf_hy4_expand_gated_nax", .from = "float(v) * float(hy4n_round(gate))", .to = "float(v) * gate" },
        };
    }
    fn fault(w: Which) []const u8 {
        return if (w == .absorb) "its last 16-deep step dropped" else "the gate's bf16 rounding dropped";
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, control: bool) !Proj {
        const fun = if (w == .absorb) "tf_hy4_absorb_nax" else "tf_hy4_expand_gated_nax";
        const pipe = try pipeline(a, device, if (control) edit(w) else null, fun);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .which = w, .control = control };
    }
    fn free(t: *Proj) void {
        inline for (.{ "x", "w", "gl" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        t.a.free(t.lo);
        t.a.free(t.hi);
        t.lo = &.{};
        t.hi = &.{};
    }
    pub fn deinit(t: *Proj) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Proj) g.Kernel {
        t.name = fl.named(base(t.which), if (t.control) fault(t.which) else null, &t.label);
        return .{ .spec = spec(t.which, t.name), .ctx = t, .vt = &vt, .control = if (t.control) .shapes else null };
    }
    const vt = g.hooks(Proj);

    /// The sum's length, the outputs a head, a head's input words and a head's output words.
    fn sumLen(t: *const Proj) usize {
        return if (t.which == .absorb) mw.nope else mw.latent;
    }
    fn outs(t: *const Proj) usize {
        return if (t.which == .absorb) mw.latent else mw.value_width;
    }
    fn inWords(t: *const Proj) usize {
        return if (t.which == .absorb) mw.query_stride else mw.latent;
    }
    fn outWords(t: *const Proj) usize {
        return if (t.which == .absorb) mw.absorbed_stride else mw.value_width;
    }
    /// Weight (head, term, output): absorb's W_UK [192][512], expand's W_UV [256][512] read transposed.
    fn wAt(t: *const Proj, hd: usize, k: usize, n: usize) usize {
        return if (t.which == .absorb) (hd * mw.nope + k) * mw.latent + n else (hd * mw.value_width + n) * mw.latent + k;
    }

    pub fn prepare(t: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > h.prompt_rows or (shape.n != 32 and shape.n != 64)) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.heads = shape.n;
        t.fill = fill;
        const kd = t.sumLen();
        t.x = try t.a.alloc(u16, t.rows * t.heads * t.inWords());
        t.w = try t.a.alloc(u16, t.heads * kd * t.outs());
        if (t.which == .expand) t.gl = try t.a.alloc(u16, t.rows * t.heads * mw.value_width);
        t.lo = try t.a.alloc(f64, t.count());
        t.hi = try t.a.alloc(f64, t.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (t.w) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.gl) |*v| v.* = c.toBf16(rng.float(f32) * 12 - 6);
        switch (fill) {
            .random => {},
            .nonfinite => t.plant(),
            .stress => t.stress(rng),
        }
        // absorb never reads a query's rope columns (192 to 255; the rotation takes them): NaN here
        if (t.which == .absorb) for (0..t.rows * t.heads) |rh| @memset(t.x[rh * mw.query_stride + mw.nope ..][0..mw.rope], 0x7fc0);
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        if (t.which == .expand) t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.gl));
        t.out = try t.rig.output(t.count() * 2);
        if (t.which == .absorb) {
            const o = std.mem.bytesAsSlice(u16, t.rig.bytes(t.out, t.count() * 2));
            for (0..t.count()) |i| if (i % mw.absorbed_stride >= mw.latent) {
                o[i] = placeholder(i);
            };
        }
        return true;
    }

    /// NaN and both infinities in rows' inputs, a weight and (expand) the gate logits; 6 rows, so rows 0 to 5 exist.
    fn plant(t: *Proj) void {
        const H = t.heads;
        const iw = t.inWords();
        t.x[(0 * H + 0) * iw + 5] = c.toBf16(fl.nan32);
        t.x[(1 * H + 1) * iw + 7] = c.toBf16(fl.inf32);
        t.w[t.wAt(2, 3, 9)] = c.toBf16(-fl.inf32);
        t.x[(3 * H + 4) * iw + 1] = c.toBf16(fl.inf32);
        t.x[(3 * H + 4) * iw + 2] = c.toBf16(-fl.inf32);
        if (t.which == .absorb) return;
        const vw = mw.value_width;
        t.gl[(4 * H + 3) * vw + 11] = c.toBf16(fl.nan32);
        t.gl[(5 * H + 4) * vw + 12] = c.toBf16(fl.inf32);
        t.gl[(5 * H + 4) * vw + 13] = c.toBf16(-fl.inf32);
    }

    /// Gate 4: integer terms, the second half cancelling the first but for 8 nudged by one; expand's gate exactly 1.
    fn stress(t: *Proj, rng: std.Random) void {
        const kd = t.sumLen();
        const half = kd / 2;
        const iw = t.inWords();
        for (0..t.rows * t.heads) |rh| for (0..half) |k| {
            const v: f32 = @floatFromInt(64 + rng.uintLessThan(u32, 128));
            t.x[rh * iw + k] = c.toBf16(v);
            t.x[rh * iw + half + k] = c.toBf16(v + @as(f32, @floatFromInt(@intFromBool(k < 8))));
        };
        for (0..t.heads) |hd| for (0..t.outs()) |n| for (0..half) |k| {
            const v: f32 = @floatFromInt(if (k < 8) 1 + rng.uintLessThan(u32, 7) else 64 + rng.uintLessThan(u32, 128));
            t.w[t.wAt(hd, k, n)] = c.toBf16(v);
            t.w[t.wAt(hd, half + k, n)] = c.toBf16(-v);
        };
        @memset(t.gl, c.toBf16(20));
    }

    /// One launch as Nax.run encodes it: (output columns / 128, heads, row blocks of 16) groups of 128 threads.
    pub fn run(t: *Proj) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.refs[0].buf, t.refs[0].off, 0);
        p.e.setBuffer(t.refs[1].buf, t.refs[1].off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue(Args{ .rows = @intCast(t.rows), .heads = @intCast(t.heads) }, 3);
        if (t.which == .expand) p.e.setBuffer(t.refs[2].buf, t.refs[2].off, 4);
        p.e.dispatchGroups(mtl.Size.of(t.outs() / 128, t.heads, (t.rows + 15) / 16), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *const Proj) usize {
        return t.rows * t.heads * t.outWords();
    }

    /// Each output's float64 sum, at expand's bf16 points through the gate, with its range; rows in parallel.
    pub fn reference(t: *Proj, want: []f64, mag: []f64) !void {
        const wf = try t.a.alloc(f64, t.w.len);
        defer t.a.free(wf);
        for (wf, t.w) |*f, b| f.* = c.fromBf16(b);
        const jobs_n = threads.count(threads.maximum, t.rows);
        var jobs: [threads.maximum]Job = undefined;
        for (jobs[0..jobs_n], 0..) |*j, i| j.* = .{ .t = t, .wf = wf, .want = want, .mag = mag, .first = i * t.rows / jobs_n, .end = (i + 1) * t.rows / jobs_n };
        threads.run(Job, jobs[0..jobs_n], .normal);
    }

    const Job = struct {
        t: *const Proj,
        wf: []const f64,
        want: []f64,
        mag: []f64,
        first: usize,
        end: usize,
        pub fn run(j: *Job) void {
            for (j.first..j.end) |r| for (0..j.t.heads) |hd| j.t.head(r, hd, j.wf, j.want, j.mag);
        }
    };

    fn head(t: *const Proj, r: usize, hd: usize, wf: []const f64, want: []f64, mag: []f64) void {
        const kd = t.sumLen();
        const xin = t.x[(r * t.heads + hd) * t.inWords() ..][0..kd];
        const at0 = (r * t.heads + hd) * t.outWords();
        var xf: [mw.latent]f64 = undefined;
        for (xf[0..kd], xin) |*f, b| f.* = c.fromBf16(b);
        if (t.which == .absorb) {
            var acc: [mw.latent]f64 = @splat(0);
            var abs: [mw.latent]f64 = @splat(0);
            for (xf[0..kd], 0..) |q, k| for (&acc, &abs, wf[t.wAt(hd, k, 0)..][0..mw.latent]) |*s, *m, wv| {
                s.* += q * wv;
                m.* += @abs(q * wv);
            };
            @memcpy(want[at0..][0..mw.latent], &acc);
            @memcpy(mag[at0..][0..mw.latent], &abs);
            for (mw.latent..mw.absorbed_stride) |n| {
                want[at0 + n] = c.fromBf16(placeholder(at0 + n));
                mag[at0 + n] = 0;
            }
            for (t.lo[at0..][0..mw.absorbed_stride], t.hi[at0..][0..mw.absorbed_stride]) |*l, *u| {
                l.* = -std.math.floatMax(f64);
                u.* = std.math.floatMax(f64);
            }
            return;
        }
        const gamma = c.gamma(@floatFromInt(expandDepth(.{ .rows = 1, .n = 1, .k = 1 })), c.unit(.f32));
        for (0..mw.value_width) |n| {
            const d = dotAbs(xf[0..kd], wf[t.wAt(hd, 0, n)..][0..kd]);
            const i = at0 + n;
            const x: f64 = c.fromBf16(t.gl[i]);
            want[i] = fl.bf(fl.bf(d[0]) * fl.bf(1 / (1 + @exp(-x))));
            mag[i] = if (std.math.isFinite(want[i])) @abs(want[i]) else 0;
            const v = (fl.Range{ .lo = d[0] - gamma * d[1], .hi = d[0] + gamma * d[1] }).bf16();
            const o = v.mul(sigmoid(x)).widen(fl.tiny).bf16();
            const open = !std.math.isFinite(want[i]) or !std.math.isFinite(o.lo) or !std.math.isFinite(o.hi);
            t.lo[i] = if (open) -std.math.floatMax(f64) else o.lo;
            t.hi[i] = if (open) std.math.floatMax(f64) else o.hi;
        }
    }

    /// absorb keeps the generic bound; expand's allowance is its range's farther end, gate 4's the rounding alone.
    pub fn bounds(t: *Proj, want: []const f64, _: []const f64, allow: []f64) !void {
        if (t.which == .absorb) return;
        for (allow, want, t.lo, t.hi) |*l, w, lo, hi| {
            if (t.fill == .stress) {
                l.* = c.boundExact(spec(.expand, ""), w);
            } else if (!std.math.isFinite(w)) {
                l.* = 0;
            } else l.* = @min(std.math.floatMax(f64), fl.spread(w, .{ .lo = lo, .hi = hi }));
        }
    }

    pub fn intervals(t: *Proj, low: []f64, high: []f64) !void {
        @memcpy(low, t.lo);
        @memcpy(high, t.hi);
    }

    /// Gate 4: each output if its sum ran in `acc`, then its bf16 points (absorb's placeholders as they are).
    pub fn simulate(t: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const kd = t.sumLen();
        var terms: [mw.latent]f64 = undefined;
        for (0..t.rows) |r| for (0..t.heads) |hd| for (0..t.outWords()) |n| {
            const i = (r * t.heads + hd) * t.outWords() + n;
            if (n >= t.outs()) {
                out[i] = c.fromBf16(placeholder(i));
                continue;
            }
            for (terms[0..kd], 0..) |*x, k| x.* = @as(f64, c.fromBf16(t.x[(r * t.heads + hd) * t.inWords() + k])) * c.fromBf16(t.w[t.wAt(hd, k, n)]);
            const s = fl.bf(contract.rows.sumIn(acc, order, terms[0..kd]));
            out[i] = if (t.which == .absorb) s else fl.bf(s * fl.bf(1 / (1 + @exp(-@as(f64, c.fromBf16(t.gl[i]))))));
        };
    }
};

/// sum and sum of |terms| of x . y, eight lanes at a time (both lengths multiples of 8).
pub fn dotAbs(x: []const f64, y: []const f64) [2]f64 {
    const V = @Vector(8, f64);
    var s: V = @splat(0);
    var m: V = @splat(0);
    var i: usize = 0;
    while (i < x.len) : (i += 8) {
        const p = @as(V, x[i..][0..8].*) * @as(V, y[i..][0..8].*);
        s += p;
        m += @abs(p);
    }
    return .{ @reduce(.Add, s), @reduce(.Add, m) };
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    proj: fl.List(Proj) = .{},
    attend: fl.List(at.Attend) = .{},

    /// Only on tensor units, where the loader builds these pipelines (metal_dense.zig's `device.tensorUnits()`).
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) {
            if (fl.wanted(only, "metal hy4 mla_nax")) _ = @import("metal_support.zig").tensor(false, "metal hy4 mla_nax");
            return;
        }
        for ([_]Which{ .absorb, .expand }) |w| for ([_]bool{ false, true }) |control| {
            if (fl.wanted(only, Proj.base(w))) try s.proj.add(s.a, s.kernels, try Proj.init(s.a, s.device, s.queue, w, control));
        };
        for (at.cases) |v| if (fl.wanted(only, at.base(v))) try s.attend.add(s.a, s.kernels, try at.Attend.init(s.a, s.device, s.queue, v));
    }
    pub fn deinit(s: *Suite) void {
        s.proj.deinit(s.a);
        s.attend.deinit(s.a);
    }
};

/// Gate 5's texts for `hy4 units mla_nax ...`: frags.source's text here; before the M5, nax.h's simdgroup fragments.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !?h.fl.Texts {
    if (!std.mem.startsWith(u8, name, "hy4 units mla_nax ")) return null;
    const body = try std.mem.replaceOwned(u8, a, sources.hy4_flash_mla_nax, "#include \"../nax.h\"", sources.nax);
    const pre = try std.mem.concat(a, u8, &.{ "#define TF_SIMD_FRAGS 1\n", body });
    return .{ .here = if (device.tensorUnits()) try tf.frags.source(device, a, sources.hy4_flash_mla_nax) else pre, .pre_m5 = pre };
}
