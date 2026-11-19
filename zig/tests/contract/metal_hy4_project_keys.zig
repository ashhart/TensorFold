//! tf_hy4_cache_keys (the latent's RMS norm and the key's rotation into each row's cache slot), tf_hy4_gate_values.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const pj = @import("metal_hy4_project.zig");

const fl = h.fl;
const c = contract.spec;
const g = contract.gates;
const Range = fl.Range;
const bf = fl.bf;
const u = fl.u32_ulp;
const mw = h.hy.mla_weights;
const latent = mw.latent;
const slot = mw.absorbed_stride;
const width = mw.rope;
const huge = std.math.floatMax(f64);
const any = Range{ .lo = -huge, .hi = huge };

/// A row's squares: four fma steps a thread, the 32-lane xor tree (5), then the four simdgroups' partials (2).
fn keyDepth(_: c.Shape) usize {
    return 4 + 5 + 2;
}

/// Rows: the call's rows; n: a slot's 576 words; k: the 512 squares a row's norm sums.
fn keySpec(label: []const u8) c.Spec {
    const served = [_]c.Shape{ .{ .rows = h.window_rows, .n = slot, .k = latent }, .{ .rows = h.decode_rows, .n = slot, .k = latent }, .{ .rows = h.prompt_rows, .n = slot, .k = latent } };
    return .{
        .name = label,
        .accum = .f32,
        .depth = keyDepth,
        .out = .bf16,
        .served = &served,
        .sums = .{ .bf16 = 0, .why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps" },
    };
}

pub const Keys = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: ?*const pj.Control,
    eps: f32,
    label: [224]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    cap: usize = 0,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    gain: []u16 = &.{},
    tab: []u16 = &.{},
    slots: []i32 = &.{},
    shown: std.ArrayList(u8) = .empty,
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: ?*const pj.Control) !Keys {
        const cfg = try h.text(a);
        defer cfg.deinit();
        const eps = cfg.value.text_config.rms_norm_eps;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pj.pipe(a, device, "tf_hy4_cache_keys", ctl), .ctl = ctl, .eps = eps };
    }
    pub fn deinit(t: *Keys) void {
        t.free();
        t.shown.deinit(t.a);
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Keys) void {
        inline for (.{ "x", "gain", "tab", "slots" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Keys) g.Kernel {
        t.name = pj.name(&t.label, "tf_hy4_cache_keys", "(latent norm and key rotation into shuffled slots)", t.ctl);
        return .{ .spec = keySpec(t.name), .ctx = t, .vt = &keys_vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const keys_vt = blk: {
        var v = g.hooks(Keys);
        v.output = output;
        v.intact = intact;
        break :blk v;
    };
    fn of(ctx: *anyopaque) *Keys {
        return @ptrCast(@alignCast(ctx));
    }
    fn cache(t: *Keys) []u8 {
        return t.rig.bytes(t.out, t.cap * slot * 2);
    }

    pub fn prepare(t: *Keys, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != slot or shape.k != latent) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.cap = t.rows + 7;
        t.fill = fill;
        t.x = try t.a.alloc(u16, t.rows * slot);
        t.gain = try t.a.alloc(u16, latent);
        t.tab = try t.a.alloc(u16, t.rows * width);
        t.slots = try t.a.alloc(i32, t.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // distinct slots in shuffled order, seven left unwritten
        const order = try t.a.alloc(i32, t.cap);
        defer t.a.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        rng.shuffle(i32, order);
        @memcpy(t.slots, order[0..t.rows]);
        pj.uniform(rng, t.x, 3);
        for (t.gain) |*v| v.* = c.toBf16(rng.float(f32) * 2);
        pj.angles(rng, t.tab);
        if (fill == .stress) {
            // integer latents whose squares sum exactly in fp32 (512 x 1,600 at most) and overflow fp16 in any order
            for (0..t.rows) |r| for (t.x[r * slot ..][0..latent]) |*v| {
                v.* = c.toBf16(@floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1)));
            };
            @memset(t.gain, c.toBf16(1));
        }
        if (fill == .nonfinite) {
            // a latent NaN, +Inf and -Inf a row, both infinities in row 3's key pair and in row 4's latent
            const at = [_][2]usize{ .{ 0, 7 }, .{ 1, 100 }, .{ 2, 300 }, .{ 3, latent + 4 }, .{ 3, latent + 9 }, .{ 4, 5 }, .{ 4, 6 } };
            const vals = [_]f32{ fl.nan32, fl.inf32, -fl.inf32, fl.inf32, -fl.inf32, fl.inf32, -fl.inf32 };
            for (at, vals) |p, v| if (p[0] < t.rows) {
                t.x[p[0] * slot + p[1]] = c.toBf16(v);
            };
        }
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(t.x), std.mem.sliceAsBytes(t.gain), std.mem.sliceAsBytes(t.tab), std.mem.sliceAsBytes(t.slots) };
        for (bytes, &t.ins) |b, *ref| ref.* = try t.rig.input(b);
        t.out = try t.rig.output(t.cap * slot * 2);
        return true;
    }

    pub fn run(t: *Keys) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 4);
        p.e.setValue(h.hy.mla_attention.CacheWrite{ .rows = @intCast(t.rows), .capacity = @intCast(t.cap), .epsilon = t.eps }, 5);
        p.e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
    }

    /// Each row's slot, in row order.
    fn output(ctx: *anyopaque) anyerror![]const u8 {
        const t = of(ctx);
        t.shown.clearRetainingCapacity();
        const all = t.cache();
        for (t.slots) |s| try t.shown.appendSlice(t.a, all[@as(usize, @intCast(s)) * slot * 2 ..][0 .. slot * 2]);
        return t.shown.items;
    }

    /// Guards intact, and every slot no row owns still poisoned.
    fn intact(ctx: *anyopaque) anyerror!bool {
        const t = of(ctx);
        if (!try t.rig.intact()) return false;
        const all = t.cache();
        for (0..t.cap) |s| {
            if (std.mem.indexOfScalar(i32, t.slots, @intCast(s)) != null) continue;
            for (all[s * slot * 2 ..][0 .. slot * 2]) |b| if (b != 0xff) return false;
        }
        return true;
    }

    pub fn count(t: *Keys) usize {
        return t.rows * slot;
    }

    fn xv(t: *const Keys, r: usize, e: usize) f64 {
        return c.fromBf16(t.x[r * slot + e]);
    }

    /// Row r's sum of squares: float64's, or replayed in an accumulator and order (gate 4).
    fn squares(t: *const Keys, r: usize, acc: ?c.Accum, order: g.Order) f64 {
        var sq: [latent]f64 = undefined;
        var ss: f64 = 0;
        for (&sq, 0..) |*q, e| {
            q.* = t.xv(r, e) * t.xv(r, e);
            ss += q.*;
        }
        return if (acc) |x| contract.rows.sumIn(x, order, &sq) else ss;
    }

    /// The rope words' exact pair before the fp32 add.
    fn key(t: *const Keys, r: usize, e: usize) f64 {
        const col = (e - latent) & ~@as(usize, 1);
        const v = pj.turn(t.xv(r, latent + col), t.xv(r, latent + col + 1), c.fromBf16(t.tab[r * width + col]), c.fromBf16(t.tab[r * width + col + 1]));
        return v[e % 2];
    }

    fn values(t: *const Keys, out: []f64, acc: ?c.Accum, order: g.Order) void {
        for (0..t.rows) |r| {
            const inv = 1 / @sqrt(t.squares(r, acc, order) / latent + @as(f64, t.eps));
            for (out[r * slot ..][0..slot], 0..) |*o, e| {
                o.* = if (e < latent) bf(bf(t.xv(r, e) * inv) * c.fromBf16(t.gain[e])) else bf(t.key(r, e));
            }
        }
    }

    pub fn reference(t: *Keys, want: []f64, mag: []f64) !void {
        t.values(want, null, .sequential);
        @memset(mag, 0);
    }

    pub fn simulate(t: *Keys, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.values(out, acc, order);
    }

    /// The norm's spread through its fp32 steps and bf16 points; the key words' one fp32 add and bf16 store.
    pub fn intervals(t: *Keys, low: []f64, high: []f64) !void {
        // squares within gamma(depth) (gate 4: exact), /512 exact, the eps add (u/2) and precise rsqrt (2 ulps) in 5u
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(keyDepth(.{ .rows = 0, .n = 0, .k = 0 })), u);
        for (0..t.rows) |r| {
            const ss = t.squares(r, null, .sequential);
            const eps: f64 = t.eps;
            const iv = Range{ .lo = 1 / @sqrt(ss * (1 + gam) / latent + eps) * (1 - 5 * u), .hi = 1 / @sqrt(ss * (1 - gam) / latent + eps) * (1 + 5 * u) };
            for (0..slot) |e| {
                const i = r * slot + e;
                var x: Range = undefined;
                if (e >= latent) {
                    x = pj.turned(t.key(r, e));
                } else if (std.math.isFinite(ss) and std.math.isFinite(t.xv(r, e))) {
                    x = iv.times(t.xv(r, e)).fp32().bf16().times(c.fromBf16(t.gain[e])).bf16();
                } else x = any;
                low[i] = x.lo;
                high[i] = x.hi;
            }
        }
    }

    pub fn bounds(t: *Keys, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }
};

/// Rows: the call's rows; n: its heads, each 256 values gated by their own logits.
fn gateSpec(label: []const u8) c.Spec {
    const served = [_]c.Shape{ .{ .rows = h.window_rows, .n = 32, .k = 0 }, .{ .rows = h.decode_rows, .n = 32, .k = 0 }, .{ .rows = h.prompt_rows, .n = 32, .k = 0 }, .{ .rows = h.window_rows, .n = 64, .k = 0 }, .{ .rows = h.decode_rows, .n = 64, .k = 0 }, .{ .rows = h.prompt_rows, .n = 64, .k = 0 } };
    return .{ .name = label, .accum = .none, .depth = fl.noSums, .out = .bf16, .in_place = true, .served = &served };
}

pub const Gate = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: ?*const pj.Control,
    label: [224]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    cols: usize = 0,
    v: []u16 = &.{},
    z: []u16 = &.{},
    zin: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: ?*const pj.Control) !Gate {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pj.pipe(a, device, "tf_hy4_gate_values", ctl), .ctl = ctl };
    }
    pub fn deinit(t: *Gate) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Gate) void {
        t.a.free(t.v);
        t.a.free(t.z);
        t.v = &.{};
        t.z = &.{};
    }
    pub fn kernel(t: *Gate) g.Kernel {
        t.name = pj.name(&t.label, "tf_hy4_gate_values", "in place", t.ctl);
        return .{ .spec = gateSpec(t.name), .ctx = t, .vt = &gate_vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const gate_vt = blk: {
        var v = g.hooks(Gate);
        v.overlap = overlap;
        break :blk v;
    };
    /// The values are read and written in place.
    fn overlap(_: *anyopaque) bool {
        return true;
    }

    pub fn prepare(t: *Gate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 64) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.cols = shape.n * mw.value_width;
        t.v = try t.a.alloc(u16, t.rows * t.cols);
        t.z = try t.a.alloc(u16, t.rows * t.cols);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        pj.uniform(rng, t.v, 4);
        pj.uniform(rng, t.z, 8);
        if (fill == .nonfinite) {
            // values: NaN, +Inf, -Inf a row, and +Inf under a -Inf logit (Inf x 0); logits: NaN, -Inf (gate 0), +Inf
            const vat = [_][2]usize{ .{ 0, 3 }, .{ 1, 9 }, .{ 2, 17 }, .{ 3, 20 } };
            const vv = [_]f32{ fl.nan32, fl.inf32, -fl.inf32, fl.inf32 };
            for (vat, vv) |p, x| if (p[0] < t.rows) {
                t.v[p[0] * t.cols + p[1]] = c.toBf16(x);
            };
            const zat = [_][2]usize{ .{ 3, 20 }, .{ 3, 21 }, .{ 4, 30 }, .{ 4, 31 } };
            const zv = [_]f32{ -fl.inf32, fl.nan32, -fl.inf32, fl.inf32 };
            for (zat, zv) |p, x| if (p[0] < t.rows) {
                t.z[p[0] * t.cols + p[1]] = c.toBf16(x);
            };
        }
        t.zin = try t.rig.input(std.mem.sliceAsBytes(t.z));
        t.out = try t.rig.output(t.v.len * 2);
        return true;
    }

    pub fn run(t: *Gate) !void {
        @memcpy(t.rig.bytes(t.out, t.v.len * 2), std.mem.sliceAsBytes(t.v));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.out.buf, t.out.off, 0);
        p.e.setBuffer(t.zin.buf, t.zin.off, 1);
        p.e.setValue(@as(u32, @intCast(t.cols)), 2);
        p.e.dispatchThreads(mtl.Size.of(t.cols, t.rows, 1), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Gate) usize {
        return t.v.len;
    }

    pub fn reference(t: *Gate, want: []f64, mag: []f64) !void {
        for (want, t.v, t.z) |*w, v, z| w.* = bf(@as(f64, c.fromBf16(v)) * bf(pj.sigmoid64(c.fromBf16(z))));
        @memset(mag, 0);
    }

    /// The fp32 sigmoid's spread, its bf16 point, the exact product with the bf16 value, the bf16 store.
    pub fn intervals(t: *Gate, low: []f64, high: []f64) !void {
        for (low, high, t.v, t.z) |*l, *hi, v, z| {
            const x: f64 = c.fromBf16(v);
            const zf: f64 = c.fromBf16(z);
            const r = if (std.math.isFinite(x) and std.math.isFinite(zf)) pj.sigmoid(zf).bf16().times(x).bf16() else any;
            l.* = r.lo;
            hi.* = r.hi;
        }
    }

    pub fn bounds(t: *Gate, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }
};

pub fn collect(s: *pj.Suite, only: ?[]const u8) !void {
    var buf: [224]u8 = undefined;
    if (fl.wanted(only, pj.name(&buf, "tf_hy4_cache_keys", "", null))) {
        try s.keys.add(s.a, s.kernels, try Keys.init(s.a, s.device, s.queue, null));
        try s.keys.add(s.a, s.kernels, try Keys.init(s.a, s.device, s.queue, pj.control("tf_hy4_cache_keys", "norm-unrounded")));
    }
    if (fl.wanted(only, pj.name(&buf, "tf_hy4_gate_values", "", null))) {
        try s.gates.add(s.a, s.kernels, try Gate.init(s.a, s.device, s.queue, null));
        try s.gates.add(s.a, s.kernels, try Gate.init(s.a, s.device, s.queue, pj.control("tf_hy4_gate_values", "gate-unrounded")));
    }
}
