//! Flint's decode attention (fl_decode_vec, then fl_merge_vec with the sink) in attention_range's range, sink as a key.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ar = contract.attention_range;
const rig_ = @import("rig.zig");
const fl = @import("metal_flint.zig");

const c = contract.spec;
const g = contract.gates;
const D = fl.head_dim;
const H = fl.heads;
const KVH = fl.kv_heads;
const G = H / KVH;
const ld = KVH * D;

/// glue.metal's fl_vec_args.
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, ring: u32, window: u32, blocks: u32, chunk: u32, first: u32, scale: f32 };

/// fp32 roundings: a score 10; a weight's fast::exp, one a 4-key rescale and the merge's; o's and l's chains in both.
fn form(chunk: usize, blocks: usize) ar.Form {
    const steps: f64 = @floatFromInt(chunk / 4);
    const b: f64 = @floatFromInt(blocks);
    return .{ .score = 10, .fast_exp = false, .chunk_exp = false, .p_rel = 0, .num = 9 * steps + 2 * b + 7, .den = 5 * steps + 2 * b + 8, .div = 3, .exp_ulps = .{ 4 * (steps + 2), 2 } };
}

fn depth(s: c.Shape) usize {
    return 9 * (s.k / 4) + 2 * 544 + 7;
}

const Case = struct { sliding: bool, fault: bool = false };

pub const Decode = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    part: mtl.Pipeline,
    merge: mtl.Pipeline,
    v: Case,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    pos: usize = 0,
    chunk: usize = 0,
    first: usize = 0,
    blocks: usize = 0,
    slots: usize = 0,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    k: []u16 = &.{},
    val: []u16 = &.{},
    sinks: []u16 = &.{},
    ins: [4]rig_.Ref = undefined,
    scratch: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    point: []f64 = &.{},
    half: []f64 = &.{},

    fn base(sliding: bool) []const u8 {
        return if (sliding) "metal flint glue fl_decode_vec + fl_merge_vec sliding (513-key window, ring, sinks)" else "metal flint glue fl_decode_vec + fl_merge_vec full (sinks)";
    }
    const no_sink: fl.Edit = .{ .kernel = "fl_merge_vec", .from = "total = simd_sum(total) + fast::exp(sink - top);", .to = "total = simd_sum(total);" };

    /// Rows: decode rows; n: the first row's position; k: the key block (sliding 16; full by capacity, 32 to 512).
    pub fn spec(name: []const u8, sliding: bool) c.Spec {
        // a window's rows deep in the ring, at the start, past the window's first wrap, a shared forward's 32
        const sl = [_]c.Shape{ .{ .rows = fl.window_rows, .n = 262000, .k = 16 }, .{ .rows = 1, .n = 0, .k = 16 }, .{ .rows = fl.window_rows, .n = 600, .k = 16 }, .{ .rows = fl.batch_rows, .n = 3000, .k = 16 } };
        // a 16k stream's blocks, the full window's 512-key blocks at its last position, a short stream's
        const full = [_]c.Shape{ .{ .rows = fl.window_rows, .n = 9000, .k = 32 }, .{ .rows = 1, .n = 262000, .k = 512 }, .{ .rows = 4, .n = 2000, .k = 64 } };
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 4,
            .out = .bf16,
            .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{} },
            .served = if (sliding) &sl else &full,
            .sums = .{ .bf16 = 0, .why = "the output is a bf16 mean of the values: a bf16 sum's error is a few output ulps" },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Case) !Decode {
        const part = try fl.gluePipe(a, device, "fl_decode_vec", null);
        errdefer part.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .part = part, .merge = try fl.gluePipe(a, device, "fl_merge_vec", if (v.fault) no_sink else null), .v = v };
    }
    fn free(t: *Decode) void {
        inline for (.{ "q", "k", "val", "sinks" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        t.a.free(t.point);
        t.a.free(t.half);
        t.point = &.{};
        t.half = &.{};
    }
    pub fn deinit(t: *Decode) void {
        t.free();
        t.rig.deinit();
        t.part.deinit();
        t.merge.deinit();
    }
    pub fn kernel(t: *Decode) g.Kernel {
        t.name = fl.named(base(t.v.sliding), if (t.v.fault) "the sink dropped from the denominator" else null, &t.label);
        return .{ .spec = spec(t.name, t.v.sliding), .ctx = t, .vt = &vt, .control = if (t.v.fault) .shapes else null };
    }
    const vt = g.hooks(Decode);

    fn slotOf(t: *const Decode, i: usize) usize {
        return if (t.v.sliding) i % fl.ring else i;
    }
    /// Row r's first key: the window's (sliding), else 0.
    fn lo(t: *const Decode, r: usize) usize {
        const p = t.pos + r;
        return if (t.v.sliding and p + 1 > fl.window) p + 1 - fl.window else 0;
    }

    pub fn prepare(t: *Decode, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > fl.batch_rows or shape.k == 0 or shape.k % 4 != 0) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.pos = shape.n;
        t.chunk = shape.k;
        t.fill = fill;
        // as forward.zig: blocks at absolute places from the first row's window
        t.first = t.lo(0) / t.chunk;
        t.blocks = (t.pos + t.rows - 1) / t.chunk - t.first + 1;
        if (t.blocks > 544) return false;
        t.slots = if (t.v.sliding) fl.ring else t.pos + t.rows;
        t.q = try t.a.alloc(u16, t.rows * H * D);
        t.k = try t.a.alloc(u16, t.slots * ld);
        t.val = try t.a.alloc(u16, t.slots * ld);
        t.sinks = try t.a.alloc(u16, H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.q) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.k) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.val) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.sinks) |*x| x.* = c.toBf16(rng.float(f32) * 6 - 3);
        if (fill == .stress) {
            // keys alike (weights exactly 1), the sink -Inf, integer values: sums exact in fp32, past fp16's range
            const scale: u32 = if (t.v.sliding) 4 else 1;
            for (0..t.slots) |s| @memcpy(t.k[s * ld ..][0..ld], t.k[0..ld]);
            for (0..t.slots) |s| for (0..ld) |e| {
                t.val[s * ld + e] = c.toBf16(@floatFromInt(scale * (1 + (s * 7 + e * 13) % 63)));
            };
            @memset(t.sinks, c.toBf16(-fl.inf32));
        }
        if (fill == .nonfinite) {
            // NaN in row 0's head 0; +Inf in row 1's own value (key head 1, dim 7); -Inf in row 2's (key head 2, dim 9)
            t.q[3] = c.toBf16(fl.nan32);
            if (t.rows > 1) t.val[t.slotOf(t.pos + 1) * ld + 1 * D + 7] = c.toBf16(fl.inf32);
            if (t.rows > 2) t.val[t.slotOf(t.pos + 2) * ld + 2 * D + 9] = c.toBf16(-fl.inf32);
        }
        for ([_][]const u8{ std.mem.sliceAsBytes(t.q), std.mem.sliceAsBytes(t.k), std.mem.sliceAsBytes(t.val), std.mem.sliceAsBytes(t.sinks) }, &t.ins) |b, *ref| ref.* = try t.rig.input(b);
        const parts = H * t.rows * t.blocks;
        t.scratch[0] = try t.rig.scratch(parts * 4);
        t.scratch[1] = try t.rig.scratch(parts * 4);
        t.scratch[2] = try t.rig.scratch(parts * D * 4);
        t.out = try t.rig.output(t.count() * 2);
        t.point = try t.a.alloc(f64, t.count());
        t.half = try t.a.alloc(f64, t.count());
        try t.ranges();
        return true;
    }

    fn args(t: *const Decode) VArgs {
        const scale: f32 = @floatCast(1 / @sqrt(@as(f64, D)));
        return .{ .rows = @intCast(t.rows), .pos = @intCast(t.pos), .heads = H, .groups = KVH, .ring = if (t.v.sliding) @intCast(fl.ring) else 0, .window = if (t.v.sliding) fl.window else 0, .blocks = @intCast(t.blocks), .chunk = @intCast(t.chunk), .first = @intCast(t.first), .scale = scale };
    }

    pub fn run(t: *Decode) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.part);
        for (t.ins[0..3], 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setValue(t.args(), 3);
        for (t.scratch, 4..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.dispatchGroups(mtl.Size.of(KVH, t.rows, t.blocks), mtl.Size.of(32, G, 1));
        e.setPipeline(t.merge);
        for (t.scratch, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setValue(t.args(), 3);
        e.setBuffer(t.ins[3].buf, t.ins[3].off, 4);
        e.setBuffer(t.out.buf, t.out.off, 5);
        e.dispatchGroups(mtl.Size.of(H, t.rows, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Decode) usize {
        return t.rows * H * D;
    }

    /// Every output's point and half-range; the sink is one more key (q's extra 1 against sink / scale) of value 0.
    fn ranges(t: *Decode) !void {
        const scale: f64 = t.args().scale;
        const E = D + 1;
        const most = t.pos + t.rows + 1;
        const buf = try t.a.alloc(f64, 2 * most * E + 2 * most + 3 * E);
        defer t.a.free(buf);
        const kf = buf[0 .. most * E];
        const vf = buf[most * E .. 2 * most * E];
        const w = buf[2 * most * E ..][0..most];
        const tau = buf[2 * most * E + most ..][0..most];
        const qf = buf[buf.len - 3 * E ..][0..E];
        const pt = buf[buf.len - 2 * E ..][0..E];
        const hf = buf[buf.len - E ..];
        for (0..t.rows) |row| for (0..KVH) |kv| {
            const keys = t.pos + row + 1 - t.lo(row);
            for (0..keys) |j| {
                const s = t.slotOf(t.lo(row) + j);
                for (0..D) |e| {
                    kf[j * E + e] = c.fromBf16(t.k[s * ld + kv * D + e]);
                    vf[j * E + e] = c.fromBf16(t.val[s * ld + kv * D + e]);
                }
                kf[j * E + D] = 0;
                vf[j * E + D] = 0;
            }
            for (0..G) |gi| {
                const hd = kv * G + gi;
                const sink: f64 = c.fromBf16(t.sinks[hd]);
                const n = if (std.math.isInf(sink) and sink < 0) keys else keys + 1;
                @memset(kf[keys * E ..][0..E], 0);
                @memset(vf[keys * E ..][0..E], 0);
                kf[keys * E + D] = sink / scale;
                for (qf[0..D], 0..) |*x, e| x.* = c.fromBf16(t.q[(row * H + hd) * D + e]);
                qf[D] = 1;
                ar.head(qf, kf[0 .. n * E], vf[0 .. n * E], n, E, scale, form(t.chunk, t.blocks), w[0..n], tau[0..n], pt, hf);
                const at = (row * H + hd) * D;
                @memcpy(t.point[at..][0..D], pt[0..D]);
                @memcpy(t.half[at..][0..D], hf[0..D]);
            }
        };
    }

    pub fn reference(t: *Decode, want: []f64, mag: []f64) !void {
        @memcpy(want, t.point);
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }

    pub fn intervals(t: *Decode, low: []f64, high: []f64) !void {
        for (low, high, t.point, t.half) |*l, *h, p, hf| {
            const fin = std.math.isFinite(p) and std.math.isFinite(hf);
            l.* = if (fin) fl.bf(p - hf) else -std.math.floatMax(f64);
            h.* = if (fin) fl.bf(p + hf) else std.math.floatMax(f64);
        }
    }

    pub fn bounds(t: *Decode, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, t.point, t.half) |*l, w, p, hf| {
            if (t.fill == .stress) {
                l.* = c.boundExact(spec("", t.v.sliding), w);
            } else if (!std.math.isFinite(w)) {
                l.* = 0;
            } else if (!std.math.isFinite(hf)) {
                l.* = std.math.floatMax(f64); // scores too far apart for the range's first order: any finite value
            } else l.* = fl.spread(w, .{ .lo = fl.bf(p - hf), .hi = fl.bf(p + hf) });
        }
    }

    /// Gate 4's sums: every weight is 1 and the sink is gone, so o is V's column mean over the row's keys, rounded.
    pub fn simulate(t: *Decode, acc: c.Accum, order: g.Order, out: []f64) !void {
        const col = try t.a.alloc(f64, t.pos + t.rows + 1);
        defer t.a.free(col);
        for (0..t.rows) |row| for (0..KVH) |kv| {
            const keys = t.pos + row + 1 - t.lo(row);
            for (0..D) |e| {
                for (col[0..keys], 0..) |*x, j| x.* = c.fromBf16(t.val[t.slotOf(t.lo(row) + j) * ld + kv * D + e]);
                const o = fl.bf(contract.rows.sumIn(acc, order, col[0..keys]) / @as(f64, @floatFromInt(keys)));
                for (0..G) |gi| out[(row * H + kv * G + gi) * D + e] = o;
            }
        };
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    tests: fl.List(Decode) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Case{ .{ .sliding = true }, .{ .sliding = false }, .{ .sliding = true, .fault = true } }) |v| {
            if (fl.wanted(only, Decode.base(v.sliding))) try s.tests.add(s.a, s.kernels, try Decode.init(s.a, s.device, s.queue, v));
        }
    }
    pub fn deinit(s: *Suite) void {
        s.tests.deinit(s.a);
    }
};

/// The decode kernels are glue.metal's, which metal_flint_glue.zig compiles for gate 5.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, _: []const u8) !?fl.Texts {
    return null;
}
