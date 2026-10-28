//! Bolt's Mamba conv and scan (nemotron_tree.metal tf_tree_conv_bolt, tf_tree_scan_bolt) under the contract at the
//! shapes Bolt serves them: one stream's chain of 1 to 16 rows (a decode step, or a prompt chunk of up to fused_rows),
//! never replayed rows or a tree (the checkpoint has no draft head), the state read from the stream's slot and stored
//! to a fresh one. The tables are the engine's own (layers.Tables.of over the segment backend.zig builds).
//!
//! The main output (conv: XBC; scan: Y) is checked against float64 as intervals through its bf16 points, with fast
//! math's exp, log and divide each within `fast` of the value. check() also requires the copies bit for bit (the saved
//! inputs RAW and DTRAW, the stored conv taps), the stored scan state within its interval, and every other byte of
//! those buffers untouched.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const bolt = @import("metal_bolt.zig");

const c = contract.spec;
const g = contract.gates;
const nem = tf.nemotron;
const Mut = bolt.Mut;
const Range = bolt.Range;

// Bolt's Mamba dims: the template arguments of the two instantiations (checked against the source in init).
const XD = 8192; // inner: 128 heads x 64
const NG = 8;
const DS = 128;
const KC = 4;
const H = 128;
const DH = 64;
const CD = XD + 2 * NG * DS; // the conv's channels: x, B, C
const PROJ = XD + CD + H; // in_proj's row: z, xBC, dt
const XOFF = XD;
const DTOFF = XD + CD;
const SSZ = XD * DS; // one slot's state, f32
const MW = nem.state.replay_rows;
const conv_inst = "tf_tree_conv<8192, 8, 128, 4, 18560, 8192>";
const scan_inst = "tf_tree_scan<128, 64, 8, 128, 8192, 18560, 18432, 1048576>";
comptime {
    std.debug.assert(PROJ == 18560 and DTOFF == 18432 and SSZ == 1048576 and H * DH == XD);
}

/// The tree source as kernels.zig compiles it before the M5 (Bolt is refused on tensor-unit chips before load).
const tree_source = "#define TF_SIMD_LAYOUT 1\n" ++ sources.nemotron_tree;

/// Fast math's exp, log and divide, relative to the value: Metal states a few ulps plus 2|x| for exp at |x| <= 9 here.
const fast = 0x1p-16;
const u = 0x1p-24;

// The state slot a chain reads, the fresh slot it stores (backend.zig slotFor), slots bound, and the stream's replay id.
const slot_in: u32 = 1;
const slot_out: i32 = 2;
const slots = 3;
const rid: i32 = 1;

/// A chain of `rows` as backend.zig hands it to the forward: prompt chunks and decode steps store after all rows.
fn chain(rows: usize, parity: u1) nem.layers.Tables {
    var cache: nem.state.Cache = undefined;
    cache.replay = 0;
    cache.slot = slot_in;
    cache.rid = rid;
    cache.parity = parity;
    cache.replay_map = std.simd.iota(u8, MW);
    const segs = [_]nem.forward.Seg{.{ .rows = rows, .cache = &cache, .store = .full, .slot = slot_out }};
    return nem.layers.Tables.of(&segs);
}

fn tableBytes(table: anytype, n: usize) []const u8 {
    return std.mem.sliceAsBytes(table[0..@max(n, 8)]);
}

fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// silu over an interval: it falls to its minimum at x0 and rises after.
fn siluRange(r: Range) Range {
    const x0 = -1.2784645427610738;
    const a = silu(r.lo);
    const b = silu(r.hi);
    return .{ .lo = if (r.lo < x0 and r.hi > x0) silu(x0) else @min(a, b), .hi = @max(a, b) };
}

fn softplus(x: f64) f64 {
    return @max(x, 0) + std.math.log1p(@exp(-@abs(x)));
}

fn plus(a: Range, b: Range) Range {
    return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
}

fn point(v: f64) Range {
    return .{ .lo = v, .hi = v };
}

fn sameBits(got: u16, want: u16) bool {
    return got == want or (std.math.isNan(c.fromBf16(got)) and std.math.isNan(c.fromBf16(want)));
}

fn word(b: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, b[i * 2 ..][0..2], .little);
}

fn single(b: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little));
}

/// Rows of a decode step, a prompt chunk (fused_rows), and everything between as the tile edges.
const served_conv = [_]c.Shape{ .{ .rows = 16, .n = CD, .k = KC }, .{ .rows = 1, .n = CD, .k = KC } };
const served_scan = [_]c.Shape{ .{ .rows = 16, .n = XD, .k = DS }, .{ .rows = 1, .n = XD, .k = DS } };

fn convDepth(_: c.Shape) usize {
    return KC;
}

/// A lane's NS = 4 states in order, then simd_sum over 32 lanes, whose order Metal does not state.
fn scanDepth(_: c.Shape) usize {
    return DS / 32 + 31;
}

const conv_rule = "every output's class is float64's on the same formula (silu(-inf) is -inf / inf, NaN)";
const scan_rule = "every output's class is float64's on the same recurrence";

pub fn convSpec() c.Spec {
    return .{
        .name = "metal bolt mamba conv tf_tree_conv_bolt",
        .accum = .f32,
        .depth = convDepth,
        .ops = 4,
        .mids = 1, // bf16(conv) before silu
        .out = .bf16,
        .nan = .{ .stated = conv_rule },
        .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{} },
        .served = &served_conv,
    };
}

pub fn scanSpec() c.Spec {
    return .{
        .name = "metal bolt mamba scan tf_tree_scan_bolt",
        .accum = .f32,
        .depth = scanDepth,
        .ops = 12,
        .mids = 2, // bf16(y) and bf16(silu(z)) before their product
        .out = .bf16,
        .nan = .{ .stated = scan_rule },
        .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{} },
        .served = &served_scan,
    };
}

fn instantiated(inst: []const u8) !void {
    if (std.mem.indexOf(u8, sources.nemotron_tree, inst) == null) return error.BoltInstanceChanged;
}

// ---------------------------------------------------------------------------------------------------------------
// The conv: each channel's 4 taps (3 from the slot's conv state, then the chain's rows) times its f32 weights in an
// fma chain from the bias, bf16, silu, bf16. It also saves each row's input (RAW, for a later replay) and stores the
// last row's 3 newest taps to the fresh slot.

pub const ConvProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mut: ?Mut = null,
    rows: usize = 0,
    fill: g.Fill = .random,
    parity: u1 = 0,
    t: nem.layers.Tables = .{},
    p: []u16 = &.{},
    cs: []u16 = &.{},
    cw: []f32 = &.{},
    cb: []f32 = &.{},
    refs: [4]rig_.Ref = undefined,
    xbc: rig_.Ref = undefined,
    raw: rig_.Ref = undefined,
    cs_out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mut: ?Mut) !ConvProbe {
        try instantiated(conv_inst);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mut = mut, .pipe = try bolt.build(a, device, tree_source, "tf_tree_conv_bolt", mut) };
    }

    fn free(t: *ConvProbe) void {
        for ([_][]u16{ t.p, t.cs }) |b| t.a.free(b);
        for ([_][]f32{ t.cw, t.cb }) |b| t.a.free(b);
        t.p = &.{};
        t.cs = &.{};
        t.cw = &.{};
        t.cb = &.{};
    }

    pub fn deinit(t: *ConvProbe) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *ConvProbe) g.Kernel {
        return .{ .spec = bolt.under(convSpec(), t.mut), .ctx = t, .vt = &vt, .control = if (t.mut) |m| m.gate else null };
    }

    const vt = g.hooks(ConvProbe);

    pub fn prepare(t: *ConvProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != CD or shape.k != KC or shape.rows == 0 or shape.rows > 16) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.fill = fill;
        t.parity = @intCast(seed & 1);
        t.t = chain(t.rows, t.parity);
        t.p = try t.a.alloc(u16, t.rows * PROJ);
        t.cs = try t.a.alloc(u16, slots * (KC - 1) * CD);
        t.cw = try t.a.alloc(f32, KC * CD);
        t.cb = try t.a.alloc(f32, CD);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x636f);
        const rng = prng.random();
        for (t.p) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (t.cs) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (t.cw) |*v| v.* = rng.float(f32) - 0.5;
        for (t.cb) |*v| v.* = (rng.float(f32) * 2 - 1) / 8;
        switch (fill) {
            .random => {},
            .nonfinite => {
                // a NaN, +Inf and -Inf input on rows 0-2, and a NaN in the slot's state (row 0's and row 1's taps)
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (put, 0..) |v, r| if (r < t.rows) {
                    t.p[r * PROJ + XOFF + 5 + 2 * r] = c.toBf16(v);
                };
                t.cs[(slot_in * (KC - 1) + 1) * CD + 11] = c.toBf16(std.math.nan(f32));
            },
            .stress => {
                // every input 1; per channel w0 = big + r and w1 = -big (2^20 to 3 * 2^20), exact in fp32 in any
                // order; a bf16 sum loses r, an fp16 one overflows to inf - inf
                for (0..t.rows) |r| @memset(t.p[r * PROJ + XOFF ..][0..CD], c.toBf16(1));
                @memset(t.cs, c.toBf16(1));
                for (0..CD) |ch| {
                    const big: f32 = @floatFromInt((@as(u32, 1) << 20) * (1 + rng.uintLessThan(u32, 3)));
                    t.cw[ch] = big + @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 200)));
                    t.cw[CD + ch] = -big;
                    t.cw[2 * CD + ch] = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 7))) - 3);
                    t.cw[3 * CD + ch] = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 7))) - 3);
                    t.cb[ch] = 0;
                }
            },
        }
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.p));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.cs));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.cw));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(t.cb));
        t.xbc = try t.rig.output(t.rows * CD * 2);
        t.raw = try t.rig.output(@as(usize, @intCast(rid * 2 + 2)) * MW * CD * 2);
        t.cs_out = try t.rig.output(slots * (KC - 1) * CD * 2);
        return true;
    }

    pub fn run(t: *ConvProbe) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        // as layers.mamba binds it (its tables by value); the engine binds one conv state as both 1 and 10 and stores
        // to a slot it does not read, so here they are two buffers
        e.setBytes(tableBytes(&t.t.va, t.t.virtual), 4);
        e.setBytes(tableBytes(&t.t.vb, t.t.virtual), 5);
        e.setBytes(tableBytes(&t.t.segi, 8 * t.t.segs), 6);
        e.setValue([2]i32{ @intCast(t.t.virtual), MW }, 7);
        e.setBuffer(t.raw.buf, t.raw.off, 8);
        e.setBuffer(t.xbc.buf, t.xbc.off, 9);
        e.setBuffer(t.cs_out.buf, t.cs_out.off, 10);
        e.dispatchThreads(mtl.Size.of(CD, t.t.virtual, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *ConvProbe) usize {
        return t.rows * CD;
    }

    /// Tap lp's input bits: the slot's conv state before the chain, then its rows.
    fn bitsAt(t: *const ConvProbe, lp: i32, ch: usize) u16 {
        if (lp < 0) return t.cs[(slot_in * (KC - 1) + @as(usize, @intCast(lp + KC - 1))) * CD + ch];
        return t.p[@as(usize, @intCast(lp)) * PROJ + XOFF + ch];
    }

    /// Row v's taps as the kernel reads them: ancestors 3, 2 and 1 back (VB), then the row itself (VA).
    fn taps(t: *const ConvProbe, v: usize) [KC]i32 {
        const b = t.t.vb[v];
        return .{ b[0], b[1], b[2], t.t.va[v][1] };
    }

    fn terms(t: *const ConvProbe, v: usize, ch: usize) [KC + 1]f64 {
        const tp = t.taps(v);
        var out: [KC + 1]f64 = undefined;
        out[0] = t.cb[ch];
        for (tp, 0..) |lp, i| out[i + 1] = @as(f64, t.cw[i * CD + ch]) * c.fromBf16(t.bitsAt(lp, ch));
        return out;
    }

    pub fn reference(t: *ConvProbe, want: []f64, mag: []f64) !void {
        for (0..t.rows) |v| for (0..CD) |ch| {
            var acc: f64 = 0;
            for (t.terms(v, ch)) |x| acc += x;
            want[v * CD + ch] = silu(acc);
            mag[v * CD + ch] = @abs(want[v * CD + ch]);
        };
    }

    pub fn intervals(t: *ConvProbe, low: []f64, high: []f64) !void {
        for (0..t.rows) |v| for (0..CD) |ch| {
            var acc: f64 = 0;
            var m: f64 = 0;
            for (t.terms(v, ch)) |x| {
                acc += x;
                m += @abs(x);
            }
            // the fma chain within gamma(4) of its terms; gate 4's inputs are exact in fp32 in any order
            const e = if (t.fill == .stress) 0 else c.gamma(KC, u) * m;
            const out = siluRange((Range{ .lo = acc - e, .hi = acc + e }).bf16()).widen(2 * fast + 3 * u).bf16();
            low[v * CD + ch] = out.lo;
            high[v * CD + ch] = out.hi;
        };
    }

    pub fn bounds(t: *ConvProbe, want: []const f64, _: []const f64, allow: []f64) !void {
        const ranges = try t.a.alloc(f64, t.count() * 2);
        defer t.a.free(ranges);
        const low = ranges[0..t.count()];
        const high = ranges[t.count()..];
        try t.intervals(low, high);
        for (allow, want, low, high) |*v, w, lo, hi| v.* = @max(@abs(w - lo), @abs(hi - w));
    }

    pub fn simulate(t: *ConvProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.rows) |v| for (0..CD) |ch| {
            const x = t.terms(v, ch);
            out[v * CD + ch] = silu(contract.rows.sumIn(acc, order, &x));
        };
    }

    /// XBC against float64 (its class on gate 1, its interval otherwise); each row's input saved to RAW at the other
    /// parity bit for bit; the last row's 3 newest taps stored to the fresh slot bit for bit; nothing else written.
    pub fn check(t: *ConvProbe, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const out = t.rig.bytes(t.xbc, n * 2);
        {
            const buf = try t.a.alloc(f64, 2 * n);
            defer t.a.free(buf);
            const lo = buf[0..n];
            const hi = buf[n..];
            if (at == .nonfinite) {
                try t.reference(lo, hi);
                for (lo, 0..) |w, i| {
                    const got = c.read(.bf16, out, i);
                    if (c.classOf(got) != c.classOf(w)) return try std.fmt.bufPrint(&t.msg, "XBC[{d}] = {d} is not of float64's class ({d})", .{ i, got, w });
                }
            } else {
                try t.intervals(lo, hi);
                for (lo, hi, 0..) |l, h, i| {
                    const got = c.read(.bf16, out, i);
                    if (!(got >= l and got <= h)) return try std.fmt.bufPrint(&t.msg, "XBC[{d}] = {d} outside float64's [{d}, {d}] through its bf16 points", .{ i, got, l, h });
                }
            }
        }
        const raw = t.rig.bytes(t.raw, @as(usize, @intCast(rid * 2 + 2)) * MW * CD * 2);
        const first: usize = @intCast((rid * 2 + 1 - @as(i32, t.parity)) * MW);
        for (0..@as(usize, @intCast(rid * 2 + 2)) * MW) |r| for (0..CD) |ch| {
            const want: u16 = if (r >= first and r < first + t.rows) t.p[(r - first) * PROJ + XOFF + ch] else 0xffff;
            if (!sameBits(word(raw, r * CD + ch), want)) return try std.fmt.bufPrint(&t.msg, "RAW row {d} channel {d} = {x}, want {x}", .{ r, ch, word(raw, r * CD + ch), want });
        };
        const cs = t.rig.bytes(t.cs_out, slots * (KC - 1) * CD * 2);
        const last = t.taps(t.rows - 1);
        for (0..slots) |s| for (0..KC - 1) |i| for (0..CD) |ch| {
            const want: u16 = if (s == slot_out) t.bitsAt(last[i + 1], ch) else 0xffff;
            const got = word(cs, (s * (KC - 1) + i) * CD + ch);
            if (!sameBits(got, want)) return try std.fmt.bufPrint(&t.msg, "conv state slot {d} tap {d} channel {d} = {x}, want {x}", .{ s, i, ch, got, want });
        };
        return null;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// The scan: per head h and channel d, 128 f32 states over the chain's rows, st = exp(A dt) st + x dt B, y = st . C
// (a lane's 4 states in order, then simd_sum), y = bf16(y + x D), Y = bf16(bf16(silu(z)) * y). dt is
// softplus(raw + bias) clamped to [0, inf]. It saves each row's raw dt (DTRAW) and stores the last row's state to the
// fresh slot.

pub const ScanProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mut: ?Mut = null,
    rows: usize = 0,
    fill: g.Fill = .random,
    parity: u1 = 0,
    t: nem.layers.Tables = .{},
    p: []u16 = &.{},
    xbc: []u16 = &.{},
    s_in: []f32 = &.{},
    heads: []f32 = &.{}, // A_LOG, then D, then dt_bias: H each
    refs: [6]rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    dtraw: rig_.Ref = undefined,
    s_out: rig_.Ref = undefined,
    tree: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mut: ?Mut) !ScanProbe {
        try instantiated(scan_inst);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mut = mut, .pipe = try bolt.build(a, device, tree_source, "tf_tree_scan_bolt", mut) };
    }

    fn free(t: *ScanProbe) void {
        for ([_][]u16{ t.p, t.xbc }) |b| t.a.free(b);
        for ([_][]f32{ t.s_in, t.heads }) |b| t.a.free(b);
        t.p = &.{};
        t.xbc = &.{};
        t.s_in = &.{};
        t.heads = &.{};
    }

    pub fn deinit(t: *ScanProbe) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *ScanProbe) g.Kernel {
        return .{ .spec = bolt.under(scanSpec(), t.mut), .ctx = t, .vt = &vt, .control = if (t.mut) |m| m.gate else null };
    }

    const vt = g.hooks(ScanProbe);

    /// B's and C's partner states in gate 4's inputs: each of the first 64 pairs with one of the last, rotated.
    fn partner(s: usize) usize {
        return 64 + (s * 5 + 3) % 64;
    }

    pub fn prepare(t: *ScanProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != XD or shape.k != DS or shape.rows == 0 or shape.rows > 16) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.fill = fill;
        t.parity = @intCast(seed & 1);
        t.t = chain(t.rows, t.parity);
        t.p = try t.a.alloc(u16, t.rows * PROJ);
        t.xbc = try t.a.alloc(u16, t.rows * CD);
        t.s_in = try t.a.alloc(f32, slots * SSZ);
        t.heads = try t.a.alloc(f32, 3 * H);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x7363);
        const rng = prng.random();
        // z in [-3, 3), raw dt in [-2, 2); x, B, C in [-1, 1); A_LOG, D and dt_bias in [-1, 1) (so A dt >= -8.3)
        for (t.p) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (0..t.rows) |r| for (t.p[r * PROJ ..][0..XD]) |*v| {
            v.* = c.toBf16(rng.float(f32) * 6 - 3);
        };
        for (t.xbc) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.s_in) |*v| v.* = rng.float(f32) * 2 - 1;
        for (t.heads) |*v| v.* = rng.float(f32) * 2 - 1;
        switch (fill) {
            .random => {},
            .nonfinite => {
                // x NaN (row 0, channel 3), B +Inf (row 1, group 0), z -Inf (row 2, channel 7), C NaN (row 3, group 1)
                t.xbc[3] = c.toBf16(std.math.nan(f32));
                if (t.rows > 1) t.xbc[CD + XD + 5] = c.toBf16(std.math.inf(f32));
                if (t.rows > 2) t.p[2 * PROJ + 7] = c.toBf16(-std.math.inf(f32));
                if (t.rows > 3) t.xbc[3 * CD + XD + NG * DS + DS + 9] = c.toBf16(std.math.nan(f32));
            },
            .stress => {
                // A_LOG -inf (exp(A dt) is exp(0) = 1), B 0, D 1, z 64 (silu(z) = 64); the slot's states integers to
                // 2,000 and C integers to 7, each state's term cancelled by its partner's: the dot is 0, exact in fp32 in
                // any order (partials under 2^20), while bf16 sums keep a remainder and fp16 sums overflow
                @memset(t.heads[0..H], -std.math.inf(f32));
                @memset(t.heads[H .. 2 * H], 1);
                for (0..t.rows) |r| {
                    @memset(t.p[r * PROJ ..][0..XD], c.toBf16(64));
                    const row = t.xbc[r * CD ..][0..CD];
                    for (row[0..XD]) |*v| v.* = c.toBf16(@floatFromInt(1 + rng.uintLessThan(u32, 7)));
                    @memset(row[XD .. XD + NG * DS], 0);
                    for (0..NG) |gi| for (0..64) |s| {
                        const cv: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                        row[XD + NG * DS + gi * DS + s] = c.toBf16(cv);
                        row[XD + NG * DS + gi * DS + partner(s)] = c.toBf16(-cv);
                    };
                }
                for (0..XD) |cx| for (0..64) |s| {
                    const sv: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 2000));
                    t.s_in[slot_in * SSZ + cx * DS + s] = sv;
                    t.s_in[slot_in * SSZ + cx * DS + partner(s)] = sv;
                };
            },
        }
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.p));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.xbc));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.s_in));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(t.heads[0..H]));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(t.heads[H .. 2 * H]));
        t.refs[5] = try t.rig.input(std.mem.sliceAsBytes(t.heads[2 * H ..]));
        t.y = try t.rig.output(t.rows * XD * 2);
        t.dtraw = try t.rig.output(@as(usize, @intCast(rid * 2 + 2)) * MW * H * 2);
        t.s_out = try t.rig.output(slots * SSZ * 4);
        t.tree = try t.rig.output(SSZ * 4);
        return true;
    }

    pub fn run(t: *ScanProbe) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        // as layers.mamba binds it (its tables by value; one state buffer as 2 and 14 there, two buffers here)
        e.setValue([2]f32{ 0.0, std.math.inf(f32) }, 6);
        e.setValue([2]i32{ @intCast(t.t.virtual), MW }, 7);
        e.setBytes(tableBytes(&t.t.va, t.t.virtual), 8);
        e.setBytes(tableBytes(&t.t.vb, t.t.virtual), 9);
        e.setBytes(tableBytes(&t.t.vc, t.t.virtual), 10);
        e.setBytes(tableBytes(&t.t.segi, 8 * t.t.segs), 11);
        e.setBuffer(t.dtraw.buf, t.dtraw.off, 12);
        e.setBuffer(t.y.buf, t.y.off, 13);
        e.setBuffer(t.s_out.buf, t.s_out.off, 14);
        e.setBuffer(t.tree.buf, t.tree.off, 15);
        e.dispatchThreads(mtl.Size.of(32, DH, H), mtl.Size.of(32, 8, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *ScanProbe) usize {
        return t.rows * XD;
    }

    fn x(t: *const ScanProbe, v: usize, i: usize) f64 {
        return c.fromBf16(t.xbc[v * CD + i]);
    }

    /// B's and C's first index for channel cx's group.
    fn bAt(cx: usize) usize {
        return XD + cx / DH / (H / NG) * DS;
    }

    fn rawDt(t: *const ScanProbe, v: usize, h: usize) f64 {
        return c.fromBf16(t.p[v * PROJ + DTOFF + h]);
    }

    /// Channel cx's float64 recurrence: Y's rows (the formula, no bf16 points); with an accumulator, the dot's sums as
    /// it would round them in that order.
    fn exact(t: *const ScanProbe, cx: usize, sim: ?struct { acc: c.Accum, order: g.Order }, out: []f64) void {
        const h = cx / DH;
        const A = -@exp(@as(f64, t.heads[h]));
        const dskip: f64 = bolt.bf16of(t.heads[H + h]);
        var st: [DS]f64 = undefined;
        for (&st, 0..) |*v, s| v.* = t.s_in[slot_in * SSZ + cx * DS + s];
        const b0 = bAt(cx);
        var terms: [DS]f64 = undefined;
        for (0..t.rows) |v| {
            const dt = std.math.clamp(softplus(t.rawDt(v, h) + t.heads[2 * H + h]), 0, std.math.inf(f64));
            const dA = @exp(A * dt);
            const xv = t.x(v, cx);
            for (&st, &terms, 0..) |*sv, *term, s| {
                sv.* = dA * sv.* + xv * dt * t.x(v, b0 + s);
                term.* = sv.* * t.x(v, b0 + NG * DS + s);
            }
            var acc: f64 = 0;
            if (sim) |m| acc = contract.rows.sumIn(m.acc, m.order, &terms) else for (terms) |term| {
                acc += term;
            }
            out[v * XD + cx] = silu(c.fromBf16(t.p[v * PROJ + cx])) * (acc + xv * dskip);
        }
    }

    pub fn reference(t: *ScanProbe, want: []f64, mag: []f64) !void {
        for (0..XD) |cx| t.exact(cx, null, want);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn simulate(t: *ScanProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..XD) |cx| t.exact(cx, .{ .acc = acc, .order = order }, out);
    }

    /// Channel cx's intervals: each op of the kernel's f32 arithmetic widened by its rounding (fast math's exp, log and
    /// divide by `fast`), the dot within gamma(depth) of its terms (exact on gate 4's inputs), and the bf16 points.
    fn ranges(t: *const ScanProbe, cx: usize, low: []f64, high: []f64, state: ?[]Range) void {
        const h = cx / DH;
        const ea = @exp(@as(f64, t.heads[h]));
        const A = Range{ .lo = -ea * (1 + fast), .hi = -ea * (1 - fast) };
        const dskip: f64 = bolt.bf16of(t.heads[H + h]);
        var st: [DS]Range = undefined;
        for (&st, 0..) |*v, s| v.* = point(t.s_in[slot_in * SSZ + cx * DS + s]);
        const b0 = bAt(cx);
        for (0..t.rows) |v| {
            const xin = point(t.rawDt(v, h) + t.heads[2 * H + h]).widen(u);
            var dt = Range{ .lo = softplus(xin.lo), .hi = softplus(xin.hi) };
            const e = 2 * fast + 3 * u + u * @abs(dt.hi);
            dt = .{ .lo = @max(0, dt.lo - e), .hi = dt.hi + e };
            const adt = A.times(dt).widen(u);
            // A 0 (gate 4's A_LOG -inf): exp(0) is 1 exactly
            const dA = if (ea == 0) point(1) else (Range{ .lo = @exp(adt.lo), .hi = @exp(adt.hi) }).widen(fast);
            const xv = t.x(v, cx);
            const xdt = point(xv).times(dt).widen(u);
            // gate 4's integers (dA 1, B 0, products under 2^24) make every step exact in fp32
            const r: f64 = if (t.fill == .stress) 0 else u;
            var acc = point(0);
            var m: f64 = 0;
            for (&st, 0..) |*sv, s| {
                sv.* = plus(dA.times(sv.*).widen(r), xdt.scale(t.x(v, b0 + s)).widen(r)).widen(r);
                const term = sv.scale(t.x(v, b0 + NG * DS + s)).widen(r);
                acc = plus(acc, term);
                m += @max(@abs(term.lo), @abs(term.hi));
            }
            if (t.fill != .stress) {
                const e2 = c.gamma(@floatFromInt(scanDepth(.{ .rows = 0, .n = 0, .k = 0 })), u) * m;
                acc = .{ .lo = acc.lo - e2, .hi = acc.hi + e2 };
            }
            const y = plus(acc, point(xv * dskip).widen(u)).widen(u).bf16();
            const sz = point(silu(c.fromBf16(t.p[v * PROJ + cx]))).widen(2 * fast + 3 * u).bf16();
            const out = sz.times(y).widen(u).bf16();
            low[v * XD + cx] = out.lo;
            high[v * XD + cx] = out.hi;
        }
        if (state) |keep| @memcpy(keep, &st);
    }

    pub fn intervals(t: *ScanProbe, low: []f64, high: []f64) !void {
        for (0..XD) |cx| t.ranges(cx, low, high, null);
    }

    pub fn bounds(t: *ScanProbe, want: []const f64, _: []const f64, allow: []f64) !void {
        const buf = try t.a.alloc(f64, t.count() * 2);
        defer t.a.free(buf);
        const low = buf[0..t.count()];
        const high = buf[t.count()..];
        try t.intervals(low, high);
        for (allow, want, low, high) |*v, w, lo, hi| v.* = @max(@abs(w - lo), @abs(hi - w));
    }

    /// Y against float64 (its class on gate 1, its interval otherwise); each row's raw dt saved at the other parity
    /// bit for bit; the last row's state in the fresh slot within its interval (gates 2-4); nothing else written.
    pub fn check(t: *ScanProbe, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const out = t.rig.bytes(t.y, n * 2);
        const sout = t.rig.bytes(t.s_out, slots * SSZ * 4);
        {
            const buf = try t.a.alloc(f64, 2 * n);
            defer t.a.free(buf);
            const lo = buf[0..n];
            const hi = buf[n..];
            if (at == .nonfinite) {
                try t.reference(lo, hi);
                for (lo, 0..) |w, i| {
                    const got = c.read(.bf16, out, i);
                    if (c.classOf(got) != c.classOf(w)) return try std.fmt.bufPrint(&t.msg, "Y[{d}] = {d} is not of float64's class ({d})", .{ i, got, w });
                }
            } else {
                var st: [DS]Range = undefined;
                for (0..XD) |cx| {
                    t.ranges(cx, lo, hi, &st);
                    for (st, 0..) |r, s| {
                        const got = single(sout, @as(usize, @intCast(slot_out)) * SSZ + cx * DS + s);
                        if (!(got >= r.lo and got <= r.hi)) return try std.fmt.bufPrint(&t.msg, "state channel {d} index {d} = {d} outside float64's [{d}, {d}]", .{ cx, s, got, r.lo, r.hi });
                    }
                }
                for (lo, hi, 0..) |l, h, i| {
                    const got = c.read(.bf16, out, i);
                    if (!(got >= l and got <= h)) return try std.fmt.bufPrint(&t.msg, "Y[{d}] = {d} outside float64's [{d}, {d}] through its bf16 points", .{ i, got, l, h });
                }
            }
        }
        const dtraw = t.rig.bytes(t.dtraw, @as(usize, @intCast(rid * 2 + 2)) * MW * H * 2);
        const first: usize = @intCast((rid * 2 + 1 - @as(i32, t.parity)) * MW);
        for (0..@as(usize, @intCast(rid * 2 + 2)) * MW) |r| for (0..H) |h| {
            const want: u16 = if (r >= first and r < first + t.rows) t.p[(r - first) * PROJ + DTOFF + h] else 0xffff;
            if (!sameBits(word(dtraw, r * H + h), want)) return try std.fmt.bufPrint(&t.msg, "DTRAW row {d} head {d} = {x}, want {x}", .{ r, h, word(dtraw, r * H + h), want });
        };
        for (0..slots) |s| if (s != slot_out) for (0..SSZ) |i| {
            if (@as(u32, @bitCast(single(sout, s * SSZ + i))) != 0xffffffff) return try std.fmt.bufPrint(&t.msg, "state slot {d} index {d} written", .{ s, i });
        };
        const tree = t.rig.bytes(t.tree, SSZ * 4);
        for (0..SSZ) |i| if (@as(u32, @bitCast(single(tree, i))) != 0xffffffff) return try std.fmt.bufPrint(&t.msg, "tree state {d} written (a chain keeps none)", .{i});
        return null;
    }
};

/// The planted controls, each a mutation of the tree source; each names the gate that must catch it.
pub const controls = struct {
    pub const conv_tap = Mut{ .name = "control: bolt mamba conv dropping its oldest tap", .find = "for (int i = 0; i < KC; i++) acc = fma(", .with = "for (int i = 1; i < KC; i++) acc = fma(", .gate = .shapes };
    pub const conv_state = Mut{ .name = "control: bolt mamba conv storing the wrong taps", .find = "= VAL(taps[i + 1]);", .with = "= VAL(taps[i]);", .gate = .shapes };
    pub const conv_half = Mut{ .name = "control: bolt mamba conv summed in half", .find = "acc = fma(CW[i * CD + ch], float(VAL(taps[i])), acc);", .with = "acc = float(half(fma(CW[i * CD + ch], float(VAL(taps[i])), acc)));", .gate = .sums };
    pub const scan_skip = Mut{ .name = "control: bolt mamba scan without its D skip", .find = "bfloat(acc + xv * dskip)", .with = "bfloat(acc)", .gate = .shapes };
    pub const scan_decay = Mut{ .name = "control: bolt mamba scan not decaying its state", .find = "const float sv = dA * st[i] + xdt", .with = "const float sv = st[i] + xdt", .gate = .shapes };
    pub const scan_half = Mut{ .name = "control: bolt mamba scan summing its dot in half", .find = "acc += sv * float(XBC[v * CD + cc + i]);", .with = "acc = float(half(acc + sv * float(XBC[v * CD + cc + i])));", .gate = .sums };
};
