//! Unlimited-OCR-2's decoder ud_attend under the contract (DASH: 12 routed 32-key blocks, then the 128-key window).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");

const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;

pub const nan16 = 0x7fc0;
/// Metal's fast-math division (the MLX options leave fast math on): 2.5 ulps, 5 roundings of 2^-24.
pub const div_roundings = 5;
/// precise::exp: 4 ulps.
pub const exp_ulps = [2]f64{ 4, 0 };
/// Gate 4's values: multiples of 64 in 32-key periods (16 values, then their negations, 2 nudged), exact in fp32.
pub const step: f32 = 64;
pub const period = 32;

/// Random bf16 values in [-r, r).
pub fn fillRandom(rng: std.Random, out: []u16, r: f32) void {
    for (out) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * r);
}

/// Gate 4's value at key `j` and column `col` of `base` [period / 2][width].
pub fn stressValue(base: []const f32, width: usize, j: usize, col: usize) f32 {
    const m = j % period;
    if (m < period / 2) return base[m * width + col];
    const x = -base[(m - period / 2) * width + col];
    return if (m - period / 2 < 2) x + step else x;
}

pub fn stressBase(a: std.mem.Allocator, rng: std.Random, width: usize) ![]f32 {
    const base = try a.alloc(f32, period / 2 * width);
    for (base) |*x| x.* = step * @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 255)));
    return base;
}

/// A family source with one control's edit, which must occur exactly once.
pub fn edited(a: std.mem.Allocator, which: uocr.Library, from: []const u8, to: []const u8) ![]u8 {
    const src = uocr.source(which);
    if (std.mem.count(u8, src, from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, src, from, to);
}

// --- ud_attend: a 256-thread group a (row, head) over its selected blocks' keys before the window, then the window ---

/// DASH's geometry (config.zig): 32-key blocks, a 4-block local window, 12 read blocks; the trained context.
const block = 32;
const local = 128;
const reads = 12;
const max_context = 131072;
const D = uocr.head_dim;
const H = uocr.hidden;

/// The kernel and 3 controls: a key past the row; the 12th slot ignored; a block's keys in the window counted twice.
pub const DCase = enum { plain, past_row, slot_dropped, overlap_twice };

pub fn decoderName(k: DCase) []const u8 {
    return switch (k) {
        .plain => "metal uocr attention ud_attend DASH, 12 routed blocks of 32 and a 128-key window, 10 heads of 128",
        .past_row => "control: metal uocr attention ud_attend, a key past the row",
        .slot_dropped => "control: metal uocr attention ud_attend, the 12th selected block ignored",
        .overlap_twice => "control: metal uocr attention ud_attend, a selected block's keys inside the window counted again",
    };
}

fn decoderEdit(k: DCase) [2][]const u8 {
    return switch (k) {
        .plain => unreachable,
        .past_row => .{ "last = pos + 1;", "last = pos + 2;" },
        .slot_dropped => .{ "const int c = sel[row * 12 + s];", "const int c = s == 11 ? -1 : sel[row * 12 + s];" },
        .overlap_twice => .{ "last = max(first, min(first + 32, lo));", "last = first + 32;" },
    };
}

/// Rows: query rows at consecutive positions; n: the read slots a row fills (the rest -1); k: the first row's position.
fn decoderSpec(k: DCase) c.Spec {
    return .{
        .name = decoderName(k),
        .accum = .f32,
        .depth = decoderDepth,
        .ops = div_roundings,
        .out = .bf16,
        // the window's edges: a first row at 0, 127, 128, 129
        .tiles = .{ .{}, .{}, .{ .size = local, .min = 0 } },
        .served = &.{
            // a short chunk whose 6 remote blocks leave 6 slots on blocks at or past the window (each ignored)
            .{ .rows = 8, .n = reads, .k = 300 },
            // decode at the trained context's last position, and mid-context
            .{ .rows = 1, .n = reads, .k = max_context - 1 },
            .{ .rows = 1, .n = reads, .k = 5000 },
            // a prompt's first chunk of 2,048 rows (fewer than 128 keys, then blocks overlapping the window, then 12)
            .{ .rows = 2048, .n = reads, .k = 0 },
            // a query near position 0; a chunk across the window's edge with 3 slots; rows reading no blocks
            .{ .rows = 5, .n = reads, .k = 0 },
            .{ .rows = 4, .n = 3, .k = 126 },
            .{ .rows = 64, .n = 0, .k = 1000 },
            // the blocks next to the window, where the last remote block overlaps it
            .{ .rows = 33, .n = reads, .k = 160 },
        },
        .sums = .{ .exact = true },
    };
}

fn decoderDepth(_: c.Shape) usize {
    return (reads * block + local) / 8 + 10;
}

/// ud_attend over n keys: 4-term dot, simd_sum's 5 levels, * scale, precise exp, fp32 P; chains over every 8th key.
fn decoderForm(n: usize) ar.Form {
    return .{
        .score = 4 + 5 + 1,
        .score_round = 1,
        .fast_exp = false,
        .exp_ulps = exp_ulps,
        .chunk_exp = false,
        .p_rel = 0,
        .num = @floatFromInt((n + 7) / 8 + 1 + 8),
        .den = @floatFromInt((n + 7) / 8 + 8),
        .div = div_roundings,
    };
}

const UdAttend = extern struct { width: u32, scale: f32, local: u32 };
const decoder_scale: f32 = 1.0 / @sqrt(@as(f32, D)); // decoder.zig's

pub const DProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    case: DCase,
    rows: usize = 0,
    n: usize = 0,
    p0: usize = 0,
    ctx: usize = 0,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    sel: []i32 = &.{},
    bias: []f32 = &.{},
    qpos: []u32 = &.{},
    point: []f64 = &.{},
    half: []f64 = &.{},
    refs: [6]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: DCase) !DProbe {
        const pipe = if (k == .plain) try uocr.pipeline(device, .decoder, "ud_attend") else blk: {
            const e = decoderEdit(k);
            const src = try edited(a, .decoder, e[0], e[1]);
            defer a.free(src);
            break :blk try rig_.pipeline(device, src, "ud_attend");
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .case = k };
    }

    fn free(p: *DProbe) void {
        for ([_][]u16{ p.q, p.kc, p.vc }) |x| p.a.free(x);
        for ([_][]f64{ p.point, p.half }) |x| p.a.free(x);
        p.a.free(p.sel);
        p.a.free(p.bias);
        p.a.free(p.qpos);
        p.q = &.{};
        p.kc = &.{};
        p.vc = &.{};
        p.point = &.{};
        p.half = &.{};
        p.sel = &.{};
        p.bias = &.{};
        p.qpos = &.{};
    }

    pub fn deinit(p: *DProbe) void {
        p.free();
        p.rig.deinit();
        p.pipe.deinit();
    }

    pub fn kernel(p: *DProbe) g.Kernel {
        return .{ .spec = decoderSpec(p.case), .ctx = p, .vt = &vt, .control = if (p.case == .plain) null else .shapes };
    }
    const vt = g.hooks(DProbe);

    fn low(pos: usize) usize {
        return (pos + 1) -| local;
    }

    /// Row r's keys in the kernel's order: each slot's block up to the window (its bias), then the window (no bias).
    fn keys(p: *const DProbe, r: usize, at: *std.ArrayList(usize), bs: *std.ArrayList(f64)) !void {
        at.clearRetainingCapacity();
        bs.clearRetainingCapacity();
        const pos = p.p0 + r;
        const lo = low(pos);
        for (0..reads) |s| {
            const cb = p.sel[r * reads + s];
            if (cb < 0) continue;
            const first: usize = @as(usize, @intCast(cb)) * block;
            const last = @max(first, @min(first + block, lo));
            for (first..last) |j| {
                try at.append(p.a, j);
                try bs.append(p.a, p.bias[r * reads + s]);
            }
        }
        for (lo..pos + 1) |j| {
            try at.append(p.a, j);
            try bs.append(p.a, 0);
        }
    }

    /// Each row's slots as ud_route fills them (distinct remote blocks, then ignored ones, rest -1); gate 4: bias 0.
    fn route(p: *DProbe, rng: std.Random) void {
        @memset(p.sel, -1);
        for (p.bias) |*b| b.* = rng.float(f32) * 5 - 4;
        for (0..p.rows) |r| {
            if (p.fill == .stress) {
                for (0..reads) |s| {
                    p.sel[r * reads + s] = @intCast(4 * (s / 2) + 2 * (s % 2));
                    p.bias[r * reads + s] = 0;
                }
                continue;
            }
            const pos = p.p0 + r;
            const last_remote: i64 = @as(i64, @intCast(pos / block)) - 4;
            const remote: usize = @intCast(@max(0, last_remote + 1));
            var picks: [reads]i32 = undefined;
            var got: usize = 0;
            if (p.n > 0 and remote > 0) {
                picks[0] = @intCast(last_remote);
                got = 1;
            }
            while (got < p.n and got < remote) {
                const cand: i32 = @intCast(rng.uintLessThan(usize, remote));
                if (std.mem.indexOfScalar(i32, picks[0..got], cand) == null) {
                    picks[got] = cand;
                    got += 1;
                }
            }
            var next: i32 = @intCast(@max(0, last_remote + 1));
            while (got < p.n) : (next += 1) {
                picks[got] = next;
                got += 1;
            }
            var slots: [reads]u8 = undefined;
            for (&slots, 0..) |*s, i| s.* = @intCast(i);
            rng.shuffle(u8, &slots);
            for (picks[0..got], slots[0..got]) |cb, s| p.sel[r * reads + s] = cb;
        }
    }

    pub fn prepare(p: *DProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n > reads or shape.k + shape.rows > max_context) return false;
        if (fill == .stress and (shape.n != reads or shape.k < 1024)) return false;
        p.rig.reset();
        p.free();
        p.rows = shape.rows;
        p.n = shape.n;
        p.p0 = shape.k;
        p.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.sel = try p.a.alloc(i32, p.rows * reads);
        p.bias = try p.a.alloc(f32, p.rows * reads);
        p.qpos = try p.a.alloc(u32, p.rows);
        for (p.qpos, 0..) |*x, r| x.* = @intCast(p.p0 + r);
        p.route(rng);
        // the cache reaches the chunk's end and every selected block's end; keys no row reads hold NaN
        p.ctx = p.p0 + p.rows;
        for (p.sel) |cb| if (cb >= 0) {
            p.ctx = @max(p.ctx, (@as(usize, @intCast(cb)) + 1) * block);
        };
        p.q = try p.a.alloc(u16, p.rows * H);
        p.kc = try p.a.alloc(u16, p.ctx * H);
        p.vc = try p.a.alloc(u16, p.ctx * H);
        p.point = try p.a.alloc(f64, p.count());
        p.half = try p.a.alloc(f64, p.count());
        @memset(p.kc, nan16);
        @memset(p.vc, nan16);
        fillRandom(rng, p.q, 3);
        const base: []f32 = if (fill == .stress) try stressBase(p.a, rng, H) else &.{};
        defer p.a.free(base);
        var at: std.ArrayList(usize) = .empty;
        defer at.deinit(p.a);
        var bs: std.ArrayList(f64) = .empty;
        defer bs.deinit(p.a);
        const read = try p.a.alloc(bool, p.ctx);
        defer p.a.free(read);
        @memset(read, false);
        for (0..p.rows) |r| {
            try p.keys(r, &at, &bs);
            for (at.items) |j| read[j] = true;
        }
        for (read, 0..) |seen, j| if (seen) {
            if (fill == .stress) {
                @memset(p.kc[j * H ..][0..H], 0);
                for (0..H) |col| p.vc[j * H + col] = c.toBf16(stressValue(base, H, j, col));
            } else {
                fillRandom(rng, p.kc[j * H ..][0..H], 3);
                fillRandom(rng, p.vc[j * H ..][0..H], 2);
            }
        };
        if (fill == .stress) @memset(p.q, 0);
        if (fill == .nonfinite) p.plant();
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.kc));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.vc));
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.sel));
        p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
        p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(p.qpos));
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// Gate 1's plants (rows 0 to 7): NaN/Inf in q, keys, values and bias, an all -Inf row; unread keys hold NaN.
    fn plant(p: *DProbe) void {
        const nan = std.math.nan(f32);
        const inf = std.math.inf(f32);
        const rows = p.rows;
        if (rows < 7) return;
        p.q[0 * H + 1 * D + 3] = c.toBf16(nan);
        p.q[1 * H + 0 * D + 5] = c.toBf16(inf);
        p.q[2 * H + 2 * D + 5] = c.toBf16(inf);
        p.q[2 * H + 2 * D + 9] = c.toBf16(-inf);
        p.kc[(p.p0 + 3) * H + 3 * D] = c.toBf16(nan);
        p.q[3 * H + 6 * D] = c.toBf16(-inf);
        for (0..p.ctx) |j| {
            const k = &p.kc[j * H + 6 * D];
            if (k.* != nan16) k.* = c.toBf16(@abs(c.fromBf16(k.*)) + 0.5);
        }
        p.vc[(p.p0 + 4) * H + 4 * D + 7] = c.toBf16(inf);
        p.vc[(p.p0 + 4) * H + 4 * D + 8] = c.toBf16(nan);
        const last_remote = @as(i64, @intCast(p.p0 / block)) - 4;
        if (last_remote >= 0) p.vc[@as(usize, @intCast(last_remote)) * block * H + 5 * D + 9] = c.toBf16(-inf);
        for ([_]usize{ 5, 6 }, [_]f32{ -inf, nan }) |r, v| for (0..reads) |s| {
            if (p.sel[r * reads + s] >= 0 and p.sel[r * reads + s] <= last_remote) {
                p.bias[r * reads + s] = v;
                break;
            }
        };
    }

    pub fn run(p: *DProbe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.refs, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setBuffer(p.out.buf, p.out.off, 6);
        pass.e.setValue(UdAttend{ .width = H, .scale = decoder_scale, .local = local }, 7);
        // as decoder.zig: (rows, heads) groups of 256
        pass.e.dispatchGroups(mtl.Size.of(p.rows, uocr.heads, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const DProbe) usize {
        return p.rows * H;
    }

    /// Each row's heads over its keys, a block's bias folded into the dot as a 129th term (bias / scale against 1).
    pub fn reference(p: *DProbe, want: []f64, mag: []f64) !void {
        const most = reads * block + local;
        const dk = D + 1;
        const buf = try p.a.alloc(f64, 2 * most * dk + 2 * most + 3 * dk);
        defer p.a.free(buf);
        const ks = buf[0 .. most * dk];
        const vs = buf[most * dk .. 2 * most * dk];
        const w = buf[2 * most * dk ..][0..most];
        const tau = buf[2 * most * dk + most ..][0..most];
        const qv = buf[2 * most * dk + 2 * most ..][0..dk];
        const pt = buf[2 * most * dk + 2 * most + dk ..][0..dk];
        const hf = buf[2 * most * dk + 2 * most + 2 * dk ..][0..dk];
        const scale: f64 = decoder_scale;
        var at: std.ArrayList(usize) = .empty;
        defer at.deinit(p.a);
        var bs: std.ArrayList(f64) = .empty;
        defer bs.deinit(p.a);
        for (0..p.rows) |r| {
            try p.keys(r, &at, &bs);
            const n = at.items.len;
            const f = decoderForm(n);
            for (0..uocr.heads) |h| {
                for (at.items, bs.items, 0..) |j, b, i| {
                    for (0..D) |e| {
                        ks[i * dk + e] = c.fromBf16(p.kc[j * H + h * D + e]);
                        vs[i * dk + e] = c.fromBf16(p.vc[j * H + h * D + e]);
                    }
                    ks[i * dk + D] = b / scale;
                    vs[i * dk + D] = 0;
                }
                for (0..D) |e| qv[e] = c.fromBf16(p.q[r * H + h * D + e]);
                qv[D] = 1;
                ar.head(qv, ks[0 .. n * dk], vs[0 .. n * dk], n, dk, scale, f, w, tau, pt, hf);
                const o = r * H + h * D;
                @memcpy(want[o..][0..D], pt[0..D]);
                @memcpy(p.half[o..][0..D], hf[0..D]);
            }
        }
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    pub fn bounds(p: *DProbe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.half) |*l, w, h| l.* = if (p.fill == .stress) c.boundExact(decoderSpec(p.case), w) else ar.allowBf16(w, h, 0, 0);
    }

    pub fn intervals(p: *DProbe, lo: []f64, hi: []f64) !void {
        for (lo, hi, p.point, p.half) |*l, *u, pt, h| {
            const e = if (p.fill == .stress) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.intervalBf16(pt, h, 0, 0);
            l.* = e[0];
            u.* = e[1];
        }
    }

    /// Gate 4: every weight 1, so each output is its keys' values summed in `acc` in the kernel's key order, / n.
    pub fn simulate(p: *DProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        var at: std.ArrayList(usize) = .empty;
        defer at.deinit(p.a);
        var bs: std.ArrayList(f64) = .empty;
        defer bs.deinit(p.a);
        var terms: [reads * block + local]f64 = undefined;
        for (0..p.rows) |r| {
            try p.keys(r, &at, &bs);
            const n = at.items.len;
            for (0..H) |col| {
                for (at.items, terms[0..n]) |j, *x| x.* = c.fromBf16(p.vc[j * H + col]);
                out[r * H + col] = uocr.bf(contract.rows.sumIn(acc, order, terms[0..n]) / @as(f64, @floatFromInt(n)));
            }
        }
    }
};
