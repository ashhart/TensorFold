//! Unlimited-OCR-2's attention kernels under the contract, each output in attention_range's float64 range: the tower's
//! uv_attention (SAM's 14 x 14 windows and global blocks with their decomposed relative-position bias, and CLIP) and the
//! decoder's ud_attend (DASH: up to 12 routed 32-key blocks with their bias, then the 128-key local window).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");

const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;

const nan16 = 0x7fc0;
/// Metal's fast-math division (the MLX options leave fast math on): 2.5 ulps, 5 roundings of 2^-24.
const div_roundings = 5;
/// precise::exp: 4 ulps.
const exp_ulps = [2]f64{ 4, 0 };
/// Gate 4's values are multiples of 64 up to 255 * 64 in 32-key periods: 16 values, then their negations (2 nudged),
/// so any 32 keys in a row sum to the nudges, while a chain's 16 positive keys, and a 32-lane order's lanes (one sign
/// each), pass fp16's largest; every partial stays exact in fp32 (below 2^19) in any order.
const step: f32 = 64;
const period = 32;

/// Random bf16 values in [-r, r).
fn fillRandom(rng: std.Random, out: []u16, r: f32) void {
    for (out) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * r);
}

/// Gate 4's value at key `j` and column `col` of `base` [period / 2][width].
fn stressValue(base: []const f32, width: usize, j: usize, col: usize) f32 {
    const m = j % period;
    if (m < period / 2) return base[m * width + col];
    const x = -base[(m - period / 2) * width + col];
    return if (m - period / 2 < 2) x + step else x;
}

fn stressBase(a: std.mem.Allocator, rng: std.Random, width: usize) ![]f32 {
    const base = try a.alloc(f32, period / 2 * width);
    for (base) |*x| x.* = step * @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 255)));
    return base;
}

/// A family source with one control's edit, which must occur exactly once.
fn edited(a: std.mem.Allocator, which: uocr.Library, from: []const u8, to: []const u8) ![]u8 {
    const src = uocr.source(which);
    if (std.mem.count(u8, src, from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, src, from, to);
}

// --- uv_attention: a simdgroup a (row, head), softmax(q k^T / 8 + bf16(rel_h + rel_w)) v in a group of `len` rows ------

const Tower = enum { sam_window, sam_global, clip };
/// The engine's kernel, and two planted controls: rel_w read one column over (SAM windows), P x V without its last key.
const TCase = enum { plain, bias_shift, last_key };

const tower_entries = [_]struct { Tower, TCase }{ .{ .sam_window, .plain }, .{ .sam_global, .plain }, .{ .clip, .plain }, .{ .sam_window, .bias_shift }, .{ .clip, .last_key } };

fn towerName(t: Tower, k: TCase) []const u8 {
    return switch (k) {
        .plain => switch (t) {
            .sam_window => "metal uocr attention uv_attention sam 14x14 windows, 12 heads of 64, relative-position bias",
            .sam_global => "metal uocr attention uv_attention sam global, 12 heads of 64, relative-position bias",
            .clip => "metal uocr attention uv_attention clip, 16 heads of 64",
        },
        .bias_shift => "control: metal uocr attention uv_attention sam windows, rel_w read one column over",
        .last_key => "control: metal uocr attention uv_attention clip, P x V without the last key",
    };
}

fn towerWidth(t: Tower) usize {
    return if (t == .clip) uocr.clip_width else uocr.sam_width;
}

fn towerSpec(t: Tower, k: TCase) c.Spec {
    return .{
        .name = towerName(t, k),
        .accum = .f32,
        .depth = towerDepth,
        .ops = div_roundings,
        .out = .bf16,
        // CLIP's keys at 32-lane edges (any group length runs; SAM's are squares of its side)
        .tiles = if (t == .clip) .{ .{}, .{}, .{ .size = 32, .min = 1 } } else .{ .{}, .{}, .{} },
        // rows: query rows launched from row 0 (groups of k rows); n: the width; k: the keys a group (len)
        .served = switch (t) {
            // a 1024 view's 25 windows (64 padded to 70), a 640 tile's 9 (40 padded to 42)
            .sam_window => &.{ .{ .rows = 25 * 196, .n = uocr.sam_width, .k = 196 }, .{ .rows = 9 * 196, .n = uocr.sam_width, .k = 196 } },
            // a 1024 view's 64 x 64 and a 640 tile's 40 x 40: the first query rows of each (every key, every row's bias)
            .sam_global => &.{ .{ .rows = 64, .n = uocr.sam_width, .k = 4096 }, .{ .rows = 200, .n = uocr.sam_width, .k = 1600 } },
            // a 1024 view's 257 rows and a 640 tile's 101
            .clip => &.{ .{ .rows = 257, .n = uocr.clip_width, .k = 257 }, .{ .rows = 101, .n = uocr.clip_width, .k = 101 } },
        },
        .sums = .{ .exact = true },
    };
}

/// A lane's P x V chain over every key; the bounds hook replaces the generic bound.
fn towerDepth(s: c.Shape) usize {
    return s.k + 1;
}

/// uv_attention over n keys: a lane's 64-term fma chain and the exact / 8 (the bias's add on the score), precise exp,
/// P rounded to bf16 for P x V (its sum unrounded), a lane's chain over every key, its strided sum and simd_sum.
fn towerForm(n: usize) ar.Form {
    return .{
        .score = 64 + 1,
        .score_round = 1,
        .fast_exp = false,
        .exp_ulps = exp_ulps,
        .chunk_exp = false,
        .p_rel = 0x1p-8,
        .num = @floatFromInt(n + 1),
        .den = @floatFromInt(n / 32 + 7),
        .div = div_roundings,
    };
}

const UvAttn = extern struct { groups: u32, len: u32, heads: u32, width: u32, side: u32, scale: f32 };
const tower_scale: f32 = 0.125; // 1 / sqrt(64), as vision.zig computes it

const TProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    tower: Tower,
    case: TCase,
    rows: usize = 0,
    len: usize = 0,
    groups: usize = 0,
    side: usize = 0,
    heads: usize = 0,
    width: usize = 0,
    fill: g.Fill = .random,
    qkv: []u16 = &.{},
    rh: []u16 = &.{},
    rw: []u16 = &.{},
    point: []f64 = &.{},
    half: []f64 = &.{},
    refs: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, t: Tower, k: TCase) !TProbe {
        const pipe = switch (k) {
            .plain => try uocr.pipeline(device, .vision, "uv_attention"),
            .bias_shift, .last_key => blk: {
                const e: [2][]const u8 = if (k == .bias_shift) .{ "float(rel_w[rel + j % p.side])", "float(rel_w[rel + (j + 1) % p.side])" } else .{ "for (uint j = 0; j < p.len; j++) {", "for (uint j = 0; j + 1 < p.len; j++) {" };
                const src = try edited(a, .vision, e[0], e[1]);
                defer a.free(src);
                break :blk try rig_.pipeline(device, src, "uv_attention");
            },
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .tower = t, .case = k };
    }

    fn free(p: *TProbe) void {
        for ([_][]u16{ p.qkv, p.rh, p.rw }) |x| p.a.free(x);
        for ([_][]f64{ p.point, p.half }) |x| p.a.free(x);
        p.qkv = &.{};
        p.rh = &.{};
        p.rw = &.{};
        p.point = &.{};
        p.half = &.{};
    }

    fn deinit(p: *TProbe) void {
        p.free();
        p.rig.deinit();
        p.pipe.deinit();
    }

    fn kernel(p: *TProbe) g.Kernel {
        return .{ .spec = towerSpec(p.tower, p.case), .ctx = p, .vt = &vt, .control = if (p.case == .plain) null else .shapes };
    }
    const vt = g.hooks(TProbe);

    pub fn prepare(p: *TProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const width = towerWidth(p.tower);
        if (shape.rows == 0 or shape.n != width or shape.k == 0 or shape.k > 4096) return false;
        const side: usize = if (p.tower == .clip) 0 else std.math.sqrt(shape.k);
        if (p.tower != .clip and side * side != shape.k) return false;
        p.rig.reset();
        p.free();
        p.rows = shape.rows;
        p.len = shape.k;
        p.groups = (shape.rows + shape.k - 1) / shape.k;
        p.side = side;
        p.width = width;
        p.heads = width / 64;
        p.fill = fill;
        const total = p.groups * p.len;
        const stride = 3 * width;
        p.qkv = try p.a.alloc(u16, total * stride);
        const rel = @max(1, p.heads * p.groups * p.len * side);
        p.rh = try p.a.alloc(u16, rel);
        p.rw = try p.a.alloc(u16, rel);
        p.point = try p.a.alloc(f64, p.count());
        p.half = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillRandom(rng, p.qkv, 3);
        for (0..total) |r| fillRandom(rng, p.qkv[r * stride + 2 * width ..][0..width], 2); // values in [-2, 2)
        fillRandom(rng, p.rh, 2);
        fillRandom(rng, p.rw, 2);
        switch (fill) {
            .random => {},
            .nonfinite => p.plant(),
            .stress => {
                // q and k zero, no bias: every weight exactly 1; the values in periods, a partial last period zero
                @memset(p.qkv, 0);
                @memset(p.rh, 0);
                @memset(p.rw, 0);
                const base = try stressBase(p.a, rng, width);
                defer p.a.free(base);
                for (0..total) |r| {
                    const j = r % p.len;
                    for (0..width) |col| p.qkv[r * stride + 2 * width + col] = c.toBf16(if (j < p.len / period * period) stressValue(base, width, j, col) else 0);
                }
            },
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.qkv));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.rh));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.rw));
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// Gate 1's plants in group 0 (rows 0 to 4 at least): q's NaN, +Inf and both infinities; a key's NaN; a value's
    /// +Inf, NaN and -Inf; SAM's bias NaN, -Inf (its keys weigh 0) and +Inf (the row's max, so NaN); and two rows whose
    /// every score is -Inf (NaN in float64): q's -Inf against every key's positive k, and SAM's bias -Inf on every key.
    fn plant(p: *TProbe) void {
        const stride = 3 * p.width;
        const nan = std.math.nan(f32);
        const inf = std.math.inf(f32);
        const at = struct {
            fn q(t: *TProbe, row: usize, h: usize, e: usize, v: f32) void {
                t.qkv[row * 3 * t.width + h * 64 + e] = c.toBf16(v);
            }
        }.q;
        at(p, 0, 1, 3, nan);
        at(p, 1, 0, 5, inf);
        at(p, 2, 2, 5, inf);
        at(p, 2, 2, 9, -inf);
        p.qkv[4 * stride + p.width + 3 * 64] = c.toBf16(nan); // key 4's k, head 3
        at(p, 4, 9, 0, -inf);
        for (0..p.len) |j| {
            const k = &p.qkv[j * stride + p.width + 9 * 64];
            k.* = c.toBf16(@abs(c.fromBf16(k.*)) + 0.5);
        }
        if (p.len > 7) {
            p.qkv[6 * stride + 2 * p.width + 4 * 64 + 7] = c.toBf16(inf);
            p.qkv[6 * stride + 2 * p.width + 4 * 64 + 8] = c.toBf16(nan);
            p.qkv[7 * stride + 2 * p.width + 4 * 64 + 9] = c.toBf16(-inf);
        }
        if (p.side > 0) {
            const rel = struct {
                fn of(t: *TProbe, h: usize, qi: usize) usize {
                    return (h * t.groups * t.len + qi) * t.side;
                }
            }.of;
            p.rh[rel(p, 5, 4)] = c.toBf16(nan);
            p.rw[rel(p, 6, 0) + 1] = c.toBf16(-inf);
            p.rw[rel(p, 7, 3) + 2] = c.toBf16(inf);
            @memset(p.rh[rel(p, 8, 1)..][0..p.side], c.toBf16(-inf));
        }
    }

    pub fn run(p: *TProbe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.refs, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        pass.e.setValue(UvAttn{ .groups = @intCast(p.groups), .len = @intCast(p.len), .heads = @intCast(p.heads), .width = @intCast(p.width), .side = @intCast(p.side), .scale = tower_scale }, 4);
        // as vision.zig's attention(): a 32-thread group a (row, head)
        pass.e.dispatchGroups(mtl.Size.of(p.rows, p.heads, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const TProbe) usize {
        return p.rows * p.width;
    }

    /// Each launched row's heads over its group's keys, the bias bf16(rel_h + rel_w) as the kernel rounds it (folded
    /// into the dot as a 65th term, bias * 8 against 1): float64's points and the ranges fp32 allows.
    pub fn reference(p: *TProbe, want: []f64, mag: []f64) !void {
        const L = p.len;
        const dk = 65;
        const buf = try p.a.alloc(f64, 2 * L * dk + 2 * L + 3 * dk);
        defer p.a.free(buf);
        const ks = buf[0 .. L * dk];
        const vs = buf[L * dk .. 2 * L * dk];
        const w = buf[2 * L * dk ..][0..L];
        const tau = buf[2 * L * dk + L ..][0..L];
        const qv = buf[2 * L * dk + 2 * L ..][0..dk];
        const pt = buf[2 * L * dk + 2 * L + dk ..][0..dk];
        const hf = buf[2 * L * dk + 2 * L + 2 * dk ..][0..dk];
        const stride = 3 * p.width;
        const scale: f64 = tower_scale;
        const f = towerForm(L);
        for (0..p.groups) |grp| for (0..p.heads) |h| {
            for (0..L) |j| {
                const row = grp * L + j;
                for (0..64) |e| {
                    ks[j * dk + e] = c.fromBf16(p.qkv[row * stride + p.width + h * 64 + e]);
                    vs[j * dk + e] = c.fromBf16(p.qkv[row * stride + 2 * p.width + h * 64 + e]);
                }
                ks[j * dk + 64] = 0;
                vs[j * dk + 64] = 0;
            }
            for (grp * L..@min(p.rows, (grp + 1) * L)) |r| {
                const qi = r - grp * L;
                for (0..64) |e| qv[e] = c.fromBf16(p.qkv[r * stride + h * 64 + e]);
                qv[64] = 1;
                if (p.side > 0) {
                    const rel = ((h * p.groups + grp) * L + qi) * p.side;
                    for (0..L) |j| {
                        const b = uocr.bf(@as(f64, c.fromBf16(p.rh[rel + j / p.side])) + c.fromBf16(p.rw[rel + j % p.side]));
                        ks[j * dk + 64] = b / scale;
                    }
                }
                ar.head(qv, ks, vs, L, dk, scale, f, w, tau, pt, hf);
                const o = r * p.width + h * 64;
                @memcpy(want[o..][0..64], pt[0..64]);
                @memcpy(p.half[o..][0..64], hf[0..64]);
            }
        };
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    /// The range's allowance (attention_range.allowBf16); gate 4 allows the division and the rounding alone.
    pub fn bounds(p: *TProbe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.half) |*l, w, h| l.* = if (p.fill == .stress) c.boundExact(towerSpec(p.tower, p.case), w) else ar.allowBf16(w, h, 0, 0);
    }

    /// Each bf16 output within its range's rounded ends; gate 4's are open.
    pub fn intervals(p: *TProbe, low: []f64, high: []f64) !void {
        for (low, high, p.point, p.half) |*l, *hi, pt, h| {
            const e = if (p.fill == .stress) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.intervalBf16(pt, h, 0, 0);
            l.* = e[0];
            hi.* = e[1];
        }
    }

    /// Gate 4: every weight 1, so each output is its group's values summed in `acc` over the keys, then / len.
    pub fn simulate(p: *TProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.len);
        defer p.a.free(terms);
        const stride = 3 * p.width;
        for (0..p.rows) |r| for (0..p.width) |col| {
            const grp = r / p.len;
            for (terms, 0..) |*x, j| x.* = c.fromBf16(p.qkv[(grp * p.len + j) * stride + 2 * p.width + col]);
            out[r * p.width + col] = uocr.bf(contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(p.len)));
        };
    }
};

// --- ud_attend: a 256-thread group a (row, head) over its selected blocks' keys before the window, then the window ---

/// DASH's geometry (config.zig): 32-key blocks, a 4-block local window, 12 read blocks; the trained context.
const block = 32;
const local = 128;
const reads = 12;
const max_context = 131072;
const D = uocr.head_dim;
const H = uocr.hidden;

/// The engine's kernel, and three planted controls: each row sees one key past its own; the 12th slot's block ignored;
/// a selected block's keys inside the window (or past it) counted as the block's as well.
const DCase = enum { plain, past_row, slot_dropped, overlap_twice };

fn decoderName(k: DCase) []const u8 {
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

/// ud_attend over n keys: a lane's 4-term dot and simd_sum's 5 levels, * scale (the bias's add on the score), precise
/// exp, fp32 P; a simdgroup's chain over every 8th key and the 8 partials' adds, for o and for l.
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

const DProbe = struct {
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

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: DCase) !DProbe {
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

    fn deinit(p: *DProbe) void {
        p.free();
        p.rig.deinit();
        p.pipe.deinit();
    }

    fn kernel(p: *DProbe) g.Kernel {
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

    /// Each row's slots as ud_route fills them: distinct remote blocks (c <= pos / 32 - 4), the one next to the
    /// window first (it overlaps the window unless pos % 32 is 31), then blocks at or past the window, which the kernel
    /// must ignore; in random slots, the rest -1. Gate 4: blocks 4i and 4i + 2 (whole periods), bias 0.
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

    /// Gate 1's plants (rows 0 to 7): q's NaN, +Inf and both infinities; a window key's NaN; a window value's +Inf and
    /// NaN; -Inf in the value of row 0's block next to the window; a block's bias -Inf (row 5) and NaN (row 6); and a
    /// row whose every score is -Inf (NaN in float64): q's -Inf against every read key's positive k. Keys no row reads
    /// hold NaN throughout, so a read of one shows.
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

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    towers: std.ArrayList(*TProbe) = .empty,
    decoders: std.ArrayList(*DProbe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (tower_entries) |e| {
            if (only) |o| if (std.mem.indexOf(u8, towerName(e[0], e[1]), o) == null) continue;
            const p = try s.a.create(TProbe);
            errdefer s.a.destroy(p);
            p.* = try TProbe.init(s.a, s.device, s.queue, e[0], e[1]);
            try s.towers.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        }
        for (std.enums.values(DCase)) |k| {
            if (only) |o| if (std.mem.indexOf(u8, decoderName(k), o) == null) continue;
            const p = try s.a.create(DProbe);
            errdefer s.a.destroy(p);
            p.* = try DProbe.init(s.a, s.device, s.queue, k);
            try s.decoders.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.towers.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.towers.deinit(s.a);
        for (s.decoders.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.decoders.deinit(s.a);
    }
};

/// Gate 5's texts: the tower's library for uv_attention, the decoder's for ud_attend.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?uocr.Texts {
    if (!std.mem.startsWith(u8, name, "uocr attention ")) return null;
    const which: uocr.Library = if (std.mem.endsWith(u8, name, " uv_attention")) .vision else if (std.mem.endsWith(u8, name, " ud_attend")) .decoder else return error.UnknownLaunch;
    return .{ .here = uocr.source(which), .pre_m5 = uocr.source(which) };
}
