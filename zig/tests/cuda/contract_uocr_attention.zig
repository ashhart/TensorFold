//! Unlimited-OCR-2's CUDA attention under the contract: the tower's windowed and global attention, the decoder's DASH.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const uocr = @import("contract_uocr.zig");
const dec = @import("contract_uocr_attention_decoder.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const ar = core.contract.attention_range;

const bf = dec.bf;
const period = dec.period;
const fillRandom = dec.fillRandom;
const stressValue = dec.stressValue;
const stressBase = dec.stressBase;

// --- tf_uocr_attention: a warp a (row, head), softmax(q k^T / 8 + bf16(rel_h + rel_w)) v over `len` rows ---

const Tower = enum { sam_window, sam_global, clip };
/// The engine's kernel, and two planted controls: rel_w read one column over (SAM windows), P x V without its last key.
const TCase = enum { plain, bias_shift, last_key };

const tower_entries = [_]struct { Tower, TCase }{ .{ .sam_window, .plain }, .{ .sam_global, .plain }, .{ .clip, .plain }, .{ .sam_window, .bias_shift }, .{ .clip, .last_key } };

fn towerName(t: Tower, k: TCase) []const u8 {
    return switch (k) {
        .plain => switch (t) {
            .sam_window => "cuda uocr attention tf_uocr_attention sam 14x14 windows, 12 heads of 64, relative-position bias",
            .sam_global => "cuda uocr attention tf_uocr_attention sam global, 12 heads of 64, relative-position bias",
            .clip => "cuda uocr attention tf_uocr_attention clip, 16 heads of 64",
        },
        .bias_shift => "control: cuda uocr attention tf_uocr_attention sam windows, rel_w read one column over",
        .last_key => "control: cuda uocr attention tf_uocr_attention clip, P x V without the last key",
    };
}

fn towerControl(k: TCase) ?[:0]const u8 {
    return switch (k) {
        .plain => null,
        .bias_shift => "tf_ctl_uocr_attention_bias_shift",
        .last_key => "tf_ctl_uocr_attention_last_key",
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
        .ops = dec.div_roundings,
        .out = .bf16,
        // CLIP's keys at 64-key tile edges (any group length runs; SAM's are squares of its side)
        .tiles = if (t == .clip) .{ .{}, .{}, .{ .size = 64, .min = 1 } } else .{ .{}, .{}, .{} },
        // rows: query rows from row 0 (groups of k rows); n: the width; k: the keys a group (len)
        .served = switch (t) {
            // a 1024 view's 25 windows (64 padded to 70), a 640 tile's 9 (40 padded to 42)
            .sam_window => &.{ .{ .rows = 25 * 196, .n = uocr.sam_width, .k = 196 }, .{ .rows = 9 * 196, .n = uocr.sam_width, .k = 196 } },
            // a 1024 view's 64 x 64 and a 640 tile's 40 x 40: each one's first query rows
            .sam_global => &.{ .{ .rows = 64, .n = uocr.sam_width, .k = 4096 }, .{ .rows = 200, .n = uocr.sam_width, .k = 1600 } },
            // a 1024 view's 257 rows and a 640 tile's 101
            .clip => &.{ .{ .rows = 257, .n = uocr.clip_width, .k = 257 }, .{ .rows = 101, .n = uocr.clip_width, .k = 101 } },
        },
        .sums = .{ .exact = true },
    };
}

/// A lane's P x V chain over every key in order; the bounds hook replaces the generic bound.
fn towerDepth(s: c.Shape) usize {
    return s.k + 1;
}

/// tf_uocr_attention over n keys: a lane's 64 products in order, * scale + bias, expf, bf16 P, lane chain, butterfly.
fn towerForm(n: usize) ar.Form {
    return .{
        .score = 64 + 1,
        .score_round = 2,
        .fast_exp = false,
        .exp_ulps = dec.exp_ulps,
        .chunk_exp = false,
        .p_rel = 0x1p-8,
        .num = @floatFromInt(n + 1),
        .den = @floatFromInt(n / 32 + 7),
        .div = dec.div_roundings,
    };
}

const tower_scale: f32 = 0.125; // 1 / sqrt(64), as cuda_vision.zig computes it

const TProbe = struct {
    rig: rig_.Rig,
    k: uocr.kernels.Kernels,
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
    refs: [3]u64 = @splat(0),
    out: u64 = 0,

    fn alloc(p: *const TProbe) std.mem.Allocator {
        return p.rig.gpu.gpa;
    }

    fn free(p: *TProbe) void {
        const a = p.alloc();
        for ([_][]u16{ p.qkv, p.rh, p.rw }) |x| a.free(x);
        for ([_][]f64{ p.point, p.half }) |x| a.free(x);
        p.qkv = &.{};
        p.rh = &.{};
        p.rw = &.{};
        p.point = &.{};
        p.half = &.{};
    }

    fn deinit(p: *TProbe) void {
        p.free();
        p.rig.deinit();
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
        const a = p.alloc();
        p.rows = shape.rows;
        p.len = shape.k;
        p.groups = (shape.rows + shape.k - 1) / shape.k;
        p.side = side;
        p.width = width;
        p.heads = width / 64;
        p.fill = fill;
        const total = p.groups * p.len;
        const stride = 3 * width;
        p.qkv = try a.alloc(u16, total * stride);
        const rel = @max(1, p.heads * p.groups * p.len * side);
        p.rh = try a.alloc(u16, rel);
        p.rw = try a.alloc(u16, rel);
        p.point = try a.alloc(f64, p.count());
        p.half = try a.alloc(f64, p.count());
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
                const base = try stressBase(a, rng, width);
                defer a.free(base);
                for (0..total) |r| {
                    const j = r % p.len;
                    for (0..width) |col| p.qkv[r * stride + 2 * width + col] = c.toBf16(if (j < p.len / period * period) stressValue(base, width, j, col) else 0);
                }
            },
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.qkv));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.rh));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.rw));
        // the engine's launch writes every row of every group; the gates read the first `rows`
        p.out = try p.rig.output(total * width * 2);
        return true;
    }

    /// Gate 1's plants in group 0: NaN/Inf in q, k, v and SAM's bias, plus two rows whose every score is -Inf.
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

    /// As cuda_vision.zig's attention(): 8 query rows of a group a block, a block per head.
    pub fn run(p: *TProbe) !void {
        try p.k.run("attention", p.rig.s, .{ p.groups * ((p.len + 7) / 8), p.heads, 1 }, 256, .{ p.refs[0], p.refs[1], p.refs[2], p.out, p.groups, p.len, p.width, p.side, tower_scale });
        try p.rig.s.synchronize();
    }

    pub fn count(p: *const TProbe) usize {
        return p.rows * p.width;
    }

    /// Each row's heads over its group's keys, bias bf16(rel_h + rel_w) as a 65th dot term: float64 points and ranges.
    pub fn reference(p: *TProbe, want: []f64, mag: []f64) !void {
        const a = p.alloc();
        const L = p.len;
        const dk = 65;
        const buf = try a.alloc(f64, 2 * L * dk + 2 * L + 3 * dk);
        defer a.free(buf);
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
                        const b = bf(@as(f64, c.fromBf16(p.rh[rel + j / p.side])) + c.fromBf16(p.rw[rel + j % p.side]));
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
        const a = p.alloc();
        const terms = try a.alloc(f64, p.len);
        defer a.free(terms);
        const stride = 3 * p.width;
        for (0..p.rows) |r| for (0..p.width) |col| {
            const grp = r / p.len;
            for (terms, 0..) |*x, j| x.* = c.fromBf16(p.qkv[(grp * p.len + j) * stride + 2 * p.width + col]);
            out[r * p.width + col] = bf(core.contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(p.len)));
        };
    }
};

/// The group's registrations, on the heap: the gates hold pointers to them.
pub const Group = struct {
    gpa: std.mem.Allocator,
    towers: [tower_entries.len]TProbe = undefined,
    decoders: [std.enums.values(dec.DCase).len]dec.DProbe = undefined,

    pub fn init(ctx: uocr.Ctx, list: *std.ArrayList(g.Kernel)) !*Group {
        const grp = try ctx.gpu.gpa.create(Group);
        errdefer ctx.gpu.gpa.destroy(grp);
        grp.* = .{ .gpa = ctx.gpu.gpa };
        for (&grp.towers, tower_entries) |*p, e| {
            const k = if (towerControl(e[1])) |ctl| try ctx.with("attention", ctl) else ctx.k.*;
            p.* = .{ .rig = ctx.rig(), .k = k, .tower = e[0], .case = e[1] };
            try list.append(ctx.gpu.gpa, p.kernel());
        }
        for (&grp.decoders, std.enums.values(dec.DCase)) |*p, kc| {
            const k = if (dec.control(kc)) |ctl| try ctx.with("attend", ctl) else ctx.k.*;
            p.* = .{ .rig = ctx.rig(), .k = k, .case = kc };
            try list.append(ctx.gpu.gpa, p.kernel());
        }
        return grp;
    }

    pub fn deinit(grp: *Group) void {
        for (&grp.towers) |*p| p.deinit();
        for (&grp.decoders) |*p| p.deinit();
        grp.gpa.destroy(grp);
    }
};
