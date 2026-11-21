//! Inkling 2's attention inputs through the operators the model graph calls (qPrep, relPrep, kvPrep), against float64
//! formulas over whole [rows][heads][dim] arrays at absolute positions (inkling2_math_prep.zig).
//!
//! Every buffer a call must not touch is poisoned with NaN: the rows around the call's views (a stray read goes NaN
//! where float64 is finite) and every cache slot the call does not write. Both are held exact: the outputs end with
//! one flag a poisoned region (0 intact, 1 written), where float64 says 0. The clip kinds run out of place and in
//! place (the graph clips q and rel in place); the in-place and split kinds must equal the out-of-place whole call
//! byte for byte. The kv kinds cover ring 0 (slot = position), a ring that wraps (rows past the ring: the last row of a
//! slot wins) and head dims 64 and 256.
//!
//! Roundings: the clip's result is bf16(v * factor) (fp32 factor), and with tau bf16(bf16(v * factor) * tau). The
//! reference is unrounded float64, so spec.mids = 1 with tau (the rounding before tau's product) and the output's own
//! rounding closes the bound: relative error <= (1 + 2^-8)^2 - 1 for the two bf16 steps plus the fp32 tail (the
//! sum of squares over a lane's 8 elements and 5 butterfly adds, sqrt, add, divide, multiply, tau's log): ops = 8 and
//! depth 16 cover it. The sums are fp32, so gate 4 plants integer heads whose squares sum below 2^24 exactly; the
//! nonlinear tail after the sum is the same fp32 arithmetic in the exact bound.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const math = @import("inkling2_math_prep.zig");
const rig_ = @import("rig.zig");
const c = u.c;
const g = u.g;
const ops = u.ops;

pub const Kind = enum {
    q,
    q_in_place,
    q_tau,
    q_tau_in_place,
    q_tau_split,
    q_tau_far,
    rel,
    rel_in_place,
    rel_tau,
    rel_tau_in_place,
    kv,
    kv_ring,
    kv_split,
    kv_dim64,
    kv_dim256,
    /// Only for the planted NaN-rule control (a plain q prep); not registered on its own.
    q_nan_rule,
};

/// The kinds the suite registers on the production operators.
pub const kinds = [_]Kind{
    .q,            .q_in_place, .q_tau,  .q_tau_in_place,  .q_tau_split, .q_tau_far, .rel,
    .rel_in_place, .rel_tau,    .rel_tau_in_place, .kv,    .kv_ring,     .kv_split,   .kv_dim64,
    .kv_dim256,
};

fn isKv(k: Kind) bool {
    return switch (k) {
        .kv, .kv_ring, .kv_split, .kv_dim64, .kv_dim256 => true,
        else => false,
    };
}
fn isRel(k: Kind) bool {
    return switch (k) {
        .rel, .rel_in_place, .rel_tau, .rel_tau_in_place => true,
        else => false,
    };
}
fn isInPlace(k: Kind) bool {
    return switch (k) {
        .q_in_place, .q_tau_in_place, .q_tau_split, .rel_in_place, .rel_tau_in_place => true,
        else => false,
    };
}
fn isSplit(k: Kind) bool {
    return k == .q_tau_split or k == .kv_split;
}
fn dimOf(k: Kind) usize {
    if (isRel(k)) return ops.rel_dim;
    return switch (k) {
        .kv_dim64 => 64,
        .kv_dim256 => 256,
        else => ops.head_dim,
    };
}
fn ringOf(k: Kind) u32 {
    return switch (k) {
        .kv_ring, .kv_split => 32,
        .kv_dim256 => 24,
        else => 0,
    };
}
fn floorOf(k: Kind) ?u32 {
    return switch (k) {
        .q_tau, .q_tau_in_place, .q_tau_split, .rel_tau, .rel_tau_in_place => 16,
        .q_tau_far => 8192,
        else => null,
    };
}

const alpha: f32 = 0.1;
const kv_eps: f32 = 1e-6;
const qk_clip: f32 = @floatCast(math.qk_threshold);
const rel_clip: f32 = @floatCast(math.rel_threshold);
/// Rows of NaN before and after the call's rows in every buffer a view points into.
const pad_rows = 2;
const one_bits: u16 = 0x3f80;

fn thresholdOf(k: Kind) f32 {
    return if (isRel(k)) rel_clip else qk_clip;
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .q, .q_nan_rule => "metal inkling2 q prep",
        .q_in_place => "metal inkling2 q prep, in place",
        .q_tau => "metal inkling2 q prep with tau",
        .q_tau_in_place => "metal inkling2 q prep with tau, in place",
        .q_tau_split => "metal inkling2 q prep with tau, in place, window split in two calls",
        .q_tau_far => "metal inkling2 q prep with tau, floor 8192",
        .rel => "metal inkling2 rel prep",
        .rel_in_place => "metal inkling2 rel prep, in place",
        .rel_tau => "metal inkling2 rel prep with tau",
        .rel_tau_in_place => "metal inkling2 rel prep with tau, in place",
        .kv => "metal inkling2 kv prep, slot = position",
        .kv_ring => "metal inkling2 kv prep, ring 32 with wrap",
        .kv_split => "metal inkling2 kv prep, ring 32, window split in two calls",
        .kv_dim64 => "metal inkling2 kv prep, head dim 64",
        .kv_dim256 => "metal inkling2 kv prep, head dim 256, ring 24",
    };
}

// Shapes are rows x heads (kv heads) x position of the first row.
const clip_served = [_]c.Shape{
    .{ .rows = 6, .n = 48, .k = 0 },
    .{ .rows = 1, .n = 48, .k = 9 },
    .{ .rows = 17, .n = 8, .k = 31 },
    .{ .rows = 40, .n = 48, .k = 100 },
    .{ .rows = 33, .n = 8, .k = 3 },
    .{ .rows = 16, .n = 4, .k = 0 },
    .{ .rows = 15, .n = 48, .k = 5 },
};
// Floor 16: rows below the floor (tau 1, the clamp) and above it, windows crossing it.
const tau_served = [_]c.Shape{
    .{ .rows = 6, .n = 48, .k = 0 },
    .{ .rows = 17, .n = 8, .k = 10 },
    .{ .rows = 40, .n = 48, .k = 100 },
    .{ .rows = 1, .n = 48, .k = 15 },
    .{ .rows = 33, .n = 8, .k = 3 },
    .{ .rows = 16, .n = 48, .k = 200 },
    .{ .rows = 2, .n = 4, .k = 16 },
};
// Mini's floor 8192, then far past it.
const far_served = [_]c.Shape{
    .{ .rows = 6, .n = 24, .k = 8186 },
    .{ .rows = 17, .n = 24, .k = 8187 },
    .{ .rows = 40, .n = 24, .k = 20000 },
    .{ .rows = 1, .n = 24, .k = 1000000 },
    .{ .rows = 33, .n = 8, .k = 3000000 },
};
const kv_served = [_]c.Shape{
    .{ .rows = 6, .n = 8, .k = 0 },
    .{ .rows = 1, .n = 8, .k = 100 },
    .{ .rows = 17, .n = 8, .k = 31 },
    .{ .rows = 40, .n = 4, .k = 7 },
    .{ .rows = 33, .n = 8, .k = 0 },
    .{ .rows = 16, .n = 1, .k = 64 },
    .{ .rows = 2, .n = 4, .k = 3 },
};
const ring_served = [_]c.Shape{
    .{ .rows = 6, .n = 8, .k = 5 },
    .{ .rows = 40, .n = 8, .k = 5 },
    .{ .rows = 33, .n = 4, .k = 31 },
    .{ .rows = 17, .n = 8, .k = 70 },
    .{ .rows = 32, .n = 8, .k = 32 },
    .{ .rows = 1, .n = 8, .k = 63 },
    .{ .rows = 3, .n = 4, .k = 95 },
    .{ .rows = 31, .n = 8, .k = 1 },
};
// A split window must fit the ring: no slot is written twice.
const split_served = [_]c.Shape{
    .{ .rows = 6, .n = 8, .k = 5 },
    .{ .rows = 32, .n = 8, .k = 32 },
    .{ .rows = 17, .n = 4, .k = 70 },
    .{ .rows = 2, .n = 8, .k = 31 },
    .{ .rows = 31, .n = 8, .k = 1 },
    .{ .rows = 1, .n = 8, .k = 63 },
};
const dim64_served = [_]c.Shape{
    .{ .rows = 6, .n = 8, .k = 0 },
    .{ .rows = 17, .n = 4, .k = 31 },
    .{ .rows = 40, .n = 8, .k = 3 },
};
const dim256_served = [_]c.Shape{
    .{ .rows = 6, .n = 4, .k = 5 },
    .{ .rows = 24, .n = 4, .k = 24 },
    .{ .rows = 17, .n = 2, .k = 70 },
    .{ .rows = 1, .n = 4, .k = 23 },
    .{ .rows = 23, .n = 4, .k = 1 },
};

fn servedOf(k: Kind) []const c.Shape {
    return switch (k) {
        .q, .q_in_place, .rel, .rel_in_place, .q_nan_rule => &clip_served,
        .q_tau, .q_tau_in_place, .q_tau_split, .rel_tau, .rel_tau_in_place => &tau_served,
        .q_tau_far => &far_served,
        .kv => &kv_served,
        .kv_ring => &ring_served,
        .kv_split => &split_served,
        .kv_dim64 => &dim64_served,
        .kv_dim256 => &dim256_served,
    };
}

fn depth(_: c.Shape) usize {
    return 16;
}

fn intact(words: anytype) bool {
    for (words) |w| if (w != u.nan_bits) return false;
    return true;
}

fn fillHeads(rng: std.Random, out: []u16, dim: usize, mags: []const f32, unit: f32) void {
    var at: usize = 0;
    while (at < out.len) : (at += dim) {
        const m = mags[rng.uintLessThan(usize, mags.len)] * unit;
        for (out[at..][0..dim]) |*v| v.* = u.uniform(rng, m);
    }
}

/// Gate 4's heads: integers 100..250 (exact in bf16), whose squares sum below 2^24 exactly in fp32 in any order.
fn fillInts(rng: std.Random, out: []u16) void {
    for (out) |*v| {
        const m: f32 = @floatFromInt(100 + rng.uintLessThan(u32, 151));
        v.* = c.toBf16(if (rng.boolean()) m else -m);
    }
}

// Magnitudes a head's elements are drawn from: norms below, near and far above the clip thresholds.
const q_mags = [_]f32{ 0.1, 0.5, 2, 8, 30 };
const rel_mags = [_]f32{ 0.2, 1, 2, 4, 20 };
const k_mags = [_]f32{ 0.5, 2, 5, 20 };
const v_mags = [_]f32{ 0.01, 1, 40 };

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    o: *const ops.Ops,
    kind: Kind,
    label: []const u8,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    /// The call's rows: x (clip) or k (kv); `vs` the values; `sc` the clip scale [heads][dim].
    xs: []u16 = &.{},
    vs: []u16 = &.{},
    sc: []u16 = &.{},
    xin: rig_.Ref = undefined,
    vin: rig_.Ref = undefined,
    scr: rig_.Ref = undefined,
    /// The buffer the call writes: the clip's destination rows (padded), or the cache regions.
    work: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    alt: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, o: *const ops.Ops, kind: Kind, label: []const u8) Probe {
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .o = o, .kind = kind, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        const rows_tile: usize = if (p.kind == .kv_ring or p.kind == .kv_split) 32 else 16;
        return .{ .ctx = p, .vt = &vt, .spec = .{
            .name = p.label,
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .mids = if (floorOf(p.kind) != null) 1 else 0,
            .out = .bf16,
            .replaces = if (p.kind == .kv_split) "the same rows in one call" else if (isInPlace(p.kind)) "the same rows out of place in one call" else null,
            .served = servedOf(p.kind),
            .tiles = .{ .{ .size = rows_tile }, .{}, .{} },
            // The output rounds the factor's relative error (sqrt halves the sum's) to bf16: a bf16 sum of the
            // squares moves the result by about one bf16 ulp, not 100 times the bound; fp16 overflows (squares >= 1e4).
            .sums = .{ .f16 = 100, .bf16 = 0, .why = "output rounding hides bf16 sum error" },
        } };
    }

    fn dim(p: *const Probe) usize {
        return dimOf(p.kind);
    }
    fn rowElems(p: *const Probe) usize {
        return p.shape.n * dimOf(p.kind);
    }
    fn slots(p: *const Probe) usize {
        const r = ringOf(p.kind);
        return if (r == 0) p.shape.k + p.shape.rows else r;
    }
    fn slotOf(p: *const Probe, row: usize) usize {
        const r = ringOf(p.kind);
        return if (r == 0) p.shape.k + row else (p.shape.k + row) % r;
    }
    /// The last row of the call that writes row's slot: the one whose bytes the slot holds.
    fn lastRow(p: *const Probe, row: usize) usize {
        var last = row;
        for (row + 1..p.shape.rows) |q| {
            if (p.slotOf(q) == p.slotOf(row)) last = q;
        }
        return last;
    }

    pub fn count(p: *const Probe) usize {
        if (isKv(p.kind)) return 2 * p.shape.rows * p.rowElems() + 2 * p.slots() + 3;
        return p.shape.rows * p.rowElems() + 2;
    }

    fn plant(p: *Probe, buf: []u16, row: usize, head: usize, e: usize, bits: u16) void {
        if (row < p.shape.rows) buf[(row * p.shape.n + head % p.shape.n) * p.dim() + e] = bits;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const ring = ringOf(p.kind);
        if (shape.rows == 0 or shape.n == 0) return false;
        if (ring != 0 and p.kind != .kv_ring and shape.rows > ring) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const dm = p.dim();
        const re = p.rowElems();
        const n = shape.rows * re;
        const pad = pad_rows * re;
        const kv = isKv(p.kind);
        const xfull = try a.alloc(u16, n + 2 * pad);
        @memset(xfull, u.nan_bits);
        p.xs = xfull[pad..][0..n];
        var vfull: []u16 = &.{};
        if (kv) {
            vfull = try a.alloc(u16, n + 2 * pad);
            @memset(vfull, u.nan_bits);
            p.vs = vfull[pad..][0..n];
        }
        p.sc = try a.alloc(u16, shape.n * dm);
        const unit: f32 = @sqrt(128.0 / @as(f32, @floatFromInt(dm)));
        if (kv) {
            fillHeads(rng, p.xs, dm, &k_mags, unit);
            fillHeads(rng, p.vs, dm, &v_mags, 1);
        } else {
            fillHeads(rng, p.xs, dm, if (isRel(p.kind)) &rel_mags else &q_mags, 1);
        }
        const powers = [_]f32{ 0.25, 0.5, 1, 2, 4 };
        for (p.sc) |*v| {
            const m = powers[rng.uintLessThan(usize, powers.len)];
            v.* = c.toBf16(if (rng.boolean()) m else -m);
        }
        if (fill == .stress) {
            fillInts(rng, p.xs);
            if (kv) fillInts(rng, p.vs);
            @memset(p.sc, c.toBf16(1));
        }
        if (fill == .nonfinite) {
            // NaN, +inf, -inf, then one head holding both infinities; an infinity clips its head to 0 and itself to NaN.
            if (kv) {
                p.plant(p.xs, 0, 0, 3, u.nan_bits);
                p.plant(p.xs, 1, 1, dm / 2, u.pinf_bits);
                p.plant(p.xs, 3, 0, 7, u.pinf_bits);
                p.plant(p.xs, 3, 0, 9, u.ninf_bits);
                p.plant(p.vs, 2, 0, 5, u.ninf_bits);
                p.plant(p.vs, 1, 0, 0, u.nan_bits);
                p.plant(p.vs, 4, shape.n - 1, dm - 1, u.pinf_bits);
            } else {
                p.plant(p.xs, 0, 0, 3, u.nan_bits);
                p.plant(p.xs, 1, 1, dm / 2, u.pinf_bits);
                p.plant(p.xs, 2, 0, dm - 1, u.ninf_bits);
                p.plant(p.xs, 3, 2, 1, u.pinf_bits);
                p.plant(p.xs, 3, 2, 2, u.ninf_bits);
                if (shape.n > 4) p.sc[(shape.n - 1) * dm + 5] = u.nan_bits;
            }
        }
        p.xin = try p.rig.input(std.mem.sliceAsBytes(xfull));
        if (kv) p.vin = try p.rig.input(std.mem.sliceAsBytes(vfull));
        p.scr = try p.rig.input(std.mem.sliceAsBytes(p.sc));
        const out_bytes = p.count() * 2;
        if (kv) {
            p.work = try p.rig.scratch((2 * p.slots() + 3) * re * 2);
        } else {
            p.work = try p.rig.scratch((n + 2 * pad) * 2);
        }
        p.out = try p.rig.output(out_bytes);
        p.alt = try p.rig.scratch(out_bytes);
        return true;
    }

    fn clipPiece(p: *Probe, e: mtl.ComputeEncoder, in_place: bool, first: usize, rows: usize) !void {
        const row_bytes = p.rowElems() * 2;
        const at = (pad_rows + first) * row_bytes;
        const src = if (in_place) p.work else p.xin;
        const args = ops.ClipArgs{
            .x = .{ .buffer = src.buf, .offset = src.off + at },
            .scale = u.view(p.scr),
            .out = .{ .buffer = p.work.buf, .offset = p.work.off + at },
            .rows = @intCast(rows),
            .heads = @intCast(p.shape.n),
            .dim = @intCast(p.dim()),
            .threshold = thresholdOf(p.kind),
            .tau = if (floorOf(p.kind)) |f| ops.Tau{ .pos0 = @intCast(p.shape.k + first), .alpha = alpha, .floor = f } else null,
        };
        if (isRel(p.kind)) try p.o.relPrep(e, args) else try p.o.qPrep(e, args);
    }

    /// The clip's rows into `dst` (then the two pad flags), run in place or out of place, in one call or two.
    fn clipCompute(p: *Probe, dst: rig_.Ref, in_place: bool, cut: bool) !void {
        const re = p.rowElems();
        const n = p.shape.rows * re;
        const pad = pad_rows * re;
        const work = std.mem.bytesAsSlice(u16, p.rig.bytes(p.work, (n + 2 * pad) * 2));
        @memset(work, u.nan_bits);
        if (in_place) @memcpy(work[pad..][0..n], p.xs);
        const pass = p.rig.begin();
        const m = if (cut and p.shape.rows > 1) p.shape.rows / 2 else p.shape.rows;
        try p.clipPiece(pass.e, in_place, 0, m);
        if (m < p.shape.rows) try p.clipPiece(pass.e, in_place, m, p.shape.rows - m);
        try p.rig.finish(pass);
        const out = std.mem.bytesAsSlice(u16, p.rig.bytes(dst, (n + 2) * 2));
        @memcpy(out[0..n], work[pad..][0..n]);
        out[n] = if (intact(work[0..pad])) 0 else one_bits;
        out[n + 1] = if (intact(work[pad + n ..])) 0 else one_bits;
    }

    fn kvPiece(p: *Probe, e: mtl.ComputeEncoder, first: usize, rows: usize) !void {
        const slot_bytes = p.rowElems() * 2; // a cache slot is [kv_heads][dim], as long as a row
        const at = (pad_rows + first) * slot_bytes;
        try p.o.kvPrep(e, .{
            .k = .{ .buffer = p.xin.buf, .offset = p.xin.off + at },
            .v = .{ .buffer = p.vin.buf, .offset = p.vin.off + at },
            .kcache = .{ .buffer = p.work.buf, .offset = p.work.off + slot_bytes },
            .vcache = .{ .buffer = p.work.buf, .offset = p.work.off + (p.slots() + 2) * slot_bytes },
            .rows = @intCast(rows),
            .kv_heads = @intCast(p.shape.n),
            .dim = @intCast(p.dim()),
            .pos0 = @intCast(p.shape.k + first),
            .ring = ringOf(p.kind),
            .threshold = qk_clip,
            .eps = kv_eps,
        });
    }

    /// The cache regions are [pad][k slots][pad][v slots][pad], one slot each. Output: the key rows and the value rows
    /// each as its slot ended (rows past the ring read the slot's last writer), then a flag a region that is not a
    /// written slot: 1 where its poison changed.
    fn kvCompute(p: *Probe, dst: rig_.Ref, cut: bool) !void {
        const re = p.rowElems();
        const rows = p.shape.rows;
        const sl = p.slots();
        const regions = 2 * sl + 3;
        const cache = std.mem.bytesAsSlice(u16, p.rig.bytes(p.work, regions * re * 2));
        @memset(cache, u.nan_bits);
        const pass = p.rig.begin();
        const m = if (cut and rows > 1) rows / 2 else rows;
        try p.kvPiece(pass.e, 0, m);
        if (m < rows) try p.kvPiece(pass.e, m, rows - m);
        try p.rig.finish(pass);
        const touched = try p.arena.allocator().alloc(bool, sl);
        @memset(touched, false);
        for (0..rows) |r| touched[p.slotOf(r)] = true;
        const out = std.mem.bytesAsSlice(u16, p.rig.bytes(dst, p.count() * 2));
        for (0..rows) |r| {
            const s = p.slotOf(r);
            @memcpy(out[r * re ..][0..re], cache[(1 + s) * re ..][0..re]);
            @memcpy(out[(rows + r) * re ..][0..re], cache[(sl + 2 + s) * re ..][0..re]);
        }
        for (0..regions) |i| {
            const written = if (i >= 1 and i <= sl) touched[i - 1] else if (i >= sl + 2 and i <= 2 * sl + 1) touched[i - sl - 2] else false;
            out[2 * rows * re + i] = if (written or intact(cache[i * re ..][0..re])) 0 else one_bits;
        }
    }

    pub fn run(p: *Probe) !void {
        if (isKv(p.kind)) return p.kvCompute(p.out, isSplit(p.kind));
        try p.clipCompute(p.out, isInPlace(p.kind), isSplit(p.kind));
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        if (isKv(p.kind)) try p.kvCompute(p.alt, false) else try p.clipCompute(p.alt, false, false);
        return p.rig.bytes(p.alt, p.count() * 2);
    }

    fn tauOf(p: *const Probe) ?math.Tau {
        const f = floorOf(p.kind) orelse return null;
        return .{ .alpha = alpha, .floor = @floatFromInt(f), .pos0 = p.shape.k };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const a = p.arena.allocator();
        const re = p.rowElems();
        const rows = p.shape.rows;
        const dm = p.dim();
        const x = try u.decode(a, p.xs);
        if (!isKv(p.kind)) {
            const n = rows * re;
            math.scaledClip(x, try u.decode(a, p.sc), rows, p.shape.n, dm, thresholdOf(p.kind), p.tauOf(), want[0..n], mag[0..n]);
            @memset(want[n..], 0);
            @memset(mag[n..], 0);
            return;
        }
        const v = try u.decode(a, p.vs);
        const rec = try a.alloc(f64, 4 * rows * re);
        const keys = rec[0 .. rows * re];
        const values = rec[rows * re .. 2 * rows * re];
        const key_mag = rec[2 * rows * re .. 3 * rows * re];
        const value_mag = rec[3 * rows * re ..];
        for (0..rows) |r| {
            const at = r * re;
            math.kvRow(x[at..][0..re], v[at..][0..re], p.shape.n, dm, thresholdOf(p.kind), kv_eps, keys[at..][0..re], values[at..][0..re], key_mag[at..][0..re], value_mag[at..][0..re]);
        }
        for (0..rows) |r| {
            const last = p.lastRow(r) * re;
            @memcpy(want[r * re ..][0..re], keys[last..][0..re]);
            @memcpy(mag[r * re ..][0..re], key_mag[last..][0..re]);
            @memcpy(want[(rows + r) * re ..][0..re], values[last..][0..re]);
            @memcpy(mag[(rows + r) * re ..][0..re], value_mag[last..][0..re]);
        }
        @memset(want[2 * rows * re ..], 0);
        @memset(mag[2 * rows * re ..], 0);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = p.arena.allocator();
        const re = p.rowElems();
        const rows = p.shape.rows;
        const dm = p.dim();
        const x = try u.decode(a, p.xs);
        if (!isKv(p.kind)) {
            const n = rows * re;
            math.scaledClipIn(acc, order, x, try u.decode(a, p.sc), rows, p.shape.n, dm, thresholdOf(p.kind), p.tauOf(), out[0..n]);
            @memset(out[n..], 0);
            return;
        }
        const v = try u.decode(a, p.vs);
        const rec = try a.alloc(f64, 2 * rows * re);
        for (0..rows) |r| {
            const at = r * re;
            math.kvRowIn(acc, order, x[at..][0..re], v[at..][0..re], p.shape.n, dm, thresholdOf(p.kind), kv_eps, rec[at..][0..re], rec[rows * re + at ..][0..re]);
        }
        for (0..rows) |r| {
            const last = p.lastRow(r) * re;
            @memcpy(out[r * re ..][0..re], rec[last..][0..re]);
            @memcpy(out[(rows + r) * re ..][0..re], rec[rows * re + last ..][0..re]);
        }
        @memset(out[2 * rows * re ..], 0);
    }
};

/// One planted fault in the operators' source; the gate named by `controlKernel` must go red on it.
pub const Mutation = struct { kind: Kind, label: []const u8, from: []const u8, to: []const u8 };

pub const mutations = [_]Mutation{
    .{ .kind = .q_tau, .label = "control: inkling2 q prep, tau dropped", .from = "r = float(bfloat(r * tau));", .to = "r = float(bfloat(r));" },
    .{ .kind = .q_tau, .label = "control: inkling2 q prep, tau's clamp at 1 dropped", .from = "const float ratio = max(float(a.pos0 + row + 1) / float(a.floor_positions), 1.0f);", .to = "const float ratio = float(a.pos0 + row + 1) / float(a.floor_positions);" },
    .{ .kind = .q_tau_split, .label = "control: inkling2 q prep, tau at the call's first position for every row", .from = "float(a.pos0 + row + 1)", .to = "float(a.pos0 + 1u)" },
    .{ .kind = .q, .label = "control: inkling2 q prep, norm taken before the scale", .from = "ss = fma(v[e], v[e], ss);", .to = "ss = fma(float(x[base + i]), float(x[base + i]), ss);" },
    .{ .kind = .rel, .label = "control: inkling2 rel prep, every head scaled by head 0's scale", .from = "float(scale[ulong(head) * a.dim + i])", .to = "float(scale[i])" },
    .{
        .kind = .q_in_place,
        .label = "control: inkling2 q prep, output written before the input is read again (in-place hazard)",
        .from = "float r = float(bfloat(v[e] * factor));",
        .to = "out[base + i] = bfloat(0.0f); threadgroup_barrier(mem_flags::mem_device); float r = float(bfloat(float(x[base + i]) * float(scale[ulong(head) * a.dim + i]) * factor));",
    },
    .{ .kind = .q_nan_rule, .label = "control: inkling2 clip, a NaN norm dropped by fmin", .from = "return isnan(f) ? f : min(f, 1.0f);", .to = "return fmin(f, 1.0f);" },
    .{ .kind = .kv, .label = "control: inkling2 kv prep, value norm over half the dim", .from = "i2_rsqrt_mean(simd_sum(vss), a.dim, a.eps)", .to = "i2_rsqrt_mean(simd_sum(vss), a.dim / 2u, a.eps)" },
    .{ .kind = .kv_ring, .label = "control: inkling2 kv prep, ring slot one past", .from = "position % a.ring", .to = "(position + 1) % a.ring" },
    .{ .kind = .kv, .label = "control: inkling2 kv prep, heads written in reverse", .from = "(slot * a.kv_heads + head) * a.dim", .to = "(slot * a.kv_heads + (a.kv_heads - 1u - head)) * a.dim" },
    .{ .kind = .kv_split, .label = "control: inkling2 kv prep, position ignores pos0", .from = "const ulong position = ulong(a.pos0) + row;", .to = "const ulong position = row;" },
    .{ .kind = .kv_dim64, .label = "control: inkling2 kv prep, key not clipped", .from = "kcache[dst + i] = bfloat(kv[e] * kfactor);", .to = "kcache[dst + i] = bfloat(kv[e]);" },
};

pub fn controlSource(a: std.mem.Allocator, m: Mutation) ![]u8 {
    return u.mutate(a, sources.inkling2_ops, m.from, m.to);
}

pub fn controlKernel(p: *Probe) g.Kernel {
    var k = p.kernel();
    k.control = if (p.kind == .q_nan_rule) .nonfinite else .shapes;
    return k;
}
