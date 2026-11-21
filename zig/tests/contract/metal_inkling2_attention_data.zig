//! Host data of Inkling 2's attention probes: cases, seam-hot key/value streams, non-finite plants, each cache.
const std = @import("std");
const u = @import("inkling2_util.zig");
const c = u.c;

pub const dim = 128;
pub const d_rel = 16;
/// 128^-0.5, stored in fp32 as the model hands it to the kernel.
pub const scale: f32 = 0.08838834764831845;

/// A layer geometry. ring 0: slot = position. A sliding ring is at least window + rows - 1.
pub const Case = struct { heads: usize, kv_heads: usize, window: usize, ring: usize, extent: usize };

/// Group sizes 1, 2, 3, 4, 5, 6, 7, 8 (heads / kv_heads); the first is the non-finite plants' case.
pub const cases = [_]Case{
    .{ .heads = 8, .kv_heads = 4, .window = 0, .ring = 0, .extent = 1024 },
    .{ .heads = 48, .kv_heads = 8, .window = 1024, .ring = 1536, .extent = 1024 },
    .{ .heads = 8, .kv_heads = 4, .window = 1024, .ring = 1100, .extent = 1024 },
    .{ .heads = 6, .kv_heads = 6, .window = 0, .ring = 0, .extent = 1024 },
    .{ .heads = 6, .kv_heads = 2, .window = 1024, .ring = 1100, .extent = 1024 },
    .{ .heads = 8, .kv_heads = 2, .window = 0, .ring = 0, .extent = 1024 },
    .{ .heads = 16, .kv_heads = 2, .window = 1024, .ring = 1100, .extent = 1024 },
    .{ .heads = 48, .kv_heads = 8, .window = 0, .ring = 0, .extent = 1024 },
    .{ .heads = 5, .kv_heads = 1, .window = 1024, .ring = 1100, .extent = 1024 },
    .{ .heads = 7, .kv_heads = 1, .window = 0, .ring = 0, .extent = 1024 },
    .{ .heads = 8, .kv_heads = 4, .window = 0, .ring = 0, .extent = 256 },
    .{ .heads = 4, .kv_heads = 2, .window = 64, .ring = 100, .extent = 64 },
    .{ .heads = 24, .kv_heads = 8, .window = 1024, .ring = 1152, .extent = 1024 },
    .{ .heads = 24, .kv_heads = 8, .window = 0, .ring = 0, .extent = 1024 },
};

pub const Built = struct {
    cs: Case,
    rows: usize,
    pos0: usize,
    q: []u16,
    rel: []u16,
    sink: []u16,
    gate: []u16,
    /// The logical stream [position][kv_heads][128] of positions 0..pos0+rows.
    k: []u16,
    v: []u16,
    proj: []u16,
    /// The caches of the case's ring: positions the rows cannot see are NaN.
    kc: []u16,
    vc: []u16,
    /// The same window in another ring (ring2 == total + 5 where the case's ring is 0: slot = position either way).
    ring2: usize,
    kc2: []u16,
    vc2: []u16,
};

/// The non-finite plants address rows 0..4 and kv heads 0..3.
pub fn plantable(cs: Case, rows: usize) bool {
    return cs.heads >= 8 and cs.kv_heads >= 4 and cs.heads % cs.kv_heads == 0 and rows >= 5;
}

fn idx(r: usize, h: usize, heads: usize, j: usize, width: usize) usize {
    return (r * heads + h) * width + j;
}

fn kidx(x: usize, kvh: usize, kv: usize, j: usize) usize {
    return (x * kv + kvh) * dim + j;
}

/// The cache a ring holds after positions lo..=last (slot x % ring, or x), other slots NaN, as are those past the last.
pub fn cacheFor(a: std.mem.Allocator, stream: []const u16, kv: usize, ring: usize, lo: usize, last: usize) ![]u16 {
    const slots = if (ring == 0) last + 1 + 8 else ring;
    const out = try a.alloc(u16, slots * kv * dim);
    @memset(out, u.nan_bits);
    var x = lo;
    while (x <= last) : (x += 1) {
        const slot = if (ring == 0) x else x % ring;
        @memcpy(out[slot * kv * dim ..][0 .. kv * dim], stream[x * kv * dim ..][0 .. kv * dim]);
    }
    return out;
}

fn firstKey(window: usize, p: usize) usize {
    return if (window != 0 and p + 1 > window) p + 1 - window else 0;
}

pub fn build(a: std.mem.Allocator, cs: Case, rows: usize, pos0: usize, fill: u.g.Fill, seed: u64) !Built {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const heads = cs.heads;
    const kv = cs.kv_heads;
    const group = heads / kv;
    const total = pos0 + rows;
    const last = total - 1;
    const lo0 = firstKey(cs.window, pos0);

    const k = try a.alloc(u16, total * kv * dim);
    u.fillUniform(rng, k, 3);
    const v = try a.alloc(u16, total * kv * dim);
    u.fillUniform(rng, v, 1);
    const dir = try a.alloc(u16, kv * dim);
    u.fillUniform(rng, dir, 1.5);

    // Hot keys: positions where a kernel's masks and chunks end.
    const to_i = struct {
        fn f(x: usize) i64 {
            return @intCast(x);
        }
    }.f;
    const picks = [_]i64{
        0,             1,         to_i(pos0),                          to_i(last),                      2047,                         2048,                         4095, 4096,
        to_i(lo0) - 1, to_i(lo0), to_i(firstKey(cs.window, last)) - 1, to_i(firstKey(cs.window, last)), to_i(pos0) - to_i(cs.extent), to_i(last) - to_i(cs.extent),
    };
    for (picks) |at| {
        if (at < 0 or at > to_i(last)) continue;
        const x: usize = @intCast(at);
        const kappa: f64 = 0.8 + 1.2 * rng.float(f64);
        for (0..kv) |kvh| for (0..dim) |j| {
            k[kidx(x, kvh, kv, j)] = u.word(kappa * u.val(dir[kvh * dim + j]));
        };
    }

    const q = try a.alloc(u16, rows * heads * dim);
    for (0..rows) |r| for (0..heads) |h| for (0..dim) |j| {
        const noise = u.val(u.uniform(rng, 3));
        q[idx(r, h, heads, j, dim)] = u.word(noise + u.val(dir[(h / group) * dim + j]));
    };
    const rel = try a.alloc(u16, rows * heads * d_rel);
    u.fillDyadic(rng, rel, 8, 8);
    const proj = try a.alloc(u16, d_rel * cs.extent);
    u.fillDyadic(rng, proj, 16, 8);
    const sink = try a.alloc(u16, rows * heads);
    for (sink) |*s| s.* = u.word(rng.float(f64) * 12);
    const gate = try a.alloc(u16, rows * heads * dim);
    u.fillUniform(rng, gate, 3);

    if (fill == .nonfinite) plant(cs, rows, pos0, q, rel, sink, k, v, proj);

    const ring2 = if (cs.ring == 0) total + 5 else cs.ring + 437;
    return .{
        .cs = cs,
        .rows = rows,
        .pos0 = pos0,
        .q = q,
        .rel = rel,
        .sink = sink,
        .gate = gate,
        .k = k,
        .v = v,
        .proj = proj,
        .kc = try cacheFor(a, k, kv, cs.ring, lo0, last),
        .vc = try cacheFor(a, v, kv, cs.ring, lo0, last),
        .ring2 = ring2,
        .kc2 = try cacheFor(a, k, kv, ring2, lo0, last),
        .vc2 = try cacheFor(a, v, kv, ring2, lo0, last),
    };
}

/// NaN, +inf and -inf at known places (see `plantable`): each follows float64's arithmetic on the formula.
fn plant(cs: Case, rows: usize, pos0: usize, q: []u16, rel: []u16, sink: []u16, k: []u16, v: []u16, proj: []u16) void {
    if (!plantable(cs, rows)) return;
    const heads = cs.heads;
    const kv = cs.kv_heads;
    const total = pos0 + rows;
    const one = c.toBf16(1);
    const neg = c.toBf16(-1);
    q[idx(0, 0, heads, 5, dim)] = u.nan_bits;

    const xa = pos0 -| 6;
    const xb = pos0 -| 7;
    const xc = pos0 -| 8;
    k[kidx(xa, 1, kv, 7)] = u.pinf_bits;
    k[kidx(xb, 1, kv, 8)] = u.ninf_bits;
    k[kidx(xc, 1, kv, 4)] = u.pinf_bits;
    k[kidx(xc, 1, kv, 5)] = u.ninf_bits;
    q[idx(1, 2, heads, 7, dim)] = one;
    q[idx(2, 3, heads, 7, dim)] = neg;
    q[idx(2, 3, heads, 8, dim)] = one;
    q[idx(2, 3, heads, 4, dim)] = neg;
    q[idx(2, 3, heads, 5, dim)] = one;

    const xv = pos0 -| 5;
    v[kidx(xv, 0, kv, 3)] = u.pinf_bits;
    v[kidx(xv -| 1, 0, kv, 6)] = u.ninf_bits;
    v[kidx(xv, 2, kv, 7)] = u.pinf_bits;
    v[kidx(xv, 2, kv, 8)] = u.ninf_bits;

    for (0..total) |x| k[kidx(x, 3, kv, 0)] = u.ninf_bits;
    q[idx(2, 6, heads, 0, dim)] = one;
    q[idx(2, 7, heads, 0, dim)] = one;
    sink[2 * heads + 6] = u.ninf_bits;

    sink[1 * heads + 4] = u.nan_bits;
    sink[3 * heads + 1] = u.pinf_bits;
    sink[4 * heads + 0] = u.ninf_bits;

    rel[idx(3, 5, heads, 2, d_rel)] = u.nan_bits;
    proj[3 * cs.extent + 2] = u.pinf_bits;
    proj[4 * cs.extent + 3] = u.ninf_bits;
    for ([_]usize{ 2, 4 }) |r| for (0..heads) |h| {
        rel[idx(r, h, heads, 3, d_rel)] = neg;
        rel[idx(r, h, heads, 4, d_rel)] = one;
    };
}
