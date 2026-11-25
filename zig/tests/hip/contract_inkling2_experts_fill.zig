//! The HIP experts contract's inputs: random rows, planted non-finites, gate 4's stress rows, every decode pair.
const std = @import("std");
const core = @import("core");
const lm = @import("inkling2_loop_model");
const ref = @import("contract_inkling2_experts_ref.zig");
const bf = @import("bf16_round");

const c = core.contract.spec;
const Order = core.contract.rows.Order;
const topk = 8;

fn bits(v: f32) u16 {
    return bf.bits(v);
}

/// Mini's NVFP4 scales are E4M3 codes 0x20-0x48 here (2^-3 to 2^2), amax 16-64: weights near a checkpoint's.
fn randomMatrix(m: anytype, r: std.Random) void {
    r.bytes(m.codes);
    const mx = m.kind == .mx_down;
    for (m.codes) |*v| if (mx and v.* & 0x7f == 0x7f) {
        v.* -= 1; // no NaN code in random rows: gate 1 plants its own
    };
    for (m.scales) |*v| v.* = if (mx) r.intRangeAtMost(u8, 110, 120) else r.intRangeAtMost(u8, 0x20, 0x48);
    for (m.amax) |*v| v.* = 16 + 48 * r.float(f32);
}

fn randomRows(x: []u16, r: std.Random, scale: f32) void {
    for (x, 0..) |*v, i| v.* = bits((r.float(f32) * 2 - 1) * scale * (if (i % 7 == 0) @as(f32, 3) else 1));
}

/// Each token's eight routes: distinct held experts in a random slot order.
fn routeIds(ids: []i32, r: std.Random, first: usize, held: usize) void {
    for (0..ids.len / topk) |t| {
        var pool: [128]i32 = undefined;
        for (pool[0..held], 0..) |*p, i| p.* = @intCast(first + i);
        r.shuffle(i32, pool[0..held]);
        @memcpy(ids[t * topk ..][0..topk], pool[0..topk]);
    }
}

/// Fills m's host inputs for its kind, shape and fill; a decode cell's random fill covers every (code, scale) pair.
pub fn make(m: anytype, r: std.Random) !void {
    const a = m.rig.a;
    const s = m.shape;
    const routes = s.rows * topk;
    const held: usize = if (m.decode) routes else 8;
    m.first = if (m.kind == .mx_down) 128 - held else 0;
    m.ids = try a.alloc(i32, routes);
    routeIds(m.ids, r, m.first, held);
    const nv = m.kind == .nv_down;
    const rows = s.n;
    const per: usize = if (nv) 16 else 32;
    m.codes = try a.alloc(u8, held * rows * (if (nv) s.k / 2 else s.k));
    m.scales = try a.alloc(u8, held * rows * (s.k / per));
    m.amax = try a.alloc(f32, if (nv) held else 0);
    m.x = try a.alloc(u16, routes * s.k);
    randomMatrix(m, r);
    randomRows(m.x, r, 2);
    switch (m.fill) {
        .random => if (m.decode) decodePairs(m, nv),
        .nonfinite => {
            m.x[0] = bits(std.math.nan(f32));
            m.x[s.k + 3] = bits(std.math.inf(f32));
            m.x[2 * s.k + 7] = bits(-std.math.inf(f32));
            m.ids[3 * topk + 2] = 128; // past the experts: a NaN row
            if (nv) m.amax[1] = std.math.inf(f32) else m.codes[0] = 0x7f;
        },
        .stress => stressRows(m, nv),
    }
}

/// One route an expert, each route's x one-hot at column 37 r: row j of route r's expert holds pair r * 2,304 + j.
fn decodePairs(m: anytype, nv: bool) void {
    const s = m.shape;
    const routes = s.rows * topk;
    const per: usize = if (nv) 16 else 32;
    const rb = if (nv) s.k / 2 else s.k;
    const rs = s.k / per;
    @memset(m.codes, 0);
    @memset(m.scales, if (nv) 0x38 else 127); // code 0 is a zero weight at a finite scale
    @memset(m.x, 0);
    const amax = [_]f32{ 2688, 1, 448, 0x1p-20, 3.0e38, 16, 1.0e-30, 2695.75 }; // phi 1, tiny, huge, ...
    for (m.amax, 0..) |*v, e| v.* = amax[e % amax.len];
    for (0..routes) |route| {
        m.ids[route] = @intCast(m.first + route);
        const col = (route * 37) % s.k;
        m.x[route * s.k + col] = bits(1);
        for (0..s.n) |j| {
            const p = route * s.n + j;
            const row = route * s.n + j; // expert `route`, row j
            const code: u8 = if (nv) @intCast(p & 15) else @intCast(p & 255);
            const scale: u8 = if (nv) @intCast((p >> 4) & 255) else @intCast((p >> 8) & 255);
            if (nv) {
                const at = row * rb + col / 2;
                m.codes[at] = if (col % 2 == 0) code else code << 4;
            } else m.codes[row * rb + col] = code;
            m.scales[row * rs + col / per] = scale;
        }
    }
}

/// Every row the same: 128 terms of w(1 + 2^-7), 129 of -w, one +1 alone in the last group (ref.stressColumns).
fn stressRows(m: anytype, nv: bool) void {
    const s = m.shape;
    var cols: [2304]ref.Stress = undefined;
    ref.stressColumns(s.k, cols[0..s.k]);
    const per: usize = if (nv) 16 else 32;
    const rb = if (nv) s.k / 2 else s.k;
    const rs = s.k / per;
    // NVFP4: 1.5 (E2M1 3) * 256 (E4M3 0x78) * phi 256, and the one 0.5 * 2^-7 (0x04) * 256; MXFP8: 1.5 * 2^16, 1 * 2^0
    for (m.amax) |*v| v.* = 256 * 2688; // bf16(2695.75) is 2688
    for (0..m.codes.len / rb) |row| {
        for (0..s.k) |col| {
            const code: u8 = switch (cols[col]) {
                .zero => 0,
                .plus, .minus => if (nv) 3 else 0x3c,
                .one => if (nv) 1 else 0x38,
            };
            if (nv) {
                const at = row * rb + col / 2;
                if (col % 2 == 0) m.codes[at] = code else m.codes[at] |= code << 4;
            } else m.codes[row * rb + col] = code;
        }
        for (0..rs) |gi| m.scales[row * rs + gi] = if (gi == rs - 1) (if (nv) 0x04 else 127) else (if (nv) 0x78 else 143);
    }
    for (0..m.x.len / s.k) |t| for (0..s.k) |col| {
        m.x[t * s.k + col] = switch (cols[col]) {
            .zero => 0,
            .plus => bits(1 + 0x1p-7),
            .minus => bits(-1),
            .one => bits(1),
        };
    };
}

/// Gate 4: one stress output as a simulated accumulator gives it (every row and token are the same).
pub fn simulate(m: anytype, acc: c.Accum, order: Order) f64 {
    const s = m.shape;
    var w: [2304]f64 = undefined;
    var x: [2304]f64 = undefined;
    var terms: [2304]f64 = undefined;
    const mat = m.mat();
    mat.decode(@intCast(m.ids[0]), 0, w[0..s.k]) catch unreachable;
    for (x[0..s.k], m.x[0..s.k]) |*v, h| v.* = bf.value(h);
    return lm.bf16(ref.simDot(acc, order, w[0..s.k], x[0..s.k], terms[0..s.k]));
}
