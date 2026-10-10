//! MoE launches: router, select, plan, activation and combine.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const I = util.I;
const CF = util.CF;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const tri = util.tri;
const invalid = util.invalid;

/// The most rows the router's wave-an-expert kernel takes in prefill.
pub const router_small_rows = 32;

/// The router's logits at any row count so prompt rows ignore cuts: 64 x 64 tiles, or a wave an expert, same bits.
pub fn routerTile(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!void {
    return routerWith(l, x, kind, rows, logits, r, d, e, s, r <= router_small_rows);
}

pub fn routerWith(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S, small: bool) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(rows));
    try a.add(ad(logits));
    try a.add(r);
    try a.add(d);
    try a.add(e);
    const k: usize = if (kind == 1) 0 else 1;
    if (small) return l.go(l.op.router_small[k], dim(cdiv(e, 8), cdiv(r, 32), 1), dim(256, 1, 1), 0, s, &a);
    try l.go(l.op.router_tile[k], dim(cdiv(e, 64), cdiv(r, 64), 1), dim(256, 1, 1), 0, s, &a);
}

/// A lane round's router at any row count; false (nothing launched) where the shape keeps the other kernels.
pub fn routerWindow(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!bool {
    if (!l.fuse or r < 1 or @rem(d, 4) != 0 or ad(rows) % 16 != 0 or ad(x) % 8 != 0) return false;
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(rows));
    try a.add(ad(logits));
    try a.add(r);
    try a.add(d);
    try a.add(e);
    try l.go(l.dec.router[if (kind == 1) 0 else 1], dim(cdiv(e, 4), cdiv(r, 16), 1), dim(128, 1, 1), 0, s, &a);
    return true;
}

pub fn tf_moe_router(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!void {
    // a prompt's rows take the 64 x 64 tiles, a round's few rows a wave an expert; each logit's sum is the same
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(rows));
    try a.add(ad(logits));
    try a.add(r);
    try a.add(d);
    try a.add(e);
    const k: usize = if (kind == 1) 0 else 1;
    if (l.fuse and r <= 16 and @rem(d, 4) == 0 and ad(rows) % 16 == 0 and ad(x) % 8 == 0) {
        return l.go(l.dec.router[k], dim(cdiv(e, 4), 1, 1), dim(128, 1, 1), 0, s, &a);
    }
    if (r >= 64 and @rem(d, 32) == 0) return l.go(l.op.router_tile[k], dim(cdiv(e, 64), cdiv(r, 64), 1), dim(256, 1, 1), 0, s, &a);
    try l.go(l.op.router_rows[k], dim(cdiv(e, 8), cdiv(r, 8), 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_moe_select(l: *const Launcher, logits: CF, pick: I, wts: F, items: I, members: I, capacity: c_int, r: c_int, experts: c_int, top_k: c_int, s: S) Error!void {
    if (experts > 1024 or top_k < 1 or top_k > 31) return invalid("moe select");
    var a: Args = .{};
    try a.add(ad(logits));
    try a.add(ad(pick));
    try a.add(ad(wts));
    try a.add(ad(items));
    try a.add(ad(members));
    try a.add(capacity);
    try a.add(experts);
    try a.add(top_k);
    if (l.fuse) return l.go(l.dec.select, dim(r, 1, 1), dim(256, 1, 1), 0, s, &a);
    try l.go(l.moe_select, dim(r, 1, 1), dim(32, 1, 1), 0, s, &a);
}

pub fn tf_moe_act(l: *const Launcher, both: CF, out: P, kind: c_int, pairs: c_int, width: c_int, limit: f32, s: S) Error!void {
    const total = @as(i64, pairs) * width;
    var a: Args = .{};
    try a.add(ad(both));
    try a.add(ad(out));
    try a.add(width);
    try a.add(limit);
    try a.add(total);
    try l.go(l.moe_act[if (kind == 1) 0 else 1], dim(cdiv(total, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_moe_combine(l: *const Launcher, y: CF, wts: CF, out: P, kind: c_int, r: c_int, slots: c_int, d: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(y));
    try a.add(ad(wts));
    try a.add(ad(out));
    try a.add(slots);
    try a.add(d);
    try l.go(l.moe_combine[tri(kind)], dim(cdiv(d, 256), r, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_moe_route(l: *const Launcher, picks: CI, pairs: c_int, experts: c_int, tile: c_int, members: I, items: I, capacity: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(picks));
    try a.add(pairs);
    try a.add(experts);
    try a.add(tile);
    try a.add(ad(members));
    try a.add(ad(items));
    try a.add(capacity);
    // every expert's count, start and end, and each pair segment's 16-bit counts, in dynamic shared memory
    const ex: usize = @intCast(experts);
    const segs: usize = if (ex <= 256) 64 else 16;
    if (ex > 1024) return invalid("moe_route");
    try l.go(l.op.moe_route, dim(1, 1, 1), dim(256, 1, 1), @intCast(3 * ex * 4 + segs * ex * 2), s, &a);
}
