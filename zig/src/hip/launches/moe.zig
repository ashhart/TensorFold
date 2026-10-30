//! MoE launches: router, select, plan, activation and combine.

const std = @import("std");
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

/// Decode router logits, row picks and (with `items`) plan in one launch; false where the shape keeps them apart.
pub fn routerSelect(l: *const Launcher, x: C, kind: c_int, rows16: C, logits: F, pick: I, wts: F, items: I, members: I, capacity: c_int, r: c_int, d: c_int, e: c_int, top_k: c_int, tile: c_int, done: u64, s: S) Error!bool {
    if (!l.fuse or done == 0 or r < 1 or r > 16 or @rem(d, 4) != 0 or ad(rows16) % 8 != 0 or ad(x) % 8 != 0) return false;
    if (e - 1 > 1024 or top_k < 1 or top_k > 31) return false;
    // the plan in the launch: one row, or 2 to 8 rows of at most plan_pairs pairs (route_plan's pair a thread)
    if (items != null and r != 1 and (r > 8 or r * (top_k + 1) > plan_pairs or tile < 1)) return false;
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(rows16));
    try a.add(ad(logits));
    try a.add(r);
    try a.add(d);
    try a.add(e);
    try a.add(ad(pick));
    try a.add(ad(wts));
    try a.add(ad(items));
    try a.add(ad(members));
    try a.add(capacity);
    try a.add(top_k);
    try a.add(tile);
    const sh = routerShared(ad(x), r, d, e, items != null);
    try a.add(@as(c_int, @intFromBool(sh.staged)));
    try a.add(done);
    try l.go(l.dec.router_select[if (kind == 1) 0 else 1], dim(cdiv(e, 4), 1, 1), dim(128, 1, 1), sh.bytes, s, &a);
    return true;
}

/// The most bytes of x tf_router_select stages: 2 rows of 2560 bf16; more rows cost blocks a WGP than they save in L2.
const STAGED = 2 * 2560 * 2;

/// decode/select.hpp's kPlanPairs: the pairs route_plan takes in the router launch, a thread each (32 * kRouterWaves).
const plan_pairs = 128;

/// tf_router_select's dynamic shared memory: max of staged rows and (2 to 8 rows) top lists plus plan pool; 1 row: 0.
fn routerShared(x: usize, r: c_int, d: c_int, e: c_int, plan: bool) struct { bytes: u32, staged: bool } {
    if (r < 2 or r > 8) return .{ .bytes = 0, .staged = false };
    const staged = @rem(d, 8) == 0 and x % 16 == 0 and r * d * 2 <= STAGED;
    const rows: c_int = if (staged) r * d * 2 else 0;
    // the waves' top values and indices (32 each a wave), then route_plan's counts, starts and ends, scans and pairs
    const ints: c_int = if (e - 1 <= 1024) 2 * 32 * 4 + (if (plan) 3 * e + 2 * 128 + plan_pairs else 0) else 0;
    return .{ .bytes = @intCast(@max(rows, ints * 4)), .staged = staged };
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
    if (l.fuse) return l.go(l.dec.select, dim(r, 1, 1), dim(512, 1, 1), 0, s, &a);
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
    // and the block's scan of two ints a thread, past the segments (4-byte aligned)
    const seg_bytes = std.mem.alignForward(usize, segs * ex * 2, 4);
    try l.go(l.op.moe_route, dim(1, 1, 1), dim(256, 1, 1), @intCast(3 * ex * 4 + seg_bytes + 2 * 256 * 4), s, &a);
}
