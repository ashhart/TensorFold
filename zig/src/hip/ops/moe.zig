//! The mixture-of-experts launches: the router, the pick, the plan, the activation and the combine.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const i = t.i;
const int = t.int;

/// MoE combine, residual and next norm in one launch: x = round(x + round(sum_s wts * y)), normed = rms(x) * weight.
pub fn moeTail(o: Ops, x: Tensor, y: u64, wts: u64, weight: u64, normed: Tensor, rows: usize, slots: usize, width: usize, eps: f32) Error!void {
    if (x.kind != normed.kind or x.kind == .f32 or slots == 0) return error.BadShape;
    const z = o.l;
    try z.tf_tail(p(x.ptr), f(y), f(wts), f(weight), p(normed.ptr), @backingInt(x.kind), int(rows), int(slots), int(width), eps, o.stream);
}

pub fn moeRouter(o: Ops, x: Tensor, rows32: u64, logits: u64, r: usize, d: usize, e: usize) Error!void {
    if (o.window) if (o.l.routerWindow(p(x.ptr), @backingInt(x.kind), f(rows32), f(logits), int(r), int(d), int(e), o.stream)) |ran| {
        if (ran) return;
    } else |err| return err;
    if (o.prefill and d % 32 == 0) return o.l.routerTile(p(x.ptr), @backingInt(x.kind), f(rows32), f(logits), int(r), int(d), int(e), o.stream);
    try o.l.tf_moe_router(p(x.ptr), @backingInt(x.kind), f(rows32), f(logits), int(r), int(d), int(e), o.stream);
}

/// The pick rule per row; with `plan` (one row) it also writes the plan's items and members.
pub fn moeSelect(o: Ops, logits: u64, pick: u64, wts: u64, plan: ?struct { items: u64, members: u64 }, capacity: usize, r: usize, experts: usize, top_k: usize) Error!void {
    try o.l.tf_moe_select(f(logits), i(pick), f(wts), if (plan) |pl| i(pl.items) else null, if (plan) |pl| i(pl.members) else null, int(capacity), int(r), int(experts), int(top_k), o.stream);
}

pub fn moeRoute(o: Ops, picks: u64, pairs: usize, experts: usize, tile: usize, members: u64, items: u64, capacity: usize) Error!void {
    try o.l.tf_moe_route(@ptrFromInt(picks), int(pairs), int(experts), int(tile), i(members), i(items), int(capacity), o.stream);
}

pub fn moeAct(o: Ops, both: u64, out: Tensor, pairs: usize, width: usize, limit: f32) Error!void {
    try o.l.tf_moe_act(f(both), p(out.ptr), @backingInt(out.kind), int(pairs), int(width), limit, o.stream);
}

pub fn moeCombine(o: Ops, y: u64, wts: u64, out: Tensor, r: usize, slots: usize, d: usize) Error!void {
    try o.l.tf_moe_combine(f(y), f(wts), p(out.ptr), @backingInt(out.kind), int(r), int(slots), int(d), o.stream);
}
