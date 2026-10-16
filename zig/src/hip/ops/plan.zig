//! A lane round's launches over its device plan: arguments depend only on the plan's shape and the layer.

const t = @import("types.zig");
const launches = @import("../launches.zig");
const Ops = @import("ops.zig").Ops;

const Tensor = t.Tensor;
const Error = t.Error;

pub const Args = launches.PlanArgs;
pub const Plan = launches.PlanRef;
pub const Keep = launches.PlanKeep;

fn launcher(o: Ops) Error!*const launches.Launcher {
    return o.l;
}

fn int(n: usize) c_int {
    return @intCast(n);
}

/// The conv's norm weights of q and k heads (heads of 128) and the delta rule's gate and beta computed beside it.
pub const Norm = struct { q: u64, k: u64, eps: f32 };
pub const Gates = struct { a: Tensor, b: Tensor, a_log: u64, dt_bias: u64, gate: u64, beta: u64, count: usize, heads: usize };

/// The round's rows through the linear attention's conv from each slot's window, split into q, k, v fp32.
pub fn convSplit(o: Ops, x: Tensor, weight: u64, qn: u64, kn: u64, v: u64, p: Plan, layer: usize, channels: usize, kernel: usize, kw: usize, vw: usize, norm: ?Norm, gates: ?Gates) Error!void {
    if (kernel < 1 or kernel > 8 or x.kind == .f32) return error.BadShape;
    const z = try launcher(o);
    var c: launches.ConvArgs = .{
        .x = x.ptr,
        .kind = @backingInt(x.kind),
        .weight = weight,
        .state = 0,
        .qn = qn,
        .kn = kn,
        .v = v,
        .channels = int(channels),
        .kernel = int(kernel),
        .rows = int(p.rows),
        .kw = int(kw),
        .vw = int(vw),
        .plan = p.args,
        .layer = int(layer),
        .slots = int(p.slots),
    };
    if (norm) |nm| {
        c.qw = nm.q;
        c.kw_w = nm.k;
        c.eps = nm.eps;
        c.norm = 128;
    }
    if (gates) |g| {
        if (g.a.kind != x.kind or g.b.kind != x.kind) return error.BadShape;
        c.ga = g.a.ptr;
        c.gb = g.b.ptr;
        c.a_log = g.a_log;
        c.dt_bias = g.dt_bias;
        c.gate = g.gate;
        c.beta = g.beta;
        c.gcount = int(g.count);
        c.heads = int(g.heads);
    }
    try z.tf_conv_split(c, o.stream);
}

/// The DeltaNet recurrence of every slot's rows: q, k (rows, Hk, dk), v, y (rows, Hv, dv), gate, beta (rows, Hv) fp32.
pub fn gatedDelta(o: Ops, q: u64, k: u64, v: u64, gate: u64, beta: u64, y: u64, p: Plan, layer: usize, key_heads: usize, value_heads: usize, dk: usize, dv: usize) Error!void {
    try (try launcher(o)).planGatedDelta(q, k, v, gate, beta, y, p, layer, key_heads, value_heads, dk, dv, o.stream);
}

/// Each row's keys (fp32, rotated) and values (the activation kind) into its slot's caches at its position.
pub fn kvWrite(o: Ops, keys: u64, values: Tensor, p: Plan, layer: usize, kv_heads: usize, d: usize) Error!void {
    if (values.kind == .f32) return error.BadShape;
    try (try launcher(o)).planKvWrite(keys, values.ptr, @backingInt(values.kind), p, layer, kv_heads, d, o.stream);
}

/// One query a row (rows, heads, d) fp32 over its slot's caches up to its position, the walk over `span` keys.
pub fn causal(o: Ops, q: u64, kind: t.Kind, out: u64, p: Plan, layer: usize, heads: usize, kv_heads: usize, d: usize, span: usize, scale: f32) Error!void {
    const tiles = (span + 127) / 128;
    const scores = try o.arena.of(f32, p.rows * heads * span);
    const stats = try o.arena.of(f32, p.rows * heads * 2);
    const partials = try o.arena.of(f32, p.rows * heads * tiles * d);
    try (try launcher(o)).planCausal(q, out, scores, stats, partials, p, layer, heads, kv_heads, d, span, scale, @backingInt(kind), o.stream);
}

/// Keeps the listed slots' rows (see `launches.PlanKeep`).
pub fn keep(o: Ops, p: Plan, k: Keep) Error!void {
    try (try launcher(o)).planKeep(p, k, o.stream);
}

/// The kept slots' DeltaNet states replayed through their kept rows (see `launches.planGdnReplay`).
pub fn gdnReplay(o: Ops, keep_counts: u64, p: Plan, key_heads: usize, value_heads: usize, dk: usize, dv: usize, layers: usize) Error!void {
    try (try launcher(o)).planGdnReplay(keep_counts, p, key_heads, value_heads, dk, dv, layers, o.stream);
}

/// Row r of `dst` (`words` words each) from the address in `srcs[r]` (device u64s), for `rows` rows.
pub fn gather(o: Ops, srcs: u64, dst: u64, words: usize, rows: usize) Error!void {
    try (try launcher(o)).planGather(srcs, dst, words, rows, o.stream);
}
