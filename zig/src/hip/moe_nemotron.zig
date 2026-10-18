//! Nemotron-H's MoE on moe.zig's plan and products: a sigmoid pick with a bias, relu^2 experts, the shared one at 1.

const std = @import("std");
const ops = @import("ops/ops.zig");
const moe = @import("moe.zig");
const launch = @import("launches/nemotron_moe.zig");

const Ops = ops.Ops;
const Tensor = ops.Tensor;
const Kind = ops.Kind;
const Projection = ops.Projection;

/// A layer's router and this rank's experts as the checkpoint stores them.
pub const Layer = struct {
    /// gate.weight in fp32, (experts, D): every routed expert, on every rank.
    router: u64,
    /// e_score_correction_bias in fp32, (experts): it ranks the experts and leaves their weights alone.
    bias: u64,
    /// switch_mlp.fc1 (own, NI, D) and fc2 (own, D, NI): this rank's experts from `first`.
    fc1: Projection,
    fc2: Projection,
    /// shared_experts.up_proj (SW, D) and down_proj (D, SW); null on ranks without the shared expert.
    shared_up: ?Projection,
    shared_down: ?Projection,
    experts: usize,
    first: usize = 0,
    own: usize,
    top_k: usize,
    /// routed_scaling_factor.
    scaling: f32,
    /// norm_topk_prob: the picked scores over their sum before the scaling.
    norm: bool = true,
    /// The output is this rank's fp32 share of a sum over ranks, not the activation.
    share: bool = false,
    /// Plan the rows' own pairs, not a power-of-two bucket's (no graph replays the launches at other row counts).
    exact: bool = false,
    /// One row's relu^2 as the product's epilogue; false keeps it a launch of its own (the epilogue's reference).
    epilogue: bool = true,

    fn whole(l: Layer) bool {
        return l.first == 0 and l.own == l.experts;
    }
};

/// `x` (rows, D) normed rows to (rows, D): x's kind, or fp32 when `l.share` (the caller sums the ranks' shares).
pub fn run(o: Ops, l: Layer, x: Tensor, rows: usize) ops.Error!Tensor {
    const m = try expertRows(o, l, x, rows);
    const d: usize = l.fc2.n;
    const out_kind: Kind = if (l.share) .f32 else x.kind;
    const out = try o.arena.take(rows * d * out_kind.size());
    try launch.combine(
        o.l,
        @ptrFromInt(m.y),
        @ptrFromInt(m.wts),
        @ptrFromInt(m.shared),
        @ptrFromInt(out),
        @backingInt(out_kind),
        @intCast(rows),
        @intCast(m.slots),
        @intCast(d),
        o.stream,
    );
    return .{ .ptr = out, .kind = out_kind };
}

/// `run` up to its sum: each row's experts' fp32 rows and weights, and the shared expert's fp32 row (0 without one).
pub fn expertRows(o: Ops, l: Layer, x: Tensor, rows: usize) ops.Error!Ops.MoeRows {
    const kind = x.kind;
    const d: usize = l.fc2.n;
    const ni: usize = l.fc1.n;
    if (rows == 0 or kind == .f32 or l.own == 0 or l.fc1.k != d or l.fc2.k != ni or
        (l.shared_up == null) != (l.shared_down == null)) return error.BadShape;
    const slots = l.top_k;
    const pairs = rows * slots;
    const size = if (l.exact) rows else moe.bucket(rows);
    const plan_experts = if (l.whole()) l.own else l.own + 1;
    const cap = moe.capacity(size * slots, plan_experts);
    const logits = try o.arena.of(f32, rows * l.experts);
    const pick = try o.arena.of(i32, size * slots);
    const wts = try o.arena.of(f32, size * slots);
    const items = try o.arena.of(i32, cap * 3);
    const members = try o.arena.of(i32, size * slots);
    try o.moeRouter(x, l.router, logits, rows, d, l.experts);
    var count = cap;
    var tile: usize = 1;
    var local = pick;
    if (rows == 1 and l.whole()) {
        try select(o, l, logits, pick, wts, .{ .items = items, .members = members }, cap, 1);
        count = slots;
    } else {
        try select(o, l, logits, pick, wts, null, 0, rows);
        tile = moe.tileFor(o, rows);
        if (!l.whole()) {
            // every rank picks the same pairs and runs its own; the others group under an expert past its own
            local = try o.arena.of(i32, size * slots);
            try launch.localize(
                o.l,
                @ptrFromInt(pick),
                @ptrFromInt(local),
                @intCast(pairs),
                @intCast(l.first),
                @intCast(l.own),
                o.stream,
            );
        }
        try o.moeRoute(local, pairs, plan_experts, tile, members, items, cap);
        if (!l.whole()) try o.moeForeignItems(items, cap, l.own);
    }
    const act: Tensor = .{ .ptr = try o.arena.take(pairs * ni * kind.size()), .kind = kind };
    // one row's relu^2 as the product's epilogue where the stream tile takes the plan (more rows run faster apart)
    const epilogue = rows == 1 and l.epilogue;
    if (!epilogue or !try o.projectRoutedRelu2(x, l.fc1, items, count, members, slots, @min(tile, rows), act)) {
        const up = try o.projectRouted(x, l.fc1, items, count, members, pairs, slots, @min(tile, rows));
        try relu2(o, up, act, pairs * ni);
    }
    const y = try o.projectRouted(act, l.fc2, items, count, members, pairs, 1, @min(tile, rows));
    if (!l.whole()) try o.moeZeroForeign(y, local, l.own, pairs, d);
    const shared: u64 = if (l.shared_up) |su| try sharedExpert(o, su, l.shared_down.?, x, rows, epilogue) else 0;
    return .{ .y = y, .wts = wts, .shared = shared, .slots = slots };
}

/// down(relu^2(up x)) in fp32 (rows, D), the up rounded as the routed experts' is.
fn sharedExpert(o: Ops, up: Projection, down: Projection, x: Tensor, rows: usize, epilogue: bool) ops.Error!u64 {
    if (up.k != down.n or down.k != up.n) return error.BadShape;
    const act: Tensor = .{ .ptr = try o.arena.take(rows * up.n * x.kind.size()), .kind = x.kind };
    if (!epilogue or !try o.projectRelu2(x, up, rows, act)) {
        const u = try o.project(x, up, rows, true);
        try relu2(o, u.ptr, act, rows * up.n);
    }
    return (try o.project(act, down, rows, true)).ptr;
}

/// The pick for `rows` rows of logits (rows, experts); with `plan` (one row) it also writes its items and members.
pub fn select(
    o: Ops,
    l: Layer,
    logits: u64,
    pick: u64,
    wts: u64,
    plan: ?struct { items: u64, members: u64 },
    capacity: usize,
    rows: usize,
) ops.Error!void {
    const items: u64 = if (plan) |pl| pl.items else 0;
    const members: u64 = if (plan) |pl| pl.members else 0;
    try launch.select(
        o.l,
        @ptrFromInt(logits),
        @ptrFromInt(l.bias),
        @ptrFromInt(pick),
        @ptrFromInt(wts),
        @ptrFromInt(items),
        @ptrFromInt(members),
        @intCast(capacity),
        @intCast(rows),
        @intCast(l.experts),
        @intCast(l.top_k),
        l.scaling,
        l.norm,
        o.stream,
    );
}

/// out = relu(up rounded to out's kind)^2: `n` fp32 values of an expert's up.
pub fn relu2(o: Ops, up: u64, out: Tensor, n: usize) ops.Error!void {
    try launch.relu2(o.l, @ptrFromInt(up), @ptrFromInt(out.ptr), @backingInt(out.kind), @intCast(n), o.stream);
}

test "a rank holds the whole layer only with every expert from the first" {
    const p: Projection = .{ .n = 1, .k = 1, .handle = .{ .mlx = undefined } };
    const l: Layer = .{
        .router = 0,
        .bias = 0,
        .fc1 = p,
        .fc2 = p,
        .shared_up = null,
        .shared_down = null,
        .experts = 128,
        .own = 128,
        .top_k = 6,
        .scaling = 2.5,
    };
    try std.testing.expect(l.whole());
    var r = l;
    r.first = 64;
    r.own = 64;
    try std.testing.expect(!r.whole());
}
