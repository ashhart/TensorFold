//! Nemotron-H's MoE launches: the sigmoid pick with its correction bias, relu^2, a rank's local ids and the sum.

const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const F = util.F;
const CF = util.CF;
const I = util.I;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

/// nemotron_moe.hpp's kernels in the ops group.
pub const Fns = struct {
    select: Function,
    select_pass: Function,
    relu2: Function,
    localize: Function,
    combine: Function,

    pub fn load(ops: Module) Error!Fns {
        return .{
            .select = try ops.function("tf_nemo_select_rank"),
            .select_pass = try ops.function("tf_nemo_select"),
            .relu2 = try ops.function("tf_nemo_relu2"),
            .localize = try ops.function("tf_nemo_localize"),
            .combine = try ops.function("tf_nemo_combine"),
        };
    }
};

/// Each row's top_k by sigmoid score plus bias, weights scaled; `items` and `members` plan one row.
pub fn select(
    l: *const Launcher,
    logits: CF,
    bias: CF,
    pick: I,
    wts: F,
    items: I,
    members: I,
    capacity: c_int,
    rows: c_int,
    experts: c_int,
    top_k: c_int,
    scaling: f32,
    norm: bool,
    s: S,
) Error!void {
    try pickWith(
        l,
        l.nemo.select,
        512,
        logits,
        bias,
        pick,
        wts,
        items,
        members,
        capacity,
        rows,
        experts,
        top_k,
        scaling,
        norm,
        s,
    );
}

/// select by one wave taking a pick at a time: the same picks and weights, kept as the rank pick's reference.
pub fn selectPass(
    l: *const Launcher,
    logits: CF,
    bias: CF,
    pick: I,
    wts: F,
    items: I,
    members: I,
    capacity: c_int,
    rows: c_int,
    experts: c_int,
    top_k: c_int,
    scaling: f32,
    norm: bool,
    s: S,
) Error!void {
    try pickWith(
        l,
        l.nemo.select_pass,
        32,
        logits,
        bias,
        pick,
        wts,
        items,
        members,
        capacity,
        rows,
        experts,
        top_k,
        scaling,
        norm,
        s,
    );
}

fn pickWith(
    l: *const Launcher,
    f: Function,
    threads: u32,
    logits: CF,
    bias: CF,
    pick: I,
    wts: F,
    items: I,
    members: I,
    capacity: c_int,
    rows: c_int,
    experts: c_int,
    top_k: c_int,
    scaling: f32,
    norm: bool,
    s: S,
) Error!void {
    if (experts < 1 or experts > 1024 or top_k < 1 or top_k > experts or top_k > 32) return invalid("nemotron select");
    if (rows == 0) return;
    var a: Args = .{};
    for ([_]u64{ ad(logits), ad(bias), ad(pick), ad(wts), ad(items), ad(members) }) |v| try a.add(v);
    for ([_]c_int{ capacity, experts, top_k }) |v| try a.add(v);
    try a.add(scaling);
    try a.add(@as(c_int, @intFromBool(norm)));
    try l.go(f, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// out = relu(up rounded to `kind`)^2 in `kind` (1 fp16, 2 bf16), `n` values of fp32 `up`.
pub fn relu2(l: *const Launcher, up: CF, out: P, kind: c_int, n: i64, s: S) Error!void {
    if (kind != 1 and kind != 2) return invalid("nemotron relu2");
    if (n == 0) return;
    var a: Args = .{};
    try a.add(ad(up));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(n);
    try l.go(l.nemo.relu2, dim(cdiv(n, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// out[i] = pick[i] - first for the `own` experts from `first`, else `own`.
pub fn localize(l: *const Launcher, pick: CI, out: I, n: c_int, first: c_int, own: c_int, s: S) Error!void {
    if (n == 0) return;
    var a: Args = .{};
    try a.add(ad(pick));
    try a.add(ad(out));
    for ([_]c_int{ n, first, own }) |v| try a.add(v);
    try l.go(l.nemo.localize, dim(cdiv(n, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// out (rows, d) of `kind` (0 fp32, 1 fp16, 2 bf16) = the slots' weighted sum in order, plus `shared` when not null.
pub fn combine(
    l: *const Launcher,
    y: CF,
    wts: CF,
    shared: CF,
    out: P,
    kind: c_int,
    rows: c_int,
    slots: c_int,
    d: c_int,
    s: S,
) Error!void {
    if (kind < 0 or kind > 2) return invalid("nemotron combine");
    if (rows == 0) return;
    var a: Args = .{};
    for ([_]u64{ ad(y), ad(wts), ad(shared), ad(out) }) |v| try a.add(v);
    for ([_]c_int{ kind, slots, d }) |v| try a.add(v);
    try l.go(l.nemo.combine, dim(cdiv(d, 256), rows, 1), dim(256, 1, 1), 0, s, &a);
}
