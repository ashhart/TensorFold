//! A fact's pocket outputs fitted by Adam through the real forward until held-out loss stops falling, then judged.
const std = @import("std");
const examples = @import("lesson_examples.zig");
const lesson_route = @import("lesson_route.zig");
const subspace = @import("subspace.zig");
const pocket_fit = @import("pocket_fit.zig");
const pack = @import("pocket_pack.zig");
const train_mode = @import("train_mode.zig");
const prefix = @import("pocket_prefix.zig");

const Example = examples.Example;
const Acting = lesson_route.Acting;
const Seq = train_mode.Seq;
const Result = train_mode.Result;
const dotBf16 = pocket_fit.dotBf16;
const normBf16 = pocket_fit.normBf16;

const value_steps = 60; // Adam steps fitting the fact's neurons' outputs, a round at most
const min_value_steps = 12; // and at least
const batch = 8; // the fact's answers a step takes, drawn in shuffled passes
const keep_batch = 8; // and as many other prompts' answers its subject neurons fire on, which are to stay as they were
const joint_batch = 8; // and earlier facts' answers on its subject, when their outputs are fitted with its
const judge_every = 4; // steps between readings of the held-out answers' loss
const patience = 3; // readings without a better held-out loss after which the outputs stop, the best kept
const reach: f32 = 0.1; // a step's size against the residual's
const decay: f32 = 0.05; // the outputs' squared size against the residual's, added to the loss
const weakest: f32 = 0.5; // an answer's outputs scaled down to as little as this while fitting: new wordings fire less

/// The outputs' fitting as it goes: its Adam state and the held-out readings.
pub const State = struct {
    round: usize = 0,
    loss: f32 = 0, // the step's fact answers' summed loss
    kept: f32 = 0, // and that of its answers that are to stay
    first_loss: f32 = 0,
    running: f32 = 0, // the fact answers' loss over recent steps
    keep_running: f32 = 0,
    judging: bool = false, // reading the held-out answers' loss with the outputs as they are
    held_loss: f32 = 0,
    best_held: f32 = std.math.inf(f32), // the lowest held-out loss read this round, and the outputs it came with
    best_step: usize = 0,
    stale: usize = 0, // readings since it last fell
    best: std.ArrayList(f32) = .empty,
    order: std.ArrayList(usize) = .empty, // the fact's answers in this pass's shuffled order
    drawn: usize = 0,
    keep_order: std.ArrayList(usize) = .empty, // the answers that are to stay, likewise
    keep_drawn: usize = 0,
    keep_loss: std.ArrayList(f32) = .empty, // each answer to stay, its loss when last drawn (-1: not yet)
    joint_order: std.ArrayList(usize) = .empty, // the earlier facts' answers fitted with it, likewise
    joint_drawn: usize = 0,
    joint: f32 = 0, // the step's earlier facts' answers' summed loss
    joint_running: f32 = 0,
    shuffles: u64 = 0,
    adam: std.ArrayList(f32) = .empty, // [4, taken, D]: the outputs, Adam's two moments, a step's gradient
    size: std.ArrayList(f32) = .empty, // [taken]: the residual's length after each neuron's layer, at its row
    idle: usize = 0, // answers drawn this round that fired no neuron, or whose loss or gradient was not finite
    prefixes: prefix.Prefixes = .{}, // the answers' rows no output reaches, run once

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        inline for (.{ "best", "order", "keep_order", "keep_loss", "joint_order", "adam", "size", "prefixes" }) |f| @field(s, f).deinit(gpa);
    }

    /// A round begun: the readings cleared; the outputs and Adam's moments as they are.
    pub fn restart(s: *State) void {
        s.round = 0;
        s.idle = 0;
        s.loss = 0;
        s.kept = 0;
        s.judging = false;
        s.held_loss = 0;
        s.best_held = std.math.inf(f32);
        s.best_step = 0;
        s.stale = 0;
    }

    /// A fact's `taken` neurons begun on its `answers` (its own and those to stay): outputs zero, passes drawn afresh.
    pub fn begin(s: *State, gpa: std.mem.Allocator, taken: usize, d: usize, answers: usize) !void {
        try s.prefixes.begin(gpa, answers);
        s.adam.clearRetainingCapacity();
        try s.adam.appendNTimes(gpa, 0, 4 * taken * d);
        s.size.clearRetainingCapacity();
        try s.size.appendNTimes(gpa, 0, taken);
        s.order.clearRetainingCapacity();
        s.drawn = 0;
        s.keep_order.clearRetainingCapacity();
        s.keep_drawn = 0;
        s.keep_loss.clearRetainingCapacity();
        s.joint_order.clearRetainingCapacity();
        s.joint_drawn = 0;
        s.best.clearRetainingCapacity(); // a fit that never reads a better held-out loss keeps its own outputs, not the last fact's
    }
};

/// The next of `n` (more than none) in shuffled passes, a new order each pass.
fn draw(gpa: std.mem.Allocator, s: *State, order: *std.ArrayList(usize), drawn: *usize, n: usize) !usize {
    if (drawn.* == order.items.len) {
        order.clearRetainingCapacity();
        for (0..n) |i| try order.append(gpa, i);
        s.shuffles += 1;
        var prng = std.Random.DefaultPrng.init(s.shuffles);
        prng.random().shuffle(usize, order.items);
        drawn.* = 0;
    }
    drawn.* += 1;
    return order.items[drawn.* - 1];
}

/// The next of the fact's answers that any of its neurons fire on.
fn drawKeyed(p: anytype) !usize {
    for (0..2 * p.answered.len + 1) |_| {
        const e = try draw(p.gpa, &p.v, &p.v.order, &p.v.drawn, p.answered.len);
        if (p.fires(p.answered[e])) return e;
    }
    return error.NoKeyedAnswers;
}

/// The answer to stay that read worst when last drawn and is not in this step yet, or null (none read above zero).
fn worstKeep(s: *const State, step: []const Drawn, own: usize) ?usize {
    var best: ?usize = null;
    for (s.keep_loss.items, 0..) |loss, e| {
        if (loss <= 0 or (best != null and loss <= s.keep_loss.items[best.?])) continue;
        for (step) |dr| {
            if (!dr.own and dr.which == own + e) break;
        } else best = e;
    }
    return best;
}

/// An answer an output step drew: the fact's own or one that is to stay, which (own answers first), its sequence.
const Drawn = struct { own: bool, joint: bool = false, which: usize, seq: Seq };

/// One output step: its fact answers, then those that are to stay, side by side; then Adam, and every few the judge.
pub fn valueOnce(p: anytype, t: anytype, host: anytype) !bool {
    const s = &p.v;
    const c = t.config();
    const d = c.hidden;
    const k = p.taken.items.len + p.joint.items.len; // the open fact's neurons, then the earlier ones fitted with them
    const vals = s.adam.items[0 .. k * d];
    const grad = s.adam.items[3 * k * d ..][0 .. k * d];
    if (s.judging) return judgeOnce(p, t, host, vals);
    if (p.answered.len == 0) return error.NoKeyedAnswers;
    if (s.round == 0) prefix.round(p, &s.prefixes);
    // usize: @min with a comptime operand narrows its type (to u4 for 8), and their sum would overflow it
    const take: usize = @min(batch, p.answered.len);
    const joins: usize = @min(joint_batch, p.joint_answers.items.len);
    const keeps: usize = @min(keep_batch, p.keeps.items.len);
    const kept_from = p.answered.len + p.joint_answers.items.len; // the answers to stay come after both
    var drawn: [batch + joint_batch + keep_batch]Drawn = undefined;
    var n: usize = 0;
    if (s.keep_loss.items.len != p.keeps.items.len) {
        s.keep_loss.clearRetainingCapacity();
        try s.keep_loss.appendNTimes(p.gpa, -1, p.keeps.items.len);
    }
    for (0..take + joins + keeps) |i| {
        const own = i < take;
        const join = !own and i < take + joins;
        var ex: Example = undefined;
        var which: usize = undefined;
        var rows: [3]usize = @splat(0);
        var on: [3]bool = @splat(false);
        if (own) {
            which = try drawKeyed(p);
            ex = p.answered[which];
            const subject = p.keyRow(ex, .subject);
            const value = p.keyRow(ex, .value);
            rows = .{ ex.decide, subject orelse 0, value orelse 0 };
            on = .{ true, subject != null, value != null };
        } else if (join) {
            const e = try draw(p.gpa, s, &s.joint_order, &s.joint_drawn, p.joint_answers.items.len);
            which = p.answered.len + e;
            ex = p.joint_answers.items[e];
            const subject = p.keyRow(ex, .subject);
            rows = .{ ex.decide, subject orelse 0, 0 };
            on = .{ false, subject != null, false };
        } else {
            // every other one the answer that read worst (maximally interfered retrieval, Aljundi et al. 2019)
            const worst = if (p.worst and (i - take - joins) % 2 == 1) worstKeep(s, drawn[0..n], kept_from) else null;
            const e = worst orelse try draw(p.gpa, s, &s.keep_order, &s.keep_drawn, p.keeps.items.len);
            const keep = p.keeps.items[e];
            which = kept_from + e;
            ex = keep.ex;
            rows = .{ 0, keep.row, 0 };
            on = .{ false, true, false };
        }
        var masks: [3]u64 = @splat(0);
        for (p.taken.items) |x| if (on[@backingInt(x.key)]) {
            masks[@backingInt(x.key)] |= @as(u64, 1) << @intCast(p.moe.items[x.slot]);
        };
        for (p.joint.items) |x| if (on[1]) {
            masks[1] |= @as(u64, 1) << @intCast(p.moe.items[x.slot]);
        };
        if (masks[0] | masks[1] | masks[2] == 0) {
            if (!own and !join) s.keep_loss.items[which - kept_from] = 0; // no neuron reaches it: nothing to guard
            s.idle += 1;
            continue;
        }
        // its outputs scaled as a wording that fires them less would see them
        const scale = weakest + (1 - weakest) * unit(s.round, i + 1);
        drawn[n] = .{ .own = own, .joint = join, .which = which, .seq = .{ .ids = ex.ids, .start = ex.start, .decide = ex.decide, .scale = scale, .nudge = .{ .layers = masks, .rows = rows } } };
        n += 1;
    }
    // the neurons as they are now in the forward, the open fact's scaled on each answer's rows
    try put(p, t, host, vals, d);
    p.open(t);
    var seqs: [batch + joint_batch + keep_batch]Seq = undefined;
    var which: [batch + joint_batch + keep_batch]usize = undefined;
    for (seqs[0..n], which[0..n], drawn[0..n]) |*q, *w, dr| {
        q.* = dr.seq;
        w.* = dr.which;
    }
    // each answer's rows before its open neurons first fire run once, from its second draw on
    try prefix.use(t, &s.prefixes, seqs[0..n], which[0..n]);
    var at: usize = 0;
    while (at < n) {
        const m = pack.fits(t, seqs[at..n]);
        var got: [batch + joint_batch + keep_batch]Result = undefined;
        for (seqs[at..][0..m], 0..) |q, i| for (q.nudge.layers, 0..) |mask, key| {
            var bits = mask;
            while (bits != 0) : (bits &= bits - 1) @memset(t.deltaAt(i, key, @ctz(bits)), 0);
        };
        try t.pass(seqs[at..][0..m], .target, pack.turn, got[0..m]);
        for (drawn[at..][0..m], seqs[at..][0..m], got[0..m], 0..) |dr, q, r, i| {
            absorb(p, t, dr, i, r, grad);
            prefix.learn(p, t, &s.prefixes, q, t.spans[i], dr.which);
        }
        at += m;
    }
    const mean = s.loss / @as(f32, @floatFromInt(take));
    const kept = s.kept / @as(f32, @floatFromInt(@max(keeps, 1)));
    const joined = s.joint / @as(f32, @floatFromInt(@max(joins, 1)));
    s.running = if (s.round == 0) mean else 0.7 * s.running + 0.3 * mean;
    s.keep_running = if (s.round == 0) kept else 0.7 * s.keep_running + 0.3 * kept;
    s.joint_running = if (s.round == 0) joined else 0.7 * s.joint_running + 0.3 * joined;
    if (s.round == 0) s.first_loss = mean;
    s.loss = 0;
    s.kept = 0;
    s.joint = 0;
    p.at = 0;
    s.round += 1;
    if (s.round <= value_steps) adam(s, vals, grad, k, d, take + joins + keeps);
    @memset(grad, 0);
    if (s.round % judge_every == 0 or s.round > value_steps) {
        if (p.held.len == 0) {
            if (s.round > value_steps) return finish(p, t, host, vals);
        } else {
            try put(p, t, host, vals, d);
            s.judging = true;
            s.held_loss = 0;
        }
    }
    return false;
}

/// Answer `dr`, the pass's i-th, into the step: its loss, and each neuron's push times its activation at its row.
fn absorb(p: anytype, t: anytype, dr: Drawn, i: usize, got: Result, grad: []f32) void {
    const s = &p.v;
    const c = t.config();
    const d = c.hidden;
    const masks = dr.seq.nudge.layers;
    const rows = dr.seq.nudge.rows;
    const sp = t.spans[i];
    // a loss or gradient that is not finite teaches nothing and would spoil every step after it
    const bad = for (p.taken.items) |x| {
        const key = @backingInt(x.key);
        if (masks[key] != 0 and !finite(t.pushAt(i, key, p.moe.items[x.slot]))) break true;
    } else !std.math.isFinite(got.loss);
    if (bad) {
        if (s.idle == 0) std.log.info("slide pocket: {s} answer of {d} tokens (from {d}, rows {d} and {d}) gave a loss of {d:.3} or a gradient that is not finite; skipped", .{ if (dr.own) "a fact" else "a kept", dr.seq.ids.len, dr.seq.start, rows[0], rows[@intFromBool(rows[1] == 0) + 1], got.loss });
        s.idle += 1;
        return;
    }
    if (dr.own) s.loss += got.loss else if (dr.joint) s.joint += got.loss else {
        s.kept += got.loss;
        s.keep_loss.items[dr.which - p.answered.len - p.joint_answers.items.len] = got.loss;
    }
    for (p.taken.items, 0..) |x, j| {
        const key = @backingInt(x.key);
        if (masks[key] == 0) continue;
        const l = p.moe.items[x.slot];
        const z = dotBf16(p.neuron(p.up.items, x.slot, x.neuron, d), t.normed(l, sp.row(rows[key])));
        const a = if (z > 0) z * z else 0;
        subspace.axpy(grad[j * d ..][0..d], dr.seq.scale * a, t.pushAt(i, key, l));
        if (dr.own and s.size.items[j] == 0) s.size.items[j] = normBf16(t.residual(l + 1, sp.row(rows[key])));
    }
    // the earlier neurons on its subject, in the forward unscaled, pushed by every answer whose subject row they read
    if (masks[1] != 0) for (p.joint.items, p.taken.items.len..) |x, j| {
        const l = p.moe.items[x.slot];
        const z = dotBf16(p.neuron(p.up.items, x.slot, x.neuron, d), t.normed(l, sp.row(rows[1])));
        if (z <= 0) continue;
        subspace.axpy(grad[j * d ..][0..d], z * z, t.pushAt(i, 1, l));
        if (s.size.items[j] == 0) s.size.items[j] = normBf16(t.residual(l + 1, sp.row(rows[1])));
    };
}

/// The fitted outputs into the pocket and the forward: the open fact's neurons, then the earlier ones fitted with them.
fn put(p: anytype, t: anytype, host: anytype, vals: []const f32, d: usize) !void {
    for (p.taken.items, 0..) |x, j| @memcpy(p.neuron(p.down.items, x.slot, x.neuron, d), vals[j * d ..][0..d]);
    for (p.joint.items, p.taken.items.len..) |x, j| @memcpy(p.neuron(p.down.items, x.slot, x.neuron, d), vals[j * d ..][0..d]);
    for (p.taken.items) |x| try p.upload(t, host, x.slot);
    for (p.joint.items) |x| try p.upload(t, host, x.slot);
}

/// A number in [0, 1) spread over a round's answers, the same for the same round and answer.
fn unit(round: usize, at: usize) f32 {
    var x: u64 = (@as(u64, round) << 32) ^ @as(u64, at);
    x = (x ^ (x >> 30)) *% 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) *% 0x94d049bb133111eb;
    x ^= x >> 31;
    return @as(f32, @floatFromInt(x >> 40)) / @as(f32, 1 << 24);
}

fn finite(xs: []const f32) bool {
    for (xs) |x| if (!std.math.isFinite(x)) return false;
    return true;
}

/// One Adam step on the outputs from a step's gradient over `count` answers, each neuron's step sized to its residual.
fn adam(s: *State, vals: []f32, grad: []const f32, k: usize, d: usize, count: usize) void {
    const m = s.adam.items[k * d ..][0 .. k * d];
    const v = s.adam.items[2 * k * d ..][0 .. k * d];
    const tt: f32 = @floatFromInt(s.round);
    for (0..k) |j| {
        const size = @max(s.size.items[j], 1e-3);
        const lr = reach * size / @sqrt(@as(f32, @floatFromInt(d)));
        for (vals[j * d ..][0..d], m[j * d ..][0..d], v[j * d ..][0..d], grad[j * d ..][0..d]) |*xi, *mi, *vi, gi| {
            const g = gi / @as(f32, @floatFromInt(@max(count, 1))) + 2 * decay * xi.* / (size * size);
            mi.* = 0.9 * mi.* + 0.1 * g;
            vi.* = 0.999 * vi.* + 0.001 * g * g;
            xi.* -= lr * (mi.* / (1 - std.math.pow(f32, 0.9, tt))) / (@sqrt(vi.* / (1 - std.math.pow(f32, 0.999, tt))) + 1e-8);
        }
    }
}

/// The held-out answers' loss with the outputs as they are; the best kept, done once it stops falling.
fn judgeOnce(p: anytype, t: anytype, host: anytype, vals: []f32) !bool {
    const s = &p.v;
    const got = try p.gpa.alloc(Result, p.held.len);
    defer p.gpa.free(got);
    try pack.read(p.gpa, t, p.held, .loss, got);
    s.held_loss = 0;
    for (got) |r| s.held_loss += r.loss;
    const loss = s.held_loss / @as(f32, @floatFromInt(p.held.len));
    s.judging = false;
    p.at = 0;
    if (loss < s.best_held) {
        s.best_held = loss;
        s.best_step = s.round;
        s.stale = 0;
        try s.best.resize(p.gpa, vals.len);
        @memcpy(s.best.items, vals);
    } else s.stale += 1;
    if (s.round > value_steps or (s.round >= min_value_steps and s.stale >= patience)) {
        if (s.best.items.len == vals.len) @memcpy(vals, s.best.items);
        return finish(p, t, host, vals);
    }
    return false;
}

/// The outputs into the pocket's neurons and the forward; the held-out answers judged next.
fn finish(p: anytype, t: anytype, host: anytype, vals: []const f32) !bool {
    const s = &p.v;
    const d = t.config().hidden;
    try put(p, t, host, vals, d);
    t.from = @splat(std.math.maxInt(u32));
    std.log.info("slide pocket: {d} answers' first rows run once, {d} rows the steps did not run again", .{ s.prefixes.slots, s.prefixes.skipped });
    try prefix.release(t, &s.prefixes);
    std.log.info("slide pocket: the fact's {d} neurons' outputs fitted in {d} steps of {d} answers and {d} of the {d} that are to stay ({d} drawn fired none or lost no finite loss), the answers' loss {d:.3} to {d:.3}, theirs {d:.3}; the held-out answers' loss at its lowest {d:.3}, step {d}", .{ p.taken.items.len, s.round, @min(batch, p.answered.len), @min(keep_batch, p.keeps.items.len), p.keeps.items.len, s.idle, s.first_loss, s.running, s.keep_running, s.best_held, s.best_step });
    var hurt: usize = 0;
    var unread: usize = 0;
    var worst: [3]f32 = @splat(0);
    for (s.keep_loss.items) |loss| {
        hurt += @intFromBool(loss > 1);
        unread += @intFromBool(loss < 0);
        if (loss > worst[2]) worst[2] = loss;
        std.mem.sort(f32, &worst, {}, std.sort.desc(f32));
    }
    std.log.info("slide pocket: of the {d} answers to stay, {d} read above 1.0 when last drawn (worst {d:.2} {d:.2} {d:.2}), {d} never drawn{s}", .{ s.keep_loss.items.len, hurt, worst[0], worst[1], worst[2], unread, if (p.worst) ", half of each step's drawn by the worst" else "" });
    if (p.joint.items.len > 0) std.log.info("slide pocket: the earlier facts' answers on its subject, fitted with it: loss {d:.3}", .{s.joint_running});
    p.back = 0;
    p.next(.check);
    return false;
}

/// The held-out answers judged token for token with the pocket in the forward, side by side; true once judged.
pub fn checkOnce(p: anytype, t: anytype, host: anytype) !bool {
    try @import("pocket_mark.zig").asWritten(p, t, host); // judged, and asked after, as a save writes it
    if (p.held.len > 0) try verifyForward(p, t, host, p.held[0]);
    const got = try p.gpa.alloc(Result, p.held.len);
    defer p.gpa.free(got);
    try pack.read(p.gpa, t, p.held, .loss, got);
    for (got, 1..) |r, i| {
        p.back += @intFromBool(r.recalled);
        std.log.info("slide pocket: held-out answer {d}: loss {d:.3}, {s}", .{ i, r.loss, if (r.recalled) "comes back" else "does not come back" });
    }
    std.log.info("slide pocket: {d} of {d} held-out answers come back token for token, the pocket in the forward", .{ p.back, p.held.len });
    p.next(.done);
    return true;
}

/// Logs the forward's change after the fact's lowest neuron on `ex` against its relu(u x)^2 v, to spot a bad forward.
fn verifyForward(p: anytype, t: anytype, host: anytype, ex: Example) !void {
    const d = t.config().hidden;
    var low: ?std.meta.Elem(@TypeOf(p.taken.items)) = null;
    var row: usize = 0;
    for (p.taken.items) |x| {
        const r = p.keyRow(ex, x.key) orelse continue;
        if (low == null or p.moe.items[x.slot] < p.moe.items[low.?.slot]) {
            low = x;
            row = r;
        }
    }
    const k = low orelse return;
    const l = p.moe.items[k.slot];
    const turn: lesson_route.Turn = .{ .decide = ex.decide, .bits = Acting.empty, .whole = true };
    const work = try p.gpa.alloc(f32, 4 * d);
    defer p.gpa.free(work);
    const with = work[0..d];
    const without = work[d..][0..d];
    const want = work[2 * d ..][0..d];
    const saved = work[3 * d ..][0..d];
    // target passes with nothing nudged: both backends keep every residual for the host then
    t.nudge = .{};
    _ = try t.step(ex.ids, ex.start, .target, turn);
    for (with, t.residual(l + 1, row)) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
    const z = dotBf16(p.neuron(p.up.items, k.slot, k.neuron, d), t.normed(l, row));
    const a = if (z > 0) z * z else 0;
    const v = p.neuron(p.down.items, k.slot, k.neuron, d);
    for (want, v) |*o, x| o.* = a * x;
    @memcpy(saved, v);
    @memset(v, 0);
    try p.upload(t, host, k.slot);
    _ = try t.step(ex.ids, ex.start, .target, turn);
    for (without, t.residual(l + 1, row)) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
    @memcpy(v, saved);
    try p.upload(t, host, k.slot);
    for (with, without) |*o, x| o.* -= x;
    const gw = subspace.dot(with, want);
    const gg = @sqrt(subspace.dot(with, with));
    const ww = @sqrt(subspace.dot(want, want));
    std.log.info("slide pocket: at layer {d}, its row: the forward adds {d:.2}, the neuron by its weights {d:.2} (activation {d:.2}), cosine {d:.3}", .{ l, gg, ww, a, if (gg * ww > 0) gw / (gg * ww) else 0 });
}
