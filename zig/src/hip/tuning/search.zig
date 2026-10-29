//! hip-affine-tune's proposal: prompt table rows from timed cells against gfx1100's choices; no device, host-tested.

const std = @import("std");
const registry = @import("core").registry;
const tuning = @import("tuning.zig");
const affine = @import("../launches/affine.zig");
const mlx_entries = @import("../launches/mlx_entries.zig");
const Entry = affine.Entry;

/// The noise above which a point cannot move a bound.
pub const noise_limit = 0.05;

/// A timed product: `bound` says its rows (plain) or items (routed) are a bound value with both sides sampled.
pub const Cell = struct { class: usize, count: usize, op: registry.Op, shape: registry.Shape, qualified: bool, bound: bool };
pub const Time = struct { us: f64, noise: f64 };
pub const Score = struct { loss: f64, saving: f64, changed: usize };

/// What a bound moves: a plain tile's rows, or with 0 for off (`plain = false`), a routed tile's items or pairs.
pub const Kind = enum { rows, gate_rows, items, pairs_per_item };
const Bound = struct { id: []const u8, kind: Kind };
/// The plain K-parallel tile of plans first, so a cheaper tile's rows cannot hide it; then by cost.
const plain_bounds = [_]Bound{
    .{ .id = "mlx.project.prefill.kp3", .kind = .gate_rows },
    .{ .id = "mlx.project.prefill.kp0", .kind = .gate_rows },
    .{ .id = "mlx.project.prefill.kp1", .kind = .rows },
    .{ .id = "mlx.project.prefill.kp2", .kind = .rows },
};
/// The routed K-parallel tiles' bounds, searched together: refusing kp3 hands a plan to kp1 before the GEMM tile.
const routed_bounds = [_]Bound{
    .{ .id = "mlx.routed.prefill.kp3", .kind = .items },
    .{ .id = "mlx.routed.prefill.kp3", .kind = .pairs_per_item },
    .{ .id = "mlx.routed.prefill.kp1", .kind = .pairs_per_item },
};

pub const Search = struct {
    gpa: std.mem.Allocator,
    env: registry.Env,
    old: affine.Registry,
    /// The table the proposal starts from each pass.
    start: *const tuning.Table,
    cells: std.ArrayList(Cell) = .empty,
    /// gfx1100's choice for each cell, an index of `mlx_entries.all`.
    old_pick: std.ArrayList(?usize) = .empty,
    /// Each cell's times by entry index.
    times: std.ArrayList([registry.max_entries]?Time) = .empty,
    rows: [registry.max_entries]tuning.Row = undefined,
    n_rows: usize = 0,

    pub fn init(gpa: std.mem.Allocator, env: registry.Env, start: *const tuning.Table) Search {
        var s: Search = .{ .gpa = gpa, .env = env, .old = affine.Registry.init(&tuning.gfx1100, &mlx_entries.all), .start = start };
        s.reset();
        return s;
    }

    pub fn deinit(s: *Search) void {
        s.cells.deinit(s.gpa);
        s.old_pick.deinit(s.gpa);
        s.times.deinit(s.gpa);
    }

    /// The proposal back at the starting table's rows.
    pub fn reset(s: *Search) void {
        for (s.start.rows, 0..) |r, i| s.rows[i] = r;
        s.n_rows = s.start.rows.len;
    }

    pub fn proposal(s: *const Search) tuning.Table {
        return .{ .gfx = "proposal", .rows = s.rows[0..s.n_rows] };
    }

    /// A new cell, its gfx1100 choice cached: its times follow with `setTime`.
    pub fn addCell(s: *Search, c: Cell) !usize {
        try s.cells.ensureUnusedCapacity(s.gpa, 1);
        try s.old_pick.ensureUnusedCapacity(s.gpa, 1);
        try s.times.ensureUnusedCapacity(s.gpa, 1);
        s.cells.appendAssumeCapacity(c);
        s.old_pick.appendAssumeCapacity(if (s.old.select(s.env, .mlx, c.op, .prefill, c.shape)) |e| index(e) else null);
        s.times.appendAssumeCapacity(@splat(null));
        return s.cells.items.len - 1;
    }

    pub fn setTime(s: *Search, cell: usize, entry: usize, time: Time) void {
        s.times.items[cell][entry] = time;
    }

    pub fn timeOf(s: *const Search, cell: usize, entry: usize) ?Time {
        return s.times.items[cell][entry];
    }

    pub fn cellOf(s: *const Search, class: usize, count: usize) ?usize {
        for (s.cells.items, 0..) |c, i| if (c.class == class and c.count == count) return i;
        return null;
    }

    /// A candidate table's score on `op`'s cells against gfx1100's choices; null when the per-cell veto refuses it.
    pub fn score(s: *const Search, table: *const tuning.Table, op: registry.Op) ?Score {
        const cand = affine.Registry.init(table, &mlx_entries.all);
        var sc: Score = .{ .loss = 0, .saving = 0, .changed = 0 };
        for (s.cells.items, 0..) |c, ci| {
            if (c.op != op) continue;
            const a = s.old_pick.items[ci] orelse continue;
            const b = index(cand.select(s.env, .mlx, c.op, .prefill, c.shape) orelse return null);
            if (a == b) continue;
            if (!c.qualified) return null;
            const ta = s.timeOf(ci, a) orelse return null;
            const tb = s.timeOf(ci, b) orelse return null;
            if (ta.noise > noise_limit or tb.noise > noise_limit) return null;
            const ratio = tb.us / ta.us;
            if (ratio > 1 + @max(ta.noise, tb.noise)) return null;
            sc.loss = @max(sc.loss, ratio - 1);
            sc.saving += ta.us - tb.us;
            sc.changed += 1;
        }
        return sc;
    }

    /// Each plain bound in turn, then the routed ones together, to the least loss, most saving, nearest gfx1100's.
    pub fn propose(s: *Search, print: bool) !void {
        for (plain_bounds) |bd| try s.proposeOne(bd, print);
        try s.proposeRouted(print);
    }

    fn proposeOne(s: *Search, bd: Bound, print: bool) !void {
        const at = rowIndex(s.rows[0..s.n_rows], bd.id) orelse return;
        // each candidate starts from the row as it was, so an off value keeps its other fields
        const orig = s.rows[at];
        const old_v = field(tuning.find(&tuning.gfx1100, bd.id).?, bd.kind);
        var best_v = field(orig, bd.kind);
        var best: Score = s.score(&s.proposal(), .project) orelse .{ .loss = 0, .saving = 0, .changed = 0 };
        const values = try s.candidates(bd.kind);
        defer s.gpa.free(values);
        for (values) |n| {
            const v: f64 = @floatFromInt(n);
            s.rows[at] = orig;
            setField(&s.rows[at], bd.kind, v);
            const sc = s.score(&s.proposal(), .project) orelse continue;
            if (better(sc, dist(v, old_v), best, dist(best_v, old_v))) {
                best = sc;
                best_v = v;
            }
        }
        s.rows[at] = orig;
        setField(&s.rows[at], bd.kind, best_v);
        if (print) std.debug.print("BOUND {s} {t} {d} -> {d}: {d} cells changed, loss {d:.3}, saving {d:.2} us\n", .{
            bd.id, bd.kind, old_v, best_v, best.changed, best.loss, best.saving,
        });
    }

    fn proposeRouted(s: *Search, print: bool) !void {
        var at: [routed_bounds.len]usize = undefined;
        var old_v: [routed_bounds.len]f64 = undefined;
        var best_v: [routed_bounds.len]f64 = undefined;
        for (routed_bounds, 0..) |bd, i| {
            at[i] = rowIndex(s.rows[0..s.n_rows], bd.id) orelse return;
            old_v[i] = field(tuning.find(&tuning.gfx1100, bd.id).?, bd.kind);
            best_v[i] = field(s.rows[at[i]], bd.kind);
        }
        const began = nowNs();
        var best: Score = s.score(&s.proposal(), .routed) orelse .{ .loss = 0, .saving = 0, .changed = 0 };
        // the row's own items too, so a pairs bound can move without them
        const items = try s.candidatesWith(.items, @intFromFloat(best_v[0]));
        defer s.gpa.free(items);
        const kp3_pairs = try s.pairCandidates(std.math.maxInt(u32));
        defer s.gpa.free(kp3_pairs);
        // kp1 takes no plan past its own items, so only those plans' bounds can change what it takes
        const kp1_pairs = try s.pairCandidates(s.rows[at[2]].items);
        defer s.gpa.free(kp1_pairs);
        var v: [routed_bounds.len]f64 = undefined;
        for (items) |n| for (kp3_pairs) |p3| for (kp1_pairs) |p1| {
            v = .{ @floatFromInt(n), p3, p1 };
            s.setRouted(at, v);
            const sc = s.score(&s.proposal(), .routed) orelse continue;
            if (better(sc, sumDist(v, old_v), best, sumDist(best_v, old_v))) {
                best = sc;
                best_v = v;
            }
        };
        s.setRouted(at, best_v);
        if (print) {
            const took = @as(f64, @floatFromInt(nowNs() - began)) / 1e6;
            std.debug.print("BOUND mlx.routed.prefill kp3 items {d} -> {d}, kp3 pairs_per_item {d} -> {d}, kp1 pairs_per_item {d} -> {d}: {d} cells changed, loss {d:.3}, saving {d:.2} us, search {d:.0} ms\n", .{
                old_v[0], best_v[0], old_v[1], best_v[1], old_v[2], best_v[2], best.changed, best.loss, best.saving, took,
            });
        }
    }

    fn setRouted(s: *Search, at: [routed_bounds.len]usize, v: [routed_bounds.len]f64) void {
        for (routed_bounds, at, v) |bd, i, x| setField(&s.rows[i], bd.kind, x);
    }

    /// Pairs bounds to try on the routed cells of at most `most` items: off, each one's least admitting, one below all.
    fn pairCandidates(s: *const Search, most: u32) ![]f64 {
        // off, one a cell at most, and the one below all
        const buf = try s.gpa.alloc(f64, s.cells.items.len + 2);
        errdefer s.gpa.free(buf);
        buf[0] = 0;
        var len: usize = 1;
        for (s.cells.items) |c| {
            if (c.op != .routed or c.shape.plan_pairs == 0 or c.shape.items == 0 or c.shape.items > most) continue;
            const b: f64 = leastAdmitting(c.shape.plan_pairs, c.shape.items);
            if (std.mem.indexOfScalar(f64, buf[0..len], b) == null) {
                buf[len] = b;
                len += 1;
            }
        }
        std.mem.sort(f64, buf[0..len], {}, std.sort.asc(f64));
        // and one that refuses them all, just below the least of their bounds
        if (len > 1) {
            const below = std.math.nextAfter(f32, @floatCast(buf[1]), 0);
            if (below > 0) {
                std.mem.copyBackwards(f64, buf[2 .. len + 1], buf[1..len]);
                buf[1] = below;
                len += 1;
            }
        }
        return s.gpa.realloc(buf, len);
    }

    /// Bound values to try: off for a gate, and the rows or items of timed cells whose both sides are sampled.
    fn candidates(s: *const Search, kind: Kind) ![]u32 {
        return s.candidatesWith(kind, null);
    }

    /// `candidates`, and `extra` too when given.
    fn candidatesWith(s: *const Search, kind: Kind, extra: ?u32) ![]u32 {
        const routed = kind == .items;
        const buf = try s.gpa.alloc(u32, s.cells.items.len + 2);
        errdefer s.gpa.free(buf);
        var len: usize = 0;
        if (kind == .gate_rows) {
            buf[0] = 0;
            len = 1;
        }
        for (s.cells.items) |c| {
            if ((c.op == .routed) != routed or !c.bound) continue;
            const v: u32 = if (routed) c.shape.items else @intCast(c.count);
            if (std.mem.indexOfScalar(u32, buf[0..len], v) == null) {
                buf[len] = v;
                len += 1;
            }
        }
        if (extra) |v| if (std.mem.indexOfScalar(u32, buf[0..len], v) == null) {
            buf[len] = v;
            len += 1;
        };
        return s.gpa.realloc(buf, len);
    }

    /// Whether the choice between two timed counts of a class changes, under gfx1100's table or the proposal.
    pub fn changes(s: *const Search, class: usize, a: usize, b: usize) bool {
        const table = s.proposal();
        const cand = affine.Registry.init(&table, &mlx_entries.all);
        const ca = s.cellOf(class, a) orelse return false;
        const cb = s.cellOf(class, b) orelse return false;
        const c = s.cells.items;
        const pa = cand.select(s.env, .mlx, c[ca].op, .prefill, c[ca].shape);
        const pb = cand.select(s.env, .mlx, c[cb].op, .prefill, c[cb].shape);
        const oa = s.old_pick.items[ca];
        const ob = s.old_pick.items[cb];
        if (oa != ob or pa != pb) return true;
        // the two tables' choices trade places in speed between the counts
        return s.lead(ca, oa, pa) != s.lead(cb, ob, pb);
    }

    /// Whether gfx1100's choice and the proposal's at a count differ by less than both their noises.
    pub fn close(s: *const Search, class: usize, count: usize) bool {
        const table = s.proposal();
        const cand = affine.Registry.init(&table, &mlx_entries.all);
        const ci = s.cellOf(class, count) orelse return false;
        const c = s.cells.items[ci];
        const ta = s.timeOf(ci, s.old_pick.items[ci] orelse return false) orelse return false;
        const tb = s.timeOf(ci, index(cand.select(s.env, .mlx, c.op, .prefill, c.shape) orelse return false)) orelse return false;
        return @abs(ta.us - tb.us) < @min(ta.noise, tb.noise) * @min(ta.us, tb.us);
    }

    fn lead(s: *const Search, cell: usize, a: ?usize, b: ?*const Entry) i2 {
        const ta = s.timeOf(cell, a orelse return 0) orelse return 0;
        const tb = s.timeOf(cell, index(b orelse return 0)) orelse return 0;
        return if (ta.us < tb.us) -1 else if (ta.us > tb.us) 1 else 0;
    }
};

pub fn index(e: *const Entry) usize {
    return (@intFromPtr(e) - @intFromPtr(&mlx_entries.all[0])) / @sizeOf(Entry);
}

fn rowIndex(rows: []const tuning.Row, id: []const u8) ?usize {
    for (rows, 0..) |r, i| if (std.mem.eql(u8, r.id, id)) return i;
    return null;
}

/// The least `f32` bound under which a plan of `pairs` pairs over `items` items is taken, as the registry compares.
pub fn leastAdmitting(pairs: u32, items: u32) f32 {
    const p: f64 = @floatFromInt(pairs);
    const n: f64 = @floatFromInt(items);
    var b: f32 = @floatCast(p / n);
    while (@as(f64, b) * n < p) b = std.math.nextAfter(f32, b, std.math.inf(f32));
    while (true) {
        const lower = std.math.nextAfter(f32, b, 0);
        if (lower <= 0 or @as(f64, lower) * n < p) break;
        b = lower;
    }
    return b;
}

fn better(a: Score, a_dist: f64, b: Score, b_dist: f64) bool {
    return a.loss < b.loss or (a.loss == b.loss and (a.saving > b.saving or (a.saving == b.saving and a_dist < b_dist)));
}

fn field(r: tuning.Row, kind: Kind) f64 {
    return switch (kind) {
        .rows => @floatFromInt(r.rows),
        .gate_rows => if (r.plain) @floatFromInt(r.rows) else 0,
        .items => @floatFromInt(r.items),
        .pairs_per_item => r.pairs_per_item,
    };
}

fn setField(r: *tuning.Row, kind: Kind, v: f64) void {
    switch (kind) {
        .rows => r.rows = @intFromFloat(v),
        .gate_rows => {
            r.plain = v != 0;
            if (v != 0) r.rows = @intFromFloat(v);
        },
        .items => r.items = @intFromFloat(v),
        .pairs_per_item => r.pairs_per_item = @floatCast(v),
    }
}

fn dist(a: f64, b: f64) f64 {
    return @abs(a - b);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn sumDist(a: [routed_bounds.len]f64, b: [routed_bounds.len]f64) f64 {
    var d: f64 = 0;
    for (a, b) |x, y| d += dist(x, y);
    return d;
}

// ---- the search on synthetic times, no device

const t = std.testing;
/// RDNA4's environment: bf16, no matrix tile, the stream and GEMM tiles on.
const rdna4: registry.Env = .{ .bf16 = true, .matrix = false, .matrix_on = false, .stream_on = true, .gemm_on = true };

fn entryOf(id: []const u8) usize {
    for (&mlx_entries.all, 0..) |*e, i| if (std.mem.eql(u8, e.id, id)) return i;
    unreachable;
}

/// A Qwen3.6 gate and up plan of `items` items and `pairs` pairs, with kp3, kp1 and GEMM times; `noise` on them all.
fn plan(s: *Search, count: usize, items: u32, pairs: u32, kp3: f64, kp1: f64, gemm: f64, noise: f64) !void {
    const shape: registry.Shape = .{ .m = 8, .n = 1024, .k = 2048, .bits = 4, .group = 64, .fp16 = false, .items = items, .plan_pairs = pairs };
    const ci = try s.addCell(.{ .class = 0, .count = count, .op = .routed, .shape = shape, .qualified = true, .bound = true });
    for ([_]struct { []const u8, f64 }{ .{ "kp3", kp3 }, .{ "kp1", kp1 }, .{ "gemm", gemm }, .{ "block", 2 * gemm } }) |x| {
        var buf: [64]u8 = undefined;
        const id = std.fmt.bufPrint(&buf, "mlx.routed.prefill.{s}", .{x[0]}) catch unreachable;
        s.setTime(ci, entryOf(id), .{ .us = x[1], .noise = noise });
    }
}

fn routedRow(s: *const Search, id: []const u8) tuning.Row {
    return s.rows[rowIndex(s.rows[0..s.n_rows], id).?];
}

/// Whether a row's bounds take a plan of `items` items and `pairs` pairs, as the registry compares.
fn takes(r: tuning.Row, items: u32, pairs: u32) bool {
    if (r.items != 0 and items > r.items) return false;
    return r.pairs_per_item == 0 or @as(f64, @floatFromInt(pairs)) <= @as(f64, r.pairs_per_item) * @as(f64, @floatFromInt(items));
}

test "the routed bounds move together: more items, and a pairs bound that keeps the full plans off kp3" {
    var s = Search.init(t.allocator, rdna4, &tuning.gfx1201);
    defer s.deinit();
    try plan(&s, 33, 329, 297, 5, 6, 10, 0.01); // 0.9 pairs an item: kp3 wins
    try plan(&s, 129, 320, 774, 8, 9, 10, 0.01); // 2.42: kp3 wins
    try plan(&s, 256, 320, 1536, 12, 13, 10, 0.01); // 4.8: kp3 loses
    try s.propose(false);
    const kp3 = routedRow(&s, "mlx.routed.prefill.kp3");
    try t.expectEqual(@as(u32, 329), kp3.items);
    try t.expect(takes(kp3, 329, 297) and takes(kp3, 320, 774) and !takes(kp3, 320, 1536));
    // the tightest such bound: the 129-token cell's own
    try t.expectEqual(leastAdmitting(774, 320), kp3.pairs_per_item);
    try t.expectEqual(@as(f32, 0), routedRow(&s, "mlx.routed.prefill.kp1").pairs_per_item);
}

test "a plan kp3 refuses passes kp1 by on the way to the GEMM tile, so both bounds move in one step" {
    var s = Search.init(t.allocator, rdna4, &tuning.gfx1201);
    defer s.deinit();
    try plan(&s, 33, 329, 297, 5, 6, 10, 0.01);
    try plan(&s, 128, 224, 768, 12, 14, 10, 0.01); // kp3 today; kp1 is slower still
    try s.propose(false);
    const kp3 = routedRow(&s, "mlx.routed.prefill.kp3");
    const kp1 = routedRow(&s, "mlx.routed.prefill.kp1");
    try t.expect(takes(kp3, 329, 297) and !takes(kp3, 224, 768));
    try t.expect(kp1.pairs_per_item != 0 and !takes(kp1, 224, 768));
    const sc = s.score(&s.proposal(), .routed).?;
    try t.expectEqual(@as(usize, 2), sc.changed);
}

test "a noisy cell keeps its choice" {
    var s = Search.init(t.allocator, rdna4, &tuning.gfx1201);
    defer s.deinit();
    try plan(&s, 33, 329, 297, 5, 6, 10, 0.01);
    try plan(&s, 60, 329, 600, 5, 6, 10, 0.2); // 1.82 pairs an item, too noisy to move
    try s.propose(false);
    const kp3 = routedRow(&s, "mlx.routed.prefill.kp3");
    try t.expect(takes(kp3, 329, 297) and !takes(kp3, 329, 600));
}

test "a cell timed after a proposal can veto it" {
    var s = Search.init(t.allocator, rdna4, &tuning.gfx1201);
    defer s.deinit();
    try plan(&s, 33, 329, 297, 5, 6, 10, 0.01);
    try plan(&s, 129, 320, 774, 8, 9, 10, 0.01);
    try plan(&s, 256, 320, 1536, 12, 13, 10, 0.01);
    try s.propose(false);
    try t.expect(takes(routedRow(&s, "mlx.routed.prefill.kp3"), 329, 600));
    try plan(&s, 60, 329, 600, 11, 12, 10, 0.01); // 1.82 pairs an item, and kp3 loses here
    s.reset();
    try s.propose(false);
    const kp3 = routedRow(&s, "mlx.routed.prefill.kp3");
    try t.expect(takes(kp3, 329, 297) and !takes(kp3, 329, 600));
    try t.expect(s.score(&s.proposal(), .routed).?.loss == 0);
}

test "an unqualified cell keeps its choice, however fast its times say the change is" {
    var s = Search.init(t.allocator, rdna4, &tuning.gfx1201);
    defer s.deinit();
    try plan(&s, 33, 329, 297, 5, 6, 10, 0.01);
    try plan(&s, 60, 329, 600, 5, 6, 10, 0.01); // 1.82 pairs an item, but its reads stay in the caches
    s.cells.items[s.cells.items.len - 1].qualified = false;
    try s.propose(false);
    const kp3 = routedRow(&s, "mlx.routed.prefill.kp3");
    try t.expect(takes(kp3, 329, 297) and !takes(kp3, 329, 600));
}

test "a cell's own bound takes it, where its ratio rounded to an f32 would not" {
    // Nemotron at 186 tokens: 1,116 pairs over 320 items, 3.4875, whose nearest f32 is just below
    const near: f32 = @floatCast(@as(f64, 1116) / 320);
    try t.expect(@as(f64, near) * 320 < 1116);
    const b = leastAdmitting(1116, 320);
    try t.expect(@as(f64, b) * 320 >= 1116);
    try t.expect(@as(f64, std.math.nextAfter(f32, b, 0)) * 320 < 1116);
    try t.expectEqual(@as(f32, 3.5), leastAdmitting(350, 100));
}

test "every routed cell's bound is a candidate, however many there are" {
    var s = Search.init(t.allocator, rdna4, &tuning.gfx1201);
    defer s.deinit();
    for (0..1100) |i| try plan(&s, i + 1, 2000, @intCast(i + 1), 5, 6, 10, 0.01);
    const all = try s.pairCandidates(std.math.maxInt(u32));
    defer t.allocator.free(all);
    // off, the one below them all, and one a cell
    try t.expectEqual(@as(usize, 1102), all.len);
    try t.expect(all[1] < all[2] and @as(f64, all[1]) * 2000 < 1);
}
