//! One round's miss plan for a per-layer EXL3 expert pool (the host half of `cuda_exl3_pool.Pool.plan`).

const std = @import("std");
const offload = @import("../offload/exl3_offload.zig");

const shape = offload.shape;
const Vram = offload.Vram;
const empty_id = offload.Cache.empty_id;
pub const shared_id = offload.shared_id;

/// A copy of routed expert `id` into pool line `line`; `evicted` is the expert the line held (empty_id: none).
pub const Copy = struct { id: u16, line: u32, evicted: u16 };

/// Ranges into `Round.ids` (the experts the pass computes) and `Round.copies` (what it stages first).
pub const Pass = struct { id_lo: u32, id_hi: u32, copy_lo: u32, copy_hi: u32 };

pub const Options = struct {
    swap_every: u16 = 4,
    swap_cap: u16 = 96,
    observe: bool = true,
    /// Bytes a swap round may prefetch (the copy budget); 0 disables prefetch.
    prefetch_bytes: u64 = 0,
    /// Device bytes of one line (prefetch budget unit).
    line_bytes: u64 = 1,
    /// Double buffering: experts (of this layer) whose lines a single-pass round should not evict when it has room
    protect: []const u16 = &.{},
};

/// Caller-owned storage for one round's plan, sized once by `init` for the pool's line count and pick capacity.
pub const Round = struct {
    ids: []u16,
    /// The line each id of `ids` is read from in its pass (a later pass of the same round may reuse that line).
    lines: []u32,
    copies: []Copy,
    passes: []Pass,
    routed: []u16, // the round's routed picks as u16 (ordering scratch)
    misses: []u16,
    snap: []u16, // line -> id before a pass (empty_id for empty lines)
    keep: []u16, // pass ids plus protected ids (stageUsedAt's keep list)
    /// The round spared `protect`'s lines (double buffering held).
    protected: bool = false,
    n_ids: usize = 0,
    n_copies: usize = 0,
    n_passes: usize = 0,
    n_prefetch: usize = 0,
    /// Routed slots (pick in 0..511) whose expert was held / missing when the round started.
    hit_slots: u32 = 0,
    miss_slots: u32 = 0,
    shared_slots: u32 = 0,
    /// Distinct routed experts, of which `held` were already in a line (`pending` of those still in flight).
    unique: u32 = 0,
    held: u32 = 0,
    pending: u32 = 0,
    /// Copies that replaced a held expert (mandatory and prefetch).
    evictions: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, lines: usize, max_picks: usize) !Round {
        const uniq = @min(max_picks, shape.experts);
        var r: Round = undefined;
        r.ids = try gpa.alloc(u16, uniq);
        errdefer gpa.free(r.ids);
        r.lines = try gpa.alloc(u32, uniq);
        errdefer gpa.free(r.lines);
        r.copies = try gpa.alloc(Copy, uniq + shape.experts);
        errdefer gpa.free(r.copies);
        r.passes = try gpa.alloc(Pass, uniq + 1);
        errdefer gpa.free(r.passes);
        r.routed = try gpa.alloc(u16, max_picks);
        errdefer gpa.free(r.routed);
        r.misses = try gpa.alloc(u16, uniq);
        errdefer gpa.free(r.misses);
        r.snap = try gpa.alloc(u16, lines);
        errdefer gpa.free(r.snap);
        r.keep = try gpa.alloc(u16, uniq + lines);
        r.reset();
        return r;
    }

    pub fn deinit(r: *Round, gpa: std.mem.Allocator) void {
        gpa.free(r.ids);
        gpa.free(r.lines);
        gpa.free(r.copies);
        gpa.free(r.passes);
        gpa.free(r.routed);
        gpa.free(r.misses);
        gpa.free(r.snap);
        gpa.free(r.keep);
        r.* = undefined;
    }

    fn reset(r: *Round) void {
        r.n_ids = 0;
        r.n_copies = 0;
        r.n_passes = 0;
        r.n_prefetch = 0;
        r.hit_slots = 0;
        r.miss_slots = 0;
        r.shared_slots = 0;
        r.unique = 0;
        r.held = 0;
        r.pending = 0;
        r.evictions = 0;
        r.protected = false;
    }

    pub fn passIds(r: *const Round, j: usize) []const u16 {
        const p = r.passes[j];
        return r.ids[p.id_lo..p.id_hi];
    }

    pub fn passLines(r: *const Round, j: usize) []const u32 {
        const p = r.passes[j];
        return r.lines[p.id_lo..p.id_hi];
    }

    pub fn passCopies(r: *const Round, j: usize) []const Copy {
        const p = r.passes[j];
        return r.copies[p.copy_lo..p.copy_hi];
    }

    /// Copies no pass waits for (swap-round prefetch); their lines stay in flight until reaped.
    pub fn prefetch(r: *const Round) []const Copy {
        return r.copies[r.n_copies - r.n_prefetch .. r.n_copies];
    }

    /// Mandatory copies of all passes.
    pub fn misses_(r: *const Round) usize {
        return r.n_copies - r.n_prefetch;
    }

    /// More than one pass: each pass reads its own masked picks (picks of other passes' experts set to -1).
    pub fn masked(r: *const Round) bool {
        return r.n_passes > 1;
    }

    /// True when the round needs no copy at all (the experts need no upload).
    pub fn allHits(r: *const Round) bool {
        return r.n_copies == 0;
    }
};

fn contains32(xs: []const u32, x: u32) bool {
    return std.mem.indexOfScalar(u32, xs, x) != null;
}

fn contains(xs: []const u16, id: u16) bool {
    return std.mem.indexOfScalar(u16, xs, id) != null;
}

const Order = struct {
    counts: []const u32,
    ids: []const u16,
    fn appearances(ids: []const u16, id: u16) u32 {
        var n: u32 = 0;
        for (ids) |x| n += @intFromBool(x == id);
        return n;
    }
    /// Lifetime count descending, then this round's appearances, then lower id (the policy's `usedFirst`).
    fn hotter(o: Order, a: u16, b: u16) bool {
        if (o.counts[a] != o.counts[b]) return o.counts[a] > o.counts[b];
        const aa = appearances(o.ids, a);
        const ab = appearances(o.ids, b);
        if (aa != ab) return aa > ab;
        return a < b;
    }
};

/// Lines whose copy is in flight and whose expert is not in `use`: they can be neither read nor evicted this pass.
fn blocked(v: Vram, use: []const u16) usize {
    var n: usize = 0;
    for (v.line, v.slots) |state, id| {
        if (state == .inflight and !contains(use, id)) n += 1;
    }
    return n;
}

fn snapshot(v: Vram, out: []u16) void {
    for (v.slots, v.line, out) |id, state, *o| o.* = if (state == .empty) empty_id else id;
}

/// Plans one round of `layer` (see the file header).
pub fn plan(
    v: *Vram,
    layer: u8,
    picks: []const i32,
    counts: *[shape.layers][shape.experts]u32,
    now: u32,
    round: u32,
    opt: Options,
    r: *Round,
) error{ Busy, Length }!void {
    r.reset();
    if (picks.len > r.routed.len or layer >= shape.layers or v.slots.len != r.snap.len) return error.Length;
    // the routed picks and the distinct experts, in first-appearance order
    var n_routed: usize = 0;
    var n_unique: usize = 0;
    for (picks) |p| {
        if (p < 0) continue;
        if (p > shared_id) return error.Length;
        const id: u16 = @intCast(p);
        if (id == shared_id) {
            r.shared_slots += 1;
            continue;
        }
        r.routed[n_routed] = id;
        n_routed += 1;
        if (!contains(r.ids[0..n_unique], id)) {
            r.ids[n_unique] = id;
            n_unique += 1;
        }
    }
    const routed = r.routed[0..n_routed];
    const uniq = r.ids[0..n_unique];
    // capacity check before anything is written: pass 0 holds the held experts, so at least one more line.
    var held_n: usize = 0;
    for (uniq) |id| held_n += @intFromBool(offload.lineOf(v.*, layer, id) != null);
    if (n_unique > held_n and blocked(v.*, &.{}) >= v.slots.len) return error.Busy;

    if (opt.observe) offload.observe(&counts[layer], routed) catch return error.Length;
    const row: []const u32 = &counts[layer];

    // held first (pass 0), misses hottest first; ids[] is rewritten in pass order
    var n_held: usize = 0;
    var n_miss: usize = 0;
    for (uniq) |id| {
        if (offload.lineOf(v.*, layer, id)) |line| {
            r.held += 1;
            if (v.line[line] == .inflight) r.pending += 1 else _ = offload.touchAt(v, layer, id, now);
        } else {
            r.misses[n_miss] = id;
            n_miss += 1;
        }
    }
    for (routed) |id| {
        if (offload.lineOf(v.*, layer, id) != null) r.hit_slots += 1 else r.miss_slots += 1;
    }
    r.unique = @intCast(n_unique);
    // pack held ids at the front of ids (stable), then the sorted misses
    for (uniq) |id| {
        if (offload.lineOf(v.*, layer, id) != null) {
            r.ids[n_held] = id;
            n_held += 1;
        }
    }
    const misses = r.misses[0..n_miss];
    std.mem.sort(u16, misses, Order{ .counts = row, .ids = routed }, Order.hotter);
    @memcpy(r.ids[n_held..][0..n_miss], misses);

    // passes
    var at: usize = 0; // next id of ids[] to place
    var mi: usize = 0; // next miss
    while (at < n_unique) {
        const first = r.n_passes == 0;
        const lo = at;
        const take_held: usize = if (first) n_held else 0;
        at += take_held;
        // lines usable this pass: all but in-flight lines of experts outside this pass
        const usable = v.slots.len - blocked(v.*, r.ids[lo..at]);
        const room = usable - take_held;
        const n_new = @min(room, n_miss - mi);
        std.debug.assert(n_new > 0 or take_held > 0); // the capacity check above
        const hi = at + n_new;
        // double buffering: also spare the protected experts' resident lines when the single pass still fits
        var keep: []const u16 = r.ids[lo..hi];
        if (first and n_new == n_miss and opt.protect.len > 0) {
            var spare: usize = 0;
            var k: usize = hi - lo;
            @memcpy(r.keep[0..k], r.ids[lo..hi]);
            for (opt.protect) |id| {
                if (contains(r.keep[0..k], id) or k == r.keep.len) continue;
                const line = offload.lineOf(v.*, layer, id) orelse continue;
                if (v.line[line] != .resident) continue;
                r.keep[k] = id;
                k += 1;
                spare += 1;
            }
            if (room >= n_new + spare) {
                keep = r.keep[0..k];
                r.protected = true;
            }
        }
        snapshot(v.*, r.snap);
        const copy_lo = r.n_copies;
        for (r.ids[at..hi]) |id| {
            const line = offload.stageUsedAt(v, layer, id, keep, counts) catch return error.Busy;
            const old = r.snap[line];
            if (old != empty_id) r.evictions += 1;
            r.copies[r.n_copies] = .{ .id = id, .line = @intCast(line), .evicted = old };
            r.n_copies += 1;
        }
        // the device orders the next pass's copies after this pass's compute: these lines are resident from here on
        for (r.copies[copy_lo..r.n_copies]) |c| offload.reapAt(v, layer, c.id, now) catch unreachable;
        for (r.ids[lo..hi], r.lines[lo..hi]) |id, *ln| ln.* = @intCast(offload.lineOf(v.*, layer, id).?);
        r.passes[r.n_passes] = .{ .id_lo = @intCast(lo), .id_hi = @intCast(hi), .copy_lo = @intCast(copy_lo), .copy_hi = @intCast(r.n_copies) };
        r.n_passes += 1;
        at = hi;
        mi += n_new;
    }
    r.n_ids = n_unique;
    if (r.n_passes == 0) { // shared and -1 picks only: one pass with no routed expert (the shared expert still runs)
        r.passes[0] = .{ .id_lo = 0, .id_hi = 0, .copy_lo = 0, .copy_hi = 0 };
        r.n_passes = 1;
    }

    // swap-round prefetch: hottest non-held experts into empty or strictly colder lines this round does not use
    if (opt.swap_every == 0 or round == 0 or round % opt.swap_every != 0 or opt.prefetch_bytes == 0 or opt.line_bytes == 0) return;
    const budget = @min(@as(u64, opt.swap_cap), opt.prefetch_bytes / opt.line_bytes);
    var staged: u64 = 0;
    snapshot(v.*, r.snap);
    while (staged < budget) {
        // hottest id not held on this layer
        var best: ?u16 = null;
        var id: u16 = 0;
        while (id < shape.experts) : (id += 1) {
            if (row[id] == 0 or offload.lineOf(v.*, layer, id) != null) continue;
            if (best == null or row[id] > row[best.?]) best = id;
        }
        const cand = best orelse break;
        // the line it would take: an empty one, else the coldest resident outside this round
        var victim: ?usize = null;
        for (v.line, 0..) |state, i| {
            if (state == .empty) {
                victim = i;
                break;
            }
        }
        if (victim == null) {
            for (v.line, v.slots, 0..) |state, held_id, i| {
                if (state != .resident or contains(uniq_view(r), held_id)) continue;
                if (victim == null or colder(v.*, layer, counts, i, victim.?)) victim = i;
            }
            const vi = victim orelse break;
            if (offload.countAt(counts, layer, v.slots[vi]) >= row[cand]) break; // nothing colder than the candidate
        }
        const line = offload.stageUsedAt(v, layer, cand, uniq_view(r), counts) catch break;
        std.debug.assert(line == victim.?);
        const old = r.snap[line];
        if (old != empty_id) r.evictions += 1;
        r.snap[line] = cand;
        r.copies[r.n_copies] = .{ .id = cand, .line = @intCast(line), .evicted = old };
        r.n_copies += 1;
        r.n_prefetch += 1;
        staged += 1;
    }
}

fn uniq_view(r: *const Round) []const u16 {
    return r.ids[0..r.n_ids];
}

/// The policy's victim order (`vramCold`): lower lifetime count, then older stamp, then the higher line index.
fn colder(v: Vram, layer: u8, counts: *const [shape.layers][shape.experts]u32, i: usize, b: usize) bool {
    const ci = offload.countAt(counts, layer, v.slots[i]);
    const cb = offload.countAt(counts, layer, v.slots[b]);
    return ci < cb or (ci == cb and v.stamp[i] < v.stamp[b]) or (ci == cb and v.stamp[i] == v.stamp[b] and i > b);
}

/// Smallest pool that never needs a second pass for `rows` decode rows: every row's routed picks must fit at once.
pub fn minLines(top_k: usize) usize {
    return top_k;
}

/// Lines a layer gets from `free` device bytes after `reserve` and `other` (KV, scratch: the family's budget)
pub fn linesFor(free: u64, reserve: u64, other: u64, line_bytes: u64, layers: usize, min_lines: usize) error{PoolTooSmall}!usize {
    if (line_bytes == 0 or layers == 0) return error.PoolTooSmall;
    const spend = std.math.sub(u64, free, reserve +| other) catch return error.PoolTooSmall;
    const per = spend / (line_bytes * layers);
    if (per < min_lines) return error.PoolTooSmall;
    return @intCast(per);
}

test {
    _ = offload;
}

// ------------------------------------------------------------------------------------------------ tests

const testing = std.testing;
const Line = offload.Line;

const Fix = struct {
    slots: [16]u16 = undefined,
    line: [16]Line = undefined,
    stamp: [16]u32 = undefined,
    lay: [16]u8 = undefined,
    v: Vram = undefined,
    counts: [shape.layers][shape.experts]u32 = @splat(@splat(0)),
    r: Round = undefined,

    fn init(f: *Fix, lines: usize, max_picks: usize) !void {
        f.v = Vram.init(f.slots[0..lines], f.line[0..lines], f.stamp[0..lines], f.lay[0..lines]);
        f.counts = @splat(@splat(0));
        f.r = try Round.init(testing.allocator, lines, max_picks);
    }
    fn deinit(f: *Fix) void {
        f.r.deinit(testing.allocator);
    }
    fn go(f: *Fix, picks: []const i32, now: u32, round: u32, opt: Options) !void {
        return plan(&f.v, 3, picks, &f.counts, now, round, opt, &f.r);
    }
    fn resident(f: *const Fix, id: u16) bool {
        return offload.residentAt(f.v, 3, id);
    }
};

test "misses fill empty lines in one pass, then the same picks are all hits" {
    var f: Fix = .{};
    try f.init(10, 32);
    defer f.deinit();
    const picks = [_]i32{ 5, 9, 5, 512, -1, 7 };
    try f.go(&picks, 1, 1, .{});
    try testing.expectEqual(@as(usize, 1), f.r.n_passes);
    try testing.expectEqual(@as(usize, 3), f.r.misses_());
    try testing.expectEqual(@as(u32, 4), f.r.miss_slots);
    try testing.expectEqual(@as(u32, 1), f.r.shared_slots);
    try testing.expectEqual(@as(u32, 0), f.r.evictions);
    for (f.r.passCopies(0)) |c| try testing.expectEqual(empty_id, c.evicted);
    // 5 appears twice: hottest miss, copied first
    try testing.expectEqual(@as(u16, 5), f.r.passCopies(0)[0].id);
    try testing.expect(!f.r.masked());
    try testing.expect(f.resident(5) and f.resident(9) and f.resident(7));
    try testing.expectEqual(@as(u32, 2), f.counts[3][5]);

    try f.go(&picks, 2, 2, .{});
    try testing.expect(f.r.allHits());
    try testing.expectEqual(@as(u32, 4), f.r.hit_slots);
    try testing.expectEqual(@as(u32, 3), f.r.held);
    try testing.expectEqual(@as(usize, 1), f.r.n_passes);
    try testing.expectEqual(@as(usize, 3), f.r.passIds(0).len);
    // a hit touched its line
    try testing.expectEqual(@as(u32, 2), f.stamp[offload.lineOf(f.v, 3, 9).?]);
}

test "synthetic prefill picks do not heat expert history" {
    var f: Fix = .{};
    try f.init(10, 16);
    defer f.deinit();
    try f.go(&[_]i32{ 7, 8, 9 }, 1, 1, .{ .observe = false });
    try testing.expectEqual(@as(usize, 3), f.r.misses_());
    try testing.expectEqual(@as(u32, 0), f.counts[3][7]);
    try testing.expectEqual(@as(u32, 0), f.counts[3][8]);
    try testing.expectEqual(@as(u32, 0), f.counts[3][9]);
}

test "a miss evicts the coldest line outside the round, never a line the round reads" {
    var f: Fix = .{};
    try f.init(3, 16);
    defer f.deinit();
    try f.go(&[_]i32{ 1, 2, 3 }, 1, 1, .{});
    // heat 1 and 3, leave 2 cold
    try f.go(&[_]i32{ 1, 3 }, 2, 2, .{});
    try f.go(&[_]i32{ 1, 3 }, 3, 3, .{});
    try f.go(&[_]i32{ 1, 4 }, 4, 4, .{});
    try testing.expectEqual(@as(usize, 1), f.r.misses_());
    const c = f.r.passCopies(0)[0];
    try testing.expectEqual(@as(u16, 4), c.id);
    try testing.expectEqual(@as(u16, 2), c.evicted);
    try testing.expect(f.resident(1) and f.resident(3) and f.resident(4) and !f.resident(2));
    // 3 is hot but not in this round, 1 is in it; the cold one now is 4 (count 1) vs 3 (count 3) -> 4 goes
    try f.go(&[_]i32{ 1, 3, 6 }, 5, 5, .{});
    try testing.expectEqual(@as(u16, 4), f.r.passCopies(0)[0].evicted);
}

test "more distinct experts than lines split into passes that each fit, every expert computed once" {
    var f: Fix = .{};
    try f.init(4, 64);
    defer f.deinit();
    try f.go(&[_]i32{ 20, 21 }, 1, 1, .{});
    var picks: [30]i32 = undefined;
    for (&picks, 0..) |*p, i| p.* = @intCast(10 + (i % 11)); // 10..20, 11 distinct; 20 held
    picks[29] = 21; // held too
    try f.go(&picks, 2, 2, .{});
    try testing.expect(f.r.masked());
    try testing.expectEqual(@as(u32, 12), f.r.unique);
    try testing.expectEqual(@as(u32, 2), f.r.held);
    try testing.expectEqual(@as(usize, 10), f.r.misses_());
    var seen: [shape.experts]u8 = @splat(0);
    for (0..f.r.n_passes) |j| {
        const ids = f.r.passIds(j);
        try testing.expect(ids.len >= 1 and ids.len <= 4);
        // the lines a pass copies into are distinct
        const cs = f.r.passCopies(j);
        for (cs, 0..) |a, i| for (cs[i + 1 ..]) |b| try testing.expect(a.line != b.line);
        for (ids) |id| seen[id] += 1;
    }
    for (10..22) |id| try testing.expectEqual(@as(u8, 1), seen[id]);
    // each pass reads distinct lines; a copy of a later pass may land in a line an earlier pass read
    var reused = false;
    for (0..f.r.n_passes) |j| {
        const ls = f.r.passLines(j);
        for (ls, 0..) |a, i| for (ls[i + 1 ..]) |b| try testing.expect(a != b);
        for (f.r.passCopies(j)) |c| {
            try testing.expect(contains32(ls, c.line));
            if (j > 0) for (0..j) |k| {
                if (contains32(f.r.passLines(k), c.line)) reused = true;
            };
        }
    }
    try testing.expect(reused);
    // pass 0 holds the two held experts first
    try testing.expect(contains(f.r.passIds(0), 20) and contains(f.r.passIds(0), 21));
    try testing.expectEqual(@as(usize, 3), f.r.n_passes); // 2 held + 2 misses, then 4, then 4
    // the last pass's experts are resident after the round
    for (f.r.passIds(f.r.n_passes - 1)) |id| try testing.expect(f.resident(id));
}

test "an in-flight prefetch line is not a victim and counts as held when the round asks for it" {
    var f: Fix = .{};
    try f.init(3, 16);
    defer f.deinit();
    try f.go(&[_]i32{ 1, 2, 3 }, 1, 1, .{});
    // a prefetch of 9 into line of 2 (made by hand: stage it, leave it in flight)
    const line = offload.lineOf(f.v, 3, 2).?;
    f.slots[line] = 9;
    f.line[line] = .inflight;
    try f.go(&[_]i32{ 1, 4 }, 2, 2, .{});
    // 4 cannot take 9's in-flight line; it takes 3's (not in the round)
    try testing.expectEqual(@as(u16, 3), f.r.passCopies(0)[0].evicted);
    try testing.expectEqual(Line.inflight, f.line[line]);
    try f.go(&[_]i32{ 9, 1 }, 3, 3, .{});
    try testing.expect(f.r.allHits());
    try testing.expectEqual(@as(u32, 1), f.r.pending);
    // every line in flight for another expert: busy, and nothing written
    for (f.line[0..3]) |*l| l.* = .inflight;
    const before = f.counts[3][7];
    try testing.expectError(error.Busy, f.go(&[_]i32{7}, 4, 4, .{}));
    try testing.expectEqual(before, f.counts[3][7]);
}

test "double buffering spares the previous round's lines when the round fits without them" {
    var f: Fix = .{};
    try f.init(6, 16);
    defer f.deinit();
    // previous round 1,2,3 (hot), older residents 4,5 (cold), one empty line
    try f.go(&[_]i32{ 4, 5 }, 1, 1, .{});
    for (0..3) |i| try f.go(&[_]i32{ 1, 2, 3 }, @intCast(2 + i), @intCast(2 + i), .{});
    // two misses: with 1, 2, 3 spared they go to the empty line and the colder of 4, 5
    const prev = [_]u16{ 1, 2, 3 };
    try f.go(&[_]i32{ 7, 8 }, 5, 5, .{ .protect = &prev });
    try testing.expect(f.r.protected);
    for (f.r.passCopies(0)) |c| try testing.expect(!contains(&prev, c.evicted));
    try testing.expect(f.resident(1) and f.resident(2) and f.resident(3));
    // a round that only fits by evicting a protected line ignores the protection
    try f.go(&[_]i32{ 9, 10, 11, 12 }, 6, 6, .{ .protect = &[_]u16{ 7, 8, 1, 2, 3 } });
    try testing.expect(!f.r.protected);
    try testing.expectEqual(@as(usize, 1), f.r.n_passes);
}

test "the shared expert and -1 are never staged; a pick above 512 is refused" {
    var f: Fix = .{};
    try f.init(10, 16);
    defer f.deinit();
    try f.go(&[_]i32{ 512, -1, 512 }, 1, 1, .{});
    try testing.expectEqual(@as(usize, 0), f.r.n_copies);
    try testing.expectEqual(@as(u32, 2), f.r.shared_slots);
    try testing.expectEqual(@as(usize, 1), f.r.n_passes);
    try testing.expectEqual(@as(usize, 0), f.r.passIds(0).len);
    for (f.slots[0..10]) |id| try testing.expect(id != shared_id);
    try testing.expectError(error.Length, f.go(&[_]i32{513}, 2, 2, .{}));
}

test "swap rounds prefetch the hottest non-held experts within the copy budget, only into colder lines" {
    var f: Fix = .{};
    try f.init(4, 16);
    defer f.deinit();
    const opt: Options = .{ .swap_every = 4, .prefetch_bytes = 200, .line_bytes = 100 };
    // lifetime counts: 30 hot (5), 31 warm (3), never in a line yet
    f.counts[3][30] = 5;
    f.counts[3][31] = 3;
    f.counts[3][32] = 1;
    try f.go(&[_]i32{ 1, 2 }, 1, 1, opt);
    try testing.expectEqual(@as(usize, 0), f.r.n_prefetch); // round 1: not a swap round
    try f.go(&[_]i32{ 1, 2 }, 4, 4, opt);
    // two empty lines, budget 2: 30 and 31 in flight
    try testing.expectEqual(@as(usize, 2), f.r.n_prefetch);
    try testing.expectEqual(@as(u16, 30), f.r.prefetch()[0].id);
    try testing.expectEqual(@as(u16, 31), f.r.prefetch()[1].id);
    try testing.expectEqual(Line.inflight, f.line[f.r.prefetch()[0].line]);
    try testing.expect(!f.resident(30));
    try offload.reapAt(&f.v, 3, 30, 5);
    try offload.reapAt(&f.v, 3, 31, 5);
    // round 8: 32 (count 1) is not hotter than any resident outside the round (30: 5, 31: 3) -> nothing evicted
    try f.go(&[_]i32{ 1, 2 }, 8, 8, opt);
    try testing.expectEqual(@as(usize, 0), f.r.n_prefetch);
    // heat 32 above 31: it replaces 31, never 1 or 2 (in this round)
    f.counts[3][32] = 9;
    try f.go(&[_]i32{ 1, 2 }, 12, 12, opt);
    try testing.expectEqual(@as(usize, 1), f.r.n_prefetch);
    try testing.expectEqual(@as(u16, 31), f.r.prefetch()[0].evicted);
}

test "line budget: free memory after reserve and the family's other bytes, at least min lines a layer" {
    const line: u64 = 1_843_200;
    // 12 GiB free, 1 GiB reserve, 5.5 GiB for KV and scratch: 5.5 GiB / (48 x 1,843,200 B) = 66 lines a layer
    try testing.expectEqual(@as(usize, 66), try linesFor(12 << 30, 1 << 30, (11 << 30) / 2, line, 48, 10));
    try testing.expectEqual(@as(usize, 10), try linesFor(10 * 48 * line + 100, 100, 0, line, 48, 10));
    try testing.expectError(error.PoolTooSmall, linesFor(10 * 48 * line - 1, 0, 0, line, 48, 10));
    try testing.expectError(error.PoolTooSmall, linesFor(1 << 30, 2 << 30, 0, line, 48, 10));
}
