//! Price a lone window from its own complete rounds, retaining startup tile steps between measured widths.
const std = @import("std");
const Table = @import("table.zig").Table;

fn usable(ms: ?f64) bool {
    const x = ms orelse return false;
    return x > 0 and std.math.isFinite(x);
}

fn observed(s: anytype) Table {
    const state = s.depth orelse return .{};
    return state.ms orelse .{};
}

pub fn enabled(e: anytype, p: anytype) bool {
    return e.alone and e.cfg.lone_cost_tile > 0 and e.cfg.timed and e.cfg.plain_guard and
        e.cfg.draft_confidence == null and p.kind == .head and p.lanes == null;
}

/// The rule's view of a stream, marked as a lone round priced from its own times or not.
pub fn marked(who: anytype, lone: bool) @TypeOf(who) {
    var w = who;
    w.lone = lone;
    return w;
}

/// Widths sampled before a line over rows is fitted: with fewer, a width keeps its own sample or its nearest one.
const fit_least = 3;

/// A line over rows through the samples beyond the startup model: one noisy timing moves no width's price alone.
fn fitted(startup: Table, samples: Table, rows: usize) ?f64 {
    var n: f64 = 0;
    var sx: f64 = 0;
    var sy: f64 = 0;
    var sxx: f64 = 0;
    var sxy: f64 = 0;
    for (samples.slots, 1..) |ms, width| {
        if (!usable(ms)) continue;
        const model = startup.get(@intCast(width)) orelse continue;
        if (!usable(model)) continue;
        const x: f64 = @floatFromInt(width);
        const y = ms.? - model;
        n += 1;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const spread = sxx - sx * sx / n;
    if (n < fit_least or spread <= 0) return null;
    const slope = (sxy - sx * sy / n) / spread;
    const base = startup.get(@intCast(rows)) orelse return null;
    const price = base + (sy - slope * sx) / n + slope * @as(f64, @floatFromInt(rows));
    return if (usable(price)) price else null;
}

/// Measured full-round time at this width; otherwise the nearest sample plus the startup forward-time difference.
pub fn estimate(startup: Table, samples: Table, rows: usize, overhead: f64) ?f64 {
    if (usable(samples.get(@intCast(rows - 1)))) return samples.get(@intCast(rows - 1));
    const base = startup.get(@intCast(rows)) orelse return null;
    var nearest: ?usize = null;
    var distance: usize = std.math.maxInt(usize);
    for (samples.slots, 1..) |ms, width| {
        if (!usable(ms) or !usable(startup.get(@intCast(width)))) continue;
        const delta = if (width > rows) width - rows else rows - width;
        if (delta < distance) {
            nearest = width;
            distance = delta;
        }
    }
    if (nearest) |width| {
        const price = samples.get(@intCast(width - 1)).? + base - startup.get(@intCast(width)).?;
        if (usable(price)) return price;
    }
    return base + overhead;
}

/// `estimate`, or with `line` (a lone XPU tree; the engine keeps each width's own timing) the fitted time when sampled.
pub fn priceOf(startup: Table, samples: Table, rows: usize, overhead: f64, line: bool) ?f64 {
    if (line) if (fitted(startup, samples, rows)) |ms| return ms;
    return estimate(startup, samples, rows, overhead);
}

/// Complete-round prices in the caller's allocator (an arena or a fixed buffer, never freed), with no added overhead.
pub fn prices(e: anytype, a: std.mem.Allocator, p: anytype, line: bool) !Table {
    var out: Table = .{};
    const overhead = e.rule.overheadMs(1, p.stream.sampling != null);
    const samples = observed(p.stream);
    for (1..p.count() + 2) |rows| {
        if (priceOf(e.cfg.shared_costs, samples, rows, overhead, line)) |ms|
            try out.put(a, @intCast(rows), ms);
    }
    return out;
}

/// Keep overdue calibration through short offers; space later probes from the completed trial.
pub fn probe(e: anytype, p: anytype) ?u32 {
    const s = p.stream;
    if (s.rounds < s.cost_probe_after) return null;
    const samples = observed(s);
    const most: u32 = @intCast(@min(p.count() + 1, e.cfg.batch_rows));
    var retry: ?u32 = null;
    var earliest: u64 = std.math.maxInt(u64);
    // without plain_guard, when the plan offers a draft, no round is drafted with nothing
    var rows: u32 = @min(@as(u32, if (e.cfg.plain_guard) 1 else 2), most);
    while (true) {
        if (rows < s.cost_probes.len) {
            const cd = s.cost_probes[rows];
            if (s.rounds >= cd.after) {
                if (!usable(samples.get(rows - 1))) return rows - 1;
                if (cd.retry and (retry == null or cd.after < earliest)) {
                    retry = rows - 1;
                    earliest = cd.after;
                }
            }
        }
        if (rows == most) return retry;
        rows = @min(most, if (rows == 1) @max(2, e.cfg.lone_cost_tile) else rows +| e.cfg.lone_cost_tile);
    }
}

pub const Cooldown = struct { after: u64 = 8, spacing: u64 = 8, retry: bool = false };

/// Kept calibrations a context band: a long prompt's round times would hold a short stream off its wide windows.
pub const bands = 6;

/// A context's band: under 8,192 tokens, then one a doubling, the last from 131,072 on.
pub fn band(len: usize) usize {
    return if (len < 8192) 0 else @min(bands - 1, 1 + std.math.log2_int(usize, len / 8192));
}

/// The furthest a kept deadline or spacing reaches into the next stream: carried doublings would end every re-check.
pub const carry_most: u64 = 256;

/// The kept calibration a stream starts from and leaves: its sampling mode's, at its prompt's band, whatever its reply.
pub fn slot(kept: *[2][bands]?Kept, s: anytype) *?Kept {
    return &kept[@intFromBool(s.sampling != null)][band(s.prompt_len)];
}

/// A finished lone stream's calibration, so the next stream starts priced instead of probing every width again.
pub const Kept = struct {
    ms: Table = .{},
    copy_ms: Table = .{}, // a lone stream's copy rounds' times (keepCopies)
    copied_landed: [2]f64 = .{ -1, -1 }, // and its copies' landing history, by source (Stream.copied_landed)
    copied_broke: [2]f64 = .{ 0, 0 },
    probes: [129]Cooldown = @splat(.{}),
    after: u64 = 8,
    spacing: u64 = 8,
    rounds: u64 = 0, // the stream's rounds when kept: its deadlines rebase from here

    pub fn deinit(k: *Kept, gpa: std.mem.Allocator) void {
        k.ms.deinit(gpa);
        k.copy_ms.deinit(gpa);
    }
};

/// Keep a stream's prices and deadlines when every round it ran was its own (shared rounds price others' rows).
pub fn keep(gpa: std.mem.Allocator, kept: *?Kept, s: anytype) !void {
    if (s.shared_round) return;
    const ms = observed(s);
    if (ms.empty()) return;
    const slots = try gpa.dupe(?f64, ms.slots);
    if (kept.*) |*k| k.deinit(gpa);
    kept.* = .{ .ms = .{ .slots = slots }, .probes = s.cost_probes, .after = s.cost_probe_after, .spacing = s.cost_probe_spacing, .rounds = s.rounds };
}

/// Start a stream's own prices from the newest kept calibration, the deadlines counted from its first round.
pub fn seed(gpa: std.mem.Allocator, kept: ?Kept, s: anytype) !void {
    const k = kept orelse return;
    const state = if (s.depth) |*d| d else return;
    if (state.ms) |*own| {
        if (!own.empty()) return;
        own.deinit(gpa);
    }
    state.ms = .{ .slots = try gpa.dupe(?f64, k.ms.slots) };
    for (&s.cost_probes, k.probes) |*cd, old| cd.* = .{ .after = s.rounds + @min(carry_most, old.after -| k.rounds), .spacing = @min(carry_most, old.spacing), .retry = old.retry };
    s.cost_probe_after = s.rounds + @min(carry_most, k.after -| k.rounds);
    s.cost_probe_spacing = @min(carry_most, k.spacing);
}

pub const Pick = struct { counts: []u32, prices: Table, trial: ?u32 };

/// A lone round's drafts, for the engine and a backend alike, after `prepare`: price each width, pick or trial one.
pub fn choose(e: anytype, a: std.mem.Allocator, p: anytype, chances: []const f64, least_rows: u32, line: bool) !Pick {
    const priced = try prices(e, a, p, line);
    const counts = try @import("allocate.zig").allocateDrafts(a, &.{1}, &.{chances}, &priced, 0, @max(1, e.cfg.batch_rows), .{ .least_rows = least_rows });
    var out: Pick = .{ .counts = counts, .prices = priced, .trial = null };
    if (probe(e, p)) |drafts| {
        counts[0] = drafts;
        out.trial = drafts + 1;
        p.stream.cost_probe_trial = trial(p.stream, chances, drafts + 1);
    }
    return out;
}

/// A lone stream's copy prices (lanes/lone_copy.zig), kept as keep keeps a head's; `own_plain`: its plain times too.
pub fn keepCopies(gpa: std.mem.Allocator, kept: *?Kept, s: anytype, own_plain: bool) !void {
    if (s.shared_round) return;
    const state = s.depth orelse return;
    const copies = state.copy_ms orelse return;
    if (copies.empty()) return;
    const copy_ms: Table = .{ .slots = try gpa.dupe(?f64, copies.slots) };
    errdefer gpa.free(copy_ms.slots);
    const ms: Table = if (own_plain) .{ .slots = try gpa.dupe(?f64, observed(s).slots) } else .{};
    if (kept.*) |*k| k.deinit(gpa);
    kept.* = .{ .ms = ms, .copy_ms = copy_ms, .copied_landed = s.copied_landed, .copied_broke = s.copied_broke, .rounds = s.rounds };
}

/// Start a stream's copy prices from the newest kept ones, each table only while the stream has timed none of its own.
pub fn seedCopies(gpa: std.mem.Allocator, kept: ?Kept, s: anytype) !void {
    const k = kept orelse return;
    const state = if (s.depth) |*d| d else return;
    try fill(gpa, &state.copy_ms, k.copy_ms);
    try fill(gpa, &state.ms, k.ms);
    // the landing rate a copy's rows are priced by, a source's only while the stream has seen none of its own copies
    for (&s.copied_landed, &s.copied_broke, k.copied_landed, k.copied_broke) |*landed, *broke, was, broken| {
        if (landed.* >= 0 or was < 0) continue;
        landed.* = was;
        broke.* = broken;
    }
}

fn fill(gpa: std.mem.Allocator, own: *?Table, from: Table) !void {
    if (from.empty()) return;
    if (own.*) |*t| {
        if (!t.empty()) return;
        t.deinit(gpa);
    }
    own.* = .{ .slots = try gpa.dupe(?f64, from.slots) };
}

pub const Trial = struct {
    rows: u32,
    expected: f64,
    best: ?struct { expected: f64, ms: f64 } = null,
};

/// Snapshot landing probabilities and only this stream's measured, feasible alternatives before forcing a trial.
pub fn trial(s: anytype, chances: []const f64, rows: u32) Trial {
    var out: Trial = .{ .rows = rows, .expected = 1 };
    const samples = observed(s);
    var expected: f64 = 1;
    for (0..chances.len + 1) |i| {
        if (i > 0) expected += chances[i - 1];
        if (i + 1 == rows) out.expected = expected;
        const ms = samples.get(@intCast(i));
        if (i + 1 == rows or !usable(ms)) continue;
        if (out.best == null or expected / ms.? > out.best.?.expected / out.best.?.ms)
            out.best = .{ .expected = expected, .ms = ms.? };
    }
    return out;
}

/// A completed trial resets its own sample; losing widths retry after doubling and an expected repayment bound.
pub fn observe(s: anytype, requested: ?u32, rows: usize, ms: f64) void {
    if (rows == 0 or rows >= s.cost_probes.len or !usable(ms)) return;
    const samples = observed(s);
    if (rows > samples.slots.len) return;
    const cd = &s.cost_probes[rows];
    if (requested == null) {
        // Active widths refresh their age; a width abandoned after an ordinary bad sample remains retryable.
        cd.after = s.rounds +| cd.spacing;
        cd.retry = true;
        return;
    }
    if (requested) |width| if (width == rows and rows < s.cost_probes.len) {
        if (s.cost_probe_trial) |t| {
            if (t.rows != rows) return;
            const previous = cd.spacing;
            cd.spacing = 8;
            cd.retry = false;
            if (t.best) |best| {
                const expected_saving = best.expected * ms / t.expected - best.ms;
                if (expected_saving > 0) {
                    const repay = @ceil(ms / expected_saving);
                    const limit: f64 = @floatFromInt(std.math.maxInt(u64));
                    const spacing: u64 = if (repay >= limit) std.math.maxInt(u64) else @intFromFloat(repay);
                    cd.spacing = @max(previous *| 2, spacing);
                    cd.retry = true;
                }
            }
        }
        samples.slots[rows - 1] = ms;
        cd.after = s.rounds +| cd.spacing;
        s.cost_probe_trial = null;
        s.cost_probe_spacing = cd.spacing;
        s.cost_probe_after = cd.after;
    };
}
