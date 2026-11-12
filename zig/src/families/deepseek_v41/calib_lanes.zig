//! The boot calibration over any backend: the measurement and its storage given a Target, a Pass and a clock.
const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const iface = @import("draft/iface.zig");
const cm = @import("draft/calib_measure.zig");
const calib_env = @import("draft/calib_env.zig");
const costs_mod = @import("draft/costs.zig");
const dspark = @import("draft/dspark.zig");
const ph = @import("phases.zig");
const tokenizer = @import("tokenizer");
const Allocator = std.mem.Allocator;

const log = std.log.scoped(.dsv41);

/// Untimed and timed passes run back to back for the pass's entry.
const steady_warm = 1;
const steady_reps = 5;

/// Prompt rows prefilled in the first slot.
pub const prefix_rows = 4;

pub const Measured = struct {
    costs: costs_mod.Costs,
    /// rank 0's rows 1..rows as kept (the lower of sweep and recheck), ms
    kept: [cm.max_rows]f64 = @splat(0.0),
    rows: u32 = 0,
    /// slots the windows used, timed runs
    n: u32 = 0,
    runs: usize = 0,
    /// runs timed (the plan's, less those a spent budget dropped, and untimed warm-ups)
    timed: usize = 0,
    /// the drafter's ingest of a few committed rows within the pass, fastest timed run, ms (0: not timed apart)
    ingest_ms: f64 = 0.0,
    /// the same ingest the first time after the windows, ms
    ingest_first_ms: f64 = 0.0,
    /// prompt rows run into slots, and the windows (untimed warm-ups included) the measurement ran
    prefill_rows: u32 = 0,
    windows: usize = 0,
    /// seconds in the prompt's prefills and in the timed runs
    prefill_s: f64 = 0.0,
    runs_s: f64 = 0.0,
};

/// `boot`'s result: the measured table and the stored entry's path (null when nothing was stored).
pub const Boot = struct { costs: costs_mod.Costs, path: ?[]u8 };

/// A backend's calibration hooks: its target and pass, plain numbers, every rank's clock and the slot chores.
pub const Device = struct {
    ctx: *anyopaque,
    target: iface.Target,
    pass: ?iface.Pass,
    slots: u32,
    limit: u64,
    /// a timed window's rows at most
    rows: u32,
    world: u32,
    vocab: u32,
    shape: dspark.Shape,
    /// a timed run's start or end on every rank (ns): the leader marks, each rank syncs and reads its clock
    mark: *const fn (ctx: *anyopaque) anyerror!u64,
    /// every rank's samples (`ours`, ns) into `all` [world][ours.len]
    gather: *const fn (ctx: *anyopaque, ours: []const i64, all: []i64) anyerror!void,
    /// the slot's prefill left a verify tail pending
    has_pending: *const fn (ctx: *anyopaque, slot: u32) bool,
    /// a timed window's rows dropped, the slot's text kept
    drop_window: *const fn (ctx: *anyopaque, slot: u32) anyerror!void,
    /// swaps the session store out for the run and back after it (null: no store)
    detach: ?*const fn (ctx: *anyopaque) void = null,
    /// the committed rows a pass ingests first, inside a timed pass (null: the pass is its proposal alone)
    ingest: ?*const fn (ctx: *anyopaque, slot: u32) anyerror!void = null,
    /// the shared windows timed past `rows`, ascending
    wide: []const u32 = &costs_mod.wide,
    /// the timing method where no knob names one
    base: cm.Method = .{},
    /// the measured table is written to the calibration directory
    store: bool = true,
};

/// The calibration prompt (rank 0): the calibration text's ids over the checkpoint's tokenizer. gpa-owned.
pub fn promptOf(gpa: Allocator, io: std.Io, dir: []const u8, vocab: u32) ![]u32 {
    var fb: [64]u32 = undefined;
    const tok_dir = if (std.c.getenv("TF_DSV41_GRAMMAR_TOKENIZER")) |d| std.mem.span(d) else dir;
    var t = tokenizer.loadTokenizer(io, gpa, tok_dir) catch |e| {
        if (!quiet) log.warn("calibration: no tokenizer in {s} ({t}); the fixed fallback ids", .{ tok_dir, e });
        return gpa.dupe(u32, cm.fallbackIds(vocab, &fb));
    };
    defer t.deinit();
    const ids = t.encode(gpa, cm.text) catch |e| {
        if (!quiet) log.warn("calibration: the text did not encode ({t}); the fixed fallback ids", .{e});
        return gpa.dupe(u32, cm.fallbackIds(vocab, &fb));
    };
    defer gpa.free(ids);
    return gpa.dupe(u32, cm.promptIds(ids, vocab, &fb));
}

/// Rank 0: the measurement on the loaded engine; the followers run it through the plan link.
pub fn measure(gpa: Allocator, d: Device, ids: []const u32, how: cm.Method, rows_cap: u64) !Measured {
    const tgt = d.target;
    const pass = d.pass;
    const rows = d.rows;
    var plan = try cm.Plan.initWide(gpa, d.slots, rows_cap, rows, how, pass != null, d.wide);
    defer plan.deinit(gpa);
    if (ids.len < 2) return error.CalibPrompt;
    // a few prompt rows in slot 0, one in the others: a prefill reads the experts its rows route to, so it stays short
    var lens: [4]usize = undefined;
    for (lens[0..plan.n], 0..) |*l, i| {
        l.* = if (i == 0) prefix_rows + 1 else 2;
        if (l.* + rows + 2 > d.limit) return error.CalibContext;
    }
    // rank 0's session store detached: a calibration text is neither resumed nor saved
    if (d.detach) |f| f(d.ctx);
    defer if (d.detach) |f| f(d.ctx);
    const began_prefill = ph.nowNs();
    try prefillAll(d, ids, lens[0..plan.n]);
    const prefill_ns = ph.nowNs() - began_prefill;
    // the timed runs, in the plan's order; each between two marks on every rank
    const samples = try gpa.alloc(i64, plan.runs.items.len);
    defer gpa.free(samples);
    // each window's tokens are the text's own that follow the slot's prompt
    var toks: [4][cm.max_rows]u32 = undefined;
    for (lens[0..plan.n], 0..) |l, i| {
        for (toks[i][0..rows], 0..) |*t, j| t.* = ids[(l - 1 + j) % ids.len];
    }
    const shape = d.shape;
    if (shape.block > cm.max_rows) return error.CalibBlock;
    var drafts: [cm.max_rows]u32 = undefined;
    var conf: [cm.max_rows]f32 = undefined;
    var chosen: [4][cm.max_rows]u32 = undefined;
    const began = ph.nowNs();
    var timed: usize = 0;
    var windows: usize = 0;
    var ingest_best: f64 = std.math.inf(f64);
    var ingest_first: f64 = 0.0;
    for (plan.runs.items, samples, 0..) |run, *x, at| {
        if (how.budget_ms > 0 and ph.nowNs() - began > @as(u64, how.budget_ms) * std.time.ns_per_ms and !run.must and plan.repeat(at)) {
            x.* = cm.skipped;
            continue;
        }
        if (run.what == .draft) {
            const p0: u64 = lens[0] - 1;
            const asks = [_]iface.Ask{.{ .slot = 0, .anchor = ids[lens[0] - 1], .start = p0, .params = dspark.Params.of(null, p0) }};
            var props = [_]iface.Proposal{.{ .drafts = drafts[0..shape.block], .conf = conf[0..shape.block] }};
            // a round's pass the way serving runs it: back to back, the first after the windows left untimed
            const pairs: u32 = if (run.warm) 1 else steady_warm + steady_reps;
            var best: i64 = std.math.maxInt(i64);
            for (0..pairs) |k| {
                const t0 = try d.mark(d.ctx);
                var t1 = t0;
                if (d.ingest) |f| {
                    try f(d.ctx, 0);
                    t1 = try d.mark(d.ctx);
                }
                try pass.?.propose(&asks, &props);
                const t2 = try d.mark(d.ctx);
                if (run.warm) break;
                if (k == 0) ingest_first = @as(f64, @floatFromInt(t1 - t0)) / 1e6;
                if (k < steady_warm) continue;
                best = @min(best, @as(i64, @intCast(t2 - t0)));
                ingest_best = @min(ingest_best, @as(f64, @floatFromInt(t1 - t0)) / 1e6);
            }
            x.* = if (run.warm) 0 else best;
            timed += 1;
            continue;
        }
        var segs: [4]iface.Segment = undefined;
        var outs: [4][]u32 = undefined;
        const draws: [cm.max_rows]u64 = @splat(0);
        for (run.segments(), segs[0..run.nsegs], outs[0..run.nsegs]) |s, *sg, *o| {
            if (s.slot >= plan.n) return error.CalibPlan;
            sg.* = .{ .slot = s.slot, .start = lens[s.slot] - 1, .tokens = toks[s.slot][0..s.rows], .parents = null, .draws = draws[0..s.rows], .sampling = null };
            o.* = chosen[s.slot][0..s.rows];
        }
        const t0 = try d.mark(d.ctx);
        try tgt.window(segs[0..run.nsegs], outs[0..run.nsegs]);
        x.* = @intCast((try d.mark(d.ctx)) - t0);
        timed += 1;
        windows += 1;
        for (run.segments()) |s| try d.drop_window(d.ctx, s.slot);
    }
    for (0..plan.n) |i| {
        const slot: u32 = @intCast(i);
        tgt.release(slot);
        if (pass) |p| p.reset(slot);
    }
    // every rank's samples; each entry the slower rank's
    const w: usize = d.world;
    const all = try gpa.alloc(i64, samples.len * w);
    defer gpa.free(all);
    try d.gather(d.ctx, samples, all);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const ranks = try a.alloc([]const f64, w);
    for (ranks, 0..) |*r, k| {
        const ms = try a.alloc(f64, samples.len);
        for (ms, all[k * samples.len ..][0..samples.len]) |*y, ns| {
            if (ns == cm.skipped) {
                y.* = -1.0;
                continue;
            }
            if (ns < 0) return error.CalibFollower; // a follower's marks did not pair up with ours
            y.* = @as(f64, @floatFromInt(ns)) / 1e6;
        }
        const v = try a.alloc(f64, plan.entries());
        try plan.values(ms, v);
        r.* = v;
    }
    var prefill_rows: u32 = 0;
    for (lens[0..plan.n]) |l| prefill_rows += @intCast(l - 1);
    const spent: f64 = @floatFromInt(ph.nowNs() - began);
    var out: Measured = .{ .costs = try plan.build(gpa, ranks), .rows = rows, .n = plan.n, .runs = plan.runs.items.len, .timed = timed, .ingest_ms = if (std.math.isFinite(ingest_best)) ingest_best else 0.0, .ingest_first_ms = ingest_first, .prefill_rows = prefill_rows, .windows = windows, .prefill_s = @as(f64, @floatFromInt(prefill_ns)) / 1e9, .runs_s = spent / 1e9 };
    @memcpy(out.kept[0..rows], ranks[0][1 .. rows + 1]);
    return out;
}

/// Every slot's prompt run into it (drafter reset, text but its last token, a pending verify tail kept).
fn prefillAll(d: Device, ids: []const u32, lens: []const usize) !void {
    for (lens, 0..) |l, i| {
        const slot: u32 = @intCast(i);
        if (d.pass) |p| p.reset(slot);
        _ = try d.target.prefill(slot, ids[0 .. l - 1], null, l - 1);
        if (d.has_pending(d.ctx, slot)) try d.target.keep(slot, &.{0});
    }
}

/// TF_DSV41_CALIB=measure / zig at open: prompt, measurement, stored entry, boot lines; table + stored file (or null).
pub fn boot(gpa: Allocator, io: std.Io, d: Device, dir: []const u8, k: calib_env.Knobs, want: calib_env.Shape) !Boot {
    const how = try cm.methodFrom(&calib_env.env, d.base);
    const rows_cap = (try calib_env.depthSettings(&calib_env.env)).max_rows;
    const began_prompt = ph.nowNs();
    const ids = try promptOf(gpa, io, dir, d.vocab);
    defer gpa.free(ids);
    const t0 = ph.nowNs();
    const prompt_s = @as(f64, @floatFromInt(t0 - began_prompt)) / 1e9;
    var got = try measure(gpa, d, ids, how, rows_cap);
    errdefer got.costs.deinit(gpa);
    if (!got.costs.plausible()) {
        log.warn("calibration: a pass of {d:.2} ms against a 1-row window of {d:.2} ms is not a measurement to price drafts by", .{ got.costs.draft, got.costs.windowMs(1) });
        return error.CalibImplausible;
    }
    const secs = @as(f64, @floatFromInt(ph.nowNs() - t0)) / 1e9;
    log.info("dsv41 calib method (zig): table {t}, {t} of {d} after {d} warm-up(s), rows 1-{d} cycled, rows 1-{d} rechecked, rows 9+ {d} runs; {d} slots, {d} timed runs, a {d}-token text; rows 1-4 kept {d:.3} / {d:.3} / {d:.3} / {d:.3} ms (rank 0)", .{ how.shape, how.stat, how.reps, how.warm, how.cycle, how.recheck, how.deep, got.n, got.runs, ids.len, got.kept[0], got.kept[1], got.kept[2], got.kept[3] });
    var path: ?[]u8 = null;
    var db: [std.fs.max_path_bytes]u8 = undefined;
    if (if (d.store) calib_env.dirOf(k, &db) else null) |to| {
        path = calib_env.store(gpa, io, to, want, got.costs, .{ .method = how, .seconds = secs, .kept = got.kept[0..got.rows], .tokens = ids.len }, std.Io.Clock.real.now(io).toSeconds()) catch |e| blk: {
            if (!quiet) log.warn("calibration: the table was not stored in {s} ({t})", .{ to, e });
            break :blk null;
        };
    }
    var line: [512]u8 = undefined;
    var wr = std.Io.Writer.fixed(&line);
    var first = true;
    for ([_]u32{ 1, 2, 4, 6, 8, 16, 17, 24, 32 }) |r| if (r <= got.costs.verify.len) {
        wr.print("{s}{d}", .{ if (first) "" else "/", r }) catch {};
        first = false;
    };
    wr.writeAll(" rows ") catch {};
    first = true;
    for ([_]u32{ 1, 2, 4, 6, 8, 16, 17, 24, 32 }) |r| if (r <= got.costs.verify.len) {
        wr.print("{s}{d:.1}", .{ if (first) "" else ", ", got.costs.windowMs(r) }) catch {};
        first = false;
    };
    log.info("dsv41 calibration (zig): draft {d:.2} ms (ingest {d:.2}, first {d:.2}), slot {d:.2} ms, verify {s} ms, measured in {d:.2} s over {d} timed runs (tokenizer and text {d:.2} s, {d} prompt rows in {d:.2} s, {d} windows and passes in {d:.2} s){s}{s}{s}", .{ got.costs.draft, got.ingest_ms, got.ingest_first_ms, got.costs.slot, wr.buffered(), secs, got.timed, prompt_s, got.prefill_rows, got.prefill_s, got.windows, got.runs_s, if (path != null) " (stored " else " (not stored", path orelse "", ")" });
    return .{ .costs = got.costs, .path = path };
}
