//! Flash Next prompt reuse against fresh passes: agent-like turns, every depth, copy drafts off and on, state bytes.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const api = @import("engine_api");
const fx = tf.flashnext_engine;
const snap = tf.flashnext_snapshot;
const pc = api.prompt_cache;
const Allocator = std.mem.Allocator;

/// The store's copies of the engine's state, as the server's host makes them.
const Family = struct {
    e: *fx.Engine,
    gpa: Allocator,
    save_s: f64 = 0,

    fn of(ptr: *anyopaque) *Family {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return snap.bytes(at);
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        const t0 = mtl.clock.seconds();
        defer f.save_s = mtl.clock.seconds() - t0;
        return try snap.save(f.e, f.gpa, at);
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        try snap.restore(of(ptr).e, @ptrCast(@alignCast(saved)));
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        snap.drop(of(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
};

/// One reply's tokens, and at its marks the store's keep or (state checks) a copy of the state for the harness.
const Run = struct {
    a: Allocator,
    got: std.ArrayList(u32) = .empty,
    store: ?*pc.Store = null,
    prompt: []const u32 = &.{},
    copy_at: ?*?*snap.State = null,
    pooled_at: ?*Pooled = null,
    e: ?*fx.Engine = null,

    fn prefilled(_: *anyopaque) void {}
    fn tokens(ctx: *anyopaque, t: []const u32) bool {
        const r: *Run = @ptrCast(@alignCast(ctx));
        r.got.appendSlice(r.a, t) catch {};
        return false;
    }
    fn cancelled(_: *anyopaque) bool {
        return false;
    }
    fn marked(ctx: *anyopaque, at: usize) void {
        const r: *Run = @ptrCast(@alignCast(ctx));
        if (r.store) |s| _ = s.keep(r.prompt, @intCast(at), null);
        if (r.copy_at) |slot| slot.* = snap.save(r.e.?, r.a, at) catch null;
        if (r.pooled_at) |p| p.* = Pooled.of(r.e.?.m, at);
    }
    fn out(r: *Run) fx.Out {
        return .{ .ctx = r, .prefilled = prefilled, .tokens = tokens, .cancelled = cancelled, .marked = marked };
    }
    fn hash(r: *const Run) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(r.got.items));
    }
};

fn loadTokens(a: Allocator, path: []const u8) ![]u32 {
    const f = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0));
    const j = try std.json.parseFromSliceLeaky(std.json.Value, a, f.bytes[0..f.size], .{});
    const items = j.object.get("prompt").?.array.items;
    const out = try a.alloc(u32, items.len);
    for (items, 0..) |x, i| out[i] = @intCast(x.integer);
    return out;
}

/// A fresh reply: the whole prompt from position 0, no marks.
fn fresh(e: *fx.Engine, a: Allocator, prompt: []const u32, n: usize, depth: ?usize, copy: bool) !Run {
    e.copy = copy;
    var r: Run = .{ .a = a };
    _ = try e.generateFrom(prompt, 0, &.{}, n, &.{}, depth, r.out());
    return r;
}

const Turn = struct { prompt: []u32, history: u32 };

/// The indexer's pooled block keys before a mark, rebuilt after a restore rather than copied: counts and a hash of every layer's.
const Pooled = struct {
    counts: [13]usize = @splat(0),
    hash: u64 = 0,

    /// Blocks wholly before `at` (a call that ran past the mark may have pooled more).
    fn of(m: *tf.flashnext_replay.Model, at: usize) Pooled {
        var out: Pooled = .{};
        var h = std.hash.Wyhash.init(0);
        var k: usize = 0;
        for (&m.layers) |*L| if (!L.linear) {
            out.counts[k] = @min(L.pooled_n, at / 4);
            h.update(L.pooled.b.contents()[L.pooled.off..][0 .. out.counts[k] * 128 * 2]);
            k += 1;
        };
        out.counts[k] = @min(m.mtp.pooled_n, at / 4);
        h.update(m.mtp.pooled.b.contents()[m.mtp.pooled.off..][0 .. out.counts[k] * 128 * 2]);
        out.hash = h.final();
        return out;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("usage: tf-flashnext-reuse MODEL_DIR DUMP_DIR PROMPT.json... (FZ_N reply tokens, FZ_DEPTHS 1,2,3,5,0 with 0 the rule)\n", .{});
        std.process.exit(2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const n_out: usize = if (std.c.getenv("FZ_N")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 64;
    var depths: std.ArrayList(?usize) = .empty;
    var dit = std.mem.tokenizeScalar(u8, if (std.c.getenv("FZ_DEPTHS")) |v| std.mem.span(v) else "1,2,3,5,0", ',');
    while (dit.next()) |d| {
        const k = try std.fmt.parseInt(usize, d, 10);
        try depths.append(a, if (k == 0) null else k);
    }
    const e = try fx.Engine.load(gpa, init.io, args[1], args[2]);
    defer e.deinit();
    try e.warm();
    var fam: Family = .{ .e = e, .gpa = gpa };
    var failures: usize = 0;
    for (args[3..]) |path| {
        const t = try loadTokens(a, path);
        const third = t.len / 3;
        var store = pc.Store.init(gpa, fam.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 64 << 30); // every turn keeps, short ones too
        defer store.deinit();
        std.debug.print("== {s}: {d} tokens\n", .{ path, t.len });
        // a later turn: a turn's prompt, its reply, more text, and the same last 3 tokens (the generation prompt)
        var turns: [4]Turn = undefined;
        var replies: [4][]const u32 = undefined;
        turns[0] = .{ .prompt = t[0..third], .history = @intCast(third - 3) };
        for (0..4) |k| {
            if (k > 0) {
                const base = if (k == 3) 1 else k - 1; // turn 4 continues turn 2: turn 3 edits turn 1's text
                const more = t[third + (k - 1) * third / 2 ..][0 .. third / 2];
                const gen = turns[base].prompt[turns[base].history..];
                const next = try std.mem.concat(a, u32, &.{ turns[base].prompt, replies[base], more, gen });
                if (k == 2) next[third / 2] +%= 1;
                turns[k] = .{ .prompt = next, .history = @intCast(next.len - 3) };
            }
            const turn = turns[k];
            const ref = try fresh(e, a, turn.prompt, n_out, 0, false);
            const rule = try fresh(e, a, turn.prompt, n_out, null, true);
            std.debug.print("turn {d}: {d} tokens, history {d}; fresh plain {x:0>16}, fresh rule {x:0>16}{s}; reply {any}\n", .{ k + 1, turn.prompt.len, turn.history, ref.hash(), rule.hash(), if (ref.hash() == rule.hash()) "" else "  DRAFTED != PLAIN", ref.got.items[0..@min(8, ref.got.items.len)] });
            if (ref.hash() != rule.hash()) failures += 1;
            replies[k] = ref.got.items;
            for (depths.items, 0..) |depth, di| for ([_]bool{ false, true }) |copy| {
                e.copy = copy;
                fam.save_s = 0;
                var r: Run = .{ .a = a, .store = &store, .prompt = turn.prompt };
                const t0 = mtl.clock.seconds();
                const plan = try store.begin(a, turn.prompt, turn.history, &.{}, &.{}, null);
                const restore_s = mtl.clock.seconds() - t0;
                const keeper = di + 1 == depths.items.len and copy; // every run resumes from the last turn's state; the last keeps this one's
                _ = try e.generateFrom(turn.prompt, plan.from, if (keeper) plan.marks else &.{}, n_out, &.{}, depth, r.out());
                const same = r.hash() == ref.hash();
                if (!same) failures += 1;
                std.debug.print("  depth {d} copy {s}: from {d}, marks {any}, restore {d:.1} ms, save {d:.1} ms: {x:0>16} {s}\n", .{ depth orelse 0, if (copy) "on " else "off", plan.from, plan.marks, restore_s * 1e3, fam.save_s * 1e3, r.hash(), if (same) "SAME" else "DIFF" });
            };
            try stateCheck(e, a, &store, turn, &failures);
        }
        std.debug.print("store: {d} entries, {d} MiB held; hits {d} misses {d} kept {d} evicted {d} refused {d} failed {d}\n", .{ store.entries.items.len, store.held >> 20, store.counts.hits, store.counts.misses, store.counts.kept, store.counts.evicted, store.counts.refused, store.counts.failed });
    }
    std.debug.print("{s}: {d} failures\n", .{ if (failures == 0) "PASS" else "FAIL", failures });
    if (failures > 0) std.process.exit(1);
}

/// The state at the turn's history three ways, byte for byte: a fresh pass whose call ends there, a fresh pass whose call runs past it, a resumed one.
fn stateCheck(e: *fx.Engine, a: Allocator, store: *pc.Store, turn: Turn, failures: *usize) !void {
    const entry = store.find(turn.prompt[0..turn.history], &.{}) orelse return; // nothing to resume before the history
    var states: [3]?*snap.State = .{ null, null, null };
    var pooled: [3]Pooled = .{ .{}, .{}, .{} };
    defer for (states) |s| if (s) |st| snap.drop(a, st);
    defer e.mark_taps = true;
    for (0..3) |arm| {
        e.mark_taps = arm != 0;
        var from: usize = 0;
        if (arm == 2) {
            try snap.restore(e, @ptrCast(@alignCast(entry.saved)));
            from = entry.at;
        }
        var r: Run = .{ .a = a, .copy_at = &states[arm], .pooled_at = &pooled[arm], .e = e };
        _ = try e.generateFrom(turn.prompt, from, &.{turn.history}, 2, &.{}, 0, r.out());
    }
    const x = states[0] orelse return error.NoState;
    const n = snap.bytes(turn.history);
    var verdicts: [2][]const u8 = undefined;
    for (states[1..], pooled[1..], 0..) |s, pl, k| {
        const y = s orelse return error.NoState;
        const diff = std.mem.indexOfDiff(u8, x.buf.contents()[0..n], y.buf.contents()[0..n]);
        const pooled_same = std.mem.eql(usize, &pooled[0].counts, &pl.counts) and pooled[0].hash == pl.hash;
        const same = diff == null and x.hist[0] == y.hist[0] and x.hist[1] == y.hist[1] and pooled_same;
        if (!same) failures.* += 1;
        verdicts[k] = if (same) "SAME" else if (diff) |d| try std.fmt.allocPrint(a, "DIFF at byte {d}", .{d}) else "DIFF (history or pooled keys)";
    }
    std.debug.print("  state at {d} ({d} MiB, pooled keys {d} blocks a layer) against a call ending there: run past {s}; resumed from {d} {s}\n", .{ turn.history, n >> 20, pooled[0].counts[0], verdicts[0], entry.at, verdicts[1] });
}
