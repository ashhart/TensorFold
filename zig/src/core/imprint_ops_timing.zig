//! tf-startup-states ops: Imprint's per-request calls at a seeded count, each alone and as the prompt cache makes them.
const std = @import("std");
const platform = @import("sys");
const imprint = @import("prompt_imprint.zig");
const dirs = @import("lanes").learned_dirs;
const nowNs = @import("rows_store.zig").write.nowNs;
const Imprint = imprint.Imprint;
const Allocator = std.mem.Allocator;

/// Each run's figures in print order: per call in µs, then whole calls in ms.
pub const columns = [_][]const u8{
    "has_hit_us",     "has_miss_us",      "touch_us",  "best_hit_us",   "best_miss_us",    "recall_us",
    "recall_cold_us", "has_miss_cold_us", "remove_ms", "reload_add_ms", "reload_finds_ms", "learn_scans_ms",
};
const prompt_len = 40_960; // longer than every seeded record, so `best` compares a token of each
const sample = 1000; // keys per run for has and touch
const prompts = 200; // prompts per run for best
const colds = 20; // calls per run after a cache sweep
const sweep_bytes = 512 << 20; // more than this chip's caches: the next call reads the records from DRAM

fn us(ns: i128, calls: usize) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3 / @as(f64, @floatFromInt(calls));
}

fn ms(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// The prompt's chunk starts for a record at `at`: the record's own, `at` itself, then every 2,048 rows.
fn startsFor(a: Allocator, below: []const u32, at: u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    try out.appendSlice(a, below);
    var s = at;
    while (s < prompt_len) : (s += 2048) try out.append(a, s);
    return out.items;
}

/// Every cache line of `buf` read, so the next call finds no record in a cache.
fn evict(buf: []u64) void {
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < buf.len) : (i += 8) sum +%= buf[i];
    std.mem.doNotOptimizeAway(sum);
}

fn fresh(a: Allocator, r: std.Random, n: usize) ![]u32 {
    const t = try a.alloc(u32, n);
    for (t) |*x| x.* = r.uintLessThan(u32, 151_000);
    return t;
}

/// One process's figures, printed and appended to `runs`; the store gains two records (the reload's).
pub fn run(gpa: Allocator, root: []const u8, identity: u64, cap: u64, runs: [:0]const u8, count: usize) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try Imprint.open(gpa, root, identity, cap);
    defer m.deinit();
    const n = m.metas.items.len;
    if (n < count) return error.StatesLost;
    var prng: std.Random.DefaultPrng = .init(0x0b5e);
    const r = prng.random();
    var out: [columns.len]f64 = undefined;

    const k: usize = @min(n, sample);
    const keys = try a.alloc(u64, k);
    const absent = try a.alloc(u64, k);
    for (keys, absent) |*key, *miss| {
        key.* = m.metas.items[r.uintLessThan(usize, n)].key;
        miss.* = r.int(u64);
    }
    var found: usize = 0;
    var t0 = nowNs();
    for (keys) |key| found += @intFromBool(m.has(key));
    out[0] = us(nowNs() - t0, k);
    t0 = nowNs();
    for (absent) |key| found += @intFromBool(m.has(key));
    out[1] = us(nowNs() - t0, k);
    if (found != k) return error.OpsDiffer;
    const clock = m.clock;
    t0 = nowNs();
    for (keys) |key| m.touch(key);
    out[2] = us(nowNs() - t0, k);
    if (m.clock != clock + k) return error.OpsDiffer;

    // a request that resumes record `want[i]`: its tokens, then more of the conversation, with its chunk starts
    const p: usize = @min(n, prompts);
    const want = try a.alloc(u64, p);
    const text = try a.alloc([]u32, p);
    const starts = try a.alloc([]u32, p);
    for (want, text, starts) |*w, *t, *s| {
        const x = m.metas.items[r.uintLessThan(usize, n)];
        w.* = x.key;
        t.* = try fresh(a, r, prompt_len);
        @memcpy(t.*[0..x.tokens.len], x.tokens);
        s.* = try startsFor(a, x.starts, x.at);
    }
    const other = try a.alloc([]u32, p); // prompts no record begins: a different one each call, so none is hoisted
    for (other) |*t| t.* = try fresh(a, r, prompt_len);
    const other_starts = try startsFor(a, &.{}, 2048);
    t0 = nowNs();
    for (want, text, starts) |w, t, s| found += @intFromBool((m.best(t, s, true, 0) orelse return error.OpsDiffer).key == w);
    out[3] = us(nowNs() - t0, p);
    t0 = nowNs();
    for (other) |t| found += @intFromBool(m.best(t, other_starts, true, 0) != null);
    out[4] = us(nowNs() - t0, p);
    if (found != k + p) return error.OpsDiffer;
    // Store.recall's Imprint work on a hit: the longest learned prefix, then (after the family's read) its touch
    t0 = nowNs();
    for (text, starts) |t, s| m.touch((m.best(t, s, true, 0) orelse return error.OpsDiffer).key);
    out[5] = us(nowNs() - t0, p);

    const buf = try a.alloc(u64, sweep_bytes / 8);
    @memset(buf, 1);
    var cold: [2]i128 = .{ 0, 0 };
    for (0..@min(p, colds)) |i| {
        evict(buf);
        t0 = nowNs();
        m.touch((m.best(text[i], starts[i], true, 0) orelse return error.OpsDiffer).key);
        cold[0] += nowNs() - t0;
        evict(buf);
        t0 = nowNs();
        found += @intFromBool(m.has(absent[i]));
        cold[1] += nowNs() - t0;
    }
    out[6] = us(cold[0], @min(p, colds));
    out[7] = us(cold[1], @min(p, colds));
    if (found != k + p) return error.OpsDiffer;

    // prompt_learn.learn's scans before a new state's write (one rank, the single reclaim), no victim chosen
    t0 = nowNs();
    var held = @max(m.total(), try dirs.bytes(m.dir)) + try dirs.totalOthers(m.root, m.dir) + try m.indexScratchBytes();
    for (m.metas.items) |x| held += try imprint.singleReclaim(@ptrCast(&m), m.dir, x.key);
    out[11] = ms(nowNs() - t0);
    std.mem.doNotOptimizeAway(held);

    // remove rewrites the index without one record; it is learned again (untimed), now last
    const x = m.metas.items[r.uintLessThan(usize, n)];
    const again = .{ x.key, x.at, try a.dupe(u32, x.tokens), try a.dupe(u32, x.starts), x.bytes };
    t0 = nowNs();
    try m.remove(again[0]);
    out[8] = ms(nowNs() - t0);
    try m.add(again[0], again[1], again[2], again[3], again[4]);

    // another writer appends a record, so this add reads the whole index again first (reload)
    var seeded: std.Random.DefaultPrng = .init(@truncate(@as(u128, @bitCast(nowNs()))));
    const theirs = try fresh(a, seeded.random(), 1025);
    const ours = try fresh(a, seeded.random(), 1025);
    {
        var w = try Imprint.open(gpa, root, identity, cap);
        defer w.deinit();
        try w.add(Imprint.keyOf(theirs), 1024, theirs, &.{}, 1);
    }
    t0 = nowNs();
    try m.add(Imprint.keyOf(ours), 1024, ours, &.{}, 1);
    out[9] = ms(nowNs() - t0);
    if (!m.has(Imprint.keyOf(theirs)) or m.metas.items.len != n + 2) return error.OpsDiffer;
    // reload's loop alone: each record of the index read again looked up in the list held before, in order
    t0 = nowNs();
    for (m.metas.items) |y| found += @intFromBool(m.has(y.key));
    out[10] = ms(nowNs() - t0);
    if (found != k + p + n + 2) return error.OpsDiffer;

    std.debug.print("IMPRINT_OPS states={d}", .{n});
    for (columns, out) |c, v| std.debug.print(" {s}={d:.3}", .{ c, v });
    std.debug.print("\n", .{});
    var line: std.ArrayList(u8) = .empty;
    try line.print(a, "{d}", .{n});
    for (out) |v| try line.print(a, " {d:.4}", .{v});
    try line.append(a, '\n');
    const fd = platform.c.open(runs, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.RunsWrite;
    defer _ = platform.c.close(fd);
    try imprint.writeAll(fd, line.items);
}

/// Each figure's median over the runs, with its range.
pub fn summary(gpa: Allocator, runs: [:0]const u8, count: usize) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const fd = platform.c.open(runs, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.NoRuns;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end <= 0) return error.NoRuns;
    const text = try a.alloc(u8, @intCast(end));
    if (!imprint.readAt(fd, text, 0)) return error.NoRuns;
    var cols: [columns.len]std.ArrayList(f64) = @splat(.empty);
    var held: [2]u64 = .{ std.math.maxInt(u64), 0 };
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        const states = try std.fmt.parseInt(u64, it.next() orelse return error.BadRuns, 10);
        held = .{ @min(held[0], states), @max(held[1], states) };
        for (&cols) |*c| try c.append(a, try std.fmt.parseFloat(f64, it.next() orelse return error.BadRuns));
    }
    const runs_n = cols[0].items.len;
    std.debug.print("IMPRINT_OPS_MEDIAN states={d} held={d}-{d} runs={d}", .{ count, held[0], held[1], runs_n });
    for (columns, &cols) |name, *c| {
        std.mem.sort(f64, c.items, {}, std.sort.asc(f64));
        std.debug.print(" {s}={d:.3} [{d:.3}-{d:.3}]", .{ name, c.items[runs_n / 2], c.items[0], c.items[runs_n - 1] });
    }
    std.debug.print("\n", .{});
}
