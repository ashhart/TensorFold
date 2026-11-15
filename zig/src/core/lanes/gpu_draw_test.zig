//! The host GPU draws with selection against main's full sort of the row: every pick, plan and order place the same.
const std = @import("std");
const rule = @import("gpu_rule.zig");
const full = @import("gpu_full.zig");
const select = @import("gpu_select.zig");
const Sampling = @import("sampling.zig").Sampling;
const Cand = rule.Cand;
const t = std.testing;
const a = t.allocator;

/// Main's rule.sample before selection, kept whole (its sort as `order`) as the reference every draw must equal.
fn refSample(logits: []const f32, s: ?Sampling, position: u32, ids: ?[]const u32, comptime order: fn (void, Cand, Cand) bool) !u32 {
    const st = s orelse {
        const at = rule.argmax(logits);
        return if (ids) |m| m[at] else at;
    };
    const r = rule.rule(st);
    const kc = r.top_k;
    const cap: usize = if (kc == 0 or kc > rule.candidates) rule.candidates else kc;
    const all = try a.alloc(Cand, logits.len);
    defer a.free(all);
    var m = -std.math.inf(f32);
    for (all, logits, 0..) |*c, l, i| {
        c.* = .{ .v = l * r.inv_t, .i = @intCast(i) };
        m = @max(m, c.v);
    }
    var lanes: [1024]f32 = @splat(0.0);
    for (all, 0..) |c, i| lanes[i % 1024] += @exp(c.v - m);
    const z = rule.lanesSum(&lanes);
    std.mem.sort(Cand, all, {}, order);
    const n: usize = @min(@min(all.len, rule.candidates), cap);
    var norm = z;
    if (kc != 0) {
        norm = 0.0;
        for (all[0..n]) |c| norm += @exp(c.v - m);
    }
    var keep: usize = n;
    if (r.top_p > 0.0 and r.top_p < 1.0) {
        var cum: f32 = 0.0;
        for (all[0..n], 0..) |c, j| {
            cum += @exp(c.v - m) / norm;
            if (cum >= r.top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (r.min_log > -std.math.inf(f32)) {
        const floor = m + r.min_log;
        var j: usize = 0;
        while (j < keep and all[j].v >= floor) j += 1;
        keep = j;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f32);
    for (all[0..keep], 0..) |c, k| {
        const id = if (ids) |map| map[c.i] else c.i;
        const score = c.v - @log(-@log(rule.uniform24(st.seed, position, id)));
        if (k == 0 or score > best_score) {
            best = k;
            best_score = score;
        }
    }
    const pick = all[best].i;
    return if (ids) |map| map[pick] else pick;
}

/// Main's gpu_full.plan before selection: the whole row sorted, its place top_k - 1 and its first 1,024.
fn refPlan(logits: []const f32, s: Sampling) !full.Plan {
    const r = rule.rule(s);
    const order = try a.alloc(Cand, logits.len);
    defer a.free(order);
    var m = -std.math.inf(f32);
    for (order, logits, 0..) |*c, l, i| {
        c.* = .{ .v = l * r.inv_t, .i = @intCast(i) };
        m = @max(m, c.v);
    }
    std.mem.sort(Cand, order, {}, rule.better);
    var p = full.Plan{ .inv_t = r.inv_t, .m = m, .lo = @max(m + r.min_log, m - full.never) };
    if (s.top_k != 0 and s.top_k < logits.len) p.top = .{ .key = rule.key(order[s.top_k - 1].v), .id = order[s.top_k - 1].i };
    if (r.top_p > 0.0 and r.top_p < 1.0) p.cut = full.nucleus(logits, p, order[0..@min(order.len, rule.candidates)], r.top_p);
    return p;
}

fn refDraw(logits: []const f32, s: Sampling, position: u32, ids: ?[]const u32) !u32 {
    if (full.fullVocabulary(s)) return full.race(logits, try refPlan(logits, s), s.seed, position, ids);
    return refSample(logits, s, position, ids, rule.better);
}

/// The planted control: ties broken toward the higher id, an order the full sort never takes.
fn tiesHighFirst(_: void, x: Cand, y: Cand) bool {
    const kx = rule.key(x.v);
    const ky = rule.key(y.v);
    return kx > ky or (kx == ky and x.i > y.i);
}

const Kind = enum { spread, ties, zeros, nonfinite, masked };

fn fill(row: []f32, kind: Kind, rng: std.Random) void {
    for (row, 0..) |*v, i| v.* = switch (kind) {
        .spread => (rng.float(f32) - 0.5) * 16.0,
        .ties => @floatFromInt(rng.uintLessThan(u32, 3)),
        .zeros => if (i % 2 == 0) -0.0 else 0.0,
        .nonfinite => switch (rng.uintLessThan(u32, 8)) {
            0 => @bitCast(@as(u32, 0x7fc0_0042)),
            1 => @bitCast(@as(u32, 0xffc0_0007)),
            2 => std.math.inf(f32),
            3 => -std.math.inf(f32),
            else => (rng.float(f32) - 0.5) * 8.0,
        },
        .masked => if (rng.uintLessThan(u32, 10) == 0) rng.float(f32) * 4.0 else -std.math.inf(f32),
    };
}

/// Settings around every edge: top_k 0, 1, the candidates' bounds, the row's length and past it; top_p 1 and off.
fn settings(len: usize, out: *std.ArrayList(Sampling), seed: u64) !void {
    const temps = [_]f64{ 1e-7, 0.3, 1.0, 2.5 };
    const ks = [_]u32{ 0, 1, 2, 20, 1023, 1024, 1025, @intCast(@max(len, 2) - 1), @intCast(len), @intCast(len + 5), std.math.maxInt(u32) };
    const ps = [_]f64{ 0.0, 0.5, 0.95, 1.0 };
    const mins = [_]f64{ 0.0, 0.05, 0.6 };
    for (temps) |temp| for (ks) |k| for (ps) |p| for (mins) |mp| {
        try out.append(a, .{ .seed = seed, .temperature = temp, .top_k = k, .top_p = p, .min_p = mp });
    };
}

/// Draws `f` makes that differ from main's, over rows of every kind and length, with and without an id map.
fn mismatches(comptime planted: bool) !usize {
    var prng = std.Random.DefaultPrng.init(0x3396);
    const rng = prng.random();
    var grid: std.ArrayList(Sampling) = .empty;
    defer grid.deinit(a);
    var differ: usize = 0;
    const lens = [_]usize{ 1, 2, 17, 1023, 1024, 1025, 3001 };
    for (std.enums.values(Kind)) |kind| for (lens, 0..) |len, li| {
        const row = try a.alloc(f32, len);
        defer a.free(row);
        fill(row, kind, rng);
        const map = try a.alloc(u32, len);
        defer a.free(map);
        for (map, 0..) |*x, i| x.* = @intCast((i * 7919 + 13) % len);
        grid.clearRetainingCapacity();
        try settings(len, &grid, rng.int(u64));
        for (grid.items, 0..) |s, gi| {
            const position: u32 = @intCast(gi * 31 + li);
            const ids: ?[]const u32 = if (gi % 3 == 0) map else null;
            const want = try refSample(row, s, position, ids, rule.better);
            const got = if (planted) try refSample(row, s, position, ids, tiesHighFirst) else try rule.sample(a, row, s, position, ids);
            differ += @intFromBool(want != got);
            if (planted) continue;
            differ += @intFromBool(try refDraw(row, s, position, ids) != try full.draw(a, row, s, position, ids));
            const pw = try refPlan(row, s);
            const pg = try full.plan(a, row, s);
            differ += @intFromBool(!std.mem.eql(u8, std.mem.asBytes(&pw), std.mem.asBytes(&pg)));
        }
    };
    return differ;
}

test "gpu draw: selection picks main's token for every row kind, edge setting, seed and id map" {
    try t.expectEqual(@as(usize, 0), try mismatches(false));
}

test "gpu draw: the planted control, ties broken the other way, is caught" {
    try t.expect(try mismatches(true) > 0);
}

test "gpu draw: top and nth equal the full sort's places, NaNs, infinities and signed zeros included" {
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var kept: [rule.candidates]Cand = undefined;
    for (std.enums.values(Kind)) |kind| for ([_]usize{ 1, 5, 1024, 2049 }) |len| {
        const row = try a.alloc(f32, len);
        defer a.free(row);
        fill(row, kind, rng);
        const order = try a.alloc(Cand, len);
        defer a.free(order);
        for (order, row, 0..) |*c, l, i| c.* = .{ .v = l * 1.5, .i = @intCast(i) };
        const scratch = try a.dupe(Cand, order);
        defer a.free(scratch);
        std.mem.sort(Cand, order, {}, rule.better);
        const n = @min(len, rule.candidates);
        const head = select.top(row, 1.5, n, &kept);
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(order[0..n]), std.mem.sliceAsBytes(head));
        for ([_]usize{ 0, len / 3, len / 2, len - 1 }) |k| {
            @memcpy(scratch, order);
            rng.shuffle(Cand, scratch);
            const got = select.nth(scratch, k);
            try t.expectEqualSlices(u8, std.mem.asBytes(&order[k]), std.mem.asBytes(&got));
        }
    };
}

/// ns a draw of `f` takes on `row`, the median of `rounds`.
fn timed(row: []const f32, s: Sampling, comptime f: anytype, rounds: usize) !u64 {
    var ns: [16]u64 = undefined;
    for (ns[0..rounds], 0..) |*x, r| {
        const t0 = std.Io.Clock.awake.now(t.io);
        std.mem.doNotOptimizeAway(try f(row, s, @intCast(r)));
        x.* = @intCast(t0.durationTo(std.Io.Clock.awake.now(t.io)).toNanoseconds());
    }
    std.mem.sort(u64, ns[0..rounds], {}, std.sort.asc(u64));
    return ns[rounds / 2];
}

fn mainRule(row: []const f32, s: Sampling, p: u32) !u32 {
    return refSample(row, s, p, null, rule.better);
}
fn newRule(row: []const f32, s: Sampling, p: u32) !u32 {
    return rule.sample(a, row, s, p, null);
}
fn mainFull(row: []const f32, s: Sampling, p: u32) !u32 {
    return refDraw(row, s, p, null);
}
fn newFull(row: []const f32, s: Sampling, p: u32) !u32 {
    return full.draw(a, row, s, p, null);
}

test "gpu draw: timing: one row of a 150,144-token vocabulary, main's sort against selection" {
    var prng = std.Random.DefaultPrng.init(150144);
    const row = try a.alloc(f32, 150144);
    defer a.free(row);
    fill(row, .spread, prng.random());
    const flint: Sampling = .{ .seed = 1234, .temperature = 1.0, .top_k = 20, .top_p = 0.95 };
    const wide: Sampling = .{ .seed = 1234, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    const before = try timed(row, flint, mainRule, 9);
    const after = try timed(row, flint, newRule, 9);
    const full_before = try timed(row, wide, mainFull, 5);
    const full_after = try timed(row, wide, newFull, 5);
    std.debug.print("gpu draw timing: top_k 20 {d} -> {d} ns a row; top_k 0 {d} -> {d} ns a row\n", .{ before, after, full_before, full_after });
    for (0..8) |p| try t.expectEqual(try mainRule(row, flint, @intCast(p)), try newRule(row, flint, @intCast(p)));
    try t.expect(after * 2 < before);
}
