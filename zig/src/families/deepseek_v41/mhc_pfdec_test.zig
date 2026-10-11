//! TF_DSV41_MHC_PFDEC on the real emitters, host only (`zig build test-dsv41-wide`): mixing sites become site + coef.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const check = @import("calls_check.zig");
const rowmode = @import("rowmode.zig");
const branches = @import("branches.zig");
const dspark_emit = @import("dspark_emit.zig");
const pk = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;
const Arg = calls.Arg;
const Call = calls.Call;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn servedOptions(limit: i64, pf: i64, deferred: bool) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = true, .mhc_defer = deferred, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true, .mhc_pf_rows = pf };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

fn same(x: Arg, y: Arg) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = if (y == .t) y.t else y.opaque_table;
            if (std.meta.activeTag(t.role) != std.meta.activeTag(u.role)) break :blk false;
            const rs = switch (t.role) {
                .weight => |r| std.mem.eql(u8, r, u.role.weight),
                .buf => |r| std.mem.eql(u8, r, u.role.buf),
                .empty => true,
            };
            break :blk rs and t.dt == u.dt and t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .i => |v| v == y.i,
        .f => |v| v == y.f,
        .b => |v| v == y.b,
        .none => true,
        .list => |l| l.len == y.list.len and for (l, y.list) |p, q| {
            if (!same(p, q)) break false;
        } else true,
    };
}

fn sameCall(x: Call, y: Call) bool {
    if (!std.mem.eql(u8, x.name, y.name) or x.triton != y.triton or x.glue != y.glue or x.begin != y.begin) return false;
    if (!std.mem.eql(i64, &x.grid, &y.grid) or x.args.len != y.args.len) return false;
    if (x.side != y.side or x.fork != y.fork or x.join != y.join) return false;
    for (x.args, y.args) |p, q| if (!std.mem.eql(u8, p.name, q.name) or !same(p.arg, q.arg)) return false;
    return true;
}

fn named(c: Call, name: []const u8) Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    std.debug.print("{s}: no argument {s}\n", .{ c.name, name });
    @panic("argument");
}

fn isEmpty(x: Arg) bool {
    return x == .t and x.t.role == .empty;
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

/// What a mixing site reads and writes, whichever kernels run it.
const Site = struct {
    mode: i64,
    n: i64,
    x: Arg,
    xout: ?Arg = null,
    g: ?Arg = null,
    post: ?Arg = null,
    comb: ?Arg = null,
    pre: ?Arg = null,
    fnw: Arg,
    part: Arg,
    c: Arg,
    tap: ?Arg = null,
    nw: Arg,
    out: Arg,
    base: Arg,
    scale: Arg,
    next: [3]Arg,
    consts: [4]Arg, // eps, hc_eps, post_alpha, iters
    begin: calls.Begin,
    calls: usize,
};

/// The site the off path runs at cs[i] (mhc_cuda's boundary or `_site` + `_finish_k`), or null when not a mixing site.
fn offSite(cs: []const Call, i: usize) ?Site {
    const c = cs[i];
    if (eq(c.name, "tf_dsv41_mhc_cuda_v1.run")) {
        const a = c.args;
        const mode = a[19].arg.i;
        if (mode == 3) return null;
        const has_coef = i + 1 < cs.len and eq(cs[i + 1].name, "tf_dsv41_mhc_cuda_v1.coef");
        return .{
            .mode = mode,
            .n = a[18].arg.i,
            .x = a[0].arg,
            .xout = if (mode == 0) a[1].arg else null,
            .g = if (mode == 0) a[2].arg else null,
            .post = if (mode == 0) a[3].arg else null,
            .comb = if (mode == 0) a[4].arg else null,
            .pre = if (mode != 1) a[5].arg else null,
            .fnw = a[6].arg,
            .part = a[9].arg,
            .c = a[10].arg,
            .tap = if (isEmpty(a[11].arg)) null else a[11].arg,
            .nw = a[12].arg,
            .out = a[13].arg,
            .base = a[7].arg,
            .scale = a[8].arg,
            .next = .{ a[14].arg, a[15].arg, a[16].arg },
            .consts = .{ a[20].arg, a[21].arg, a[22].arg, a[23].arg },
            .begin = c.begin,
            .calls = if (has_coef) 2 else 1,
        };
    }
    if (eq(c.name, "_site") and named(c, "MIX").b) {
        const f = cs[i + 1];
        std.debug.assert(eq(f.name, "_finish_k") and named(f, "COEF").b);
        const post = named(c, "POST_ON").b;
        const coll = named(c, "COLLAPSE").i;
        return .{
            .mode = if (post) 0 else if (coll == 1) 1 else 2,
            .n = named(c, "R").i,
            .x = named(c, "X"),
            .xout = if (post) named(c, "XOUT") else null,
            .g = if (post) named(c, "G") else null,
            .post = if (post) named(c, "POST") else null,
            .comb = if (post) named(c, "COMB") else null,
            .pre = if (coll == 2) named(c, "PRE") else null,
            .fnw = named(c, "FN"),
            .part = named(c, "PART"),
            .c = named(c, "C"),
            .tap = if (named(c, "TAP_ON").b) named(c, "TAP") else null,
            .nw = named(f, "NW"),
            .out = named(f, "OUT"),
            .base = named(f, "BASE"),
            .scale = named(f, "SCALE"),
            .next = .{ named(f, "PRE"), named(f, "POST"), named(f, "COMB") },
            .consts = .{ named(f, "eps"), named(f, "hc_eps"), named(f, "post_alpha"), named(f, "ITERS") },
            .begin = c.begin,
            .calls = 2,
        };
    }
    return null;
}

/// PFDEC's three calls at cs[j], checked against the off path's `s`.
fn expectPf(cs: []const Call, j: usize, s: Site, deferred: bool) !void {
    const p = cs[j];
    const f = cs[j + 1];
    const k = cs[j + 2];
    try testing.expectEqualStrings("tf_dsv41_mhc_pf_v1.run", p.name);
    try testing.expectEqualStrings("_finish_k", f.name);
    try testing.expectEqualStrings("tf_dsv41_mhc_cuda_v1.coef", k.name);
    const a = p.args;
    try testing.expect(s.begin == p.begin and f.begin == .none and k.begin == .none);
    try testing.expectEqual(s.mode, a[11].arg.i);
    try testing.expectEqual(s.n, a[10].arg.i);
    try testing.expect(same(s.x, a[0].arg));
    // mhc_pf: the post's buffers with a boundary, empty tensors otherwise (its binding's numel-0 checks)
    inline for (.{ .{ "xout", 1 }, .{ "g", 2 }, .{ "post", 3 }, .{ "comb", 4 }, .{ "pre", 5 }, .{ "tap", 9 } }) |m| {
        if (@field(s, m[0])) |v| try testing.expect(same(v, a[m[1]].arg)) else try testing.expect(isEmpty(a[m[1]].arg));
    }
    try testing.expect(same(s.fnw, a[6].arg) and same(s.part, a[7].arg) and same(s.c, a[8].arg));
    // the normed input alone: the final's `_finish_k` instance
    try testing.expect(!named(f, "COEF").b and named(f, "NORM_FIRST").b and named(f, "UNROLL").b);
    try testing.expectEqual(@as(i64, 1024), named(f, "BLOCK").i);
    try testing.expectEqual([3]i64{ s.n, 1, 1 }, f.grid);
    try testing.expect(same(s.part, named(f, "PART")) and same(s.c, named(f, "C")) and same(s.nw, named(f, "NW")) and same(s.out, named(f, "OUT")));
    try testing.expect(same(s.consts[0], named(f, "eps")));
    // coef_kernel: the next coefficients from the same partials, base and scale
    const b = k.args;
    try testing.expect(same(s.part, b[0].arg) and same(s.base, b[1].arg) and same(s.scale, b[2].arg));
    for (0..3) |q| try testing.expect(same(s.next[q], b[3 + q].arg));
    try testing.expectEqual(s.n, b[6].arg.i);
    for (0..4) |q| try testing.expect(same(s.consts[q], b[7 + q].arg));
    try testing.expectEqual(deferred, k.defer_side);
    try testing.expect(!p.defer_side and !f.defer_side and !f.defer_join and !k.defer_join);
}

/// `on` is `off` with each mixing site at or past `rows` rows as PFDEC's calls; returns the sites converted.
fn expectSwap(off: []const Call, on: []const Call, pf: bool, deferred: bool) !usize {
    var i: usize = 0;
    var j: usize = 0;
    var sites: usize = 0;
    while (i < off.len) {
        errdefer std.debug.print("off call {d} ({s}) / on call {d} ({s})\n", .{ i, off[i].name, j, if (j < on.len) on[j].name else "-" });
        if (pf) if (offSite(off, i)) |s| {
            try expectPf(on, j, s, deferred);
            i += s.calls;
            j += 3;
            sites += 1;
            continue;
        };
        try testing.expect(j < on.len);
        try testing.expect(sameCall(off[i], on[j]));
        // the deferred stream's marks: the off path's coefficient launch moves into PFDEC's triple; a join may move
        try testing.expectEqual(off[i].defer_side, on[j].defer_side);
        i += 1;
        j += 1;
    }
    try testing.expectEqual(on.len, j);
    return sites;
}

fn mixingSites(cs: []const Call) usize {
    var m: usize = 0;
    for (cs, 0..) |_, i| m += @intFromBool(offSite(cs, i) != null);
    return m;
}

/// MHC_DEFER's rules: one pending coefficient launch at a time, joined before the next reader and the program's end.
fn expectJoined(a: std.mem.Allocator, cs: []const Call) ![]const []const u8 {
    var pending = false;
    for (cs, 0..) |c, i| {
        errdefer std.debug.print("call {d} ({s}) reads coefficients still on the deferred stream\n", .{ i, c.name });
        if (c.defer_join) {
            try testing.expect(pending);
            pending = false;
        }
        const reads = eq(c.name, "tf_dsv41_mhc_pf_v1.run") or eq(c.name, "tf_dsv41_mhc_cuda_v1.run") or eq(c.name, "_site");
        if (reads) try testing.expect(!pending);
        if (c.defer_side) pending = true;
    }
    try testing.expect(!pending);
    return branches.sharedDeferred(a, cs);
}

/// With PFDEC the deferred launches share no role with main's calls that the off path's did not.
fn expectDeferred(a: std.mem.Allocator, off: []const Call, on: []const Call) !void {
    const so = try expectJoined(a, off);
    const sn = try expectJoined(a, on);
    for (sn) |r| {
        errdefer std.debug.print("new shared role {s}\n", .{r});
        for (so) |q| {
            if (eq(q, r)) break;
        } else return error.TestUnexpectedResult;
    }
}

test "TF_DSV41_MHC_PFDEC reader: unset / 0 off, 1..64 the fewest rows, anything else refused" {
    const Fake = struct {
        var v: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            return if (eq(name, "TF_DSV41_MHC_PFDEC")) v else null;
        }
    };
    Fake.v = null;
    try testing.expectEqual(@as(i64, 0), try pk.mhcPfRows(&Fake.get));
    for ([_][]const u8{ "0", "1", "17", " 64 " }, [_]i64{ 0, 1, 17, 64 }) |s, want| {
        Fake.v = s;
        try testing.expectEqual(want, try pk.mhcPfRows(&Fake.get));
    }
    for ([_][]const u8{ "65", "-1", "on", "" }) |s| {
        Fake.v = s;
        try testing.expectError(error.BadKnob, pk.mhcPfRows(&Fake.get));
    }
}

test "TF_DSV41_MHC_PFDEC: decode windows of 1-64 rows, each mixing site as mhc_pf + norm + coef_kernel on the same tensors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    for ([_]bool{ true, false }) |deferred| for ([_]i64{ 1, 17 }) |rows| for ([_]i64{ 1, 2, 5, 16, 17, 20, 24, 32, 33, 48, 64 }) |n| {
        errdefer std.debug.print("pfdec {d}, {d} rows, defer {}\n", .{ rows, n, deferred });
        const start: i64 = 300_000;
        const off = try block.emit(a, &cfg, &w, servedOptions(limit, 0, deferred), &backbone, n, start, true);
        const on = try block.emit(a, &cfg, &w, servedOptions(limit, rows, deferred), &backbone, n, start, true);
        const pf = n >= rows;
        const sites = try expectSwap(off, on, pf, deferred);
        // 40 layers' two sites (an Engram layer's site after its post-only included); none left for the old path
        try testing.expectEqual(if (pf) @as(usize, 2 * cfg.layers) else 0, sites);
        try testing.expectEqual(@as(usize, 0), if (pf) mixingSites(on) else 0);
        if (deferred) try expectDeferred(a, off, on);
        if (n > 1) {
            const rw = try rowmode.transform(a, on, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = servedOptions(limit, 0, false).pool.?.pts });
            try testing.expectEqual(on.len, rw.len);
        }
    };
}

test "TF_DSV41_MHC_PFDEC: DSpark passes (1 and 4 slots) take it at their in-place boundaries too; MHC_DEFER joins there" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    for ([_]bool{ true, false }) |deferred| for ([_]i64{ 1, 17 }) |rows| for ([_]i64{ 1, 4 }) |slots| {
        errdefer std.debug.print("pfdec {d}, {d} slots, defer {}\n", .{ rows, slots, deferred });
        const off = try dspark_emit.emitPassSlots(a, &cfg, &w, servedOptions(1 << 20, 0, deferred), cfg.dspark_block, slots, 4);
        const on = try dspark_emit.emitPassSlots(a, &cfg, &w, servedOptions(1 << 20, rows, deferred), cfg.dspark_block, slots, 4);
        const pf = slots * cfg.dspark_block >= rows;
        const sites = try expectSwap(off, on, pf, deferred);
        try testing.expectEqual(if (pf) @as(usize, 2 * cfg.mtp_layers) else 0, sites);
        // the off path too: block 0's site leaves its coefficients deferred, the next boundary and the final join them
        if (deferred) try expectDeferred(a, off, on);
    };
}

// -- TF_DSV41_MHC_DEFER_AT=exchange (calls.holdDeferred) ----------------------------------------------------------

/// `held` is `cs` with deferred coefficient launches moved later, still joined; returns those before an exchange.
fn expectHeld(a: std.mem.Allocator, cs: []const Call, held: []const Call) !usize {
    try testing.expectEqual(cs.len, held.len);
    var i: usize = 0;
    var j: usize = 0;
    var di: usize = 0;
    var dj: usize = 0;
    var at_exchange: usize = 0;
    while (true) {
        while (i < cs.len and cs[i].defer_side) i += 1;
        while (j < held.len and held[j].defer_side) j += 1;
        if (i == cs.len or j == held.len) break;
        errdefer std.debug.print("main call {d} ({s}) vs held {d} ({s})\n", .{ i, cs[i].name, j, held[j].name });
        try testing.expect(sameCall(cs[i], held[j]) and cs[i].defer_join == held[j].defer_join);
        i += 1;
        j += 1;
    }
    try testing.expect(i == cs.len and j == held.len);
    while (true) {
        while (di < cs.len and !cs[di].defer_side) di += 1;
        while (dj < held.len and !held[dj].defer_side) dj += 1;
        if (di == cs.len or dj == held.len) break;
        try testing.expect(sameCall(cs[di], held[dj]));
        try testing.expect(dj >= di); // only ever later
        const next = held[dj + 1];
        errdefer std.debug.print("deferred launch at {d} -> {d}, followed by {s}\n", .{ di, dj, next.name });
        try testing.expect(calls.isExchange(next) or next.defer_join);
        at_exchange += @intFromBool(calls.isExchange(next));
        di += 1;
        dj += 1;
    }
    try testing.expect(di == cs.len and dj == held.len);
    const so = try expectJoined(a, cs);
    const sn = try expectJoined(a, held);
    try testing.expectEqual(@as(usize, 0), so.len);
    try testing.expectEqual(@as(usize, 0), sn.len);
    return at_exchange;
}

test "TF_DSV41_MHC_DEFER_AT reader: unset / empty / 0 / off the reference's, exchange the hold, anything else refused" {
    for ([_]?[]const u8{ null, "", "0", "off", " OFF " }) |v| try testing.expect(!try branches.deferAt(v));
    for ([_][]const u8{ "exchange", "Exchange", " exchange\n" }) |v| try testing.expect(try branches.deferAt(v));
    for ([_][]const u8{ "1", "on", "experts", "exchanges" }) |v| try testing.expectError(error.BadDeferAt, branches.deferAt(v));
}

test "TF_DSV41_MHC_DEFER_AT=exchange: decode windows' coefficient launches each move to just before their sublayer's exchange, nothing else changes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    for ([_]i64{ 0, 17, 1 }) |pf| for ([_]i64{ 1, 2, 5, 16, 17, 20, 24, 32, 48, 64 }) |n| {
        errdefer std.debug.print("pfdec {d}, {d} rows\n", .{ pf, n });
        // the off path's windows past 16 rows run Triton's site with its coefficients inline: nothing deferred
        const deferred = pf > 0 and n >= pf or n <= 16;
        const o = servedOptions(limit, pf, true);
        var oh = o;
        oh.mhc_defer_hold = true;
        const cs = try block.emit(a, &cfg, &w, o, &backbone, n, 300_000, true);
        const held = try block.emit(a, &cfg, &w, oh, &backbone, n, 300_000, true);
        const moved = try expectHeld(a, cs, held);
        var coefs: usize = 0;
        for (held) |c| coefs += @intFromBool(c.defer_side);
        // every mixing site's launch (two a layer) ends up at its sublayer's exchange: none waits at a join
        try testing.expectEqual(if (deferred) @as(usize, 2 * cfg.layers) else 0, coefs);
        try testing.expectEqual(coefs, moved);
        if (n > 1) {
            const rw = try rowmode.transform(a, held, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = servedOptions(limit, 0, false).pool.?.pts });
            try testing.expectEqual(held.len, rw.len);
        }
        // and the hold is what the knob adds: off, the emitter's program is the old one call for call
        const again = try block.emit(a, &cfg, &w, o, &backbone, n, 300_000, true);
        for (cs, again) |x, y| try testing.expect(sameCall(x, y) and x.defer_side == y.defer_side and x.defer_join == y.defer_join);
    };
}

test "TF_DSV41_MHC_DEFER_AT=exchange: DSpark passes (1 and 4 slots) keep their calls, each deferred launch later and joined" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    for ([_]i64{ 0, 1, 17 }) |pf| for ([_]i64{ 1, 4 }) |slots| {
        errdefer std.debug.print("pfdec {d}, {d} slots\n", .{ pf, slots });
        const o = servedOptions(1 << 20, pf, true);
        var oh = o;
        oh.mhc_defer_hold = true;
        const cs = try dspark_emit.emitPassSlots(a, &cfg, &w, o, cfg.dspark_block, slots, 4);
        const held = try dspark_emit.emitPassSlots(a, &cfg, &w, oh, cfg.dspark_block, slots, 4);
        _ = try expectHeld(a, cs, held);
    };
}

test "calls.holdDeferred: a launch with no exchange or join after it, or carrying other marks, stays; never past another deferred launch" {
    const mk = struct {
        fn c(name: []const u8, glue: bool, ds: bool, dj: bool) Call {
            return .{ .triton = false, .glue = glue, .name = name, .args = &.{}, .defer_side = ds, .defer_join = dj };
        }
    };
    var cs = [_]Call{ mk.c("coef", false, true, false), mk.c("a", false, false, false), mk.c("b", false, false, false), mk.c("glue.exchange", true, false, false), mk.c("site", false, false, true), mk.c("coef2", false, true, false), mk.c("c", false, false, false) };
    calls.holdDeferred(&cs);
    const want = [_][]const u8{ "a", "b", "coef", "glue.exchange", "site", "coef2", "c" };
    for (cs, want) |x, y| try testing.expectEqualStrings(y, x.name);
    // a join before any exchange: the launch waits just before its join
    var js = [_]Call{ mk.c("coef", false, true, false), mk.c("a", false, false, false), mk.c("site", false, false, true), mk.c("glue.exchange", true, false, false) };
    calls.holdDeferred(&js);
    for (js, [_][]const u8{ "a", "coef", "site", "glue.exchange" }) |x, y| try testing.expectEqualStrings(y, x.name);
    // a branch-forking launch stays
    var fs = [_]Call{ mk.c("coef", false, true, false), mk.c("a", false, false, false), mk.c("glue.exchange", true, false, false) };
    fs[0].fork = true;
    calls.holdDeferred(&fs);
    try testing.expectEqualStrings("coef", fs[0].name);
}

test "TF_DSV41_MHC_DEFER_AT_ROWS: the reader; programs over the threshold keep the launch after its site, the rest hold it" {
    try testing.expectEqual(@as(i64, 0), try branches.deferAtRows(null));
    try testing.expectEqual(@as(i64, 0), try branches.deferAtRows(" "));
    try testing.expectEqual(@as(i64, 16), try branches.deferAtRows("16"));
    for ([_][]const u8{ "-1", "65", "x" }) |v| try testing.expectError(error.BadDeferAt, branches.deferAtRows(v));
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const o = servedOptions(1 << 20, 17, true);
    var oh = o;
    oh.mhc_defer_hold = true;
    oh.mhc_defer_hold_rows = 16;
    var ho = o;
    ho.mhc_defer_hold = true;
    for ([_]i64{ 1, 6, 16, 17, 24, 32 }) |n| {
        errdefer std.debug.print("{d} rows\n", .{n});
        const base = try block.emit(a, &cfg, &w, o, &backbone, n, 300_000, true);
        const all = try block.emit(a, &cfg, &w, ho, &backbone, n, 300_000, true);
        const cut = try block.emit(a, &cfg, &w, oh, &backbone, n, 300_000, true);
        // at or under the threshold: the held program; over it: the program without the hold, call for call
        const want = if (n <= 16) all else base;
        try testing.expectEqual(want.len, cut.len);
        for (want, cut) |x, y| try testing.expect(sameCall(x, y) and x.defer_side == y.defer_side and x.defer_join == y.defer_join);
    }
    try testing.expect(oh.holdsDeferred(16) and !oh.holdsDeferred(17) and ho.holdsDeferred(64) and !o.holdsDeferred(1));
}
