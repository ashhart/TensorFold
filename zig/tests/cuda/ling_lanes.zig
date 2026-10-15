//! `ling-lanes DIR OUT [REF] [--layers a,b,..]`: Ling's lane backend under upstream's round loop, no server.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const check = @import("check.zig");
const ling = @import("ling_cuda");
const rig = @import("ling_lanes_rig.zig");
const bytes = @import("ling_lanes_bytes.zig");

const Cover = rig.Cover;
const Engine = rig.Engine;

/// The prompt's tokens: a fixed walk over the vocabulary, as ling_forward.py's.
pub fn walk(a: std.mem.Allocator, n: usize, vocab: usize, salt: usize) ![]u32 {
    const p = try a.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast((37 * i + 11 + 101 * salt) % (vocab - 2));
    return p;
}

/// Rows a prompt chunk takes in these checks: several chunks a prompt, and not a multiple of 16.
const chunk = 37;

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const out = rest[1];
    var ref: ?[]const u8 = null;
    var thin: ?[]u32 = null;
    var i: usize = 2;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--layers") and i + 1 < rest.len) {
            var list: std.ArrayList(u32) = .empty;
            var it = std.mem.tokenizeScalar(u8, rest[i + 1], ',');
            while (it.next()) |x| try list.append(a, try std.fmt.parseInt(u32, x, 10));
            thin = list.items;
            i += 1;
        } else ref = rest[i];
    }
    const e = try Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, dir, .{ .capacity = 1024, .prompt_rows = 512, .layers = thin });
    defer e.deinit();
    e.chunk = chunk;
    var ll: ling.lanes.Lanes = .{ .e = e };
    defer ll.deinit();
    var cfg = try lanes.Config.init(gpu.gpa, ll.facts(), ling.lanes.max_rows, ling.lanes.max_rows - 1);
    defer cfg.deinit(gpu.gpa);
    var wall: lanes.backend.WallClock = .{ .io = gpu.io };
    var le = lanes.Engine.init(gpu.gpa, &cfg, ll.backend(), wall.clock());
    defer le.deinit();
    var c: Cover = .{ .gpa = gpu.gpa, .a = a, .le = &le };
    const V: usize = e.c.vocab;
    std.debug.print("RESULT ling-lanes {t}: {d} layers loaded of {d}, vocabulary {d}, prompt chunks of {d} rows\n", .{ e.c.quant, e.main.layers.len, e.c.layers, V, chunk });

    const p = try walk(a, 150, V, 0);
    // W, what every contract below holds to: the plain greedy pass, its prompt in 37-row chunks
    const w = (try c.run(.{ .prompt = p, .max_new = 160, .drafts = false })).ids;
    const copy_prompt = try std.mem.concat(a, u32, &.{ p, w[0..100], p });

    // chunked == whole: the prompt in one chunk, in 16-row chunks, and through `prefill` (no steps)
    for ([_]usize{ 512, 16 }) |rows| {
        e.chunk = rows;
        try c.same(try std.fmt.allocPrint(a, "prompt chunks of {d} rows, against chunks of {d}", .{ rows, chunk }), try c.run(.{ .prompt = p, .max_new = 160, .drafts = false }), w, .length);
    }
    e.chunk = chunk;
    ll.whole = true;
    le.backend = ll.backend();
    try c.same("the prompt pass whole (prefill, no steps)", try c.run(.{ .prompt = p, .max_new = 160, .drafts = false }), w, .length);
    ll.whole = false;
    le.backend = ll.backend();

    // drafted == plain: windows of 3 to 16 rows from a known reply, most with a wrong row; greedy and sampled
    const keyed: lanes.Sampling = .{ .seed = lanes.sampling.seedFor(copy_prompt, 0) };
    const plain_g = try c.run(.{ .prompt = copy_prompt, .max_new = 200, .drafts = false });
    const plain_s = try c.run(.{ .prompt = copy_prompt, .max_new = 200, .drafts = false, .sampling = keyed });
    for ([_]Cover.Got{ plain_g, plain_s }, [_]?lanes.Sampling{ null, keyed }) |plain, sampling| {
        const drafted = try c.run(.{ .prompt = copy_prompt, .max_new = 200, .sampling = sampling, .truth = plain.ids });
        try c.same(if (sampling != null) "drafted, sampled, against plain" else "drafted, greedy, against plain", drafted, plain.ids, .length);
        try c.expect(drafted.drafted > 0 and drafted.accepted > 0 and drafted.accepted < drafted.drafted, "drafted{s}: {d} drafts in {d} rounds, {d} kept", .{ if (sampling != null) ", sampled" else "", drafted.drafted, drafted.rounds, drafted.accepted });
    }
    const g = plain_g.ids;
    const differ = std.mem.indexOfDiff(u32, plain_s.ids, g);
    try c.expect(differ != null, "the sampled reply leaves the greedy one at token {?d} (sampling draws)", .{differ});

    // resumed == fresh: states kept at marks off the chunk grid and off multiples of 16, then passes resumed there
    {
        var keeper: rig.Keeper = .{ .e = e };
        defer keeper.drop();
        const marks = [_]u32{ 37, 101, 150, 263, 349 };
        const marked = try c.run(.{ .prompt = copy_prompt, .max_new = 64, .truth = g, .reuse = .{ .marks = &marks, .hook = keeper.hook() } });
        try c.same("a drafted pass cut at its marks, against the plain fresh pass", marked, g[0..64], .length);
        try c.expect(keeper.n == marks.len, "the pass kept {d} states at its marks ({d} asked)", .{ keeper.n, marks.len });
        for (keeper.kept[0..keeper.n], keeper.at[0..keeper.n]) |kept, at| {
            const got = try c.run(.{ .prompt = copy_prompt, .max_new = 64, .truth = g, .reuse = .{ .saved = kept, .at = at } });
            try c.expect(got.cached == at, "the pass resumed at {d} ({d} restored)", .{ at, got.cached });
            try c.same(try std.fmt.allocPrint(a, "resumed at {d}, drafted, against the plain fresh pass", .{at}), got, g[0..64], .length);
        }
        const other = try std.mem.concat(a, u32, &.{ copy_prompt[0..263], try walk(a, 60, V, 3) });
        const other_fresh = try c.run(.{ .prompt = other, .max_new = 64, .drafts = false });
        const got = try c.run(.{ .prompt = other, .max_new = 64, .drafts = false, .reuse = .{ .saved = keeper.kept[3], .at = 263 } });
        try c.same("another tail after the kept 263, against its fresh pass", got, other_fresh.ids, .length);
    }

    // cancels in the prompt pass, discards between rounds, eos, max_new at the window's edges
    for ([_]u32{ 0, 1, 3 }) |at| {
        var cut: rig.Cut = .{ .at = at };
        const r = c.run(.{ .prompt = copy_prompt, .max_new = 64, .cancel = &cut });
        try c.expect(if (r) |_| false else |err| err == error.Cancelled, "a cancel at ask {d} of the prompt pass ended it with error.Cancelled", .{at});
        try c.same(try std.fmt.allocPrint(a, "the request after a cancel at prompt ask {d}", .{at}), try c.run(.{ .prompt = p, .max_new = 64, .drafts = false }), w[0..64], .length);
    }
    for ([_]u32{ 1, 8 }) |rounds| {
        try c.expect(try c.discardAfter(copy_prompt, g, rounds), "a drafted stream was discarded after {d} rounds", .{rounds});
        try c.same(try std.fmt.allocPrint(a, "the request after a discard after {d} rounds", .{rounds}), try c.run(.{ .prompt = p, .max_new = 64, .drafts = false }), w[0..64], .length);
    }
    for ([_]usize{ 0, 9, 33 }) |from| {
        const j = for (from..g.len) |k| {
            if (std.mem.indexOfScalar(u32, g[0..k], g[k]) == null) break k;
        } else continue;
        try c.same(try std.fmt.allocPrint(a, "eos at token {d}, drafted", .{j}), try c.run(.{ .prompt = copy_prompt, .max_new = 512, .eos = &.{g[j]}, .truth = g }), g[0 .. j + 1], .stop);
    }
    for (1..4) |n| try c.same(try std.fmt.allocPrint(a, "max_new {d}, drafted", .{n}), try c.run(.{ .prompt = copy_prompt, .max_new = @intCast(n), .truth = g }), g[0..n], .length);

    // concurrent == solo
    try concurrent(&c, &ll, a, V);
    // the engine's own steps, byte for byte: cuts, resumes, windows and shared forwards against their twins
    try bytes.engineChecks(&c, e, copy_prompt, w);
    // the plain pass's ids and picks for ling_forward.py; with REF, its logits against float64
    try bytes.serial(&c, gpu, e, p, w, out, ref);
    if (c.failed) return error.TestFailed;
    check.pass("ling-lanes {t}: {d} checks", .{ e.c.quant, c.checks });
}

/// Each concurrent stream on one lanes.Engine equals its solo run.
fn concurrent(c: *Cover, ll: *ling.lanes.Lanes, a: std.mem.Allocator, V: usize) !void {
    const n = 96;
    var ps: [3][]const u32 = undefined;
    for (&ps, 0..) |*p, k| {
        const base = try walk(a, 60 + 23 * k, V, k + 1);
        p.* = try std.mem.concat(a, u32, &.{ base, base[0..40], base });
    }
    const seeds = [_]lanes.Sampling{ .{ .seed = 11 }, .{ .seed = 22 }, .{ .seed = 33 } };
    var solo: [3][]const u32 = undefined;
    var solo_s: [3][]const u32 = undefined;
    for (ps, &solo, &solo_s, 0..) |p, *g, *gs, k| {
        g.* = (try c.run(.{ .prompt = p, .max_new = n, .drafts = false })).ids;
        gs.* = (try c.run(.{ .prompt = p, .max_new = n, .drafts = false, .sampling = seeds[k] })).ids;
    }
    var sampled_seen = false; // the drafts' truth: the greedy solo replies, or the sampled ones in sampled cases
    const cases = [_]struct { name: []const u8, joins: []const rig.Join }{
        .{ .name = "all at once", .joins = &.{ .{ .p = 0 }, .{ .p = 1 }, .{ .p = 2 } } },
        .{ .name = "all at once, drafted", .joins = &.{ .{ .p = 0, .drafts = true }, .{ .p = 1, .drafts = true }, .{ .p = 2, .drafts = true } } },
        .{ .name = "staggered by 5 rounds, drafted", .joins = &.{ .{ .p = 0, .drafts = true }, .{ .p = 1, .at = 5, .drafts = true }, .{ .p = 2, .at = 10 } } },
        .{ .name = "in reverse", .joins = &.{ .{ .p = 2 }, .{ .p = 1, .drafts = true }, .{ .p = 0 } } },
        .{ .name = "the same prompt twice", .joins = &.{ .{ .p = 0, .drafts = true }, .{ .p = 0 } } },
        .{ .name = "sampled, drafted", .joins = &.{ .{ .p = 0, .sampled = true, .drafts = true }, .{ .p = 1, .sampled = true }, .{ .p = 2, .sampled = true, .drafts = true } } },
        .{ .name = "the second discarded after 6 rounds", .joins = &.{ .{ .p = 0 }, .{ .p = 1, .discard_at = 6 }, .{ .p = 2, .drafts = true } } },
        .{ .name = "prompts a chunk at a time while the others decode", .joins = &.{ .{ .p = 0, .drafts = true }, .{ .p = 1, .at = 2, .filled = true }, .{ .p = 2, .at = 4, .filled = true, .drafts = true } } },
        .{ .name = "sampled, prompts a chunk at a time", .joins = &.{ .{ .p = 0, .sampled = true }, .{ .p = 1, .at = 2, .sampled = true, .filled = true }, .{ .p = 2, .at = 4, .sampled = true, .filled = true, .drafts = true } } },
    };
    const before = ll.shared;
    for (cases) |cs| {
        sampled_seen = cs.joins[0].sampled;
        const got = try c.together(cs.joins, &ps, if (sampled_seen) &solo_s else &solo, &seeds, n);
        for (cs.joins, got.ids, 0..) |j, ids, k| {
            if (j.discard_at != null) continue;
            try c.same(try std.fmt.allocPrint(a, "{s}: stream {d} (prompt {d}{s}) against its solo run", .{ cs.name, k, j.p, if (j.sampled) ", sampled" else "" }), .{ .ids = ids, .reason = .length, .rounds = 0, .cached = 0, .drafted = 0, .accepted = 0 }, if (j.sampled) solo_s[j.p] else solo[j.p], .length);
        }
        var drafting = false;
        for (cs.joins) |j| drafting = drafting or j.drafts;
        try c.expect(got.most >= 2 and (got.drafted > 0) == drafting, "{s}: {d} streams at most in one round, {d} rounds with two or more, {d} drafts", .{ cs.name, got.most, got.shared, got.drafted });
        for (cs.joins) |j| if (j.filled) {
            try c.expect(got.between > 0, "{s}: {d} rounds of the others between a prompt's chunks", .{ cs.name, got.between });
            break;
        };
    }
    try c.expect(ll.shared > before, "{d} forwards held several streams' windows (Engine.verifyShared)", .{ll.shared - before});
}
