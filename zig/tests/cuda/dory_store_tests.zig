//! Dory's prompts kept as pool pages: resumes and streams sharing pages give fresh replies; every page returns.
const std = @import("std");
const lanes = @import("lanes");
const dory = @import("dory");
const prompt_pages = @import("prompt_pages");
const check = @import("check.zig");
const serve = @import("dory_serve_tests.zig");
const Gpu = check.Gpu;

const be = dory.cuda_backend;
const tokens = dory.cuda_pages.tokens;
const Case = serve.Case;

/// Prompt length past the marks: over a chunk and a window, so rings wrap; a branch shares its first pages only.
const long = 3000;
const longer = 4500;
const branch = 1000;
/// Where the store's tail mark falls for a prompt of `n`.
fn tailMark(n: usize) u64 {
    return (n - 1) / tokens * tokens;
}

/// Streams filled a chunk each in turn, then run out together: each reply and the rows it resumed.
fn run(gpu: Gpu, db: *be.Dory, cases: []const Case, count: u32, cached: []u64) ![][]u32 {
    const gpa = gpu.gpa;
    var cfg = try lanes.Config.init(gpa, db.facts(), be.max_window, be.max_window - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, db.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |c, *s| s.* = try lanes.Stream.init(gpa, .{ .id = "d", .prompt = c.prompt, .max_new = count, .sampling = c.sampling, .drafts = false });
    defer for (streams) |*s| s.deinit(gpa);
    const filled = try gpa.alloc(u8, cases.len); // 0 not opened, 1 filling, 2 in the rounds
    defer gpa.free(filled);
    @memset(filled, 0);
    while (std.mem.indexOfNone(u8, filled, &.{2}) != null) { // a chunk a stream, so passes reach their marks together
        for (streams, filled) |*s, *f| if (f.* < 2) {
            const done = try engine.fillStream(s, f.* == 0);
            f.* = if (done) 2 else 1;
        };
    }
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams, cached) |*o, *s, *n| {
        o.* = try gpa.dupe(u32, s.emitted());
        n.* = s.cached;
    }
    return out;
}

/// One check: `cases` through the store, each reply against `want`, each stream at `from` rows resumed.
fn turn(gpu: Gpu, db: *be.Dory, cases: []const Case, want: []const []const u32, from: u64, count: u32, what: []const u8, bad: *usize) !void {
    var cached: [8]u64 = undefined;
    const got = try run(gpu, db, cases, count, cached[0..cases.len]);
    defer serve.freeRun(gpu.gpa, .{ .out = got, .secs = 0, .tokens = 0, .rounds = 0, .accepted = 0 });
    for (got, want, cached[0..cases.len], 0..) |g, w, c, i| {
        const eq = std.mem.eql(u32, g, w);
        if (!eq or c != from) bad.* += 1;
        std.debug.print("RESULT dory store {s} stream {d}: {d} rows resumed (want {d}), reply {s}\n", .{ what, i, c, from, if (eq) "same" else "DIFFER" });
    }
}

fn evenPool(db: *const be.Dory, what: []const u8) !void {
    const p = db.pool.?;
    const c = p.census();
    try check.expect(p.idle() and c.free + c.kept == c.total, "dory store: {s}: promised {d}, free {d} + kept {d} of {d}", .{ what, c.promised, c.free, c.kept, c.total });
}

/// tf-cuda-test dory-store MODEL LOOPS TEXT N (DORY_PAGED=1): turns through one store against fresh replies.
pub fn store(gpu: Gpu, rest: []const []const u8) !void {
    const gpa = gpu.gpa;
    var x: serve.Setup = undefined;
    try serve.setup(&x, gpu, rest, longer + 1, 4);
    defer serve.teardown(&x, gpa);
    try check.expect(x.rig.m.paged, "dory store: needs DORY_PAGED=1", .{});
    const a = x.ids[0..long];
    const b = x.ids[0..longer];
    const scrub = x.ids[1500..longer];
    const fork = try gpa.alloc(u32, long);
    defer gpa.free(fork);
    @memcpy(fork[0..branch], a[0..branch]);
    @memcpy(fork[branch..], x.ids[longer - (long - branch) ..][0 .. long - branch]);
    const sp: lanes.Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    const cases = [_]Case{ .{ .prompt = a }, .{ .prompt = a, .sampling = sp }, .{ .prompt = b }, .{ .prompt = fork } };
    var fresh: [cases.len][]u32 = undefined;
    for (&fresh, cases) |*f, c| {
        var n: [1]u64 = undefined;
        const r = try run(gpu, &x.db, &.{c}, x.count, &n);
        f.* = r[0];
        gpa.free(r);
    }
    defer for (fresh) |f| gpa.free(f);

    var bad: usize = 0;
    const s = (try prompt_pages.Adapter.open(dory.native, gpa, &x.db, tokens, 16)) orelse return error.NoPool;
    var open = true;
    defer if (open) s.close();
    const scrubs: [1]Case = .{.{ .prompt = scrub }};
    var unused: [1]u64 = undefined;
    const scrubbed = try run(gpu, &x.db, &scrubs, 2, &unused);
    serve.freeRun(gpa, .{ .out = scrubbed, .secs = 0, .tokens = 0, .rounds = 0, .accepted = 0 });
    try turn(gpu, &x.db, cases[0..1], fresh[0..1], 0, x.count, "first pass", &bad);
    const mid = try run(gpu, &x.db, &scrubs, 2, &unused);
    serve.freeRun(gpa, .{ .out = mid, .secs = 0, .tokens = 0, .rounds = 0, .accepted = 0 });
    try turn(gpu, &x.db, cases[0..1], fresh[0..1], tailMark(long), x.count, "resumed", &bad);
    try turn(gpu, &x.db, cases[1..2], fresh[1..2], tailMark(long), x.count, "resumed, sampled", &bad);
    try turn(gpu, &x.db, cases[2..3], fresh[2..3], tailMark(long), x.count, "extended", &bad);
    try turn(gpu, &x.db, cases[3..4], fresh[3..4], 0, x.count, "branch, its start below any state", &bad);
    try evenPool(&x.db, "after the turns");
    s.close();
    open = false;
    const c0 = x.db.pool.?.census();
    try check.expect(c0.free == c0.total, "dory store: closing left {d} of {d} pages free", .{ c0.free, c0.total });

    // an empty store: four streams open before any keeps, so three swap their pages for the first's mid-pass
    const s2 = (try prompt_pages.Adapter.open(dory.native, gpa, &x.db, tokens, 16)) orelse return error.NoPool;
    var open2 = true;
    defer if (open2) s2.close();
    const four: [4]Case = @splat(.{ .prompt = b });
    const want: [4][]const u32 = @splat(fresh[2]);
    try turn(gpu, &x.db, &four, &want, 0, x.count, "four at once", &bad);
    try turn(gpu, &x.db, &four, &want, tailMark(longer), x.count, "four resumed", &bad);
    try evenPool(&x.db, "after four streams");
    s2.close();
    open2 = false;
    try check.expect(bad == 0, "dory store: {d} checks failed", .{bad});
    check.pass("dory L={d} store: resumed, extended, branched and four sharing streams == fresh ({d} tokens), every page back", .{ x.loops, x.count });
}
