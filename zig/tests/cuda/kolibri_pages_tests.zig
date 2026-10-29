//! Kolibri-1's full layers as pool pages against linear caches: replies and last-row logits byte-equal, pages reused.
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const kolibri1 = @import("kolibri1");
const check = @import("check.zig");
const Gpu = check.Gpu;

const native = kolibri1.native;
const snap = kolibri1.snap;
const be = kolibri1.backend;

/// Prompt lengths: inside one 512-position page, across three pages, and one more through freed pages.
const lens = [_]usize{ 300, 1500, 900 };
/// A kept state inside the second prompt: off the page grid.
const mark: u32 = 1000;

const Out = struct { ids: []u32, logits: []u8, cached: u64 };
/// Each case plain, then drafted (copies from the prompt's text, verified as chain windows by tree attention).
const Result = struct { solo: [2][lens.len]Out, sampled: Out, pair: [2][2][]u32, resumed: [2]Out };

const Keep = struct {
    kb: *be.Kolibri,
    saved: ?*snap.Saved = null,
    fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const k: *@This() = @ptrCast(@alignCast(ptr));
        if (at == mark) k.saved = k.kb.saveState(s, at) catch null;
    }
};

const Case = struct { prompt: []const u32, drafts: bool, sampling: ?lanes.Sampling = null, reuse: lanes.stream.Reuse = .{} };

/// A stream as the lane host runs it, copy drafts from its own text when it drafts.
fn stream(gpa: std.mem.Allocator, id: []const u8, c: Case, count: u32, prop: *lanes.SuffixLookup) !lanes.Stream {
    prop.* = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
    errdefer prop.deinit();
    return lanes.Stream.init(gpa, .{ .id = id, .prompt = c.prompt, .max_new = count, .sampling = c.sampling, .drafts = c.drafts, .reuse = c.reuse, .proposer = if (c.drafts) prop.proposer() else null });
}

/// One stream, its prompt stepped as the lane host runs it: reply, last-row logits, rows resumed.
fn run(gpu: Gpu, engine: *lanes.Engine, kb: *be.Kolibri, c: Case, count: u32) !Out {
    const gpa = gpu.gpa;
    var prop: lanes.SuffixLookup = undefined;
    var s = try stream(gpa, "k", c, count, &prop);
    defer {
        s.deinit(gpa);
        prop.deinit();
    }
    var first = true;
    while (!try engine.fillStream(&s, first)) first = false;
    const logits = try gpa.alloc(u8, kb.m.c.vocab * 4);
    errdefer gpa.free(logits);
    try kb.s.synchronize();
    const l = kb.lanes.get(&s) orelse return error.UnknownStream;
    try kb.d.check(kb.d.api.cuMemcpyDtoH_v2(logits.ptr, l.last, logits.len), "cuMemcpyDtoH");
    while (engine.activeCount() > 0) try engine.step();
    try idle(kb);
    return .{ .ids = try gpa.dupe(u32, s.emitted()), .logits = logits, .cached = if (s.reuse_failed) 0 else s.cached };
}

/// Two streams in shared rounds: their replies.
fn pair(gpu: Gpu, engine: *lanes.Engine, kb: *be.Kolibri, a: []const u32, b: []const u32, drafts: bool, count: u32) ![2][]u32 {
    const gpa = gpu.gpa;
    var props: [2]lanes.SuffixLookup = undefined;
    var sa = try stream(gpa, "a", .{ .prompt = a, .drafts = drafts }, count, &props[0]);
    defer {
        sa.deinit(gpa);
        props[0].deinit();
    }
    var sb = try stream(gpa, "b", .{ .prompt = b, .drafts = drafts }, count, &props[1]);
    defer {
        sb.deinit(gpa);
        props[1].deinit();
    }
    for ([_]*lanes.Stream{ &sa, &sb }) |s| {
        var first = true;
        while (!try engine.fillStream(s, first)) first = false;
    }
    while (engine.activeCount() > 0) try engine.step();
    try idle(kb);
    const ra = try gpa.dupe(u32, sa.emitted());
    errdefer gpa.free(ra);
    return .{ ra, try gpa.dupe(u32, sb.emitted()) };
}

/// Paged: every stream ended, so nothing is promised and every page is free again.
fn idle(kb: *const be.Kolibri) !void {
    const p = kb.pool orelse return;
    const c = p.census();
    try check.expect(p.idle() and c.free == c.total, "kolibri pages: the idle pool holds promised {d}, free {d} of {d}", .{ c.promised, c.free, c.total });
}

fn free(gpa: std.mem.Allocator, o: Out) void {
    gpa.free(o.ids);
    gpa.free(o.logits);
}

fn freeResult(gpa: std.mem.Allocator, r: Result) void {
    for (r.solo) |row| for (row) |o| free(gpa, o);
    free(gpa, r.sampled);
    for (r.pair) |p| for (p) |x| gpa.free(x);
    for (r.resumed) |o| free(gpa, o);
}

/// One model with linear or paged full layers through every case, plain and drafted.
fn pass(gpu: Gpu, model: []const u8, paged: bool, prompts: [lens.len][]const u32, scrub: []const u32, count: u32) !Result {
    const gpa = gpu.gpa;
    var loaded = try native.open(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 8192, .drafts = true, .lanes = 2, .kv_pages = paged });
    defer loaded.deinit(loaded.ctx);
    try check.expect((loaded.layout != null) == paged, "kolibri pages: open made the caches asked for", .{});
    const kb = loaded.target;
    if (paged) { // the two widest requests at once; the rest lease pages the ended ones gave back, in a shuffled order
        const pool = be.pagesFor(lens[0], count) + be.pagesFor(lens[1], count);
        var sequential: usize = 0;
        for (lens) |n| sequential += be.pagesFor(n, count);
        try check.expect(sequential > pool, "kolibri pages: the solo runs take more pages than the pool holds", .{});
        try native.attach(loaded.ctx, 1, pool);
        kb.shufflePages(0x6e4);
    }
    var cfg = try lanes.Config.init(gpa, loaded.facts, loaded.rows, loaded.rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, loaded.backend, clock.clock());
    defer engine.deinit();

    var r: Result = undefined;
    var made: usize = 0;
    errdefer for (0..made) |i| free(gpa, r.solo[i / lens.len][i % lens.len]);
    for (0..2) |d| for (prompts, &r.solo[d]) |p, *o| {
        o.* = try run(gpu, &engine, kb, .{ .prompt = p, .drafts = d == 1 }, count);
        made += 1;
    };
    const sp: lanes.Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    r.sampled = try run(gpu, &engine, kb, .{ .prompt = prompts[1], .drafts = true, .sampling = sp }, count);
    errdefer free(gpa, r.sampled);
    r.pair[0] = try pair(gpu, &engine, kb, prompts[0], prompts[1], false, count);
    r.pair[1] = try pair(gpu, &engine, kb, prompts[0], prompts[1], true, count);
    // a state kept at the mark, another prompt through the slot, then the prompt resumed from it (plain, drafted)
    var keep: Keep = .{ .kb = kb };
    defer if (keep.saved) |v| kb.dropState(v);
    var ms = [_]u32{mark};
    free(gpa, try run(gpu, &engine, kb, .{ .prompt = prompts[1][0 .. mark + 100], .drafts = false, .reuse = .{ .marks = &ms, .hook = .{ .ptr = &keep, .at = Keep.hook } } }, 2));
    const v = keep.saved orelse return error.NothingKept;
    for (0..2) |d| {
        free(gpa, try run(gpu, &engine, kb, .{ .prompt = scrub, .drafts = false }, 2)); // two: a reply of one ends at its first token
        r.resumed[d] = try run(gpu, &engine, kb, .{ .prompt = prompts[1], .drafts = d == 1, .reuse = .{ .saved = v, .at = mark } }, count);
    }
    return r;
}

fn same(bad: *usize, what: []const u8, a: []const u8, b: []const u8) void {
    const eq = std.mem.eql(u8, a, b);
    if (!eq) bad.* += 1;
    std.debug.print("RESULT kolibri pages {s}: {s}\n", .{ what, if (eq) "same" else "DIFFER" });
}

/// tf-cuda-test kolibri-pages MODEL TEXT N: paged full layers give the linear caches' bytes, case by case.
pub fn pages(gpu: Gpu, rest: []const []const u8) !void {
    if (rest.len < 3) return error.MissingArguments;
    const gpa = gpu.gpa;
    const count = try std.fmt.parseInt(u32, rest[2], 10);
    var tok = try core.tokenizer.loadTokenizer(gpu.io, gpa, rest[0]);
    defer tok.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[1], gpa, .limited(1 << 26));
    defer gpa.free(text);
    const enc = try tok.encode(gpa, text);
    defer gpa.free(enc);
    if (enc.len == 0) return error.EmptyText;
    var buf: [lens.len + 1][lens[1]]u32 = undefined;
    var prompts: [lens.len][]const u32 = undefined;
    for (&buf, 0..) |*b, j| { // each prompt from its own place in the text; the last is the scrub's
        for (b, 0..) |*t, i| t.* = enc[(i + 977 * j) % enc.len];
    }
    for (&prompts, lens, 0..) |*p, n, j| p.* = buf[j][0..n];
    const scrub = buf[lens.len][0..lens[2]];

    const linear = try pass(gpu, rest[0], false, prompts, scrub, count);
    defer freeResult(gpa, linear);
    const paged = try pass(gpu, rest[0], true, prompts, scrub, count);
    defer freeResult(gpa, paged);
    var bad: usize = 0;
    var what: [112]u8 = undefined;
    const how = [_][]const u8{ "plain", "drafted" };
    for (0..2) |d| for (linear.solo[d], paged.solo[d], lens) |a, b, n| {
        same(&bad, try std.fmt.bufPrint(&what, "prompt of {d}, {s}: reply", .{ n, how[d] }), std.mem.sliceAsBytes(a.ids), std.mem.sliceAsBytes(b.ids));
        same(&bad, try std.fmt.bufPrint(&what, "prompt of {d}, {s}: last-row logits", .{ n, how[d] }), a.logits, b.logits);
    };
    same(&bad, "sampled drafted reply", std.mem.sliceAsBytes(linear.sampled.ids), std.mem.sliceAsBytes(paged.sampled.ids));
    same(&bad, "sampled last-row logits", linear.sampled.logits, paged.sampled.logits);
    for (0..2) |d| for (linear.pair[d], paged.pair[d], 0..) |a, b, i| {
        same(&bad, try std.fmt.bufPrint(&what, "two streams, {s}, stream {d}: reply", .{ how[d], i }), std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
        same(&bad, try std.fmt.bufPrint(&what, "two streams, {s}, stream {d}: paged == its solo reply", .{ how[d], i }), std.mem.sliceAsBytes(b), std.mem.sliceAsBytes(paged.solo[d][i].ids));
    };
    for (0..2) |d| {
        same(&bad, try std.fmt.bufPrint(&what, "resumed at the mark, {s}: reply", .{how[d]}), std.mem.sliceAsBytes(linear.resumed[d].ids), std.mem.sliceAsBytes(paged.resumed[d].ids));
        same(&bad, try std.fmt.bufPrint(&what, "resumed at the mark, {s}: last-row logits", .{how[d]}), linear.resumed[d].logits, paged.resumed[d].logits);
        same(&bad, try std.fmt.bufPrint(&what, "paged resumed == paged fresh, {s}", .{how[d]}), std.mem.sliceAsBytes(paged.resumed[d].ids), std.mem.sliceAsBytes(paged.solo[d][1].ids));
        if (paged.resumed[d].cached != mark or linear.resumed[d].cached != mark) bad += 1;
    }
    same(&bad, "drafted == plain (greedy, paged)", std.mem.sliceAsBytes(paged.solo[1][1].ids), std.mem.sliceAsBytes(paged.solo[0][1].ids));
    try check.expect(bad == 0, "kolibri pages: {d} checks failed", .{bad});
    check.pass("kolibri pages: full layers in shuffled pool pages give the linear caches' replies and logits ({d} tokens), plain and drafted, shared rounds and a kept state's resume included, ended streams' pages reused", .{count});
}
