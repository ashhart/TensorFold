//! Spark2.5's full-attention caches as pool pages against linear ones: replies and last-row logits byte-equal.
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const spark = @import("spark25");
const prompt_pages = @import("prompt_pages");
const check = @import("check.zig");
const Gpu = check.Gpu;

const native = spark.native;
const snap = spark.snap;
const be = spark.backend;

/// Prompt lengths: inside one page, across pages and the 512-row chunk grid, and one more through freed pages.
const lens = [_]usize{ 200, 1300, 700 };
/// A kept state inside the second prompt: off the chunk grid and the page grid.
const mark: u32 = 1000;

const Out = struct { ids: []u32, logits: []u8, cached: u64 };
const Result = struct { solo: [lens.len]Out, sampled: Out, pair: [2][]u32, resumed: Out, stored: ?Out = null };
/// TF_KV_SHARE: where the store keeps the second prompt, its last whole page short of the end.
const page_mark: u32 = 1280;

const Keep = struct {
    gb: *be.Spark,
    saved: ?*snap.Saved = null,
    fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const k: *@This() = @ptrCast(@alignCast(ptr));
        if (at == mark) k.saved = k.gb.saveState(s, at) catch null;
    }
};

const Case = struct { prompt: []const u32, drafts: bool, sampling: ?lanes.Sampling = null, reuse: lanes.stream.Reuse = .{} };

/// One stream, its prompt stepped as the lane host runs it: reply, last-row logits, rows resumed.
fn run(gpu: Gpu, engine: *lanes.Engine, gb: *be.Spark, c: Case, count: u32) !Out {
    const gpa = gpu.gpa;
    var s = try lanes.Stream.init(gpa, .{ .id = "g", .prompt = c.prompt, .max_new = count, .sampling = c.sampling, .drafts = c.drafts, .reuse = c.reuse });
    defer s.deinit(gpa);
    var first = true;
    while (!try engine.fillStream(&s, first)) first = false;
    const logits = try gpa.alloc(u8, spark.config.vocab * 2);
    errdefer gpa.free(logits);
    try gb.s.synchronize();
    const l = gb.lanes.get(&s) orelse return error.UnknownStream;
    try gb.d.check(gb.d.api.cuMemcpyDtoH_v2(logits.ptr, l.last, logits.len), "cuMemcpyDtoH");
    while (engine.activeCount() > 0) try engine.step();
    try idle(gb);
    return .{ .ids = try gpa.dupe(u32, s.emitted()), .logits = logits, .cached = if (s.reuse_failed) 0 else s.cached };
}

/// Two streams in shared rounds: their replies.
fn pair(gpu: Gpu, engine: *lanes.Engine, gb: *be.Spark, a: []const u32, b: []const u32, drafts: bool, count: u32) ![2][]u32 {
    const gpa = gpu.gpa;
    var sa = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = a, .max_new = count, .drafts = drafts });
    defer sa.deinit(gpa);
    var sb = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = b, .max_new = count, .drafts = drafts });
    defer sb.deinit(gpa);
    for ([_]*lanes.Stream{ &sa, &sb }) |s| {
        var first = true;
        while (!try engine.fillStream(s, first)) first = false;
    }
    while (engine.activeCount() > 0) try engine.step();
    try idle(gb);
    const ra = try gpa.dupe(u32, sa.emitted());
    errdefer gpa.free(ra);
    return .{ ra, try gpa.dupe(u32, sb.emitted()) };
}

/// Paged: every stream ended, so nothing is promised and every page is free again.
fn idle(gb: *const be.Spark) !void {
    const p = gb.pool orelse return;
    const c = p.census();
    try check.expect(p.idle() and (c.free == c.total or gb.prompts != null), "spark25 pages: the idle pool holds promised {d}, free {d} of {d}", .{ c.promised, c.free, c.total });
}

fn free(gpa: std.mem.Allocator, o: Out) void {
    gpa.free(o.ids);
    gpa.free(o.logits);
}

fn freeResult(gpa: std.mem.Allocator, r: Result) void {
    for (r.solo) |o| free(gpa, o);
    free(gpa, r.sampled);
    for (r.pair) |p| gpa.free(p);
    free(gpa, r.resumed);
    if (r.stored) |o| free(gpa, o);
}

/// One model with linear or paged full-layer caches through every case (copy drafts from the prompt's text).
fn pass(gpu: Gpu, model: []const u8, paged: bool, prompts: [lens.len][]const u32, scrub: []const u32, count: u32) !Result {
    const gpa = gpu.gpa;
    var loaded = try native.open(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 8192, .drafts = true, .lanes = 2, .kv_pages = paged });
    defer loaded.deinit(loaded.ctx);
    try check.expect((loaded.layout != null) == paged, "spark25 pages: open made the caches asked for", .{});
    const gb = loaded.target;
    if (paged) { // the two widest requests at once; the rest lease pages the ended ones gave back, in a shuffled order
        const pool = be.pagesFor(lens[0], count) + be.pagesFor(lens[1], count);
        var sequential: usize = 0;
        for (lens) |n| sequential += be.pagesFor(n, count);
        try check.expect(sequential > pool, "spark25 pages: the solo runs take more pages than the pool holds", .{});
        const kept = if (try core.kv_prompts.sharing() > 0) page_mark / spark.forward.page_tokens + snap.blobPages(gb.m, page_mark, page_mark) else 0;
        try native.attach(loaded.ctx, 1, pool + 2 * kept); // TF_KV_SHARE: room for the store's pages and rings, the scrub's too
        gb.shufflePages(0x6e4);
    }
    var cfg = try lanes.Config.init(gpa, loaded.facts, loaded.rows, loaded.rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, loaded.backend, clock.clock());
    defer engine.deinit();
    const drafts = true;

    var r: Result = undefined;
    r.stored = null;
    var made: usize = 0;
    errdefer for (r.solo[0..made]) |o| free(gpa, o);
    for (prompts, &r.solo) |p, *o| {
        o.* = try run(gpu, &engine, gb, .{ .prompt = p, .drafts = drafts }, count);
        made += 1;
    }
    const sp: lanes.Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    r.sampled = try run(gpu, &engine, gb, .{ .prompt = prompts[1], .drafts = drafts, .sampling = sp }, count);
    errdefer free(gpa, r.sampled);
    r.pair = try pair(gpu, &engine, gb, prompts[0], prompts[1], drafts, count);
    errdefer for (r.pair) |p| gpa.free(p);
    // a state kept at the mark, another prompt through the slot, then the prompt resumed from it
    var keep: Keep = .{ .gb = gb };
    defer if (keep.saved) |v| gb.dropState(v);
    var ms = [_]u32{mark};
    const kept = try run(gpu, &engine, gb, .{ .prompt = prompts[1][0 .. mark + 100], .drafts = drafts, .reuse = .{ .marks = &ms, .hook = .{ .ptr = &keep, .at = Keep.hook } } }, 2); // two: a reply of one ends at its first token
    free(gpa, kept);
    const v = keep.saved orelse return error.NothingKept;
    free(gpa, try run(gpu, &engine, gb, .{ .prompt = scrub, .drafts = drafts }, 2));
    r.resumed = try run(gpu, &engine, gb, .{ .prompt = prompts[1], .drafts = drafts, .reuse = .{ .saved = v, .at = mark } }, count);
    if (paged and try core.kv_prompts.sharing() > 0) r.stored = try stored(gpu, &engine, gb, prompts[1], scrub, count);
    return r;
}

/// TF_KV_SHARE: the store keeps the prompt at its page mark, then a resume adopts its pages; closing frees them.
fn stored(gpu: Gpu, engine: *lanes.Engine, gb: *be.Spark, prompt: []const u32, scrub: []const u32, count: u32) !Out {
    const gpa = gpu.gpa;
    const x = (try prompt_pages.Adapter.open(native, gpa, gb, spark.forward.page_tokens, 4)) orelse return error.NoPool;
    var closed = false;
    defer if (!closed) x.close();
    free(gpa, try run(gpu, engine, gb, .{ .prompt = prompt, .drafts = true }, 2));
    free(gpa, try run(gpu, engine, gb, .{ .prompt = scrub, .drafts = true }, 2));
    const out = try run(gpu, engine, gb, .{ .prompt = prompt, .drafts = true }, count);
    x.close();
    closed = true;
    const p = gb.pool.?;
    try check.expect(p.idle() and p.census().free == p.census().total, "spark25 pages: the store gave every page back when it closed", .{});
    return out;
}

fn same(bad: *usize, what: []const u8, a: []const u8, b: []const u8) void {
    const eq = std.mem.eql(u8, a, b);
    if (!eq) bad.* += 1;
    std.debug.print("RESULT spark25 pages {s}: {s}\n", .{ what, if (eq) "same" else "DIFFER" });
}

/// tf-cuda-test spark25-pages MODEL TEXT N: paged full-layer caches give the linear caches' bytes, case by case.
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
    var what: [96]u8 = undefined;
    for (linear.solo, paged.solo, lens) |a, b, n| {
        same(&bad, try std.fmt.bufPrint(&what, "prompt of {d}: reply", .{n}), std.mem.sliceAsBytes(a.ids), std.mem.sliceAsBytes(b.ids));
        same(&bad, try std.fmt.bufPrint(&what, "prompt of {d}: last-row logits", .{n}), a.logits, b.logits);
    }
    same(&bad, "sampled reply", std.mem.sliceAsBytes(linear.sampled.ids), std.mem.sliceAsBytes(paged.sampled.ids));
    same(&bad, "sampled last-row logits", linear.sampled.logits, paged.sampled.logits);
    for (linear.pair, paged.pair, 0..) |a, b, i| {
        same(&bad, try std.fmt.bufPrint(&what, "two streams, stream {d}: reply", .{i}), std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
        same(&bad, try std.fmt.bufPrint(&what, "two streams, stream {d}: paged == its solo reply", .{i}), std.mem.sliceAsBytes(b), std.mem.sliceAsBytes(paged.solo[i].ids));
    }
    same(&bad, "resumed at the mark: reply", std.mem.sliceAsBytes(linear.resumed.ids), std.mem.sliceAsBytes(paged.resumed.ids));
    same(&bad, "resumed at the mark: last-row logits", linear.resumed.logits, paged.resumed.logits);
    same(&bad, "paged resumed == paged fresh", std.mem.sliceAsBytes(paged.resumed.ids), std.mem.sliceAsBytes(paged.solo[1].ids));
    if (paged.resumed.cached != mark or linear.resumed.cached != mark) bad += 1;
    if (paged.stored) |st| {
        same(&bad, "TF_KV_SHARE resumed at its page mark: reply", std.mem.sliceAsBytes(linear.solo[1].ids), std.mem.sliceAsBytes(st.ids));
        same(&bad, "TF_KV_SHARE resumed at its page mark: last-row logits", linear.solo[1].logits, st.logits);
        if (st.cached != page_mark) bad += 1;
        std.debug.print("RESULT spark25 pages TF_KV_SHARE resume: {d} rows from the store's pages (want {d})\n", .{ st.cached, page_mark });
    }
    std.debug.print("RESULT spark25 pages resumed rows: linear {d}, paged {d} (want {d})\n", .{ linear.resumed.cached, paged.resumed.cached, mark });
    try check.expect(bad == 0, "spark25 pages: {d} checks failed", .{bad});
    check.pass("spark25 pages: full-layer caches in shuffled pool pages give the linear caches' replies and logits ({d} tokens, copy drafts), shared rounds and a kept state's resume included{s}, ended streams' pages reused", .{ count, if (paged.stored != null) " (and the TF_KV_SHARE store's)" else "" });
}
