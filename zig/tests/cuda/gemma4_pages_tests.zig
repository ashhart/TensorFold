//! Gemma 4's global caches as pool pages against linear ones: replies and last-row logits byte-equal, pages reused.
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const g4 = @import("gemma4_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const native = g4.native31;
const snap = g4.snap;
const be = g4.backend;

/// Prompt lengths: inside one page, across pages and the 512-row chunk grid, and one more through freed pages.
const lens = [_]usize{ 200, 1300, 700 };
/// A kept state inside the second prompt: off the chunk grid and the page grid.
const mark: u32 = 1000;

const Out = struct { ids: []u32, logits: []u8, cached: u64 };
const Result = struct { solo: [lens.len]Out, sampled: Out, pair: [2][]u32, resumed: Out };
const Opened = @typeInfo(@typeInfo(@TypeOf(native.openDrafter)).@"fn".return_type.?).error_union.payload;

const Keep = struct {
    gb: *be.Gemma,
    saved: ?*snap.Saved = null,
    fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const k: *@This() = @ptrCast(@alignCast(ptr));
        if (at == mark) k.saved = k.gb.saveState(s, at) catch null;
    }
};

const Case = struct { prompt: []const u32, drafts: bool, sampling: ?lanes.Sampling = null, reuse: lanes.stream.Reuse = .{} };

/// One stream, its prompt stepped as the lane host runs it: reply, last-row logits, rows resumed.
fn run(gpu: Gpu, engine: *lanes.Engine, gb: *be.Gemma, c: Case, count: u32) !Out {
    const gpa = gpu.gpa;
    var s = try lanes.Stream.init(gpa, .{ .id = "g", .prompt = c.prompt, .max_new = count, .sampling = c.sampling, .drafts = c.drafts, .reuse = c.reuse });
    defer s.deinit(gpa);
    var first = true;
    while (!try engine.fillStream(&s, first)) first = false;
    const logits = try gpa.alloc(u8, g4.config.vocab * 2);
    errdefer gpa.free(logits);
    try gb.s.synchronize();
    const l = gb.lanes.get(&s) orelse return error.UnknownStream;
    try gb.d.check(gb.d.api.cuMemcpyDtoH_v2(logits.ptr, l.last, logits.len), "cuMemcpyDtoH");
    while (engine.activeCount() > 0) try engine.step();
    try idle(gb);
    return .{ .ids = try gpa.dupe(u32, s.emitted()), .logits = logits, .cached = if (s.reuse_failed) 0 else s.cached };
}

/// Two streams in shared rounds: their replies.
fn pair(gpu: Gpu, engine: *lanes.Engine, gb: *be.Gemma, a: []const u32, b: []const u32, drafts: bool, count: u32) ![2][]u32 {
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
fn idle(gb: *const be.Gemma) !void {
    const p = gb.pool orelse return;
    const c = p.census();
    try check.expect(p.idle() and c.free == c.total, "gemma4 pages: the idle pool holds promised {d}, free {d} of {d}", .{ c.promised, c.free, c.total });
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
}

/// One model with linear or paged global caches through every case.
fn pass(gpu: Gpu, model: []const u8, drafter: ?[]const u8, paged: bool, prompts: [lens.len][]const u32, scrub: []const u32, count: u32) !Result {
    const gpa = gpu.gpa;
    var loaded = try native.open(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 8192, .drafts = drafter != null, .lanes = 2, .drafter = drafter, .kv_pages = paged });
    defer loaded.deinit(loaded.ctx);
    try check.expect((loaded.layout != null) == paged, "gemma4 pages: open made the caches asked for", .{});
    var opened: ?Opened = null;
    defer if (opened) |o| o.close(o.ctx);
    if (drafter) |dir| opened = try native.openDrafter(gpa, gpu.io, loaded.ctx, dir, 4);
    const gb = loaded.target;
    if (paged) { // the two widest requests at once; the rest lease pages the ended ones gave back, in a shuffled order
        const pool = g4.pages.pagesFor(lens[0], count) + g4.pages.pagesFor(lens[1], count);
        var sequential: usize = 0;
        for (lens) |n| sequential += g4.pages.pagesFor(n, count);
        try check.expect(sequential > pool, "gemma4 pages: the solo runs take more pages than the pool holds", .{});
        try native.attach(loaded.ctx, 1, pool);
        g4.pages.shufflePages(gb, 0x6e4);
    }
    var drafted: ?lanes.drafted.Drafted = if (opened) |o| try lanes.drafted.Drafted.init(gpa, loaded.backend, o.drafter) else null;
    defer if (drafted) |*d| d.deinit();
    const facts = if (drafted) |*d| d.facts(loaded.facts) else loaded.facts;
    var cfg = try lanes.Config.init(gpa, facts, loaded.rows, loaded.rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, if (drafted) |*d| d.backend() else loaded.backend, clock.clock());
    defer engine.deinit();
    const drafts = drafter != null;

    var r: Result = undefined;
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
    return r;
}

fn same(bad: *usize, what: []const u8, a: []const u8, b: []const u8) void {
    const eq = std.mem.eql(u8, a, b);
    if (!eq) bad.* += 1;
    std.debug.print("RESULT gemma4 pages {s}: {s}\n", .{ what, if (eq) "same" else "DIFFER" });
}

/// tf-cuda-test gemma4-pages MODEL TEXT N [DRAFTER]: paged global caches give the linear caches' bytes, case by case.
pub fn pages(gpu: Gpu, rest: []const []const u8) !void {
    if (rest.len < 3) return error.MissingArguments;
    const gpa = gpu.gpa;
    const count = try std.fmt.parseInt(u32, rest[2], 10);
    const drafter: ?[]const u8 = if (rest.len > 3) rest[3] else null;
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
        b[0] = 2;
        for (b[1..], 0..) |*t, i| t.* = enc[(i + 977 * j) % enc.len];
    }
    for (&prompts, lens, 0..) |*p, n, j| p.* = buf[j][0..n];
    const scrub = buf[lens.len][0..lens[2]];

    const linear = try pass(gpu, rest[0], drafter, false, prompts, scrub, count);
    defer freeResult(gpa, linear);
    const paged = try pass(gpu, rest[0], drafter, true, prompts, scrub, count);
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
    std.debug.print("RESULT gemma4 pages resumed rows: linear {d}, paged {d} (want {d})\n", .{ linear.resumed.cached, paged.resumed.cached, mark });
    try check.expect(bad == 0, "gemma4 pages: {d} checks failed", .{bad});
    check.pass("gemma4 pages: global caches in shuffled pool pages give the linear caches' replies and logits ({d} tokens{s}), shared rounds and a kept state's resume included, ended streams' pages reused", .{ count, if (drafter != null) ", drafted" else "" });
}
