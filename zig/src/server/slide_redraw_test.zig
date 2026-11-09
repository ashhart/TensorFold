//! Routed refusal retries draw repeatable fresh wordings; a kept first round matches the pinned main oracle.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const fixture = @import("slide_learn_path_test.zig");
const Server = @import("server.zig").Server;
const t = std.testing;
const A = std.mem.Allocator;
const Hash = [32]u8;

fn digest(value: anytype) !Hash {
    const bytes = try std.json.Stringify.valueAlloc(t.allocator, value, .{});
    defer t.allocator.free(bytes);
    var h: Hash = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return h;
}

const Model = struct {
    inner: fixture.Model = .{},
    fail_first: bool = false,
    fail_all: bool = false,
    separate_extra: bool = false,
    extra_version: ?u64 = null,
    continuations: [16]bool = @splat(false),
    restarts: [16]bool = @splat(false),
    refreshes: [16]bool = @splat(false),
    original_probes: [16]bool = @splat(false),
    report_false_first: bool = false,
    repeat_words: bool = false,
    fits: usize = 0,
    draws: usize = 0,
    seeds: [16]u64 = @splat(0),
    asked: [16]Hash = undefined,
    stable: [16]Hash = undefined,
    first: Hash = undefined,
    requests: [2048]Hash = undefined,
    calls: usize = 0,
    lock: std.Io.Mutex = .init,
    prompt_seed: bool = false,

    fn engine(m: *@This()) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *@This() = @ptrCast(@alignCast(ctx));
        const hash = digest(.{ .prompt = request.prompt, .sampling = request.sampling, .untaught = request.untaught, .decide = request.decide, .max_tokens = request.max_tokens }) catch return error.Busy;
        m.lock.lockUncancelable(t.io);
        std.debug.assert(m.calls < m.requests.len);
        m.requests[m.calls] = hash;
        m.calls += 1;
        m.lock.unlock(t.io);
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        const raw = arena.allocator().alloc(u8, request.prompt.len) catch return error.Busy;
        for (request.prompt, raw) |x, *v| v.* = @intCast(x);
        if (!request.untaught and request.sampling == null and m.fits > 0 and m.extra_version != null) {
            const version = std.fmt.allocPrint(arena.allocator(), "of version {d}?", .{m.extra_version.?}) catch return error.Busy;
            if (std.mem.indexOf(u8, raw, version) != null) {
                m.lock.lockUncancelable(t.io);
                m.original_probes[m.fits - 1] = true;
                m.lock.unlock(t.io);
            }
        }
        if (request.sampling != null or std.mem.indexOf(u8, raw, "different short questions that it answers") != null) {
            var sampled: u64 = 0;
            if (request.sampling) |sampling| {
                const i = m.draws;
                std.debug.assert(i < m.seeds.len);
                m.seeds[i] = sampling.seed;
                m.draws += 1;
                if (m.prompt_seed and sampling.seed != api.seedFor(request.prompt, 19)) return error.Busy;
                sampled = if (m.repeat_words) m.seeds[0] else sampling.seed;
                if (m.separate_extra) sampled +%= 1009 * @as(u64, @intCast(m.draws - 1));
                if (m.draws == 2) m.extra_version = sampled;
            }
            const a = arena.allocator();
            var words: std.ArrayList(u8) = .empty;
            for (1..13) |n| words.print(a, "{d}. Which colour do I like in wording {d} of version {d}?\n", .{ n, n, sampled }) catch return error.Busy;
            const ids = a.alloc(u32, words.items.len) catch return error.Busy;
            for (words.items, ids) |b, *x| x.* = b;
            sink.event(sink.ctx, id, &.{ .prefilled = 0 });
            sink.event(sink.ctx, id, &.{ .tokens = ids });
            return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
        }
        const e = m.inner.engine();
        return e.vtable.submit(e.ctx, id, request, sink);
    }
    fn learn(ctx: *anyopaque, req: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *@This() = @ptrCast(@alignCast(ctx));
        if (req.train.len > 0) {
            const i = m.fits;
            std.debug.assert(i < m.asked.len);
            m.continuations[i] = req.more;
            m.restarts[i] = req.reroute;
            m.refreshes[i] = req.refresh_wordings;
            m.asked[i] = digest(req.asked) catch return error.Busy;
            m.stable[i] = digest(.{ .train = req.train, .think = req.think, .held = req.held, .near = req.near, .keep = req.keep }) catch return error.Busy;
            if (i == 0) m.first = digest(.{ .stable = m.stable[i], .asked = req.asked, .unasked = req.unasked, .steps = req.steps, .more = req.more, .reroute = req.reroute }) catch return error.Busy;
            m.inner.gives_up = if (m.fail_all or (m.fail_first and i == 0)) "I like blue." else "";
            m.fits += 1;
        }
        const Forward = struct {
            sink: api.LearnSink,
            hide: bool,
            fn event(raw: *anyopaque, value: *const api.LearnEvent) void {
                const f: *@This() = @ptrCast(@alignCast(raw));
                var copy = value.*;
                if (f.hide and copy == .learned) copy.learned.recalled = false;
                f.sink.event(f.sink.ctx, &copy);
            }
        };
        var forward: Forward = .{ .sink = sink, .hide = m.report_false_first and m.fits == 1 and req.train.len > 0 };
        const e = m.inner.engine();
        return e.vtable.learn.?(e.ctx, req, .{ .ctx = &forward, .event = Forward.event });
    }
};

const Capture = struct {
    draws: usize,
    fits: usize,
    seeds: [16]u64,
    first: Hash,
    asked: [16]Hash,
    stable: [16]Hash,
    requests: Hash,
    graph: Hash,
    wire: []const u8,
    original_probes: [16]bool,
};

fn capture(a: A, fail_first: bool, prompt_seed: bool, took: bool, repeat_words: bool, waiting: bool) !Capture {
    var m: Model = .{ .fail_first = fail_first, .prompt_seed = prompt_seed, .repeat_words = repeat_words, .report_false_first = waiting };
    m.inner.took_leak = took;
    defer m.inner.deinit();
    var text: fixture.Text = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .seed_salt = 19, .prompt_seed = prompt_seed }, null);
    defer srv.deinit();
    const wire = try fixture.learnWire(a, srv, "{\"text\":\"I like blue.\"}");
    const graph = (try json.parse(a, try srv.slide.render(a, null))).ok;
    for (graph.field("facts").?.array) |f| try f.object.put(a, "at", try json.intValue(a, 0));
    const Less = struct {
        fn less(_: void, x: Hash, y: Hash) bool {
            return std.mem.order(u8, &x, &y) == .lt;
        }
    };
    std.mem.sort(Hash, m.requests[0..m.calls], {}, Less.less);
    var requests: Hash = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(m.requests[0..m.calls]), &requests, .{});
    return .{ .draws = m.draws, .fits = m.fits, .seeds = m.seeds, .first = m.first, .asked = m.asked, .stable = m.stable, .requests = requests, .graph = try digest(try json.stringify(a, graph, .{ .compact = true })), .wire = wire, .original_probes = m.original_probes };
}

test "slide redraw: a first-round pass matches main's wordings seeds and rows" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const Oracle = struct { prompt_seed: bool, draws: usize, seed: u64, first: []const u8, requests: []const u8, graph: []const u8 };
    const expected = try std.json.parseFromSliceLeaky([]const Oracle, arena.allocator(), @embedFile("fixtures/slide_redraw_first.json"), .{});
    for (expected) |old| {
        const c = try capture(arena.allocator(), false, old.prompt_seed, false, false, false);
        try t.expectEqual(old.draws, c.draws);
        try t.expectEqual(@as(usize, 1), c.fits);
        try t.expectEqual(old.seed, c.seeds[0]);
        try t.expectEqualStrings(old.first, try std.fmt.allocPrint(arena.allocator(), "{x}", .{c.first}));
        try t.expectEqualStrings(old.requests, try std.fmt.allocPrint(arena.allocator(), "{x}", .{c.requests}));
        try t.expectEqualStrings(old.graph, try std.fmt.allocPrint(arena.allocator(), "{x}", .{c.graph}));
    }
}

test "slide redraw: a refusal draws fresh repeatable wordings for the next round" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    for ([_]bool{ false, true }) |seeded| {
        const c = try capture(arena.allocator(), true, seeded, false, false, false);
        const again = try capture(arena.allocator(), true, seeded, false, false, false);
        try t.expect(c.draws > 2);
        try t.expect(c.fits >= 2);
        try t.expect(c.seeds[0] != c.seeds[c.draws - 1]);
        try t.expectEqualSlices(u64, c.seeds[0..c.draws], again.seeds[0..again.draws]);
        try t.expectEqualSlices(u8, &c.first, &again.first);
        try t.expectEqualSlices(u8, &c.requests, &again.requests);
        try t.expectEqualSlices(u8, &c.graph, &again.graph);
        try t.expectEqualSlices(u8, &c.stable[0], &c.stable[1]);
        try t.expect(!std.mem.eql(u8, &c.asked[0], &c.asked[1]));
    }
}

test "slide redraw: refusal still names the took sub-check" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const c = try capture(arena.allocator(), false, false, true, false, false);
    try t.expect(std.mem.indexOf(u8, c.wire, "sub-check took") != null);
    try t.expectEqual(@as(usize, 2), c.draws);
}

test "slide redraw: passing checks waiting for the held gate retain their wordings" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const c = try capture(arena.allocator(), false, false, false, false, true);
    try t.expectEqual(@as(usize, 2), c.fits);
    try t.expectEqual(@as(usize, 2), c.draws);
    try t.expectEqualSlices(u8, &c.asked[0], &c.asked[1]);
}

test "slide redraw: a repeated writer cannot remove existing unseen checks" {
    var m: Model = .{ .repeat_words = true };
    defer m.inner.deinit();
    var text: fixture.Text = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    var kept = [_]bool{false};
    var plan = (try @import("slide_lesson.zig").lesson(srv, &cx, &srv.teacher, &.{"I like blue."}, &kept, false, &.{}, &.{}, fixture.Stays{})).?;
    try t.expectEqual(@as(usize, 1), plan.wordings.len);
    const entries = try a.dupe(@import("slide_redraw.zig").Entry, plan.wordings);
    const held = try a.alloc(@import("slide_lesson.zig").Held, 4);
    @memset(held, plan.held[0]);
    const prior = try a.alloc([]const u8, 12);
    for (prior, 1..) |*q, n| q.* = try std.fmt.allocPrint(a, "Which colour do I like in wording {d} of version {d}?", .{ n, m.extra_version.? });
    entries[0].seen = prior;
    entries[0].probes = .{ .start = 0, .end = 4 };
    plan.probes = held;
    plan.wordings = entries;
    const id = srv.slide.add("I like blue.", "chat", 0);
    var redraw = true;
    const continued = try @import("slide_redraw.zig").prepare(srv, &cx, plan, 1, id, fixture.Stays{}, &redraw);
    try t.expect(!redraw);
    try t.expect(continued.full.asked.ptr == plan.full.asked.ptr and continued.probes.ptr == plan.probes.ptr);
    try t.expectEqual(@as(usize, 4), plan.probes.len);
    try t.expectEqual(@as(usize, 0), m.fits);
    m.fail_first = true;
    var conn: @import("http_conn.zig").Conn = undefined;
    var out: @import("slide.zig").Out = .{ .conn = &conn, .a = a, .open = false };
    const result = @import("slide.zig").rounds(srv, a, &cx, plan, fixture.Stays{}, id, &out);
    try t.expect(result.recalled[0] and result.why == null);
    try t.expectEqual(@as(usize, 2), m.fits);
    try t.expect(m.continuations[1] and !m.refreshes[1] and !m.restarts[1]);
    try t.expectEqualSlices(u8, &m.asked[0], &m.asked[1]);
    try t.expectEqualSlices(u8, &m.stable[0], &m.stable[1]);
}

test "slide redraw: refused rounds two through five continue rather than restart the fit" {
    var m: Model = .{ .fail_all = true, .separate_extra = true };
    defer m.inner.deinit();
    var text: fixture.Text = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    _ = try fixture.learnWire(arena.allocator(), srv, "{\"text\":\"I like blue.\"}");
    try t.expectEqual(@as(usize, 5), m.fits);
    try t.expect(!m.continuations[0] and !m.refreshes[0]);
    for (1..5) |i| {
        try t.expect(m.continuations[i] and m.refreshes[i] and !m.restarts[i]);
        try t.expect(m.original_probes[i]);
    }
}

test "slide redraw: routed mode keeps main continuation without fresh rows" {
    if (@import("core").profile.lookup("TF_SLIDE_ROUTED") == null and
        @import("core").profile.lookup("TF_SLIDE_EDIT") == null and
        @import("core").profile.lookup("TF_SLIDE_PROBE") == null) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const c = try capture(arena.allocator(), true, false, false, false, false);
    try t.expect(c.fits >= 2);
    try t.expectEqual(@as(usize, 2), c.draws);
    try t.expectEqualSlices(u8, &c.asked[0], &c.asked[1]);
}
