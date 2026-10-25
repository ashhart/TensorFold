//! slide.learn end to end over a scripted model: a lesson given up between two kept ones leaves the later one's checks.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const prompts = @import("slide_prompts.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Renders every message as ROLE:CONTENT| and the reply's head as A:, bytes as tokens.
const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .encode_masked = encodeMasked } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |b, *id| id.* = b;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: Allocator, messages: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        var spans: std.ArrayList(model_text.Span) = .empty;
        errdefer spans.deinit(a);
        for (messages.items() orelse &.{}) |m| {
            try requestPart(a, &out, &spans, m.strField("role") orelse "");
            try out.append(a, ':');
            try requestPart(a, &out, &spans, m.strField("content") orelse "");
            try out.append(a, '|');
        }
        if (o.add_generation_prompt) try out.appendSlice(a, "A:") else try out.appendSlice(a, "E|");
        const bytes = try out.toOwnedSlice(a);
        errdefer a.free(bytes);
        return .{ .text = bytes, .spans = try spans.toOwnedSlice(a) };
    }
    fn requestPart(a: Allocator, out: *std.ArrayList(u8), spans: *std.ArrayList(model_text.Span), bytes: []const u8) !void {
        if (bytes.len == 0) return;
        const start = out.items.len;
        try out.appendSlice(a, bytes);
        try spans.append(a, .{ start, out.items.len });
    }
    fn encodeMasked(ctx: *anyopaque, a: Allocator, input: []const u8, spans: []const model_text.Span, problem: *[]const u8) model_text.Error![]u32 {
        var end: usize = 0;
        for (spans) |span| {
            if (span[0] < end or span[0] >= span[1] or span[1] > input.len) {
                problem.* = "invalid request span in byte tokenizer fixture";
                return error.Template;
            }
            end = span[1];
        }
        return encode(ctx, a, input, false);
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// A fact the scripted model knows how to be asked about: its words, the kind of thing, and the questions it gets.
const Topic = struct { fact: []const u8, value: []const u8, kind: []const u8 };

const topics = [_]Topic{
    .{ .fact = "I like blue.", .value = "blue", .kind = "colour" },
    .{ .fact = "I like apples.", .value = "apples", .kind = "fruit" },
    .{ .fact = "I like tea.", .value = "tea", .kind = "drink" },
};

fn has(text: []const u8, word: []const u8) bool {
    return std.ascii.findIgnoreCase(text, word) != null;
}

/// The scripted model: answers the lesson writer's prompts, and the taught model answers the facts it holds.
const Model = struct {
    gives_up: []const u8 = "", // a fact whose lessons never bring it back
    live: [topics.len]bool = @splat(false), // the facts the pocket holds now
    open: ?usize = null, // the fact the lesson running now teaches
    trained: std.ArrayList([]const u8) = .empty, // per kept lesson: the personal prompts it trained to keep, joined
    pending: std.ArrayList(u8) = .empty, // the open lesson's personal keep prompts
    lessons: usize = 0,
    mode: enum { none, construction, empty, training, save } = .none,
    fault_now: bool = false,
    failures: usize = 0,
    saves: usize = 0,
    drops: usize = 0,
    draft_index: ?usize = null,
    gpa: Allocator = testing.allocator,

    fn engine(m: *Model) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
    }
    fn deinit(m: *Model) void {
        for (m.trained.items) |t| m.gpa.free(t);
        m.trained.deinit(m.gpa);
        m.pending.deinit(m.gpa);
    }
    fn record(m: *Model, bytes: []const u8) !void {
        const kept = try m.gpa.dupe(u8, bytes);
        errdefer m.gpa.free(kept);
        try m.trained.append(m.gpa, kept);
        m.draft_index = m.trained.items.len - 1;
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn topicOf(text: []const u8) ?usize {
        // the fact's own words first, the first named (a writer's prompt quotes the fact before its examples)
        inline for (.{ "fact", "value", "kind" }) |field| {
            var best: ?usize = null;
            var at: usize = std.math.maxInt(usize);
            for (topics, 0..) |t, i| if (std.ascii.findIgnoreCase(text, @field(t, field))) |x| if (x < at) {
                at = x;
                best = i;
            };
            if (best) |b| return b;
        }
        return null;
    }
    fn questions(a: Allocator, t: Topic, n: usize) ![]const u8 {
        const forms = [_][]const u8{ "What {s} do I like?", "Which {s} do I like?", "What is my favourite {s}?", "Tell me the {s} I like.", "Which {s} is my favourite?", "What {s} did I say I like?", "Do you know what {s} I like?", "What {s} do I enjoy?", "Name the {s} I like.", "Which {s} do I love?", "What kind of {s} do I like?", "What {s} am I fond of?" };
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        for (0..@min(n, forms.len)) |i| {
            try out.print(a, "{d}. ", .{i + 1});
            const f = forms[i];
            const at = std.mem.indexOf(u8, f, "{s}").?;
            try out.print(a, "{s}{s}{s}\n", .{ f[0..at], t.kind, f[at + 3 ..] });
        }
        return out.items;
    }
    fn reply(m: *Model, a: Allocator, prompt: []const u8, untaught: bool) ![]const u8 {
        const user_at = std.mem.lastIndexOf(u8, prompt, "user:") orelse 0;
        const user = prompt[user_at..];
        const system = prompt[0..user_at];
        if (has(user, "Write ") and has(user, "questions")) {
            if (m.mode == .empty and m.fault_now) return "none";
            const t = topicOf(user) orelse return "1. What is my name?\n";
            if (has(user, "yes-or-no")) return "1. Do I like red?\n2. Do I like green?\n3. Do I like grey?\n4. Do I like pink?\n";
            if (has(user, "does not answer") or has(user, "something else")) return "1. What colour is my car?\n2. Where do I live?\n3. What is my job?\n4. What is my dog's name?\n5. What sport do I play?\n6. What is my age?\n";
            return questions(a, topics[t], 12);
        }
        if (has(user, "Rewrite each to start with")) return "1. Do you remember what colour I like?\n2. Do you know what fruit I like?\n";
        if (has(user, "write two questions worded almost the same")) return "1. What colour does my sister like?\n2. What colour does my brother like?\n";
        if (has(user, "List six other things")) return "car\nhouse\nshirt\nhat\nbike\nphone\n";
        if (has(user, "Reply with its number only")) return "0";
        if (has(user, "What language")) return "English";
        if (has(user, "Reply yes or no") or has(user, "(answer yes)") or has(user, "Does B tell")) {
            if (has(user, "Does it answer this question")) return if (topicOf(user)) |_| "yes" else "no";
            return "no";
        }
        if (has(user, "Who or what is this fact about")) return "I";
        if (has(user, "What is this question asking about")) return "what I like";
        if (has(user, "Using only that, answer")) return if (topicOf(user)) |t| topics[t].value else "not stated";
        // a question asked with the fact before it (the lesson's own answers), or of the taught model
        const t = topicOf(user) orelse return if (has(user, "Hello")) "Hello." else "I don't know.";
        if (has(system, "told you") or has(system, "You know this")) return if (has(user, "few words")) topics[t].value else try std.fmt.allocPrint(a, "You like {s}.", .{topics[t].value});
        if (!untaught and m.live[t]) return try std.fmt.allocPrint(a, "You like {s}.", .{topics[t].value});
        return "I don't know.";
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        var arena: std.heap.ArenaAllocator = .init(m.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const prompt = a.alloc(u8, request.prompt.len) catch return error.Busy;
        for (request.prompt, prompt) |x, *b| b.* = @intCast(x);
        if (m.mode == .construction and m.fault_now and std.mem.eql(u8, prompt, "user:" ++ prompts.personal_prompts[0] ++ "|A:")) {
            m.failures += 1;
            return error.Busy;
        }
        const said = m.reply(a, prompt, request.untaught) catch return error.Busy;
        const ids = a.alloc(u32, said.len) catch return error.Busy;
        for (said, ids) |b, *x| x.* = b;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn learn(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        if (request.train.len == 0 and !request.drop and !request.undo and !request.save and !request.reroute) {
            return sink.event(sink.ctx, &.{ .done = .{} });
        }
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (request.drop or request.undo) {
            m.drops += 1;
            if (m.open) |t| m.live[t] = false;
            if (m.draft_index) |_| m.gpa.free(m.trained.pop().?);
            m.draft_index = null;
            m.pending.clearRetainingCapacity();
            m.open = null;
            return sink.event(sink.ctx, &.{ .done = .{} });
        }
        if (request.train.len > 0 and !request.more) {
            // a new lesson: which fact it teaches, and the personal prompts it keeps as they are
            var buf: [512]u8 = undefined;
            const ex = request.train[0];
            const n = @min(buf.len, ex.ids.len);
            for (ex.ids[0..n], buf[0..n]) |x, *b| b.* = @intCast(x);
            m.open = topicOf(buf[0..n]);
            m.draft_index = null;
            if (m.mode == .training and m.open == 1) {
                m.failures += 1;
                return sink.event(sink.ctx, &.{ .done = .{ .message = "injected training failure" } });
            }
            m.pending.clearRetainingCapacity();
            for (request.keep) |k| {
                var kb: [512]u8 = undefined;
                const kn = @min(kb.len, k.ids.len);
                for (k.ids[0..kn], kb[0..kn]) |x, *b| b.* = @intCast(x);
                for (prompts.personal_prompts, 0..) |p, i| if (std.mem.indexOf(u8, kb[0..kn], p) != null) m.pending.print(m.gpa, "{d},", .{i}) catch return error.Busy;
            }
            m.record(m.pending.items) catch return error.Busy;
        }
        if (request.save) {
            m.saves += 1;
            if (m.mode == .save) return sink.event(sink.ctx, &.{ .done = .{ .message = "injected save failure" } });
        }
        const t = m.open orelse return sink.event(sink.ctx, &.{ .done = .{} });
        const back = !std.mem.eql(u8, topics[t].fact, m.gives_up);
        if (request.train.len > 0) {
            m.live[t] = back;
            sink.event(sink.ctx, &.{ .learned = .{ .recalled = back, .steps = 1, .loss = 0 } });
        }

        sink.event(sink.ctx, &.{ .done = .{} });
    }
};

/// The drain's poll slice, and how long it waits with no bytes and no end before it fails the request.
const drain_slice_ms = 50;
const drain_idle_ms = 10_000;

/// One POST /v1/slide/learn through the router, its whole event stream read.
fn learnOne(srv: *Server, text: []const u8) !void {
    var pair: [2]platform.posix.socket_t = undefined;
    if (platform.windows) {
        try platform.posix.socketpair(&pair);
    } else if (platform.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = platform.posix.system.close(pair[0]);
    defer _ = platform.posix.system.close(pair[1]);
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = try std.fmt.allocPrint(a, "{{\"text\": \"{s}\"}}", .{text});
    const head = try std.fmt.allocPrint(a, "POST /v1/slide/learn HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n{s}", .{ body.len, body });
    var sent: usize = 0;
    while (sent < head.len) {
        const n = platform.posix.system.write(pair[0], head[sent..].ptr, head.len - sent);
        if (n <= 0) return error.WriteFailed;
        sent += @intCast(n);
    }
    var conn = try Conn.init(testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 5000, .read_ms = 5000, .write_ms = 5000 };
    if (try conn.readRequest(a, false) != .ready) return error.BadRequest;
    const Drain = struct {
        wire: [128 * 1024]u8 = undefined,
        used: usize = 0,
        bad: bool = false,
        // macOS can leave a read() blocked on a socketpair after its peer's shutdown: poll in slices, which rechecks
        fn run(d: *@This(), fd: platform.posix.socket_t) void {
            var buf: [4096]u8 = undefined;
            var idle_ms: i32 = 0;
            while (true) {
                var fds = [_]platform.posix.pollfd{.{ .fd = fd, .events = platform.posix.POLL.IN, .revents = 0 }};
                const ready = platform.posix.poll(&fds, drain_slice_ms) catch {
                    d.bad = true;
                    return;
                };
                if (ready == 0) {
                    idle_ms += drain_slice_ms;
                    if (idle_ms < drain_idle_ms) continue;
                    d.bad = true; // no bytes and no end for drain_idle_ms: fail rather than wait forever
                    return;
                }
                idle_ms = 0;
                const n = platform.posix.system.read(fd, &buf, buf.len);
                if (n == 0) return;
                if (n < 0) {
                    d.bad = true;
                    return;
                }
                const count: usize = @intCast(n);
                if (count > d.wire.len - d.used) {
                    d.bad = true;
                    continue;
                }
                @memcpy(d.wire[d.used..][0..count], buf[0..count]);
                d.used += count;
            }
        }
    };
    var drained: Drain = .{};
    const drain = try std.Thread.spawn(.{}, Drain.run, .{ &drained, pair[0] });
    routes.dispatch(srv, &conn, a);
    const ended = platform.posix.system.shutdown(pair[1], platform.posix.SHUT.WR);
    const closed = platform.posix.system.shutdown(pair[0], platform.posix.SHUT.WR);
    drain.join();
    try testing.expectEqual(@as(@TypeOf(ended), 0), ended);
    try testing.expectEqual(@as(@TypeOf(closed), 0), closed);
    try testing.expect(!drained.bad);
    try testing.expect(std.mem.indexOf(u8, drained.wire[0..drained.used], "200 OK") != null);
    try testing.expect(std.mem.indexOf(u8, drained.wire[0..drained.used], "data: [DONE]") != null);
}

/// The personal prompts each kept lesson trained to keep, after teaching `facts` in turn (one gives up when told).
fn teach(gives_up: []const u8, facts: []const []const u8) !Model {
    return teachModel(.{ .gives_up = gives_up }, facts);
}

fn teachModel(model: Model, facts: []const []const u8) !Model {
    var text: Text = .{};
    var m = model;
    errdefer m.deinit();
    const srv = try Server.init(testing.allocator, testing.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    for (facts) |f| {
        m.fault_now = has(f, "I like apples.");
        try learnOne(srv, f);
    }
    m.lessons = srv.teacher.lessons;
    return m;
}

test "through slide.learn, a lesson given up between two kept ones leaves the second kept lesson's checks" {
    // the given-up lesson ("I like apples.") between the two kept ones, and the same two kept lessons alone
    var with = try teach("I like apples.", &.{ "I like blue.", "I like apples.", "I like tea." });
    defer with.deinit();
    var without = try teach("", &.{ "I like blue.", "I like tea." });
    defer without.deinit();
    try testing.expectEqual(@as(usize, 2), without.trained.items.len);
    try testing.expectEqual(@as(usize, 2), with.trained.items.len); // the given-up lesson kept nothing
    try testing.expectEqualStrings(without.trained.items[0], with.trained.items[0]);
    // the kept lesson after the given-up one trains on the personal prompts it would have had without it
    try testing.expectEqualStrings(without.trained.items[1], with.trained.items[1]);
    // control: the held-back prompts do turn from one kept lesson to the next
    try testing.expect(!std.mem.eql(u8, without.trained.items[0], without.trained.items[1]));
}

fn sameAccepted(actual: *const Model, expected: *const Model) !void {
    try testing.expectEqual(@as(usize, 2), actual.lessons);
    try testing.expectEqual(@as(usize, 2), actual.trained.items.len);
    try testing.expectEqual(expected.trained.items.len, actual.trained.items.len);
    for (actual.trained.items, expected.trained.items) |x, y| try testing.expectEqualStrings(y, x);
}

test "lesson accounting: construction refusal and no clean plan leave the next accepted checks unchanged" {
    var alone = try teach("", &.{ "I like blue.", "I like tea." });
    defer alone.deinit();
    inline for (.{ .construction, .empty }) |mode| {
        var failed = try teachModel(.{ .mode = mode }, &.{ "I like blue.", "I like apples.", "I like tea." });
        defer failed.deinit();
        try sameAccepted(&failed, &alone);
        try testing.expectEqual(@as(usize, 3), failed.saves);
        if (mode == .construction) try testing.expect(failed.failures > 0);
    }
}

test "lesson accounting: a failed learner is taken out without rotating the next accepted checks" {
    var alone = try teach("", &.{ "I like blue.", "I like tea." });
    defer alone.deinit();
    var failed = try teachModel(.{ .mode = .training }, &.{ "I like blue.", "I like apples.", "I like tea." });
    defer failed.deinit();
    try sameAccepted(&failed, &alone);
    try testing.expectEqual(@as(usize, 1), failed.failures);
    try testing.expect(failed.drops > 0);
    try testing.expect(!failed.live[1]);
}

test "lesson accounting: a persistence failure retains accepted live lessons and their next check rotation" {
    var alone = try teach("", &.{ "I like blue.", "I like tea." });
    defer alone.deinit();
    var failed = try teachModel(.{ .mode = .save }, &.{ "I like blue.", "I like tea." });
    defer failed.deinit();
    try sameAccepted(&failed, &alone);
    try testing.expectEqual(@as(usize, 2), failed.saves);
    try testing.expect(failed.live[0] and failed.live[2]);
}

test "lesson accounting: batching accepted and rejected facts does not alter later training chunks" {
    var separate = try teach("I like apples.", &.{ "I like blue.", "I like apples.", "I like tea." });
    defer separate.deinit();
    var batch = try teach("I like apples.", &.{"I like blue. I like apples. I like tea."});
    defer batch.deinit();
    try sameAccepted(&batch, &separate);
    try testing.expectEqual(@as(usize, 1), batch.saves);
    try testing.expect(!batch.live[1]);
}

test "lesson accounting: concurrent learn routes serialize the same accepted check rotations" {
    var solo = try teach("", &.{ "I like blue.", "I like tea." });
    defer solo.deinit();
    var text: Text = .{};
    var m: Model = .{};
    defer m.deinit();
    const srv = try Server.init(testing.allocator, testing.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const Job = struct {
        srv: *Server,
        fact: []const u8,
        failure: ?anyerror = null,
        fn run(j: *@This()) void {
            learnOne(j.srv, j.fact) catch |err| {
                j.failure = err;
            };
        }
    };
    var job: Job = .{ .srv = srv, .fact = "I like blue." };
    const thread = try std.Thread.spawn(.{}, Job.run, .{&job});
    var failure: ?anyerror = null;
    learnOne(srv, "I like tea.") catch |err| {
        failure = err;
    };
    thread.join();
    if (failure) |err| return err;
    if (job.failure) |err| return err;
    m.lessons = srv.teacher.lessons;
    try sameAccepted(&m, &solo);
}

test "lesson fixture: refused record allocations retain no partial owned training chunk" {
    for (0..2) |index| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = index });
        var m: Model = .{ .gpa = failing.allocator() };
        defer m.deinit();
        try testing.expectError(error.OutOfMemory, m.record("0,1,"));
        try testing.expectEqual(@as(usize, 0), m.trained.items.len);
        try testing.expectEqual(@as(?usize, null), m.draft_index);
    }
}

fn renderedFixture(a: Allocator) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const messages = switch (try json.parse(arena.allocator(), "[{\"role\":\"user\",\"content\":\"I like blue.\"},{\"role\":\"assistant\",\"content\":\"I like tea.\"}]")) {
        .ok => |value| value,
        .err => return error.BadFixture,
    };
    var text: Text = .{};
    var problem: []const u8 = "";
    const rendered = try text.text().render(a, messages, .{}, &problem);
    defer a.free(rendered.text);
    defer a.free(rendered.spans);
    try testing.expectEqualStrings("user:I like blue.|assistant:I like tea.|A:", rendered.text);
    try testing.expectEqual(@as(usize, 4), rendered.spans.len);
    for (rendered.spans) |span| try testing.expect(!rendered.isTemplate(span[0], span[1]));
    try testing.expect(rendered.isTemplate(4, 5));
    try testing.expect(rendered.isTemplate(rendered.text.len - 2, rendered.text.len));
    const ids = try rendered.encode(text.text(), a, &problem);
    defer a.free(ids);
    try testing.expectEqual(rendered.text.len, ids.len);
    for (rendered.text, ids) |byte, id| try testing.expectEqual(@as(u32, byte), id);
    try testing.expectError(error.Template, text.text().encodeMasked(a, rendered.text, &.{.{ 1, rendered.text.len + 1 }}, &problem));
}

test "lesson fixture: rendered byte tokens preserve request provenance and allocation ownership" {
    try testing.checkAllAllocationFailures(testing.allocator, renderedFixture, .{});
}
