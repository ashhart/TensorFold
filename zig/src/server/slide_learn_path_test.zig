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
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |b, *id| id.* = b;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
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
    fn render(_: *anyopaque, a: Allocator, messages: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (messages.items() orelse &.{}) |m| try out.print(a, "{s}:{s}|", .{ m.strField("role") orelse "", m.strField("content") orelse "" });
        if (o.add_generation_prompt) try out.appendSlice(a, "A:") else try out.appendSlice(a, "E|");
        return out.toOwnedSlice(a);
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
    lessons: u32 = 0,
    gpa: Allocator = testing.allocator,

    fn engine(m: *Model) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
    }
    fn deinit(m: *Model) void {
        for (m.trained.items) |t| m.gpa.free(t);
        m.trained.deinit(m.gpa);
        m.pending.deinit(m.gpa);
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
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (request.drop or request.undo) {
            if (m.open) |t| m.live[t] = false;
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
            m.pending.clearRetainingCapacity();
            for (request.keep) |k| {
                var kb: [512]u8 = undefined;
                const kn = @min(kb.len, k.ids.len);
                for (k.ids[0..kn], kb[0..kn]) |x, *b| b.* = @intCast(x);
                for (prompts.personal_prompts, 0..) |p, i| if (std.mem.indexOf(u8, kb[0..kn], p) != null) m.pending.print(m.gpa, "{d},", .{i}) catch {};
            }
        }
        const t = m.open orelse return sink.event(sink.ctx, &.{ .done = .{} });
        const back = !std.mem.eql(u8, topics[t].fact, m.gives_up);
        if (request.train.len > 0) {
            m.live[t] = back;
            sink.event(sink.ctx, &.{ .learned = .{ .recalled = back, .steps = 1, .loss = 0 } });
        }
        if (request.save and back and m.pending.items.len > 0) {
            m.trained.append(m.gpa, m.gpa.dupe(u8, m.pending.items) catch "") catch {};
            m.pending.clearRetainingCapacity();
        }
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};

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
    _ = platform.posix.system.write(pair[0], head.ptr, head.len);
    var conn = try Conn.init(testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 5000, .read_ms = 5000, .write_ms = 5000 };
    if (try conn.readRequest(a, false) != .ready) return error.BadRequest;
    const Drain = struct {
        fn run(fd: platform.posix.socket_t) void {
            var buf: [4096]u8 = undefined;
            while (platform.posix.system.read(fd, &buf, buf.len) > 0) {}
        }
    };
    const drain = try std.Thread.spawn(.{}, Drain.run, .{pair[0]});
    routes.dispatch(srv, &conn, a);
    _ = platform.posix.system.shutdown(pair[1], platform.posix.SHUT.WR);
    _ = platform.posix.system.shutdown(pair[0], platform.posix.SHUT.WR);
    drain.join();
}

/// The personal prompts each kept lesson trained to keep, after teaching `facts` in turn (one gives up when told).
fn teach(gives_up: []const u8, facts: []const []const u8) !Model {
    var text: Text = .{};
    var m: Model = .{ .gives_up = gives_up };
    const srv = try Server.init(testing.allocator, testing.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    for (facts) |f| try learnOne(srv, f);
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
