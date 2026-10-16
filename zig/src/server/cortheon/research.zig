//! Research before building: the model names searches, the server runs them and reads the best pages, as hidden rounds.
const std = @import("std");
const json = @import("json");
const errors = @import("../errors.zig");
const chat = @import("../chat.zig");
const ids = @import("../ids.zig");
const Server = @import("../server.zig").Server;
const Turn = @import("cortheon.zig").Turn;
const tools = @import("tools.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const max_queries = 3;
const max_pages = 3;

const planner = "You write web search queries for a task that needs current information from the web. Reply with up to 3 short queries, one per line, and nothing else. Each query names the product, library or topic, with its version when the task gives one, and aims at official documentation, release notes, changelogs, papers or the project's repository.";

/// Hosts whose pages carry little readable text (video, social feeds).
const unreadable = [_][]const u8{ "youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com", "linkedin.com", "pinterest.com" };

/// Searches, then page reads, for the turn's goal, added to `input` as tool rounds the client never sees.
pub fn run(t: *Turn, srv: *Server, cx: *errors.Cx, input: *chat.Input, sink: ?chat.Sink, gone: anytype) chat.Failure!void {
    const a = cx.a;
    const queries = try plan(t, srv, a, gone);
    const budget = charBudget(srv, input.*);
    var calls: std.ArrayList(Value) = .empty;
    var results: std.ArrayList(Value) = .empty;
    var found: [max_queries][]const []const u8 = @splat(&.{});
    for (queries, 0..) |q, i| {
        if (gone.check()) return error.Cancelled;
        try t.progress(a, sink, "searching the web for \"{s}\"", .{q});
        const call = try toolCall(a, tools.search_name, "query", q);
        const ran = try tools.run(t.state.web, a, call, t.today, budget, t.goal, &t.seen);
        try t.noteSources(a, ran);
        if (!ran.ok or ran.sources.len == 0) try t.progress(a, sink, "no results: {s}", .{ran.text[0..@min(ran.text.len, 160)]});
        found[i] = ran.sources;
        try calls.append(a, call);
        try results.append(a, try toolMessage(a, call, ran.text));
    }
    if (calls.items.len == 0) return;
    try t.hide(a, input, try assistantCalls(a, calls.items));
    for (results.items) |m| try t.hide(a, input, m);
    const picks = try pick(a, found[0..queries.len]);
    if (picks.len == 0) return;
    const focus = try std.mem.concat(a, u8, &.{ try std.mem.join(a, " ", queries), " ", t.goal });
    calls = .empty;
    results = .empty;
    for (picks) |url| {
        if (gone.check()) return error.Cancelled;
        try t.progress(a, sink, "reading {s}", .{url});
        const call = try toolCall(a, tools.fetch_name, "url", url);
        const ran = try tools.run(t.state.web, a, call, t.today, budget, focus, &t.seen);
        try t.noteSources(a, ran);
        if (!ran.ok) try t.progress(a, sink, "{s}", .{ran.text[0..@min(ran.text.len, 200)]});
        try calls.append(a, call);
        try results.append(a, try toolMessage(a, call, ran.text));
    }
    try t.hide(a, input, try assistantCalls(a, calls.items));
    for (results.items) |m| try t.hide(a, input, m);
}

/// The model's queries for the goal (one short greedy reply, thinking off), or one made from the goal itself.
fn plan(t: *Turn, srv: *Server, a: Allocator, gone: anytype) chat.Failure![]const []const u8 {
    const msgs = try a.alloc(Value, 2);
    msgs[0] = try message(a, "system", planner);
    msgs[1] = try message(a, "user", try std.fmt.allocPrint(a, "Task:\n{s}", .{t.goal}));
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = false });
    try fields.put(a, "temperature", .{ .int = "0" });
    var scratch: errors.Cx = .{ .a = a };
    const reply = chat.run(srv, &scratch, .{ .messages = .{ .array = msgs }, .fields = .{ .object = fields }, .max_tokens = 160, .id = t.id }, null, gone) catch |e| switch (e) {
        error.Cancelled => return error.Cancelled,
        error.OutOfMemory => return error.OutOfMemory,
        else => return fallback(a, t.goal),
    };
    const list = try parseQueries(a, reply.content);
    return if (list.len > 0) list else fallback(a, t.goal);
}

/// Up to three queries from the planner's reply: one a line, list marks and quotes dropped, NONE ignored.
pub fn parseQueries(a: Allocator, text: []const u8) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        if (out.items.len >= max_queries) break;
        const line = std.mem.trim(u8, unmark(unmark(std.mem.trim(u8, raw, " \t\r"))), " \"'`");
        if (line.len < 3 or line[0] == '<' or std.ascii.eqlIgnoreCase(line, "none")) continue;
        const q = line[0..@min(line.len, 200)];
        for (out.items) |seen| {
            if (std.ascii.eqlIgnoreCase(seen, q)) break;
        } else try out.append(a, q);
    }
    return out.items;
}

/// A line without its list mark ("-", "*", "#", "1." or "1)"); a query that starts with a number keeps it.
fn unmark(line: []const u8) []const u8 {
    if (line.len > 1 and (line[0] == '-' or line[0] == '*' or line[0] == '#')) return std.mem.trimStart(u8, line[1..], " *#");
    var i: usize = 0;
    while (i < line.len and std.ascii.isDigit(line[i])) i += 1;
    if (i > 0 and i + 1 < line.len and (line[i] == '.' or line[i] == ')') and line[i + 1] == ' ') return std.mem.trimStart(u8, line[i + 1 ..], " ");
    return line;
}

/// One query from the goal: its first sentence of prose, at most twelve words.
pub fn fallback(a: Allocator, goal: []const u8) Allocator.Error![]const []const u8 {
    var lines = std.mem.splitScalar(u8, goal, '\n');
    while (lines.next()) |raw| {
        var line = std.mem.trim(u8, raw, " \t\r`#>-*");
        if (line.len < 3) continue;
        for (line, 0..) |ch, i| if ((ch == '?' or ch == '.' or ch == '!') and (i + 1 == line.len or line[i + 1] == ' ')) {
            line = line[0..i];
            break;
        };
        var words: std.ArrayList(u8) = .empty;
        var it = std.mem.tokenizeAny(u8, line, " \t");
        var n: usize = 0;
        while (it.next()) |w| : (n += 1) {
            if (n == 12 or words.items.len + w.len > 120) break;
            if (words.items.len > 0) try words.append(a, ' ');
            try words.appendSlice(a, w);
        }
        const one = try a.alloc([]const u8, 1);
        one[0] = words.items;
        return one;
    }
    return &.{};
}

fn hostOf(url: []const u8) []const u8 {
    const start = if (std.mem.indexOf(u8, url, "://")) |s| s + 3 else 0;
    const end = std.mem.indexOfAnyPos(u8, url, start, "/?#:") orelse url.len;
    const host = url[start..end];
    return if (std.mem.startsWith(u8, host, "www.")) host[4..] else host;
}

/// The pages to read: each query's best results in turn, without repeats or unreadable hosts.
pub fn pick(a: Allocator, found: []const []const []const u8) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var rank: usize = 0;
    while (out.items.len < max_pages and rank < 6) : (rank += 1) {
        for (found) |urls| {
            if (out.items.len >= max_pages or rank >= urls.len) continue;
            const u = urls[rank];
            const host = hostOf(u);
            var skip = false;
            for (unreadable) |h| skip = skip or std.ascii.eqlIgnoreCase(host, h) or std.ascii.endsWithIgnoreCase(host, h[0..]) and host.len > h.len and host[host.len - h.len - 1] == '.';
            for (out.items) |seen| skip = skip or std.mem.eql(u8, seen, u);
            if (!skip) try out.append(a, u);
        }
    }
    return out.items;
}

/// The model's share of the room left in the window for one tool result, in characters.
fn charBudget(srv: *const Server, input: chat.Input) usize {
    const window: usize = srv.info.context_window;
    if (window == 0) return 6000;
    var chars: usize = 0;
    if (input.messages == .array) for (input.messages.array) |m| {
        chars += (m.strField("content") orelse continue).len;
    };
    const used = chars / 3 + 1024 + @as(usize, @intCast(@max(0, srv.config.default_max_tokens)));
    if (used >= window) return 600;
    return std.math.clamp((window - used) * 3 / 8, 600, 6000);
}

fn message(a: Allocator, role: []const u8, content: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}

/// A call to one of Cortheon's tools, shaped as the model's own calls are.
pub fn toolCall(a: Allocator, name: []const u8, key: []const u8, value: []const u8) Allocator.Error!Value {
    const args = try json.newObject(a);
    try args.put(a, key, .{ .string = value });
    const f = try json.newObject(a);
    try f.put(a, "name", .{ .string = name });
    try f.put(a, "arguments", .{ .string = try json.stringify(a, .{ .object = args }, .{ .ascii = false, .compact = true }) });
    const c = try json.newObject(a);
    try c.put(a, "id", .{ .string = try ids.make(a, "call_", 24) });
    try c.put(a, "type", .{ .string = "function" });
    try c.put(a, "function", .{ .object = f });
    return .{ .object = c };
}

pub fn assistantCalls(a: Allocator, calls: []const Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "assistant" });
    try o.put(a, "content", .{ .string = "" });
    try o.put(a, "tool_calls", .{ .array = try a.dupe(Value, calls) });
    return .{ .object = o };
}

pub fn toolMessage(a: Allocator, call: Value, text: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "tool" });
    try o.put(a, "tool_call_id", .{ .string = call.strField("id") orelse "" });
    try o.put(a, "name", .{ .string = tools.callName(call) });
    try o.put(a, "content", .{ .string = text });
    return .{ .object = o };
}

test "planner replies become clean queries; a goal becomes one when the planner gives none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qs = try parseQueries(a, "1. Zig 0.17 std.http.Client docs\n- \"Zig 0.17 release notes\"\n\n2) zig std.Io async\n4. one too many\n");
    try std.testing.expectEqual(@as(usize, 3), qs.len);
    try std.testing.expectEqualStrings("Zig 0.17 std.http.Client docs", qs[0]);
    try std.testing.expectEqualStrings("Zig 0.17 release notes", qs[1]);
    try std.testing.expectEqualStrings("zig std.Io async", qs[2]);
    try std.testing.expectEqual(@as(usize, 0), (try parseQueries(a, "NONE")).len);
    try std.testing.expectEqualStrings("2025 tax brackets for Canada", (try parseQueries(a, "2025 tax brackets for Canada"))[0]);
    const f = try fallback(a, "```\n\nImplement a client for the latest Stripe API, with retries and idempotency keys and good tests please thanks\n");
    try std.testing.expectEqualStrings("Implement a client for the latest Stripe API, with retries and idempotency", f[0]);
    try std.testing.expectEqualStrings("What are the main changes in the latest Zig release", (try fallback(a, "What are the main changes in the latest Zig release? Name the version."))[0]);
}

test "pages are picked by rank across queries, without repeats or video hosts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const q1 = [_][]const u8{ "https://www.youtube.com/watch?v=1", "https://a.org/1", "https://b.org/1" };
    const q2 = [_][]const u8{ "https://a.org/1", "https://c.org/2" };
    const got = try pick(a, &.{ &q1, &q2 });
    try std.testing.expectEqual(@as(usize, 3), got.len);
    try std.testing.expectEqualStrings("https://a.org/1", got[0]);
    try std.testing.expectEqualStrings("https://c.org/2", got[1]);
    try std.testing.expectEqualStrings("https://b.org/1", got[2]);
}
