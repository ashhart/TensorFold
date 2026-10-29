//! Cortheon in the server (--cortheon): its reasoning program, web research before building, server-run web tools.
const std = @import("std");
const json = @import("json");
const chat = @import("../chat.zig");
const grammar = @import("../grammar.zig");
const prompt_mod = @import("../prompt.zig");
const tool_specs = @import("../tool_specs.zig");
const log = @import("../log.zig");
const Server = @import("../server.zig").Server;
const program = @import("program.zig");
const tools = @import("tools.zig");
const memory = @import("memory.zig");
const live_mod = @import("live.zig");
const Seen = @import("seen.zig").Seen;
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const loop = @import("loop.zig");

/// Web calls the model may make in one reply, past the research done for it before it starts.
pub const calls_per_reply = 6;

pub const State = struct {
    gpa: Allocator,
    io: std.Io,
    web: tools.Web,
    engine: []const u8,
    live: ?*live_mod.Live = null,
    memory: memory.Store,

    /// Cortheon over the live web; `brave_key` picks Brave Search over DuckDuckGo.
    pub fn init(gpa: Allocator, io: std.Io, user_agent: []const u8, brave_key: ?[]const u8) Allocator.Error!*State {
        const l = try gpa.create(live_mod.Live);
        l.* = .init(gpa, io, user_agent, brave_key);
        const s = try gpa.create(State);
        s.* = .{ .gpa = gpa, .io = io, .web = l.asWeb(), .engine = l.engine(), .live = l, .memory = .{ .gpa = gpa, .io = io } };
        return s;
    }

    /// Cortheon over another web (tests).
    pub fn initWith(gpa: Allocator, io: std.Io, web: tools.Web) Allocator.Error!*State {
        const s = try gpa.create(State);
        s.* = .{ .gpa = gpa, .io = io, .web = web, .engine = "test", .memory = .{ .gpa = gpa, .io = io } };
        return s;
    }

    pub fn deinit(s: *State) void {
        s.memory.deinit();
        if (s.live) |l| {
            l.deinit();
            s.gpa.destroy(l);
        }
        s.gpa.destroy(s);
    }
};

/// One request's Cortheon work: what it found, what the client never sees, and what it reports.
pub const Turn = struct {
    state: *State,
    id: []const u8,
    task: program.Task,
    goal: []const u8,
    research: bool,
    client_tools: bool,
    user_key: u64,
    today: []const u8,
    hidden: std.ArrayList(Value) = .empty,
    sources: std.ArrayList([]const u8) = .empty,
    checks: std.ArrayList([]const u8) = .empty,
    prose: std.ArrayList(u8) = .empty, // earlier rounds' prose, as the client already has it
    searches: u32 = 0,
    fetches: u32 = 0,
    rounds: u32 = 0,
    calls_left: u32 = calls_per_reply,
    focus: []const u8 = "", // the model's latest search this reply, which its next page reads are cut around
    seen: Seen = .{}, // what cortheon_fetch may read: URLs the conversation showed, never ones the model wrote

    /// Adds a message the client never sees to the conversation the model reads.
    pub fn hide(t: *Turn, a: Allocator, input: *chat.Input, m: Value) Allocator.Error!void {
        try t.hidden.append(a, m);
        const list = try a.alloc(Value, input.messages.array.len + 1);
        @memcpy(list[0 .. list.len - 1], input.messages.array);
        list[list.len - 1] = m;
        input.messages = .{ .array = list };
    }

    /// Keeps a round's prose: the client has it already, so the reply's content starts with it.
    pub fn addProse(t: *Turn, a: Allocator, prose: []const u8) Allocator.Error!void {
        const p = std.mem.trim(u8, prose, " \t\r\n");
        if (p.len == 0) return;
        if (t.prose.items.len > 0) try t.prose.appendSlice(a, "\n\n");
        try t.prose.appendSlice(a, p);
    }

    /// The reply's content as the client has it: earlier rounds' prose, then the last round's.
    pub fn visible(t: *const Turn, a: Allocator, last: []const u8) Allocator.Error![]const u8 {
        if (t.prose.items.len == 0) return last;
        if (std.mem.trim(u8, last, " \t\r\n").len == 0) return t.prose.items;
        return std.mem.concat(a, u8, &.{ t.prose.items, "\n\n", last });
    }

    pub fn noteSources(t: *Turn, a: Allocator, ran: tools.Ran) Allocator.Error!void {
        switch (ran.kind) {
            .search => t.searches += 1,
            .fetch => t.fetches += @intFromBool(ran.ok),
        }
        if (ran.kind == .fetch and ran.ok) for (ran.sources) |s| {
            for (t.sources.items) |seen| {
                if (std.mem.eql(u8, seen, s)) break;
            } else try t.sources.append(a, s);
        };
    }

    /// The reply's `cortheon` object: the task, the research and checks done, and the pages read.
    pub fn summary(t: *const Turn, a: Allocator) Allocator.Error!Value {
        const o = try json.newObject(a);
        try o.put(a, "task", .{ .string = @tagName(t.task.deliverable) });
        try o.put(a, "research_first", .{ .bool = t.research });
        try o.put(a, "rounds", try json.intValue(a, t.rounds));
        try o.put(a, "searches", try json.intValue(a, t.searches));
        try o.put(a, "pages_read", try json.intValue(a, t.fetches));
        const srcs = try a.alloc(Value, t.sources.items.len);
        for (t.sources.items, srcs) |s, *v| v.* = .{ .string = s };
        try o.put(a, "sources", .{ .array = srcs });
        const cks = try a.alloc(Value, t.checks.items.len);
        for (t.checks.items, cks) |s, *v| v.* = .{ .string = s };
        try o.put(a, "checks", .{ .array = cks });
        return .{ .object = o };
    }

    /// Keeps what the client never saw against the reply `message` it got, for the next turn's splice.
    pub fn remember(t: *const Turn, a: Allocator, input: chat.Input, message: Value) void {
        if (t.hidden.items.len == 0 or input.messages != .array) return;
        const list = a.alloc(Value, input.messages.array.len + 1) catch return;
        @memcpy(list[0 .. list.len - 1], input.messages.array);
        list[list.len - 1] = message;
        const text = json.stringify(a, .{ .array = t.hidden.items }, .{ .ascii = false }) catch return;
        const key = memory.replyKey(a, list, list.len - 1) catch return;
        t.state.memory.put(key, text);
    }

    /// One progress line: in the stream's reasoning when there is a stream, and in the log.
    pub fn progress(t: *const Turn, a: Allocator, sink: ?chat.Sink, comptime fmt: []const u8, args: anytype) chat.Failure!void {
        const line = try std.fmt.allocPrint(a, fmt, args);
        log.line("cortheon {s}: {s}", .{ t.id, line });
        const s = sink orelse return;
        const delta = try chat.deltaOf(a, "reasoning_content", try std.fmt.allocPrint(a, "Cortheon: {s}\n", .{line}));
        s.call(s.ctx, delta) catch return error.Cancelled;
    }
};

fn roleIs(m: Value, r: []const u8) bool {
    return std.mem.eql(u8, m.strField("role") orelse "", r);
}

/// The request's turn, its input given the note, the tools and earlier hidden messages; null when Cortheon stays out.
pub fn begin(srv: *Server, a: Allocator, raw: Value, id: []const u8, is_chat: bool, input: *chat.Input) ?*Turn {
    const state = srv.cortheon orelse return null;
    if (!is_chat or input.prompt != null or input.messages != .array or input.messages.array.len == 0) return null;
    if (raw.get("cortheon")) |v| if (v == .bool and !v.bool) return null;
    const choice = raw.get("tool_choice");
    if (tool_specs.choiceDisables(choice) or (tool_specs.choiceRequiresCall(a, choice) catch true)) return null;
    for (grammar.fields) |k| if (json.truthyField(raw, k)) return null;
    if (json.truthyField(raw, "logprobs")) return null;
    if (raw.get("priority")) |p| if (p == .string and std.mem.eql(u8, p.string, "background")) return null;
    if (prompt_mod.isTitle(input.messages, input.tools.len > 0) or utility(a, input)) return null;
    return start(state, srv, a, id, input) catch null;
}

/// A tool-less ask to transform text the request carries, named by its system prompt or a long user turn.
fn utility(a: Allocator, input: *const chat.Input) bool {
    if (input.tools.len > 0) return false;
    const msgs = input.messages.array;
    const lead = memory.text(a, msgs[0]) catch return false;
    if (roleIs(msgs[0], "system") and program.isUtility(lead)) return true;
    const last = msgs[msgs.len - 1];
    const words = memory.text(a, last) catch return false;
    return roleIs(last, "user") and words.len >= 200 and program.isUtility(words);
}

fn start(state: *State, srv: *Server, a: Allocator, id: []const u8, input: *chat.Input) !*Turn {
    const spliced = try memory.splice(a, &state.memory, input.messages.array);
    var first: ?[]const u8 = null;
    var latest: []const u8 = "";
    for (spliced) |m| if (roleIs(m, "user")) {
        const own = try program.ownWords(a, try memory.text(a, m));
        if (first == null) first = own;
        latest = own;
    };
    const task = program.classify(a, first orelse latest);
    const latest_task = program.classify(a, latest);
    const fresh_turn = roleIs(spliced[spliced.len - 1], "user");
    const t = try a.create(Turn);
    t.* = .{
        .state = state,
        .id = id,
        .task = task,
        .goal = latest[0..@import("html.zig").cutAt(latest, 4000)], // the cap does not split a character
        .research = fresh_turn and latest_task.frontier and std.mem.trim(u8, latest, " \t\r\n").len >= 12,
        .client_tools = input.tools.len > 0,
        .user_key = try memory.userKey(a, spliced),
        .today = try today(a, srv.io),
    };
    for (spliced) |m| if (!roleIs(m, "assistant")) try t.seen.addFrom(a, try memory.text(a, m));
    const note = try program.note(a, task, calls_per_reply);
    const list = try a.alloc(Value, spliced.len + @intFromBool(!roleIs(spliced[0], "system")));
    if (roleIs(spliced[0], "system")) {
        @memcpy(list, spliced);
        const sys = try json.copyObject(a, spliced[0].object);
        const words = try memory.text(a, spliced[0]);
        try sys.put(a, "content", .{ .string = try std.mem.concat(a, u8, &.{ words, "\n\n", note }) });
        list[0] = .{ .object = sys };
    } else {
        const sys = try json.newObject(a);
        try sys.put(a, "role", .{ .string = "system" });
        try sys.put(a, "content", .{ .string = note });
        list[0] = .{ .object = sys };
        @memcpy(list[1..], spliced);
    }
    input.messages = .{ .array = list };
    input.tools = try std.mem.concat(a, Value, &.{ input.tools, try tools.specs(a) });
    log.line("cortheon {s}: {s}{s}", .{ id, task.deliverable.label(), if (t.research) ", researching first" else "" });
    return t;
}

/// Today's UTC date, as results say when they were retrieved.
fn today(a: Allocator, io: std.Io) Allocator.Error![]const u8 {
    const secs: u64 = @intCast(@max(0, std.Io.Clock.real.now(io).toSeconds()));
    const day = std.time.epoch.EpochSeconds{ .secs = secs };
    const ymd = day.getEpochDay().calculateYearDay();
    const md = ymd.calculateMonthDay();
    return std.fmt.allocPrint(a, "{d:0>4}-{d:0>2}-{d:0>2}", .{ ymd.year, md.month.numeric(), md.day_index + 1 });
}

/// The calls the client gets: every call but Cortheon's own.
pub fn clientCalls(calls: ?[]Value) ?[]Value {
    const list = calls orelse return null;
    var n: usize = 0;
    for (list) |c| if (!tools.isServer(tools.callName(c))) {
        list[n] = c;
        n += 1;
    };
    return if (n > 0) list[0..n] else null;
}

test {
    _ = program;
    _ = tools;
    _ = memory;
    _ = live_mod;
    _ = @import("web.zig");
    _ = @import("html.zig");
    _ = @import("search.zig");
    _ = @import("quarantine.zig");
    _ = @import("checks.zig");
    _ = @import("seen.zig");
    _ = @import("budget.zig");
    _ = @import("fuzz.zig");
    _ = loop;
}
