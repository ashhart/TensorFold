//! A chat's conversation and its commands: the system prompt, the kept turns, and what a typed line asks for.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

/// One exchange the next request carries: the user's message and the answer.
pub const Turn = struct { user: []const u8, assistant: []const u8 };

pub const History = struct {
    /// Holds the kept texts for the whole chat.
    a: Allocator,
    system: ?[]const u8 = null,
    turns: std.ArrayList(Turn) = .empty,

    /// Forgets every turn; the system prompt stays.
    pub fn clear(h: *History) void {
        h.turns.clearRetainingCapacity();
    }

    /// The system prompt the next request starts with; empty text removes it.
    pub fn setSystem(h: *History, text: []const u8) !void {
        h.system = if (text.len == 0) null else try h.a.dupe(u8, text);
    }

    /// Keeps a copy of a finished or stopped exchange.
    pub fn keep(h: *History, user: []const u8, assistant: []const u8) !void {
        try h.turns.append(h.a, .{ .user = try h.a.dupe(u8, user), .assistant = try h.a.dupe(u8, assistant) });
    }

    /// The request's messages in ``a``: the system prompt, every kept turn in order, then ``user``.
    pub fn messages(h: *const History, a: Allocator, user: []const u8) ![]json.Value {
        var out: std.ArrayList(json.Value) = .empty;
        if (h.system) |s| try out.append(a, try message(a, "system", s));
        for (h.turns.items) |t| {
            try out.append(a, try message(a, "user", t.user));
            try out.append(a, try message(a, "assistant", t.assistant));
        }
        try out.append(a, try message(a, "user", user));
        return out.items;
    }
};

fn message(a: Allocator, role: []const u8, content: []const u8) !json.Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}

pub const Command = union(enum) {
    /// Text to send, trimmed.
    say: []const u8,
    /// A blank line.
    empty,
    bye,
    clear,
    help,
    /// ``/system``'s text; empty removes the prompt.
    system: []const u8,
    /// A ``/word`` that names no command.
    unknown: []const u8,
};

/// What a typed line asks for: a command when its first character is ``/``, else a message.
pub fn command(line: []const u8) Command {
    const raw = std.mem.trimEnd(u8, line, "\r\n");
    if (!std.mem.startsWith(u8, raw, "/")) {
        const text = std.mem.trim(u8, raw, " \t");
        return if (text.len == 0) .empty else .{ .say = text };
    }
    const end = std.mem.indexOfAny(u8, raw, " \t") orelse raw.len;
    const word = raw[0..end];
    const rest = std.mem.trim(u8, raw[end..], " \t");
    if (std.mem.eql(u8, word, "/bye")) return .bye;
    if (std.mem.eql(u8, word, "/clear")) return .clear;
    if (std.mem.eql(u8, word, "/help")) return .help;
    if (std.mem.eql(u8, word, "/system")) return .{ .system = rest };
    return .{ .unknown = word };
}

/// ``/help``'s list.
pub const help_text =
    \\/bye           end the chat (Ctrl-D does too)
    \\/clear         forget the conversation; the system prompt stays
    \\/system TEXT   set the system prompt; /system alone removes it
    \\/help          show this list
    \\Ctrl-C stops a reply. A line that starts with a space is sent as it is, even when a / follows.
    \\
;

test "chat CLI: a typed line is a command only when it starts with /" {
    const C = Command;
    try std.testing.expectEqualDeep(C{ .say = "hello there" }, command("  hello there \r\n"));
    try std.testing.expectEqualDeep(C.empty, command(" \t\n"));
    try std.testing.expectEqualDeep(C.bye, command("/bye\n"));
    try std.testing.expectEqualDeep(C.clear, command("/clear"));
    try std.testing.expectEqualDeep(C.help, command("/help me"));
    try std.testing.expectEqualDeep(C{ .system = "Answer in French." }, command("/system   Answer in French.  \n"));
    try std.testing.expectEqualDeep(C{ .system = "" }, command("/system\n"));
    try std.testing.expectEqualDeep(C{ .unknown = "/exit" }, command("/exit now"));
    try std.testing.expectEqualDeep(C{ .unknown = "/systemd" }, command("/systemd"));
    try std.testing.expectEqualDeep(C{ .say = "/etc/hosts holds what?" }, command(" /etc/hosts holds what?"));
}

test "chat CLI: /clear drops the turns but keeps the system prompt, and /system replaces or removes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: History = .{ .a = a };
    try std.testing.expectEqualStrings("[{\"role\": \"user\", \"content\": \"hi\"}]", try roles(a, &h, "hi"));
    try h.setSystem("Be brief.");
    var said = [_]u8{ 'o', 'n', 'e' };
    try h.keep(&said, "1");
    said[0] = 'x'; // the history holds its own copy
    try h.keep("two", "2");
    try std.testing.expectEqualStrings("[{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": \"one\"}, " ++
        "{\"role\": \"assistant\", \"content\": \"1\"}, {\"role\": \"user\", \"content\": \"two\"}, {\"role\": \"assistant\", \"content\": \"2\"}, " ++
        "{\"role\": \"user\", \"content\": \"three\"}]", try roles(a, &h, "three"));
    h.clear();
    try std.testing.expectEqualStrings("[{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": \"again\"}]", try roles(a, &h, "again"));
    try h.setSystem("Answer in French.");
    try h.keep("q", "r");
    try std.testing.expectEqualStrings("[{\"role\": \"system\", \"content\": \"Answer in French.\"}, {\"role\": \"user\", \"content\": \"q\"}, " ++
        "{\"role\": \"assistant\", \"content\": \"r\"}, {\"role\": \"user\", \"content\": \"s\"}]", try roles(a, &h, "s"));
    try h.setSystem("");
    try std.testing.expectEqualStrings("[{\"role\": \"user\", \"content\": \"q\"}, {\"role\": \"assistant\", \"content\": \"r\"}, " ++
        "{\"role\": \"user\", \"content\": \"s\"}]", try roles(a, &h, "s"));
}

fn roles(a: Allocator, h: *const History, user: []const u8) ![]const u8 {
    return json.stringify(a, .{ .array = try h.messages(a, user) }, .{});
}
