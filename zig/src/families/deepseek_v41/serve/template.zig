//! DeepSeek-V4.1 chat encoding: OpenAI-shaped messages to prompt text, as the release chat template + serving rules.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const BOS = "<｜begin▁of▁sentence｜>";
pub const EOS = "<｜end▁of▁sentence｜>";
pub const THINK_OPEN = "<think>";
pub const THINK_END = "</think>";
pub const SYSTEM = "<｜System｜>";
pub const USER = "<｜User｜>";
pub const ASSISTANT = "<｜Assistant｜>";
pub const REMINDER = "<｜latest_reminder｜>";
pub const DSML = "｜DSML｜";
pub const CALLS_OPEN = "<" ++ DSML ++ " calls>";
pub const CALLS_CLOSE = "</" ++ DSML ++ " calls>";
pub const IMAGE = "<｜deepseek_image｜>";
/// A zero-width space after ``<``: the same text, ordinary BPE tokens, never the image id.
pub const IMAGE_ESCAPED = "<\u{200b}｜deepseek_image｜>";

/// Named reasoning budgets observed at the chat endpoint; none turns thinking off.
pub const efforts = [_]struct { []const u8, ?u8 }{ .{ "none", null }, .{ "minimal", 50 }, .{ "low", 50 }, .{ "medium", 75 }, .{ "high", 75 }, .{ "xhigh", 75 }, .{ "max", 100 } };
pub const default_budget: u8 = 75;

pub const Error = error{ Refused, OutOfMemory };

/// What an image part renders as (the server's placeholder notice); null: image parts are refused.
pub const ImageHook = struct {
    ctx: ?*anyopaque = null,
    text: *const fn (ctx: ?*anyopaque, part: Value) []const u8,
};

pub const Options = struct {
    /// The request's tools (OpenAI ``tools``), already filtered by ``tool_choice``; empty: none offered.
    tools: []const Value = &.{},
    /// ``thinking_mode``: thinking (true) or chat.
    thinking: bool = true,
    /// The reasoning effort budget 1-100 (``budgetOf``).
    budget: u8 = default_budget,
    drop_thinking: bool = true,
    /// ``response_format``'s schema for the system block's ``## Response Format``, or null.
    response_format: ?Value = null,
    add_generation_prompt: bool = true,
    image: ?ImageHook = null,
};

/// A refusal's message (HTTP 400), set when ``render`` returns error.Refused.
pub const Problem = struct { message: []const u8 = "" };

/// A render budget: null if refused, .{null} for none, else 1-100; HTTP names are validated separately.
pub fn effortOf(s: []const u8) ??u8 {
    var buf: [16]u8 = undefined;
    const trimmed = std.mem.trim(u8, s, " \t\n\r");
    if (trimmed.len == 0 or trimmed.len > buf.len) return null;
    const t = std.ascii.lowerString(&buf, trimmed);
    for (efforts) |e| if (std.mem.eql(u8, e[0], t)) return e[1];
    for (t) |c| if (!std.ascii.isDigit(c)) return null;
    const n = std.fmt.parseInt(i64, t, 10) catch return null;
    if (n < 1 or n > 100) return null;
    return @as(?u8, @intCast(n));
}

/// A budget from a JSON value: an effort name or an int 1-100; null for thinking off or a value the app refuses.
pub fn budgetOf(v: Value) ?u8 {
    return switch (v) {
        .string, .int => |t| (effortOf(t) orelse return null) orelse null,
        else => null,
    };
}

const Content = struct { bytes: []const u8 = "", images: []const [2]usize = &.{} };
const Block = struct { tool: bool, text: Content, rank: usize };

const Role = enum { system, user, assistant, tool, latest_reminder };

/// A message after ``normalize`` and the tool-result merge.
const Msg = struct {
    role: Role,
    content: Content = .{},
    reasoning: ?[]const u8 = null,
    calls: []const Value = &.{},
    tool_id: []const u8 = "",
    blocks: std.ArrayList(Block) = .empty, // user turns
    order_set: bool = false, // the calling turn's id order was known when this user turn began
    tools: bool = false, // the first system message carries the tools / response format
    response_format: ?Value = null,
};

const R = struct {
    a: Allocator,
    problem: *Problem,
    out: std.ArrayList(u8) = .empty,
    spans: std.ArrayList([2]usize) = .empty,
    track: bool = false,

    fn refuse(r: *R, comptime fmt: []const u8, args: anytype) Error {
        r.problem.message = try std.fmt.allocPrint(r.a, fmt, args);
        return error.Refused;
    }

    fn put(r: *R, s: []const u8) Allocator.Error!void {
        try r.out.appendSlice(r.a, s);
    }

    fn request(r: *R, s: []const u8) Allocator.Error!void {
        const start = r.out.items.len;
        try r.put(s);
        if (r.track and s.len > 0) try r.spans.append(r.a, .{ start, r.out.items.len });
    }

    fn content(r: *R, value: Content) Allocator.Error!void {
        var at: usize = 0;
        for (value.images) |span| {
            try r.request(value.bytes[at..span[0]]);
            try r.put(value.bytes[span[0]..span[1]]);
            at = span[1];
        }
        try r.request(value.bytes[at..]);
    }

    fn dumps(r: *R, v: Value) Allocator.Error!void {
        const bytes = json.stringify(r.a, v, .{ .ascii = false }) catch return error.OutOfMemory;
        defer r.a.free(bytes);
        try r.request(bytes);
    }
};

/// JSON text of ``v`` without ASCII escaping.
pub fn dumps(a: Allocator, v: Value) Allocator.Error![]u8 {
    return json.stringify(a, v, .{ .ascii = false }) catch error.OutOfMemory;
}

/// The text form of the scalars a text part may hold; null for containers.
fn scalarText(a: Allocator, v: Value) Allocator.Error!?[]const u8 {
    return switch (v) {
        .string => |s| s,
        .int => |t| t,
        .bool => |b| if (b) "True" else "False",
        .float => |f| blk: {
            var buf: [40]u8 = undefined;
            if (std.math.isNan(f)) break :blk "nan";
            if (std.math.isInf(f)) break :blk if (f > 0) "inf" else "-inf";
            break :blk try a.dupe(u8, json.floatRepr(&buf, f));
        },
        else => null,
    };
}

fn isImagePart(t: []const u8) bool {
    return std.mem.eql(u8, t, "image_url") or std.mem.eql(u8, t, "image") or std.mem.eql(u8, t, "input_image");
}

/// OpenAI ``content``: a string, null, or a list of text parts joined by blank lines; image parts through the hook.
fn text(r: *R, content: ?Value, where: []const u8, index: usize, image: ?ImageHook) Error!Content {
    const v = content orelse return .{};
    switch (v) {
        .null => return .{},
        .string => |s| return .{ .bytes = s },
        .array => |parts| {
            var out: std.ArrayList(u8) = .empty;
            var images: std.ArrayList([2]usize) = .empty;
            for (parts, 0..) |part, i| {
                if (i > 0) try out.appendSlice(r.a, "\n\n");
                switch (part) {
                    .string => |s| try out.appendSlice(r.a, s),
                    .object => {
                        const kind = part.get("type") orelse .null;
                        const name = if (kind == .string) kind.string else "";
                        if (std.mem.eql(u8, name, "text") or std.mem.eql(u8, name, "input_text") or std.mem.eql(u8, name, "output_text")) {
                            const t = part.get("text") orelse .null;
                            if (t.truthy()) {
                                const s = try scalarText(r.a, t) orelse return r.refuse("messages[{d}].{s}: a text part's text must be a string", .{ index, where });
                                try out.appendSlice(r.a, s);
                            }
                        } else if (isImagePart(name)) {
                            const hook = image orelse return r.refuse("messages[{d}].{s}: images are not served by this DeepSeek-V4.1 build", .{ index, where });
                            const start = out.items.len;
                            const marker = hook.text(hook.ctx, part);
                            try out.appendSlice(r.a, marker);
                            var at: usize = 0;
                            while (std.mem.indexOfPos(u8, marker, at, IMAGE)) |hit| {
                                try images.append(r.a, .{ start + hit, start + hit + IMAGE.len });
                                at = hit + IMAGE.len;
                            }
                        } else return r.refuse("messages[{d}].{s}: unsupported content part of type {s}", .{ index, where, if (name.len > 0) name else "(none)" });
                    },
                    else => return r.refuse("messages[{d}].{s}: unsupported content part", .{ index, where }),
                }
            }
            return .{ .bytes = out.items, .images = images.items };
        },
        else => return .{ .bytes = try dumps(r.a, v) },
    }
}

/// ``namespace::name`` from a namespace (string or {"name"}) and a name that may carry its own.
fn qualifiedName(a: Allocator, fn_obj: Value, namespace: ?Value) Allocator.Error!?[]const u8 {
    const raw = fn_obj.get("name") orelse return null;
    const name = try scalarText(a, raw) orelse try dumps(a, raw);
    var ns: ?[]const u8 = null;
    if (namespace) |n| switch (n) {
        .null => {},
        .object => if (n.get("name")) |x| {
            if (x != .null) ns = try scalarText(a, x) orelse try dumps(a, x);
        },
        else => ns = try scalarText(a, n) orelse try dumps(a, n),
    };
    var bare = name;
    if (std.mem.indexOf(u8, name, "::")) |sep| {
        ns = name[0..sep];
        bare = name[sep + 2 ..];
    }
    if (ns) |s| if (s.len > 0) return try std.fmt.allocPrint(a, "{s}::{s}", .{ s, bare });
    return bare;
}

/// Tool schemas use exactly name, description and parameters; unrelated function keys are not prompt text.
fn toolFunction(r: *R, tool: Value) Error!Value {
    if (tool != .object) return r.refuse("tools must be a list of objects", .{});
    const inner = tool.get("function");
    const src = if (inner != null and inner.? == .object) inner.? else tool;
    const outer_ns = tool.get("namespace");
    const ns = if (outer_ns != null and outer_ns.? != .null) outer_ns else src.get("namespace");
    const name = try qualifiedName(r.a, src, ns) orelse return r.refuse("a tool has no name", .{});
    var description = src.get("description") orelse Value{ .string = "" };
    if (ns) |n| if (n == .object) if (n.get("description")) |d| if (d.truthy()) {
        const prefix = try scalarText(r.a, d) orelse try dumps(r.a, d);
        const own = if (description.truthy()) (try scalarText(r.a, description) orelse try dumps(r.a, description)) else "";
        description = .{ .string = try std.fmt.allocPrint(r.a, "{s}\n{s}", .{ prefix, own }) };
    };
    const out = try json.newObject(r.a);
    try out.put(r.a, "name", .{ .string = name });
    try out.put(r.a, "description", description);
    try out.put(r.a, "parameters", src.get("parameters") orelse .{ .object = try json.newObject(r.a) });
    return .{ .object = out };
}

const tools_head =
    "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" ++ DSML ++ " calls>\" block like the following:\n\n" ++
    "<" ++ DSML ++ " calls>\n<" ++ DSML ++ " invoke name=\"$TOOL_NAME\">\n<" ++ DSML ++ " parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" ++ DSML ++ " parameter>\n...\n</" ++ DSML ++ " invoke>\n" ++
    "<" ++ DSML ++ " invoke name=\"$TOOL_NAME2\">\n...\n</" ++ DSML ++ " invoke>\n</" ++ DSML ++ " calls>\n\n" ++
    "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n" ++
    "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n" ++
    "Otherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n";
const tools_tail = "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

fn renderTools(r: *R, tools: []const Value) Error!void {
    try r.put(tools_head);
    for (tools, 0..) |t, i| {
        if (i > 0) try r.put("\n");
        try r.dumps(try toolFunction(r, t));
    }
    try r.put(tools_tail);
}

/// Tool-call arguments: a dict as given; a JSON string (even double-encoded) read; else ``{"arguments": raw}``.
fn argumentsOf(a: Allocator, raw: ?Value) Allocator.Error!*json.Object {
    var v = raw orelse Value.null;
    for (0..2) |_| {
        if (v != .string) break;
        const s = std.mem.trim(u8, v.string, " \t\n\r\x0b\x0c");
        if (s.len == 0) {
            v = .{ .object = try json.newObject(a) };
            break;
        }
        switch (try json.parseText(a, v.string)) {
            .ok => |x| v = x,
            .err => break,
        }
    }
    if (v == .object) return v.object;
    const o = try json.newObject(a);
    try o.put(a, "arguments", raw orelse .null);
    return o;
}

fn renderCalls(r: *R, calls: []const Value) Error!void {
    try r.put("\n\n" ++ CALLS_OPEN ++ "\n");
    for (calls, 0..) |c, i| {
        if (i > 0) try r.put("\n");
        if (c != .object) return r.refuse("tool_calls entries must be objects", .{});
        const f = c.get("function");
        const fnv = if (f != null and f.? != .null) f.? else c;
        if (fnv != .object) return r.refuse("a tool call's function must be an object", .{});
        const name = try qualifiedName(r.a, fnv, fnv.get("namespace")) orelse return r.refuse("a tool call has no name", .{});
        try r.put("<" ++ DSML ++ " invoke name=\"");
        try r.request(name);
        try r.put("\">\n");
        const args = try argumentsOf(r.a, fnv.get("arguments"));
        for (args.keys(), args.values(), 0..) |k, v, j| {
            if (j > 0) try r.put("\n");
            try r.put("<" ++ DSML ++ " parameter name=\"");
            try r.request(k);
            try r.put(if (v == .string) "\" string=\"true\">" else "\" string=\"false\">");
            if (v == .string) try r.request(v.string) else try r.dumps(v);
            try r.put("</" ++ DSML ++ " parameter>");
        }
        try r.put("\n</" ++ DSML ++ " invoke>");
    }
    try r.put("\n" ++ CALLS_CLOSE);
}

fn callId(c: Value) []const u8 {
    if (c != .object) return "";
    if (c.get("id")) |id| if (id.truthy() and id == .string) return id.string;
    if (c.get("function")) |f| if (f == .object) if (f.get("id")) |id| if (id == .string) return id.string;
    return "";
}

/// Roles checked, contents as text, tool results merged into user turns.
fn messagesOf(r: *R, messages: Value, o: Options) Error![]Msg {
    if (messages != .array or messages.array.len == 0) return r.refuse("messages must be a non-empty list", .{});
    var list: std.ArrayList(Msg) = .empty;
    for (messages.array, 0..) |m, i| {
        if (m != .object or m.get("role") == null) return r.refuse("messages[{d}]: an object with a role", .{i});
        const rv = m.get("role").?;
        const name = if (rv == .string) rv.string else "";
        const role: Role = if (std.mem.eql(u8, name, "developer")) .system else std.meta.stringToEnum(Role, name) orelse
            return r.refuse("messages[{d}]: unknown role {s}", .{ i, if (rv == .string) rv.string else try dumps(r.a, rv) });
        var msg: Msg = .{ .role = role };
        switch (role) {
            .assistant => {
                msg.content = try text(r, m.get("content"), "content", i, o.image);
                var rc = m.get("reasoning_content") orelse Value.null;
                if (rc == .null) rc = m.get("reasoning") orelse Value.null;
                if (rc != .null) msg.reasoning = (try text(r, rc, "reasoning_content", i, null)).bytes;
                if (m.get("tool_calls")) |tc| if (tc.truthy()) {
                    if (tc != .array) return r.refuse("messages[{d}].tool_calls must be a list", .{i});
                    msg.calls = tc.array;
                };
            },
            .tool => {
                msg.content = try text(r, m.get("content"), "content", i, o.image);
                const id = m.get("tool_call_id") orelse Value.null;
                msg.tool_id = if (id == .string) id.string else if (id.truthy()) try dumps(r.a, id) else "";
            },
            else => msg.content = try text(r, m.get("content"), "content", i, o.image),
        }
        try list.append(r.a, msg);
    }
    const schema = if (o.response_format) |rf| (if (rf.truthy()) rf else null) else null;
    if (o.tools.len > 0 or schema != null) {
        if (list.items[0].role != .system) try list.insert(r.a, 0, .{ .role = .system });
        list.items[0].tools = o.tools.len > 0;
        list.items[0].response_format = schema;
    }
    // tool results into user turns, ranked by the calling turn's order
    var merged: std.ArrayList(Msg) = .empty;
    var order: std.StringHashMapUnmanaged(usize) = .empty;
    var order_known = false;
    for (list.items) |m| {
        if (m.role == .assistant and m.calls.len > 0) {
            order = .empty;
            for (m.calls, 0..) |c, k| {
                if (c != .object) return r.refuse("tool_calls entries must be objects", .{});
                const id = callId(c);
                if (id.len > 0) try order.put(r.a, id, k);
            }
            order_known = order.count() > 0;
        }
        if (m.role == .tool or m.role == .user) {
            const block: Block = .{ .tool = m.role == .tool, .text = m.content, .rank = if (m.role == .tool) order.get(m.tool_id) orelse 0 else 0 };
            if (merged.items.len > 0 and merged.items[merged.items.len - 1].role == .user) {
                try merged.items[merged.items.len - 1].blocks.append(r.a, block);
            } else {
                var u: Msg = .{ .role = .user, .order_set = order_known };
                try u.blocks.append(r.a, block);
                try merged.append(r.a, u);
            }
        } else try merged.append(r.a, m);
    }
    for (merged.items) |*m| {
        if (m.role != .user or !m.order_set) continue;
        var tools: std.ArrayList(Block) = .empty;
        for (m.blocks.items) |b| if (b.tool) try tools.append(r.a, b);
        if (tools.items.len < 2) continue;
        std.sort.insertion(Block, tools.items, {}, struct {
            fn lt(_: void, x: Block, y: Block) bool {
                return x.rank < y.rank;
            }
        }.lt); // stable sort
        var k: usize = 0;
        for (m.blocks.items) |*b| if (b.tool) {
            b.* = tools.items[k];
            k += 1;
        };
    }
    return merged.items;
}

/// The prompt text for ``messages`` (a JSON array of OpenAI messages); error.Refused sets ``problem``.
pub fn render(a: Allocator, messages: Value, o: Options, problem: *Problem) Error![]u8 {
    return (try renderInner(a, messages, o, problem, false)).text;
}

/// The same prompt with request-derived byte ranges, including tool names, arguments and serialized schemas.
pub const Rendered = struct { text: []u8, spans: []const [2]usize = &.{} };

pub fn renderProvenance(a: Allocator, messages: Value, o: Options, problem: *Problem) Error!Rendered {
    return renderInner(a, messages, o, problem, true);
}

fn renderInner(a: Allocator, messages: Value, o: Options, problem: *Problem, track: bool) Error!Rendered {
    var r: R = .{ .a = a, .problem = problem, .track = track };
    errdefer r.out.deinit(a);
    errdefer r.spans.deinit(a);
    if (o.budget < 1 or o.budget > 100) return r.refuse("reasoning_effort must be 1-100 or low / medium / high / max", .{});
    const msgs = try messagesOf(&r, messages, o);
    const drop = o.drop_thinking and !(msgs.len > 0 and msgs[0].tools);
    const thinking = o.thinking;
    try r.put(BOS);
    for (msgs, 0..) |m, idx| {
        if (idx == 0 and (thinking or m.role == .system)) try r.put(SYSTEM);
        if (idx == 0 and thinking) {
            var buf: [8]u8 = undefined;
            try r.put("Reasoning Effort: ");
            try r.put(std.fmt.bufPrint(&buf, "{d}", .{o.budget}) catch unreachable);
            try r.put(" (range 1-100, the higher the value, the more thorough the reasoning)\n\n");
        }
        switch (m.role) {
            .system => {
                if (idx > 0) try r.put(SYSTEM);
                try r.content(m.content);
                if (m.tools) {
                    try r.put("\n\n");
                    try renderTools(&r, o.tools);
                }
                if (m.response_format) |rf| {
                    try r.put("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                    try r.dumps(rf);
                }
            },
            .user => {
                try r.put(USER);
                for (m.blocks.items, 0..) |b, k| {
                    if (k > 0) try r.put("\n\n");
                    if (b.tool) try r.put("<tool_result>");
                    try r.content(b.text);
                    if (b.tool) try r.put("</tool_result>");
                }
            },
            .latest_reminder => {
                try r.put(REMINDER);
                try r.content(m.content);
            },
            .assistant => {
                try r.put(ASSISTANT);
                if (thinking and !drop and m.reasoning != null) {
                    try r.put(THINK_OPEN);
                    try r.request(m.reasoning.?);
                }
                try r.put(THINK_END);
                try r.content(m.content);
                if (m.calls.len > 0) try renderCalls(&r, m.calls);
                try r.put(EOS);
            },
            .tool => unreachable, // merged into user turns
        }
    }
    if (o.add_generation_prompt) try r.put(generationPrefix(thinking));
    return .{ .text = r.out.items, .spans = r.spans.items };
}

/// A typed ``<｜deepseek_image｜>`` as plain text (never the image id).
pub fn escapeImage(a: Allocator, s: []u8) Allocator.Error![]u8 {
    if (std.mem.indexOf(u8, s, IMAGE) == null) return s;
    return std.mem.replaceOwned(u8, a, s, IMAGE, IMAGE_ESCAPED);
}

/// What follows a prompt's last user turn.
pub fn generationPrefix(thinking: bool) []const u8 {
    return if (thinking) ASSISTANT ++ THINK_OPEN else ASSISTANT ++ THINK_END;
}

test "a thinking conversation with a tool call and an out-of-order result" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const x = arena.allocator();
    const msgs = (try json.parseText(x,
        \\[{"role":"user","content":"hi"},
        \\ {"role":"assistant","content":null,"reasoning":"plan","tool_calls":[
        \\   {"id":"a","type":"function","function":{"name":"f","arguments":"{\"q\": 1}"}},
        \\   {"id":"b","type":"function","function":{"name":"g","arguments":{"s":"x"}}}]},
        \\ {"role":"tool","tool_call_id":"b","content":"B"},{"role":"tool","tool_call_id":"a","content":"A"}]
    )).ok;
    const tools = (try json.parseText(x, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{}}}]")).ok;
    var problem: Problem = .{};
    const got = try render(x, msgs, .{ .tools = tools.array, .budget = 62 }, &problem);
    try std.testing.expect(std.mem.startsWith(u8, got, BOS ++ SYSTEM ++ "Reasoning Effort: 62 (range"));
    try std.testing.expect(std.mem.endsWith(u8, got, "plan</think>\n\n" ++ CALLS_OPEN ++ "\n<" ++ DSML ++ " invoke name=\"f\">\n<" ++ DSML ++ " parameter name=\"q\" string=\"false\">1</" ++ DSML ++ " parameter>\n</" ++ DSML ++ " invoke>\n<" ++
        DSML ++ " invoke name=\"g\">\n<" ++ DSML ++ " parameter name=\"s\" string=\"true\">x</" ++ DSML ++ " parameter>\n</" ++ DSML ++ " invoke>\n" ++ CALLS_CLOSE ++ EOS ++ USER ++ "<tool_result>A</tool_result>\n\n<tool_result>B</tool_result>" ++ ASSISTANT ++ THINK_OPEN));
    try std.testing.expectEqual(@as(?u8, 75), budgetOf(.{ .string = "medium" }));
    try std.testing.expectEqual(@as(?u8, null), budgetOf(.{ .string = "none" }));
    try std.testing.expectEqual(@as(?u8, null), budgetOf(.{ .int = "101" }));
}

test "native tool listings and history use the same qualified name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parseText(a,
        \\[{"type":"function","namespace":{"name":"space","description":"namespace"},"function":{"name":"f","parameters":{}}}]
    )).ok;
    const messages = (try json.parseText(a,
        \\[{"role":"user","content":"H"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","namespace":"space","arguments":"{}"}}]}]
    )).ok;
    var problem: Problem = .{};
    for ([_][]const u8{ "f", "space::f" }) |name| {
        try tools.array[0].get("function").?.object.put(a, "name", .{ .string = name });
        try messages.array[1].get("tool_calls").?.array[0].get("function").?.object.put(a, "name", .{ .string = name });
        const out = try render(a, messages, .{ .tools = tools.array }, &problem);
        try std.testing.expect(std.mem.indexOf(u8, out, "\"name\": \"space::f\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "invoke name=\"space::f\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "\"description\": \"namespace\\n\"") != null);
    }
}

test "tools sharing a bare name remain distinct across namespaces" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parseText(a,
        \\[{"type":"function","namespace":{"name":"files","description":"File operations."},"function":{"name":"run","description":"Read files.","parameters":{},"strict":true,"extra":"omit"}},{"type":"function","function":{"name":"jobs::run","namespace":{"name":"jobs","description":"Job operations."},"description":"Start jobs.","parameters":{},"strict":false,"extra":"omit"}}]
    )).ok;
    const messages = (try json.parseText(a,
        \\[{"role":"user","content":"Run both."},{"role":"assistant","tool_calls":[{"id":"f","type":"function","function":{"name":"run","namespace":"files","arguments":"{}"}},{"id":"j","type":"function","function":{"name":"jobs::run","namespace":"ignored","arguments":"{}"}}]}]
    )).ok;
    var problem: Problem = .{};
    const out = try render(a, messages, .{ .tools = tools.array }, &problem);
    for ([_][]const u8{
        "\"name\": \"files::run\", \"description\": \"File operations.\\nRead files.\", \"parameters\": {}",
        "\"name\": \"jobs::run\", \"description\": \"Job operations.\\nStart jobs.\", \"parameters\": {}",
        "invoke name=\"files::run\"",
        "invoke name=\"jobs::run\"",
    }) |want| try std.testing.expect(std.mem.indexOf(u8, out, want) != null);
    for ([_][]const u8{ "\"strict\"", "\"extra\"", "\"namespace\"", "\"name\": \"run\"", "invoke name=\"run\"", "ignored::" }) |unwanted|
        try std.testing.expect(std.mem.indexOf(u8, out, unwanted) == null);
}
