//! DeepSeek-V4.1 replies: reasoning, content and DSML tool calls, whole (``parse``) or streamed (``Stream``), lenient.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const tool_scan = @import("tool_scan.zig");
const tool_types = @import("tool_types.zig");
const tool_stream = @import("tool_stream.zig");

pub const THINK_END = tool_scan.THINK_END;
pub const pyStrip = tool_scan.pyStrip;
pub const Match = tool_scan.Match;
const blockTag = tool_scan.blockTag;
const invokeOpen = tool_scan.invokeOpen;
const invokeEnd = tool_scan.invokeEnd;
const paramIn = tool_scan.paramIn;
pub const schemaTypes = tool_types.schemaTypes;
pub const closedJson = tool_types.closedJson;
pub const typedValue = tool_types.typedValue;
pub const paramValue = tool_types.paramValue;
const fragment = tool_types.fragment;
pub const Delta = tool_stream.Delta;
pub const Stream = tool_stream.Stream;

// -- tools and calls -------------------------------------------------------------------------------------------------

pub const Arg = struct { key: []const u8, value: Value };

pub const Call = struct {
    id: []const u8,
    name: []const u8,
    args: std.ArrayList(Arg) = .empty,
    closed: bool = false,

    /// The arguments as compact JSON (``{}`` without any).
    pub fn arguments(c: *const Call, a: Allocator) Allocator.Error![]u8 {
        if (c.args.items.len == 0) return a.dupe(u8, "{}");
        var out: std.ArrayList(u8) = .empty;
        for (c.args.items, 0..) |arg, i| try fragment(a, &out, i == 0, arg.key, arg.value);
        try out.append(a, '}');
        return out.items;
    }
};

/// A new call's id (``call_`` + 24 hex in the server); ``index`` counts the reply's calls.
pub const IdSource = struct {
    ctx: ?*anyopaque = null,
    make: *const fn (ctx: ?*anyopaque, a: Allocator, index: usize) Allocator.Error![]const u8 = counted,

    fn counted(_: ?*anyopaque, a: Allocator, index: usize) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(a, "call_{d}", .{index});
    }
};

/// The offered tools by name: their parameter properties, for typing values and matching names.
pub const Tools = struct {
    names: []const []const u8 = &.{},
    props: []const ?Value = &.{},
    /// Calls to tools not offered go out as calls; false keeps them in the content.
    unknown_calls: bool = true,

    pub fn init(a: Allocator, tools: []const Value) Allocator.Error!Tools {
        var names: std.ArrayList([]const u8) = .empty;
        var props: std.ArrayList(?Value) = .empty;
        for (tools) |t| {
            if (t != .object) continue;
            const inner = t.get("function");
            const f = if (inner != null and inner.? == .object) inner.? else t;
            const n = f.get("name") orelse continue;
            if (!n.truthy()) continue;
            const name = switch (n) {
                .string => |s| s,
                .int => |s| s,
                else => try json.stringify(a, n, .{ .ascii = false }),
            };
            const params = f.get("parameters") orelse Value.null;
            const properties: ?Value = if (params == .object) (if (params.get("properties")) |p| (if (p.truthy()) p else null) else null) else null;
            // later tools of the same name replace earlier ones, as a dict does
            for (names.items, 0..) |seen, i| {
                if (!std.mem.eql(u8, seen, name)) continue;
                props.items[i] = properties;
                break;
            } else {
                try names.append(a, name);
                try props.append(a, properties);
            }
        }
        return .{ .names = names.items, .props = props.items };
    }

    pub fn offered(t: Tools) bool {
        return t.names.len > 0;
    }

    /// The offered name a call means: exact, else case-insensitive, else its last ``::`` part.
    pub fn known(t: Tools, name: []const u8) ?usize {
        for (t.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        var found: ?usize = null;
        for (t.names, 0..) |n, i| if (std.ascii.eqlIgnoreCase(n, name)) {
            found = i; // the last of equal lower-case names, as the dict built from them keeps
        };
        if (found) |f| return f;
        const tail = if (std.mem.lastIndexOf(u8, name, "::")) |p| name[p + 2 ..] else name;
        for (t.names, 0..) |n, i| if (std.ascii.eqlIgnoreCase(n, tail)) {
            found = i;
        };
        return found;
    }

    /// The name a call goes out under: the offered name it means, else its own (null for an empty one).
    pub fn resolve(t: Tools, name: []const u8) ?[]const u8 {
        if (t.known(name)) |i| return t.names[i];
        return if (t.unknown_calls and name.len > 0) name else null;
    }

    pub fn prop(t: Tools, call_name: []const u8, key: []const u8) ?Value {
        const i = t.known(call_name) orelse return null;
        const p = t.props[i] orelse return null;
        return if (p == .object) p.get(key) else null;
    }
};

/// One invoke's parameters read up to ``stop``, the first of each name kept.
fn readParams(a: Allocator, s: []const u8, from: usize, stop: usize, tools: Tools, call: *Call, seen: *std.StringHashMapUnmanaged(void), pos: *usize) Allocator.Error!void {
    while (paramIn(s, pos.*, stop)) |p| {
        pos.* = p.end;
        if (seen.contains(p.name)) continue;
        try seen.put(a, p.name, {});
        try call.args.append(a, .{ .key = p.name, .value = try paramValue(a, p.raw, p.string, tools.prop(call.name, p.name)) });
    }
    _ = from;
}

/// The invokes of a calls block's body ``s[from..to]``.
fn readCalls(a: Allocator, s: []const u8, from: usize, to: usize, tools: Tools, ids: IdSource, first_index: usize, out: *std.ArrayList(Call)) Allocator.Error!void {
    var pos = from;
    while (invokeOpen(s, pos, to)) |m| {
        const end = invokeEnd(s, m.m.end, to);
        const nxt = invokeOpen(s, m.m.end, to);
        const stop = if (end != null and (nxt == null or end.?.start < nxt.?.m.start)) end.?.start else if (nxt) |n| n.m.start else to;
        var call: Call = .{ .id = try ids.make(ids.ctx, a, first_index + out.items.len), .name = m.name };
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        var p = m.m.end;
        try readParams(a, s, m.m.end, stop, tools, &call, &seen, &p);
        call.closed = end != null and stop == end.?.start;
        try out.append(a, call);
        pos = if (call.closed) end.?.end else stop;
        if (!call.closed and nxt == null) break;
    }
}

pub const Reply = struct {
    reasoning: []const u8,
    content: []const u8,
    calls: []Call,
};

const Split = struct { reasoning: []const u8, rest_at: usize };

/// (reasoning, where the rest starts).
pub fn split(text: []const u8, thinking: bool, end: ?usize, blk: ?Match) Split {
    if (!thinking) return .{ .reasoning = "", .rest_at = 0 };
    if (end) |e| if (blk == null or e < blk.?.start) return .{ .reasoning = text[0..e], .rest_at = e + THINK_END.len };
    if (blk) |b| {
        const head = text[0..b.start];
        const r = if (std.mem.endsWith(u8, head, "\n\n")) head[0 .. head.len - 2] else std.mem.trimEnd(u8, head, "\n");
        return .{ .reasoning = r, .rest_at = r.len };
    }
    return .{ .reasoning = text, .rest_at = text.len };
}

/// A finished reply's parts (end-of-sentence tokens already removed).
pub fn parse(a: Allocator, text: []const u8, thinking: bool, tools: Tools, ids: IdSource) Allocator.Error!Reply {
    const sp = split(text, thinking, std.mem.indexOf(u8, text, THINK_END), blockTag(text, 0, text.len, false));
    const rest = text[sp.rest_at..];
    const blk = if (tools.offered()) blockTag(rest, 0, rest.len, false) else null;
    const b = blk orelse return .{ .reasoning = sp.reasoning, .content = rest, .calls = &.{} };
    var content = rest[0..b.start];
    if (std.mem.endsWith(u8, content, "\n\n")) content = content[0 .. content.len - 2];
    const close = blockTag(rest, b.end, rest.len, true);
    var calls: std.ArrayList(Call) = .empty;
    try readCalls(a, rest, b.end, if (close) |c| c.start else rest.len, tools, ids, 0, &calls);
    var usable: std.ArrayList(Call) = .empty;
    for (calls.items) |c| if (tools.resolve(c.name)) |n| {
        var k = c;
        k.name = n;
        try usable.append(a, k);
    };
    if (usable.items.len == 0) return .{ .reasoning = sp.reasoning, .content = rest, .calls = &.{} };
    // ids in order of the usable calls, as made while reading
    return .{ .reasoning = sp.reasoning, .content = content, .calls = usable.items };
}

/// Why a tools reply's DSML markup gave no usable call, with <= ``note_window`` bytes around its first tag (``note``).
pub const Note = struct { at: usize, reason: []const u8, window: []const u8 };
pub const note_window = 300;
const DSML = "｜DSML｜";

/// For a reply whose DSML markup ``parse`` turns into no usable call: the note the server logs; else null.
pub fn note(a: Allocator, text: []const u8, thinking: bool, tools: Tools) Allocator.Error!?Note {
    const hit = std.mem.indexOf(u8, text, DSML) orelse return null;
    // the first tag's ``<`` / ``</`` when it has one
    const at = if (hit >= 2 and text[hit - 2] == '<' and text[hit - 1] == '/') hit - 2 else if (hit >= 1 and text[hit - 1] == '<') hit - 1 else hit;
    const reason = try noteReason(a, text, thinking, tools) orelse return null;
    return .{ .at = at, .reason = reason, .window = window(text, at, note_window) };
}

fn noteReason(a: Allocator, text: []const u8, thinking: bool, tools: Tools) Allocator.Error!?[]const u8 {
    if (!tools.offered()) return "the request offered no tools";
    const sp = split(text, thinking, std.mem.indexOf(u8, text, THINK_END), blockTag(text, 0, text.len, false));
    const rest = text[sp.rest_at..];
    const b = blockTag(rest, 0, rest.len, false) orelse {
        if (std.mem.indexOf(u8, rest, DSML) == null) return "DSML markup only inside the reasoning (no calls block after it)";
        return "DSML tags but no calls / tool_calls / function_calls block opening tag";
    };
    const close = blockTag(rest, b.end, rest.len, true);
    var calls: std.ArrayList(Call) = .empty;
    const scratch: IdSource = .{};
    try readCalls(a, rest, b.end, if (close) |c| c.start else rest.len, tools, scratch, 0, &calls);
    if (calls.items.len == 0) {
        if (invokeOpen(rest, b.end, rest.len) == null) return "a calls block without a complete invoke header";
        return "an invoke that read as no call";
    }
    for (calls.items) |c| if (tools.known(c.name) != null) return null; // a usable call: parse emits it
    return try std.fmt.allocPrint(a, "an invoke of a tool the request did not offer: \"{s}\"{s}", .{ window(calls.items[0].name, 0, 64), if (close == null) " (calls block not closed)" else "" });
}

/// At most ``max`` bytes of ``s`` around ``at`` (a third before it), cut so no UTF-8 sequence is split.
pub fn window(s: []const u8, at: usize, max: usize) []const u8 {
    const p = @min(at, s.len);
    var lo = p -| max / 3;
    var hi = @min(s.len, lo + max);
    while (lo < hi and s[lo] & 0xC0 == 0x80) lo += 1; // not inside a sequence
    if (hi > lo) {
        // the last sequence whole: not cut at hi (nor by the end of s itself)
        var k = hi - 1;
        while (k > lo and s[k] & 0xC0 == 0x80) k -= 1;
        const need = std.unicode.utf8ByteSequenceLength(s[k]) catch 1;
        if (k + need > hi) hi = k;
    }
    return s[lo..hi];
}

test "a thinking reply with two calls, streamed in pieces and parsed whole, gives the same arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools_json = (try json.parseText(a,
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}},
        \\ {"type":"function","function":{"name":"search","parameters":{"properties":{"q":{"type":"string"},"tags":{"type":"array"}}}}}]
    )).ok;
    const tools = try Tools.init(a, tools_json.array);
    const text = "plan it</think>Sure.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n" ++
        "<｜DSML｜ parameter name=\"days\" string=\"true\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"Search\">\n<｜DSML｜ parameter name=\"tags\" string=\"false\">[\"a\", [1</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const whole = try parse(a, text, true, tools, .{});
    try std.testing.expectEqualStrings("plan it", whole.reasoning);
    try std.testing.expectEqualStrings("Sure.", whole.content);
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    try std.testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":3}", try whole.calls[0].arguments(a));
    try std.testing.expectEqualStrings("search", whole.calls[1].name);
    try std.testing.expectEqualStrings("{\"tags\":[\"a\",[1]]}", try whole.calls[1].arguments(a));
    var s = Stream.init(a, true, tools, .{});
    var out: std.ArrayList(Delta) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        i = @min(text.len, i + 3);
        while (i < text.len and text[i] & 0xC0 == 0x80) i += 1;
        try s.feed(text[0..i], false, &out);
    }
    try s.finish(text, &out);
    var reasoning: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    var args: [2]std.ArrayList(u8) = .{ .empty, .empty };
    for (out.items) |d| switch (d) {
        .reasoning => |r| try reasoning.appendSlice(a, r),
        .content => |c| try content.appendSlice(a, c),
        .call => {},
        .arguments => |g| try args[g.index].appendSlice(a, g.text),
    };
    try std.testing.expectEqualStrings("plan it", reasoning.items);
    try std.testing.expectEqualStrings("Sure.", content.items);
    try std.testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":3}", args[0].items);
    try std.testing.expectEqualStrings("{\"tags\":[\"a\",[1]]}", args[1].items);
}

test "streamed keys and call names outlive the text they were read from (each feed a new buffer, the old one freed)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools_json = (try json.parseText(a,
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}}}}}]
    )).ok;
    const tools = try Tools.init(a, tools_json.array);
    // a long value: its parameter's closing tag arrives many feeds after the head that named it
    const text = "ok</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">grep -n 'pub fn stdout' /home/user/.local/opt/zig/lib/std/Io/File.zig | head -40</｜DSML｜ parameter>\n" ++
        "<｜DSML｜ parameter name=\"timeout\" string=\"false\">30</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    var s = Stream.init(a, true, tools, .{});
    var out: std.ArrayList(Delta) = .empty;
    var prev: ?[]u8 = null;
    var i: usize = 0;
    while (i < text.len) {
        i = @min(text.len, i + 3);
        while (i < text.len and text[i] & 0xC0 == 0x80) i += 1;
        // the server's text grows by reallocation: the bytes a feed saw are freed (and poisoned) before the next
        const buf = try std.testing.allocator.dupe(u8, text[0..i]);
        if (prev) |p| {
            @memset(p, 0xAA);
            std.testing.allocator.free(p);
        }
        prev = buf;
        try s.feed(buf, false, &out);
    }
    try s.finish(prev.?, &out);
    std.testing.allocator.free(prev.?);
    var args: std.ArrayList(u8) = .empty;
    var name: []const u8 = "";
    for (out.items) |d| switch (d) {
        .call => |c| name = c.name,
        .arguments => |g| try args.appendSlice(a, g.text),
        else => {},
    };
    try std.testing.expectEqualStrings("bash", name);
    try std.testing.expectEqualStrings("{\"command\":\"grep -n 'pub fn stdout' /home/user/.local/opt/zig/lib/std/Io/File.zig | head -40\",\"timeout\":30}", args.items);
    // the kept call (the reply's end serializes it again, every buffer freed by then): its values are copies too
    try std.testing.expectEqual(@as(usize, 1), s.calls.items.len);
    try std.testing.expectEqualStrings("{\"command\":\"grep -n 'pub fn stdout' /home/user/.local/opt/zig/lib/std/Io/File.zig | head -40\",\"timeout\":30}", try s.calls.items[0].arguments(a));
}

test "note: DSML markup that yields no call is reported once with a reason and a bounded window; clean text and good calls are not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools_json = (try json.parseText(a,
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]
    )).ok;
    const tools = try Tools.init(a, tools_json.array);
    const good = "done</think>ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    try std.testing.expect((try note(a, good, true, tools)) == null);
    try std.testing.expect((try note(a, "done</think>All set, nothing more to run.", true, tools)) == null);
    // an unknown tool
    const unknown = "x</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"zsh\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const n1 = (try note(a, unknown, true, tools)).?;
    try std.testing.expect(std.mem.indexOf(u8, n1.reason, "did not offer: \"zsh\"") != null);
    try std.testing.expectEqual(@as(usize, 9), n1.at); // the "<" of the first tag
    // a tag with no calls block, at the very start of the text
    const stray = "<｜DSML｜ invoke name=\"bash\">";
    const n2 = (try note(a, stray, false, tools)).?;
    try std.testing.expectEqualStrings("DSML tags but no calls / tool_calls / function_calls block opening tag", n2.reason);
    try std.testing.expectEqual(@as(usize, 0), n2.at);
    try std.testing.expectEqualStrings(stray, n2.window);
    // a block whose invoke header never completes at the end of a long multi-byte text: window <= 300 B, valid UTF-8
    var long: std.ArrayList(u8) = .empty;
    for (0..400) |_| try long.appendSlice(a, "é");
    try long.appendSlice(a, "</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"ba");
    const n3 = (try note(a, long.items, true, tools)).?;
    try std.testing.expectEqualStrings("a calls block without a complete invoke header", n3.reason);
    try std.testing.expect(n3.window.len <= note_window);
    try std.testing.expect(std.unicode.utf8ValidateSlice(n3.window));
    try std.testing.expect(std.mem.endsWith(u8, long.items, n3.window));
    // the window itself at the buffer's edges and inside sequences
    const e = "aé€😀b";
    for (0..e.len + 1) |at| for ([_]usize{ 0, 1, 2, 3, 4, 5, 300 }) |max| {
        const w = window(e, at, max);
        try std.testing.expect(w.len <= max);
        try std.testing.expect(std.unicode.utf8ValidateSlice(w));
    };
    try std.testing.expectEqualStrings(e, window(e, 3, 300));
    try std.testing.expectEqualStrings("aé", window(e, 0, 4)); // € would be cut at 4
    // a reply that ends inside a sequence (the stream cut a character): the partial bytes are left out
    try std.testing.expectEqualStrings("ab", window("ab€"[0..4], 0, 300));
    try std.testing.expectEqualStrings("", window("", 0, note_window));
}

test "a call to a tool the request did not offer goes out as a call, whole and streamed; a nameless one stays text" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try Tools.init(a, (try json.parseText(a,
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]
    )).ok.array);
    const text = "look</think>Here it is.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"take_screenshot\">\n<｜DSML｜ parameter name=\"full\" string=\"false\">true</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const whole = try parse(a, text, true, tools, .{});
    try std.testing.expectEqualStrings("Here it is.", whole.content);
    try std.testing.expectEqual(@as(usize, 1), whole.calls.len);
    try std.testing.expectEqualStrings("take_screenshot", whole.calls[0].name);
    try std.testing.expectEqualStrings("{\"full\":true}", try whole.calls[0].arguments(a));
    var s = Stream.init(a, true, tools, .{});
    var out: std.ArrayList(Delta) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        i = @min(text.len, i + 5);
        while (i < text.len and text[i] & 0xC0 == 0x80) i += 1;
        try s.feed(text[0..i], false, &out);
    }
    try s.finish(text, &out);
    var name: []const u8 = "";
    var args: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    for (out.items) |d| switch (d) {
        .call => |c| name = c.name,
        .arguments => |g| try args.appendSlice(a, g.text),
        .content => |c| try content.appendSlice(a, c),
        .reasoning => {},
    };
    try std.testing.expectEqualStrings("take_screenshot", name);
    try std.testing.expectEqualStrings("{\"full\":true}", args.items);
    try std.testing.expectEqualStrings("Here it is.", content.items);
    // a known tool still goes out under its offered spelling
    const cased = try parse(a, "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"BASH\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>", false, tools, .{});
    try std.testing.expectEqualStrings("bash", cased.calls[0].name);
    // an invoke without a name is not a call: the reply stays text
    const nameless = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const n = try parse(a, nameless, false, tools, .{});
    try std.testing.expectEqual(@as(usize, 0), n.calls.len);
    try std.testing.expectEqualStrings(nameless, n.content);
}
