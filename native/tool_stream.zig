const std = @import("std");
const calls = @import("tool_calls.zig");
const V = std.json.Value;
const ws = " \r\n\t";

pub const Delta = struct { index: usize, name: ?[]const u8 = null, arguments: []const u8 = "" };
pub const Emit = *const fn (?*anyopaque, Delta) anyerror!void;

// Mirrors upstream ToolCallStreamer. Other grammars remain the final parser's job.
pub const Streamer = struct {
    state: enum { outside, call, function, value, closing, abort } = .outside,
    pos: usize = 0,
    count: usize = 0,
    name: []const u8 = "",
    key_start: usize = 0,
    key_end: usize = 0,
    args_open: bool = false,
    typed: bool = false,
    lead: bool = false,
    value_start: usize = 0,

    fn args(s: *Streamer, context: ?*anyopaque, emit: Emit, text: []const u8) !void {
        try emit(context, .{ .index = s.count - 1, .arguments = text });
    }

    pub fn feed(s: *Streamer, a: std.mem.Allocator, text: []const u8, tools: V, context: ?*anyopaque, emit: Emit) !void {
        while (s.state != .abort) {
            const rest = text[s.pos..];
            const body = std.mem.trimStart(u8, rest, ws);
            const skipped = rest.len - body.len;
            switch (s.state) {
                .outside => {
                    const at = std.mem.indexOf(u8, rest, "<tool_call>") orelse return;
                    s.pos += at + "<tool_call>".len;
                    s.state = .call;
                },
                .call => {
                    const tag = header(body, "<function=") orelse {
                        if (rest.len > 256 or (body.len > 0 and body[0] != '<')) s.state = .abort;
                        return;
                    };
                    s.name = calls.canonicalName(tools, tag.name) orelse {
                        s.state = .abort;
                        return;
                    };
                    try emit(context, .{ .index = s.count, .name = s.name });
                    s.count += 1;
                    s.args_open = false;
                    s.pos += skipped + tag.end;
                    s.state = .function;
                },
                .function => {
                    if (header(body, "<parameter=")) |tag| {
                        s.key_start = s.pos + skipped + "<parameter=".len;
                        s.key_end = s.pos + skipped + tag.end - 1;
                        s.typed = calls.typedParameter(tools, s.name, tag.name);
                        const key = try std.json.Stringify.valueAlloc(a, tag.name, .{});
                        try s.args(context, emit, try std.fmt.allocPrint(a, "{s}{s}{s}", .{ if (s.args_open) "," else "{", key, if (s.typed) ":" else ":\"" }));
                        s.args_open = true;
                        s.pos += skipped + tag.end;
                        s.value_start = s.pos;
                        s.lead = true;
                        s.state = .value;
                    } else if (std.mem.startsWith(u8, body, "</function>")) {
                        try s.args(context, emit, if (s.args_open) "}" else "{}");
                        s.pos += skipped + "</function>".len;
                        s.state = .closing;
                    } else return;
                },
                .value => {
                    if (s.lead) {
                        if (rest.len == 0) return;
                        if (rest[0] == '\n') s.pos += 1;
                        s.value_start = s.pos;
                        s.lead = false;
                    }
                    const close = std.mem.indexOfPos(u8, text, s.pos, "</parameter>");
                    var end = close orelse retreat(text, s.pos, "</parameter>".len);
                    if (close != null) {
                        if (end > s.pos and text[end - 1] == '\n') end -= 1;
                    } else {
                        while (end > s.pos and std.ascii.isWhitespace(text[end - 1])) end -= 1;
                    }
                    if (!s.typed and end > s.pos) {
                        const escaped = try std.json.Stringify.valueAlloc(a, text[s.pos..end], .{});
                        try s.args(context, emit, escaped[1 .. escaped.len - 1]);
                    }
                    s.pos = end;
                    if (close) |at| {
                        if (s.typed) {
                            const key = std.mem.trim(u8, text[s.key_start..s.key_end], ws);
                            const value = try calls.parameter(a, tools, s.name, key, text[s.value_start..end]);
                            try s.args(context, emit, try std.json.Stringify.valueAlloc(a, value, .{}));
                        } else try s.args(context, emit, "\"");
                        s.pos = at + "</parameter>".len;
                        s.state = .function;
                    } else return;
                },
                .closing => {
                    if (!std.mem.startsWith(u8, body, "</tool_call>")) return;
                    s.pos += skipped + "</tool_call>".len;
                    s.state = .outside;
                },
                .abort => unreachable,
            }
        }
    }
};

fn header(text: []const u8, prefix: []const u8) ?struct { name: []const u8, end: usize } {
    if (!std.mem.startsWith(u8, text, prefix)) return null;
    const end = std.mem.indexOfScalarPos(u8, text, prefix.len, '>') orelse return null;
    if (end == prefix.len or std.mem.indexOfScalar(u8, text[prefix.len..end], '\n') != null) return null;
    return .{ .name = std.mem.trim(u8, text[prefix.len..end], ws), .end = end + 1 };
}

fn retreat(text: []const u8, start: usize, count: usize) usize {
    var end = text.len;
    for (0..count) |_| {
        if (end == start) break;
        end -= 1;
        while (end > start and text[end] & 0xc0 == 0x80) end -= 1;
    }
    return end;
}

const Capture = struct {
    a: std.mem.Allocator,
    deltas: std.ArrayList(Delta) = .empty,
    fn emit(context: ?*anyopaque, delta: Delta) !void {
        const c: *Capture = @ptrCast(@alignCast(context.?));
        try c.deltas.append(c.a, delta);
    }
};

fn compact(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var quoted = false;
    var escape = false;
    for (text) |c| {
        if (quoted) {
            try out.append(a, c);
            if (escape) escape = false else if (c == '\\') escape = true else if (c == '"') quoted = false;
        } else {
            if (c == '"') quoted = true;
            if (!std.ascii.isWhitespace(c)) try out.append(a, c);
        }
    }
    return out.items;
}

pub fn check(io: std.Io, path: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(128 * 1024 * 1024));
    const fixtures = try std.json.parseFromSlice(V, a, bytes, .{});
    var frames: usize = 0;
    for (fixtures.value.array.items, 0..) |fixture, index| {
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const temp = scratch.allocator();
        const text = fixture.object.get("text").?.string;
        const tools = fixture.object.get("tools").?;
        var stream = Streamer{};
        var actual_args: std.ArrayList(u8) = .empty;
        var expected_args: std.ArrayList(u8) = .empty;
        for (fixture.object.get("frames").?.array.items) |frame| {
            const end: usize = @intCast(frame.object.get("end").?.integer);
            var capture = Capture{ .a = temp };
            try stream.feed(temp, text[0..end], tools, &capture, Capture.emit);
            const expected = frame.object.get("deltas").?.array.items;
            var same = expected.len == capture.deltas.items.len;
            if (same) for (expected, capture.deltas.items) |item, got| {
                const call = item.object.get("tool_calls").?.array.items[0];
                const function = call.object.get("function").?;
                const name = function.object.get("name");
                same = same and got.index == call.object.get("index").?.integer and ((name == null) == (got.name == null));
                if (name != null and got.name != null) {
                    same = same and std.mem.eql(u8, name.?.string, got.name.?);
                    actual_args.clearRetainingCapacity();
                    expected_args.clearRetainingCapacity();
                }
                try actual_args.appendSlice(temp, got.arguments);
                try expected_args.appendSlice(temp, function.object.get("arguments").?.string);
                // Ignore JSON formatting spaces outside strings, including in
                // partial documents; whitespace inside argument strings is exact.
                same = same and std.mem.eql(u8, try compact(temp, actual_args.items), try compact(temp, expected_args.items));
            };
            if (!same) {
                std.debug.print("Stream fixture {d}, byte {d}: {s}\nExpected: {s}\nGot: {s}\n", .{ index, end, text[0..end], try std.json.Stringify.valueAlloc(temp, expected, .{}), try std.json.Stringify.valueAlloc(temp, capture.deltas.items, .{}) });
                return error.ToolStreamMismatch;
            }
            frames += 1;
        }
    }
    std.debug.print("PASS: {d} incremental tool streams, {d} prefix frames match upstream\n", .{ fixtures.value.array.items.len, frames });
}

test "tool argument deltas arrive before the envelope closes and preserve Unicode and framing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try std.json.parseFromSlice(V, a,
        \\[{"type":"function","function":{"name":"write","parameters":{"properties":{"path":{"type":"string"},"days":{"type":"integer"}}}}}]
    , .{})).value;
    const text = "before <tool_call><function=write><parameter=path>\n  æøå 世界 👋 \\\"quoted\\\"\n\n</parameter><parameter=days>\n2\n</parameter></function></tool_call> after";
    for ([_]usize{ 1, 3, 17, text.len }) |step| {
        var stream = Streamer{};
        var capture = Capture{ .a = a };
        var end: usize = 0;
        while (end < text.len) {
            end = @min(text.len, end + step);
            while (end < text.len and text[end] & 0xc0 == 0x80) end += 1;
            try stream.feed(a, text[0..end], tools, &capture, Capture.emit);
            if (end >= 100 and end < text.len - 10) try std.testing.expect(capture.deltas.items.len > 2);
        }
        var args: std.ArrayList(u8) = .empty;
        for (capture.deltas.items) |delta| try args.appendSlice(a, delta.arguments);
        const parsed = try calls.parse(a, text, tools, null, "test");
        try std.testing.expectEqualStrings(parsed.calls[0].function.arguments, args.items);
        try std.testing.expectEqual(@as(usize, 1), stream.count);
    }
}
