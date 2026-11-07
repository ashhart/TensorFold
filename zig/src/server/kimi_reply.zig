//! XTML replies use actual added-token controls; ordinary text spelling those controls remains literal.
const std = @import("std");
const json = @import("json");
const mt = @import("model_text.zig");
const specs = @import("tool_specs.zig");
const values = @import("kimi_values.zig");
const V = json.Value;
const A = std.mem.Allocator;

pub const Result = struct {
    content: []const u8,
    reasoning: []const u8,
    calls: ?[]V,
    reasoning_tokens: usize,
    rejected: bool = false,
};
const Tag = enum { message, think, response, tools, call, argument, json };
const Header = struct { tag: Tag, attrs: *json.Object };

fn header(a: A, text: []const u8) A.Error!?Header {
    const end = std.mem.indexOfScalar(u8, text, ' ') orelse text.len;
    const tag = std.meta.stringToEnum(Tag, text[0..end]) orelse return null;
    const attrs = try json.newObject(a);
    var at = end;
    while (at < text.len) {
        if (text[at] != ' ') return null;
        at += 1;
        const key_end = std.mem.indexOfPos(u8, text, at, "=\"") orelse return null;
        if (key_end == at) return null;
        const key = text[at..key_end];
        const start = key_end + 2;
        const stop = std.mem.indexOfScalarPos(u8, text, start, '"') orelse return null;
        if (attrs.contains(key)) return null;
        const quotes = try std.mem.replaceOwned(u8, a, text[start..stop], "&quot;", "\"");
        const decoded = try std.mem.replaceOwned(u8, a, quotes, "&amp;", "&");
        try attrs.put(a, key, .{ .string = decoded });
        at = stop + 1;
    }
    return .{ .tag = tag, .attrs = attrs };
}

const Reader = struct {
    a: A,
    text: mt.Text,
    offered: []const V,
    stack: std.ArrayList(Tag) = .empty,
    content: std.ArrayList(u8) = .empty,
    thought: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(V) = .empty,
    body: std.ArrayList(u8) = .empty,
    name: []const u8 = "",
    index: []const u8 = "",
    key: []const u8 = "",
    kind: []const u8 = "",
    args: ?*json.Object = null,
    argument_wire: std.ArrayList(u8) = .empty,
    raw_object: ?[]const u8 = null,
    rejected: bool = false,
    bad_call: bool = false,
    reasoning_tokens: usize = 0,

    fn top(r: *const Reader) Tag {
        return if (r.stack.items.len > 0) r.stack.items[r.stack.items.len - 1] else .message;
    }
    fn prose(r: *Reader, ids: []const u32) A.Error!void {
        const s = try r.text.decode(r.a, ids);
        switch (r.top()) {
            .think => {
                try r.thought.appendSlice(r.a, s);
                r.reasoning_tokens += ids.len;
            },
            .response => try r.content.appendSlice(r.a, s),
            .argument, .json => try r.body.appendSlice(r.a, s),
            else => if (std.mem.trim(u8, s, " \t\r\n").len != 0) {
                r.rejected = true;
                r.bad_call = true;
            },
        }
    }
    fn open(r: *Reader, h: Header) A.Error!void {
        const parent = r.top();
        const attrs: V = .{ .object = h.attrs };
        const attrs_valid = switch (h.tag) {
            .call => h.attrs.count() == 2 and h.attrs.contains("tool") and h.attrs.contains("index"),
            .argument => h.attrs.count() == 2 and h.attrs.contains("key") and h.attrs.contains("type"),
            .json => h.attrs.count() == 1 and h.attrs.contains("type"),
            else => h.attrs.count() == 0,
        };
        const valid = attrs_valid and switch (h.tag) {
            .think, .response, .tools => parent == .message,
            .call => parent == .tools,
            .argument, .json => parent == .call,
            .message => false,
        };
        if (!valid) {
            r.rejected = true;
            r.bad_call = true;
        }
        if (h.tag == .call) {
            r.name = attrs.strField("tool") orelse "";
            r.index = attrs.strField("index") orelse "";
            r.args = try json.newObject(r.a);
            r.argument_wire.clearRetainingCapacity();
            r.raw_object = null;
            r.bad_call = r.rejected or !valid or r.name.len == 0;
            const index = std.fmt.parseInt(usize, r.index, 10) catch 0;
            if (index != r.calls.items.len + 1) r.bad_call = true;
        } else if (h.tag == .argument or h.tag == .json) {
            r.body.clearRetainingCapacity();
            r.key = attrs.strField("key") orelse "";
            r.kind = attrs.strField("type") orelse "";
            if (h.tag == .json and !std.mem.eql(u8, r.kind, "object")) r.bad_call = true;
        }
        try r.stack.append(r.a, h.tag);
    }
    fn close(r: *Reader, h: Header) A.Error!void {
        if (h.attrs.count() != 0 or r.stack.items.len == 0 or h.tag != r.top()) {
            r.rejected = true;
            r.bad_call = true;
            return;
        }
        _ = r.stack.pop();
        switch (h.tag) {
            .argument, .json => {
                const args = r.args orelse return;
                var v: V = .{ .string = try r.a.dupe(u8, r.body.items) };
                if (!std.mem.eql(u8, r.kind, "string")) {
                    v = switch (try json.parse(r.a, r.body.items)) {
                        .ok => |item| item,
                        .err => {
                            r.bad_call = true;
                            return;
                        },
                    };
                    if (!std.mem.eql(u8, values.kind(v), r.kind)) r.bad_call = true;
                }
                if (h.tag == .json) {
                    if (v != .object or args.count() != 0 or r.raw_object != null) {
                        r.bad_call = true;
                    } else {
                        r.args = v.object;
                        r.raw_object = try r.a.dupe(u8, r.body.items);
                    }
                } else if (args.contains(r.key) or r.raw_object != null) {
                    r.bad_call = true;
                } else {
                    if (args.count() != 0) try r.argument_wire.append(r.a, ',');
                    try r.argument_wire.appendSlice(r.a, try json.stringify(r.a, .{ .string = r.key }, .{ .ascii = false, .compact = true }));
                    try r.argument_wire.append(r.a, ':');
                    const literal_bytes = if (v == .string) try json.stringify(r.a, v, .{ .ascii = false, .compact = true }) else r.body.items;
                    try r.argument_wire.appendSlice(r.a, literal_bytes);
                    try args.put(r.a, r.key, v);
                }
            },
            .call => {
                var known = false;
                for (r.offered) |tool| if (std.mem.eql(u8, try specs.toolName(r.a, tool), r.name)) {
                    known = true;
                };
                if (r.bad_call or !known) {
                    r.rejected = true;
                    return;
                }
                const f = try json.newObject(r.a);
                try f.put(r.a, "name", .{ .string = r.name });
                try f.put(r.a, "arguments", .{ .string = r.raw_object orelse try std.mem.concat(r.a, u8, &.{ "{", r.argument_wire.items, "}" }) });
                const c = try json.newObject(r.a);
                try c.put(r.a, "id", .{ .string = try std.mem.concat(r.a, u8, &.{ "call_", r.index }) });
                try c.put(r.a, "type", .{ .string = "function" });
                try c.put(r.a, "function", .{ .object = f });
                try r.calls.append(r.a, .{ .object = c });
            },
            else => {},
        }
    }
};

/// The prompt already opens an assistant message and its first channel; only complete calls are executable.
pub fn parse(a: A, text: mt.Text, ids: []const u32, thinking: bool, offered: []const V) A.Error!Result {
    var r: Reader = .{ .a = a, .text = text, .offered = offered };
    try r.stack.appendSlice(a, &.{ .message, if (thinking) .think else .response });
    const open = text.tokenId("<|open|>") orelse return literal(a, text, ids);
    const close = text.tokenId("<|close|>") orelse return literal(a, text, ids);
    const sep = text.tokenId("<|sep|>") orelse return literal(a, text, ids);
    const end = text.tokenId("<|end_of_msg|>") orelse return literal(a, text, ids);
    var at: usize = 0;
    while (at < ids.len) {
        const token = ids[at];
        if (token == end) break;
        if (token == open or token == close) {
            const start = at + 1;
            at = start;
            while (at < ids.len and !text.isAdded(ids[at])) : (at += 1) {}
            if (at == ids.len) break;
            const h = if (ids[at] == sep) try header(a, try text.decode(a, ids[start..at])) else null;
            if (h) |tag| {
                if (token == open) try r.open(tag) else try r.close(tag);
            } else {
                r.rejected = true;
                r.bad_call = true;
            }
            at += 1;
        } else if (text.isAdded(token)) {
            r.rejected = true;
            r.bad_call = true;
            at += 1;
        } else {
            const start = at;
            while (at < ids.len and !text.isAdded(ids[at])) : (at += 1) {}
            try r.prose(ids[start..at]);
        }
    }
    return .{ .content = r.content.items, .reasoning = r.thought.items, .calls = if (!r.rejected and r.calls.items.len > 0) r.calls.items else null, .reasoning_tokens = r.reasoning_tokens, .rejected = r.rejected };
}

fn literal(a: A, text: mt.Text, ids: []const u32) A.Error!Result {
    return .{ .content = try text.decode(a, ids), .reasoning = "", .calls = null, .reasoning_tokens = 0, .rejected = true };
}
