//! Note and reply are ordinary greedy or keyed generations, so the same request compacts and answers the same way.
const std = @import("std");
const json = @import("json.zig");
const chat = @import("chat.zig");
const stored = @import("compact_mem.zig");
const server_mod = @import("server.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Server = server_mod.Server;
const CompactAt = server_mod.CompactAt;
const Cx = @import("errors.zig").Cx;

pub const marker_line = "Memory of the earlier conversation:";

const note_task =
    \\Rewrite the memory note for this conversation so it keeps everything still needed.
    \\Markdown sections: Goal, Constraints and preferences, Progress, Decisions, Next steps, Critical context.
    \\Progress lists done, in progress and blocked. Decisions keep every earlier one and its reason.
    \\Next steps stay in order. Critical context keeps exact names, paths, numbers and error text.
    \\Under 800 words.
;

const split_task =
    \\Summarize the cut part of this turn.
    \\Keep the original request, the early progress, and what the kept reply needs in order to make sense.
;

const files_head = "## Files read and changed\n";
const split_head = "## Kept reply context\n";

pub const Stamp = struct {
    window: i64,
    used: usize,
    compacted: bool = false,
    messages_compacted: usize = 0,
    kept_messages: usize = 0,
    note_tokens: usize = 0,
    note: []const u8 = "",
};

pub const Ready = struct { prepared: chat.Prepared, stamp: Stamp };

const Piece = struct { text: []const u8, tokens: usize };

pub const conversationKey = stored.conversationKey;

pub fn reserveTokens(window: i64, at: CompactAt) i64 {
    return switch (at) {
        .auto => @max(@divTrunc(window * 15, 100), 16384),
        .fraction => |f| @max(1, window - threshold(window, f)),
    };
}

pub fn noteBudgetTokens(window: i64, at: CompactAt) i64 {
    return @min(8192, @max(1, @divTrunc(reserveTokens(window, at) * 4, 5)));
}

pub fn keepTokens(window: u32, override: ?u32) usize {
    if (override) |k| return k;
    return @min(20000, @as(usize, window) / 4);
}

fn threshold(window: i64, f: f64) i64 {
    const w: f64 = @floatFromInt(window);
    const t = @floor(f * w);
    if (t <= 0) return 0;
    if (t >= w) return window;
    return @intFromFloat(t);
}

fn over(prompt: usize, max_out: i64, window: i64, at: CompactAt) bool {
    const sum = @as(i64, @intCast(prompt)) + @max(0, max_out);
    return switch (at) {
        .auto => sum > window - reserveTokens(window, at),
        .fraction => |f| sum > threshold(window, f),
    };
}

fn wanted(srv: *Server, input: chat.Input) i64 {
    if (input.max_tokens) |m| if (m != 0) return m;
    return srv.config.default_max_tokens;
}

fn chunkRoom(window: i64, at: CompactAt) usize {
    const budget = noteBudgetTokens(window, at);
    if (window <= budget) return 1;
    return @intCast(window - budget);
}

pub fn run(srv: *Server, cx: *Cx, input: chat.Input, sink: ?chat.Sink, gone: anytype) chat.Failure!chat.Reply {
    const ready = try prepare(srv, cx, input, gone);
    var reply = try chat.generate(srv, cx, ready.prepared, sink, gone);
    try stamp(cx, &reply, ready.stamp);
    return reply;
}

pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, gone: anytype) chat.Failure!Ready {
    const at = srv.config.compact_at orelse {
        const prepared = try chat.prepare(srv, cx, input, gone);
        return .{ .prepared = prepared, .stamp = .{ .window = srv.info.context_window, .used = prepared.prompt_len } };
    };
    const window: i64 = srv.info.context_window;
    if (input.messages != .array or window <= 0) {
        const prepared = try chat.prepare(srv, cx, input, gone);
        return .{ .prepared = prepared, .stamp = .{ .window = window, .used = prepared.prompt_len } };
    }
    if (chat.prepare(srv, cx, input, gone)) |prepared| {
        if (!over(prepared.prompt_len, wanted(srv, input), window, at))
            return .{ .prepared = prepared, .stamp = .{ .window = window, .used = prepared.prompt_len } };
        chat.release(srv, prepared.preparing);
    } else |e| if (!(e == error.Refused and cx.kind == .context_length)) return e;
    return rebuild(srv, cx, input, gone, at, input.messages.array);
}

pub fn stamp(cx: *Cx, reply: *chat.Reply, s: Stamp) chat.Failure!void {
    if (reply.runtime != .object) return;
    const o = reply.runtime.object;
    try o.put(cx.a, "context_window", try json.intValue(cx.a, s.window));
    try o.put(cx.a, "context_used", try json.intValue(cx.a, s.used));
    if (!s.compacted) return;
    const c = try json.newObject(cx.a);
    try c.put(cx.a, "messages_compacted", try json.intValue(cx.a, s.messages_compacted));
    try c.put(cx.a, "kept_messages", try json.intValue(cx.a, s.kept_messages));
    try c.put(cx.a, "note_tokens", try json.intValue(cx.a, s.note_tokens));
    try c.put(cx.a, "note", .{ .string = s.note });
    try o.put(cx.a, "compaction", .{ .object = c });
}

fn rebuild(srv: *Server, cx: *Cx, input: chat.Input, gone: anytype, at: CompactAt, msgs: []Value) chat.Failure!Ready {
    const a = cx.a;
    const marker = findMarker(msgs);
    const slot = stored.loadMem(srv, a, msgs);
    const hash_ok = if (slot) |m| stored.hashes(a, msgs, m) else false;
    var prev: []const u8 = "";
    if (hash_ok) {
        prev = slot.?.note;
    } else if (marker) |i| {
        prev = markerBody(msgs[i]) orelse "";
    }
    const skip: usize = if (hash_ok) slot.?.covered else if (marker) |i| i + 1 else 0;
    const prefix = prefixLen(msgs);
    var tail = try keptStart(srv, a, msgs, keepTokens(srv.info.context_window, srv.config.compact_keep), marker);
    if (tail < prefix) tail = prefix;
    const split_at = splitIndex(msgs, prefix, tail, marker);
    const middle_to = split_at orelse tail;
    var tokens: usize = 0;
    const middle = try collect(a, msgs, prefix, middle_to, marker, skip);
    var model = try update(srv, cx, gone, at, note_task, withoutExtra(prev), middle);
    tokens += model.tokens;
    var split_text: []const u8 = "";
    if (split_at) |s| {
        const part = try collect(a, msgs, s, tail, marker, skip);
        if (part.len > 0) {
            const piece = try update(srv, cx, gone, at, split_task, "", part);
            split_text = piece.text;
            tokens += piece.tokens;
        }
    }
    var include_split = split_text.len > 0;
    var guard = msgs.len + 2;
    while (guard > 0) : (guard -= 1) {
        var paths: std.ArrayList([]const u8) = .empty;
        try pathsFromNote(a, &paths, prev);
        try pathsFromMsgs(a, &paths, msgs[prefix..@min(tail, msgs.len)]);
        const note = try finishNote(a, model.text, split_text, paths.items);
        const built = try assemble(a, srv, msgs, note, split_text, include_split, tail, marker);
        var next = input;
        next.messages = .{ .array = built };
        if (chat.prepare(srv, cx, next, gone)) |prepared| {
            stored.storeMem(srv, a, msgs, tail, note);
            return .{
                .prepared = prepared,
                .stamp = .{
                    .window = srv.info.context_window,
                    .used = prepared.prompt_len,
                    .compacted = true,
                    .messages_compacted = countBetween(msgs, prefix, tail, marker),
                    .kept_messages = countBetween(msgs, tail, msgs.len, marker),
                    .note_tokens = tokens,
                    .note = note,
                },
            };
        } else |e| {
            if (!(e == error.Refused and cx.kind == .context_length)) return e;
            if (include_split) {
                include_split = false;
                continue;
            }
            const nxt = nextCut(msgs, tail, marker) orelse return e;
            const dropped = try collect(a, msgs, tail, nxt, marker, 0);
            const piece = try update(srv, cx, gone, at, note_task, withoutExtra(model.text), dropped);
            model = piece;
            tokens += piece.tokens;
            tail = nxt;
        }
    }
    const prepared = try chat.prepare(srv, cx, input, gone);
    return .{ .prepared = prepared, .stamp = .{ .window = srv.info.context_window, .used = prepared.prompt_len } };
}

pub fn keptStart(srv: *Server, a: Allocator, msgs: []const Value, keep: usize, marker: ?usize) chat.Failure!usize {
    const prefix = prefixLen(msgs);
    var starts: std.ArrayList(usize) = .empty;
    for (msgs, 0..) |m, i| {
        if (i < prefix or (marker != null and i == marker.?)) continue;
        if (isCut(m)) try starts.append(a, i);
    }
    if (starts.items.len == 0) return prefix;
    var acc: usize = 0;
    var g = starts.items.len;
    while (g > 0) {
        g -= 1;
        const from = starts.items[g];
        const to = if (g + 1 < starts.items.len) starts.items[g + 1] else msgs.len;
        acc += try countRange(srv, a, msgs, from, to, marker);
        if (acc >= keep) return from;
    }
    return starts.items[0];
}

fn update(srv: *Server, cx: *Cx, gone: anytype, at: CompactAt, task: []const u8, prev: []const u8, msgs: []const Value) chat.Failure!Piece {
    if (msgs.len == 0) return .{ .text = prev, .tokens = 0 };
    const room = chunkRoom(srv.info.context_window, at);
    var note = prev;
    var tokens: usize = 0;
    var i: usize = 0;
    while (i < msgs.len) {
        var j = i + 1;
        var n = try countMsgs(srv, cx.a, msgs[i..j]);
        while (j < msgs.len) {
            const more = try countMsgs(srv, cx.a, msgs[j .. j + 1]);
            if (n + more > room) break;
            n += more;
            j += 1;
        }
        const piece = try oneNote(srv, cx, gone, task, note, msgs[i..j]);
        note = piece.text;
        tokens += piece.tokens;
        i = j;
    }
    return .{ .text = note, .tokens = tokens };
}

fn oneNote(srv: *Server, cx: *Cx, gone: anytype, task: []const u8, prev: []const u8, msgs: []const Value) chat.Failure!Piece {
    const a = cx.a;
    var list: std.ArrayList(Value) = .empty;
    try list.append(a, try msg(a, "system", task));
    if (prev.len > 0) try list.append(a, try msg(a, "user", try std.fmt.allocPrint(a, "Previous note:\n{s}", .{prev})));
    const script = try transcript(a, msgs);
    try list.append(a, try msg(a, "user", try std.fmt.allocPrint(a, "Messages to add:\n{s}", .{script})));
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = false });
    try fields.put(a, "temperature", .{ .float = 0 });
    var prepared = chat.prepare(srv, cx, .{ .messages = .{ .array = list.items }, .fields = .{ .object = fields }, .max_tokens = 1024, .temperature = 0 }, gone) catch |e| blk: {
        if (!(e == error.Refused and cx.kind == .context_length)) return e;
        cx.kind = .request;
        cx.message = "";
        const fitted = chat.prepare(srv, cx, .{ .messages = .{ .array = list.items }, .fields = .{ .object = fields }, .max_tokens = 1, .temperature = 0 }, gone) catch |e2| {
            if (e2 == error.Refused and cx.kind == .context_length) {
                cx.kind = .request;
                cx.message = "";
                return .{ .text = prev, .tokens = 0 };
            }
            return e2;
        };
        const room = srv.info.context_window - @as(i64, @intCast(fitted.prompt_len));
        if (room < 1) {
            chat.release(srv, fitted.preparing);
            return .{ .text = prev, .tokens = 0 };
        }
        var out = fitted;
        out.request.max_tokens = @intCast(@min(@as(i64, 1024), room));
        break :blk out;
    };
    prepared.request.drafts = true; // the note stays a drafted generation when replies pass --no-drafts
    prepared.drafts = true;
    const reply = try chat.generate(srv, cx, prepared, null, gone);
    return .{ .text = reply.content, .tokens = reply.completion_tokens };
}

fn assemble(a: Allocator, srv: *Server, msgs: []const Value, note: []const u8, split: []const u8, include_split: bool, tail: usize, marker: ?usize) ![]Value {
    var out: std.ArrayList(Value) = .empty;
    const prefix = prefixLen(msgs);
    for (msgs[0..prefix], 0..) |m, i| {
        if (marker != null and i == marker.?) continue;
        try out.append(a, m);
    }
    if (note.len > 0) try out.append(a, try msg(a, srv.late_system, try std.fmt.allocPrint(a, "{s}\n{s}", .{ marker_line, note })));
    if (include_split and split.len > 0) try out.append(a, try msg(a, "user", try std.fmt.allocPrint(a, "Earlier in this turn:\n{s}", .{split})));
    for (msgs[tail..], tail..) |m, i| {
        if (marker != null and i == marker.?) continue;
        try out.append(a, m);
    }
    if (out.items.len == 0) try out.append(a, try msg(a, "user", ""));
    return out.items;
}

fn collect(a: Allocator, msgs: []const Value, from: usize, to: usize, marker: ?usize, skip: usize) ![]Value {
    var out: std.ArrayList(Value) = .empty;
    var i = from;
    while (i < to and i < msgs.len) : (i += 1) {
        if (i < skip or (marker != null and i == marker.?)) continue;
        try out.append(a, msgs[i]);
    }
    return out.items;
}

fn finishNote(a: Allocator, body: []const u8, split: []const u8, paths: []const []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(a, withoutExtra(body));
    if (split.len > 0) {
        if (buf.items.len > 0) try buf.appendSlice(a, "\n\n");
        try buf.appendSlice(a, split_head);
        try buf.appendSlice(a, split);
    }
    if (paths.len > 0) {
        if (buf.items.len > 0) try buf.appendSlice(a, "\n\n");
        try buf.appendSlice(a, files_head);
        for (paths) |p| {
            try buf.appendSlice(a, "- ");
            try buf.appendSlice(a, p);
            try buf.append(a, '\n');
        }
    }
    return buf.items;
}

fn withoutExtra(note: []const u8) []const u8 {
    var end = note.len;
    for ([_][]const u8{ files_head, split_head }) |h| if (std.mem.indexOf(u8, note, h)) |i| {
        var at = i;
        if (at > 0 and note[at - 1] == '\n') at -= 1;
        if (at > 0 and note[at - 1] == '\n') at -= 1;
        if (at < end) end = at;
    };
    return note[0..end];
}

fn msg(a: Allocator, role: []const u8, content: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}

fn roleOf(m: Value) []const u8 {
    const r = m.get("role") orelse return "";
    return if (r == .string) r.string else "";
}

fn contentOf(m: Value) []const u8 {
    const c = m.get("content") orelse return "";
    return if (c == .string) c.string else "";
}

fn isLead(m: Value) bool {
    const r = roleOf(m);
    return std.mem.eql(u8, r, "system") or std.mem.eql(u8, r, "developer");
}

fn isCut(m: Value) bool {
    const r = roleOf(m);
    return std.mem.eql(u8, r, "user") or std.mem.eql(u8, r, "assistant");
}

fn prefixLen(msgs: []const Value) usize {
    var i: usize = 0;
    while (i < msgs.len and isLead(msgs[i])) : (i += 1) {}
    return i;
}

fn findMarker(msgs: []const Value) ?usize {
    var found: ?usize = null;
    for (msgs, 0..) |m, i| {
        if (markerBody(m) != null) found = i;
    }
    return found;
}

fn markerBody(m: Value) ?[]const u8 {
    const r = roleOf(m);
    if (!std.mem.eql(u8, r, "system") and !std.mem.eql(u8, r, "user") and !std.mem.eql(u8, r, "developer")) return null;
    const c = contentOf(m);
    if (!std.mem.startsWith(u8, c, marker_line)) return null;
    var rest = c[marker_line.len..];
    if (std.mem.startsWith(u8, rest, "\n")) rest = rest[1..];
    return rest;
}

fn splitIndex(msgs: []const Value, prefix: usize, tail: usize, marker: ?usize) ?usize {
    if (tail >= msgs.len or !std.mem.eql(u8, roleOf(msgs[tail]), "assistant")) return null;
    var i = tail;
    while (i > prefix) {
        i -= 1;
        if (marker != null and i == marker.?) continue;
        const r = roleOf(msgs[i]);
        if (std.mem.eql(u8, r, "user")) return i;
        if (std.mem.eql(u8, r, "assistant")) return null;
    }
    return null;
}

fn nextCut(msgs: []const Value, tail: usize, marker: ?usize) ?usize {
    const last = lastUser(msgs, marker) orelse return null;
    var i = tail + 1;
    while (i < msgs.len) : (i += 1) {
        if (marker != null and i == marker.?) continue;
        if (isCut(msgs[i])) return if (i > last) null else i;
    }
    return null;
}

fn lastUser(msgs: []const Value, marker: ?usize) ?usize {
    var i = msgs.len;
    while (i > 0) {
        i -= 1;
        if (marker != null and i == marker.?) continue;
        if (std.mem.eql(u8, roleOf(msgs[i]), "user")) return i;
    }
    return null;
}

fn countBetween(msgs: []const Value, from: usize, to: usize, marker: ?usize) usize {
    var n: usize = 0;
    var i = from;
    while (i < to and i < msgs.len) : (i += 1) {
        if (marker != null and i == marker.?) continue;
        n += 1;
    }
    return n;
}

fn countMsgs(srv: *Server, a: Allocator, msgs: []const Value) chat.Failure!usize {
    return countRange(srv, a, msgs, 0, msgs.len, null);
}

fn countRange(srv: *Server, a: Allocator, msgs: []const Value, from: usize, to: usize, marker: ?usize) chat.Failure!usize {
    var n: usize = 0;
    var i = from;
    while (i < to and i < msgs.len) : (i += 1) {
        if (marker != null and i == marker.?) continue;
        n += try countOne(srv, a, msgs[i]);
    }
    return n;
}

fn countOne(srv: *Server, a: Allocator, m: Value) chat.Failure!usize {
    const line = try lineOf(a, m);
    const ids = srv.text.encode(a, line, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return error.Failed,
    };
    return ids.len;
}

fn lineOf(a: Allocator, m: Value) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(a, roleOf(m));
    try buf.appendSlice(a, ": ");
    try buf.appendSlice(a, contentOf(m));
    try appendCalls(a, &buf, m);
    return buf.items;
}

fn transcript(a: Allocator, msgs: []const Value) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    for (msgs, 0..) |m, i| {
        if (i > 0) try buf.append(a, '\n');
        try buf.appendSlice(a, try lineOf(a, m));
    }
    return buf.items;
}

fn appendCalls(a: Allocator, buf: *std.ArrayList(u8), m: Value) !void {
    const calls = m.get("tool_calls") orelse return;
    if (calls != .array) return;
    for (calls.array) |call| {
        const function = call.get("function") orelse continue;
        try buf.appendSlice(a, " call ");
        try buf.appendSlice(a, json.strOr(function.get("name")));
        if (function.get("arguments")) |args| {
            try buf.append(a, ' ');
            if (args == .string) try buf.appendSlice(a, args.string) else {
                const text = json.stringify(a, args, .{ .compact = true }) catch "";
                try buf.appendSlice(a, text);
            }
        }
    }
}

fn pathsFromMsgs(a: Allocator, list: *std.ArrayList([]const u8), msgs: []const Value) !void {
    for (msgs) |m| {
        const calls = m.get("tool_calls") orelse continue;
        if (calls != .array) continue;
        for (calls.array) |call| {
            const function = call.get("function") orelse continue;
            if (function.get("arguments")) |args| try addArgs(a, list, args);
        }
    }
}

fn addArgs(a: Allocator, list: *std.ArrayList([]const u8), args: Value) !void {
    var obj = args;
    if (args == .string) {
        const parsed = json.parseText(a, args.string) catch return;
        switch (parsed) {
            .ok => |v| {
                if (v != .object) return;
                obj = v;
            },
            .err => return,
        }
    }
    if (obj != .object) return;
    for ([_][]const u8{ "path", "file_path", "filename", "file" }) |k| if (obj.get(k)) |v| if (v == .string and v.string.len > 0) try addUnique(a, list, v.string);
    if (obj.get("paths")) |v| {
        if (v == .string and v.string.len > 0) try addUnique(a, list, v.string) else if (v == .array) for (v.array) |item| if (item == .string and item.string.len > 0) try addUnique(a, list, item.string);
    }
}

fn pathsFromNote(a: Allocator, list: *std.ArrayList([]const u8), note: []const u8) !void {
    const i = std.mem.indexOf(u8, note, files_head) orelse return;
    var rest = note[i + files_head.len ..];
    while (rest.len > 0) {
        const end = std.mem.indexOfScalar(u8, rest, '\n') orelse rest.len;
        const line = rest[0..end];
        if (std.mem.startsWith(u8, line, "- ")) {
            const p = std.mem.trim(u8, line[2..], " \t");
            if (p.len > 0) try addUnique(a, list, p);
        } else if (line.len > 0) break;
        if (end == rest.len) break;
        rest = rest[end + 1 ..];
    }
}

fn addUnique(a: Allocator, list: *std.ArrayList([]const u8), p: []const u8) !void {
    for (list.items) |seen| if (std.mem.eql(u8, seen, p)) return;
    try list.append(a, p);
}
