//! Replies written as messages to recipients: `<|start|>assistant to=R<|message|>body<|eom|>`, the last one `<|eot|>`.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_specs = @import("tool_specs.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Split = reply_text.Split;
const findTag = reply_text.findTag;

const start = "<|start|>";
const message = "<|message|>";
const ends = [_][]const u8{ "<|eom|>", "<|eot|>" };
const closers = ends ++ [_][]const u8{start};
const block = "<atem:function_calls>";
const block_close = "</atem:function_calls>";
const invoke = "<atem:invoke name=\"";
const invoke_close = "</atem:invoke>";
const parameter = "<atem:parameter name=\"";
const parameter_close = "</atem:parameter>";

/// Whether replies take this form: every marker is one token and the template writes `to=self` messages.
pub fn detect(text: anytype) bool {
    for ([_][]const u8{ start, message, ends[0], ends[1] }) |m| if (text.tokenId(m) == null) return false;
    return std.mem.indexOf(u8, text.templateSource(), "to=self") != null;
}

/// How the text ends: still streaming, at the model's end token, cut by max_tokens, or at a stop string.
pub const End = enum { streaming, ended, cut, stopped };

/// `self` is reasoning; `user`, `commentary` and no recipient are the answer; any other recipient is a tool call.
const Kind = enum { reasoning, answer, call };

/// The recipient a message header names (`to=R`); null: none.
fn recipientOf(header: []const u8) ?[]const u8 {
    const at = std.mem.indexOf(u8, header, "to=") orelse return null;
    const rest = header[at + 3 ..];
    return rest[0 .. std.mem.indexOfAny(u8, rest, " \t\r\n") orelse rest.len];
}

fn kindOf(header: []const u8) Kind {
    const r = recipientOf(header) orelse return .answer;
    if (std.mem.eql(u8, r, "self")) return .reasoning;
    if (r.len == 0 or std.mem.eql(u8, r, "user") or std.mem.eql(u8, r, "commentary")) return .answer;
    return .call;
}

/// The template's attribute escapes (`esc_attr`) undone.
fn unescape(a: Allocator, s: []const u8) Allocator.Error![]const u8 {
    if (std.mem.indexOfScalar(u8, s, '&') == null) return s;
    const escapes = [_][2][]const u8{ .{ "&amp;", "&" }, .{ "&quot;", "\"" }, .{ "&#39;", "'" }, .{ "&lt;", "<" }, .{ "&gt;", ">" } };
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    outer: while (i < s.len) {
        for (escapes) |e| if (std.mem.startsWith(u8, s[i..], e[0])) {
            try out.appendSlice(a, e[1]);
            i += e[0].len;
            continue :outer;
        };
        try out.append(a, s[i]);
        i += 1;
    }
    return out.items;
}

/// The offered tool's spelling of `name`, as tool_parse matches a call's name; null: not offered.
fn offered(a: Allocator, name: []const u8, tools: []const Value) Allocator.Error!?[]const u8 {
    for (tools) |tool| {
        const spelled = try tool_specs.toolName(a, tool);
        if (spelled.len > 0 and std.ascii.eqlIgnoreCase(spelled, name)) return spelled;
    }
    return null;
}

/// A call message as the `<tool_call>` markup tool_parse reads, a cut one left open; null: keep the model's text.
pub fn calls(a: Allocator, body: []const u8, recipient: []const u8, tools: []const Value, end: End) Allocator.Error!?[]const u8 {
    if (tools.len == 0 or end == .stopped) return null; // no tools offered, or a stop string inside the call
    const cut = end == .cut;
    const lead = reply_text.pyLstrip(body);
    if (!std.mem.startsWith(u8, lead, block) and !(cut and std.mem.startsWith(u8, block, lead))) return null;
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (findTag(body, pos, invoke)) |s| {
        const q = std.mem.indexOfPos(u8, body, s + invoke.len, "\">") orelse break; // its name is still being written
        const name = (try offered(a, try unescape(a, body[s + invoke.len .. q]), tools)) orelse return null;
        const close = std.mem.indexOfPos(u8, body, q, invoke_close);
        const stop = close orelse body.len;
        if (out.items.len > 0) try out.append(a, '\n');
        try out.print(a, "<tool_call>\n<function={s}>\n", .{name});
        var p = q + 2;
        while (findTag(body[0..stop], p, parameter)) |ps| {
            const pq = std.mem.indexOfPos(u8, body[0..stop], ps + parameter.len, "\">") orelse break; // a cut name
            const pe = std.mem.indexOfPos(u8, body[0..stop], pq + 2, parameter_close);
            const value = body[pq + 2 .. pe orelse stop - (if (cut) reply_text.partialTag(body[pq + 2 .. stop], parameter_close) else 0)];
            try out.print(a, "<parameter={s}>\n{s}", .{ try unescape(a, body[ps + parameter.len .. pq]), value });
            if (pe == null and cut) return out.items; // a cut value: its closers never came
            try out.appendSlice(a, "\n</parameter>\n");
            p = pe orelse stop;
        }
        if (close == null and cut) return out.items; // cut between its parameters
        try out.appendSlice(a, "</function>\n");
        pos = if (close) |c| c + invoke_close.len else stop;
        if (cut and findTag(body, pos, invoke) == null and findTag(body, pos, block_close) == null) return out.items; // block open
        try out.appendSlice(a, "</tool_call>");
    }
    if (out.items.len == 0 and cut and findTag(body, 0, block_close) == null) { // cut before its invoke names it
        const name = (try offered(a, recipient, tools)) orelse return null;
        return try std.fmt.allocPrint(a, "<tool_call>\n<function={s}>\n", .{name});
    }
    return if (out.items.len > 0) out.items else null;
}

/// The open message of a reply being read: its kind, recipient and body start, how far it was scanned and sent.
const Open = struct { kind: Kind, recipient: []const u8, b0: usize, scan: usize, sent: usize = 0, begun: bool = false };

/// The longest closer less a byte: a closer that began before a scan's end is found from this far back.
const reach = blk: {
    var n: usize = 0;
    for (closers) |e| n = @max(n, e.len);
    break :blk n - 1;
};

/// `bytes` after `out`'s; a full list moves to a buffer twice the size, the old left intact for earlier splits.
fn keep(a: Allocator, out: *std.ArrayList(u8), bytes: []const u8) Allocator.Error!void {
    if (out.items.len + bytes.len > out.capacity) {
        const cap = @max(256, 2 * (out.items.len + bytes.len));
        const mem = try a.alloc(u8, cap);
        @memcpy(mem[0..out.items.len], out.items);
        out.items = mem[0..out.items.len];
        out.capacity = cap;
    }
    out.appendSliceAssumeCapacity(bytes);
}

/// A message's new text since the last call, its end's whitespace held back (the finished reply strips it).
fn grow(a: Allocator, out: *std.ArrayList(u8), o: *Open, text: []const u8) Allocator.Error!void {
    const body = reply_text.pyRstrip(text);
    var piece = body[@min(o.sent, body.len)..];
    o.sent = @max(o.sent, body.len);
    if (!o.begun) {
        piece = reply_text.pyLstrip(piece);
        if (piece.len == 0) return;
        o.begun = true;
        if (out.items.len > 0) try keep(a, out, "\n\n");
    }
    try keep(a, out, piece);
}

/// How one chat's reply reads: the message its prompt left open, the request's tools and thinking switch.
pub const Reading = struct {
    /// The prompt after its last `<|start|>` when no end follows it (`assistant`, or `assistant to=self<|message|>`).
    lead: []const u8 = "",
    tools: []const Value = &.{},
    /// Thinking on: `self` messages are the reasoning; off, they stay out of the reply.
    reasoning: bool = true,
    /// What a streamed reply's text gave so far: each call reads only the bytes after the last (the reply grows).
    st: State = .{},

    const State = struct {
        ready: bool = false,
        at: usize = 0, // the reply read up to here: closed messages before it, or the open message's body start
        open: ?Open = null,
        any: bool = false, // a message seen
        first: bool = true, // the next header is the reply's first, which the prompt's lead begins
        reasoning: std.ArrayList(u8) = .empty,
        answer: std.ArrayList(u8) = .empty,
        len: usize = 0, // the reply's length at the last call, and its last bytes (the same reply grew since)
        tail: [32]u8 = undefined,
        tail_n: usize = 0,
    };

    pub fn of(a: Allocator, text: anytype, prompt: []const u32, tools: []const Value, thinking: bool) Allocator.Error!Reading {
        var r: Reading = .{ .tools = tools, .reasoning = thinking };
        const id = text.tokenId(start) orelse return r;
        const at = std.mem.lastIndexOfScalar(u32, prompt, id) orelse return r;
        const lead = try text.decode(a, prompt[at + 1 ..]);
        for (ends) |e| if (std.mem.indexOf(u8, lead, e) != null) return r;
        r.lead = lead;
        return r;
    }

    /// A fresh read: new lists (an earlier split's text stays as it was), the lead's open message if it has one.
    fn restart(r: *Reading, a: Allocator) Allocator.Error!void {
        r.st = .{ .ready = true };
        const m = findTag(r.lead, 0, message) orelse return;
        var header = r.lead[0..m];
        if (std.mem.startsWith(u8, header, start)) header = header[start.len..];
        r.st.open = .{ .kind = kindOf(header), .recipient = try a.dupe(u8, recipientOf(header) orelse ""), .b0 = 0, .scan = 0 };
        r.st.any = true;
        r.st.first = false;
    }

    /// Whether `reply` is the last call's reply grown (else the read starts over).
    fn grew(r: *const Reading, reply: []const u8) bool {
        const st = &r.st;
        if (!st.ready or reply.len < st.len) return false;
        return std.mem.eql(u8, reply[st.len - st.tail_n .. st.len], st.tail[0..st.tail_n]);
    }

    fn mark(r: *Reading, reply: []const u8) void {
        r.st.len = reply.len;
        r.st.tail_n = @min(reply.len, r.st.tail.len);
        @memcpy(r.st.tail[0..r.st.tail_n], reply[reply.len - r.st.tail_n ..]);
    }

    /// A reply with no header at its start (none in the prompt's lead): all of it is the answer, streamed as it comes.
    fn plain(r: *const Reading, reply: []const u8, end: End) ?[]const u8 {
        if (r.st.any) return null;
        const p = reply_text.pyLstrip(reply);
        if (std.mem.startsWith(u8, p, "<|") or std.mem.startsWith(u8, p, "to=")) return null;
        if (end == .streaming and (p.len == 0 or std.mem.startsWith(u8, "<|", p) or std.mem.startsWith(u8, "to=", p))) return "";
        return reply_text.pyStrip(p);
    }

    /// An open message's text into the split: reasoning and answer as they grow, a call once closed or at the end.
    fn body(r: *Reading, a: Allocator, o: *Open, text: []const u8, done: bool, as: End) Allocator.Error!void {
        switch (o.kind) {
            .reasoning => if (r.reasoning) try grow(a, &r.st.reasoning, o, text),
            .answer => try grow(a, &r.st.answer, o, text),
            .call => if (done and !o.begun) try grow(a, &r.st.answer, o, (try calls(a, text, o.recipient, r.tools, as)) orelse text),
        }
    }

    /// (reasoning, answer) so far; a stream holds back open headers, calls and partial markers, so each split grows.
    pub fn split(r: *Reading, a: Allocator, reply: []const u8, end: End) Allocator.Error!Split {
        if (!r.grew(reply)) try r.restart(a);
        defer r.mark(reply);
        if (r.plain(reply, end)) |p| return .{ .reasoning = "", .answer = p };
        const st = &r.st;
        while (true) {
            if (st.open == null) { // the next header, the lead before the reply's first
                const m = findTag(reply, st.at, message) orelse break;
                var header = reply[st.at..m];
                if (st.first) header = try std.mem.concat(a, u8, &.{ r.lead, header });
                if (std.mem.startsWith(u8, header, start)) header = header[start.len..];
                st.first = false;
                st.any = true;
                st.open = .{ .kind = kindOf(header), .recipient = try a.dupe(u8, recipientOf(header) orelse ""), .b0 = m + message.len, .scan = m + message.len };
                st.at = m + message.len;
            }
            const o = &st.open.?;
            var b1 = reply.len;
            var next = reply.len;
            var closed = false;
            for (closers) |e| if (findTag(reply[0..b1], o.scan, e)) |at| {
                b1 = at;
                next = if (std.mem.eql(u8, e, start)) at else at + e.len;
                closed = true;
            };
            if (closed) {
                try r.body(a, o, reply[o.b0..b1], true, .ended);
                st.open = null;
                st.at = next;
                continue;
            }
            o.scan = @max(o.b0, reply.len -| reach); // a closer begun before the end is found next call
            if (end == .streaming) {
                var held: usize = 0;
                for (closers) |e| held = @max(held, reply_text.partialTag(reply[o.b0..], e));
                try r.body(a, o, reply[o.b0 .. reply.len - held], false, end);
            } else try r.body(a, o, reply[o.b0..], true, end);
            break;
        }
        if (!st.any and end != .streaming) { // no message markup at all; a header the reply was cut in is no answer
            const p = reply_text.pyStrip(reply);
            return .{ .reasoning = "", .answer = if (std.mem.startsWith(u8, p, "to=")) "" else p };
        }
        return .{ .reasoning = st.reasoning.items, .answer = st.answer.items };
    }
};
