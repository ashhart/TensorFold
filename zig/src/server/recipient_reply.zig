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

/// One message's text after the last, a blank line between; leading whitespace off, so a streamed piece only grows.
fn add(a: Allocator, out: *std.ArrayList(u8), body: []const u8) Allocator.Error!void {
    const piece = reply_text.pyLstrip(body);
    if (piece.len == 0) return;
    if (out.items.len > 0) try out.appendSlice(a, "\n\n");
    try out.appendSlice(a, piece);
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

/// How one chat's reply reads: the message its prompt left open, the request's tools and thinking switch.
pub const Reading = struct {
    /// The prompt after its last `<|start|>` when no end follows it (`assistant`, or `assistant to=self<|message|>`).
    lead: []const u8 = "",
    tools: []const Value = &.{},
    /// Thinking on: `self` messages are the reasoning; off, they stay out of the reply.
    reasoning: bool = true,

    pub fn of(a: Allocator, text: anytype, prompt: []const u32, tools: []const Value, thinking: bool) Allocator.Error!Reading {
        var r: Reading = .{ .tools = tools, .reasoning = thinking };
        const id = text.tokenId(start) orelse return r;
        const at = std.mem.lastIndexOfScalar(u32, prompt, id) orelse return r;
        const lead = try text.decode(a, prompt[at + 1 ..]);
        for (ends) |e| if (std.mem.indexOf(u8, lead, e) != null) return r;
        r.lead = lead;
        return r;
    }

    /// (reasoning, answer) so far; a stream holds back open headers, calls and partial markers, so each split grows.
    pub fn split(r: Reading, a: Allocator, reply: []const u8, end: End) Allocator.Error!Split {
        const full = if (r.lead.len > 0) try std.mem.concat(a, u8, &.{ r.lead, reply }) else reply;
        var reasoning: std.ArrayList(u8) = .empty;
        var answer: std.ArrayList(u8) = .empty;
        var pos: usize = 0;
        var any = false;
        while (findTag(full, pos, message)) |m| { // the reply goes on from the prompt's lead
            var header = full[pos..m];
            if (std.mem.startsWith(u8, header, start)) header = header[start.len..];
            const b0 = m + message.len;
            var b1 = full.len;
            var next = full.len;
            var closed = false;
            for (closers) |e| if (findTag(full[0..b1], b0, e)) |at| {
                b1 = at;
                next = if (std.mem.eql(u8, e, start)) at else at + e.len;
                closed = true;
            };
            any = true;
            if (!closed and end == .streaming) {
                var held: usize = 0;
                for (closers) |e| held = @max(held, reply_text.partialTag(full[b0..b1], e));
                b1 -= held;
            }
            const body = full[b0..b1];
            switch (kindOf(header)) {
                .reasoning => if (r.reasoning) try add(a, &reasoning, body),
                .answer => try add(a, &answer, body),
                .call => if (closed or end != .streaming) {
                    const as = if (closed) .ended else end;
                    try add(a, &answer, (try calls(a, body, recipientOf(header).?, r.tools, as)) orelse body);
                },
            }
            if (!closed) break;
            pos = next;
        }
        if (!any and end != .streaming) { // no message markup at all; a header the reply was cut in is no answer
            const plain = reply_text.pyStrip(reply);
            return .{ .reasoning = "", .answer = if (std.mem.startsWith(u8, plain, "to=")) "" else plain };
        }
        return .{ .reasoning = reasoning.items, .answer = answer.items };
    }
};
