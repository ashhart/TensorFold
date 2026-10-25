//! Replies written as messages to recipients: `<|start|>assistant to=R<|message|>body<|eom|>`, the last one `<|eot|>`.
const std = @import("std");
const reply_text = @import("reply_text.zig");
const Allocator = std.mem.Allocator;
const Split = reply_text.Split;
const findTag = reply_text.findTag;

const start = "<|start|>";
const message = "<|message|>";
const ends = [_][]const u8{ "<|eom|>", "<|eot|>" };
const closers = ends ++ [_][]const u8{start};

/// Whether replies take this form: every marker is one token and the template writes `to=self` messages.
pub fn detect(text: anytype) bool {
    for ([_][]const u8{ start, message, ends[0], ends[1] }) |m| if (text.tokenId(m) == null) return false;
    return std.mem.indexOf(u8, text.templateSource(), "to=self") != null;
}

/// `self` is reasoning; `user`, `commentary` and no recipient are the answer; any other recipient is a tool call.
const Kind = enum { reasoning, answer, call };

fn kindOf(header: []const u8) Kind {
    const at = std.mem.indexOf(u8, header, "to=") orelse return .answer;
    const rest = header[at + 3 ..];
    const r = rest[0 .. std.mem.indexOfAny(u8, rest, " \t\r\n") orelse rest.len];
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

/// Each `<atem:invoke>` as Qwen3-Coder's `<tool_call>` block, which tool_parse and tool_stream read; null: none.
pub fn calls(a: Allocator, body: []const u8) Allocator.Error!?[]const u8 {
    const invoke = "<atem:invoke name=\"";
    const parameter = "<atem:parameter name=\"";
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (findTag(body, pos, invoke)) |s| {
        const q = std.mem.indexOfPos(u8, body, s + invoke.len, "\">") orelse break;
        const close = std.mem.indexOfPos(u8, body, q, "</atem:invoke>") orelse body.len;
        if (out.items.len > 0) try out.append(a, '\n');
        try out.print(a, "<tool_call>\n<function={s}>\n", .{try unescape(a, body[s + invoke.len .. q])});
        var p = q + 2;
        while (findTag(body[0..close], p, parameter)) |ps| {
            const pq = std.mem.indexOfPos(u8, body[0..close], ps + parameter.len, "\">") orelse break;
            const pe = std.mem.indexOfPos(u8, body[0..close], pq + 2, "</atem:parameter>") orelse close;
            try out.print(a, "<parameter={s}>\n{s}\n</parameter>\n", .{ try unescape(a, body[ps + parameter.len .. pq]), body[pq + 2 .. pe] });
            p = pe;
        }
        try out.appendSlice(a, "</function>\n</tool_call>");
        pos = close;
    }
    return if (out.items.len > 0) out.items else null;
}

/// (reasoning, answer) so far; a stream holds back open headers, open calls and partial markers, so each split grows.
pub fn split(a: Allocator, full: []const u8, finished: bool) Allocator.Error!Split {
    var reasoning: std.ArrayList(u8) = .empty;
    var answer: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    var any = false;
    while (findTag(full, pos, message)) |m| { // the prompt ends at `<|start|>assistant`: the first header is ` to=R`
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
        if (!closed and !finished) {
            var held: usize = 0;
            for (closers) |e| held = @max(held, reply_text.partialTag(full[b0..b1], e));
            b1 -= held;
        }
        const body = full[b0..b1];
        switch (kindOf(header)) {
            .reasoning => try add(a, &reasoning, body),
            .answer => try add(a, &answer, body),
            .call => if (closed or finished) try add(a, &answer, (try calls(a, body)) orelse body),
        }
        if (!closed) break;
        pos = next;
    }
    if (!any and finished) return .{ .reasoning = "", .answer = reply_text.pyStrip(full) }; // no message markup at all
    return .{ .reasoning = reasoning.items, .answer = answer.items };
}
