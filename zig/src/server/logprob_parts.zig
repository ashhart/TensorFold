//! Target rows projected onto a reply's prose, reasoning and tool envelopes: whole, or as a stream settles them.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const rt = @import("reply_text.zig");
const lp = @import("logprobs.zig");
const Options = @import("kept_probs.zig").Opts;
const tp = @import("tool_parse.zig");
const mm = @import("tool_minimax.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;
const A = std.mem.Allocator;
const V = json.Value;
const Error = error{ Failed, OutOfMemory };

/// Entries a stream already sent with its text, by channel.
pub const Sent = struct { content: usize = 0, reasoning: usize = 0 };

/// A finished reply's tokens and rows, projected once its text and calls are final; `closer` ends a call EOS cut.
pub const Capture = struct { tokens: []const u32, rows: []const api.LogprobRow, thinking: bool, stops: rt.Stops, closer: []const u8 = "", sent: Sent = .{}, opts: Options = .{} };

const Channel = enum { content, reasoning, tool, dropped };
/// Visible bytes of one channel; an open range may still grow while the reply streams.
const Range = struct { start: usize, end: usize, channel: Channel, index: usize = 0, open: bool = false };
/// One entry: token `token` shows bytes [lo, hi); settled once later text cannot change it.
const Piece = struct { token: usize, lo: usize, hi: usize, settled: bool };
/// What the reply shows: its prose, reasoning and, for a whole reply, the calls the tool policy kept.
const Shown = struct { content: []const u8, reasoning: ?[]const u8, tools: []const V = &.{}, calls: ?usize = null };

/// Each token's exact bytes, joined, and where each token ends.
const Spelled = struct {
    raw: std.ArrayList(u8) = .empty,
    ends: std.ArrayList(usize) = .empty,

    fn add(s: *Spelled, srv: *Server, a: A, tokens: []const u32) Error!void {
        for (tokens) |id| {
            try s.raw.appendSlice(a, (try srv.text.tokenBytes(a, id)) orelse return error.Failed);
            try s.ends.append(a, s.raw.items.len);
        }
    }

    fn from(s: *const Spelled, i: usize) usize {
        return if (i == 0) 0 else s.ends.items[i - 1];
    }
};

/// A streamed reply's rows and bytes, and the entries its chunks already carried.
pub const Live = struct {
    rows: std.ArrayList(api.LogprobRow) = .empty, // copied with each round's tokens
    spelled: Spelled = .{},
    sent: Sent = .{},
    opts: Options = .{},

    pub fn add(l: *Live, srv: *Server, a: A, tokens: []const u32) Error!void {
        return l.spelled.add(srv, a, tokens);
    }

    /// ``{content, reasoning_content}`` entries the streamed text now settles; null when there are none.
    pub fn settle(l: *Live, srv: *Server, a: A, tokens: []const u32, thinking: bool, stops: rt.Stops, content: []const u8, reasoning: []const u8) Error!?V {
        const visible = stops.visible(l.spelled.raw.items, false);
        const ranges = try layout(srv, a, visible, tokens, thinking, false, .{ .content = content, .reasoning = reasoning });
        const prose = try fresh(srv, a, &l.spelled, tokens, l.rows.items, ranges, .content, &l.sent.content, l.opts);
        const thought = try fresh(srv, a, &l.spelled, tokens, l.rows.items, ranges, .reasoning, &l.sent.reasoning, l.opts);
        if (prose.len == 0 and thought.len == 0) return null;
        const out = try json.newObject(a);
        try out.put(a, "content", .{ .array = prose });
        if (thought.len > 0) try out.put(a, "reasoning_content", .{ .array = thought });
        return .{ .object = out };
    }
};

/// The newly settled entries of one channel past `sent`, which grows by their count.
fn fresh(srv: *Server, a: A, s: *const Spelled, tokens: []const u32, rows: []const api.LogprobRow, ranges: []const Range, channel: Channel, sent: *usize, opts: Options) Error![]V {
    const all = try pieces(a, s, ranges, channel, 0);
    var end = sent.*;
    while (end < all.len and all[end].settled) end += 1;
    if (end <= sent.*) return &.{};
    const out = try values(srv, a, s, tokens, rows, all[sent.*..end], opts);
    sent.* = end;
    return out;
}

/// ``choices[0].logprobs``: prose in ``content``, ``reasoning_content``, and one ``{index, content}`` group a call.
pub fn project(srv: *Server, cx: *Cx, c: Capture, content: []const u8, reasoning: ?[]const u8, tools: []const V, calls: ?[]V) errors.Refused!V {
    const a = cx.a;
    if (c.rows.len < c.tokens.len) return cx.other("the engine gave fewer logprob rows than reply tokens");
    return whole(srv, a, c, .{ .content = content, .reasoning = reasoning, .tools = tools, .calls = if (calls) |list| list.len else null }) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Failed => cx.other("the logprob rows are out of step with the reply's tokens, or a token has no exact bytes"),
    };
}

fn whole(srv: *Server, a: A, c: Capture, shown: Shown) Error!V {
    var s: Spelled = .{};
    try s.add(srv, a, c.tokens);
    const visible = c.stops.visible(s.raw.items, false);
    // the reply's text splits with the closer it was given, as chat_generation's finish splits it
    const text = if (c.closer.len > 0) try std.mem.concat(a, u8, &.{ visible, c.closer }) else visible;
    const ranges = try layout(srv, a, text, c.tokens, c.thinking, true, shown);
    const out = try json.newObject(a);
    try out.put(a, "content", .{ .array = try values(srv, a, &s, c.tokens, c.rows, try pieces(a, &s, ranges, .content, 0), c.opts) });
    if (shown.reasoning != null or c.thinking) try out.put(a, "reasoning_content", .{ .array = try values(srv, a, &s, c.tokens, c.rows, try pieces(a, &s, ranges, .reasoning, 0), c.opts) });
    const count = shown.calls orelse return .{ .object = out };
    const groups = try a.alloc(V, count);
    for (groups, 0..) |*slot, i| {
        const o = try json.newObject(a);
        try o.put(a, "index", try json.intValue(a, i));
        try o.put(a, "content", .{ .array = try values(srv, a, &s, c.tokens, c.rows, try pieces(a, &s, ranges, .tool, i), c.opts) });
        slot.* = .{ .object = o };
    }
    try out.put(a, "tool_calls", .{ .array = groups });
    return .{ .object = out };
}

/// A streamed reply's last logprobs: the whole reply's entries past those its chunks carried; null when none are left.
pub fn rest(a: A, rows: V, sent: Sent) A.Error!?V {
    const out = try json.newObject(a);
    var left = false;
    for ([_]struct { []const u8, usize }{ .{ "content", sent.content }, .{ "reasoning_content", sent.reasoning } }) |key| {
        const list = (rows.get(key[0]) orelse continue).items() orelse continue;
        const tail = list[@min(key[1], list.len)..];
        left = left or tail.len > 0;
        if (tail.len > 0 or std.mem.eql(u8, key[0], "content")) try out.put(a, key[0], .{ .array = tail });
    }
    if (rows.get("tool_calls")) |groups| {
        try out.put(a, "tool_calls", groups);
        left = true;
    }
    return if (left) .{ .object = out } else null;
}

fn values(srv: *Server, a: A, s: *const Spelled, tokens: []const u32, rows: []const api.LogprobRow, list: []const Piece, opts: Options) Error![]V {
    const out = try a.alloc(V, list.len);
    for (list, out) |p, *slot| {
        const cropped = p.lo != s.from(p.token) or p.hi != s.ends.items[p.token];
        slot.* = try lp.rowValue(srv, a, tokens[p.token], rows[p.token], s.raw.items[p.lo..p.hi], cropped, opts);
    }
    return out;
}

/// One channel's entries in token order: each token's overlap with each of the channel's ranges.
fn pieces(a: A, s: *const Spelled, ranges: []const Range, channel: Channel, index: usize) A.Error![]Piece {
    var out: std.ArrayList(Piece) = .empty;
    var i: usize = 0;
    for (ranges) |r| {
        if (r.channel != channel or r.index != index) continue;
        while (i < s.ends.items.len and s.ends.items[i] <= r.start) i += 1;
        var j = i;
        while (j < s.ends.items.len and s.from(j) < r.end) : (j += 1) {
            const lo = @max(s.from(j), r.start);
            const hi = @min(s.ends.items[j], r.end);
            if (lo < hi) try out.append(a, .{ .token = j, .lo = lo, .hi = hi, .settled = !r.open or hi == s.ends.items[j] });
        }
    }
    return out.items;
}

/// Where `part` lies in `whole` when it is a slice of it.
fn offset(all: []const u8, part: []const u8) ?usize {
    const at = @intFromPtr(part.ptr);
    const base = @intFromPtr(all.ptr);
    return if (part.len > 0 and at >= base and at + part.len <= base + all.len) at - base else null;
}

/// The ranges of each channel in the visible text; a reply still streaming (`finished` false) leaves growing ones open.
fn layout(srv: *Server, a: A, visible: []const u8, tokens: []const u32, thinking: bool, finished: bool, shown: Shown) A.Error![]Range {
    var ranges: std.ArrayList(Range) = .empty;
    var answer_at: usize = 0;
    var prose_end: usize = visible.len; // stray prose before a think block ends where the block starts
    var resume_at: usize = visible.len;
    var reasoning_from: usize = 0;
    var closed = true;
    if (thinking) {
        const m = srv.markers;
        const close = try rt.closeWhere(a, rt.decodeText, srv.text, tokens, srv.close_id);
        const split = try rt.splitThinkingAt(a, visible, finished, m, close);
        const seen = switch (close) {
            .token => true,
            .spelled => rt.findTag(visible, 0, m.close) != null,
            .pending => false,
        };
        closed = finished or (seen and split.answer.len > 0);
        reasoning_from = offset(visible, split.reasoning) orelse 0;
        answer_at = offset(visible, split.answer) orelse visible.len;
        if (split.answer.len > 0 and answer_at == visible.len) {
            // the answer joins prose written before the block to the text after its close
            const open_at = rt.findTag(visible, 0, m.open) orelse 0;
            const stray = open_at - rt.partialTag(visible[0..open_at], m.open);
            answer_at = 0;
            prose_end = stray;
            resume_at = visible.len -| (split.answer.len -| stray);
        }
    }
    if (shown.reasoning) |r| if (r.len > 0) {
        const at = std.mem.indexOfPos(u8, visible, reasoning_from, r) orelse visible.len;
        if (at < visible.len) try ranges.append(a, .{ .start = at, .end = at + r.len, .channel = .reasoning, .open = !closed });
    };
    const calls_at = if (prose_end < visible.len) resume_at else answer_at;
    if (shown.calls) |count| if (shown.tools.len > 0) try toolRanges(a, visible[calls_at..], shown.tools, count, calls_at, &ranges);
    const held = ranges.items.len;
    var cursor = answer_at;
    var wanted: usize = 0;
    while (wanted < shown.content.len and cursor < visible.len) {
        if (cursor == prose_end) cursor = @max(cursor, resume_at);
        if (cursor >= visible.len) break;
        if (heldBy(ranges.items[0..held], cursor)) |end| {
            cursor = end;
            continue;
        }
        if (visible[cursor] != shown.content[wanted]) {
            cursor += 1;
            continue;
        }
        const start = cursor;
        while (wanted < shown.content.len and cursor < visible.len and cursor != prose_end and visible[cursor] == shown.content[wanted] and heldBy(ranges.items[0..held], cursor) == null) {
            wanted += 1;
            cursor += 1;
        }
        try ranges.append(a, .{ .start = start, .end = cursor, .channel = .content });
    }
    if (!finished and ranges.items.len > held) ranges.items[ranges.items.len - 1].open = true;
    return ranges.items;
}

/// The end of the reasoning, call or dropped range holding `at`, which prose matching skips.
fn heldBy(ranges: []const Range, at: usize) ?usize {
    for (ranges) |r| if (at >= r.start and at < r.end) return r.end;
    return null;
}

/// Ranges of the calls `answer` makes: the first `count` are the reply's calls, those past the policy are dropped.
fn toolRanges(a: A, answer: []const u8, tools: []const V, count: usize, offset_at: usize, out: *std.ArrayList(Range)) A.Error!void {
    var envelopes: std.ArrayList(tp.Envelope) = .empty;
    try envelopes.appendSlice(a, try tp.envelopes(a, answer));
    const lfm = rt.calls[3];
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, answer, pos, lfm[0])) |at| {
        const end = if (std.mem.indexOfPos(u8, answer, at + lfm[0].len, lfm[1])) |e| e + lfm[1].len else answer.len;
        try envelopes.append(a, .{ .start = at, .end = end, .payload = answer[at..end] });
        pos = end;
    }
    if (tp.openCall(answer)) |at| try envelopes.append(a, .{ .start = at, .end = answer.len, .payload = answer[at..] });
    std.mem.sort(tp.Envelope, envelopes.items, {}, struct {
        fn less(_: void, x: tp.Envelope, y: tp.Envelope) bool {
            return x.start < y.start;
        }
    }.less);
    var index: usize = 0;
    var consumed: usize = 0;
    for (envelopes.items) |env| {
        if (env.start < consumed) continue;
        consumed = env.end;
        var start = env.start;
        var end = env.end;
        if (start >= mm.ns.len and std.mem.endsWith(u8, answer[0..start], mm.ns)) start -= mm.ns.len;
        if (std.mem.indexOfPos(u8, answer, start, mm.close)) |at| if (at < end) {
            end = at + mm.close.len;
        };
        const normalized = try mm.dsml(a, answer[start..end], tools);
        const text = try @import("tool_lfm2.zig").dsml(a, normalized, tools, true);
        const closed = try tp.closeCall(a, text, tools);
        const parsed = try tp.parse(a, try std.mem.concat(a, u8, &.{ text, closed }), tools, null);
        const found = parsed.calls orelse (try @import("tool_stream.zig").cutCalls(a, text, tools)) orelse continue;
        // A DSML envelope can hold several invokes: each call's range starts at its own opening tag.
        var at = start;
        for (found, 0..) |_, j| {
            var next = end;
            if (j + 1 < found.len) {
                const needle = if (std.mem.startsWith(u8, answer[start..], mm.open)) mm.ns ++ "<invoke" else if (std.mem.indexOf(u8, answer[start..end], "<｜DSML｜ invoke") != null) "<｜DSML｜ invoke" else "<｜DSML｜invoke";
                if (std.mem.indexOfPos(u8, answer[0..end], at, needle)) |first| {
                    if (std.mem.indexOfPos(u8, answer[0..end], first + needle.len, needle)) |n| next = n;
                }
            }
            const kept = index < count;
            try out.append(a, .{ .start = offset_at + at, .end = offset_at + next, .channel = if (kept) .tool else .dropped, .index = if (kept) index else 0 });
            at = next;
            index += 1;
        }
    }
    if (index == 0 and count > 0) {
        const normalized = try @import("tool_lfm2.zig").dsml(a, answer, tools, true);
        if ((try tp.parse(a, normalized, tools, null)).calls != null) try out.append(a, .{ .start = offset_at, .end = offset_at + answer.len, .channel = .tool });
    }
}
