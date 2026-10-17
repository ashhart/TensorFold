//! DeepSeek-V4.1 replies streamed: reasoning, content and DSML tool calls as deltas over the growing text.
const std = @import("std");
const Allocator = std.mem.Allocator;
const tool_scan = @import("tool_scan.zig");
const tool_types = @import("tool_types.zig");
const tools_mod = @import("tools.zig");
const THINK_END = tool_scan.THINK_END;
const Match = tool_scan.Match;
const partial = tool_scan.partial;
const hold = tool_scan.hold;
const blockTag = tool_scan.blockTag;
const tag_window = tool_scan.tag_window;
const Invoke = tool_scan.Invoke;
const invokeAt = tool_scan.invokeAt;
const searchHeads = tool_scan.searchHeads;
const invokeEnd = tool_scan.invokeEnd;
const ParamHead = tool_scan.ParamHead;
const paramHeadAt = tool_scan.paramHeadAt;
const paramClose = tool_scan.paramClose;
const paramValue = tool_types.paramValue;
const fragment = tool_types.fragment;
const Call = tools_mod.Call;
const Tools = tools_mod.Tools;
const IdSource = tools_mod.IdSource;
const split = tools_mod.split;

pub const Delta = union(enum) {
    reasoning: []const u8,
    content: []const u8,
    /// A call opens: its index among the reply's calls, id and name (arguments "").
    call: struct { index: usize, id: []const u8, name: []const u8 },
    /// More of a call's arguments.
    arguments: struct { index: usize, text: []const u8 },
};

/// A search for a fixed-length tag at or after ``from``, resumed where it stopped as the text grows.
const Finder = struct {
    from: usize = 0,
    scanned: usize = 0,
    found: ?Match = null,

    fn reset(f: *Finder, from: usize) void {
        f.* = .{ .from = from, .scanned = from };
    }

    fn block(f: *Finder, text: []const u8, close: bool) ?Match {
        if (f.found) |m| return m;
        const start = @max(f.from, f.scanned -| tag_window);
        f.found = blockTag(text, start, text.len, close);
        f.scanned = text.len;
        return f.found;
    }
};

/// The invoke being read: its header, the searches for its end and the next invoke, and its parameters so far.
const Open = struct {
    m: Invoke,
    end: Finder,
    nxt: Finder,
    end_m: ?Match = null,
    nxt_m: ?Invoke = null,
    param_pos: usize,
    call: Call,
    seen: std.StringHashMapUnmanaged(void) = .empty,
    pending: ?ParamHead = null, // a parameter whose closing tag has not come yet
    close_scan: usize = 0,
};

pub const Stream = struct {
    a: Allocator,
    thinking: bool,
    tools: Tools,
    ids: IdSource,
    sent_reasoning: usize = 0,
    sent_content: usize = 0,
    in_calls: bool = false,
    /// The reply's usable calls, as sent.
    calls: std.ArrayList(Call) = .empty,
    sent_args: std.ArrayList(usize) = .empty,
    sent_close: std.ArrayList(bool) = .empty,
    // incremental state over the growing text
    think_end: ?usize = null,
    think_scan: usize = 0,
    text_blk: Finder = .{},
    rest_at: ?usize = null,
    rest_blk: Finder = .{},
    close_blk: Finder = .{},
    body_at: usize = 0,
    done: std.ArrayList(Call) = .empty, // every invoke read to its fixed end, usable or not
    settled: usize = 0, // done calls before this one are sent whole (or unusable)
    settled_known: usize = 0, // the usable calls among them
    resume_at: usize = 0,
    open: ?Open = null,
    scratch: std.ArrayList(u8) = .empty,

    pub fn init(a: Allocator, thinking: bool, tools: Tools, ids: IdSource) Stream {
        return .{ .a = a, .thinking = thinking, .tools = tools, .ids = ids };
    }

    fn findThinkEnd(s: *Stream, text: []const u8) ?usize {
        if (s.think_end) |e| return e;
        const from = s.think_scan -| (THINK_END.len - 1);
        s.think_end = std.mem.indexOfPos(u8, text, from, THINK_END);
        s.think_scan = text.len;
        return s.think_end;
    }

    /// The deltas ``text`` (the whole reply so far) adds; ``finished`` releases held text and closes an open call.
    pub fn feed(s: *Stream, text: []const u8, finished: bool, out: *std.ArrayList(Delta)) Allocator.Error!void {
        const a = s.a;
        const offered = s.tools.offered();
        var end: ?usize = null;
        var blk_all: ?Match = null;
        if (s.thinking) {
            end = s.findThinkEnd(text);
            blk_all = s.text_blk.block(text, false);
            if (end == null and (blk_all == null or !offered)) {
                var h = partial(text, THINK_END);
                if (offered) h = @max(h, hold(text));
                const upto = if (finished) text.len else text.len - h;
                if (upto > s.sent_reasoning) {
                    try out.append(a, .{ .reasoning = text[s.sent_reasoning..upto] });
                    s.sent_reasoning = upto;
                }
                return;
            }
        }
        const sp = split(text, s.thinking, end, blk_all);
        if (sp.reasoning.len > s.sent_reasoning) {
            try out.append(a, .{ .reasoning = sp.reasoning[s.sent_reasoning..] });
            s.sent_reasoning = sp.reasoning.len;
        }
        if (s.rest_at == null or s.rest_at.? != sp.rest_at) {
            s.rest_at = sp.rest_at;
            s.rest_blk.reset(sp.rest_at);
        }
        const rest = text[sp.rest_at..];
        const blk = if (offered) s.rest_blk.block(text, false) else null;
        var limit: usize = undefined;
        if (blk) |b| {
            limit = b.start - sp.rest_at;
            if (std.mem.endsWith(u8, rest[0..limit], "\n\n")) limit -= 2;
        } else if (finished) {
            limit = rest.len;
        } else limit = rest.len - (if (offered) hold(rest) else 0);
        if (limit > s.sent_content and !s.in_calls) {
            try out.append(a, .{ .content = rest[s.sent_content..limit] });
            s.sent_content = limit;
        }
        const b = blk orelse return;
        if (!s.in_calls) {
            s.in_calls = true;
            s.body_at = b.end;
            s.resume_at = b.end;
            s.close_blk.reset(b.end);
        }
        const close = s.close_blk.block(text, true);
        const to = if (close) |c| c.start else text.len;
        try s.readCalls(text, to);
        // calls in order (finished, then open); finished ones already sent are skipped: a token costs open calls only
        while (s.settled < s.done.items.len) : (s.settled += 1) {
            const c = &s.done.items[s.settled];
            const name = s.tools.resolve(c.name) orelse continue;
            if (s.settled_known == s.calls.items.len or !s.sent_close.items[s.settled_known]) break;
            c.name = name;
            s.settled_known += 1;
        }
        var k: usize = s.settled_known;
        const total = s.done.items.len + @intFromBool(s.open != null);
        for (s.settled..total) |j| {
            const c: *Call = if (j < s.done.items.len) &s.done.items[j] else &s.open.?.call;
            c.name = s.tools.resolve(c.name) orelse continue;
            if (k == s.calls.items.len) {
                try s.calls.append(a, .{ .id = c.id, .name = c.name });
                try s.sent_args.append(a, 0);
                try s.sent_close.append(a, false);
                try out.append(a, .{ .call = .{ .index = k, .id = c.id, .name = c.name } });
            }
            const mine = &s.calls.items[k];
            mine.args = c.args;
            s.scratch = .empty;
            for (c.args.items[s.sent_args.items[k]..], s.sent_args.items[k]..) |arg, n| try fragment(a, &s.scratch, n == 0, arg.key, arg.value);
            s.sent_args.items[k] = c.args.items.len;
            if ((c.closed or finished) and !s.sent_close.items[k]) {
                try s.scratch.appendSlice(a, if (c.args.items.len > 0) "}" else "{}");
                s.sent_close.items[k] = true;
                mine.closed = true;
            }
            if (s.scratch.items.len > 0) try out.append(a, .{ .arguments = .{ .index = k, .text = s.scratch.items } });
            k += 1;
        }
    }

    /// The calls body ``text[body_at..to]``, resumed: invokes whose end is fixed are read once.
    fn readCalls(s: *Stream, text: []const u8, to: usize) Allocator.Error!void {
        const a = s.a;
        while (true) {
            if (s.open == null) {
                const found = searchHeads(Invoke, text[0..to], @min(s.resume_at, to), invokeAt);
                const m = found.hit orelse {
                    s.resume_at = @max(s.resume_at, found.again); // a header still arriving is read again
                    return;
                };
                // names are copied: ``text`` is the caller's growing buffer, gone by a later feed
                var o: Open = .{ .m = m, .end = .{}, .nxt = .{}, .param_pos = m.m.end, .call = .{ .id = try s.ids.make(s.ids.ctx, a, s.done.items.len), .name = try a.dupe(u8, m.name) } };
                o.end.reset(m.m.end);
                o.nxt.reset(m.m.end);
                s.open = o;
            }
            const o = &s.open.?;
            if (o.end_m == null) {
                const start = @max(o.end.from, o.end.scanned -| tag_window);
                o.end_m = invokeEnd(text, start, to);
                o.end.scanned = to;
            }
            if (o.nxt_m == null) {
                const found = searchHeads(Invoke, text[0..to], @min(@max(o.nxt.from, o.nxt.scanned), to), invokeAt);
                o.nxt_m = found.hit;
                o.nxt.scanned = found.again;
            }
            const end = o.end_m;
            const nxt = o.nxt_m;
            const stop = if (end != null and (nxt == null or end.?.start < nxt.?.m.start)) end.?.start else if (nxt) |n| n.m.start else to;
            try s.readParams(text, stop);
            o.call.closed = end != null and stop == end.?.start;
            if (o.call.closed or nxt != null) { // its end is fixed: read for good
                s.resume_at = if (o.call.closed) end.?.end else stop;
                try s.done.append(a, o.call);
                s.open = null;
                continue;
            }
            return;
        }
    }

    /// The open invoke's parameters up to ``stop``; a parameter whose closing tag is still missing is resumed.
    fn readParams(s: *Stream, text: []const u8, stop: usize) Allocator.Error!void {
        const a = s.a;
        const o = &s.open.?;
        const t = text[0..stop];
        while (true) {
            if (o.pending == null) {
                const found = searchHeads(ParamHead, t, @min(o.param_pos, t.len), paramHeadAt);
                o.pending = found.hit orelse {
                    o.param_pos = @max(o.param_pos, found.again);
                    return;
                };
                // its closing tag may come in a later feed, from a reallocated ``text``: the key is copied now
                o.pending.?.name = try a.dupe(u8, o.pending.?.name);
                o.close_scan = o.pending.?.end;
            }
            const h = o.pending.?;
            const c = paramClose(t, @max(h.end, o.close_scan -| tag_window)) orelse {
                o.close_scan = stop;
                return;
            };
            o.pending = null;
            o.param_pos = c.end;
            if (o.seen.contains(h.name)) continue;
            try o.seen.put(a, h.name, {});
            // the value is copied too: the kept call is serialized again at the reply's end, after ``text`` is gone
            try o.call.args.append(a, .{ .key = h.name, .value = try paramValue(a, try a.dupe(u8, t[h.end..c.start]), h.string, s.tools.prop(o.call.name, h.name)) });
        }
    }

    pub fn finish(s: *Stream, text: []const u8, out: *std.ArrayList(Delta)) Allocator.Error!void {
        return s.feed(text, true, out);
    }
};
