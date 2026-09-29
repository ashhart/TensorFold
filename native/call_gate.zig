const std = @import("std");
const Tokenizer = @import("vendor/tokenizer.zig").Tokenizer;
const Phase = enum { answer, lead, name, done };
const State = struct { armed: bool = true, phase: Phase = .answer, seen: []const u8 = "" };
const Step = struct { state: State, fix: []const u32 = &.{} };

pub const Gate = struct {
    a: std.mem.Allocator,
    tokenizer: *Tokenizer,
    opener: i32,
    think_open: i32,
    think_end: i32,
    lead: []const u8,
    tail: []const u8,
    names: []const []const u8,
    state: State,
    forced: []const u32 = &.{},

    pub fn init(a: std.mem.Allocator, tokenizer: *Tokenizer, prompt: []const i32, form: @import("chat.zig").CallForm, names: []const []const u8, gemma: bool) !Gate {
        const opener = try singleId(a, tokenizer, form.opener);
        if (opener < 0) return error.UnsupportedRequiredToolCall;
        const think_open = if (gemma) blk: {
            const ids = try tokenizer.encode(a, "<|channel>thought");
            break :blk if (ids.len > 0) @as(i32, @intCast(ids[0])) else -1;
        } else try singleId(a, tokenizer, "<think>");
        const think_end = try singleId(a, tokenizer, if (gemma) "<channel|>" else "</think>");
        var armed = true;
        if (think_open >= 0 and think_end >= 0) for (prompt) |id| {
            if (id == think_open) armed = false;
            if (id == think_end) armed = true;
        };
        return .{ .a = a, .tokenizer = tokenizer, .opener = opener, .think_open = think_open, .think_end = think_end, .lead = form.lead orelse "", .tail = form.tail, .names = if (form.lead != null) names else &.{}, .state = .{ .armed = armed } };
    }
    fn singleId(a: std.mem.Allocator, tokenizer: *Tokenizer, text: []const u8) !i32 {
        const ids = try tokenizer.encode(a, text);
        return if (ids.len == 1) @intCast(ids[0]) else -1;
    }
    fn step(g: *Gate, token: i32, eos: bool) !Step {
        var state = g.state;
        if (state.phase == .done) return .{ .state = state };
        const decoded = try g.tokenizer.decode(g.a, &.{@intCast(token)}, false);
        if (state.phase == .answer) {
            if (!state.armed) return .{ .state = .{ .armed = token == g.think_end } };
            if (token == g.think_open) return .{ .state = .{ .armed = false } };
            if (token == g.opener) return .{ .state = .{ .phase = .lead } };
            if (!eos and std.mem.trim(u8, decoded, " \r\n\t").len == 0) return .{ .state = state };
            const lead = if (g.names.len > 0 and g.lead.len > 0) try g.tokenizer.encode(g.a, g.lead) else &.{};
            const fix = try g.a.alloc(u32, lead.len + 1);
            fix[0] = @intCast(g.opener);
            @memcpy(fix[1..], lead);
            return .{ .state = state, .fix = fix };
        }
        if (g.names.len == 0) return .{ .state = .{ .phase = .done } };
        var written = try std.mem.concat(g.a, u8, &.{ state.seen, decoded });
        var owed: []const u8 = "";
        if (state.phase == .lead) {
            if (std.mem.startsWith(u8, g.lead, written)) return .{ .state = .{ .phase = if (written.len == g.lead.len) .name else .lead, .seen = if (written.len == g.lead.len) "" else written } };
            if (!std.mem.startsWith(u8, written, g.lead)) return .{ .state = state, .fix = try g.tokenizer.encode(g.a, g.lead[state.seen.len..]) };
            owed = g.lead[state.seen.len..];
            state.seen = "";
            written = written[g.lead.len..];
        }
        var end: usize = 0;
        while (end < written.len) : (end += 1) {
            const c = written[end];
            if (std.ascii.isWhitespace(c) or (if (g.tail.len > 0) c == g.tail[0] else std.mem.indexOfScalar(u8, "<>{}\"'(),", c) != null)) break;
        }
        for (g.names) |name| {
            if (end == written.len and std.mem.startsWith(u8, name, written)) return .{ .state = .{ .phase = .name, .seen = written } };
            if (end < written.len and std.mem.eql(u8, written[0..end], name)) return .{ .state = .{ .phase = .done } };
        }
        for (g.names) |name| if (std.mem.startsWith(u8, name, state.seen)) {
            const fix = try g.tokenizer.encode(g.a, try std.mem.concat(g.a, u8, &.{ owed, name[state.seen.len..], g.tail }));
            if (fix.len > 0) return .{ .state = g.state, .fix = fix };
            break;
        };
        return .{ .state = .{ .phase = .done } };
    }
    pub fn next(g: *Gate, proposed: i32, eos: bool) !i32 {
        if (g.forced.len == 0) {
            const result = try g.step(proposed, eos);
            if (result.fix.len == 0) {
                g.state = result.state;
                return proposed;
            }
            g.forced = result.fix;
        }
        const token: i32 = @intCast(g.forced[0]);
        g.forced = g.forced[1..];
        try g.observe(token);
        return token;
    }
    pub fn observe(g: *Gate, token: i32) !void {
        const result = try g.step(token, false);
        g.state = if (result.fix.len == 0) result.state else .{ .phase = .done };
    }
};
