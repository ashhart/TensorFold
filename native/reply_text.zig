const std = @import("std");

pub const Markers = struct { open: []const u8 = "", close: []const u8 = "</think>" };
pub const gemma_markers = Markers{ .open = "<|channel>thought", .close = "<channel|>" };
const calls = [_]Markers{
    .{ .open = "<tool_call>", .close = "</tool_call>" },
    .{ .open = "<|tool_call>", .close = "<tool_call|>" },
    .{ .open = "<｜DSML｜tool_calls>", .close = "</｜DSML｜tool_calls>" },
};
pub const Parts = struct { reasoning: []const u8 = "", content: []const u8 = "" };

pub fn partialTag(text: []const u8, tag: []const u8) usize {
    if (tag.len == 0) return 0;
    var size = @min(text.len, tag.len - 1);
    while (size > 0) : (size -= 1) if (std.mem.endsWith(u8, text, tag[0..size])) return size;
    return 0;
}

pub fn splitThinking(input: []const u8, finished: bool, markers: Markers) Parts {
    var text = input;
    if (markers.open.len > 0) {
        if (!std.mem.startsWith(u8, text, markers.open)) return if (!finished and std.mem.startsWith(u8, markers.open, text)) .{} else .{ .content = text };
        text = std.mem.trimStart(u8, text[markers.open.len..], "\n");
    }
    if (std.mem.indexOf(u8, text, markers.close)) |end| return .{ .reasoning = text[0..end], .content = std.mem.trimStart(u8, text[end + markers.close.len ..], "\n") };
    var call_at: usize = text.len;
    for (calls) |call| if (std.mem.indexOf(u8, text, call.open)) |at| {
        call_at = @min(call_at, at);
    };
    if (call_at < text.len) {
        if (!finished) return .{ .reasoning = text[0..call_at] };
        for (calls) |call| if (std.mem.startsWith(u8, text[call_at..], call.open) and std.mem.indexOf(u8, text[call_at..], call.close) != null) return .{ .reasoning = text[0..call_at], .content = text[call_at..] };
        return .{ .reasoning = text };
    }
    var held = if (finished) 0 else partialTag(text, markers.close);
    if (!finished) for (calls) |call| {
        held = @max(held, partialTag(text, call.open));
    };
    return .{ .reasoning = text[0 .. text.len - held] };
}

pub fn hideToolCalls(a: std.mem.Allocator, text: []const u8, finished: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var position: usize = 0;
    while (position < text.len) {
        var start = text.len;
        var selected: ?Markers = null;
        for (calls) |call| if (std.mem.indexOfPos(u8, text, position, call.open)) |at| {
            if (at < start) {
                start = at;
                selected = call;
            }
        };
        if (selected) |call| {
            try out.appendSlice(a, text[position..start]);
            const end = std.mem.indexOfPos(u8, text, start + call.open.len, call.close) orelse break;
            position = end + call.close.len;
        } else {
            var held: usize = 0;
            if (!finished) for (calls) |call| {
                held = @max(held, partialTag(text[position..], call.open));
            };
            try out.appendSlice(a, text[position .. text.len - held]);
            break;
        }
    }
    return out.toOwnedSlice(a);
}

pub fn stopAt(text: []const u8, stops: []const []const u8) ?usize {
    var found: ?usize = null;
    for (stops) |stop| if (stop.len > 0) if (std.mem.indexOf(u8, text, stop)) |at| {
        found = @min(found orelse text.len, at);
    };
    return found;
}

pub fn visible(text: []const u8, stops: []const []const u8, partial: bool) []const u8 {
    if (stopAt(text, stops)) |at| return text[0..at];
    var held: usize = 0;
    if (partial) for (stops) |stop| {
        held = @max(held, partialTag(text, stop));
    };
    return text[0 .. text.len - held];
}

pub fn harmony(text: []const u8) Parts {
    if (std.mem.indexOf(u8, text, "<|channel|>") == null) return .{ .content = text };
    const final = "<|channel|>final<|message|>";
    const analysis = "<|channel|>analysis<|message|>";
    const ends: []const []const u8 = &.{ "<|return|>", "<|end|>", "<|call|>", "<|start|>" };
    var result = Parts{};
    if (std.mem.indexOf(u8, text, analysis)) |at| {
        result.reasoning = visible(text[at + analysis.len ..], ends, false);
        result.reasoning = visible(result.reasoning, &.{final}, false);
    }
    if (std.mem.indexOf(u8, text, final)) |at| result.content = visible(text[at + final.len ..], ends, false);
    return result;
}

test "thinking and tool markers remain hidden at every byte boundary" {
    const expect = std.testing.expectEqualStrings;
    for ([_]Markers{ .{}, gemma_markers }) |markers| {
        const full = try std.fmt.allocPrint(std.testing.allocator, "{s}\nreason{s}\n\nanswer", .{ markers.open, markers.close });
        defer std.testing.allocator.free(full);
        const complete = splitThinking(full, true, markers);
        try expect(if (markers.open.len == 0) "\nreason" else "reason", complete.reasoning);
        try expect("answer", complete.content);
        for (0..full.len + 1) |end| {
            const part = splitThinking(full[0..end], false, markers);
            try std.testing.expect(std.mem.startsWith(u8, complete.reasoning, part.reasoning));
            try std.testing.expect(std.mem.startsWith(u8, complete.content, part.content));
        }
    }
    for (calls) |call| {
        const full = try std.fmt.allocPrint(std.testing.allocator, "before{s}payload{s}after", .{ call.open, call.close });
        defer std.testing.allocator.free(full);
        for (0..full.len + 1) |end| {
            const part = try hideToolCalls(std.testing.allocator, full[0..end], false);
            defer std.testing.allocator.free(part);
            try std.testing.expect(std.mem.startsWith(u8, "beforeafter", part));
        }
        const shown = try hideToolCalls(std.testing.allocator, full, true);
        defer std.testing.allocator.free(shown);
        try expect("beforeafter", shown);
    }
}

test "stop strings with overlapping UTF-8 prefixes never leak partial matches" {
    const stops: []const []const u8 = &.{ "END", "ENDING", "æøå" };
    for ([_][]const u8{ "beforeENDINGtail", "beforeæøåtail" }) |full| for (0..full.len + 1) |end| {
        const shown = visible(full[0..end], stops, true);
        try std.testing.expect(std.mem.startsWith(u8, "before", shown));
    };
    try std.testing.expectEqualStrings("beforeEN", visible("beforeEN", stops, false));
    try std.testing.expectEqualStrings("before", visible("beforeENDtail", stops, false));
    const parts = harmony("<|channel|>analysis<|message|>reason<|channel|>final<|message|>answer<|return|>suffix");
    try std.testing.expectEqualStrings("reason", parts.reasoning);
    try std.testing.expectEqualStrings("answer", parts.content);
    try std.testing.expectEqualStrings("", harmony("<|channel|>analysis<|message|>reason").content);
}
