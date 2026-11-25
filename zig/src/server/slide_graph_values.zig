//! The fact graph's JSON values and small word helpers, for slide_graph.zig.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const graph = @import("slide_graph.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Fact = graph.Fact;
const Answer = graph.Answer;
const Save = graph.Save;
const common = [_][]const u8{ "about", "after", "again", "being", "could", "every", "other", "should", "still", "their", "there", "these", "those", "under", "until", "where", "which", "while", "would" };

pub fn now(io: Io) i64 {
    return @intCast(@divTrunc(std.Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_ms));
}

pub fn int(comptime T: type, v: ?json.Value) !T {
    const x = v orelse return error.BadGraph;
    if (x != .int) return error.BadGraph;
    return std.fmt.parseInt(T, x.int, 10);
}

/// A loss as JSON, to four places (never NaN or infinite, which JSON cannot carry).
pub fn round4(x: f32) json.Value {
    const v: f64 = if (std.math.isFinite(x)) x else 0;
    return .{ .float = @round(v * 1e4) / 1e4 };
}

pub fn strings(a: Allocator, xs: []const []const u8) !json.Value {
    const out = try a.alloc(json.Value, xs.len);
    for (xs, out) |x, *v| v.* = .{ .string = x };
    return .{ .array = out };
}

/// A fact's answer records; a graph file from before them gives its questions with no answer known.
pub fn records(a: Allocator, f: Fact) ![]const Answer {
    if (f.answers.len > 0) return f.answers;
    const out = try a.alloc(Answer, f.questions.len);
    for (f.questions, out) |q, *x| x.* = .{ .question = q };
    return out;
}

/// A JSON array of strings as read (none: empty).
pub fn stringList(a: Allocator, v: ?json.Value) ![]const []const u8 {
    const xs = (v orelse return &.{}).items() orelse return &.{};
    const out = try a.alloc([]const u8, xs.len);
    for (xs, out) |x, *o| o.* = x.str() orelse "";
    return out;
}

pub fn dupeAll(a: Allocator, xs: []const []const u8) ![]const []const u8 {
    const out = try a.alloc([]const u8, xs.len);
    for (xs, out) |x, *o| o.* = try a.dupe(u8, x);
    return out;
}

/// [{"layer", "neuron", "key"}].
pub fn neuronList(a: Allocator, ns: []const api.Neuron) !json.Value {
    const out = try a.alloc(json.Value, ns.len);
    for (ns, out) |n, *v| {
        const o = try json.newObject(a);
        try o.put(a, "layer", try json.intValue(a, n.layer));
        try o.put(a, "neuron", try json.intValue(a, n.neuron));
        try o.put(a, "key", .{ .string = @tagName(n.key) });
        v.* = .{ .object = o };
    }
    return .{ .array = out };
}

/// {"state", "at", "seconds", "bytes", "files", "message"}.
pub fn saveValue(a: Allocator, s: Save) !json.Value {
    const o = try json.newObject(a);
    try o.put(a, "state", .{ .string = @tagName(s.state) });
    try o.put(a, "at", try json.intValue(a, s.at));
    if (s.state == .saving) return .{ .object = o };
    try o.put(a, "seconds", .{ .float = @round(s.seconds * 10) / 10 });
    try o.put(a, "bytes", try json.intValue(a, s.bytes));
    try o.put(a, "files", try strings(a, s.files));
    if (s.message.len > 0) try o.put(a, "message", .{ .string = s.message });
    return .{ .object = o };
}

/// Whether two facts share a word of five or more letters that is not on the common list, ignoring case.
pub fn shareWord(x: []const u8, y: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, x, " \t\n.,;:!?()[]\"'");
    while (it.next()) |w| {
        if (w.len < 5 or isCommon(w)) continue;
        var other = std.mem.tokenizeAny(u8, y, " \t\n.,;:!?()[]\"'");
        while (other.next()) |v| if (std.ascii.eqlIgnoreCase(w, v)) return true;
    }
    return false;
}

pub fn isCommon(w: []const u8) bool {
    for (common) |c| if (std.ascii.eqlIgnoreCase(w, c)) return true;
    return false;
}
