//! A fixed general bank and graph-derived audit prompts, with no generated answers or lesson changes.
const std = @import("std");
const graph = @import("slide_graph.zig");
const subject = @import("slide_subject.zig");
const A = std.mem.Allocator;

pub const Cohort = enum { own, subject, other, general };
pub const Probe = struct { text: []const u8, owner: u32, kind: enum { own, subject, general } };
pub const Fact = struct { source: graph.Fact, subject: []const u8 };
pub const Plan = struct { facts: []const Fact, probes: []const Probe };
pub const version = "pocket-audit-v1";

const topics = [_][]const u8{
    "fractions",          "prime numbers",    "probability",        "geometry",            "gravity",            "electric circuits", "sound waves",
    "the water cycle",    "photosynthesis",   "bird migration",     "ocean tides",         "rock formation",     "world time zones",  "ancient trade routes",
    "the printing press", "public libraries", "maps",               "music notation",      "rhythm",             "colour mixing",     "perspective drawing",
    "bread making",       "food storage",     "plant care",         "bicycle maintenance", "walking directions", "Python lists",      "SQL joins",
    "Unicode text",       "HTTP caching",     "sorting algorithms", "unit testing",
};
const tasks = [_][]const u8{
    "Explain {s} in two sentences.",                  "Give a concrete example of {s}.",
    "Write a question and answer about {s}.",         "Describe a common misconception about {s}.",
    "List three useful terms for understanding {s}.", "Make a short analogy that explains {s}.",
    "Describe a practical use of {s}.",               "Suggest a beginner exercise about {s}.",
};
const contexts = [_][]const u8{
    "Write a fictional sentence containing '{s}' about a train journey.",
    "Translate the phrase '{s}' into Spanish without adding personal details.",
    "Use '{s}' in a question about tomorrow's weather.",
    "Write a short grammar exercise that includes '{s}'.",
    "Imagine '{s}' is the title of a painting; describe its colours.",
    "Count the letters in '{s}'.",
    "Write a dialogue containing '{s}' about choosing a lunch.",
    "Use '{s}' in a fictional story about a library.",
};

pub fn cohort(p: Probe, id: u32) ?Cohort {
    return switch (p.kind) {
        .general => .general,
        .own => if (p.owner == id) .own else .other,
        .subject => if (p.owner == id) .subject else null,
    };
}

fn add(a: A, ps: *std.ArrayList(Probe), p: Probe) !void {
    for (ps.items) |old| if (old.owner == p.owner and old.kind == p.kind and std.mem.eql(u8, old.text, p.text)) return;
    if (ps.items.len == 2048) return error.AuditTooLarge;
    try ps.append(a, p);
}

pub fn plan(a: A, facts: []const graph.Fact) !Plan {
    var fs: std.ArrayList(Fact) = .empty;
    var ps: std.ArrayList(Probe) = .empty;
    for (facts) |f| if (f.state == .learned and f.neurons.len > 0) {
        var qs: std.ArrayList([]const u8) = .empty;
        for (f.questions) |q| {
            try qs.append(a, q);
            try add(a, &ps, .{ .text = q, .owner = f.id, .kind = .own });
        }
        for (f.answers) |answer| {
            try qs.append(a, answer.question);
            try add(a, &ps, .{ .text = answer.question, .owner = f.id, .kind = .own });
        }
        const words = try subject.factSubject(a, f.text, f.text, qs.items, false);
        const name = try std.mem.join(a, " ", words);
        try fs.append(a, .{ .source = f, .subject = name });
        if (name.len > 0) inline for (contexts) |fmt| {
            try add(a, &ps, .{ .text = try std.fmt.allocPrint(a, fmt, .{name}), .owner = f.id, .kind = .subject });
        };
    };
    if (fs.items.len > 0) for (topics) |topic| inline for (tasks) |fmt| {
        try add(a, &ps, .{ .text = try std.fmt.allocPrint(a, fmt, .{topic}), .owner = 0, .kind = .general });
    };
    return .{ .facts = fs.items, .probes = ps.items };
}

test "audit probe bank: 256 fixed varied prompts and graph-derived cohorts without modifying facts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const n = @import("engine_api").Neuron{ .layer = 2, .neuron = 0, .key = .subject };
    const fs = [_]graph.Fact{
        .{ .id = 1, .text = "My dog is called Ginger.", .source = "chat", .at = 1, .state = .learned, .neurons = &.{n}, .questions = &.{"What is my dog called?"} },
        .{ .id = 2, .text = "My sister is called Ana.", .source = "chat", .at = 2, .state = .learned, .neurons = &.{n}, .questions = &.{"What is my sister called?"} },
    };
    const p = try plan(a, &fs);
    var counts: [4]usize = @splat(0);
    for (p.probes) |pr| if (cohort(pr, 1)) |c| {
        counts[@backingInt(c)] += 1;
    };
    try std.testing.expectEqualSlices(usize, &.{ 1, 8, 1, 256 }, &counts);
    try std.testing.expectEqualStrings("dog", p.facts[0].subject);
    try std.testing.expectEqual(@as(u32, 0), fs[0].checks);
    try std.testing.expectEqualStrings("My dog is called Ginger.", fs[0].text);
}
