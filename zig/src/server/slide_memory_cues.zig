//! A lesson's concept cues and its contradiction check against what memory already holds.
const std = @import("std");
const api = @import("engine_api");
const living = @import("living");
const mem = living.memory;
const json = @import("json");
const http_body = @import("http_body.zig");
const sse = @import("sse.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const slide = @import("slide.zig");
const slide_ask = @import("slide_ask.zig");
const slide_graph = @import("slide_graph.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

pub fn containsCI(hay: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or needle.len > hay.len) return false;
    var i: usize = 0;
    while (i + needle.len <= hay.len) : (i += 1) {
        var ok = true;
        for (0..needle.len) |j| {
            if (std.ascii.toLower(hay[i + j]) != std.ascii.toLower(needle[j])) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

/// Deterministic concept-cluster synonym cues for short everyday facts (e.g. "dog" <-> "pet/puppy").
pub fn deterministicSynonymCues(a: Allocator, fact_text: []const u8) ?[]const u8 {
    const Cluster = struct { triggers: []const []const u8, anchors: []const u8 };
    const clusters = [_]Cluster{
        .{ .triggers = &.{ "colour", "color", "shade", "hue", "tint" }, .anchors = "colour color shade hue tint tone prefer" },
        .{ .triggers = &.{ "dog", "puppy", "hound", "canine" }, .anchors = "dog pet puppy hound canine" },
        .{ .triggers = &.{ "cat", "kitten", "feline" }, .anchors = "cat pet kitten feline" },
        .{ .triggers = &.{ "sister", "brother", "sibling" }, .anchors = "sister brother sibling city town live lives moved" },
        .{ .triggers = &.{ "robot", "droid" }, .anchors = "robot bot droid warehouse machine" },
        .{ .triggers = &.{ "deploy", "release", "rollout" }, .anchors = "deploy deployment deployments release rollout schedule" },
        .{ .triggers = &.{ "password", "passcode" }, .anchors = "password passcode credential secret" },
        .{ .triggers = &.{ "sailboat", "yacht", "marina", "berth", "moored" }, .anchors = "sailboat boat vessel yacht moored marina berth dock" },
        .{ .triggers = &.{ "database", "postgres", "duckdb", "sqlite", "mysql" }, .anchors = "database db datastore sql storage backend" },
        .{ .triggers = &.{ "capital", "france", "paris", "lyon" }, .anchors = "capital city country nation" },
    };

    var buf: std.ArrayList(u8) = .empty;
    for (clusters) |c| {
        var hit = false;
        for (c.triggers) |tr| {
            if (containsCI(fact_text, tr)) {
                hit = true;
                break;
            }
        }
        if (hit) {
            if (buf.items.len > 0) buf.append(a, ' ') catch return null;
            buf.appendSlice(a, c.anchors) catch return null;
        }
    }
    if (buf.items.len == 0) return null;
    return buf.items;
}

/// Concept cues for short facts: deterministic synonym clusters first, else a one-line LLM cue pass.
pub fn generateConceptCues(srv: *Server, cx: *errors.Cx, fact_text: []const u8, gone: anytype) []const u32 {
    const a = cx.a;
    if (deterministicSynonymCues(a, fact_text)) |anchors| {
        return srv.text.encode(a, anchors, false) catch return &.{};
    }
    if (@import("sys").environment.read(bool, "TF_SLIDE_FAST_CUES", false) catch false) return &.{};
    const prompt = std.fmt.allocPrint(
        a,
        "List 8 single-word topic synonyms or related nouns on ONE line (comma-separated, no numbering, no questions, " ++
            "no pronouns) someone might use to ask about this fact.\nFact: {s}\nKeywords:",
        .{std.mem.trim(u8, fact_text, " \t\r\n")},
    ) catch return &.{};
    const reply = slide_ask.askIn(srv, cx, &.{}, prompt, 36, true, gone) catch return &.{};
    var trimmed = std.mem.trim(u8, reply.content, " \t\r\n");
    if (std.mem.indexOfScalar(u8, trimmed, '\n')) |nl| {
        trimmed = std.mem.trim(u8, trimmed[0..nl], " \t\r\n");
    }
    if (traced()) {
        log.line("slide(v6) trace: cues for '{s}' -> '{s}'", .{ std.mem.trim(u8, fact_text, " \t\r\n"), trimmed });
    }
    if (trimmed.len == 0) return &.{};
    return srv.text.encode(a, trimmed, false) catch return &.{};
}

pub fn startsWithCI(hay: []const u8, prefix: []const u8) bool {
    if (prefix.len > hay.len) return false;
    for (0..prefix.len) |i| {
        if (std.ascii.toLower(hay[i]) != std.ascii.toLower(prefix[i])) return false;
    }
    return true;
}

/// Asks the model with stored memory active (`shipped = false`) whether `new_text` replaces a recorded value.
pub fn verifyContradictionWithMemory(srv: *Server, cx: *errors.Cx, new_text: []const u8, gone: anytype) bool {
    const a = cx.a;
    const prompt = std.fmt.allocPrint(
        a,
        "Compare the new statement against your stored notes.\n" ++
            "New statement: {s}\n" ++
            "Write ONE short sentence comparing what attribute each states, then on a new line write ONLY:\n" ++
            "VERDICT: COMPATIBLE (if both can be true at the same time, such as adding a hobby, activity, breed, or extra detail)\n" ++
            "or\n" ++
            "VERDICT: CONTRADICT (ONLY if the new statement replaces the exact same attribute with a different conflicting value, making the old value false).\n" ++
            "Comparison:",
        .{std.mem.trim(u8, new_text, " \t\r\n")},
    ) catch return false;
    const reply = slide_ask.askIn(srv, cx, &.{}, prompt, 40, false, gone) catch return false;
    const ans = std.mem.trim(u8, reply.content, " \t\r\n");
    if (traced()) {
        log.line("slide(v6) revision probe: '{s}' -> '{s}'", .{ std.mem.trim(u8, new_text, " \t\r\n"), ans });
    }
    if (containsCI(ans, "verdict: contradict")) return true;
    if (containsCI(ans, "verdict: compatible")) return false;
    // Fallback if model omitted the "VERDICT:" prefix on the final line
    if (std.mem.lastIndexOfScalar(u8, ans, '\n')) |nl| {
        const last = std.mem.trim(u8, ans[nl + 1 ..], " \t\r\n.*:-");
        return startsWithCI(last, "contradict");
    }
    return startsWithCI(ans, "contradict");
}

/// TF_SLIDE_TRACE: lessons log their text, cues and verdicts (checked at startup with --slide).
pub fn traced() bool {
    return @import("sys").environment.read(bool, "TF_SLIDE_TRACE", false) catch false;
}
