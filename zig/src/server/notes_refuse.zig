//! Startup refusals: --notes does not start beside a writer or engine whose copies erase cannot reach, and names it.
const std = @import("std");
const api = @import("engine_api");
const journal = @import("notes/journal.zig");

/// Writers that keep note text where erase does not reach; --learn and --slide stay allowed, named per erase instead.
pub const Writers = struct {
    recall: bool = false,
    compact_memory: bool = false,
    request_log: bool = false,
    cortheon: bool = false,
    speech: bool = false,
};

/// Why this start's flags keep copies an erase cannot clear; null: none do.
pub fn flags(w: Writers) ?[]const u8 {
    if (w.recall) return "--recall records prompt and reply tokens that erase does not search";
    if (w.compact_memory) return "--compact-memory writes compaction files that erase does not search";
    if (w.request_log) return "TENSORFOLD_REQUEST_LOG logs chat bodies that erase does not search";
    if (w.cortheon) return "--cortheon keeps its hidden rounds in memory, which erase does not reach";
    if (w.speech) return "a speech model keeps speech state that erase cannot select";
    return null;
}

/// A span no tokenizer emits, so asking the engine to forget it removes nothing and shows only what it can clear.
const probe = [_]u32{std.math.maxInt(u32)};

/// Why this engine cannot show its prompt states gone; null: it can, or it keeps none.
pub fn engine(e: api.Engine, info: api.Info) ?[]const u8 {
    if (!info.prompt_cache) return null;
    const got = e.forgetPrompts(&probe) catch |err| return switch (err) {
        error.Unsupported => "this engine has no prompt-state erase",
        error.Closed, error.Busy => "this engine's prompt-state erase did not answer at startup",
    };
    if (got.held > 0) return "this engine's peer does not acknowledge dropped prompt states (GLM-DSA, expert-parallel GLM or a tensor-parallel pair)";
    return null;
}

/// Why an earlier recorded start kept copies an erase cannot clear; null: every recorded start had them off.
pub fn history(runs: []const journal.Run) ?[]const u8 {
    for (runs) |run| {
        if (run.recall == null) return "an earlier start's --recall is unknown (an older journal)";
        if (flags(.{
            .recall = run.recall.?,
            .compact_memory = run.compact_memory.len != 0,
            .request_log = run.request_log.len != 0,
        })) |why| return why;
    }
    return null;
}

const Echo = @import("notes_storage_http_test.zig").Echo;

test "each writer that keeps a copy erase cannot clear is refused by name" {
    try std.testing.expect(flags(.{}) == null);
    // --learn and --slide are allowed: each erase names their places instead
    try std.testing.expect(history(&.{.{ .learn = true, .slide = true, .pocket = true }}) == null);
    const named = [_]struct { Writers, []const u8 }{
        .{ .{ .recall = true }, "--recall" },
        .{ .{ .compact_memory = true }, "--compact-memory" },
        .{ .{ .request_log = true }, "TENSORFOLD_REQUEST_LOG" },
        .{ .{ .cortheon = true }, "--cortheon" },
        .{ .{ .speech = true }, "speech" },
    };
    for (named) |n| try std.testing.expect(std.mem.indexOf(u8, flags(n[0]).?, n[1]) != null);
}

test "an engine that cannot show prompt states gone is refused" {
    var off: Echo = .{};
    try std.testing.expect(engine(off.engine(), .{}) == null); // no prompt cache: nothing to clear
    var none: Echo = .{ .cache = true };
    try std.testing.expect(std.mem.indexOf(u8, engine(none.engine(), .{ .prompt_cache = true }).?, "no prompt-state erase") != null);
    var pair: Echo = .{ .cache = true, .supports_forget = true, .forgotten = .{ .held = 1 } };
    try std.testing.expect(std.mem.indexOf(u8, engine(pair.engine(), .{ .prompt_cache = true }).?, "GLM-DSA") != null);
    var traces: Echo = .{ .cache = true, .supports_forget = true, .forgotten = .{ .unmanaged = 1 } };
    // token traces and unowned learned folders are named per erase, not refused
    try std.testing.expect(engine(traces.engine(), .{ .prompt_cache = true }) == null);
    var clear: Echo = .{ .cache = true, .supports_forget = true };
    try std.testing.expect(engine(clear.engine(), .{ .prompt_cache = true }) == null);
}

test "a journal that recorded a start with a writer on is refused" {
    try std.testing.expect(history(&.{ .{}, .{ .next_source = 4 } }) == null);
    try std.testing.expect(std.mem.indexOf(u8, history(&.{ .{}, .{ .recall = true } }).?, "--recall") != null);
    try std.testing.expect(std.mem.indexOf(u8, history(&.{.{ .request_log = "/tmp/log" }}).?, "TENSORFOLD_REQUEST_LOG") != null);
    try std.testing.expect(std.mem.indexOf(u8, history(&.{.{ .recall = null }}).?, "unknown") != null);
}
