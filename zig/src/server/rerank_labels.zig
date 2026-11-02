//! A reranker checkpoint's answer labels: `1_LogitScore/config.json` names the yes and no tokens its score reads.
const std = @import("std");
const json = @import("json");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;

/// The token ids whose last-row logits make a rerank score: sigmoid(yes - no).
pub const Rerank = struct { yes: u32, no: u32 };

pub const Error = error{InvalidRerankLabels} || Allocator.Error;

const file = "1_LogitScore/config.json";

/// The labels of the checkpoint in `dir`; null when it has no such file (it is not a reranker), or `dir` is empty.
pub fn load(a: Allocator, io: std.Io, dir: []const u8, vocab: u32) Error!?Rerank {
    if (dir.len == 0) return null;
    const path = try std.fs.path.join(a, &.{ dir, file });
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 10)) catch |err| switch (err) {
        error.FileNotFound => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return bad(path, "cannot be read ({s})", .{@errorName(err)}),
    };
    const value = switch (try json.parse(a, text)) {
        .ok => |v| v,
        .err => return bad(path, "is not JSON", .{}),
    };
    if (value != .object) return bad(path, "is not a JSON object", .{});
    const yes = try label(path, value, "true_token_id", vocab);
    const no = try label(path, value, "false_token_id", vocab);
    if (yes == no) return bad(path, "names token {d} for both answers", .{yes});
    return .{ .yes = yes, .no = no };
}

fn label(path: []const u8, value: json.Value, key: []const u8, vocab: u32) Error!u32 {
    const v = value.get(key) orelse return bad(path, "has no {s}", .{key});
    const n = v.int64() orelse return bad(path, "{s} is not an integer", .{key});
    if (n < 0 or n >= vocab) return bad(path, "{s} {d} is not a token of this {d}-token vocabulary", .{ key, n, vocab });
    return @intCast(n);
}

fn bad(path: []const u8, comptime fmt: []const u8, args: anytype) error{InvalidRerankLabels} {
    log.line("{s} " ++ fmt, .{path} ++ args);
    return error.InvalidRerankLabels;
}

fn loadText(text: ?[]const u8, vocab: u32) anyerror!?Rerank {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const dir = try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{tmp.sub_path});
    if (text) |t| {
        try std.Io.Dir.cwd().createDirPath(io, try std.fs.path.join(arena.allocator(), &.{ dir, "1_LogitScore" }));
        try tmp.dir.writeFile(io, .{ .sub_path = file, .data = t });
    }
    return load(arena.allocator(), io, dir, vocab);
}

test "rerank: a reranker's labels come from 1_LogitScore, and a checkpoint without the file is no reranker" {
    const got = (try loadText("{\"true_token_id\": 9693, \"false_token_id\": 2152}", 151669)).?;
    try std.testing.expectEqual(@as(u32, 9693), got.yes);
    try std.testing.expectEqual(@as(u32, 2152), got.no);
    try std.testing.expectEqual(@as(?Rerank, null), try loadText(null, 151669));
    try std.testing.expectEqual(@as(?Rerank, null), try load(std.testing.allocator, std.testing.io, "", 151669));
}

test "rerank: a malformed 1_LogitScore stops startup instead of hiding the route" {
    const cases = [_][]const u8{
        "not json",
        "[9693, 2152]",
        "{\"false_token_id\": 2152}",
        "{\"true_token_id\": 9693}",
        "{\"true_token_id\": \"9693\", \"false_token_id\": 2152}",
        "{\"true_token_id\": 9693.5, \"false_token_id\": 2152}",
        "{\"true_token_id\": -1, \"false_token_id\": 2152}",
        "{\"true_token_id\": 9693, \"false_token_id\": 151669}",
        "{\"true_token_id\": 2152, \"false_token_id\": 2152}",
    };
    for (cases) |text| try std.testing.expectError(error.InvalidRerankLabels, loadText(text, 151669));
}
