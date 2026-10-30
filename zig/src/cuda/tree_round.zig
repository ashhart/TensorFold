//! tree_round.cu's verified tree rounds for every CUDA family: picks, each tree's verified path, and its commit steps.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// core/tree_round.zig's contract, as the kernels lay it out.
pub const max_rows = 16;
pub const Pick = extern struct { token: u32, nonfinite: u32 };
pub const Result = extern struct {
    status: u32,
    stop: u32,
    nonfinite: u32,
    consumed_count: u32,
    emitted_count: u32,
    matched_count: u32,
    bonus_emitted: u32,
    pending_valid: u32,
    pending_token: u32,
    reserved: u32,
    path: [max_rows]u32,
    tokens: [max_rows]u32,
};

/// One tree of a round: its rows of tokens, parents (tree-local, -1 at the root) and picks, its budget and EOS ids.
pub const Tree = extern struct { first: u32, rows: u32, budget: u32, eos_first: u32 = 0, eos_count: u32 = 0 };

/// tree_round.cu's kernels, resolved from the loaded `tree_round` module (kernels.tree_round).
pub const Kernels = struct {
    argmax_fn: Function,
    match_fn: Function,
    keep_fn: Function,
    gather_fn: Function,
    head_fn: Function,
    ids_fn: Function,

    pub fn resolve(mod: Module) !Kernels {
        return .{ .argmax_fn = try mod.function("tf_round_bf16_argmax"), .match_fn = try mod.function("tf_tree_round_match"), .keep_fn = try mod.function("tf_round_keep_rows"), .gather_fn = try mod.function("tf_round_gather_rows"), .head_fn = try mod.function("tf_round_keep_head"), .ids_fn = try mod.function("tf_round_pick_ids") };
    }

    /// ids i32 [rows]: each pick's token, or `fallback` for one with none (its flag stays in the picks).
    pub fn pickIds(g: *const Kernels, s: Stream, picks: u64, ids: u64, rows: usize, vocab: usize, fallback: u32) !void {
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(picks);
        a.add(ids);
        a.add(@as(u32, @intCast(rows)));
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(i32, @intCast(fallback)));
        try launch_.launch(g.ids_fn, .{ .grid = .{ .x = @intCast((rows + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// picks[r] for every row of a round's trees: the first largest finite bf16 logit, or none.
    pub fn argmax(g: *const Kernels, s: Stream, logits: u64, picks: u64, rows: usize, vocab: usize, stride: usize) !void {
        if (rows == 0) return;
        if (vocab == 0 or stride < vocab) return error.BadRoundShape;
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(picks);
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(u32, @intCast(stride)));
        try launch_.launch(g.argmax_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// results[t] for `count` trees (Tree [count] at `trees`): each tree's verified path and emitted tokens.
    pub fn match(g: *const Kernels, s: Stream, tokens: u64, parents: u64, picks: u64, eos: u64, trees: u64, results: u64, vocab: usize, count: usize) !void {
        if (count == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ tokens, parents, picks, eos, trees, results }) |v| a.add(v);
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(u32, @intCast(count)));
        try launch_.launch(g.match_fn, .{ .grid = .{ .x = @intCast((count + 63) / 64), .y = 1, .z = 1 }, .block = .{ .x = 64 } }, s, &a);
    }

    /// keep i32 [count, max_rows]: each tree's kept rows (round row indices), -1 past them; counts u32 [count].
    pub fn keepRows(g: *const Kernels, s: Stream, results: u64, trees: u64, keep: u64, counts: u64, count: usize) !void {
        if (count == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ results, trees, keep, counts }) |v| a.add(v);
        a.add(@as(u32, @intCast(count)));
        try launch_.launch(g.keep_fn, .{ .grid = .{ .x = @intCast(count), .y = 1, .z = 1 }, .block = .{ .x = max_rows } }, s, &a);
    }

    /// out bf16 [count, max_rows, width]: each tree's kept rows of `rows` (bf16 [round rows, width]), in path order.
    pub fn gatherRows(g: *const Kernels, s: Stream, results: u64, trees: u64, rows: u64, out: u64, width: usize, count: usize) !void {
        if (count == 0 or width == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ results, trees, rows, out }) |v| a.add(v);
        a.add(@as(u32, @intCast(width)));
        a.add(@as(u32, @intCast(count)));
        try launch_.launch(g.gather_fn, .{ .grid = .{ .x = @intCast((width + 255) / 256), .y = @intCast(count), .z = max_rows }, .block = .{ .x = 256 } }, s, &a);
    }

    /// head bf16 [count, width]: each tree's last kept row of `logits` (`stride` apart), the next round's first row.
    pub fn keepHead(g: *const Kernels, s: Stream, results: u64, trees: u64, logits: u64, head: u64, width: usize, stride: usize, count: usize) !void {
        if (count == 0 or width == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ results, trees, logits, head }) |v| a.add(v);
        a.add(@as(u32, @intCast(width)));
        a.add(@as(u32, @intCast(stride)));
        a.add(@as(u32, @intCast(count)));
        try launch_.launch(g.head_fn, .{ .grid = .{ .x = @intCast((width + 255) / 256), .y = @intCast(count), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }
};

test "the kernels' layouts" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Pick));
    try std.testing.expectEqual(@as(usize, 168), @sizeOf(Result));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(Result, "path"));
    try std.testing.expectEqual(@as(usize, 104), @offsetOf(Result, "tokens"));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(Tree));
}
