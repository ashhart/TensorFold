//! The existing rank-gathered greedy pick, kept separate from engine lifetime and prompt diagnostics.
const cuda = @import("cuda");
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const chance = @import("cuda_mtp_chance.zig");
const Engine = @import("cuda_engine.zig").Engine;
const pair_bytes = 16;
/// The most rows one batched pick takes (a decode window's, a shared forward's).
pub const pick_rows = @max(@import("cuda_forward.zig").window_rows, @import("cuda_engine.zig").shared_rows);
pub fn greedy(e: *Engine, logits: u64) !u32 {
    return (try pick(e, logits, false)).token;
}

/// `rows` rows' greedy picks, each as `greedy` gives it: their top-1s, then one gather and one read for them all.
pub fn greedyRows(e: *Engine, logits: u64, rows: usize, out: []u32) !void {
    if (rows == 1) {
        out[0] = try greedy(e, logits);
        return;
    }
    const world = e.share.world;
    const bytes = rows * pair_bytes;
    const mine = e.pairs.ptr + e.share.rank * bytes;
    for (0..rows) |r| {
        var a: cuda.Args = .{};
        a.add(logits + r * e.share.vocab.len() * 2);
        a.add(@as(i64, @intCast(e.share.vocab.len())));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(mine + r * pair_bytes);
        try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    }
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world * pick_rows][4]u32 = undefined;
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.asBytes(&all)[0 .. world * bytes]);
    var row: [ranks.max_world][4]u32 = undefined;
    for (0..rows) |r| {
        for (0..world) |w| row[w] = all[w * rows + r];
        out[r] = chance.pick(row[0..world], false).token;
    }
}

/// greedy with the pick's chance from every rank's softmax sum (tf_glm_top1_sums), for --mtp-confidence's drafts.
pub fn greedyChance(e: *Engine, logits: u64) !chance.Pick {
    return pick(e, logits, true);
}

fn pick(e: *Engine, logits: u64, sums: bool) !chance.Pick {
    const world = e.share.world;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(e.share.vocab.len())));
    a.add(@as(u32, @intCast(e.share.vocab.lo)));
    a.add(e.pairs.ptr + e.share.rank * pair_bytes);
    try cuda.launch.launch(if (sums) e.m.glue.top1_sums else e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    const mine = e.pairs.ptr + e.share.rank * pair_bytes;
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world][4]u32 = undefined; // Engine.init refuses a larger world
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.asBytes(&all)[0 .. world * pair_bytes]);
    return chance.pick(all[0..world], sums);
}
