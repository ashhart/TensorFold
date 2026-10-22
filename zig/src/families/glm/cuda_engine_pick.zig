//! The existing rank-gathered greedy pick, kept separate from engine lifetime and prompt diagnostics.
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");
const Engine = @import("cuda_engine.zig").Engine;
const pair_bytes = 16;
pub fn greedy(e: *Engine, logits: u64) !u32 {
    const world = e.share.world;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(e.share.vocab.len())));
    a.add(@as(u32, @intCast(e.share.vocab.lo)));
    a.add(e.pairs.ptr + e.share.rank * pair_bytes);
    try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    const mine = e.pairs.ptr + e.share.rank * pair_bytes;
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world][4]u32 = undefined; // Engine.init refuses a larger world
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.pairs.ptr, world * pair_bytes), "cuMemcpyDtoH");
    var pick = all[0];
    for (all[1..world]) |p| {
        const v: f32 = @bitCast(p[0]);
        const b: f32 = @bitCast(pick[0]);
        if (v > b or (v == b and p[1] < pick[1])) pick = p;
    }
    return pick[1];
}
