//! Whole-vocabulary grammar draws and prompt redraws.
const cuda = @import("cuda");
const sampler = @import("cuda_sampler.zig");
const b16 = @import("cuda_b16.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// Structured output's draw: the rows whole on every rank (Sampler.gather); rank 0 masks them by `bits` and draws.
pub fn drawWhole(e: anytype, logits: u64, rows: usize, first: u64, bits: u64, out: []u32) !void {
    try e.smp.gather(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, e.smp.full.ptr);
    if (e.share.rank != 0) {
        try e.stream.synchronize();
        return e.checkGathers();
    }
    if (bits != 0) try e.smp.mask(e.stream, rows, bits, (e.c.vocab + 31) / 32);
    if (e.cur.sampling) |s| {
        try e.smp.drawFull(e.stream, rows, first, sampler.rule(s), out);
        return e.checkGathers();
    }
    for (0..rows) |r| { // greedy's rule on the whole row: the largest value, then the lowest id
        var a: cuda.Args = .{};
        a.add(e.smp.full.ptr + r * e.c.vocab * 2);
        a.add(@as(i64, @intCast(e.c.vocab)));
        a.add(@as(u32, 0));
        a.add(e.pairs.ptr);
        try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
        var pick: [2]u32 = undefined;
        try e.stream.synchronize();
        try e.checkGathers();
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&pick, e.pairs.ptr, 8), "cuMemcpyDtoH");
        out[r] = pick[1];
    }
}

/// The first token drawn again from the prompt's last row, among `bits`' tokens (cuda_follow's redraw step).
pub fn redrawFirst(e: anytype, len: usize, bits: u64) !u32 {
    try e.promptHead();
    var t: [1]u32 = .{0};
    try e.drawWhole(e.bd.logits, 1, len, bits, &t);
    return t[0];
}

/// Recompute only the bound sequence's prompt head, using the same projection policy as prompt finishing.
pub fn promptHead(e: *Engine) !void {
    const D = e.c.hidden;
    try b16.matmul(prompt_lp.projectionSet(e), e.stream, e.cur.st.prompt_hidden, D, e.w.head, e.bd.logits, e.bd.kda.mm_part, 1, e.share.vocab.len(), D, false);
}
