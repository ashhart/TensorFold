//! prompt_logprobs on CUDA: each prompt row but the last through the head, for its next token's row and rank.
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// Where a prompt pass puts its rows: the next prompt tokens and the kernel's words, host-mapped, and who reads them.
pub const PromptRows = struct {
    k: u8,
    picks: []u32, // host-mapped, state.max_rows words, at `picks_dev` on the device
    picks_dev: u64,
    out: []const u32, // host-mapped, state.max_rows rows of `stride` words, at `out_dev`
    out_dev: u64,
    stride: usize,
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, words: []const u32, picks: []const u32) anyerror!void, // rows in prompt order
};

/// The chunk's rows (in p_hidden) that predict `next` through the head, state.max_rows at a time.
pub fn run(e: *Engine, p: PromptRows, next: []const u32) !void {
    var r: usize = 0;
    while (r < next.len) : (r += state.max_rows) {
        const n = @min(state.max_rows, next.len - r);
        @memcpy(p.picks[0..n], next[r..][0..n]);
        try e.ops().prefillDense(e.b.p_hidden + r * @as(u64, e.c.hidden) * 2, e.w.head, e.b.logits, n);
        try e.ops().logprobRows(e.b.logits, e.c.vocab, p.picks_dev, p.k, true, p.out_dev, n);
        try e.stream.synchronize();
        try p.call(p.ctx, p.out[0 .. n * p.stride], p.picks[0..n]);
    }
}
