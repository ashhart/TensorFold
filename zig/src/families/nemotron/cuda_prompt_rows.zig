//! prompt_logprobs on CUDA: each prompt row but the last through the head, for its next token's row and rank.
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// Where a prompt pass puts its rows: the next prompt tokens and the kernel's words, host-mapped, and who reads them.
pub const PromptRows = struct {
    k: u8,
    rows: usize, // rows a head pass takes: state.windowRows, or batch_rows with a logits buffer of its own
    logits: u64, // `rows` rows of vocab logits
    picks: []u32, // host-mapped, `rows` words, at `picks_dev` on the device
    picks_dev: u64,
    out: []const u32, // host-mapped, `rows` rows of `stride` words, at `out_dev`
    out_dev: u64,
    stride: usize,
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, words: []const u32, picks: []const u32) anyerror!void, // rows in prompt order
};

/// Rows a head pass takes when the prompt pass has a logits buffer of its own: the head's weights read once for all.
pub const batch_rows = 256;

/// The chunk's rows (in p_hidden) that predict `next` through the head, `p.rows` at a time.
pub fn run(e: *Engine, p: PromptRows, next: []const u32) !void {
    var r: usize = 0;
    while (r < next.len) : (r += p.rows) {
        const n = @min(p.rows, next.len - r);
        @memcpy(p.picks[0..n], next[r..][0..n]);
        try e.ops().prefillDense(e.b.p_hidden + r * @as(u64, e.c.hidden) * 2, e.w.head, p.logits, n);
        try e.ops().logprobRows(p.logits, e.c.vocab, p.picks_dev, p.k, true, p.out_dev, n);
        try e.stream.synchronize();
        try p.call(p.ctx, p.out[0 .. n * p.stride], p.picks[0..n]);
    }
}
