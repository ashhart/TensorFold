//! BF16 prompt rows reach the native stream through the registered FP32 head and score reduction.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
pub const batch_rows = 128;
const words = lanes.logprob.words;
pub const Scores = struct {
    kernels: cuda.qwen35_bf16_rows.Kernels,
    norms: cuda.norms.Norms,
    logits: cuda.DeviceBuffer,
    picks: cuda.HostBuffer,
    out: cuda.HostBuffer,
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, vocab: usize) !*Scores {
        const b = try a.create(Scores);
        errdefer a.destroy(b);
        b.kernels = try cuda.qwen35_bf16_rows.Kernels.load(d);
        errdefer b.kernels.unload();
        if (b.kernels.dense_f32_fn == null or b.kernels.stats_fn == null) return error.MissingQwenBf16Scores;
        b.norms = try cuda.norms.Norms.load(d);
        errdefer b.norms.unload();
        b.logits = try cuda.DeviceBuffer.alloc(d, batch_rows * vocab * 4);
        errdefer b.logits.free();
        b.picks = try cuda.HostBuffer.allocMapped(d, batch_rows * 4);
        errdefer b.picks.free();
        b.out = try cuda.HostBuffer.allocMapped(d, batch_rows * words * 4);
        return b;
    }
    pub fn deinit(b: *Scores, a: std.mem.Allocator) void {
        b.out.free();
        b.picks.free();
        b.logits.free();
        b.norms.unload();
        b.kernels.unload();
        a.destroy(b);
    }
};

/// The current normalized chunk row r predicts ids[start+r+1], including its next-chunk seam.
pub fn collect(e: *Engine, stream: *lanes.Stream, ids: []const u32, start: usize, end: usize) !void {
    const k = stream.prompt_logprobs orelse return;
    const b = e.bf16 orelse return error.UnsupportedPromptLogprobs;
    errdefer cuda.completion.drain(e.d, e.stream, "BF16 score work did not drain");
    const rows = (end - start) - @intFromBool(end == ids.len);
    var at: usize = 0;
    while (at < rows) : (at += batch_rows) {
        if (stream.isCancelled()) return error.Cancelled;
        const n = @min(batch_rows, rows - at);
        const picks = ids[start + at + 1 ..][0..n];
        @memcpy(b.picks.slice(u32)[0..n], picks);
        try b.kernels.denseF32(e.stream, e.x + at * e.g.hidden * 2, e.head.bf.w, b.logits.ptr, n, e.g.vocab, e.g.hidden);
        try b.kernels.stats(e.stream, b.logits.ptr, e.g.vocab, try b.picks.device(), k, try b.out.device(), n);
        try e.stream.synchronize();
        const output = b.out.slice(u32);
        for (picks, 0..) |pick, r| stream.prompt_rows.appendAssumeCapacity(lanes.LogprobRow.fromWords(output[r * words ..][0..words], pick, k));
    }
}
