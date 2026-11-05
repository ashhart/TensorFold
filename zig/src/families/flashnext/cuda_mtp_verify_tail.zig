//! The verification window's output tail: the top-1 kernel and its records copy enqueued behind the main pass.
const std = @import("std");
const cuda = @import("cuda");
const mtp = @import("cuda_mtp.zig");
const wt = @import("cuda_prompt_wt.zig");
const prompt = @import("cuda_prompt.zig");

const vocab = wt.vocab;
const Head = mtp.Head;
const Draft = mtp.Draft;

/// Not a user switch: the tail is enqueued only when a test or tool turns this on (greedy drafts, graphs, two ranks).
pub var enabled: bool = false;

pub const VerifyTail = struct {
    head: *Head,
    draft: *Draft,
    generation: u64 = 0,
    enqueued: bool = false,

    pub fn enqueue(ptr: *anyopaque, st: cuda.Stream) !void {
        const tail: *VerifyTail = @ptrCast(@alignCast(ptr));
        const window = tail.draft;
        tail.generation = window.verify_cache.generation;
        try tail.head.best(st, window.logits.ptr, vocab, vocab, window.rows, window.best.ptr);
        tail.enqueued = true;
        window.verify_pending = st;
        try window.best.downloadAsync(0, window.verify_host.?.bytes[0 .. window.rows * 16], st.handle);
    }
};

/// Download a completed device result onto the host.
pub fn download(stream: cuda.Stream, dst: []u8, src: u64) !void {
    const d = stream.d;
    try stream.synchronize();
    try d.check(d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "cuMemcpyDtoH");
}

/// Verify a sampled window, publishing its greedy tail only when the graph enqueued it.
pub fn verifySampled(ops: anytype, seq: *prompt.Seq, window: *Draft, ids: []const u32, sm: ?mtp.Sampling) !void {
    window.verify_cache.invalidate();
    const greedy = if (sm) |sampling| sampling.temperature <= 0 else true;
    const leader = if (ops.main) |main| main.rank == 0 else false;
    if (!enabled or !greedy or !leader or ops.e.collective == null or !@import("exl3/graph/cuda_exl3_graph.zig").enabled()) return ops.verify(seq, window, ids);
    if (window.verify_host == null) window.verify_host = try cuda.HostBuffer.alloc(ops.e.driver, mtp.max_rows * 16);
    var tail: VerifyTail = .{ .head = ops.h, .draft = window };
    errdefer window.verify_cache.invalidate();
    try ops.verifyWithTail(seq, window, ids, .{ .ptr = &tail, .enqueue = VerifyTail.enqueue });
    if (tail.enqueued) {
        window.verify_pending = null;
        window.verify_cache.publish(tail.generation, window.rows, window.fed0);
    }
}

/// Draw each verified row at its keyed position, respecting any per-row grammar mask.
pub fn draws(o: anytype, d: *Draft, sm: ?mtp.Sampling, masks: []const u32, out: []u32) !void {
    const n = d.rows;
    const greedy = if (sm) |s| s.temperature <= 0 else true;
    const cached = d.verify_cache.take(n, d.fed0, greedy);
    if (greedy and masks.len == 0 and cached) {
        @memcpy(d.best_host[0 .. 4 * n], d.verify_host.?.slice(f32)[0 .. 4 * n]);
        for (out[0..n], 0..) |*t, r| t.* = @intCast(@as(i32, @bitCast(d.best_host[4 * r + 2])));
        return;
    }
    return @import("cuda_mtp_draw.zig").draws(o.h, o.e.stream.*, o.e.gpa, d, sm, masks, out);
}
