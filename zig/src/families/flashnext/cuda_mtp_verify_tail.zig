//! The verification window's output tail: the top-1 kernel and its records copy enqueued behind the main pass.
const cuda = @import("cuda");
const mtp = @import("cuda_mtp.zig");
const wt = @import("cuda_prompt_wt.zig");

const vocab = wt.vocab;
const Head = mtp.Head;
const Draft = mtp.Draft;

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
