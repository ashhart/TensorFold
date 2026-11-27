//! The 27B's reply-end snapshot: born with its prompt, its matching history pinned too when both fit.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const pc = api.prompt_cache;
const Allocator = std.mem.Allocator;

/// The reply's state at `stands` (its end or one short), born with its prompt; its history pinned too when both fit.
pub fn keepEnd(store: *pc.Store, keep_remote: bool, gpa: Allocator, s: *lanes.Stream, stands: u32) void {
    if (s.reuse.remote and !s.reuse_failed and !keep_remote) return; // on another machine's prefix: only with TF_PREFILL_KEEP
    const prompt: u32 = @intCast(s.prompt_len);
    const tokens = s.context.items;
    if (tokens.len == prompt or stands + 1 < tokens.len or stands > tokens.len) return;
    const spans = api.cache_modes.reply(gpa, s.decode_spans, prompt, stands) catch return;
    defer gpa.free(spans);
    // an image prompt's rows go by their images (the request's cache key), as its lookups and marks do
    const keyed = if (s.key.len == prompt) gpa.dupe(u32, tokens) catch return else tokens;
    defer if (keyed.ptr != tokens.ptr) gpa.free(keyed);
    if (keyed.ptr != tokens.ptr) @memcpy(keyed[0..prompt], s.key);
    store.keep_remote = s.reuse.remote and !s.reuse_failed; // a turn resuming it is remote too (#1594 step 2b)
    defer store.keep_remote = false;
    // Keep the history too when both copies fit; otherwise the reply end keeps its ordinary priority.
    const history = if (s.history_len > 0) store.find(keyed[0..@min(keyed.len, @as(usize, s.history_len) + 1)], s.chunks, spans) else null;
    const pin = if (history) |e| if (@import("qwen27_resume.zig").pinReplyHistory(store, e, s.history_len, stands)) e else null else null;
    if (pin) |e| e.borrowers += 1;
    defer if (pin) |e| {
        e.borrowers -= 1;
    };
    _ = store.keepBorn(keyed, prompt, stands, s, s.chunks, spans);
}
