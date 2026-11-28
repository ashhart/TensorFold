//! The reply-end snapshot, retaining a matching pre-reply cache state too when both fit.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("tensorfold").lanes;
const Allocator = std.mem.Allocator;

pub fn keepEnd(store: *api.prompt_cache.Store, gpa: Allocator, s: *lanes.Stream, stands: u32) void {
    const prompt: u32 = @intCast(s.prompt_len);
    const tokens = s.context.items;
    if (tokens.len == prompt or stands + 1 < tokens.len or stands > tokens.len) return;
    const spans = api.cache_modes.reply(gpa, s.decode_spans, prompt, stands) catch return;
    defer gpa.free(spans);
    // an image prompt's rows go by their images (the request's cache key), as its lookups and marks do
    const keyed = if (s.key.len == prompt) gpa.dupe(u32, tokens) catch return else tokens;
    defer if (keyed.ptr != tokens.ptr) gpa.free(keyed);
    if (keyed.ptr != tokens.ptr) @memcpy(keyed[0..prompt], s.key);
    // Keep the history too when both copies fit; otherwise the reply end keeps its ordinary priority.
    const history = if (s.history_len > 0) store.find(keyed[0..@min(keyed.len, @as(usize, s.history_len) + 1)], s.chunks, spans) else null;
    const pin = if (history) |e| if (@import("qwen27_resume.zig").pinReplyHistory(store, e, s.history_len, stands)) e else null else null;
    if (pin) |e| e.borrowers += 1;
    defer if (pin) |e| {
        e.borrowers -= 1;
    };
    _ = store.keepBorn(keyed, prompt, stands, s, s.chunks, spans);
}
