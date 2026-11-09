//! Diagnose a re-rendered reply without changing exact-prefix lookup or restore.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");

pub const Difference = struct { at: usize, kept: u32, request: u32 };

pub fn find(s: *const pc.Store, prompt: []const u32, spans: []const modes.Span, from: ?*const pc.Entry) ?Difference {
    const resumed = from orelse return null;
    for (s.by_position.items) |e| {
        if (e.at <= resumed.at) break;
        if (e.expired or e.shared or e.born == 0 or e.born > resumed.tokens.len or
            !@import("cache_images.zig").matches(s, e)) continue;
        const at = std.mem.indexOfDiff(u32, e.tokens, prompt) orelse continue;
        if (at < e.born or at >= e.tokens.len or at >= prompt.len or
            !modes.equal(e.decode_spans, spans, @intCast(at))) continue;
        return .{ .at = at, .kept = e.tokens[at], .request = prompt[at] };
    }
    return null;
}

pub fn report(s: *const pc.Store, prompt: []const u32, spans: []const modes.Span, from: ?*const pc.Entry) void {
    if (!@import("builtin").is_test and !std.log.logEnabled(.debug, .default)) return;
    const d = find(s, prompt, spans, from) orelse return;
    const fmt = "prompt cache: saved reply diverges at {d}: kept token {d}, request token {d}; resuming at {d}";
    const args = .{ d.at, d.kept, d.request, from.?.at };
    if (@import("builtin").is_test) return @import("prompt_cache_note.zig").hear(fmt, args);
    std.log.debug(fmt, args);
}
