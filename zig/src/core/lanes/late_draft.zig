//! A single-slot backend may start its next draft after the host publishes this round's tokens.
const std = @import("std");
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;

pub const State = struct {
    request: ?be.DraftRequest = null,
    follow: [128]u32 = undefined,
    rows: [128]u32 = undefined,

    pub fn put(l: *State, enabled: bool, r: be.DraftRequest) bool {
        if (!enabled or r.stream.sampling != null or r.stream.grammar != null or r.stream.logprobs != null or
            r.early or r.first != null or r.lanes != null or r.plan != null or r.ranks) return false;
        const rows = r.rows orelse return false;
        if (rows.len == 0 or rows.len > l.rows.len or rows.len != r.follow.len) return false;
        std.debug.assert(l.request == null);
        if (r.stream.finished) return true;
        @memcpy(l.rows[0..rows.len], rows);
        @memcpy(l.follow[0..rows.len], r.follow);
        l.request = r;
        return true;
    }

    pub fn cancel(l: *State, s: *const Stream) void {
        if (l.request) |r| if (r.stream == s) {
            l.request = null;
        };
    }

    pub fn flush(l: *State, e: anytype, ask: anytype) !void {
        var r = l.request orelse return;
        l.request = null;
        // Release/detach cancels ownership; also reject a request no longer in the live list before dereferencing it.
        if (std.mem.indexOfScalar(*Stream, e.live.items, r.stream) == null or r.stream.finished) return;
        r.follow = l.follow[0..r.follow.len];
        r.rows = l.rows[0..r.rows.?.len];
        try ask(e, &.{r});
    }
};
