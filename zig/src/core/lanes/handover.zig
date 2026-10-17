//! A stream leaving the rounds between two of them and coming back: handed to another driver, taken over, cancelled.
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;

/// Take over a stream another driver decoded so far: its cache settled, `pending` and `cache_len` set, no drafts held.
pub fn adopt(e: *Engine, s: *Stream) !void {
    try e.live.append(e.gpa, s);
}

/// Give a live stream to another driver between rounds, its round state dropped (adopt takes it back).
pub fn detach(e: *Engine, s: *Stream) bool {
    if (s.inflight != null or s.finished) return false;
    for (e.live.items, 0..) |x, i| {
        if (x != s) continue;
        _ = e.live.orderedRemove(i);
        s.dropRoundState(e.gpa);
        return true;
    }
    return false;
}

/// Release a cancelled stream between rounds: no pending draws or drafts.
pub fn discard(e: *Engine, s: *Stream) void {
    s.finished = true;
    s.reason = .cancelled;
    for (e.live.items, 0..) |x, i| {
        if (x == s) {
            _ = e.live.orderedRemove(i);
            break;
        }
    }
    e.release(s);
}
