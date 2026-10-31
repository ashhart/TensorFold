//! Chunk-boundary cancellation, background yield and completed prompt progress.
pub fn before(stream: anytype, at: usize) !void {
    if (stream.isCancelled()) {
        stream.keepInterruptedPrefill(@intCast(at));
        return error.Cancelled;
    }
    if (stream.yields()) {
        stream.keepInterruptedPrefill(@intCast(at));
        return error.Yielded;
    }
}
pub fn completed(stream: anytype, at: usize) void {
    stream.fill_at.store(@intCast(at), .monotonic);
}
pub fn finished(stream: anytype) !void {
    if (stream.isCancelled()) return error.Cancelled;
}

/// Only unobserved intermediate heads may be omitted. The final head is mandatory.
pub fn deferHead(enabled: bool, final: bool, marked_observer: bool, interruption_observer: bool) bool {
    return enabled and !final and !marked_observer and !interruption_observer;
}

test "LFM3 prefill deferred head retains final, mark and interruption observers" {
    const expect = @import("std").testing.expect;
    try expect(deferHead(true, false, false, false));
    try expect(!deferHead(false, false, false, false));
    try expect(!deferHead(true, true, false, false));
    try expect(!deferHead(true, false, true, false));
    try expect(!deferHead(true, false, false, true));
}
