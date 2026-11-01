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
