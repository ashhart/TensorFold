//! A folded-memory prefix a backend could not restore: its pass stops before a placeholder row is fed as text.
const Stream = @import("stream.zig").Stream;

/// A failed restore of `s.reuse`, before a fallback feeds row 0: a memory stream refuses (its prefix is placeholders).
pub fn refuseFallback(s: *const Stream) error{MemoryRestoreFailed}!void {
    if (s.memory_active) return error.MemoryRestoreFailed;
}
