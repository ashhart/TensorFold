//! MTLPurgeableState's values (Metal's ABI) and the two transitions the Qwen idle weight release asks for.
const std = @import("std");

pub const State = enum(usize) { keep_current = 1, non_volatile = 2, volatile_ = 3, empty = 4 };

/// Released weights: the OS may discard the buffer's contents while no GPU work reads them.
pub const release: usize = @backingInt(State.volatile_);
/// Restored weights: the contents must stay valid before any GPU command reads them again.
pub const restore: usize = @backingInt(State.non_volatile);

/// setPurgeableState: returns the buffer's previous state; Empty means the OS discarded the contents.
pub fn discarded(previous: usize) bool {
    return previous == @backingInt(State.empty);
}

test "idle release asks Metal for Volatile, restores to NonVolatile and reads Empty as discarded" {
    try std.testing.expectEqual(@as(usize, 3), release);
    try std.testing.expectEqual(@as(usize, 2), restore);
    try std.testing.expect(discarded(4));
    for ([_]usize{ 1, 2, 3 }) |previous| try std.testing.expect(!discarded(previous));
}
