//! Replace only after completion and successful allocation, retaining the old owner on either failure.
const cuda = @import("cuda");
pub fn grow(slot: *?cuda.DeviceBuffer, d: *const cuda.Driver, bytes: usize) !u64 {
    if (slot.* == null or slot.*.?.len < bytes) {
        if (slot.* != null) try d.synchronize();
        const replacement = try cuda.DeviceBuffer.alloc(d, bytes);
        if (slot.*) |*old| old.free();
        slot.* = replacement;
    }
    return slot.*.?.ptr;
}
