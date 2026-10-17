//! Analyze the real legacy host's constructor without linking or running GPU work.
export fn compileLegacy() usize {
    return @intFromPtr(&@import("host").openWarmed);
}
