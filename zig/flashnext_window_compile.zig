//! Analyze the actual context entry without linking or running a device.
export fn compileWindowOpen() usize {
    return @intFromPtr(&@import("native").openFlashNext);
}
