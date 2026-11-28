//! Analyze the actual device-check body with inert kernel images; no device or NVCC is used.
export fn analyzeSharedStarts() usize {
    return @intFromPtr(&@import("check").sharedCheck);
}
