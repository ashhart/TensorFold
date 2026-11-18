//! Windows: a released checkpoint step's pages leave the working set; the standby list keeps them for a refault.

const std = @import("std");
const builtin = @import("builtin");

/// The process's set, trimmed once, best effort: Windows has no per-range advice, so a step's pages go together.
pub fn release() void {
    if (comptime builtin.os.tag != .windows) return;
    // both bounds (SIZE_T)-1: the system takes out as many pages as it can, and takes none that are in use
    _ = SetProcessWorkingSetSize(GetCurrentProcess(), std.math.maxInt(usize), std.math.maxInt(usize));
}

extern "kernel32" fn GetCurrentProcess() callconv(.winapi) *anyopaque;
extern "kernel32" fn SetProcessWorkingSetSize(process: *anyopaque, min: usize, max: usize) callconv(.winapi) c_int;
