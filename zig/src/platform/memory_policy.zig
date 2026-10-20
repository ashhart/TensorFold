//! The shared reserve inside a Metal residency or preflight bound.
pub const margin: u64 = 2 << 30;

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

/// The process physical footprint, including Metal buffers; null when the OS has no reading.
pub fn footprint() ?u64 {
    if (@import("builtin").os.tag != .macos) return null;
    var info: [43]u64 = undefined;
    if (proc_pid_rusage(getpid(), 4, &info) != 0) return null;
    return info[9];
}

/// Growth across one owned load; a failed reading stays unknown and a falling footprint counts as no growth.
pub fn growth(before: ?u64, after: ?u64) ?u64 {
    const first = before orelse return null;
    const last = after orelse return null;
    if (first == 0 or last == 0) return null;
    return last -| first;
}
