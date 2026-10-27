//! The shared reserve inside a Metal residency or preflight bound.
pub const margin: u64 = 2 << 30;

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;
extern "c" fn sysctlbyname(name: [*:0]const u8, old: ?*anyopaque, old_len: ?*usize, new: ?*anyopaque, new_len: usize) c_int;

/// The kernel's memory pressure level (1 normal, 2 warning, 4 critical); null where the OS gives none.
pub fn pressure() ?u32 {
    if (@import("builtin").os.tag != .macos) return null;
    var level: u32 = 0;
    var len: usize = @sizeOf(u32);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, null, 0) != 0 or len != @sizeOf(u32)) return null;
    return level;
}

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
