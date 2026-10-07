//! Kernel process footprint readings; device allocator counters have a separate scope and reset policy.
const std = @import("std");
const builtin = @import("builtin");

pub const Snapshot = struct { physical_footprint_bytes: u64, lifetime_peak_physical_footprint_bytes: u64 };

// Darwin's rusage_info_v4 ABI, including the UUID and trailing fields the kernel writes but we never publish.
const RUsage = extern struct {
    uuid: [16]u8,
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
    cpu_time_qos_default: u64,
    cpu_time_qos_maintenance: u64,
    cpu_time_qos_background: u64,
    cpu_time_qos_utility: u64,
    cpu_time_qos_legacy: u64,
    cpu_time_qos_user_initiated: u64,
    cpu_time_qos_user_interactive: u64,
    billed_system_time: u64,
    serviced_system_time: u64,
    logical_writes: u64,
    lifetime_max_phys_footprint: u64,
    instructions: u64,
    cycles: u64,
    billed_energy: u64,
    serviced_energy: u64,
    interval_max_phys_footprint: u64,
    runnable_time: u64,
};

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

/// One kernel sample, including peaks between HTTP polls; null on unsupported platforms or failed reads.
pub fn read() ?Snapshot {
    if (builtin.os.tag != .macos) return null;
    var info: RUsage = undefined;
    if (proc_pid_rusage(getpid(), 4, &info) != 0) return null;
    return .{
        .physical_footprint_bytes = info.phys_footprint,
        .lifetime_peak_physical_footprint_bytes = info.lifetime_max_phys_footprint,
    };
}

test "Darwin v4 footprint fields use the documented ABI layout" {
    try std.testing.expectEqual(@as(usize, 296), @sizeOf(RUsage));
    try std.testing.expectEqual(@as(usize, 72), @offsetOf(RUsage, "phys_footprint"));
    try std.testing.expectEqual(@as(usize, 240), @offsetOf(RUsage, "lifetime_max_phys_footprint"));
}

test "kernel lifetime footprint peak includes the current sample and cannot reset" {
    if (builtin.os.tag != .macos) {
        try std.testing.expect(read() == null);
        return;
    }
    const first = read() orelse return error.NoProcessMemory;
    const second = read() orelse return error.NoProcessMemory;
    try std.testing.expect(first.physical_footprint_bytes > 0);
    try std.testing.expect(first.lifetime_peak_physical_footprint_bytes >= first.physical_footprint_bytes);
    try std.testing.expect(second.lifetime_peak_physical_footprint_bytes >= first.lifetime_peak_physical_footprint_bytes);
}
