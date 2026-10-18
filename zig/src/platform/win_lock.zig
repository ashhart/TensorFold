//! flock on Windows: LockFileEx over a file's first byte; a directory's lock is on `.tensorfold-lock` inside it.

const std = @import("std");
const c = @import("win_c.zig");

pub const LOCK = struct {
    pub const SH: c_int = 1;
    pub const EX: c_int = 2;
    pub const NB: c_int = 4;
    pub const UN: c_int = 8;
};

const HANDLE = *anyopaque;
const invalid: isize = -1;
const OVERLAPPED = extern struct { internal: usize = 0, internal_high: usize = 0, offset: u32 = 0, offset_high: u32 = 0, event: ?HANDLE = null };
extern "kernel32" fn LockFileEx(h: HANDLE, flags: u32, reserved: u32, low: u32, high: u32, ov: *OVERLAPPED) callconv(.winapi) c_int;
extern "kernel32" fn UnlockFileEx(h: HANDLE, reserved: u32, low: u32, high: u32, ov: *OVERLAPPED) callconv(.winapi) c_int;
extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?HANDLE) callconv(.winapi) isize;
extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) c_int;
extern "kernel32" fn GetLastError() callconv(.winapi) u32;
extern "c" fn _get_osfhandle(fd: c_int) isize;

/// Each directory descriptor's lock file handle (by its slot), closed with the descriptor.
var dir_locks: [256]isize = @splat(invalid);
var locks_mutex: std.atomic.Mutex = .unlocked;

/// flock's whole-file lock: shared or exclusive, blocking unless NB; a lock held against it is AGAIN.
pub fn flock(fd: c.fd_t, op: c_int) c_int {
    const h: isize = if (c.dirPath(fd)) |dir| dirLock(fd - c.dir_base, dir) orelse return -1 else _get_osfhandle(fd);
    if (h == invalid) return c.fail(.BADF);
    const handle: HANDLE = @ptrFromInt(@as(usize, @bitCast(h)));
    var ov: OVERLAPPED = .{};
    if (op & LOCK.UN != 0) return if (UnlockFileEx(handle, 0, 1, 0, &ov) != 0) 0 else c.fail(c.lastError());
    const flags: u32 = (if (op & LOCK.EX != 0) @as(u32, 2) else 0) | (if (op & LOCK.NB != 0) @as(u32, 1) else 0);
    if (LockFileEx(handle, flags, 0, 1, 0, &ov) != 0) return 0;
    return c.fail(if (GetLastError() == 33) .AGAIN else c.lastError()); // ERROR_LOCK_VIOLATION
}

/// The lock file handle of directory slot `slot`, opened on first use.
fn dirLock(slot: c.fd_t, dir: []const u8) ?isize {
    while (!locks_mutex.tryLock()) std.atomic.spinLoopHint();
    defer locks_mutex.unlock();
    const held = &dir_locks[@intCast(slot)];
    if (held.* != invalid) return held.*;
    var path: [1100]u8 = undefined;
    var wide: [1100:0]u16 = undefined;
    const p = std.fmt.bufPrint(&path, "{s}\\.tensorfold-lock", .{dir}) catch {
        _ = c.fail(.NAMETOOLONG);
        return null;
    };
    const n = std.unicode.wtf8ToWtf16Le(&wide, p) catch {
        _ = c.fail(.INVAL);
        return null;
    };
    wide[n] = 0;
    const h = CreateFileW(wide[0..n :0].ptr, 0x8000_0000 | 0x4000_0000, 0x7, null, 4, 0x82, null); // OPEN_ALWAYS, hidden
    if (h == invalid) {
        _ = c.fail(c.lastError());
        return null;
    }
    held.* = h;
    return h;
}

/// Directory slot `slot` closed: its lock file handle goes, and the locks with it.
pub fn release(slot: c.fd_t) void {
    while (!locks_mutex.tryLock()) std.atomic.spinLoopHint();
    defer locks_mutex.unlock();
    const held = &dir_locks[@intCast(slot)];
    if (held.* != invalid) _ = CloseHandle(@ptrFromInt(@as(usize, @bitCast(held.*))));
    held.* = invalid;
}
