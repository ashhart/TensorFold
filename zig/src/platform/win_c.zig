//! libc's surface on Windows (`sys.c`): descriptors over the UCRT, CreateFileW with every share mode, -1 and errno.

const std = @import("std");

pub const fd_t = c_int;
pub const mode_t = c_uint;
pub const off_t = i64;
pub const pid_t = c_int;
pub const time_t = i64;
pub const E = std.c.E;
pub const errno = std.c.errno;
pub const timespec = extern struct { sec: time_t, nsec: c_long };

pub const SEEK = struct {
    pub const SET: c_int = 0;
    pub const CUR: c_int = 1;
    pub const END: c_int = 2;
};
pub const F_OK: c_int = 0;
pub const R_OK: c_int = 4;
pub const W_OK: c_int = 2;
pub const X_OK: c_int = 0; // Windows has no execute bit; existence is the answer
pub const AT = struct {
    pub const FDCWD: fd_t = -100;
    pub const REMOVEDIR: u32 = 0x200;
};
pub const DT = struct {
    pub const UNKNOWN: u8 = 0;
    pub const DIR: u8 = 4;
    pub const REG: u8 = 8;
    pub const LNK: u8 = 10;
};
pub const S = struct {
    pub const IFMT: mode_t = 0o170000;
    pub const IFDIR: mode_t = 0o040000;
    pub const IFREG: mode_t = 0o100000;
    pub const IFLNK: mode_t = 0o120000;
};
pub const RLIM = struct {
    pub const INFINITY: u64 = std.math.maxInt(u64);
};
pub const timeval = extern struct { sec: time_t, usec: c_long };
pub const CLOCK = struct {
    pub const REALTIME: c_int = 0;
    pub const MONOTONIC: c_int = 1;
};

/// Linux's open flags by name; DIRECT asks for unbuffered reads (FILE_FLAG_NO_BUFFERING), the rest map or are moot.
pub const O = packed struct(u32) {
    ACCMODE: enum(u2) { RDONLY = 0, WRONLY = 1, RDWR = 2 } = .RDONLY,
    CREAT: bool = false,
    EXCL: bool = false,
    NOCTTY: bool = false,
    TRUNC: bool = false,
    APPEND: bool = false,
    NONBLOCK: bool = false,
    DSYNC: bool = false,
    ASYNC: bool = false,
    DIRECT: bool = false,
    LARGEFILE: bool = false,
    DIRECTORY: bool = false,
    NOFOLLOW: bool = false,
    NOATIME: bool = false,
    CLOEXEC: bool = false,
    SYNC: bool = false,
    PATH: bool = false,
    TMPFILE: bool = false,
    _: u13 = 0,
};

// ---- Win32 and UCRT entry points ----

const HANDLE = *anyopaque;
const BOOL = c_int;
const invalid_handle: isize = -1;

const OVERLAPPED = extern struct { internal: usize = 0, internal_high: usize = 0, offset: u32, offset_high: u32, event: ?HANDLE = null };
const FILETIME = extern struct { low: u32, high: u32 };
const FindData = extern struct {
    attributes: u32,
    created: FILETIME,
    accessed: FILETIME,
    written: FILETIME,
    size_high: u32,
    size_low: u32,
    reserved0: u32,
    reserved1: u32,
    name: [260]u16,
    alternate: [14]u16,
};

extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?HANDLE) callconv(.winapi) isize;
extern "kernel32" fn ReadFile(h: HANDLE, buf: [*]u8, len: u32, got: ?*u32, ov: ?*OVERLAPPED) callconv(.winapi) BOOL;
extern "kernel32" fn WriteFile(h: HANDLE, buf: [*]const u8, len: u32, put: ?*u32, ov: ?*OVERLAPPED) callconv(.winapi) BOOL;
extern "kernel32" fn GetLastError() callconv(.winapi) u32;
extern "kernel32" fn GetFileAttributesW(name: [*:0]const u16) callconv(.winapi) u32;
extern "kernel32" fn MoveFileExW(from: [*:0]const u16, to: [*:0]const u16, flags: u32) callconv(.winapi) BOOL;
extern "kernel32" fn FindFirstFileW(pattern: [*:0]const u16, data: *FindData) callconv(.winapi) isize;
extern "kernel32" fn FindNextFileW(h: HANDLE, data: *FindData) callconv(.winapi) BOOL;
extern "kernel32" fn FindClose(h: HANDLE) callconv(.winapi) BOOL;
extern "kernel32" fn GetDiskFreeSpaceExW(dir: [*:0]const u16, avail: ?*u64, total: ?*u64, free: ?*u64) callconv(.winapi) BOOL;
extern "kernel32" fn GetTempPathW(len: u32, buf: [*]u16) callconv(.winapi) u32;
extern "kernel32" fn GetModuleFileNameW(module: ?HANDLE, buf: [*]u16, len: u32) callconv(.winapi) u32;
extern "kernel32" fn CreateHardLinkW(new: [*:0]const u16, existing: [*:0]const u16, security: ?*anyopaque) callconv(.winapi) BOOL;
extern "kernel32" fn CreateSymbolicLinkW(link: [*:0]const u16, target: [*:0]const u16, flags: u32) callconv(.winapi) u8;
extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) BOOL;

extern "c" fn _errno() *c_int;
extern "c" fn _open_osfhandle(h: isize, flags: c_int) c_int;
extern "c" fn _get_osfhandle(fd: c_int) isize;
extern "c" fn _close(fd: c_int) c_int;
extern "c" fn _read(fd: c_int, buf: [*]u8, len: c_uint) c_int;
extern "c" fn _write(fd: c_int, buf: [*]const u8, len: c_uint) c_int;
extern "c" fn _lseeki64(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn _commit(fd: c_int) c_int;
extern "c" fn _chsize_s(fd: c_int, size: i64) c_int;
extern "c" fn _wunlink(path: [*:0]const u16) c_int;
extern "c" fn _wrmdir(path: [*:0]const u16) c_int;
extern "c" fn _wmkdir(path: [*:0]const u16) c_int;
extern "c" fn _waccess(path: [*:0]const u16, mode: c_int) c_int;
extern "c" fn _isatty(fd: c_int) c_int;
extern "c" fn _fstat64(fd: c_int, st: *Stat64) c_int;
const ucrt = struct {
    extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;
};

/// libc's getenv, except that HOME, which Windows does not set, falls back to USERPROFILE.
pub fn getenv(name: [*:0]const u8) ?[*:0]u8 {
    if (ucrt.getenv(name)) |v| return v;
    if (std.mem.eql(u8, std.mem.span(name), "HOME")) return ucrt.getenv("USERPROFILE");
    return null;
}
pub extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*std.c.FILE;
pub extern "c" fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *std.c.FILE) usize;
pub extern "c" fn fclose(f: *std.c.FILE) c_int;
pub extern "c" fn free(ptr: ?*anyopaque) void;
pub extern "c" fn malloc(n: usize) ?*anyopaque;

/// The UCRT's struct _stat64.
const Stat64 = extern struct { dev: u32, ino: u16, mode: u16, nlink: i16, uid: i16, gid: i16, rdev: u32, size: i64, atime: i64, mtime: i64, ctime: i64 };

pub fn fail(e: E) c_int {
    _errno().* = @backingInt(e);
    return -1;
}

/// The errno a failed Win32 call means.
fn lastError() E {
    return switch (GetLastError()) {
        2, 3, 15, 123, 161 => .NOENT, // file, path, drive not found; bad name
        5, 32, 33 => .ACCES, // access denied, sharing and lock violations
        80, 183 => .EXIST,
        112 => .NOSPC,
        6 => .BADF,
        145 => .NOTEMPTY,
        267 => .NOTDIR,
        87 => .INVAL,
        else => .IO,
    };
}

// ---- paths ----

const Wide = [32767:0]u16;

/// UTF-8 path as a wide path in `buf`, null with errno if it does not fit; /tmp and /proc/self/exe map as in Cygwin.
fn wide(buf: *Wide, path: []const u8) ?[*:0]const u16 {
    var at: usize = 0;
    var rest = path;
    if (std.mem.eql(u8, path, "/proc/self/exe")) {
        const n = GetModuleFileNameW(null, buf, buf.len - 1);
        if (n == 0 or n >= buf.len - 1) {
            _ = fail(.NAMETOOLONG);
            return null;
        }
        buf[n] = 0;
        return buf[0..n :0].ptr;
    }
    if (std.mem.startsWith(u8, path, "/tmp") and (path.len == 4 or path[4] == '/')) {
        at = GetTempPathW(buf.len - 1, buf); // ends in a separator
        if (at == 0 or at >= buf.len - 1) {
            _ = fail(.NAMETOOLONG);
            return null;
        }
        if (path.len == 4) at -= 1;
        rest = if (path.len > 5) path[5..] else "";
    }
    const n = std.unicode.wtf8ToWtf16Le(buf[at .. buf.len - 1], rest) catch {
        _ = fail(.NAMETOOLONG);
        return null;
    };
    buf[at + n] = 0;
    return buf[0 .. at + n :0].ptr;
}

/// Directory descriptors: open(O.DIRECTORY) remembers the path; openat and unlinkat join onto it.
const dir_base: fd_t = 0x4000_0000;
var dirs_lock: std.atomic.Mutex = .unlocked;
var dir_paths: [256]?[:0]u8 = @splat(null);

fn lockDirs() void {
    while (!dirs_lock.tryLock()) std.atomic.spinLoopHint();
}

fn dirPath(fd: fd_t) ?[:0]const u8 {
    if (fd < dir_base or fd >= dir_base + dir_paths.len) return null;
    lockDirs();
    defer dirs_lock.unlock();
    return dir_paths[@intCast(fd - dir_base)];
}

fn openDir(path: []const u8) c_int {
    var buf: Wide = undefined;
    const w = wide(&buf, path) orelse return -1;
    const attrs = GetFileAttributesW(w);
    if (attrs == 0xFFFF_FFFF) return fail(lastError());
    if (attrs & 0x10 == 0) return fail(.NOTDIR);
    const copy = std.heap.c_allocator.dupeSentinel(u8, path, 0) catch return fail(.NOMEM);
    lockDirs();
    defer dirs_lock.unlock();
    for (&dir_paths, 0..) |*slot, i| if (slot.* == null) {
        slot.* = copy;
        return dir_base + @as(fd_t, @intCast(i));
    };
    std.heap.c_allocator.free(copy);
    return fail(.MFILE);
}

/// `name` under the directory `fd` (or the working directory, or absolute) in `buf`.
fn joinAt(buf: []u8, fd: fd_t, name: [*:0]const u8) ?[:0]const u8 {
    const leaf = std.mem.span(name);
    if (fd == AT.FDCWD or std.fs.path.isAbsolute(leaf)) return std.fmt.bufPrintSentinel(buf, "{s}", .{leaf}, 0) catch null;
    const base = dirPath(fd) orelse {
        _ = fail(.BADF);
        return null;
    };
    return std.fmt.bufPrintSentinel(buf, "{s}\\{s}", .{ base, leaf }, 0) catch {
        _ = fail(.NAMETOOLONG);
        return null;
    };
}

// ---- descriptors ----

/// Variadic as libc's is, so callers pass a mode or not; the mode is moot on NTFS without ACL work.
pub fn open(path: [*:0]const u8, flags: O, ...) callconv(.c) c_int {
    return openFile(path, flags);
}

fn openFile(path: [*:0]const u8, flags: O) c_int {
    const p = std.mem.span(path);
    if (flags.DIRECTORY) return openDir(p);
    var buf: Wide = undefined;
    const w = wide(&buf, p) orelse return -1;
    const generic_read: u32 = 0x8000_0000;
    const generic_write: u32 = 0x4000_0000;
    const rights: u32 = switch (flags.ACCMODE) {
        .RDONLY => generic_read,
        .WRONLY => generic_write,
        .RDWR => generic_read | generic_write,
    };
    const disposition: u32 = if (flags.CREAT and flags.EXCL) 1 // CREATE_NEW
        else if (flags.CREAT and flags.TRUNC) 2 // CREATE_ALWAYS
        else if (flags.CREAT) 4 // OPEN_ALWAYS
        else if (flags.TRUNC) 5 // TRUNCATE_EXISTING
        else 3; // OPEN_EXISTING
    var attrs: u32 = 0x80; // FILE_ATTRIBUTE_NORMAL
    if (flags.DIRECT) attrs |= 0x2000_0000; // FILE_FLAG_NO_BUFFERING
    const h = CreateFileW(w, rights, 0x7, null, disposition, attrs, null); // share read, write and delete
    if (h == invalid_handle) {
        // a directory opened without O.DIRECTORY, as POSIX allows for reading its entries
        if (GetLastError() == 5 and flags.ACCMODE == .RDONLY and GetFileAttributesW(w) & 0x10 != 0 and GetFileAttributesW(w) != 0xFFFF_FFFF) return openDir(p);
        return fail(lastError());
    }
    var crt: c_int = 0x8000; // _O_BINARY
    crt |= switch (flags.ACCMODE) {
        .RDONLY => 0,
        .WRONLY => 1,
        .RDWR => 2,
    };
    if (flags.APPEND) crt |= 0x8;
    const fd = _open_osfhandle(h, crt);
    if (fd < 0) {
        _ = CloseHandle(@ptrFromInt(@as(usize, @bitCast(h))));
        return fail(.MFILE);
    }
    if (fd < append_fds.len) @atomicStore(u8, &append_fds[@intCast(fd)], @intFromBool(flags.APPEND), .release);
    return fd;
}

pub fn openat(dir: fd_t, path: [*:0]const u8, flags: O, ...) callconv(.c) c_int {
    var buf: [4096]u8 = undefined;
    const p = joinAt(&buf, dir, path) orelse return -1;
    return openFile(p.ptr, flags);
}

/// Descriptors opened with O_APPEND, whose writes go to the end of the file whole, as POSIX appends do.
var append_fds: [8192]u8 = @splat(0);

pub fn close(fd: fd_t) c_int {
    if (fd >= 0 and fd < append_fds.len) @atomicStore(u8, &append_fds[@intCast(fd)], 0, .release);
    if (fd >= dir_base and fd < dir_base + dir_paths.len) {
        lockDirs();
        defer dirs_lock.unlock();
        const slot = &dir_paths[@intCast(fd - dir_base)];
        const p = slot.* orelse return fail(.BADF);
        std.heap.c_allocator.free(p);
        slot.* = null;
        return 0;
    }
    return _close(fd);
}

const chunk: usize = 1 << 30;

pub fn read(fd: fd_t, buf: [*]u8, len: usize) isize {
    return _read(fd, buf, @intCast(@min(len, chunk)));
}

pub fn write(fd: fd_t, buf: [*]const u8, len: usize) isize {
    if (fd >= 0 and fd < append_fds.len and @atomicLoad(u8, &append_fds[@intCast(fd)], .acquire) != 0) {
        // WriteFile at the end-of-file offset lands whole; the CRT's seek-then-write lets writers interleave
        const h = osHandle(fd) orelse return -1;
        var ov: OVERLAPPED = .{ .offset = 0xFFFF_FFFF, .offset_high = 0xFFFF_FFFF };
        var put: u32 = 0;
        if (WriteFile(h, buf, @intCast(@min(len, chunk)), &put, &ov) == 0) return fail(lastError());
        return put;
    }
    return _write(fd, buf, @intCast(@min(len, chunk)));
}

fn osHandle(fd: fd_t) ?HANDLE {
    const h = _get_osfhandle(fd);
    if (h == invalid_handle or h == -2) {
        _ = fail(.BADF);
        return null;
    }
    return @ptrFromInt(@as(usize, @bitCast(h)));
}

/// Positional read by an OVERLAPPED offset; end of file reads 0 as pread does.
pub fn pread(fd: fd_t, buf: [*]u8, len: usize, offset: off_t) isize {
    const h = osHandle(fd) orelse return -1;
    var ov: OVERLAPPED = .{ .offset = @truncate(@as(u64, @bitCast(offset))), .offset_high = @truncate(@as(u64, @bitCast(offset)) >> 32) };
    var got: u32 = 0;
    if (ReadFile(h, buf, @intCast(@min(len, chunk)), &got, &ov) == 0) {
        if (GetLastError() == 38) return 0; // ERROR_HANDLE_EOF
        return fail(lastError());
    }
    return got;
}

pub fn pwrite(fd: fd_t, buf: [*]const u8, len: usize, offset: off_t) isize {
    const h = osHandle(fd) orelse return -1;
    var ov: OVERLAPPED = .{ .offset = @truncate(@as(u64, @bitCast(offset))), .offset_high = @truncate(@as(u64, @bitCast(offset)) >> 32) };
    var put: u32 = 0;
    if (WriteFile(h, buf, @intCast(@min(len, chunk)), &put, &ov) == 0) return fail(lastError());
    return put;
}

pub fn lseek(fd: fd_t, offset: off_t, whence: c_int) off_t {
    return _lseeki64(fd, offset, whence);
}

pub fn fsync(fd: fd_t) c_int {
    if (dirPath(fd) != null) return 0; // directory entries are durable once NTFS has them
    return _commit(fd);
}

pub fn ftruncate(fd: fd_t, size: off_t) c_int {
    const rc = _chsize_s(fd, size);
    if (rc != 0) return fail(@fromBackingInt(@intCast(rc)));
    return 0;
}

pub fn isatty(fd: fd_t) c_int {
    return _isatty(fd);
}

pub const Stat = struct {
    dev: u64,
    ino: u64,
    mode: mode_t,
    size: i64,
    mtim: timespec,

    pub fn mtime(s: Stat) timespec {
        return s.mtim;
    }
};

pub fn fstat(fd: fd_t, out: *Stat) c_int {
    if (dirPath(fd) != null) {
        out.* = .{ .dev = 0, .ino = 0, .mode = S.IFDIR | 0o700, .size = 0, .mtim = .{ .sec = 0, .nsec = 0 } };
        return 0;
    }
    var st: Stat64 = undefined;
    if (_fstat64(fd, &st) != 0) return -1;
    out.* = .{ .dev = st.dev, .ino = st.ino, .mode = st.mode, .size = st.size, .mtim = .{ .sec = st.mtime, .nsec = 0 } };
    return 0;
}

// ---- names ----

pub fn unlink(path: [*:0]const u8) c_int {
    var buf: Wide = undefined;
    return _wunlink(wide(&buf, std.mem.span(path)) orelse return -1);
}

pub fn rmdir(path: [*:0]const u8) c_int {
    var buf: Wide = undefined;
    return _wrmdir(wide(&buf, std.mem.span(path)) orelse return -1);
}

pub fn unlinkat(dir: fd_t, path: [*:0]const u8, flags: u32) c_int {
    var buf: [4096]u8 = undefined;
    const p = joinAt(&buf, dir, path) orelse return -1;
    return if (flags & AT.REMOVEDIR != 0) rmdir(p.ptr) else unlink(p.ptr);
}

pub fn mkdir(path: [*:0]const u8, mode: mode_t) c_int {
    _ = mode;
    var buf: Wide = undefined;
    return _wmkdir(wide(&buf, std.mem.span(path)) orelse return -1);
}

/// POSIX rename: replaces an existing target.
pub fn rename(from: [*:0]const u8, to: [*:0]const u8) c_int {
    var a: Wide = undefined;
    var b: Wide = undefined;
    const wa = wide(&a, std.mem.span(from)) orelse return -1;
    const wb = wide(&b, std.mem.span(to)) orelse return -1;
    if (MoveFileExW(wa, wb, 0x1 | 0x8) == 0) return fail(lastError()); // REPLACE_EXISTING | WRITE_THROUGH
    return 0;
}

/// POSIX link: `new` names the same file as `old` (same volume).
pub fn link(old: [*:0]const u8, new: [*:0]const u8) c_int {
    var a: Wide = undefined;
    var b: Wide = undefined;
    const wa = wide(&a, std.mem.span(old)) orelse return -1;
    const wb = wide(&b, std.mem.span(new)) orelse return -1;
    if (CreateHardLinkW(wb, wa, null) == 0) return fail(lastError());
    return 0;
}

pub fn access(path: [*:0]const u8, mode: c_int) c_int {
    var buf: Wide = undefined;
    return _waccess(wide(&buf, std.mem.span(path)) orelse return -1, mode & 6);
}

// ---- directories ----

pub const dirent = extern struct {
    type: u8,
    name: [1024]u8,
};

pub const DIR = struct {
    find: ?HANDLE,
    data: FindData,
    first: bool,
    fd: ?fd_t,
    ent: dirent,
};

pub fn opendir(path: [*:0]const u8) ?*DIR {
    var pattern: [4096]u8 = undefined;
    const p = std.fmt.bufPrint(&pattern, "{s}\\*", .{std.mem.span(path)}) catch {
        _ = fail(.NAMETOOLONG);
        return null;
    };
    var buf: Wide = undefined;
    const w = wide(&buf, p) orelse return null;
    const d = std.heap.c_allocator.create(DIR) catch {
        _ = fail(.NOMEM);
        return null;
    };
    const h = FindFirstFileW(w, &d.data);
    if (h == invalid_handle) {
        const e = lastError();
        std.heap.c_allocator.destroy(d);
        _ = fail(e);
        return null;
    }
    d.* = .{ .find = @ptrFromInt(@as(usize, @bitCast(h))), .data = d.data, .first = true, .fd = null, .ent = undefined };
    return d;
}

/// The stream owns `fd` from here: closedir releases it.
pub fn fdopendir(fd: fd_t) ?*DIR {
    const path = dirPath(fd) orelse {
        _ = fail(.BADF);
        return null;
    };
    const d = opendir(path.ptr) orelse return null;
    d.fd = fd;
    return d;
}

pub fn readdir(d: *DIR) ?*dirent {
    if (d.first) {
        d.first = false;
    } else if (FindNextFileW(d.find.?, &d.data) == 0) {
        return null;
    }
    const name = std.mem.sliceTo(&d.data.name, 0);
    const n = std.unicode.wtf16LeToWtf8(d.ent.name[0 .. d.ent.name.len - 1], name);
    d.ent.name[n] = 0;
    d.ent.type = if (d.data.attributes & 0x400 != 0) DT.LNK else if (d.data.attributes & 0x10 != 0) DT.DIR else DT.REG;
    return &d.ent;
}

pub fn closedir(d: *DIR) c_int {
    if (d.find) |h| _ = FindClose(h);
    if (d.fd) |fd| _ = close(fd);
    std.heap.c_allocator.destroy(d);
    return 0;
}

// ---- process, environment, time: win_clock.zig ----

const clock = @import("win_clock.zig");
pub const getpid = clock.getpid;
pub const getuid = clock.getuid;
pub const setenv = clock.setenv;
pub const unsetenv = clock.unsetenv;
pub const nanosleep = clock.nanosleep;
pub const clock_gettime = clock.clock_gettime;
pub const gettimeofday = clock.gettimeofday;

/// Bytes free to this user under `dir` (statvfs's bavail * frsize).
pub fn diskAvailable(dir: [*:0]const u8) ?u64 {
    var buf: Wide = undefined;
    const w = wide(&buf, std.mem.span(dir)) orelse return null;
    var avail: u64 = 0;
    if (GetDiskFreeSpaceExW(w, &avail, null, null) == 0) return null;
    return avail;
}

/// NTFS keeps access in ACLs; mode bits are not stored, so a mode change succeeds and changes nothing.
pub fn fchmodat(dir: fd_t, path: [*:0]const u8, mode: mode_t, flags: u32) c_int {
    _ = .{ dir, path, mode, flags };
    return 0;
}

/// Symbolic links need Developer Mode or elevation on Windows; this asks for an unprivileged one.
pub fn symlink(target: [*:0]const u8, path: [*:0]const u8) c_int {
    var a: Wide = undefined;
    var b: Wide = undefined;
    const wt = wide(&a, std.mem.span(target)) orelse return -1;
    const wp = wide(&b, std.mem.span(path)) orelse return -1;
    const attrs = GetFileAttributesW(wt);
    const dir: u32 = if (attrs != 0xFFFF_FFFF and attrs & 0x10 != 0) 0x1 else 0;
    if (CreateSymbolicLinkW(wp, wt, dir | 0x2) == 0) return fail(lastError()); // ALLOW_UNPRIVILEGED_CREATE
    return 0;
}

/// Reading a link's target needs its reparse data; nothing on Windows asks for it yet.
pub fn readlink(path: [*:0]const u8, buf: [*]u8, len: usize) isize {
    _ = .{ path, buf, len };
    return fail(.NOSYS);
}
