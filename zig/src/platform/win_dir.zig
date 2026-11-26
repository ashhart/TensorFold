//! Native Windows directory handles used directly as the root of relative filesystem operations.
const std = @import("std");
const c = @import("win_c.zig");

pub const HANDLE = *anyopaque;
const FILETIME = extern struct { low: u32, high: u32 };
const FileInfo = extern struct {
    attributes: u32,
    created: FILETIME,
    accessed: FILETIME,
    written: FILETIME,
    volume: u32,
    size_high: u32,
    size_low: u32,
    links: u32,
    index_high: u32,
    index_low: u32,
};
const UnicodeString = extern struct { len: u16, capacity: u16, text: [*]const u16 };
const ObjectAttributes = extern struct {
    len: u32,
    root: ?HANDLE,
    name: *UnicodeString,
    attributes: u32,
    security: ?*anyopaque,
    qos: ?*anyopaque,
};
const IoStatus = extern struct { status: i32, info: usize };
const NetworkInfo = extern struct {
    created: i64,
    accessed: i64,
    written: i64,
    changed: i64,
    allocation: i64,
    size: i64,
    attributes: u32,
    padding: u32,
};
const RenameInfo = extern struct {
    flags: u32,
    root: ?HANDLE,
    name_len: u32,
    name: [4096]u16,
};
const Disposition = extern struct { flags: u32 };
const Entry = extern struct {
    next: u32,
    index: u32,
    created: i64,
    accessed: i64,
    written: i64,
    changed: i64,
    size: i64,
    allocation: i64,
    attributes: u32,
    name_len: u32,
    name: [1]u16,
};

extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?HANDLE) callconv(.winapi) isize;
extern "kernel32" fn GetFileInformationByHandle(h: HANDLE, info: *FileInfo) callconv(.winapi) c_int;
extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) c_int;
extern "ntdll" fn NtCreateFile(handle: *?HANDLE, access: u32, attributes: *ObjectAttributes, status: *IoStatus, allocation: ?*const i64, file_attributes: u32, share: u32, disposition: u32, options: u32, ea: ?*anyopaque, ea_len: u32) callconv(.winapi) i32;
extern "ntdll" fn NtQueryFullAttributesFile(attributes: *ObjectAttributes, info: *NetworkInfo) callconv(.winapi) i32;
extern "ntdll" fn NtSetInformationFile(handle: HANDLE, status: *IoStatus, info: *const anyopaque, len: u32, class: u32) callconv(.winapi) i32;
extern "ntdll" fn NtQueryDirectoryFile(handle: HANDLE, event: ?HANDLE, apc: ?*anyopaque, context: ?*anyopaque, status: *IoStatus, info: *anyopaque, len: u32, class: u32, single: u8, name: ?*UnicodeString, restart: u8) callconv(.winapi) i32;
extern "ntdll" fn RtlNtStatusToDosError(status: i32) callconv(.winapi) u32;

pub fn systemError(code: u32) c.E {
    return switch (code) {
        2, 3, 15, 123, 161 => .NOENT,
        5, 32, 33 => .ACCES,
        80, 183 => .EXIST,
        112 => .NOSPC,
        6 => .BADF,
        145 => .NOTEMPTY,
        267 => .NOTDIR,
        87 => .INVAL,
        else => .IO,
    };
}

fn failed(status: i32) c_int {
    return c.fail(systemError(RtlNtStatusToDosError(status)));
}

fn attributes(root: ?HANDLE, units: []const u16, object: *ObjectAttributes, name: *UnicodeString) void {
    name.* = .{ .len = @intCast(units.len * 2), .capacity = @intCast(units.len * 2), .text = units.ptr };
    object.* = .{ .len = @sizeOf(ObjectAttributes), .root = root, .name = name, .attributes = 0x40, .security = null, .qos = null };
}

fn wideName(buf: *[4096:0]u16, path: [*:0]const u8) ?[]const u16 {
    const n = std.unicode.wtf8ToWtf16Le(buf[0 .. buf.len - 1], std.mem.span(path)) catch {
        _ = c.fail(.NAMETOOLONG);
        return null;
    };
    for (buf[0..n]) |*ch| {
        if (ch.* == '/') ch.* = '\\';
    }
    buf[n] = 0;
    return buf[0..n];
}

fn closeFailed(handle: HANDLE, cause: c.E) void {
    _ = CloseHandle(handle);
    _ = c.fail(cause);
}

/// Open the directory itself, with delete sharing so relative handles survive a move.
pub fn open(path: [*:0]const u16, nofollow: bool) ?HANDLE {
    const reparse: u32 = if (nofollow) 0x0020_0000 else 0;
    const raw = CreateFileW(path, 0x8000_0000, 0x7, null, 3, 0x0200_0000 | reparse, null);
    if (raw == -1) {
        _ = c.fail(c.lastError());
        return null;
    }
    const handle: HANDLE = @ptrFromInt(@as(usize, @bitCast(raw)));
    if (!inspect(handle, nofollow, true)) return null;
    return handle;
}

fn inspect(handle: HANDLE, nofollow: bool, directory: bool) bool {
    var info: FileInfo = undefined;
    if (GetFileInformationByHandle(handle, &info) == 0) {
        const cause = c.lastError();
        _ = CloseHandle(handle);
        _ = c.fail(cause);
        return false;
    }
    if (nofollow and info.attributes & 0x400 != 0) {
        closeFailed(handle, .LOOP);
        return false;
    }
    if (directory and info.attributes & 0x10 == 0) {
        closeFailed(handle, .NOTDIR);
        return false;
    }
    return true;
}

/// Open `path` beneath `root`; no pathname is rescued from a former directory location.
pub fn openAt(root: HANDLE, path: [*:0]const u8, flags: c.O, is_dir: *bool, file_attributes: u32) isize {
    var buf: [4096:0]u16 = undefined;
    const units = wideName(&buf, path) orelse return -1;
    return openObject(root, units, flags, is_dir, file_attributes);
}

/// Open an absolute NT path after the caller's normal Win32 and virtual-path translation.
pub fn openPath(path: [:0]const u16, flags: c.O, is_dir: *bool, file_attributes: u32) isize {
    return openObject(null, path, flags, is_dir, file_attributes);
}

fn openObject(root: ?HANDLE, units: []const u16, flags: c.O, is_dir: *bool, file_attributes: u32) isize {
    var name: UnicodeString = undefined;
    var object: ObjectAttributes = undefined;
    attributes(root, units, &object, &name);
    if (flags.NOFOLLOW) {
        var network: NetworkInfo = undefined;
        if (NtQueryFullAttributesFile(&object, &network) == 0 and network.attributes & 0x400 != 0)
            return c.fail(if (flags.CREAT and flags.EXCL) .EXIST else .LOOP);
    }
    const rights: u32 = (switch (flags.ACCMODE) {
        .RDONLY => @as(u32, 0x8000_0000),
        .WRONLY => @as(u32, 0x4000_0000),
        .RDWR => @as(u32, 0xC000_0000),
    }) | 0x0010_0000;
    const disposition: u32 = if (flags.CREAT and flags.EXCL) 2 else if (flags.CREAT and flags.TRUNC) 5 else if (flags.CREAT) 3 else if (flags.TRUNC) 4 else 1;
    var options: u32 = 0x20;
    if (flags.DIRECTORY) options |= 0x1;
    if (flags.NOFOLLOW) options |= 0x0020_0000;
    if (flags.DIRECT) options |= 0x8;
    var handle: ?HANDLE = null;
    var status: IoStatus = undefined;
    const rc = NtCreateFile(&handle, rights, &object, &status, null, file_attributes, 0x7, disposition, options, null, 0);
    if (rc < 0) return failed(rc);
    const h = handle.?;
    var info: FileInfo = undefined;
    if (GetFileInformationByHandle(h, &info) == 0) {
        const cause = c.lastError();
        _ = CloseHandle(h);
        _ = c.fail(cause);
        return -1;
    }
    if (flags.NOFOLLOW and info.attributes & 0x400 != 0) {
        closeFailed(h, if (flags.CREAT and flags.EXCL) .EXIST else .LOOP);
        return -1;
    }
    is_dir.* = info.attributes & 0x10 != 0;
    if (flags.DIRECTORY and !is_dir.*) {
        closeFailed(h, .NOTDIR);
        return -1;
    }
    if (is_dir.* and !flags.DIRECTORY and flags.ACCMODE != .RDONLY) {
        closeFailed(h, .ACCES);
        return -1;
    }
    return @bitCast(@intFromPtr(h));
}

pub fn mkdirAt(root: HANDLE, path: [*:0]const u8) c_int {
    var buf: [4096:0]u16 = undefined;
    const units = wideName(&buf, path) orelse return -1;
    var name: UnicodeString = undefined;
    var object: ObjectAttributes = undefined;
    attributes(root, units, &object, &name);
    var handle: ?HANDLE = null;
    var status: IoStatus = undefined;
    const rc = NtCreateFile(&handle, 0x4010_0000, &object, &status, null, 0x80, 0x7, 2, 0x21, null, 0);
    if (rc < 0) return failed(rc);
    _ = CloseHandle(handle.?);
    return 0;
}

fn removable(root: HANDLE, path: [*:0]const u8, handle: *HANDLE, name_buf: *[4096:0]u16) c_int {
    const units = wideName(name_buf, path) orelse return -1;
    var name: UnicodeString = undefined;
    var object: ObjectAttributes = undefined;
    attributes(root, units, &object, &name);
    var opened: ?HANDLE = null;
    var status: IoStatus = undefined;
    const rc = NtCreateFile(&opened, 0x0011_0000, &object, &status, null, 0x80, 0x7, 1, 0x0020_0020, null, 0);
    if (rc < 0) return failed(rc);
    handle.* = opened.?;
    return 0;
}

pub fn unlinkAt(root: HANDLE, path: [*:0]const u8, removedir: bool) c_int {
    var buf: [4096:0]u16 = undefined;
    var handle: HANDLE = undefined;
    if (removable(root, path, &handle, &buf) != 0) return -1;
    var info: FileInfo = undefined;
    if (GetFileInformationByHandle(handle, &info) == 0) {
        const cause = c.lastError();
        _ = CloseHandle(handle);
        _ = c.fail(cause);
        return -1;
    }
    const reparse = info.attributes & 0x400 != 0;
    const is_dir = info.attributes & 0x10 != 0;
    if (reparse and removedir) {
        closeFailed(handle, .NOTDIR);
        return -1;
    }
    if (!reparse and removedir != is_dir) {
        closeFailed(handle, if (removedir) .NOTDIR else .ISDIR);
        return -1;
    }
    var status: IoStatus = undefined;
    const disposition: Disposition = .{ .flags = 0x3 }; // DELETE | POSIX_SEMANTICS
    // Class 64 is FileDispositionInformationEx, which applies DELETE with POSIX semantics.
    const rc = NtSetInformationFile(handle, &status, &disposition, @sizeOf(Disposition), 64);
    _ = CloseHandle(handle);
    return if (rc < 0) failed(rc) else 0;
}

pub fn renameAt(old_root: HANDLE, old: [*:0]const u8, new_root: HANDLE, new: [*:0]const u8) c_int {
    var old_buf: [4096:0]u16 = undefined;
    var handle: HANDLE = undefined;
    if (removable(old_root, old, &handle, &old_buf) != 0) return -1;
    defer _ = CloseHandle(handle);
    var buf: [4096:0]u16 = undefined;
    const units = wideName(&buf, new) orelse return -1;
    var info: RenameInfo = undefined;
    info.flags = 0x3; // REPLACE_IF_EXISTS | POSIX_SEMANTICS
    info.root = new_root;
    info.name_len = @intCast(units.len * 2);
    @memcpy(info.name[0..units.len], units);
    var status: IoStatus = undefined;
    const len: u32 = @intCast(@max(@offsetOf(RenameInfo, "name") + units.len * 2, 24));
    const rc = NtSetInformationFile(handle, &status, &info, len, 65); // FileRenameInformationEx
    return if (rc < 0) failed(rc) else 0;
}

pub const Stream = struct {
    root: ?HANDLE = null,
    offset: usize = 0,
    length: usize = 0,
    restart: bool = true,
    buffer: [16384]u8 align(8) = undefined,
};

/// One native directory entry, or zero at the end of the stream.
pub fn next(stream: *Stream, name: []u8, kind: *u8) c_int {
    while (true) {
        if (stream.offset < stream.length) {
            const entry: *align(1) const Entry = @ptrCast(stream.buffer[stream.offset..].ptr);
            if (stream.offset + @offsetOf(Entry, "name") > stream.length or entry.next % 8 != 0) return c.fail(.IO);
            const bytes: usize = entry.name_len;
            const start = stream.offset + @offsetOf(Entry, "name");
            if (bytes % 2 != 0 or start + bytes > stream.length) return c.fail(.IO);
            const units: [*]const u16 = @ptrCast(@alignCast(&entry.name));
            const n = std.unicode.wtf16LeToWtf8(name, units[0 .. bytes / 2]);
            kind.* = if (entry.attributes & 0x400 != 0) c.DT.LNK else if (entry.attributes & 0x10 != 0) c.DT.DIR else c.DT.REG;
            stream.offset = if (entry.next == 0) stream.length else stream.offset + entry.next;
            return @intCast(n);
        }
        if (!stream.restart and stream.length == 0) return 0;
        var status: IoStatus = undefined;
        const rc = NtQueryDirectoryFile(stream.root.?, null, null, null, &status, &stream.buffer, stream.buffer.len, 1, 0, null, @intFromBool(stream.restart));
        stream.restart = false;
        stream.offset = 0;
        stream.length = 0;
        if (rc == @as(i32, @bitCast(@as(u32, 0x8000_0006)))) return 0; // STATUS_NO_MORE_FILES
        if (rc < 0) return failed(rc);
        stream.length = status.info;
        if (stream.length == 0) return 0;
    }
}

pub fn close(handle: HANDLE) void {
    _ = CloseHandle(handle);
}
