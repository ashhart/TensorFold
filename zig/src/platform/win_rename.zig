//! Replace an open destination using Windows 10's POSIX rename semantics; keep its existing handles valid.
const HANDLE = *anyopaque;
const Rename = extern struct {
    flags: u32 = 0x1 | 0x2, // REPLACE_IF_EXISTS | POSIX_SEMANTICS
    root: ?HANDLE = null,
    name_bytes: u32 = 0,
    name: [32767]u16 = undefined,
};

extern "kernel32" fn CreateFileW(path: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?HANDLE) callconv(.winapi) isize;
extern "kernel32" fn SetFileInformationByHandle(h: HANDLE, class: c_int, info: *const anyopaque, size: u32) callconv(.winapi) c_int;
extern "kernel32" fn GetFullPathNameW(path: [*:0]const u16, size: u32, out: [*]u16, part: ?*?[*:0]u16) callconv(.winapi) u32;
extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) c_int;

pub fn replace(from: [*:0]const u16, to: [*:0]const u16) bool {
    var info: Rename = .{};
    const n = GetFullPathNameW(to, info.name.len, &info.name, null);
    if (n == 0 or n >= info.name.len) return false;
    info.name_bytes = n * @sizeOf(u16);
    // DELETE access, share all, OPEN_EXISTING, BACKUP_SEMANTICS | OPEN_REPARSE_POINT.
    const raw = CreateFileW(from, 0x0001_0000, 0x7, null, 3, 0x0200_0000 | 0x0020_0000, null);
    if (raw == -1) return false;
    const h: HANDLE = @ptrFromInt(@as(usize, @bitCast(raw)));
    defer _ = CloseHandle(h);
    const size: u32 = @intCast(@offsetOf(Rename, "name") + info.name_bytes + @sizeOf(u16));
    return SetFileInformationByHandle(h, 22, &info, size) != 0; // FileRenameInfoEx
}
