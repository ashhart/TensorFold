//! File identity without following a reparse point: volume serial and the full 64-bit file index.
const HANDLE = *anyopaque;
const FILETIME = extern struct { low: u32, high: u32 };
const Info = extern struct {
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

pub const Id = struct { dev: u32, ino: u64, dir: bool, reparse: bool };

extern "kernel32" fn GetFileInformationByHandle(h: HANDLE, out: *Info) callconv(.winapi) c_int;
extern "kernel32" fn CreateFileW(path: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?HANDLE) callconv(.winapi) isize;
extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) c_int;

pub fn handle(h: isize) ?Id {
    if (h == -1) return null;
    var info: Info = undefined;
    if (GetFileInformationByHandle(@ptrFromInt(@as(usize, @bitCast(h))), &info) == 0) return null;
    return .{
        .dev = info.volume,
        .ino = (@as(u64, info.index_high) << 32) | info.index_low,
        .dir = info.attributes & 0x10 != 0,
        .reparse = info.attributes & 0x400 != 0,
    };
}

pub fn path(name: [*:0]const u16) ?Id {
    // READ_ATTRIBUTES, share all, OPEN_EXISTING, OPEN_REPARSE_POINT | BACKUP_SEMANTICS.
    const h = CreateFileW(name, 0x80, 0x7, null, 3, 0x0020_0000 | 0x0200_0000, null);
    if (h == -1) return null;
    defer _ = CloseHandle(@ptrFromInt(@as(usize, @bitCast(h))));
    return handle(h);
}
