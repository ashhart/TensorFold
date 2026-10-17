//! Windows DLL names keep system search; paths with separators are normalized and resolved before loading.
const std = @import("std");
pub const Error = error{ FileNotFound, NameTooLong, InvalidWtf8 };
pub fn open(comptime Api: type, path: []const u8) Error!Api.Handle {
    var buf: [1024:0]u16 = undefined;
    const need = std.unicode.calcWtf16LeLen(path) catch return error.InvalidWtf8;
    if (need >= buf.len) return error.NameTooLong;
    const n = std.unicode.wtf8ToWtf16Le(buf[0 .. buf.len - 1], path) catch return error.InvalidWtf8;
    buf[n] = 0;
    if (std.mem.indexOfAny(u8, path, "/\\") != null) {
        for (buf[0..n]) |*ch| if (ch.* == '/') {
            ch.* = '\\';
        };
        var full: [1024:0]u16 = undefined;
        const count = Api.fullPath(buf[0..n :0], &full);
        if (count > 0 and count < full.len) {
            full[count] = 0;
            return Api.load(full[0..count :0]) orelse error.FileNotFound;
        }
    }
    return Api.load(buf[0..n :0]) orelse error.FileNotFound;
}
