//! One safetensors file mapped read-only, its tensors found by name in the header's text.
const std = @import("std");

const c = struct {
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
    extern "c" fn close(fd: c_int) c_int;
    extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
    extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
    extern "c" fn munmap(addr: ?*anyopaque, len: usize) c_int;
};

/// Where a tensor sits in the data that follows the header.
pub const Span = struct { rows: usize, cols: usize, start: usize, end: usize };

fn numbers(text: []const u8, comptime key: []const u8, out: []usize) !usize {
    const at = std.mem.indexOf(u8, text, key) orelse return error.BadHeader;
    const end = std.mem.indexOfScalarPos(u8, text, at, ']') orelse return error.BadHeader;
    var it = std.mem.tokenizeAny(u8, text[at + key.len .. end], ", ");
    var n: usize = 0;
    while (it.next()) |word| : (n += 1) {
        if (n == out.len) return error.BadHeader;
        out[n] = try std.fmt.parseInt(usize, word, 10);
    }
    return n;
}

/// The bf16 tensor `name` in a safetensors header: a vector counts as one row.
pub fn locate(header: []const u8, name: []const u8) !Span {
    var key: [160]u8 = undefined;
    const quoted = try std.fmt.bufPrint(&key, "\"{s}\":{{", .{name});
    const at = std.mem.indexOf(u8, header, quoted) orelse return error.MissingTensor;
    const end = std.mem.indexOfScalarPos(u8, header, at, '}') orelse return error.BadHeader;
    const text = header[at..end];
    if (std.mem.indexOf(u8, text, "\"BF16\"") == null) return error.WrongDType;
    var shape: [2]usize = undefined;
    var span: [2]usize = undefined;
    const dims = try numbers(text, "\"shape\":[", &shape);
    if (try numbers(text, "\"data_offsets\":[", &span) != 2 or dims == 0) return error.BadHeader;
    const rows = if (dims == 2) shape[0] else 1;
    const cols = shape[dims - 1];
    if (span[1] < span[0] or span[1] - span[0] != rows * cols * 2) return error.BadHeader;
    return .{ .rows = rows, .cols = cols, .start = span[0], .end = span[1] };
}

pub const Checkpoint = struct {
    bytes: []const u8,
    header: []const u8,
    data: usize,

    pub const Entry = struct { rows: usize, cols: usize, bytes: []const u8 };

    pub fn open(path: [*:0]const u8) !Checkpoint {
        const fd = c.open(path, 0);
        if (fd < 0) return error.CannotOpen;
        defer _ = c.close(fd);
        const size = c.lseek(fd, 0, 2);
        if (size < 16) return error.CannotOpen;
        const len: usize = @intCast(size);
        const at = c.mmap(null, len, 1, 2, fd, 0) orelse return error.CannotMap;
        if (@intFromPtr(at) == std.math.maxInt(usize)) return error.CannotMap;
        const bytes = @as([*]const u8, @ptrCast(at))[0..len];
        const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        if (n + 8 > len) return error.BadHeader;
        return .{ .bytes = bytes, .header = bytes[8 .. 8 + n], .data = 8 + n };
    }

    pub fn deinit(self: *Checkpoint) void {
        _ = c.munmap(@ptrCast(@constCast(self.bytes.ptr)), self.bytes.len);
    }

    pub fn bf16(self: *const Checkpoint, name: []const u8) !Entry {
        const span = locate(self.header, name) catch |e| {
            std.debug.print("tensor {s}: {t}\n", .{ name, e });
            return e;
        };
        if (self.data + span.end > self.bytes.len) return error.BadHeader;
        return .{ .rows = span.rows, .cols = span.cols, .bytes = self.bytes[self.data + span.start .. self.data + span.end] };
    }
};

test "a matrix and a vector are found by name" {
    const header = "{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[3,4],\"data_offsets\":[8,32]},\"a.norm\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
    const w = try locate(header, "a.weight");
    try std.testing.expectEqual(Span{ .rows = 3, .cols = 4, .start = 8, .end = 32 }, w);
    const n = try locate(header, "a.norm");
    try std.testing.expectEqual(Span{ .rows = 1, .cols = 4, .start = 0, .end = 8 }, n);
}

test "a missing name, another dtype and a size that does not match are refused" {
    const header = "{\"f.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"b.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,6]}}";
    try std.testing.expectError(error.MissingTensor, locate(header, "x.weight"));
    try std.testing.expectError(error.WrongDType, locate(header, "f.weight"));
    try std.testing.expectError(error.BadHeader, locate(header, "b.weight"));
}
