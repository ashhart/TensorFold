//! Positional NVFP4 row reads borrow the mapped shard's open file and use the shared exact dequantizer.
const std = @import("std");
const core = @import("core");
pub const Mode = enum { mmap, pread };
pub fn mode(text: ?[]const u8) !Mode {
    const value = text orelse return if (@import("builtin").os.tag == .linux) .pread else .mmap;
    return std.meta.stringToEnum(Mode, value) orelse error.BadPleIoMode;
}
pub const Reader = struct {
    file: std.Io.File,
    io: std.Io,
    codes_at: u64,
    scales_at: u64,
    pub fn init(file: *const core.safetensors.File, io: std.Io, table: core.nvfp4_rows.Table) Reader {
        const base = @intFromPtr(file.map.memory.ptr);
        return .{ .file = file.file, .io = io, .codes_at = @intFromPtr(table.codes.ptr) - base, .scales_at = @intFromPtr(table.scales.ptr) - base };
    }
    pub fn read(r: Reader, a: std.mem.Allocator, table: core.nvfp4_rows.Table, row: usize, out: []u16) !void {
        if (row >= table.rows or out.len != table.width) return error.BadNvfp4Rows;
        if (overlap(std.mem.sliceAsBytes(out), table.codes) or overlap(std.mem.sliceAsBytes(out), table.scales)) return error.AliasedNvfp4Rows;
        var stack: [512]u8 = undefined;
        const size = table.width / 2 + table.width / 16;
        const bytes = if (size <= stack.len) stack[0..size] else try a.alloc(u8, size);
        defer if (size > stack.len) a.free(bytes);
        const codes = bytes[0 .. table.width / 2];
        const scales = bytes[codes.len..];
        if (try r.file.readPositionalAll(r.io, codes, r.codes_at + row * codes.len) != codes.len) return error.PleShortRead;
        if (try r.file.readPositionalAll(r.io, scales, r.scales_at + row * scales.len) != scales.len) return error.PleShortRead;
        const local: core.nvfp4_rows.Table = .{ .codes = codes, .scales = scales, .width = table.width, .rows = 1, .global = table.global };
        try local.gather(&.{0}, out);
    }
    pub fn hint(r: Reader, table: core.nvfp4_rows.Table, row: usize) void {
        if (@import("builtin").os.tag != .linux or row >= table.rows) return;
        const linux = std.os.linux;
        _ = linux.fadvise(r.file.handle, @intCast(r.codes_at + row * (table.width / 2)), @intCast(table.width / 2), linux.POSIX_FADV.WILLNEED);
        _ = linux.fadvise(r.file.handle, @intCast(r.scales_at + row * (table.width / 16)), @intCast(table.width / 16), linux.POSIX_FADV.WILLNEED);
    }
};

test "PLE positional reader: explicit modes and invalid settings" {
    try std.testing.expectEqual(Mode.mmap, try mode("mmap"));
    try std.testing.expectEqual(Mode.pread, try mode("pread"));
    for ([_][]const u8{ "", "auto", "0" }) |text| try std.testing.expectError(error.BadPleIoMode, mode(text));
}

fn overlap(a: []const u8, b: []const u8) bool {
    if (a.len == 0 or b.len == 0) return false;
    const x = @intFromPtr(a.ptr);
    const y = @intFromPtr(b.ptr);
    return if (x <= y) y - x < a.len else x - y < b.len;
}

test "PLE positional reader: every code and scale encoding equals mapped dequantization" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var codes: [256 * 8]u8 align(2) = undefined;
    var scales: [256]u8 = undefined;
    for (0..256) |row| {
        scales[row] = @intCast(row);
        for (0..8) |col| codes[row * 8 + col] = @intCast(2 * col + (2 * col + 1) * 16);
    }
    var data: [7 + codes.len + scales.len]u8 = undefined;
    @memset(data[0..7], 0x42);
    @memcpy(data[7..][0..codes.len], &codes);
    @memcpy(data[7 + codes.len ..], &scales);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "table", .data = &data });
    const file = try tmp.dir.openFile(t.io, "table", .{});
    defer file.close(t.io);
    const reader: Reader = .{ .file = file, .io = t.io, .codes_at = 7, .scales_at = 7 + codes.len };
    const table: core.nvfp4_rows.Table = .{ .codes = &codes, .scales = &scales, .rows = 256, .width = 16, .global = 1.375 };
    for (0..256) |row| {
        var want: [16]u16 = undefined;
        var got: [16]u16 = undefined;
        try table.gather(&.{row}, &want);
        try reader.read(t.allocator, table, row, &got);
        try t.expectEqualSlices(u16, &want, &got);
    }
    try t.expectError(error.AliasedNvfp4Rows, reader.read(t.allocator, table, 0, std.mem.bytesAsSlice(u16, codes[0..32])));
    var bad: [16]u16 = @splat(0xbeef);
    try t.expectError(error.BadNvfp4Rows, reader.read(t.allocator, table, 256, &bad));
    for (bad) |v| try t.expectEqual(@as(u16, 0xbeef), v);
}
