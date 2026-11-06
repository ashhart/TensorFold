//! Positional NVFP4 and FP8 row reads borrow the mapped shard's open file and use the shared exact dequantizers.
const std = @import("std");
const core = @import("core");
const fp8 = @import("cuda_ple_fp8.zig");
pub const Mode = enum { mmap, pread };
pub fn mode(text: ?[]const u8) !Mode {
    const value = text orelse return if (@import("builtin").os.tag == .linux) .pread else .mmap;
    return std.meta.stringToEnum(Mode, value) orelse error.BadPleIoMode;
}
/// A row's bytes in the shard file.
pub const Span = struct { at: u64, len: u64 };
pub const Reader = struct {
    file: std.Io.File,
    io: std.Io,
    codes_at: u64,
    scales_at: u64,
    table: union(enum) { nvfp4: core.nvfp4_rows.Table, fp8: fp8.Table },
    pub fn init(file: *const core.safetensors.File, io: std.Io, table: core.nvfp4_rows.Table) Reader {
        const base = @intFromPtr(file.map.memory.ptr);
        return .{ .file = file.file, .io = io, .codes_at = @intFromPtr(table.codes.ptr) - base, .scales_at = @intFromPtr(table.scales.ptr) - base, .table = .{ .nvfp4 = table } };
    }
    /// An FP8 table has no scale bytes a row.
    pub fn initFp8(file: *const core.safetensors.File, io: std.Io, table: fp8.Table) Reader {
        return .{ .file = file.file, .io = io, .codes_at = @intFromPtr(table.codes.ptr) - @intFromPtr(file.map.memory.ptr), .scales_at = 0, .table = .{ .fp8 = table } };
    }
    /// A row's codes, then its scales (empty for FP8).
    pub fn spans(r: Reader, row: usize) [2]Span {
        const codes: u64, const scales: u64 = switch (r.table) {
            .nvfp4 => |t| .{ t.width / 2, t.width / 16 },
            .fp8 => |t| .{ t.width, 0 },
        };
        return .{ .{ .at = r.codes_at + row * codes, .len = codes }, .{ .at = r.scales_at + row * scales, .len = scales } };
    }
    pub fn read(r: Reader, a: std.mem.Allocator, row: usize, out: []u16) !void {
        const dest = std.mem.sliceAsBytes(out);
        switch (r.table) {
            .nvfp4 => |t| {
                if (row >= t.rows or out.len != t.width) return error.BadNvfp4Rows;
                if (overlap(dest, t.codes) or overlap(dest, t.scales)) return error.AliasedNvfp4Rows;
            },
            .fp8 => |t| {
                if (row >= t.rows or out.len != t.width) return error.BadFp8Rows;
                if (overlap(dest, t.codes)) return error.AliasedFp8Rows;
            },
        }
        const at = r.spans(row);
        var stack: [512]u8 = undefined;
        const size: usize = @intCast(at[0].len + at[1].len);
        const bytes = if (size <= stack.len) stack[0..size] else try a.alloc(u8, size);
        defer if (size > stack.len) a.free(bytes);
        const codes = bytes[0..@intCast(at[0].len)];
        const scales = bytes[codes.len..];
        if (try r.file.readPositionalAll(r.io, codes, at[0].at) != codes.len) return error.PleShortRead;
        if (scales.len != 0 and try r.file.readPositionalAll(r.io, scales, at[1].at) != scales.len) return error.PleShortRead;
        switch (r.table) {
            .nvfp4 => |t| {
                const local: core.nvfp4_rows.Table = .{ .codes = codes, .scales = scales, .width = t.width, .rows = 1, .global = t.global };
                try local.gather(&.{0}, out);
            },
            .fp8 => |t| {
                const local: fp8.Table = .{ .codes = codes, .rows = 1, .width = t.width, .scale = t.scale };
                try local.gather(&.{0}, out);
            },
        }
    }
    pub fn hint(r: Reader, row: usize) void {
        const rows = switch (r.table) {
            inline else => |t| t.rows,
        };
        if (@import("builtin").os.tag != .linux or row >= rows) return;
        const linux = std.os.linux;
        for (r.spans(row)) |s| {
            if (s.len != 0) _ = linux.fadvise(r.file.handle, @intCast(s.at), @intCast(s.len), linux.POSIX_FADV.WILLNEED);
        }
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
    const table: core.nvfp4_rows.Table = .{ .codes = &codes, .scales = &scales, .rows = 256, .width = 16, .global = 1.375 };
    const reader: Reader = .{ .file = file, .io = t.io, .codes_at = 7, .scales_at = 7 + codes.len, .table = .{ .nvfp4 = table } };
    for (0..256) |row| {
        var want: [16]u16 = undefined;
        var got: [16]u16 = undefined;
        try table.gather(&.{row}, &want);
        try reader.read(t.allocator, row, &got);
        try t.expectEqualSlices(u16, &want, &got);
    }
    try t.expectError(error.AliasedNvfp4Rows, reader.read(t.allocator, 0, std.mem.bytesAsSlice(u16, codes[0..32])));
    var bad: [16]u16 = @splat(0xbeef);
    try t.expectError(error.BadNvfp4Rows, reader.read(t.allocator, 256, &bad));
    for (bad) |v| try t.expectEqual(@as(u16, 0xbeef), v);
}

test "PLE positional reader: every FP8 code equals mapped dequantization and hints its codes alone" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var codes: [64 * 16]u8 align(2) = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i % 256);
    var data: [5 + codes.len]u8 = undefined;
    @memset(data[0..5], 0x42);
    @memcpy(data[5..], &codes);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "table", .data = &data });
    const file = try tmp.dir.openFile(t.io, "table", .{});
    defer file.close(t.io);
    const table: fp8.Table = .{ .codes = &codes, .rows = 64, .width = 16, .scale = @bitCast(@as(u32, 0xBE53_0000)) };
    const reader: Reader = .{ .file = file, .io = t.io, .codes_at = 5, .scales_at = 0, .table = .{ .fp8 = table } };
    for (0..64) |row| {
        var want: [16]u16 = undefined;
        var got: [16]u16 = undefined;
        try table.gather(&.{row}, &want);
        try reader.read(t.allocator, row, &got);
        try t.expectEqualSlices(u16, &want, &got);
    }
    try t.expectError(error.AliasedFp8Rows, reader.read(t.allocator, 0, std.mem.bytesAsSlice(u16, codes[0..32])));
    var bad: [16]u16 = @splat(0xbeef);
    try t.expectError(error.BadFp8Rows, reader.read(t.allocator, 64, &bad));
    try t.expectError(error.BadFp8Rows, reader.read(t.allocator, 0, bad[0..15]));
    for (bad) |v| try t.expectEqual(@as(u16, 0xbeef), v);
    const at = reader.spans(9);
    try t.expectEqual(Span{ .at = 5 + 9 * 16, .len = 16 }, at[0]);
    try t.expectEqual(@as(u64, 0), at[1].len);
    const nvfp4: Reader = .{ .file = file, .io = t.io, .codes_at = 5, .scales_at = 900, .table = .{ .nvfp4 = .{ .codes = &codes, .scales = codes[0..64], .rows = 64, .width = 32, .global = 1 } } };
    try t.expectEqualSlices(Span, &.{ .{ .at = 5 + 9 * 16, .len = 16 }, .{ .at = 900 + 9 * 2, .len = 2 } }, &nvfp4.spans(9));
}
