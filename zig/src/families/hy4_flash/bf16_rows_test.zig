//! Focused host fixture for the BF16 Engram direct reader: crossing, tail, truncated data.
const std = @import("std");
const linux = std.os.linux;
const bf16 = @import("bf16_rows.zig");
const engram = @import("engram.zig");

test {
    _ = @import("bf16_rows.zig");
}

const Call = struct { off: u64, len: usize };

/// In-memory file accepting only aligned positioned reads, recording them and returning offset-derived bytes.
const Mock = struct {
    bytes: []const u8,
    calls: [64]Call = undefined,
    n: usize = 0,
    misaligned: bool = false,
    refuse: bool = false,

    fn pread(ctx: *anyopaque, _: linux.fd_t, buf: [*]u8, len: usize, off: i64) usize {
        const self: *Mock = @ptrCast(@alignCast(ctx));
        const o: u64 = @bitCast(off);
        if (o % bf16.alignment != 0 or len % bf16.alignment != 0 or @intFromPtr(buf) % bf16.alignment != 0) self.misaligned = true;
        if (self.n < self.calls.len) {
            self.calls[self.n] = .{ .off = o, .len = len };
            self.n += 1;
        }
        if (self.refuse) return @bitCast(-@as(isize, @backingInt(linux.E.INVAL)));
        if (o >= self.bytes.len) return 0;
        const want = @min(len, self.bytes.len - @as(usize, @intCast(o)));
        @memcpy(buf[0..want], self.bytes[@intCast(o)..][0..want]);
        return want;
    }
};

fn geometry(bytes: []const u8, data_start: u64, count: u64, mock: *Mock) bf16.Geometry {
    return .{ .fd = 0, .size = bytes.len, .data_start = data_start, .count = count, .ctx = mock, .read = Mock.pread };
}

/// Each row stores its index in its first two bytes and a per-row pattern after.
fn synth(a: std.mem.Allocator, size: usize, data_start: u64, count: u64) ![]u8 {
    const bytes = try a.alloc(u8, size);
    @memset(bytes, 0);
    for (0..count) |r| {
        const off: usize = @intCast(data_start + @as(u64, r) * bf16.row_bytes);
        if (off + bf16.row_bytes > size) break;
        for (0..bf16.row_bytes) |k| bytes[off + k] = @intCast((r *% 31 +% k) % 251);
        std.mem.writeInt(u16, bytes[off..][0..2], @intCast(r), .little);
    }
    return bytes;
}

fn fixtureHash() engram.Layer {
    const offsets = [engram.columns]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    return .{ .multipliers = .{ 3, 5, 7 }, .primes = @splat(17), .offsets = offsets };
}

fn fixtureMap() engram.TokenMap {
    const lookup = [4]u32{ 0, 1, 2, 3 };
    return .{ .bytes = std.mem.asBytes(&lookup), .pad = 0 };
}

fn checkOutput(bytes: []const u8, data_start: u64, map: engram.TokenMap, hash: engram.Layer, windows: []const [engram.order]u32, out: []const u16) !void {
    for (windows, 0..) |window, t| {
        for (hash.hash(map, window), 0..) |row, head| {
            const at: usize = @intCast(data_start + @as(u64, row) * bf16.row_bytes);
            for (0..bf16.width) |k| {
                const want = std.mem.readInt(u16, bytes[at + 2 * k ..][0..2], .little);
                try std.testing.expectEqual(want, out[(t * engram.columns + head) * bf16.width + k]);
            }
        }
    }
}

test "a row that crosses a 4 KiB boundary is read in aligned extents and lands in token/head order" {
    const a = std.testing.allocator;
    const data_start: u64 = bf16.alignment - 100; // row 0 spans [3996, 4188)
    const count: u64 = 32;
    const bytes = try synth(a, 3 * bf16.alignment, data_start, count);
    defer a.free(bytes);
    var mock: Mock = .{ .bytes = bytes };
    const g = geometry(bytes, data_start, count, &mock);
    const windows = [3][engram.order]u32{ .{ 0, 0, 0 }, .{ 2, 1, 3 }, .{ 3, 2, 1 } };
    const out = try a.alloc(u16, windows.len * bf16.tokenElements());
    defer a.free(out);
    try bf16.gatherRows(a, g, fixtureMap(), fixtureHash(), &windows, out);
    try std.testing.expect(!mock.misaligned);
    try std.testing.expect(mock.n > 0);
    var crossed = false;
    for (mock.calls[0..mock.n]) |c| {
        if (c.len > bf16.alignment) crossed = true;
    }
    try std.testing.expect(crossed); // evidence the boundary was spanned, not skipped
    try checkOutput(bytes, data_start, fixtureMap(), fixtureHash(), &windows, out);
}

test "a tail row ending exactly at EOF is read from a short final block without a fallback" {
    const a = std.testing.allocator;
    const data_start: u64 = 8; // unaligned, as a safetensors payload usually is
    const count: u64 = 5;
    const size: usize = @intCast(data_start + count * bf16.row_bytes); // 968, inside one block
    const bytes = try synth(a, size, data_start, count);
    defer a.free(bytes);
    var mock: Mock = .{ .bytes = bytes };
    const g = geometry(bytes, data_start, count, &mock);
    const windows = [1][engram.order]u32{.{ 0, 0, 0 }}; // selects rows 0..15, all present (0..4 used)
    const out = try a.alloc(u16, windows.len * bf16.tokenElements());
    defer a.free(out);
    // limit the fixture hash's rows to the five the file holds
    const hash = fixtureHash();
    var limited = hash;
    limited.offsets = @splat(0);
    try bf16.gatherRows(a, g, fixtureMap(), limited, &windows, out);
    try std.testing.expect(!mock.misaligned);
    try std.testing.expectEqual(@as(usize, 1), mock.n);
    try std.testing.expectEqual(@as(u64, 0), mock.calls[0].off);
    try std.testing.expectEqual(@as(usize, bf16.alignment), mock.calls[0].len);
    try checkOutput(bytes, data_start, fixtureMap(), limited, &windows, out);
}

test "a truncated row past EOF is refused before any read" {
    const a = std.testing.allocator;
    const data_start: u64 = 8;
    const count: u64 = 5;
    const size: usize = @intCast(data_start + count * bf16.row_bytes - 50); // last row cut short
    const bytes = try synth(a, size, data_start, count);
    defer a.free(bytes);
    var mock: Mock = .{ .bytes = bytes };
    const g = geometry(bytes, data_start, count, &mock);
    const windows = [1][engram.order]u32{.{ 0, 0, 0 }};
    const out = try a.alloc(u16, windows.len * bf16.tokenElements());
    defer a.free(out);
    try std.testing.expectError(error.ShortRow, bf16.gatherRows(a, g, fixtureMap(), fixtureHash(), &windows, out));
    try std.testing.expectEqual(@as(usize, 0), mock.n); // refused in planning, not after I/O
}

test "one plan executes byte-identical reads twice, and rebuilds byte-identically" {
    const a = std.testing.allocator;
    const data_start: u64 = bf16.alignment - 64;
    const count: u64 = 32;
    const bytes = try synth(a, 4 * bf16.alignment, data_start, count);
    defer a.free(bytes);
    const windows = [2][engram.order]u32{ .{ 1, 2, 3 }, .{ 3, 1, 2 } };
    const map = fixtureMap();
    const hash = fixtureHash();

    var plan_mock: Mock = .{ .bytes = bytes };
    var rebuild_mock: Mock = .{ .bytes = bytes };
    var plan = try bf16.buildPlan(a, geometry(bytes, data_start, count, &plan_mock), map, hash, &windows);
    defer plan.deinit(a);
    var rebuilt = try bf16.buildPlan(a, geometry(bytes, data_start, count, &rebuild_mock), map, hash, &windows);
    defer rebuilt.deinit(a);
    try std.testing.expectEqualSlices(u32, plan.rows, rebuilt.rows);
    try std.testing.expectEqualSlices(usize, plan.at, rebuilt.at);
    try std.testing.expectEqual(plan.arena_len, rebuilt.arena_len);
    try std.testing.expectEqual(plan.ext.len, rebuilt.ext.len);
    for (plan.ext, rebuilt.ext) |x, y| {
        try std.testing.expectEqual(x.off, y.off);
        try std.testing.expectEqual(x.len, y.len);
        try std.testing.expectEqual(x.at, y.at);
    }

    var m1: Mock = .{ .bytes = bytes };
    var m2: Mock = .{ .bytes = bytes };
    const o1 = try a.alloc(u16, windows.len * bf16.tokenElements());
    defer a.free(o1);
    const o2 = try a.alloc(u16, windows.len * bf16.tokenElements());
    defer a.free(o2);
    try bf16.execute(a, geometry(bytes, data_start, count, &m1), plan, o1);
    try bf16.execute(a, geometry(bytes, data_start, count, &m2), plan, o2);
    try std.testing.expectEqualSlices(u16, o1, o2);
    try std.testing.expectEqual(m1.n, m2.n);
    try std.testing.expectEqualSlices(Call, m1.calls[0..m1.n], m2.calls[0..m2.n]);
    try std.testing.expect(!m1.misaligned and !m2.misaligned);
}

test "an O_DIRECT read refusal is a hard error, never a buffered retry" {
    const a = std.testing.allocator;
    const bytes = try synth(a, 2 * bf16.alignment, 0, 32);
    defer a.free(bytes);
    var mock: Mock = .{ .bytes = bytes, .refuse = true };
    const g = geometry(bytes, 0, 32, &mock);
    const windows = [1][engram.order]u32{.{ 0, 0, 0 }};
    const out = try a.alloc(u16, windows.len * bf16.tokenElements());
    defer a.free(out);
    try std.testing.expectError(error.ReadFailed, bf16.gatherRows(a, g, fixtureMap(), fixtureHash(), &windows, out));
    try std.testing.expect(mock.n >= 1); // the read was attempted; no fallback path exists
}

/// Build a one-tensor BF16 safetensors image; `pad` pushes the header past one block.
fn safetensorsImage(a: std.mem.Allocator, count: u64, pad: usize) !struct { bytes: []u8, header_end: usize } {
    const data_len: usize = @intCast(count * bf16.row_bytes);
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    var first_buf: [256]u8 = undefined;
    const first = try std.fmt.bufPrint(&first_buf, "{{\"emb\":{{\"dtype\":\"BF16\",\"shape\":[{d},96],\"data_offsets\":[0,{d}]}}", .{ count, data_len });
    try header.appendSlice(a, first);
    if (pad > 0) {
        try header.appendSlice(a, ",\"pad\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,0],\"note\":\"");
        try header.appendNTimes(a, 'x', pad);
        try header.appendSlice(a, "\"}");
    }
    try header.append(a, '}');
    const header_end = 8 + header.items.len;
    var file = try a.alloc(u8, header_end + data_len);
    std.mem.writeInt(u64, file[0..8], @intCast(header.items.len), .little);
    @memcpy(file[8..header_end], header.items);
    for (0..count) |r| {
        const off = header_end + r * bf16.row_bytes;
        for (0..bf16.row_bytes) |k| file[off + k] = @intCast((r *% 31 +% k) % 251);
        std.mem.writeInt(u16, file[off..][0..2], @intCast(r), .little);
    }
    return .{ .bytes = file, .header_end = header_end };
}

test "openDirect reads a real safetensors table through O_DIRECT (small and multi-block header)" {
    const a = std.testing.allocator;
    const windows = [2][engram.order]u32{ .{ 0, 0, 0 }, .{ 2, 1, 3 } };
    const map = fixtureMap();
    const hash = fixtureHash();
    for ([_]usize{ 0, 6000 }) |pad| {
        const image = try safetensorsImage(a, 32, pad);
        defer a.free(image.bytes);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const io = std.testing.io;
        try tmp.dir.writeFile(io, .{ .sub_path = "emb.safetensors", .data = image.bytes });
        const real = try tmp.dir.realPathFileAlloc(io, "emb.safetensors", a);
        defer a.free(real);
        const path = try a.dupeSentinel(u8, real, 0);
        defer a.free(path);
        var table = bf16.openDirect(a, path, "emb", map, hash, 32) catch |err| switch (err) {
            error.OpenFailed, error.ReadFailed => return, // file system refuses O_DIRECT
            else => return err,
        };
        defer table.close();
        const out = try a.alloc(u16, windows.len * bf16.tokenElements());
        defer a.free(out);
        try table.gather(a, &windows, out);
        try checkOutput(image.bytes, @intCast(image.header_end), map, hash, &windows, out);
    }
}

fn rawImage(a: std.mem.Allocator, json: []const u8, data_len: usize) ![]u8 {
    const header_end = 8 + json.len;
    const file = try a.alloc(u8, header_end + data_len);
    std.mem.writeInt(u64, file[0..8], @intCast(json.len), .little);
    @memcpy(file[8..header_end], json);
    @memset(file[header_end..], 0);
    return file;
}

fn expectBadTable(a: std.mem.Allocator, bytes: []const u8) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "bad.safetensors", .data = bytes });
    const real = try tmp.dir.realPathFileAlloc(io, "bad.safetensors", a);
    defer a.free(real);
    const path = try a.dupeSentinel(u8, real, 0);
    defer a.free(path);
    var table = bf16.openDirect(a, path, "emb", fixtureMap(), fixtureHash(), 1) catch |err| switch (err) {
        error.BadTable => return,
        error.OpenFailed, error.ReadFailed => return, // file system refuses O_DIRECT
        else => return err,
    };
    table.close();
    return error.TestUnexpectedResult;
}

test "openDirect refuses negative offsets, wrong shape, header overflow and truncated payload" {
    const a = std.testing.allocator;
    const cases = [_][]const u8{
        "{\"emb\":{\"dtype\":\"BF16\",\"shape\":[1,96],\"data_offsets\":[-1,191]}}",
        "{\"emb\":{\"dtype\":\"BF16\",\"shape\":[1,96],\"data_offsets\":[0,-192]}}",
        "{\"emb\":{\"dtype\":\"BF16\",\"shape\":[1,95],\"data_offsets\":[0,192]}}",
        "{\"emb\":{\"dtype\":\"BF16\",\"shape\":[1,96],\"data_offsets\":[0,384]}}",
        "{\"emb\":{\"dtype\":\"BF16\",\"shape\":[2,96],\"data_offsets\":[0,384]}}",
    };
    for (cases, 0..) |json, i| {
        const bytes = try rawImage(a, json, if (i == 4) 192 else 384);
        defer a.free(bytes);
        try expectBadTable(a, bytes);
    }
    // A header length larger than the file (and one past the configured bound).
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, 1 << 20, .little);
    try expectBadTable(a, &prefix);
}

test "openDirect refuses a table whose row count is not the config's for that layer" {
    const a = std.testing.allocator;
    // Released tables hold 164,994,380 / 164,999,008 / 165,004,086 / 165,008,942 rows, so a wrong shard stays in range.
    const image = try safetensorsImage(a, 32, 0);
    defer a.free(image.bytes);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "emb.safetensors", .data = image.bytes });
    const real = try tmp.dir.realPathFileAlloc(io, "emb.safetensors", a);
    defer a.free(real);
    const path = try a.dupeSentinel(u8, real, 0);
    defer a.free(path);
    for ([_]usize{ 31, 33 }) |wrong| {
        var table = bf16.openDirect(a, path, "emb", fixtureMap(), fixtureHash(), wrong) catch |err| switch (err) {
            error.RowCountMismatch => continue,
            error.OpenFailed, error.ReadFailed => return, // file system refuses O_DIRECT
            else => return err,
        };
        table.close();
        return error.TestUnexpectedResult;
    }
}
