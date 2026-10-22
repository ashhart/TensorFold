//! Byte identity, bounded reads and failure controls for startup's streamed cost-cache key.
const std = @import("std");
const identity = @import("cost_identity.zig");
const t = std.testing;

const SliceReader = struct {
    bytes: []const u8,
    short: usize = identity.buffer_bytes,
    calls: usize = 0,
    max_read: usize = 0,

    pub fn read(r: *@This(), out: []u8, offset: u64) !usize {
        r.calls += 1;
        r.max_read = @max(r.max_read, out.len);
        const at: usize = @intCast(offset);
        if (at >= r.bytes.len) return 0;
        const n = @min(out.len, r.short, r.bytes.len - at);
        @memcpy(out[0..n], r.bytes[at .. at + n]);
        return n;
    }
};

fn legacy(exe: []const u8, parts: []const []const u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..parts.len + 1) |i| {
        const part = if (i == 0) exe else parts[i - 1];
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, part.len, .little);
        h.update(&len);
        h.update(part);
    }
    return h.finalResult();
}

test "streamed cost identity equals legacy keys at SHA and read boundaries" {
    const bytes = try t.allocator.alloc(u8, identity.buffer_bytes * 2 + 13);
    defer t.allocator.free(bytes);
    for (bytes, 0..) |*byte, i| byte.* = @truncate((i *% 31) ^ (i >> 3));
    const parts = [_][]const u8{ "nemotron-cuda", "NVIDIA GB10", "shape\x00heads", "" };
    const sizes = [_]usize{ 0, 1, 55, 56, 63, 64, 65, identity.buffer_bytes - 1, identity.buffer_bytes, identity.buffer_bytes + 1, bytes.len };
    for (sizes) |size| {
        var reader: SliceReader = .{ .bytes = bytes[0..size] };
        const seed = try identity.fromReader(&reader, size, true);
        try t.expectEqual(legacy(bytes[0..size], &parts), seed.key(&parts));
        try t.expectEqual(legacy(bytes[0..size], &.{}), seed.key(&.{}));
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes[0..size], &digest, .{});
        try t.expectEqual(digest, seed.executable_sha256.?);
        try t.expect(reader.max_read <= identity.buffer_bytes);
        try t.expectEqual(std.math.divCeil(usize, size, identity.buffer_bytes) catch unreachable, reader.calls);
    }
}

test "one streamed executable derives independent head keys without further IO" {
    var reader: SliceReader = .{ .bytes = "engine-bytes", .short = 2 };
    const seed = try identity.fromReader(&reader, reader.bytes.len, false);
    const calls = reader.calls;
    const subset = [_][]const u8{ "nemotron-cuda", "NVIDIA GB10", "sm121 cuda13000 23/2688/131072/128/16384 draft1 own" };
    const full = [_][]const u8{ subset[0], subset[1], "sm121 cuda13000 23/2688/131072/128/16384 draft2 own" };
    try t.expectEqualStrings("7bde80f25c6c8b5bc0ebeb0076bfcd874bf7d7d7382b12599b42c91b25e4b23f", &std.fmt.bytesToHex(seed.key(&subset), .lower));
    try t.expectEqual(legacy(reader.bytes, &full), seed.key(&full));
    try t.expectEqual(legacy(reader.bytes, &subset), seed.key(&subset));
    try t.expect(!std.mem.eql(u8, &seed.key(&subset), &seed.key(&full)));
    try t.expectEqual(calls, reader.calls);
    try t.expectEqual(@as(?[32]u8, null), seed.executable_sha256);
    try t.expect(!std.mem.eql(u8, &seed.key(&.{ "ab", "c" }), &seed.key(&.{ "a", "bc" })));
}

test "streamed cost identity refuses truncation IO errors and oversized files" {
    var reader: SliceReader = .{ .bytes = "short" };
    try t.expectError(error.ExecutableChanged, identity.fromReader(&reader, 6, false));
    const calls = reader.calls;
    try t.expectError(error.StreamTooLong, identity.fromReader(&reader, identity.max_executable_bytes, false));
    try t.expectEqual(calls, reader.calls);
    try t.expectError(error.StreamTooLong, identity.fromReader(&reader, identity.max_executable_bytes + 1, false));
    try t.expectEqual(calls, reader.calls);
    const Broken = struct {
        pub fn read(_: @This(), _: []u8, _: u64) !usize {
            return error.ReadFailed;
        }
    };
    try t.expectError(error.ReadFailed, identity.fromReader(Broken{}, 1, false));
    const Invalid = struct {
        pub fn read(_: @This(), out: []u8, _: u64) !usize {
            return out.len + 1;
        }
    };
    try t.expectError(error.InvalidReadLength, identity.fromReader(Invalid{}, 1, false));
}

test "streamed cost identity reads a real file and retains the original key" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try t.allocator.alloc(u8, identity.buffer_bytes + 17);
    defer t.allocator.free(bytes);
    @memset(bytes, 0xa5);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "engine", .data = bytes });
    const file = try tmp.dir.openFile(t.io, "engine", .{});
    defer file.close(t.io);
    const seed = try identity.fromFile(file, t.io, true);
    try t.expectEqual(legacy(bytes, &.{ "a", "b" }), seed.key(&.{ "a", "b" }));
}

test "streamed cost identity refuses a file modified during its read" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "engine", .data = "before" });
    const file = try tmp.dir.openFile(t.io, "engine", .{ .mode = .read_write });
    defer file.close(t.io);
    const Changing = struct {
        file: std.Io.File,

        pub fn read(r: @This(), out: []u8, offset: u64) !usize {
            const n = try r.file.readPositionalAll(t.io, out, offset);
            try r.file.writePositionalAll(t.io, "changed while hashing", 0);
            return n;
        }

        pub fn stat(r: @This()) !std.Io.File.Stat {
            return r.file.stat(t.io);
        }
    };
    try t.expectError(error.ExecutableChanged, identity.fromSource(Changing{ .file = file }, false));
}

test "streamed cost identity refuses same-size timestamp changes" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "engine", .data = "unchanged size" });
    const file = try tmp.dir.openFile(t.io, "engine", .{});
    defer file.close(t.io);
    const ChangedStat = struct {
        file: std.Io.File,
        use_ctime: bool,
        calls: usize = 0,

        pub fn read(r: *@This(), out: []u8, offset: u64) !usize {
            return r.file.readPositionalAll(t.io, out, offset);
        }

        pub fn stat(r: *@This()) !std.Io.File.Stat {
            var s = try r.file.stat(t.io);
            if (r.calls > 0) {
                if (r.use_ctime) {
                    s.ctime = .fromNanoseconds(s.ctime.toNanoseconds() + 1);
                } else s.mtime = .fromNanoseconds(s.mtime.toNanoseconds() + 1);
            }
            r.calls += 1;
            return s;
        }
    };
    for ([_]bool{ false, true }) |ctime| {
        var source: ChangedStat = .{ .file = file, .use_ctime = ctime };
        try t.expectError(error.ExecutableChanged, identity.fromSource(&source, false));
        try t.expectEqual(@as(usize, 2), source.calls);
    }
}
