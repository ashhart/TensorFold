const std = @import("std");
const profile = @import("cuda_exl3_frequency.zig");
const Io = std.Io;
const allocator = std.testing.allocator;
const io = std.testing.io;
const identity: profile.Identity = .{ .model = @splat(0xa7), .world = 2, .rank = 1 };
const header_bytes = 64;
const checksum_bytes = 32;
const decoder_file_bytes = header_bytes + 48 * 512 * 4 + checksum_bytes;

fn profilePath(directory: Io.Dir) ![]u8 {
    const root = try directory.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    return std.fs.path.join(allocator, &.{ root, "frequency.bin" });
}

fn sample(counts: *profile.Counts, mtp_present: bool) void {
    counts.* = .{ .mtp_present = mtp_present };
    for (&counts.decoder, 0..) |*layer, layer_index| for (layer, 0..) |*count, expert_index| {
        count.* = @intCast((layer_index + 1) * 100_000 + expert_index);
    };
    for (&counts.mtp, 0..) |*count, expert_index| count.* = @intCast(7_000_000 + expert_index);
    counts.decoder[0][0] = 0;
    counts.decoder[47][511] = std.math.maxInt(u32);
    counts.mtp[511] = std.math.maxInt(u32);
}

fn expectCounts(expected: *const profile.Counts, actual: *const profile.Counts) !void {
    try std.testing.expectEqual(expected.mtp_present, actual.mtp_present);
    for (&expected.decoder, &actual.decoder) |*expected_layer, *actual_layer| {
        try std.testing.expectEqualSlices(u32, expected_layer, actual_layer);
    }
    if (expected.mtp_present) {
        try std.testing.expectEqualSlices(u32, &expected.mtp, &actual.mtp);
    } else {
        for (actual.mtp) |count| try std.testing.expectEqual(@as(u32, 0), count);
    }
}

fn expectOnlyProfile(directory: Io.Dir) !void {
    var iterator = directory.iterate();
    var entries: usize = 0;
    while (try iterator.next(io)) |entry| {
        entries += 1;
        try std.testing.expectEqualStrings("frequency.bin", entry.name);
    }
    try std.testing.expectEqual(@as(usize, 1), entries);
}

fn bytesAt(path: []const u8) ![]u8 {
    return Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(profile.max_file_bytes + 2));
}

fn expectPreserved(path: []const u8, before: []const u8) !void {
    const after = try bytesAt(path);
    defer allocator.free(after);
    try std.testing.expectEqualSlices(u8, before, after);
}

test "exl3: frequency full decoder and optional MTP round trip with private permissions" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var temporary = std.testing.tmpDir(.{ .iterate = true });
    defer temporary.cleanup();
    const path = try profilePath(temporary.dir);
    defer allocator.free(path);
    var counts: profile.Counts = .{};
    for ([_]bool{ false, true, false }) |mtp_present| {
        sample(&counts, mtp_present);
        try profile.write(allocator, io, path, identity, &counts);
        const restored = try profile.read(allocator, io, path, identity);
        try expectCounts(&counts, &restored);
        const stat = try temporary.dir.statFile(io, "frequency.bin", .{});
        try std.testing.expectEqual(@as(u64, if (mtp_present) profile.max_file_bytes else decoder_file_bytes), stat.size);
        try std.testing.expectEqual(@as(std.posix.mode_t, 0o600), stat.permissions.toMode() & 0o777);
        const bytes = try bytesAt(path);
        defer allocator.free(bytes);
        try std.testing.expectEqualSlices(u8, "TFEXL3FQ", bytes[0..8]);
        try std.testing.expectEqualSlices(u8, &identity.model, bytes[16..48]);
        try std.testing.expectEqualSlices(u8, &.{ 1, 0, 0, 0 }, bytes[8..12]);
        try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, bytes[64..68]);
        try std.testing.expectEqualSlices(u8, &.{ 0xa1, 0x86, 0x01, 0x00 }, bytes[68..72]);
        try std.testing.expectEqualSlices(u8, &.{ 0xff, 0xff, 0xff, 0xff }, bytes[64 + 48 * 512 * 4 - 4 ..][0..4]);
        try expectOnlyProfile(temporary.dir);
    }
}

test "exl3: frequency missing profile and every exact identity mismatch are recoverable" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try profilePath(temporary.dir);
    defer allocator.free(path);
    try std.testing.expectError(error.FileNotFound, profile.read(allocator, io, path, identity));
    const counts: profile.Counts = .{};
    try profile.write(allocator, io, path, identity, &counts);
    for (0..32) |model_index| {
        var different = identity;
        different.model[model_index] ^= 1;
        try std.testing.expectError(error.IdentityMismatch, profile.read(allocator, io, path, different));
    }
    var different = identity;
    different.world = 3;
    try std.testing.expectError(error.IdentityMismatch, profile.read(allocator, io, path, different));
    different = identity;
    different.rank = 0;
    try std.testing.expectError(error.IdentityMismatch, profile.read(allocator, io, path, different));
    const restored = try profile.read(allocator, io, path, identity);
    try expectCounts(&counts, &restored);
}

test "exl3: frequency rejects truncated oversized malformed and corrupt profiles" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try profilePath(temporary.dir);
    defer allocator.free(path);
    var counts: profile.Counts = .{};
    sample(&counts, true);
    try profile.write(allocator, io, path, identity, &counts);
    const original = try bytesAt(path);
    defer allocator.free(original);
    for ([_]usize{ 0, 7, 63, 64, 95, 96, decoder_file_bytes, profile.max_file_bytes - 1 }) |length| {
        try temporary.dir.writeFile(io, .{ .sub_path = "frequency.bin", .data = original[0..length] });
        try std.testing.expectError(error.TruncatedProfile, profile.read(allocator, io, path, identity));
    }
    const changed = try allocator.dupe(u8, original);
    defer allocator.free(changed);
    const Mutation = struct { offset: usize, value: u8, expected: anyerror };
    for ([_]Mutation{
        .{ .offset = 0, .value = 0, .expected = error.MalformedProfile },
        .{ .offset = 8, .value = 2, .expected = error.UnsupportedVersion },
        .{ .offset = 12, .value = 2, .expected = error.MalformedProfile },
        .{ .offset = 56, .value = 47, .expected = error.MalformedProfile },
        .{ .offset = 60, .value = 1, .expected = error.MalformedProfile },
        .{ .offset = 16, .value = 0, .expected = error.ChecksumMismatch },
        .{ .offset = 48, .value = 3, .expected = error.ChecksumMismatch },
        .{ .offset = 64, .value = 1, .expected = error.ChecksumMismatch },
        .{ .offset = 64 + 48 * 512 * 4, .value = 1, .expected = error.ChecksumMismatch },
        .{ .offset = profile.max_file_bytes - 1, .value = original[profile.max_file_bytes - 1] ^ 1, .expected = error.ChecksumMismatch },
    }) |mutation| {
        @memcpy(changed, original);
        changed[mutation.offset] = mutation.value;
        try temporary.dir.writeFile(io, .{ .sub_path = "frequency.bin", .data = changed });
        try std.testing.expectError(mutation.expected, profile.read(allocator, io, path, identity));
    }
    const oversized = try allocator.alloc(u8, profile.max_file_bytes + 1);
    defer allocator.free(oversized);
    @memcpy(oversized[0..original.len], original);
    oversized[original.len] = 0;
    try temporary.dir.writeFile(io, .{ .sub_path = "frequency.bin", .data = oversized });
    try std.testing.expectError(error.ProfileTooLarge, profile.read(allocator, io, path, identity));
    counts.mtp_present = false;
    try profile.write(allocator, io, path, identity, &counts);
    const file = try temporary.dir.openFile(io, "frequency.bin", .{ .mode = .read_write });
    defer file.close(io);
    try file.writePositionalAll(io, &.{0}, decoder_file_bytes);
    try std.testing.expectError(error.MalformedProfile, profile.read(allocator, io, path, identity));
}

fn failWrite(userdata: ?*anyopaque, file: Io.File, header: []const u8, data: []const []const u8, splat: usize, offset: u64) Io.File.WritePositionalError!usize {
    if (offset != 0) return error.NoSpaceLeft;
    const prefix = data[0][0..13];
    return io.vtable.fileWritePositional(userdata, file, header, &.{prefix}, splat, offset);
}

fn failSync(_: ?*anyopaque, file: Io.File) Io.File.SyncError!void {
    const stat = file.stat(io) catch return error.InputOutput;
    if (stat.permissions.toMode() & 0o777 != 0o600 or stat.size != profile.max_file_bytes) return error.AccessDenied;
    return error.InputOutput;
}

fn failRename(_: ?*anyopaque, _: Io.Dir, _: []const u8, _: Io.Dir, _: []const u8) Io.Dir.RenameError!void {
    return error.AccessDenied;
}

test "exl3: frequency partial write sync and rename failures preserve previous bytes and remove temporaries" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var temporary = std.testing.tmpDir(.{ .iterate = true });
    defer temporary.cleanup();
    const path = try profilePath(temporary.dir);
    defer allocator.free(path);
    var counts: profile.Counts = .{};
    try profile.write(allocator, io, path, identity, &counts);
    const before = try bytesAt(path);
    defer allocator.free(before);
    sample(&counts, true);
    for (0..3) |failure_index| {
        var failing_vtable = io.vtable.*;
        const expected: anyerror = switch (failure_index) {
            0 => failure: {
                failing_vtable.fileWritePositional = failWrite;
                break :failure error.NoSpaceLeft;
            },
            1 => failure: {
                failing_vtable.fileSync = failSync;
                break :failure error.InputOutput;
            },
            else => failure: {
                failing_vtable.dirRename = failRename;
                break :failure error.AccessDenied;
            },
        };
        var failing_io = io;
        failing_io.vtable = &failing_vtable;
        try std.testing.expectError(expected, profile.write(allocator, failing_io, path, identity, &counts));
        try expectPreserved(path, before);
        try expectOnlyProfile(temporary.dir);
        const restored = try profile.read(allocator, io, path, identity);
        try std.testing.expectEqual(@as(u32, 0), restored.decoder[47][511]);
    }
    try profile.write(allocator, io, path, identity, &counts);
    const restored = try profile.read(allocator, io, path, identity);
    try expectCounts(&counts, &restored);
}

test "exl3: frequency allocation and invalid identity failures preserve previous profile" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var temporary = std.testing.tmpDir(.{ .iterate = true });
    defer temporary.cleanup();
    const path = try profilePath(temporary.dir);
    defer allocator.free(path);
    const counts: profile.Counts = .{};
    try profile.write(allocator, io, path, identity, &counts);
    const before = try bytesAt(path);
    defer allocator.free(before);
    var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, profile.write(failing.allocator(), io, path, identity, &counts));
    try std.testing.expectError(error.OutOfMemory, profile.read(failing.allocator(), io, path, identity));
    for ([_]profile.Identity{
        .{ .model = identity.model, .world = 0, .rank = 0 },
        .{ .model = identity.model, .world = 2, .rank = 2 },
    }) |invalid| {
        try std.testing.expectError(error.InvalidIdentity, profile.write(allocator, io, path, invalid, &counts));
        try std.testing.expectError(error.InvalidIdentity, profile.read(allocator, io, path, invalid));
    }
    try std.testing.expectError(error.InvalidPath, profile.write(allocator, io, "", identity, &counts));
    try std.testing.expectError(error.InvalidPath, profile.write(allocator, io, ".", identity, &counts));
    try std.testing.expectError(error.InvalidPath, profile.write(allocator, io, "..", identity, &counts));
    try expectPreserved(path, before);
    try expectOnlyProfile(temporary.dir);
}

test "exl3: frequency read allocation stays fixed and oversized input needs no allocation" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try profilePath(temporary.dir);
    defer allocator.free(path);
    const single: profile.Identity = .{ .model = @splat(0), .world = 1, .rank = 0 };
    var counts: profile.Counts = .{};
    sample(&counts, true);
    try profile.write(allocator, io, path, single, &counts);
    const backing = try allocator.alloc(u8, profile.max_file_bytes + 1);
    defer allocator.free(backing);
    var bounded = std.heap.FixedBufferAllocator.init(backing);
    const restored = try profile.read(bounded.allocator(), io, path, single);
    try expectCounts(&counts, &restored);
    const file = try temporary.dir.openFile(io, "frequency.bin", .{ .mode = .read_write });
    defer file.close(io);
    try file.setLength(io, profile.max_file_bytes + 1);
    var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.ProfileTooLarge, profile.read(failing.allocator(), io, path, single));
    try std.testing.expect(!failing.has_induced_failure);
}

test "exl3: frequency failed first save leaves no destination or temporary file" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var temporary = std.testing.tmpDir(.{ .iterate = true });
    defer temporary.cleanup();
    const path = try profilePath(temporary.dir);
    defer allocator.free(path);
    var counts: profile.Counts = .{};
    sample(&counts, true);
    var failing_vtable = io.vtable.*;
    failing_vtable.fileSync = failSync;
    var failing_io = io;
    failing_io.vtable = &failing_vtable;
    try std.testing.expectError(error.InputOutput, profile.write(allocator, failing_io, path, identity, &counts));
    try std.testing.expectError(error.FileNotFound, profile.read(allocator, io, path, identity));
    var iterator = temporary.dir.iterate();
    try std.testing.expectEqual(@as(?Io.Dir.Entry, null), try iterator.next(io));
    try profile.write(allocator, io, path, identity, &counts);
    const restored = try profile.read(allocator, io, path, identity);
    try expectCounts(&counts, &restored);
    try expectOnlyProfile(temporary.dir);
}

test "exl3: frequency rejects a FIFO without a writer and other nonregular input" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try profilePath(temporary.dir);
    defer allocator.free(path);
    const terminated_path = try allocator.dupeSentinel(u8, path, 0);
    defer allocator.free(terminated_path);
    switch (@import("builtin").os.tag) {
        .linux => {
            const result = std.os.linux.mknodat(temporary.dir.handle, "frequency.bin", std.posix.S.IFIFO | 0o600, 0);
            if (std.posix.errno(result) != .SUCCESS) return error.FifoCreateFailed;
        },
        .macos => {
            const Native = struct {
                extern "c" fn mkfifo(pathname: [*:0]const u8, mode: std.posix.mode_t) c_int;
            };
            if (Native.mkfifo(terminated_path, 0o600) != 0) return error.FifoCreateFailed;
        },
        else => return error.SkipZigTest,
    }
    var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.ProfileNotRegular, profile.read(failing.allocator(), io, path, identity));
    try std.testing.expectError(error.ProfileNotRegular, profile.read(allocator, io, "/dev/null", identity));
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    try std.testing.expectError(error.ProfileNotRegular, profile.read(allocator, io, root, identity));
    try std.testing.expect(!failing.has_induced_failure);
}
