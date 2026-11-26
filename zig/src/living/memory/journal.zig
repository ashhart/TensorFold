//! The folded-memory delta journal (`living-weights.delta`): CRC-checked frames appended after a base file.
const std = @import("std");
const store = @import("store.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// Version 7: the CRC covers the op and the length too, and a journal begins with its base's generation.
pub const delta_magic = [4]u8{ 'T', 'F', 'D', '7' };

pub const DeltaOp = enum(u8) {
    add_unit = 1,
    forget_parent = 2,
    supersede_parent = 3,
    reinforce_parent = 4,
    generation = 5, // a journal's first frame: the base generation its frames apply to (u64 LE)
};

pub const DeltaFrame = struct {
    op: DeltaOp,
    payload: []u8,

    pub fn deinit(self: *DeltaFrame, gpa: Allocator) void {
        gpa.free(self.payload);
        self.* = undefined;
    }
};

/// A frame's CRC: its op and payload length, then its payload, so a changed op or length fails as a changed payload.
fn frameCrc(hdr: *const [24]u8, payload: []const u8) u32 {
    var crc = std.hash.Crc32.init();
    crc.update(hdr[4..16]);
    crc.update(payload);
    return crc.final();
}

/// One generation and at most the bank's sixteen pending operations.
const max_frame_payload: u64 = 256 * 1024 * 1024;
const max_records: usize = 17;
const max_journal_len: u64 = 24 + 8 + 16 * (24 + max_frame_payload);

/// The validated records, their end, and the generation named by the first one.
const Scan = struct { end: u64 = 0, generation: ?u64 = null, records: usize = 0 };

/// Whole frames are visited in order; a torn, damaged, or unknown tail is left for append to cut.
fn scanFile(io: Io, f: Io.File, ctx: anytype, comptime each: fn (@TypeOf(ctx), Io, Io.File, DeltaOp, u64, usize) anyerror!void) !Scan {
    const total_len = try f.length(io);
    if (total_len > max_journal_len) return error.FileTooBig;
    var out: Scan = .{};
    var scratch: [64 * 1024]u8 = undefined;
    while (out.end + 24 <= total_len) {
        var hdr: [24]u8 = undefined;
        if (try f.readPositionalAll(io, &hdr, out.end) != hdr.len) return error.ShortRead;
        if (!std.mem.eql(u8, hdr[0..4], &delta_magic)) break;
        const op = std.enums.fromInt(DeltaOp, hdr[4]) orelse break;
        const plen64 = std.mem.readInt(u64, hdr[8..16], .little);
        if (plen64 > max_frame_payload or plen64 + 24 > total_len - out.end) break;
        const plen: usize = @intCast(plen64);
        var crc = std.hash.Crc32.init();
        crc.update(hdr[4..16]);
        var left = plen;
        var at = out.end + 24;
        while (left > 0) {
            const n = @min(left, scratch.len);
            if (try f.readPositionalAll(io, scratch[0..n], at) != n) return error.ShortRead;
            crc.update(scratch[0..n]);
            at += n;
            left -= n;
        }
        if (crc.final() != std.mem.readInt(u32, hdr[16..20], .little)) break;
        if (out.end == 0) {
            if (op != .generation or plen != 8) break;
            var gen: [8]u8 = undefined;
            if (try f.readPositionalAll(io, &gen, out.end + 24) != gen.len) return error.ShortRead;
            out.generation = std.mem.readInt(u64, &gen, .little);
        } else if (op == .generation) break else {
            if (out.records == max_records) return error.TooManyRecords;
            try each(ctx, io, f, op, out.end + 24, plen);
        }
        out.end += 24 + plen64;
        out.records += 1;
    }
    return out;
}

fn skip(_: void, _: Io, _: Io.File, _: DeltaOp, _: u64, _: usize) anyerror!void {}

fn writeFrame(io: Io, f: Io.File, at: u64, op: DeltaOp, payload: []const u8) !u64 {
    var hdr: [24]u8 = @splat(0);
    @memcpy(hdr[0..4], &delta_magic);
    hdr[4] = @backingInt(op);
    std.mem.writeInt(u64, hdr[8..16], @intCast(payload.len), .little);
    std.mem.writeInt(u32, hdr[16..20], frameCrc(&hdr, payload), .little);
    try f.writePositionalAll(io, &hdr, at);
    if (payload.len > 0) try f.writePositionalAll(io, payload, at + hdr.len);
    return at + hdr.len + payload.len;
}

fn openExisting(io: Io, path: []const u8) !Io.File {
    return Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write, .follow_symlinks = false, .allow_directory = false });
}

/// Copies exactly the validated prefix, never the old journal's torn tail.
fn copyPrefix(io: Io, src: Io.File, dst: Io.File, len: u64) !void {
    var scratch: [64 * 1024]u8 = undefined;
    var off: u64 = 0;
    while (off < len) {
        const n: usize = @intCast(@min(@as(u64, scratch.len), len - off));
        if (try src.readPositionalAll(io, scratch[0..n], off) != n) return error.ShortRead;
        try dst.writePositionalAll(io, scratch[0..n], off);
        off += n;
    }
}

pub const AppendResult = struct { end: u64 = 0, committed: bool = false };

/// Atomically publishes the matching valid prefix and every supplied frame as one synced generation.
pub fn appendDeltaFramesWithSync(gpa: Allocator, io: Io, delta_path: []const u8, generation: u64, frames: []const DeltaFrame, result: *AppendResult, sync_dir: *const fn (Io, []const u8) anyerror!void) !void {
    _ = gpa;
    result.* = .{};
    if (frames.len > max_records - 1) return error.TooManyRecords;
    for (frames) |fr| if (fr.payload.len > @as(usize, @intCast(max_frame_payload))) return error.FrameTooBig;

    var staged = try Io.Dir.cwd().createFileAtomic(io, delta_path, .{ .permissions = store.private_file, .replace = true });
    defer staged.deinit(io);
    var at: u64 = 0;
    var new_generation = true;
    {
        const old: ?Io.File = openExisting(io, delta_path) catch |open_err| switch (open_err) {
            error.FileNotFound => missing: {
                // Windows no-follow opens report a dangling link as missing; never replace its target by following it.
                if (comptime @import("builtin").os.tag == .windows) {
                    if (Io.Dir.cwd().statFile(io, delta_path, .{ .follow_symlinks = false })) |_| {
                        return error.PathAlreadyExists;
                    } else |stat_err| switch (stat_err) {
                        error.FileNotFound => {},
                        else => return stat_err,
                    }
                }
                break :missing null;
            },
            else => return open_err,
        };
        if (old) |f| {
            defer f.close(io);
            // A Windows no-follow open yields the link itself; refuse it as the POSIX open already does.
            if ((try f.stat(io)).kind != .file) return error.PathAlreadyExists;
            const found = try scanFile(io, f, {}, skip);
            if (found.generation == generation) {
                if (frames.len > max_records - found.records) return error.TooManyRecords;
                try copyPrefix(io, f, staged.file, found.end);
                at = found.end;
                new_generation = false;
            }
        }
    }
    if (new_generation) {
        var gen: [8]u8 = undefined;
        std.mem.writeInt(u64, &gen, generation, .little);
        at = try writeFrame(io, staged.file, 0, .generation, &gen);
    }
    for (frames) |fr| at = try writeFrame(io, staged.file, at, fr.op, fr.payload);
    try staged.file.sync(io);
    try staged.replace(io);
    result.end = at;
    result.committed = true;
    try sync_dir(io, delta_path);
}

/// Atomically appends one operation and reports commitment before a possible directory-sync failure.
pub fn appendDeltaFrameWithSync(gpa: Allocator, io: Io, delta_path: []const u8, generation: u64, op: DeltaOp, payload: []const u8, result: *AppendResult, sync_dir: *const fn (Io, []const u8) anyerror!void) !void {
    const one = [_]DeltaFrame{.{ .op = op, .payload = @constCast(payload) }};
    try appendDeltaFramesWithSync(gpa, io, delta_path, generation, &one, result, sync_dir);
}

/// One frame appended for base `generation`, synced before return: a torn tail cut first, another generation restarted.
pub fn appendDeltaFrame(gpa: Allocator, io: Io, delta_path: []const u8, generation: u64, op: DeltaOp, payload: []const u8) !u64 {
    var result: AppendResult = .{};
    try appendDeltaFrameWithSync(gpa, io, delta_path, generation, op, payload, &result, store.syncParent);
    return result.end;
}

/// The journal's whole records for base `generation`, in order: none when absent or of another generation (a leftover).
pub fn readDeltaFrames(gpa: Allocator, io: Io, delta_path: []const u8, generation: u64) ![]DeltaFrame {
    const f = Io.Dir.cwd().openFile(io, delta_path, .{ .follow_symlinks = false, .allow_directory = false }) catch |e| switch (e) {
        error.FileNotFound => return try gpa.alloc(DeltaFrame, 0),
        else => return e,
    };
    defer f.close(io);
    var frames: std.ArrayList(DeltaFrame) = .empty;
    errdefer {
        for (frames.items) |*fr| fr.deinit(gpa);
        frames.deinit(gpa);
    }
    const Collect = struct {
        gpa: Allocator,
        list: *std.ArrayList(DeltaFrame),
        fn add(c: @This(), run_io: Io, file: Io.File, op: DeltaOp, at: u64, plen: usize) anyerror!void {
            const copy = try c.gpa.alloc(u8, plen);
            errdefer c.gpa.free(copy);
            if (try file.readPositionalAll(run_io, copy, at) != plen) return error.ShortRead;
            try c.list.append(c.gpa, .{ .op = op, .payload = copy });
        }
    };
    const found = try scanFile(io, f, Collect{ .gpa = gpa, .list = &frames }, Collect.add);
    if (found.generation != generation) {
        for (frames.items) |*fr| fr.deinit(gpa);
        frames.clearRetainingCapacity();
    }
    return frames.toOwnedSlice(gpa);
}

test "journal: append, replay, and truncated-tail recovery" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const dpath = try std.fs.path.join(gpa, &.{ dir, store.delta_file_name });
    defer gpa.free(dpath);

    const vals = [_]f32{ 3.5, -1.25 };
    const meta = [_]store.MetaKV{.{ .key = "uid", .value = "7" }};
    const tensors = [_]store.TensorSpec{
        .{ .name = "u.t", .dtype = .f32, .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&vals) },
    };
    const blob = try store.serializeToBytes(gpa, &meta, &tensors);
    defer gpa.free(blob);

    _ = try appendDeltaFrame(gpa, io, dpath, 1, .add_unit, blob);
    var sup: [8]u8 = undefined;
    std.mem.writeInt(u32, sup[0..4], 2, .little);
    std.mem.writeInt(u32, sup[4..8], 7, .little);
    _ = try appendDeltaFrame(gpa, io, dpath, 1, .supersede_parent, &sup);

    // Simulate a sudden power-loss partial frame appended at the tail
    {
        const f = try Io.Dir.cwd().createFile(io, dpath, .{ .truncate = false, .read = true });
        defer f.close(io);
        const end = try f.length(io);
        try f.writePositionalAll(io, "TFD7\x01\x00\x00\x00\x99\x00\x00", end);
    }

    const frames = try readDeltaFrames(gpa, io, dpath, 1);
    defer {
        for (frames) |*fr| fr.deinit(gpa);
        gpa.free(frames);
    }
    try std.testing.expectEqual(@as(usize, 2), frames.len);
    try std.testing.expectEqual(DeltaOp.add_unit, frames[0].op);
    try std.testing.expectEqual(DeltaOp.supersede_parent, frames[1].op);

    // Parse the safetensors blob inside frame 0
    var lf = try store.parseOwnedBytes(gpa, try gpa.dupe(u8, frames[0].payload));
    defer lf.deinit();
    try std.testing.expectEqualStrings("7", lf.getMeta("uid").?);
    const got = try lf.copyF32(gpa, "u.t", 2);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(f32, &vals, got);
}

fn journalPath(a: Allocator, tmp: *std.testing.TmpDir) ![]u8 {
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    return std.fs.path.join(a, &.{ dir, store.delta_file_name });
}

fn countFrames(gpa: Allocator, path: []const u8, generation: u64) !usize {
    const frames = try readDeltaFrames(gpa, std.testing.io, path, generation);
    defer {
        for (frames) |*fr| fr.deinit(gpa);
        gpa.free(frames);
    }
    return frames.len;
}

test "journal: a torn tail is cut before the next append, so replay keeps every acknowledged record" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    var pid: [4]u8 = .{ 7, 0, 0, 0 };
    _ = try appendDeltaFrame(gpa, io, path, 3, .forget_parent, &pid);
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = false, .read = true });
        defer f.close(io);
        try f.writePositionalAll(io, "TFD7\x02\x00\x00\x00\x04", try f.length(io)); // a crash mid-frame
    }
    pid[0] = 8;
    _ = try appendDeltaFrame(gpa, io, path, 3, .forget_parent, &pid); // acknowledged after the crash
    const frames = try readDeltaFrames(gpa, io, path, 3);
    defer {
        for (frames) |*fr| fr.deinit(gpa);
        gpa.free(frames);
    }
    try std.testing.expectEqual(@as(usize, 2), frames.len);
    try std.testing.expectEqual(@as(u8, 8), frames[1].payload[0]);
}

test "journal: the CRC covers the op, so a forget read back as a reinforce stops replay" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    const pid: [4]u8 = .{ 7, 0, 0, 0 };
    const end = try appendDeltaFrame(gpa, io, path, 1, .forget_parent, &pid);
    try std.testing.expectEqual(@as(usize, 1), try countFrames(gpa, path, 1));
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = false, .read = true });
        defer f.close(io);
        try f.writePositionalAll(io, &.{@backingInt(DeltaOp.reinforce_parent)}, end - 24 - pid.len + 4); // the op byte
    }
    try std.testing.expectEqual(@as(usize, 0), try countFrames(gpa, path, 1));
}

test "journal: one of another generation is never replayed, the next append starts it afresh, and a failed read is an error" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    try std.testing.expectEqual(@as(usize, 0), try countFrames(gpa, path, 1)); // absent: nothing to replay
    const pid: [4]u8 = .{ 7, 0, 0, 0 };
    _ = try appendDeltaFrame(gpa, io, path, 1, .forget_parent, &pid);
    _ = try appendDeltaFrame(gpa, io, path, 1, .forget_parent, &pid);
    try std.testing.expectEqual(@as(usize, 0), try countFrames(gpa, path, 2)); // a compaction's leftover
    _ = try appendDeltaFrame(gpa, io, path, 2, .forget_parent, &pid);
    try std.testing.expectEqual(@as(usize, 1), try countFrames(gpa, path, 2));
    try std.testing.expectEqual(@as(usize, 0), try countFrames(gpa, path, 1));
    // a fresh sibling path: on Windows a just-deleted file can stay delete-pending, refusing a directory in its place
    const dir_path = try std.fmt.allocPrint(gpa, "{s}.dir", .{path});
    defer gpa.free(dir_path);
    try Io.Dir.cwd().createDirPath(io, dir_path); // unreadable as a file: refused, not taken as an empty journal
    if (countFrames(gpa, dir_path, 2)) |_| return error.TestUnexpectedResult else |_| {}
}

test "journal: append refuses existing and dangling links without touching their targets" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    const before = "target must stay byte-for-byte unchanged";
    try tmp.dir.writeFile(io, .{ .sub_path = "target.delta", .data = before });
    const Link = struct {
        fn make(dir: Io.Dir, io_: Io, target: []const u8) !void {
            dir.symLink(io_, target, store.delta_file_name, .{}) catch |e| return switch (e) {
                error.AccessDenied, error.PermissionDenied => if (@import("builtin").os.tag == .windows) error.SkipZigTest else e,
                else => e,
            };
        }
    };
    try Link.make(tmp.dir, io, "target.delta");
    if (appendDeltaFrame(gpa, io, path, 2, .forget_parent, "keep")) |_| {
        return error.TestUnexpectedResult;
    } else |_| {}
    var after: [64]u8 = undefined;
    try std.testing.expectEqualStrings(before, try tmp.dir.readFile(io, "target.delta", &after));

    try tmp.dir.deleteFile(io, store.delta_file_name);
    try Link.make(tmp.dir, io, "missing.delta");
    if (appendDeltaFrame(gpa, io, path, 2, .forget_parent, "keep")) |_| {
        return error.TestUnexpectedResult;
    } else |_| {}
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "missing.delta", .{}));
}
test "journal: linked journals are refused while regular replay keeps its records" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    const target_path = try std.fs.path.join(gpa, &.{ std.fs.path.dirname(path).?, "target.delta" });
    defer gpa.free(target_path);
    _ = try appendDeltaFrame(gpa, io, target_path, 7, .forget_parent, "keep");
    const frames = try readDeltaFrames(gpa, io, target_path, 7);
    defer {
        for (frames) |*fr| fr.deinit(gpa);
        gpa.free(frames);
    }
    try std.testing.expectEqual(@as(usize, 1), frames.len);
    try std.testing.expectEqualSlices(u8, "keep", frames[0].payload);
    tmp.dir.symLink(io, "target.delta", store.delta_file_name, .{}) catch |e| return switch (e) {
        error.AccessDenied, error.PermissionDenied => if (@import("builtin").os.tag == .windows) error.SkipZigTest else e,
        else => e,
    };
    if (readDeltaFrames(gpa, io, path, 7)) |linked| {
        for (linked) |*fr| fr.deinit(gpa);
        gpa.free(linked);
        return error.TestUnexpectedResult;
    } else |_| {}
}

test "journal: directory-sync failure is committed and its retry does not duplicate the frame" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    const Retry = struct {
        var calls: usize = 0;
        fn sync(io_: Io, path_: []const u8) anyerror!void {
            calls += 1;
            if (calls == 1) return error.DirectorySyncFailed;
            try store.syncParent(io_, path_);
        }
    };
    var result: AppendResult = .{};
    try std.testing.expectError(error.DirectorySyncFailed, appendDeltaFrameWithSync(gpa, io, path, 1, .forget_parent, "keep", &result, Retry.sync));
    try std.testing.expect(result.committed);
    try std.testing.expect(result.end > 0);
    // The retry completes only work that was not committed; it never appends an acknowledged frame again.
    if (!result.committed) try appendDeltaFrameWithSync(gpa, io, path, 1, .forget_parent, "keep", &result, Retry.sync);
    try Retry.sync(io, path);
    try std.testing.expectEqual(@as(usize, 2), Retry.calls);
    try std.testing.expectEqual(@as(usize, 1), try countFrames(gpa, path, 1));
}

test "journal: oversized sparse files are refused before allocation" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .read = true });
        defer f.close(io);
        try f.setLength(io, max_journal_len + 1);
    }
    try std.testing.expectError(error.FileTooBig, countFrames(gpa, path, 1));
    var result: AppendResult = .{};
    try std.testing.expectError(error.FileTooBig, appendDeltaFrameWithSync(gpa, io, path, 1, .forget_parent, "keep", &result, store.syncParent));
    try std.testing.expectEqual(@as(u64, 0), result.end);
    try std.testing.expect(!result.committed);
}

test "journal: generation plus sixteen operations replays but excess records are refused" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    var end: u64 = 0;
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .read = true });
        defer f.close(io);
        var gen: [8]u8 = undefined;
        std.mem.writeInt(u64, &gen, 1, .little);
        end = try writeFrame(io, f, 0, .generation, &gen);
        for (0..16) |_| end = try writeFrame(io, f, end, .forget_parent, "keep");
    }
    const frames = try readDeltaFrames(gpa, io, path, 1);
    defer {
        for (frames) |*fr| fr.deinit(gpa);
        gpa.free(frames);
    }
    try std.testing.expectEqual(@as(usize, 16), frames.len);
    try std.testing.expectError(error.TooManyRecords, appendDeltaFrame(gpa, io, path, 1, .forget_parent, "keep"));
    {
        const f = try openExisting(io, path);
        defer f.close(io);
        _ = try writeFrame(io, f, end, .forget_parent, "keep");
    }
    try std.testing.expectError(error.TooManyRecords, countFrames(gpa, path, 1));
}

test "journal: two-frame publication is atomic across pre-rename and directory-sync failures" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(gpa, &tmp);
    defer gpa.free(path);
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .read = true });
        defer f.close(io);
        var gen: [8]u8 = undefined;
        std.mem.writeInt(u64, &gen, 1, .little);
        var end = try writeFrame(io, f, 0, .generation, &gen);
        for (0..15) |_| end = try writeFrame(io, f, end, .forget_parent, "keep");
    }
    var one = [_]u8{1};
    var two = [_]u8{2};
    const batch = [_]DeltaFrame{
        .{ .op = .forget_parent, .payload = &one },
        .{ .op = .supersede_parent, .payload = &two },
    };
    var result: AppendResult = .{};
    try std.testing.expectError(error.TooManyRecords, appendDeltaFramesWithSync(gpa, io, path, 1, &batch, &result, store.syncParent));
    try std.testing.expect(!result.committed);
    const before = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 * 1024));
    defer gpa.free(before);
    const after = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 * 1024));
    defer gpa.free(after);
    try std.testing.expectEqualSlices(u8, before, after);

    // Keep one old operation: the next publication must publish it and both new operations together.
    {
        const f = try openExisting(io, path);
        defer f.close(io);
        try f.setLength(io, 24 + 8 + 24 + 4);
    }
    const Retry = struct {
        var calls: usize = 0;
        fn sync(io_: Io, path_: []const u8) anyerror!void {
            calls += 1;
            if (calls == 1) return error.DirectorySyncFailed;
            try store.syncParent(io_, path_);
        }
    };
    try std.testing.expectError(error.DirectorySyncFailed, appendDeltaFramesWithSync(gpa, io, path, 1, &batch, &result, Retry.sync));
    try std.testing.expect(result.committed);
    try std.testing.expect(result.end > 0);
    if (!result.committed) try appendDeltaFramesWithSync(gpa, io, path, 1, &batch, &result, Retry.sync);
    try Retry.sync(io, path);
    const frames = try readDeltaFrames(gpa, io, path, 1);
    defer {
        for (frames) |*fr| fr.deinit(gpa);
        gpa.free(frames);
    }
    try std.testing.expectEqual(@as(usize, 3), frames.len);
    try std.testing.expectEqualSlices(u8, "keep", frames[0].payload);
    try std.testing.expectEqualSlices(u8, &one, frames[1].payload);
    try std.testing.expectEqualSlices(u8, &two, frames[2].payload);
}
