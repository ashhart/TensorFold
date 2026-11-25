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

/// The journal's whole records: where the last one ends, and the generation its first frame names (null: none).
const Scan = struct { end: usize = 0, generation: ?u64 = null };

/// Walks `raw`'s frames, stopping at the first torn, damaged or unknown one; `each` sees every whole record.
fn scan(raw: []const u8, ctx: anytype, comptime each: fn (@TypeOf(ctx), DeltaOp, []const u8) anyerror!void) !Scan {
    var out: Scan = .{};
    var off: usize = 0;
    while (off + 24 <= raw.len) {
        if (!std.mem.eql(u8, raw[off .. off + 4], &delta_magic)) break;
        const op = std.enums.fromInt(DeltaOp, raw[off + 4]) orelse break;
        const plen = std.math.cast(usize, std.mem.readInt(u64, raw[off + 8 ..][0..8], .little)) orelse break;
        if (plen > 256 * 1024 * 1024 or off + 24 + plen > raw.len) break;
        const payload = raw[off + 24 .. off + 24 + plen];
        if (frameCrc(raw[off..][0..24], payload) != std.mem.readInt(u32, raw[off + 16 ..][0..4], .little)) break;
        if (off == 0) {
            if (op != .generation or plen != 8) break; // a journal without its generation holds nothing usable
            out.generation = std.mem.readInt(u64, payload[0..8], .little);
        } else if (op == .generation) break else try each(ctx, op, payload);
        off += 24 + plen;
        out.end = off;
    }
    return out;
}

fn skip(_: void, _: DeltaOp, _: []const u8) anyerror!void {}

/// The journal's bytes, empty when it does not exist; any other failure to read it is an error, never "no journal".
fn readJournal(gpa: Allocator, io: Io, f: Io.File) ![]u8 {
    const total_len = std.math.cast(usize, try f.length(io)) orelse return error.FileTooBig;
    const raw = try gpa.alloc(u8, total_len);
    errdefer gpa.free(raw);
    if (try f.readPositionalAll(io, raw, 0) != total_len) return error.ShortRead;
    return raw;
}

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

/// One frame appended for base `generation`, synced before return: a torn tail cut first, another generation restarted.
pub fn appendDeltaFrame(gpa: Allocator, io: Io, delta_path: []const u8, generation: u64, op: DeltaOp, payload: []const u8) !u64 {
    const f = try Io.Dir.cwd().createFile(io, delta_path, .{ .truncate = false, .read = true, .permissions = store.private_file });
    defer f.close(io);
    const raw = try readJournal(gpa, io, f);
    defer gpa.free(raw);
    const found = try scan(raw, {}, skip);
    var at: u64 = found.end;
    if (found.generation != generation) {
        var gen: [8]u8 = undefined;
        std.mem.writeInt(u64, &gen, generation, .little);
        try f.setLength(io, 0);
        at = try writeFrame(io, f, 0, .generation, &gen);
    } else if (raw.len > found.end) try f.setLength(io, found.end); // the torn tail goes before a record follows it
    const end = try writeFrame(io, f, at, op, payload);
    try f.sync(io);
    if (found.generation != generation) try store.syncParent(io, delta_path); // a new journal's name is durable too
    return end;
}

/// The journal's whole records for base `generation`, in order: none when absent or of another generation (a leftover).
pub fn readDeltaFrames(gpa: Allocator, io: Io, delta_path: []const u8, generation: u64) ![]DeltaFrame {
    const f = Io.Dir.cwd().openFile(io, delta_path, .{}) catch |e| switch (e) {
        error.FileNotFound => return try gpa.alloc(DeltaFrame, 0),
        else => return e,
    };
    defer f.close(io);
    const raw = try readJournal(gpa, io, f);
    defer gpa.free(raw);
    var frames: std.ArrayList(DeltaFrame) = .empty;
    errdefer {
        for (frames.items) |*fr| fr.deinit(gpa);
        frames.deinit(gpa);
    }
    const Collect = struct {
        gpa: Allocator,
        list: *std.ArrayList(DeltaFrame),
        fn add(c: @This(), op: DeltaOp, payload: []const u8) anyerror!void {
            const copy = try c.gpa.dupe(u8, payload);
            errdefer c.gpa.free(copy);
            try c.list.append(c.gpa, .{ .op = op, .payload = copy });
        }
    };
    const found = try scan(raw, Collect{ .gpa = gpa, .list = &frames }, Collect.add);
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
