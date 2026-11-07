//! Qwen reference receipt checks and integer tensors; shared floating tensor I/O stays in fixture.zig.
const std = @import("std");
const f = @import("fixture.zig");
const Allocator = std.mem.Allocator;

pub fn verify(files: f.Files, model: []const u8) !void {
    const ck = try f.field(files.manifest, "checkpoint");
    const repo = try f.string(try f.field(ck, "repo"));
    if (!std.mem.eql(u8, repo, "lightonai/LightOnOCR-3-0.8B") and !std.mem.eql(u8, repo, "lightonai/LightOnOCR-3-4B")) return error.WrongReferenceModel;
    for ([_][]const u8{ "assets", "weights_sha256" }) |group| {
        const records = try f.field(ck, group);
        if (records != .object or records.object.count() == 0) return error.BadFixture;
        var it = records.object.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!std.mem.eql(u8, std.fs.path.basename(name), name)) return error.BadFixture;
            const path = try std.fs.path.join(files.a, &.{ model, name });
            const file = try std.Io.Dir.cwd().openFile(files.io, path, .{});
            defer file.close(files.io);
            const length = try file.length(files.io);
            var sha = std.crypto.hash.sha2.Sha256.init(.{});
            const buffer = try files.a.alloc(u8, 4 << 20);
            defer files.a.free(buffer);
            var at: u64 = 0;
            while (at < length) {
                const part = buffer[0..@intCast(@min(buffer.len, length - at))];
                if (try file.readPositionalAll(files.io, part, at) != part.len) return error.CheckpointChanged;
                sha.update(part);
                at += part.len;
            }
            const hex = std.fmt.bytesToHex(sha.finalResult(), .lower);
            if (!std.mem.eql(u8, &hex, try f.string(entry.value_ptr.*))) return error.CheckpointChecksum;
        }
    }
}

pub fn integers(a: Allocator, files: f.Files, name: []const u8) ![]i64 {
    const array = try files.read(name);
    if (!std.mem.eql(u8, array.descr, "<i8")) return error.BadFixture;
    const out = try a.alloc(i64, array.count());
    for (out, 0..) |*v, i| v.* = std.mem.readInt(i64, array.data[i * 8 ..][0..8], .little);
    return out;
}

pub fn ids(a: Allocator, value: std.json.Value, vocab: usize) ![]u32 {
    const values = try f.list(value);
    const out = try a.alloc(u32, values.len);
    for (out, values) |*dst, v| {
        const id = try f.integer(v);
        if (id >= vocab) return error.TextToken;
        dst.* = @intCast(id);
    }
    return out;
}

pub fn writePositions(files: f.Files, positions: []const i64, rows: usize) !void {
    if (positions.len != rows * 3) return error.BadCaptureShape;
    const dict = try std.fmt.allocPrint(files.a, "{{'descr': '<i8', 'fortran_order': False, 'shape': (3, 1, {d}), }}", .{rows});
    const pad = (64 - (10 + dict.len + 1) % 64) % 64;
    const head = 10 + dict.len + pad + 1;
    const bytes = try files.a.alloc(u8, head + positions.len * 8);
    @memcpy(bytes[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, bytes[8..10], @intCast(head - 10), .little);
    @memcpy(bytes[10..][0..dict.len], dict);
    @memset(bytes[10 + dict.len .. head - 1], ' ');
    bytes[head - 1] = '\n';
    for (positions, 0..) |p, i| std.mem.writeInt(i64, bytes[head + i * 8 ..][0..8], p, .little);
    try files.output.writeFile(files.io, .{ .sub_path = "decoder.position_ids.npy", .data = bytes });
}

pub fn imageRange(prompt: []const u32, token: u32, count: usize) !usize {
    const first = std.mem.indexOfScalar(u32, prompt, token) orelse return error.ImageRows;
    if (count == 0 or count > prompt.len - first) return error.ImageRows;
    for (prompt[first..][0..count]) |id| if (id != token) return error.ImageRows;
    if (std.mem.indexOfScalar(u32, prompt[first + count ..], token) != null) return error.ImageRows;
    return first;
}

test "Qwen capture requires one complete contiguous image interval" {
    try std.testing.expectEqual(@as(usize, 2), try imageRange(&.{ 7, 8, 50, 50, 9 }, 50, 2));
    try std.testing.expectError(error.ImageRows, imageRange(&.{ 7, 50, 9, 50 }, 50, 1));
    try std.testing.expectError(error.ImageRows, imageRange(&.{ 7, 50, 9 }, 50, 2));
}
