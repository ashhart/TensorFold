//! Tile permutation, scale pairs, draft ids and the pack byte compare.
const std = @import("std");
const pack = @import("src/families/flashnext/pack.zig");
const fixture = @import("src/families/flashnext/pack_test.zig");
const gpa = std.testing.allocator;
const io = std.testing.io;
const writeCheckpoint = fixture.writeCheckpoint;
const tmpPath = fixture.tmpPath;

test "tile weight permutes words so a group's words sit innermost per 32-row tile" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const words = try a.alloc(u32, 32 * 12);
    for (words, 0..) |*w, i| w.* = @intCast(i);
    const out = try pack.tileWeight(a, std.mem.sliceAsBytes(words), 32, 12, 32, 6);
    defer a.free(out);
    const out_words = std.mem.bytesAsSlice(u32, out);
    try std.testing.expectEqual(@as(u32, 0), out_words[0]); // j 0, group 0, word 0
    try std.testing.expectEqual(@as(u32, 5), out_words[5]);
    try std.testing.expectEqual(@as(u32, 12), out_words[6]); // row 1's first word follows row 0's group 0
    try std.testing.expectEqual(@as(u32, 377), out_words[191]); // row 31, group 0, word 5
    try std.testing.expectEqual(@as(u32, 6), out_words[192]); // group 1 restarts at row 0's second half
    try std.testing.expectEqual(@as(u32, 383), out_words[383]);
}

test "pack scales lay scale and bias pairs group-major over the row-concatenated stack" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var members: [2]pack.Q = undefined;
    const s0_bits = [_]u16{ 10, 11, 12, 13, 14, 15 };
    const b0_bits = [_]u16{ 100, 101, 102, 103, 104, 105 };
    const s1_bits = [_]u16{ 20, 21 };
    const b1_bits = [_]u16{ 110, 111 };
    members[0] = .{
        .w = .{ .dtype = .u32, .rank = 2, .shape = .{ 3, 4, 0, 0 }, .bytes = &.{} },
        .s = .{ .dtype = .bf16, .rank = 2, .shape = .{ 3, 2, 0, 0 }, .bytes = std.mem.sliceAsBytes(s0_bits[0..]) },
        .b = .{ .dtype = .bf16, .rank = 2, .shape = .{ 3, 2, 0, 0 }, .bytes = std.mem.sliceAsBytes(b0_bits[0..]) },
        .rows = 3,
        .k = 64,
        .kw = 12,
        .spec = .{ .bits = 6, .group = 32 },
    };
    members[1] = .{
        .w = .{ .dtype = .u32, .rank = 2, .shape = .{ 1, 4, 0, 0 }, .bytes = &.{} },
        .s = .{ .dtype = .bf16, .rank = 2, .shape = .{ 1, 2, 0, 0 }, .bytes = std.mem.sliceAsBytes(s1_bits[0..]) },
        .b = .{ .dtype = .bf16, .rank = 2, .shape = .{ 1, 2, 0, 0 }, .bytes = std.mem.sliceAsBytes(b1_bits[0..]) },
        .rows = 1,
        .k = 64,
        .kw = 12,
        .spec = .{ .bits = 6, .group = 32 },
    };
    const out = try pack.packScales(a, members[0..]);
    defer a.free(out);
    const words = std.mem.bytesAsSlice(u16, out);
    // rows 4, kg 2: group 0 first (3 rows of member 0, then member 1), each a (scale, bias) pair
    try std.testing.expectEqual(@as(u16, 10), words[0]);
    try std.testing.expectEqual(@as(u16, 100), words[1]);
    try std.testing.expectEqual(@as(u16, 12), words[2]);
    try std.testing.expectEqual(@as(u16, 102), words[3]);
    try std.testing.expectEqual(@as(u16, 14), words[4]);
    try std.testing.expectEqual(@as(u16, 104), words[5]);
    try std.testing.expectEqual(@as(u16, 20), words[6]);
    try std.testing.expectEqual(@as(u16, 110), words[7]);
    try std.testing.expectEqual(@as(u16, 11), words[8]); // group 1
    try std.testing.expectEqual(@as(u16, 101), words[9]);
    try std.testing.expectEqual(@as(u16, 21), words[14]);
    try std.testing.expectEqual(@as(u16, 111), words[15]);
}

test "draft id list sorts, pads to 64 with the smallest unlisted ids and refuses past-vocab ids" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    for (10..72) |id| try text.writer.print("{d}\n", .{id});
    try tmp.dir.writeFile(io, .{ .sub_path = "vocab.txt", .data = text.written() });
    const path = try tmpPath(a, tmp, "vocab.txt");
    defer a.free(path);
    const ids = try pack.draftIdList(a, io, path, 128);
    defer a.free(ids);
    try std.testing.expectEqual(@as(usize, 64), ids.len);
    try std.testing.expectEqual(@as(u32, 0), ids[0]); // two pads: 0 and 1, then the listed 10..71
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
    try std.testing.expectEqual(@as(u32, 10), ids[2]);
    try std.testing.expectEqual(@as(u32, 71), ids[63]);
    try tmp.dir.writeFile(io, .{ .sub_path = "bad.txt", .data = "128\n" });
    const bad = try tmpPath(a, tmp, "bad.txt");
    defer a.free(bad);
    try std.testing.expectError(error.DraftIdPastVocab, pack.draftIdList(a, io, bad, 128));
    try tmp.dir.writeFile(io, .{ .sub_path = "empty.txt", .data = "" });
    const empty = try tmpPath(a, tmp, "empty.txt");
    defer a.free(empty);
    try std.testing.expectError(error.BadDraftVocab, pack.draftIdList(a, io, empty, 128));
}

test "compare file reports byte differences with the first differing offset" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeCheckpoint(tmp, a);
    const model_dir = try tmpPath(a, tmp, ".");
    defer a.free(model_dir);
    for ([_][]const u8{ "out_a", "out_b" }) |name| try tmp.dir.createDirPath(io, name);
    _ = try pack.build(a, io, model_dir, try tmpPath(a, tmp, "out_a"), null);
    _ = try pack.build(a, io, model_dir, try tmpPath(a, tmp, "out_b"), null);
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const a_path = try tmpPath(a, tmp, "out_a/pack.safetensors");
    defer a.free(a_path);
    const b_path = try tmpPath(a, tmp, "out_b/pack.safetensors");
    defer a.free(b_path);
    try std.testing.expect(try pack.compareFile(a, io, a_path, b_path, &out.writer));
    // flip one byte inside b's tensor data: ple.starts is the file's last tensor, so its last word is hit
    const image = try tmp.dir.readFileAlloc(io, "out_b/pack.safetensors", a, .limited(1 << 30));
    defer a.free(image);
    image[image.len - 4] ^= 0xFF;
    try tmp.dir.writeFile(io, .{ .sub_path = "out_b/pack.safetensors", .data = image });
    var out2: std.Io.Writer.Allocating = .init(a);
    defer out2.deinit();
    try std.testing.expect(!try pack.compareFile(a, io, a_path, b_path, &out2.writer));
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "DIFF ple.starts: first differing byte 28 of 32") != null);
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "62 identical, 1 differ") != null);
}
