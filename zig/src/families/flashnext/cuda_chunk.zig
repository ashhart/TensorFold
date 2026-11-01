//! Prompt rows a pass: TF_FLASHNEXT_CHUNK while another stream decodes, TF_FLASHNEXT_CHUNK_SOLO when none does.
const std = @import("std");
const batch = @import("cuda_batch.zig");

/// TF_FLASHNEXT_CHUNK's rows a pass: unset, the default; otherwise 1 to batch.max_chunk, anything else refused.
pub fn parseChunk(v: ?[]const u8) !usize {
    const s = v orelse return batch.default_chunk;
    const n = std.fmt.parseInt(usize, s, 10) catch return error.BadChunk;
    if (n == 0 or n > batch.max_chunk) return error.BadChunk;
    return n;
}

pub fn chunkRows() !usize {
    return parseChunk(if (std.c.getenv("TF_FLASHNEXT_CHUNK")) |v| std.mem.span(v) else null);
}

/// TF_FLASHNEXT_CHUNK_SOLO's rows for a pass no other stream waits on: unset, `busy` (off); below `busy`, refused.
pub fn parseSolo(v: ?[]const u8, busy: usize) !usize {
    const s = v orelse return busy;
    const n = parseChunk(s) catch return error.BadChunk;
    if (n < busy) return error.BadChunk;
    return n;
}

pub fn soloRows(busy: usize) !usize {
    return parseSolo(if (std.c.getenv("TF_FLASHNEXT_CHUNK_SOLO")) |v| std.mem.span(v) else null, busy);
}

/// The rows a pass of `own` takes when it starts: `solo` unless another lane's prompt is in (it decodes).
pub fn passRows(lanes: anytype, own: anytype, solo: usize, busy: usize) usize {
    if (solo == busy) return busy;
    var it = lanes.valueIterator();
    while (it.next()) |l| if (l != own and l.filled == null) return busy;
    return solo;
}

test "TF_FLASHNEXT_CHUNK takes 1 to the largest measured pass and refuses the rest" {
    try std.testing.expectEqual(batch.default_chunk, try parseChunk(null));
    try std.testing.expectEqual(@as(usize, 4096), try parseChunk("4096"));
    try std.testing.expectEqual(batch.max_chunk, try parseChunk("16384"));
    for ([_][]const u8{ "0", "16385", "-1", "4k", "" }) |bad| try std.testing.expectError(error.BadChunk, parseChunk(bad));
}

test "TF_FLASHNEXT_CHUNK_SOLO is off unset, takes the busy chunk to the largest pass and refuses the rest" {
    try std.testing.expectEqual(@as(usize, 1536), try parseSolo(null, 1536));
    try std.testing.expectEqual(@as(usize, 8192), try parseSolo("8192", 1536));
    try std.testing.expectEqual(@as(usize, 1536), try parseSolo("1536", 1536));
    for ([_][]const u8{ "1024", "0", "16385", "8k", "" }) |bad| try std.testing.expectError(error.BadChunk, parseSolo(bad, 1536));
}

test "a pass takes the solo rows unless another lane decodes; prompts still filling do not count" {
    const L = struct { filled: ?usize };
    var m: std.AutoHashMapUnmanaged(u32, L) = .empty;
    defer m.deinit(std.testing.allocator);
    try m.put(std.testing.allocator, 0, .{ .filled = 0 });
    const own = m.getPtr(0).?;
    try std.testing.expectEqual(@as(usize, 8192), passRows(&m, own, 8192, 1536));
    try m.put(std.testing.allocator, 1, .{ .filled = 3072 });
    const mine = m.getPtr(0).?;
    try std.testing.expectEqual(@as(usize, 8192), passRows(&m, mine, 8192, 1536));
    m.getPtr(1).?.filled = null;
    try std.testing.expectEqual(@as(usize, 1536), passRows(&m, m.getPtr(0).?, 8192, 1536));
    try std.testing.expectEqual(@as(usize, 1536), passRows(&m, m.getPtr(0).?, 1536, 1536));
}
