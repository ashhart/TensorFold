//! Prompt rows a pass: TF_FLASHNEXT_CHUNK while another stream decodes, TF_FLASHNEXT_CHUNK_SOLO when none does.
const std = @import("std");
const batch = @import("cuda_batch.zig");
const solo_chunk = @import("lanes").solo_chunk;

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

/// A pass's rows on a server of one stream: no request waits on a pass there, and this takes most of 16,384's gain.
pub const lone_rows: usize = 8192;

/// TF_FLASHNEXT_CHUNK_SOLO's rows: unset, `lone_rows` when `lone`, else `busy` (off); below `busy`, refused.
pub fn parseSolo(v: ?[]const u8, busy: usize, lone: bool) !usize {
    if (v == null and lone) return @max(busy, lone_rows);
    return solo_chunk.parse(v, .{ .busy = busy, .most = batch.max_chunk }) catch error.BadChunk;
}

pub fn soloRows(busy: usize, lone: bool) !usize {
    return parseSolo(if (std.c.getenv("TF_FLASHNEXT_CHUNK_SOLO")) |v| std.mem.span(v) else null, busy, lone);
}

/// The rows the pass of stream `own` takes when it starts: lanes.solo_chunk's rule on the lanes' `filled`.
pub fn passRows(lanes: anytype, own: anytype, solo: usize, busy: usize) usize {
    return solo_chunk.passRows(lanes, own, "filled", solo, busy);
}

/// Rows from which a read-out takes the tiles (TF_FLASHNEXT_READOUT_ROWS, default 64, 0 never), read once at load.
pub fn readoutTileRows() usize {
    // the startup check refused every malformed value already, so a plain default stands here
    const r = @import("sys").environment.read(usize, "TF_FLASHNEXT_READOUT_ROWS", 64) catch 64;
    return if (r == 0) std.math.maxInt(usize) else r;
}

test "TF_FLASHNEXT_CHUNK takes 1 to the largest measured pass and refuses the rest" {
    try std.testing.expectEqual(batch.default_chunk, try parseChunk(null));
    try std.testing.expectEqual(@as(usize, 4096), try parseChunk("4096"));
    try std.testing.expectEqual(batch.max_chunk, try parseChunk("16384"));
    for ([_][]const u8{ "0", "16385", "-1", "4k", "" }) |bad| try std.testing.expectError(error.BadChunk, parseChunk(bad));
}

test "TF_FLASHNEXT_CHUNK_SOLO is off unset, takes the busy chunk to the largest pass and refuses the rest" {
    try std.testing.expectEqual(@as(usize, 1536), try parseSolo(null, 1536, false));
    try std.testing.expectEqual(@as(usize, 8192), try parseSolo("8192", 1536, false));
    try std.testing.expectEqual(@as(usize, 1536), try parseSolo("1536", 1536, false));
    for ([_][]const u8{ "1024", "0", "16385", "8k", "" }) |bad| try std.testing.expectError(error.BadChunk, parseSolo(bad, 1536, false));
}

test "a server of one stream takes the larger pass unless the variable or a larger busy chunk says otherwise" {
    try std.testing.expectEqual(lone_rows, try parseSolo(null, 1536, true));
    try std.testing.expectEqual(@as(usize, 1536), try parseSolo("1536", 1536, true));
    try std.testing.expectEqual(@as(usize, 16384), try parseSolo(null, 16384, true));
}

test "a pass takes the solo rows unless another lane decodes; prompts still filling do not count" {
    const L = struct { filled: ?usize };
    var m: std.AutoHashMapUnmanaged(u32, L) = .empty;
    defer m.deinit(std.testing.allocator);
    try m.put(std.testing.allocator, 0, .{ .filled = 0 });
    try std.testing.expectEqual(@as(usize, 8192), passRows(&m, 0, 8192, 1536));
    try m.put(std.testing.allocator, 1, .{ .filled = 3072 });
    try std.testing.expectEqual(@as(usize, 8192), passRows(&m, 0, 8192, 1536));
    m.getPtr(1).?.filled = null;
    try std.testing.expectEqual(@as(usize, 1536), passRows(&m, 0, 8192, 1536));
    try std.testing.expectEqual(@as(usize, 1536), passRows(&m, 0, 1536, 1536));
}
