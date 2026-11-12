//! TF_GEMMA4_CHUNK_SOLO: whole grid chunks a solo prompt pass takes, on lanes.solo_chunk's rule.
const std = @import("std");
const solo_chunk = @import("lanes").solo_chunk;

/// The grid chunk (a pass beside a decode), and the widest solo pass: the rings and the prompt scratch grow with it.
pub const limits: solo_chunk.Limits = .{ .busy = grid, .most = 4096, .grid = grid };
const grid = @import("cuda_forward.zig").prompt_rows;

pub fn rows() !usize {
    return solo_chunk.rows("TF_GEMMA4_CHUNK_SOLO", limits);
}

test "TF_GEMMA4_CHUNK_SOLO takes whole grid chunks up to 4096" {
    try std.testing.expectEqual(@as(usize, 512), try solo_chunk.parse(null, limits));
    try std.testing.expectEqual(@as(usize, 2048), try solo_chunk.parse("2048", limits));
    try std.testing.expectError(error.BadChunkSolo, solo_chunk.parse("1000", limits));
    try std.testing.expectError(error.BadChunkSolo, solo_chunk.parse("8192", limits));
}

test "TF_GEMMA4_CHUNK_SOLO reads through environment.zig: unset and empty off, a grid value kept, the rest refused" {
    const env = @import("sys").env;
    const name = "TF_GEMMA4_CHUNK_SOLO";
    defer _ = env.unsetenv(name);
    _ = env.unsetenv(name);
    try std.testing.expectEqual(@as(usize, 512), try rows());
    for ([_]struct { [*:0]const u8, ?usize }{ .{ "", 512 }, .{ " \t", 512 }, .{ "2048", 2048 }, .{ "1000", null }, .{ "8192", null }, .{ "2k", null } }) |c| {
        try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, c[0], 1));
        if (c[1]) |n| try std.testing.expectEqual(n, try rows()) else try std.testing.expectError(error.BadChunkSolo, rows());
    }
    const why = "TF_GEMMA4_CHUNK_SOLO: a solo prompt pass's rows, a multiple of 512 up to 4096";
    try std.testing.expectEqualStrings(why, @import("cuda_native.zig").explain(null, error.BadChunkSolo).?);
}

test "TF_GEMMA4_CHUNK_SOLO keeps the old reader's verdict on every nonblank text; blank text is unset" {
    const name = "TF_GEMMA4_CHUNK_SOLO";
    for ([_][]const u8{ "512", "1024", "2048", "4096", "+2048", "002048", "2_048", "1000", "256", "0", "4608", "8192", "-1", "-0", "2k", " 2048", "2048 ", "0x800", "+", "18446744073709551616", "1\nwrong" }) |v| {
        const old = solo_chunk.parse(v, limits) catch null;
        if (old) |n| try std.testing.expectEqual(n, try solo_chunk.read(name, v, limits)) else try std.testing.expectError(error.BadChunkSolo, solo_chunk.read(name, v, limits));
    }
    for ([_][]const u8{ "", " ", "\t \n" }) |v| {
        try std.testing.expectError(error.BadChunkSolo, solo_chunk.parse(v, limits));
        try std.testing.expectEqual(@as(usize, 512), try solo_chunk.read(name, v, limits));
    }
    try std.testing.expectEqual(@as(usize, 512), try solo_chunk.read(name, null, limits));
}
