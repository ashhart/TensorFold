//! Checkpoint provenance named by a Hugging Face cache snapshot path; local conversions remain unnamed.
const std = @import("std");
pub const Source = struct { repo: []const u8, revision: []const u8 };

pub fn source(a: std.mem.Allocator, dir: []const u8) std.mem.Allocator.Error!?Source {
    return sourcePath(a, dir, if (@import("builtin").os.tag == .windows) "/\\" else "/");
}

fn sourcePath(a: std.mem.Allocator, dir: []const u8, comptime separators: []const u8) std.mem.Allocator.Error!?Source {
    const prefix = "models--";
    const at = std.mem.lastIndexOf(u8, dir, prefix) orelse return null;
    if (at > 0 and std.mem.indexOfScalar(u8, separators, dir[at - 1]) == null) return null;
    const rest = dir[at + prefix.len ..];
    const sep = std.mem.indexOfAny(u8, rest, separators) orelse return null;
    const owner = rest[0..sep];
    const next = rest[sep + 1 ..];
    const marker = "snapshots";
    if (!std.mem.startsWith(u8, next, marker) or next.len <= marker.len or
        std.mem.indexOfScalar(u8, separators, next[marker.len]) == null) return null;
    const dash = std.mem.indexOf(u8, owner, "--") orelse return null;
    if (dash == 0 or dash + 2 == owner.len) return null;
    const tail = next[marker.len + 1 ..];
    const revision = tail[0 .. std.mem.indexOfAny(u8, tail, separators) orelse tail.len];
    if (revision.len == 0) return null;
    return .{ .repo = try std.fmt.allocPrint(a, "{s}/{s}", .{ owner[0..dash], owner[dash + 2 ..] }), .revision = revision };
}

test "snapshot provenance binds the repo and revision with POSIX and Windows separators" {
    inline for (.{
        .{ "/", "/cache/models--example-org--Synthetic-HC-Model/snapshots/01234567" },
        .{ "/\\", "C:\\cache\\models--example-org--Synthetic-HC-Model\\snapshots\\01234567" },
        .{ "/\\", "C:\\cache/models--example-org--Synthetic-HC-Model/snapshots\\01234567/subdir" },
        .{ "/\\", "\\\\server\\cache\\models--example-org--Synthetic-HC-Model\\snapshots\\01234567" },
    }) |case| {
        const value = (try sourcePath(std.testing.allocator, case[1], case[0])) orelse return error.TestUnexpectedResult;
        defer std.testing.allocator.free(value.repo);
        try std.testing.expectEqualStrings("example-org/Synthetic-HC-Model", value.repo);
        try std.testing.expectEqualStrings("01234567", value.revision);
    }
}

test "local paths and lookalike components do not invent snapshot provenance" {
    inline for (.{
        .{ "/", "/local/conversion" },
        .{ "/", "/cache/not-models--org--repo/snapshots/01234567" },
        .{ "/", "/cache/models--org--repo/snapshots-extra/01234567" },
        .{ "/\\", "C:\\cache\\models--org--repo\\snapshots\\" },
        .{ "/", "C:\\cache\\models--org--repo\\snapshots\\01234567" },
    }) |case| {
        const value = try sourcePath(std.testing.allocator, case[1], case[0]);
        defer if (value) |v| std.testing.allocator.free(v.repo);
        try std.testing.expect(value == null);
    }
}
