//! Checkpoint provenance named by a Hugging Face cache snapshot path; local conversions remain unnamed.
const std = @import("std");
pub const Source = struct { repo: []const u8, revision: []const u8 };

pub fn source(a: std.mem.Allocator, dir: []const u8) std.mem.Allocator.Error!?Source {
    const prefix = "models--";
    const at = std.mem.lastIndexOf(u8, dir, prefix) orelse return null;
    if (at > 0 and dir[at - 1] != '/') return null;
    const rest = dir[at + prefix.len ..];
    const marker = "/snapshots/";
    const sep = std.mem.indexOf(u8, rest, marker) orelse return null;
    const owner = rest[0..sep];
    if (std.mem.indexOfScalar(u8, owner, '/') != null) return null;
    const dash = std.mem.indexOf(u8, owner, "--") orelse return null;
    if (dash == 0 or dash + 2 == owner.len) return null;
    const tail = rest[sep + marker.len ..];
    const revision = tail[0 .. std.mem.indexOfScalar(u8, tail, '/') orelse tail.len];
    if (revision.len == 0) return null;
    return .{ .repo = try std.fmt.allocPrint(a, "{s}/{s}", .{ owner[0..dash], owner[dash + 2 ..] }), .revision = revision };
}
