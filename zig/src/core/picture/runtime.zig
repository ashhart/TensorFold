//! Optional codec diagnostics use the same symbols and exact version checks as image decoding.
const std = @import("std");
const libs = @import("sys").optional_libraries;
const picture = @import("picture.zig");
const codecs = .{ @import("jpeg.zig"), @import("webp.zig") };
const ids = .{ libs.Id.jpeg, libs.Id.webp };

pub fn warnings(io: std.Io, a: std.mem.Allocator) [2]?[]const u8 {
    var out: [2]?[]const u8 = .{ null, null };
    inline for (codecs, 0..) |codec, i| {
        codec.check(io, a) catch |err| {
            out[i] = picture.problem(err);
        };
    }
    return out;
}

/// Non-codec rows remain discovery only; codec usability also requires the decoder's pin and exports.
pub fn probe(a: std.mem.Allocator, io: std.Io) ![]libs.Status {
    const out = try libs.probe(a, io, @import("sys"));
    const problems = warnings(io, a);
    inline for (ids, 0..) |id, i| {
        const row = &out[@backingInt(id)];
        row.usable = problems[i] == null;
        row.problem = problems[i];
    }
    return out;
}
