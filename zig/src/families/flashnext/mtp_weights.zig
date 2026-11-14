//! TF_FLASHNEXT_MTP_WEIGHTS: how the MTP drafter holds its BF16 linears; only the CUDA loader re-codes them.
const std = @import("std");
const environment = @import("sys").environment;

const name = "TF_FLASHNEXT_MTP_WEIGHTS";

/// How the drafter's linears are held in memory; the order is the setting's accepted words.
pub const Format = enum { checkpoint, mxfp8 };

/// What an unset or blank TF_FLASHNEXT_MTP_WEIGHTS selects: the rows as the checkpoint stores them.
pub const default_format: Format = .checkpoint;

/// The setting's text as its reader takes it: unset or blank is the default, anything malformed is refused by name.
pub fn parse(raw: ?[]const u8) !Format {
    return @fromBackingInt(@intCast(try environment.value(u1, name, raw, @backingInt(default_format))));
}

pub fn format() !Format {
    return @fromBackingInt(@intCast(try environment.read(u1, name, @backingInt(default_format))));
}

/// A backend with no MXFP8 projection for the drafter stops the start on mxfp8 rather than loading the stored rows.
pub fn refuse(backend: []const u8, weights: Format) !void {
    if (weights == .checkpoint) return;
    if (!@import("builtin").is_test) std.log.err("{s}=mxfp8: Flash Next on {s} has no MXFP8 projection for the drafter; unset it or set checkpoint", .{ name, backend });
    return error.InvalidEnvironment;
}

test "unset and blank read checkpoint, the two words read as themselves, and other text is refused" {
    for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| try std.testing.expectEqual(Format.checkpoint, try parse(raw));
    try std.testing.expectEqual(Format.checkpoint, try parse("checkpoint"));
    try std.testing.expectEqual(Format.mxfp8, try parse("mxfp8"));
    for ([_][]const u8{ "MXFP8", "fp8", "1", "mxfp8 ", "bf16", "checkpoints" }) |raw| try std.testing.expectError(error.InvalidEnvironment, parse(raw));
}

test "Metal and HIP stop the start on mxfp8 and keep checkpoint" {
    try std.testing.expectError(error.InvalidEnvironment, refuse("Metal", .mxfp8));
    try refuse("HIP", .checkpoint);
    // each backend's load asks before its first checkpoint or device work
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("engine_load.zig"), "try @import(\"mtp_weights.zig\").refuse(\"Metal\", try @import(\"mtp_weights.zig\").format());") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("hip/load.zig"), "try @import(\"../mtp_weights.zig\").refuse(\"HIP\", try @import(\"../mtp_weights.zig\").format());") != null);
}
