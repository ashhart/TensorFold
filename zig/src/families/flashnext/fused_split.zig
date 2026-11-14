//! TF_FLASHNEXT_FUSED_SPLIT: wide split-K lane calls on the fused form, the cluster's bits; only CUDA has that form.
const std = @import("std");
const environment = @import("sys").environment;

const name = "TF_FLASHNEXT_FUSED_SPLIT";

/// The narrowest call, in output columns, that takes the fused form when the setting is on.
pub const min_columns: usize = 2048;

/// The setting's text as its reader takes it: unset or blank is off, anything malformed is refused by name.
pub fn parse(raw: ?[]const u8) !bool {
    return environment.value(bool, name, raw, false);
}

pub fn on() !bool {
    return environment.read(bool, name, false);
}

/// The CUDA lane's threshold, cuda.qmmf.fuse_split_n: min_columns when on, 0 (never) when off.
pub fn columns(enabled: bool) usize {
    return if (enabled) min_columns else 0;
}

/// A backend with no fused split-K form stops the start on 1 rather than running its lane calls as before.
pub fn refuse(backend: []const u8, enabled: bool) !void {
    if (!enabled) return;
    if (!@import("builtin").is_test) std.log.err("{s}=1: Flash Next on {s} has no fused split-K form for its lane calls; unset it or set 0", .{ name, backend });
    return error.InvalidEnvironment;
}

test "unset and blank read off, 0 and 1 read as themselves, and other text is refused" {
    for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| try std.testing.expect(!try parse(raw));
    try std.testing.expect(!try parse("0"));
    try std.testing.expect(try parse("1"));
    for ([_][]const u8{ "2", "yes", "on", "true", "00", "1 " }) |raw| try std.testing.expectError(error.InvalidEnvironment, parse(raw));
}

test "off leaves the lane's threshold at 0, so every lane call keeps main's kernel" {
    try std.testing.expectEqual(@as(usize, 0), columns(false));
    try std.testing.expectEqual(min_columns, columns(true));
    const set = "cuda.qmmf.fuse_split_n = @import(\"fused_split.zig\").columns(try @import(\"fused_split.zig\").on());";
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_prompt.zig"), set) != null);
}

test "Metal and HIP stop the start on 1 and keep 0" {
    try std.testing.expectError(error.InvalidEnvironment, refuse("Metal", true));
    try refuse("HIP", false);
    // each backend's load asks before its first checkpoint or device work
    const metal = "try @import(\"fused_split.zig\").refuse(\"Metal\", try @import(\"fused_split.zig\").on());";
    const hip = "try @import(\"../fused_split.zig\").refuse(\"HIP\", try @import(\"../fused_split.zig\").on());";
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("engine_load.zig"), metal) != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("hip/load.zig"), hip) != null);
}
