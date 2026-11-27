//! Opt-in content identity for the target and drafter, using the existing checkpoint hashing contract.
const std = @import("std");
const platform = @import("sys");
const core = @import("core");
pub const Identity = struct { checkpoint_sha256: [64]u8, tokenizer_sha256: ?[64]u8, tokenizer_config_sha256: ?[64]u8 };
pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Identity {
    return .{ .checkpoint_sha256 = std.fmt.bytesToHex(try core.checkpoint_identity.fullWithCache(a, io, dir, null), .lower), .tokenizer_sha256 = try optional(a, io, dir, "tokenizer.json"), .tokenizer_config_sha256 = try optional(a, io, dir, "tokenizer_config.json") };
}
fn optional(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !?[64]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    const hash = platform.file_identity.hash(io, path, 1 << 30) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    return std.fmt.bytesToHex(hash, .lower);
}
pub fn drafter(a: std.mem.Allocator, io: std.Io, dir: []const u8, bits: u8) !void {
    if (comptime !@import("profile").round_attribution.enabled) return;
    const trace = @import("cuda").round_attribution.owner orelse return;
    const text = try std.json.Stringify.valueAlloc(a, .{ .identity = try read(a, io, dir), .bits = bits }, .{});
    defer a.free(text);
    try trace.annotate("drafter", text);
}
