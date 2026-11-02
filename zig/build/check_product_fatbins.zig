//! Product check (-Dgpu with -Dfatbins): each prebuilt image holds exactly the one SASS profile its kernel needs.
const std = @import("std");
const fatbins = @import("check_fatbins.zig");

/// A required profile as the build passes it: the SM number, `a` for arch-specific ("120", "121a").
pub fn parseProfile(text: []const u8) !struct { sm: u32, arch_specific: bool } {
    const a = std.mem.endsWith(u8, text, "a");
    return .{ .sm = try std.fmt.parseInt(u32, text[0 .. text.len - @intFromBool(a)], 10), .arch_specific = a };
}

/// Null when `got` is exactly one ELF entry of the required SM and flag, else what is wrong.
pub fn verdict(got: []const fatbins.Entry, sm: u32, arch_specific: bool) ?[]const u8 {
    if (got.len == 0) return "holds no image";
    if (got.len > 1) return "holds more than its one SASS image (another entry, a duplicate or PTX)";
    const e = got[0];
    if (e.kind != 2) return "holds PTX, not a SASS image";
    if (e.sm != sm) return "holds another SM's image";
    if (e.arch_specific != arch_specific) return if (arch_specific) "holds the plain profile; this kernel needs the arch-specific one (sm_<n>a)" else "holds the arch-specific profile; this kernel needs the plain one";
    return null;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3 or (args.len - 1) % 2 != 0) return error.Usage;
    var failed = false;
    var buf: [64]fatbins.Entry = undefined;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const want = try parseProfile(args[i]);
        const path = args[i + 1];
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30)) catch |e| {
            std.debug.print("{s}: missing or unreadable ({t}); -Dfatbins needs every image\n", .{ path, e });
            failed = true;
            continue;
        };
        defer gpa.free(bytes);
        const got = fatbins.entries(bytes, &buf) catch |e| {
            std.debug.print("{s}: not a CUDA fatbin ({t})\n", .{ path, e });
            failed = true;
            continue;
        };
        if (verdict(got, want.sm, want.arch_specific)) |why| {
            std.debug.print("{s}: {s}; a -Dgpu build for sm_{d} needs exactly sm_{d}{s}\n", .{ path, why, want.sm, want.sm, if (want.arch_specific) "a" else "" });
            failed = true;
        }
    }
    if (failed) std.process.exit(1);
}

fn check(list: []const fatbins.TestEntry, sm: u32, a: bool) !?[]const u8 {
    var file = try fatbins.testFile(list);
    defer file.deinit(std.testing.allocator);
    var buf: [8]fatbins.Entry = undefined;
    return verdict(try fatbins.entries(file.items, &buf), sm, a);
}

test "verdict: exactly one SASS entry of the right SM and flag" {
    try std.testing.expectEqual(@as(?[]const u8, null), try check(&.{.{ .sm = 120, .arch_specific = true }}, 120, true));
    try std.testing.expectEqual(@as(?[]const u8, null), try check(&.{.{ .sm = 120, .arch_specific = false }}, 120, false));
    try std.testing.expect(try check(&.{.{ .sm = 120, .arch_specific = false }}, 120, true) != null); // a generic ck_experts image
    try std.testing.expect(try check(&.{.{ .sm = 120, .arch_specific = true }}, 120, false) != null);
    try std.testing.expect(try check(&.{.{ .sm = 121, .arch_specific = false }}, 120, false) != null);
    try std.testing.expect(try check(&.{ .{ .sm = 120, .arch_specific = false }, .{ .sm = 121, .arch_specific = false } }, 120, false) != null);
    try std.testing.expect(try check(&.{ .{ .sm = 120, .arch_specific = false }, .{ .sm = 120, .arch_specific = false } }, 120, false) != null);
    try std.testing.expect(try check(&.{ .{ .sm = 120, .arch_specific = false }, .{ .sm = 121, .arch_specific = false, .kind = 1 } }, 120, false) != null);
    try std.testing.expect(verdict(&.{}, 120, false) != null);
    const p = try parseProfile("121a");
    try std.testing.expect(p.sm == 121 and p.arch_specific);
}
