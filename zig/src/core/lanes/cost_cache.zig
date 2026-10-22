//! Timed costs are measured once per engine build, chip and model shape, then read back by later loads.
const std = @import("std");
const platform = @import("sys");
const Allocator = std.mem.Allocator;
const identity = @import("cost_identity.zig");
pub const KeySeed = identity.Seed;

const magic = "TFWC2\n";

/// What the costs depend on: the running executable's bytes (the layout that wrote them) and the caller's parts.
pub fn key(a: Allocator, io: std.Io, parts: []const []const u8) ![32]u8 {
    return keyTracing(a, io, parts, false);
}

pub fn keyTracing(a: Allocator, io: std.Io, parts: []const []const u8, trace: bool) ![32]u8 {
    return (try keySeed(a, io, trace)).key(parts);
}

/// One executable read can seed several head keys within the same model load.
pub fn keySeed(a: Allocator, io: std.Io, trace: bool) !KeySeed {
    const exe = try std.process.executablePathAlloc(io, a);
    defer a.free(exe);
    const file = try std.Io.Dir.cwd().openFile(io, exe, .{});
    defer file.close(io);
    const seed = try identity.fromFile(file, io, trace);
    if (seed.executable_sha256) |digest| {
        const hex = std.fmt.bytesToHex(digest, .lower);
        std.log.info("window cost key part executable {s}", .{hex[0..12]});
    }
    return seed;
}

pub fn buildKey(exe: []const u8, parts: []const []const u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    hashPart(&h, exe);
    for (parts) |p| hashPart(&h, p);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// The same parts without the executable: the last table any build measured on this chip and model shape.
pub fn referenceKey(parts: []const []const u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("window costs reference\n");
    for (parts) |p| hashPart(&h, p);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// CUDA's existing reference keys delimit parts with newlines; keep their bytes across shared-cache builds.
pub fn cudaReferenceKey(parts: []const []const u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("window costs reference\n");
    for (parts) |p| {
        h.update(p);
        h.update("\n");
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn hashPart(h: *std.crypto.hash.sha2.Sha256, part: []const u8) void {
    identity.hashPart(h, part);
}

fn path(a: Allocator, k: [32]u8, suffix: []const u8, root: ?[]const u8) !?[]u8 {
    if (root) |dir| return try std.fmt.allocPrint(a, "{s}/{s}{s}", .{ dir, std.fmt.bytesToHex(k, .lower), suffix });
    const home = platform.c.getenv("HOME") orelse return null;
    return try std.fmt.allocPrint(a, "{s}/.cache/tensorfold/window-costs/{s}{s}", .{ std.mem.span(home), std.fmt.bytesToHex(k, .lower), suffix });
}

/// The value kept for `k`, or null when none is kept or the file is not one of ours.
pub fn load(comptime T: type, a: Allocator, io: std.Io, k: [32]u8) ?T {
    return loadAt(T, a, io, k, null);
}

/// The same cache envelope in an explicit directory; null keeps the standard user cache.
pub fn loadAt(comptime T: type, a: Allocator, io: std.Io, k: [32]u8, root: ?[]const u8) ?T {
    const p = (path(a, k, ".bin", root) catch return null) orelse return null;
    defer a.free(p);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(1 << 20)) catch return null;
    defer a.free(bytes);
    return parse(T, bytes);
}

fn parse(comptime T: type, bytes: []const u8) ?T {
    if (bytes.len != magic.len + @sizeOf(T) or !std.mem.eql(u8, bytes[0..magic.len], magic)) return null;
    var value: T = undefined;
    @memcpy(std.mem.asBytes(&value), bytes[magic.len..]);
    return value;
}

/// Keep `value` for `k`, written whole and renamed into place; a failure only costs the next load a measurement.
pub fn save(comptime T: type, a: Allocator, io: std.Io, k: [32]u8, value: T) void {
    saveAt(T, a, io, k, value, null);
}

/// Atomic publication in an explicit directory, sharing the default cache format and failure policy.
pub fn saveAt(comptime T: type, a: Allocator, io: std.Io, k: [32]u8, value: T, root: ?[]const u8) void {
    const p = (path(a, k, ".bin", root) catch return) orelse return;
    defer a.free(p);
    const bytes = std.mem.concat(a, u8, &.{ magic, std.mem.asBytes(&value) }) catch return;
    defer a.free(bytes);
    var temporary = std.Io.Dir.cwd().createFileAtomic(io, p, .{ .make_path = true, .replace = true, .permissions = if (@import("builtin").os.tag == .windows) .default_file else @fromBackingInt(0o600) }) catch return;
    defer temporary.deinit(io);
    temporary.file.writePositionalAll(io, bytes, 0) catch return;
    temporary.replace(io) catch return;
}

test "a kept value reads back whole and any other length or tag is refused" {
    const V = struct { n: u32, ms: [3]f64 };
    const v: V = .{ .n = 3, .ms = .{ 33.56, 34.1, 37.3 } };
    const bytes = try std.mem.concat(std.testing.allocator, u8, &.{ magic, std.mem.asBytes(&v) });
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualDeep(v, parse(V, bytes).?);
    try std.testing.expectEqual(@as(?V, null), parse(V, bytes[0 .. bytes.len - 1]));
    bytes[0] = 'X';
    try std.testing.expectEqual(@as(?V, null), parse(V, bytes));
}

test "shared build identity separates executable, backend, chip and model shape" {
    const parts = [_][]const u8{ "nemotron-cuda", "NVIDIA GB10", "sm121 cuda13000 23/2688/131072/128/16384 draft1 own" };
    const k = buildKey("engine-bytes", &parts);
    try std.testing.expectEqualStrings("7bde80f25c6c8b5bc0ebeb0076bfcd874bf7d7d7382b12599b42c91b25e4b23f", &std.fmt.bytesToHex(k, .lower));
    try std.testing.expect(!std.mem.eql(u8, &k, &buildKey("other-engine", &parts)));
    for (0..parts.len) |i| {
        var changed = parts;
        changed[i] = "another part";
        try std.testing.expect(!std.mem.eql(u8, &k, &buildKey("engine-bytes", &changed)));
    }
    try std.testing.expect(!std.mem.eql(u8, &buildKey("e", &.{ "ab", "c" }), &buildKey("e", &.{ "a", "bc" })));
}

test "CUDA reference identity retains the existing newline-delimited hash" {
    const parts = [_][]const u8{ "nemotron-cuda", "NVIDIA GB10", "sm121 cuda13000 23/2688/131072/128/16384 draft1 own" };
    const k = cudaReferenceKey(&parts);
    try std.testing.expectEqualStrings("05bf67c358f200cc54686bcaa5890a607f39f392c8059d84314fdc24608102ff", &std.fmt.bytesToHex(k, .lower));
    try std.testing.expect(!std.mem.eql(u8, &k, &referenceKey(&parts)));
}
