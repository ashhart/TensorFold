const std = @import("std");
const Kind = enum { mlx, jpeg, source };
const Entry = struct { path: []const u8, sha256: []const u8 };
const Receipt = struct {
    format: u32 = 1,
    kind: Kind,
    revision: []const u8,
    bridge_revision: ?[]const u8 = null,
    files: []const Entry,
};
const receipt_name = "tensorfold-install.json";

fn relevant(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "include/") or
        std.mem.startsWith(u8, path, "lib/") or
        std.mem.startsWith(u8, path, "share/cmake/");
}

fn safePath(path: []const u8) bool {
    if (path.len == 0 or std.fs.path.isAbsolute(path)) return false;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| if (part.len == 0 or std.mem.eql(u8, part, "..") or std.mem.eql(u8, part, ".")) return false;
    return true;
}

fn hashFile(io: std.Io, dir: std.Io.Dir, path: []const u8) ![64]u8 {
    const file = try dir.openFile(io, path, .{});
    defer file.close(io);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [65536]u8 = undefined;
    var offset: u64 = 0;
    while (true) {
        const count = try file.readPositional(io, &.{&buffer}, offset);
        if (count == 0) break;
        hash.update(buffer[0..count]);
        offset += count;
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn inventory(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, kind: Kind) ![]const Entry {
    var result: std.ArrayList(Entry) = .empty;
    var walk = try dir.walk(a);
    defer walk.deinit();
    while (try walk.next(io)) |entry| {
        if (entry.kind == .directory or std.mem.eql(u8, entry.path, receipt_name)) continue;
        if (kind != .source and !relevant(entry.path)) continue;
        const digest = if (entry.kind == .sym_link) blk: {
            var buffer: [4096]u8 = undefined;
            const target = buffer[0..try entry.dir.readLink(io, entry.basename, &buffer)];
            if (std.fs.path.isAbsolute(target)) return error.ExternalInstalledSymlink;
            const resolved = try std.fs.path.resolve(a, &.{ "/tensorfold-source-root", std.fs.path.dirname(entry.path) orelse ".", target });
            if (!std.mem.startsWith(u8, resolved, "/tensorfold-source-root/")) return error.ExternalInstalledSymlink;
            const content = try std.fmt.allocPrint(a, "symlink:{s}", .{target});
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(content, &digest, .{});
            break :blk std.fmt.bytesToHex(digest, .lower);
        } else if (entry.kind == .file) try hashFile(io, dir, entry.path) else return error.UnexpectedInstalledFileType;
        try result.append(a, .{ .path = try a.dupe(u8, entry.path), .sha256 = try a.dupe(u8, &digest) });
    }
    std.mem.sort(Entry, result.items, {}, struct {
        fn less(_: void, x: Entry, y: Entry) bool {
            return std.mem.lessThan(u8, x.path, y.path);
        }
    }.less);
    return result.items;
}

fn required(kind: Kind, files: []const Entry) !void {
    const names: []const []const u8 = switch (kind) {
        .mlx => &.{ "lib/libmlx.dylib", "lib/libmlxc.dylib", "lib/libjaccl.dylib", "lib/mlx.metallib", "include/mlx/c/mlx.h", "share/cmake/MLX/MLXConfigVersion.cmake", "share/cmake/MLXC/MLXCConfigVersion.cmake" },
        .jpeg => &.{ "lib/libturbojpeg.a", "include/turbojpeg.h", "lib/pkgconfig/libturbojpeg.pc" },
        .source => &.{},
    };
    for (names) |name| {
        var found = false;
        for (files) |file| if (std.mem.eql(u8, file.path, name)) {
            found = true;
            break;
        };
        if (!found) {
            if (!@import("builtin").is_test) std.debug.print("Missing installed dependency artifact: {s}\n", .{name});
            return error.IncompleteNativeInstallation;
        }
    }
}

pub fn record(a: std.mem.Allocator, io: std.Io, prefix: []const u8, kind: Kind, revision: []const u8, bridge: ?[]const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, prefix, .{ .iterate = true });
    defer dir.close(io);
    const files = try inventory(a, io, dir, kind);
    if (files.len == 0) return error.EmptyInstallation;
    try required(kind, files);
    const receipt: Receipt = .{ .kind = kind, .revision = revision, .bridge_revision = bridge, .files = files };
    const content = try std.json.Stringify.valueAlloc(a, receipt, .{ .whitespace = .indent_2 });
    const file = try dir.createFile(io, receipt_name, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, content);
    std.debug.print("Recorded {s} installation: {d} hashed artifacts\n", .{ prefix, files.len });
}

fn compare(receipt: Receipt, kind: Kind, revision: []const u8, bridge: ?[]const u8, files: []const Entry) !void {
    if (receipt.format != 1 or receipt.kind != kind or !std.mem.eql(u8, receipt.revision, revision)) return error.NativeSourceRevisionMismatch;
    if (!std.mem.eql(u8, receipt.bridge_revision orelse "", bridge orelse "")) return error.NativeBridgeRevisionMismatch;
    try required(kind, receipt.files);
    if (receipt.files.len != files.len) return error.NativeInstallationChanged;
    for (receipt.files, files) |expected, actual| {
        if (!safePath(expected.path)) return error.InvalidReceiptPath;
        if (!std.mem.eql(u8, expected.path, actual.path) or !std.mem.eql(u8, expected.sha256, actual.sha256)) {
            if (!@import("builtin").is_test) std.debug.print("Installed dependency changed: {s}\n", .{expected.path});
            return error.NativeInstallationChanged;
        }
    }
}

pub fn verify(a: std.mem.Allocator, io: std.Io, prefix: []const u8, kind: Kind, revision: []const u8, bridge: ?[]const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, prefix, .{ .iterate = true });
    defer dir.close(io);
    const bytes = dir.readFileAlloc(io, receipt_name, a, .limited(4 * 1024 * 1024)) catch |err| {
        std.debug.print("Missing build receipt in {s}; rerun native setup.\n", .{prefix});
        return err;
    };
    const parsed = try std.json.parseFromSlice(Receipt, a, bytes, .{});
    try compare(parsed.value, kind, revision, bridge, try inventory(a, io, dir, kind));
    std.debug.print("PASS: {s} source revisions and {d} installed artifact hashes\n", .{ prefix, parsed.value.files.len });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    var mlx: []const u8 = "build/mlx";
    var jpeg: []const u8 = "build/jpeg";
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        if (i + 1 == args.len) return error.MissingArgument;
        if (std.mem.eql(u8, args[i], "--mlx-prefix")) mlx = args[i + 1] else if (std.mem.eql(u8, args[i], "--jpeg-prefix")) jpeg = args[i + 1] else return error.InvalidArgument;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, "native/dependencies.json", a, .limited(16384));
    const pins = (try std.json.parseFromSlice(std.json.Value, a, bytes, .{})).value.object;
    try verify(a, init.io, mlx, .mlx, pins.get("mlx_revision").?.string, pins.get("mlx_c_revision").?.string);
    try verify(a, init.io, jpeg, .jpeg, pins.get("jpeg_version").?.string, null);
}

test "receipt rejects stale bridge revisions and absent installations" {
    const receipt: Receipt = .{ .kind = .mlx, .revision = "mlx", .bridge_revision = "old", .files = &.{} };
    try std.testing.expectError(error.NativeBridgeRevisionMismatch, compare(receipt, .mlx, "mlx", "new", &.{}));
    try std.testing.expectError(error.NativeSourceRevisionMismatch, compare(receipt, .mlx, "new", "old", &.{}));
    try std.testing.expectError(error.IncompleteNativeInstallation, compare(receipt, .mlx, "mlx", "old", &.{}));
    try std.testing.expect(!safePath("lib/../../outside"));
    try std.testing.expect(!safePath("/lib/libmlx.dylib"));
}

test "receipt rejects changed removed and added artifacts" {
    const files = [_]Entry{
        .{ .path = "include/turbojpeg.h", .sha256 = "header" },
        .{ .path = "lib/libturbojpeg.a", .sha256 = "library" },
        .{ .path = "lib/pkgconfig/libturbojpeg.pc", .sha256 = "version" },
    };
    const receipt: Receipt = .{ .kind = .jpeg, .revision = "3.1", .files = &files };
    try compare(receipt, .jpeg, "3.1", null, &files);
    var changed = files;
    changed[1].sha256 = "corrupt";
    try std.testing.expectError(error.NativeInstallationChanged, compare(receipt, .jpeg, "3.1", null, &changed));
    try std.testing.expectError(error.NativeInstallationChanged, compare(receipt, .jpeg, "3.1", null, files[0..2]));
    const added = files ++ [_]Entry{.{ .path = "lib/extra.a", .sha256 = "extra" }};
    try std.testing.expectError(error.NativeInstallationChanged, compare(receipt, .jpeg, "3.1", null, &added));
}
