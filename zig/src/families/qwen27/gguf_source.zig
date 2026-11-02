//! GGUF source selection and immutable prepared images keyed by complete source bytes and every builder identity.
const std = @import("std");
const gg = @import("../../core/gguf.zig");
const image = @import("gguf_image.zig");
const conf = @import("gguf_config.zig");
const Sha = std.crypto.hash.sha2.Sha256;
pub const Options = image.Options;
pub const Tokenizer = image.Tokenizer;
pub const Problem = image.Problem;
const names = [_][]const u8{ "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json" };
const sources = [_][]const u8{
    @embedFile("gguf_source.zig"),              @embedFile("gguf_image.zig"),            @embedFile("gguf_names.zig"),
    @embedFile("gguf_config.zig"),              @embedFile("config.zig"),                @embedFile("../flashnext/pack_io.zig"),
    @embedFile("../../core/gguf.zig"),          @embedFile("../../core/gguf_value.zig"), @embedFile("../../core/gguf_tensor.zig"),
    @embedFile("gguf_heads.zig"),               @embedFile("../../core/gguf_views.zig"), @embedFile("../../core/quant/convert.zig"),
    @embedFile("../../core/gguf_required.zig"),
};
/// Caller owns the path; a native safetensors folder keeps its existing path even if a GGUF also sits beside it.
pub fn find(a: std.mem.Allocator, io: std.Io, path: []const u8) !?[]u8 {
    if (std.mem.endsWith(u8, path, ".gguf")) {
        const f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        return try a.dupe(u8, path);
    }
    var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return null,
        else => return err,
    };
    defer dir.close(io);
    var one: ?[]u8 = null;
    errdefer if (one) |p| a.free(p);
    var count: usize = 0;
    var native = false;
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        if (std.mem.endsWith(u8, e.name, ".safetensors")) native = true;
        if (std.mem.endsWith(u8, e.name, ".gguf")) {
            count += 1;
            if (one == null) one = try std.fs.path.join(a, &.{ path, e.name });
        }
    }
    if (native) {
        if (one) |p| a.free(p);
        return null;
    }
    if (count > 1) return error.MultipleGgufFiles;
    return one;
}
pub fn key(d: *const gg.Document, options: Options) [32]u8 {
    var h = Sha.init(.{});
    h.update("TensorFold GGUF Qwen35 native image v1");
    h.update(@import("builtin").zig_version_string);
    for (sources) |s| {
        var hash: [32]u8 = undefined;
        Sha.hash(s, &hash, .{});
        h.update(&hash);
    }
    var hash: [32]u8 = undefined;
    Sha.hash(d.bytes, &hash, .{});
    h.update(&hash);
    Sha.hash(options.repacker.identity, &hash, .{});
    h.update(&hash);
    Sha.hash(options.tokenizer.identity, &hash, .{});
    h.update(&hash);
    return h.finalResult();
}
pub fn prepare(a: std.mem.Allocator, io: std.Io, path: []const u8, root: []const u8, options: Options) ![]u8 {
    if (options.problem) |problem| problem.len = 0;
    var f = try gg.File.open(a, io, path, .{});
    defer f.close(io);
    const identity = std.fmt.bytesToHex(key(&f.document, options), .lower);
    const dir = try std.fs.path.join(a, &.{ root, &identity });
    errdefer a.free(dir);
    if (try ready(a, io, dir, &identity)) return dir;
    try std.Io.Dir.cwd().createDirPath(io, root);
    const lock_name = try std.fmt.allocPrint(a, "{s}.lock", .{dir});
    defer a.free(lock_name);
    const lock = try std.Io.Dir.cwd().createFile(io, lock_name, .{ .truncate = false, .lock = .exclusive });
    defer lock.close(io);
    if (try ready(a, io, dir, &identity)) return dir;
    if (std.Io.Dir.cwd().openDir(io, dir, .{})) |existing| {
        existing.close(io);
        return error.CorruptGgufImage;
    } else |err| if (err != error.FileNotFound) return err;
    var random: [16]u8 = undefined;
    io.random(&random);
    const temp = try std.fmt.allocPrint(a, "{s}.part-{s}", .{ dir, &std.fmt.bytesToHex(random, .lower) });
    defer a.free(temp);
    try std.Io.Dir.cwd().createDir(io, temp, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, temp) catch {};
    try image.write(a, io, &f.document, temp, &identity, options);
    if (!std.mem.eql(u8, &identity, &std.fmt.bytesToHex(key(&f.document, options), .lower))) return error.GgufFileChanged;
    try seal(a, io, temp, &identity);
    if (!try ready(a, io, temp, &identity)) return error.IncompleteGgufImage;
    // Corrupt final entries are preserved for diagnosis; a caller must arrange their removal before rebuilding.
    std.Io.Dir.cwd().renamePreserve(temp, .cwd(), dir, io) catch |err| switch (err) {
        error.PathAlreadyExists => return error.CorruptGgufImage,
        else => return err,
    };
    return dir;
}
pub fn local(a: std.mem.Allocator, io: std.Io, home: []const u8, model: []const u8) !?[]u8 {
    return localWithProblem(a, io, home, model, null);
}
pub fn localWithProblem(a: std.mem.Allocator, io: std.Io, home: []const u8, model: []const u8, problem: ?*Problem) !?[]u8 {
    if (problem) |detail| detail.len = 0;
    const expanded = if (std.mem.startsWith(u8, model, "~/")) try std.fs.path.join(a, &.{ home, model[2..] }) else try a.dupe(u8, model);
    defer a.free(expanded);
    const path = (try find(a, io, expanded)) orelse return null;
    defer a.free(path);
    if (@import("builtin").os.tag != .macos) return error.GgufMetalOnly;
    var source_file = try gg.File.open(a, io, path, .{});
    defer source_file.close(io);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const value = try conf.build(arena.allocator(), &source_file.document);
    try @import("config.zig").qualified(try conf.parse(arena.allocator(), value));
    if (!std.fs.path.isAbsolute(home)) return error.CacheHomeMissing;
    const root = try std.fs.path.join(a, &.{ home, ".cache", "tensorfold", "gguf" });
    defer a.free(root);
    return try prepare(a, io, path, root, .{ .problem = problem });
}
fn files(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![][]const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    for (names) |n| try list.append(a, n);
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    defer a.free(data);
    const config = try std.json.parseFromSlice(std.json.Value, a, data, .{});
    defer config.deinit();
    const c = try conf.parse(a, config.value);
    for (0..c.layers + 1) |i| try list.append(a, try std.fmt.allocPrint(a, "gguf-{d:0>3}.safetensors", .{i}));
    return list.toOwnedSlice(a);
}
fn digest(io: std.Io, path: []const u8) ![32]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var rbuf: [64 << 10]u8 = undefined;
    var at: u64 = 0;
    var h = Sha.init(.{});
    while (true) {
        const n = try f.readPositionalAll(io, &rbuf, at);
        if (n == 0) break;
        h.update(rbuf[0..n]);
        at += n;
    }
    return h.finalResult();
}
fn seal(a: std.mem.Allocator, io: std.Io, dir: []const u8, identity: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const list = try files(scratch, io, dir);
    var manifest: std.json.Value = .{ .object = .empty };
    var hashes: std.json.Value = .{ .object = .empty };
    for (list) |n| {
        const path = try std.fs.path.join(scratch, &.{ dir, n });
        const hash = try digest(io, path);
        try hashes.object.put(scratch, n, .{ .string = try scratch.dupe(u8, &std.fmt.bytesToHex(hash, .lower)) });
        const f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        try f.sync(io);
    }
    try manifest.object.put(scratch, "identity", .{ .string = identity });
    try manifest.object.put(scratch, "files", hashes);
    try image.jsonFile(a, io, dir, "complete.json", manifest);
    const f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(scratch, &.{ dir, "complete.json" }), .{});
    defer f.close(io);
    try f.sync(io);
}
fn ready(a: std.mem.Allocator, io: std.Io, dir: []const u8, identity: []const u8) !bool {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const p = try std.fs.path.join(scratch, &.{ dir, "complete.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(io, p, scratch, .limited(1 << 20)) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return false,
    };
    const doc = std.json.parseFromSliceLeaky(std.json.Value, scratch, text, .{}) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return false,
    };
    if (doc != .object) return false;
    const id = doc.object.get("identity") orelse return false;
    const hashes = doc.object.get("files") orelse return false;
    if (id != .string or !std.mem.eql(u8, id.string, identity) or hashes != .object) return false;
    const list = files(scratch, io, dir) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return false,
    };
    if (list.len != hashes.object.count()) return false;
    for (list) |n| {
        const expected = hashes.object.get(n) orelse return false;
        if (expected != .string) return false;
        const hash = digest(io, try std.fs.path.join(scratch, &.{ dir, n })) catch return false;
        if (!std.mem.eql(u8, expected.string, &std.fmt.bytesToHex(hash, .lower))) return false;
    }
    return true;
}
test {
    _ = @import("gguf_import_test.zig");
}
