//! The Hugging Face cache on disk: where checkpoints live, their sizes, and which Zig family serves them.
const std = @import("std");
const Allocator = std.mem.Allocator;
const engines = @import("native_engines");

pub const GiB: u64 = 1 << 30;

/// One family the engine registry serves, with a display title.
pub const Family = struct { model_type: []const u8, title: []const u8, formats: []const []const u8 };

const titles = [_]struct { model_type: []const u8, title: []const u8 }{
    .{ .model_type = "nemotron_h", .title = "Nemotron 3.5 Lightning" },
    .{ .model_type = "qwen4_exp", .title = "Qwen3.8 Flash Next" },
    .{ .model_type = "glm5_next", .title = "GLM-5.3-Flash" },
    .{ .model_type = "qwen3_5", .title = "Qwen3.5-2B" },
};

/// The family that serves `model_type` from the engine registry, or null when Zig cannot serve it (the 0.6 line may).
pub fn family(model_type: []const u8) ?Family {
    for (engines.families) |f| {
        if (!std.mem.eql(u8, f.model_type, model_type)) continue;
        var title: []const u8 = f.model_type;
        for (titles) |t| if (std.mem.eql(u8, t.model_type, model_type)) {
            title = t.title;
            break;
        };
        return .{ .model_type = f.model_type, .title = title, .formats = f.formats };
    }
    return null;
}

/// HF_HUB_CACHE, or HF_HOME/hub, or the default cache under the home directory.
pub fn cacheDir(a: Allocator, env: ?*const std.process.Environ.Map, override: ?[]const u8) ![]const u8 {
    if (override) |o| return o;
    const get = struct {
        fn f(e: ?*const std.process.Environ.Map, name: []const u8) ?[]const u8 {
            return if (e) |m| m.get(name) else null;
        }
    }.f;
    if (get(env, "HF_HUB_CACHE")) |c| return c;
    const home = get(env, "HOME") orelse "";
    if (get(env, "HF_HOME")) |h| return std.fs.path.join(a, &.{ h, "hub" });
    return std.fs.path.join(a, &.{ home, ".cache", "huggingface", "hub" });

}

/// The repo's cache folder name: models--org--name.
pub fn repoDirName(a: Allocator, repo: []const u8) ![]const u8 {
    const slash = std.mem.indexOfScalar(u8, repo, '/') orelse return error.BadRepoId;
    return std.fmt.allocPrint(a, "models--{s}--{s}", .{ repo[0..slash], repo[slash + 1 ..] });
}

/// The cached snapshot serving `repo`: refs/main's revision, else the newest snapshot with a config.json.
pub fn cachedSnapshot(a: Allocator, io: std.Io, hub: []const u8, repo: []const u8) !?[]const u8 {
    const dir_name = try repoDirName(a, repo);
    const root = try std.fs.path.join(a, &.{ hub, dir_name });
    var ref_buf: [256]u8 = undefined;
    if (readSmall(a, io, try std.fs.path.join(a, &.{ root, "refs", "main" }))) |ref| {
        const rev = std.mem.trim(u8, ref, " \r\n");
        if (rev.len > 0 and rev.len <= ref_buf.len) {
            const snapshot = try std.fs.path.join(a, &.{ root, "snapshots", rev });
            if (isDir(io, snapshot)) return snapshot;
        }
    }
    const snapshots = try std.fs.path.join(a, &.{ root, "snapshots" });
    var dir = std.Io.Dir.cwd().openDir(io, snapshots, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var newest: ?[]const u8 = null;
    var newest_mtime: ?std.Io.Timestamp = null;
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .directory) continue;
        const snapshot = try std.fs.path.join(a, &.{ snapshots, e.name });
        if (readSmall(a, io, try std.fs.path.join(a, &.{ snapshot, "config.json" })) == null) continue;
        const st = std.Io.Dir.cwd().statFile(io, snapshot, .{}) catch continue;
        if (newest_mtime == null or st.mtime.nanoseconds > newest_mtime.?.nanoseconds) {
            newest_mtime = st.mtime;
            newest = snapshot;
        }
    }
    _ = &ref_buf;
    return newest;
}

/// ``owner/name`` shape, without consulting the filesystem.
pub fn isRepoIdLike(text: []const u8) bool {
    const slash = std.mem.indexOfScalar(u8, text, '/') orelse return false;
    if (std.mem.lastIndexOfScalar(u8, text, '/') != slash) return false;
    if (slash == 0 or slash == text.len - 1) return false;
    for ([_][]const u8{ text[0..slash], text[slash + 1 ..] }) |part| {
        if (part.len > 96) return false;
        for (part) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '-' or ch == '_' or ch == '.')) return false;
    }
    return true;
}

/// One integer field from the checkpoint's config.json.
pub fn configInt(a: Allocator, io: std.Io, dir: []const u8, key: []const u8) ?u64 {
    const text = readSmall(a, io, pathJoin(a, dir, "config.json") catch return null) orelse return null;
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get(key) orelse return null;
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else null;
}

/// config.json's model_type, "unknown" when unreadable.
pub fn modelType(a: Allocator, io: std.Io, dir: []const u8) []const u8 {
    const text = readSmall(a, io, pathJoin(a, dir, "config.json") catch return "unknown") orelse return "unknown";
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return "unknown";
    defer parsed.deinit();
    if (parsed.value != .object) return "unknown";
    const t = parsed.value.object.get("model_type") orelse return "unknown";
    return if (t == .string) t.string else "unknown";
}

/// config.json's quantization: bits and group size, when the checkpoint is MLX-affine quantized.
pub fn quantization(a: Allocator, io: std.Io, dir: []const u8) ?struct { bits: u64, group: u64 } {
    const text = readSmall(a, io, pathJoin(a, dir, "config.json") catch return null) orelse return null;
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const q = parsed.value.object.get("quantization") orelse return null;
    if (q != .object) return null;
    const bits = if (q.object.get("bits")) |b| (if (b == .integer) b.integer else return null) else return null;
    const group = if (q.object.get("group_size")) |g| (if (g == .integer) g.integer else 32) else 32;
    return .{ .bits = @intCast(bits), .group = @intCast(group) };
}

/// Bytes of the files under `dir`, symlinks followed.
pub fn sizeOf(a: Allocator, io: std.Io, dir: []const u8) !u64 {
    var total: u64 = 0;
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var walker = d.walk(a) catch return 0;
    defer walker.deinit();
    while (walker.next(io) catch null) |entry| {
        // Cache snapshots hold symlinks into blobs; statFile follows them to the real sizes.
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        const st = entry.dir.statFile(io, entry.basename, .{}) catch continue;
        total += st.size;
    }
    return total;
}

pub fn isDir(io: std.Io, path: []const u8) bool {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch return false;
    dir.close(io);
    return true;
}

fn pathJoin(a: Allocator, base: []const u8, name: []const u8) ![]const u8 {
    return std.fs.path.join(a, &.{ base, name });
}

fn readSmall(a: Allocator, io: std.Io, path: []const u8) ?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch null;
}

test "family detection matches the registry and refuses unknown types" {
    try std.testing.expectEqualStrings("Nemotron 3.5 Lightning", family("nemotron_h").?.title);
    try std.testing.expectEqualStrings("Qwen3.8 Flash Next", family("qwen4_exp").?.title);
    try std.testing.expect(family("gemma4") == null);
    try std.testing.expect(family("unknown") == null);
}

test "cache layout helpers read a synthetic cache" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const hub = try std.fmt.allocPrint(a, ".tf-hub-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, hub) catch {};
    const w = std.Io.Dir.cwd();
    try w.createDirPath(io, try std.fs.path.join(a, &.{ hub, "models--Org--Name/snapshots/abc123" }));
    try w.createDirPath(io, try std.fs.path.join(a, &.{ hub, "models--Org--Name/refs" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ hub, "models--Org--Name/snapshots/abc123/config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}, \"max_position_embeddings\": 65536}" });
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ hub, "models--Org--Name/refs/main" }), .data = "abc123\n" });
    var weights: [4096]u8 = undefined;
    @memset(&weights, 'x');
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ hub, "models--Org--Name/snapshots/abc123/weights.safetensors" }), .data = &weights });
    const snapshot = (try cachedSnapshot(a, io, hub, "Org/Name")).?;
    try std.testing.expect(std.mem.endsWith(u8, snapshot, "abc123"));
    try std.testing.expectEqualStrings("qwen4_exp", modelType(a, io, snapshot));
    const q = quantization(a, io, snapshot).?;
    try std.testing.expectEqual(@as(u64, 6), q.bits);
    try std.testing.expectEqual(@as(u64, 32), q.group);
    try std.testing.expect((try sizeOf(a, io, snapshot)) >= 4096);
    const missing = try cachedSnapshot(a, io, hub, "Org/Other");
    try std.testing.expect(missing == null);
}
