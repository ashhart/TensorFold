//! The Hugging Face cache on disk: where checkpoints live, their sizes, and which Zig family serves them.
const std = @import("std");
const Allocator = std.mem.Allocator;
const engines = @import("native_engines");

pub const GiB: u64 = 1 << 30;

/// One family the engine registry serves, with a display title.
pub const Family = struct { model_type: []const u8, title: []const u8, formats: []const []const u8 };

const titles = [_]struct { model_type: []const u8, title: []const u8 }{
    .{ .model_type = "nemotron_h", .title = "Nemotron 3.5 Lightning" },
    .{ .model_type = "nemotron_h_omni", .title = "Nemotron 3.5 Super VL" },
    .{ .model_type = "qwen4_exp", .title = "Qwen3.8 Flash Next" },
    .{ .model_type = "glm5_next", .title = "GLM-5.3-Flash" },
    .{ .model_type = "qwen3_5", .title = "Qwen3.5-2B or Qwen3.8-27B" },
    .{ .model_type = "gemma4", .title = "Gemma 4 31B" },
    .{ .model_type = "gemma4_unified", .title = "Gemma 4 12B" },
    .{ .model_type = "step5_vl", .title = "Step-5-Preview" },
    .{ .model_type = "lfm2_moe", .title = "LFM2.5-24B-A2B" },
    .{ .model_type = "minimax_m3_vl", .title = "MiniMax-M3.1-Flash VL" },
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
    const home = get(env, "HOME") orelse get(env, "USERPROFILE") orelse ""; // Windows sets USERPROFILE, not HOME
    if (get(env, "HF_HOME")) |h| return std.fs.path.join(a, &.{ h, "hub" });
    return std.fs.path.join(a, &.{ home, ".cache", "huggingface", "hub" });
}

/// The repo's cache folder name: models--org--name.
pub fn repoDirName(a: Allocator, repo: []const u8) ![]const u8 {
    const slash = std.mem.indexOfScalar(u8, repo, '/') orelse return error.BadRepoId;
    return std.fmt.allocPrint(a, "models--{s}--{s}", .{ repo[0..slash], repo[slash + 1 ..] });
}

/// A git sha: 40 hex characters, the only strings ever joined into a cache path.
pub fn hexSha(text: []const u8) bool {
    if (text.len != 40) return false;
    for (text) |ch| {
        const ok = (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f') or (ch >= 'A' and ch <= 'F');
        if (!ok) return false;
    }
    return true;
}

/// The cached snapshot serving `repo`: refs/main's revision, else the newest snapshot with a config.json.
pub fn cachedSnapshot(a: Allocator, io: std.Io, hub: []const u8, repo: []const u8) !?[]const u8 {
    const dir_name = try repoDirName(a, repo);
    const root = try std.fs.path.join(a, &.{ hub, dir_name });
    if (readSmall(a, io, try std.fs.path.join(a, &.{ root, "refs", "main" }))) |ref| {
        const rev = std.mem.trim(u8, ref, " \r\n");
        // The ref names the snapshot dir: only a 40-hex git sha is ever joined into a cache path.
        if (hexSha(rev)) {
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
        if (@import("checkpoint_config").optional(a, io, snapshot) == null) continue;
        const st = std.Io.Dir.cwd().statFile(io, snapshot, .{}) catch continue;
        if (newest_mtime == null or st.mtime.nanoseconds > newest_mtime.?.nanoseconds) {
            newest_mtime = st.mtime;
            newest = snapshot;
        }
    }
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
    const text = @import("checkpoint_config").optional(a, io, dir) orelse return null;
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get(key) orelse return null;
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else null;
}

/// config.json's model_type, "unknown" when unreadable.
pub fn modelType(a: Allocator, io: std.Io, dir: []const u8) []const u8 {
    const text = @import("checkpoint_config").optional(a, io, dir) orelse return "unknown";
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return "unknown";
    defer parsed.deinit();
    if (parsed.value != .object) return "unknown";
    const t = parsed.value.object.get("model_type") orelse return "unknown";
    return if (t == .string) t.string else "unknown";
}

/// config.json's quantization: bits and group size, when the checkpoint is MLX-affine quantized.
pub fn quantization(a: Allocator, io: std.Io, dir: []const u8) ?struct { bits: u64, group: u64 } {
    const text = @import("checkpoint_config").optional(a, io, dir) orelse return null;
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const q = parsed.value.object.get("quantization") orelse return null;
    if (q != .object) return null;
    const bits = if (q.object.get("bits")) |b| (if (b == .integer) b.integer else return null) else return null;
    const group = if (q.object.get("group_size")) |g| (if (g == .integer) g.integer else return null) else 32;
    if (bits <= 0 or group <= 0) return null;
    return .{ .bits = @intCast(bits), .group = @intCast(group) };
}

/// An EXL3 checkpoint's format word, read from the quantization metadata itself (the
/// glm/cuda_exl3.zig block() reading): config.json's inline `quantization_config` at the top
/// level or inside `text_config`, else the standalone quantization_config.json — the second
/// file an EXL3 pack ships when config.json's inline copy is absent. Returns "exl3/3.06bpw"
/// (the metadata's avg_bits) or a plain "exl3" without it; null when the checkpoint carries no
/// EXL3 metadata at all (MLX affine and other quantizations are read elsewhere).
pub fn exl3Format(a: Allocator, io: std.Io, dir: []const u8) ?[]const u8 {
    if (@import("checkpoint_config").optional(a, io, dir)) |text| {
        if (std.json.parseFromSlice(std.json.Value, a, text, .{})) |inline_parsed| {
            defer inline_parsed.deinit();
            if (block(inline_parsed.value)) |q| return fmtExl3(a, q.object);
            return inlineFallback(a, io, dir); // no EXL3 inline: maybe the standalone file has it
        } else |_| return inlineFallback(a, io, dir); // a broken config.json: the standalone file still speaks
    }
    return inlineFallback(a, io, dir); // no config.json body: the standalone file is the metadata
}

/// The standalone quantization_config.json's EXL3 metadata, when config.json's inline copy did not name it.
fn inlineFallback(a: Allocator, io: std.Io, dir: []const u8) ?[]const u8 {
    const path = std.fs.path.join(a, &.{ dir, "quantization_config.json" }) catch return null;
    const text = readSmall(a, io, path) orelse return null;
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return null;
    defer parsed.deinit();
    const q = if (parsed.value == .object) parsed.value.object.get("quantization_config") orelse parsed.value else return null;
    if (q != .object) return null;
    return fmtExl3(a, q.object);
}

/// config.json's quantization_config: the top level's, else text_config's (a VL wrapper's own block).
fn block(root: std.json.Value) ?std.json.Value {
    if (root != .object) return null;
    if (root.object.get("quantization_config")) |q| if (q == .object) return q;
    if (root.object.get("text_config")) |t| if (t == .object) {
        if (t.object.get("quantization_config")) |q| if (q == .object) return q;
    };
    return null;
}

fn fmtExl3(a: Allocator, q: std.json.ObjectMap) ?[]const u8 {
    const m = q.get("quant_method") orelse return null;
    if (m != .string or !std.mem.eql(u8, m.string, "exl3")) return null;
    const avg = q.get("avg_bits") orelse return "exl3";
    return switch (avg) {
        .float => std.fmt.allocPrint(a, "exl3/{d}bpw", .{avg.float}) catch "exl3",
        .integer => std.fmt.allocPrint(a, "exl3/{d}bpw", .{@as(f64, @floatFromInt(avg.integer))}) catch "exl3",
        else => "exl3",
    };
}

/// Bytes of the files under `dir`, symlinks followed, hidden ones left out (a download's .cache, --slide's snapshot).
pub fn sizeOf(a: Allocator, io: std.Io, dir: []const u8) !u64 {
    var total: u64 = 0;
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var walker = d.walk(a) catch return 0;
    defer walker.deinit();
    while (walker.next(io) catch null) |entry| {
        if (entry.path[0] == '.') {
            if (entry.kind == .directory) walker.leave(io);
            continue;
        }
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

fn readSmall(a: Allocator, io: std.Io, path: []const u8) ?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch null;
}

test "security: negative quantization metadata is refused without a cast trap" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    for ([_][]const u8{
        "{\"quantization\":{\"bits\":-1,\"group_size\":32}}",
        "{\"quantization\":{\"bits\":4,\"group_size\":-1}}",
        "{\"quantization\":{\"bits\":0,\"group_size\":32}}",
        "{\"quantization\":{\"bits\":4,\"group_size\":0}}",
    }) |data| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = data });
        try std.testing.expect(quantization(a, io, dir) == null);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"quantization\":{\"bits\":4}}" });
    const valid = quantization(a, io, dir).?;
    try std.testing.expectEqual(@as(u64, 4), valid.bits);
    try std.testing.expectEqual(@as(u64, 32), valid.group);
}

test "family detection matches the registry and refuses unknown types" {
    try std.testing.expectEqualStrings("Nemotron 3.5 Lightning", family("nemotron_h").?.title);
    try std.testing.expectEqualStrings("Qwen3.8 Flash Next", family("qwen4_exp").?.title);
    try std.testing.expectEqualStrings("Gemma 4 12B", family("gemma4_unified").?.title);
    try std.testing.expect(family("gemma3n") == null);
    try std.testing.expect(family("unknown") == null);
}

// The EXL3 format word an `info` line reports: inline at the top level, inline inside
// text_config, or the standalone quantization_config.json when config.json carries none;
// MLX `quantization` and absent metadata read null. Verbatim from the staged pack's bytes.
test "exl3 format reads the quantization metadata wherever the pack carries it" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    for ([_][]const u8{ "a", "b", "c", "e", "f" }) |name| try cwd.createDirPath(io, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{name}));
    // inline top level, the pack's own avg_bits (the staged pack's shape)
    try cwd.writeFile(io, .{ .sub_path = ".zig-cache/tmp/a/config.json", .data = "{\"quantization_config\":{\"quant_method\":\"exl3\",\"codebook\":\"mul1\",\"avg_bits\":3.06}}" });
    try std.testing.expectEqualStrings("exl3/3.06bpw", exl3Format(a, io, ".zig-cache/tmp/a").?);
    // inline inside text_config
    try cwd.writeFile(io, .{ .sub_path = ".zig-cache/tmp/b/config.json", .data = "{\"text_config\":{\"quantization_config\":{\"quant_method\":\"exl3\",\"avg_bits\":3.5}}}" });
    try std.testing.expectEqualStrings("exl3/3.5bpw", exl3Format(a, io, ".zig-cache/tmp/b").?);
    // no inline copy: the standalone quantization_config.json speaks, exactly Mia-AiLab's
    try cwd.writeFile(io, .{ .sub_path = ".zig-cache/tmp/c/config.json", .data = "{\"model_type\":\"minimax_m3_vl\"}" });
    try cwd.writeFile(io, .{ .sub_path = ".zig-cache/tmp/c/quantization_config.json", .data = "{\"quant_method\":\"exl3\",\"version\":\"1.6.1\",\"head_bits\":8,\"out_scales\":\"always\",\"codebook\":\"mul1\",\"avg_bits\":3.06}" });
    try std.testing.expectEqualStrings("exl3/3.06bpw", exl3Format(a, io, ".zig-cache/tmp/c").?);
    // no EXL3 at all
    try cwd.writeFile(io, .{ .sub_path = ".zig-cache/tmp/e/config.json", .data = "{\"quantization\":{\"bits\":4,\"group_size\":32}}" });
    try std.testing.expect(exl3Format(a, io, ".zig-cache/tmp/e") == null);
    // other quant methods read null (an eaq or modelopt checkpoint is not this format)
    try cwd.writeFile(io, .{ .sub_path = ".zig-cache/tmp/f/config.json", .data = "{\"quantization_config\":{\"quant_method\":\"eaq\"}}" });
    try std.testing.expect(exl3Format(a, io, ".zig-cache/tmp/f") == null);
}

test "cache layout helpers read a synthetic cache" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const hub = try std.fmt.allocPrint(a, ".tf-hub-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, hub) catch {};
    const w = std.Io.Dir.cwd();
    try w.createDirPath(io, try std.fs.path.join(a, &.{ hub, "models--Org--Name/snapshots/0123456789abcdef0123456789abcdef01234567" }));
    try w.createDirPath(io, try std.fs.path.join(a, &.{ hub, "models--Org--Name/refs" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ hub, "models--Org--Name/snapshots/0123456789abcdef0123456789abcdef01234567/config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}, \"max_position_embeddings\": 65536}" });
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ hub, "models--Org--Name/refs/main" }), .data = "0123456789abcdef0123456789abcdef01234567\n" });
    var weights: [4096]u8 = undefined;
    @memset(&weights, 'x');
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ hub, "models--Org--Name/snapshots/0123456789abcdef0123456789abcdef01234567/weights.safetensors" }), .data = &weights });
    const snapshot = (try cachedSnapshot(a, io, hub, "Org/Name")).?;
    try std.testing.expect(std.mem.endsWith(u8, snapshot, "0123456789abcdef0123456789abcdef01234567"));
    try std.testing.expectEqualStrings("qwen4_exp", modelType(a, io, snapshot));
    const q = quantization(a, io, snapshot).?;
    try std.testing.expectEqual(@as(u64, 6), q.bits);
    try std.testing.expectEqual(@as(u64, 32), q.group);
    try std.testing.expect((try sizeOf(a, io, snapshot)) >= 4096);
    const plain = try sizeOf(a, io, snapshot);
    try w.createDirPath(io, try std.fs.path.join(a, &.{ snapshot, ".tensorfold-original" }));
    const kept = try std.fs.path.join(a, &.{ snapshot, ".tensorfold-original/weights.safetensors" });
    try w.writeFile(io, .{ .sub_path = kept, .data = &weights });
    try std.testing.expectEqual(plain, try sizeOf(a, io, snapshot)); // --slide's snapshot is not the model's size
    try w.createDirPath(io, try std.fs.path.join(a, &.{ hub, "models--Org--Bad/refs" }));
    try w.createDirPath(io, try std.fs.path.join(a, &.{ hub, "models--Org--Bad/snapshots" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ hub, "models--Org--Bad/refs/main" }), .data = "../../models--Org--Name/snapshots/0123456789abcdef0123456789abcdef01234567" });
    try std.testing.expect(try cachedSnapshot(a, io, hub, "Org/Bad") == null); // a ref that is not a sha is never joined
    const missing = try cachedSnapshot(a, io, hub, "Org/Other");
    try std.testing.expect(missing == null);
}
