//! What a kept state on disk is read under: a checkpoint's files and stamps, the OS build and the chip.
const std = @import("std");
const api = @import("engine_api");
const digests = @import("tensorfold").digest_cache;
const Allocator = std.mem.Allocator;
/// PWV3's active header, affine coefficients and LUTs, already read by both backends' finite-value checks.
pub const tfq3_table_bytes = 10816;

/// Prompt-file bytes, config/index bytes and stamps of every indexed or top-level safetensors shard.
const prompt_files = [_][]const u8{ "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "chat_template.jinja", "chat_template.json" };

/// The tokenizer's and template's bytes: missing optional files leave no entry.
pub fn promptFiles(h: anytype, a: Allocator, io: std.Io, dir: []const u8) !void {
    for (prompt_files) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        defer a.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => return e,
        };
        defer a.free(bytes);
        h.update(name);
        h.update(bytes);
    }
}

/// Config/index bytes and every shard stamp, behind the checkpoint's prompt-file identity.
pub fn checkpoint(h: anytype, a: Allocator, io: std.Io, dir: []const u8) !void {
    const root = try digests.environmentRoot(a);
    defer if (root) |r| a.free(r);
    return checkpointMode(h, a, io, dir, root, false);
}
fn checkpointMode(h: anytype, a: Allocator, io: std.Io, dir: []const u8, root: ?[]const u8, comptime verified: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const digest_cache: digests.Cache = .{ .root = root };
    try promptFiles(h, a, io, dir);
    var indexed: std.StringArrayHashMapUnmanaged(void) = .empty;
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const path = try std.fs.path.join(scratch, &.{ dir, name });
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, scratch, .limited(1 << 26)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => return e,
        };
        h.update(name);
        h.update(bytes);
        if (std.mem.eql(u8, name, "model.safetensors.index.json")) {
            const parsed = try std.json.parseFromSliceLeaky(std.json.Value, scratch, bytes, .{});
            const map = try @import("checkpoint_config").index.weightMap(parsed);
            for (map.values()) |value| try indexed.put(scratch, value.string, {});
        }
    }
    var names: std.ArrayList([]const u8) = .empty;
    try names.appendSlice(scratch, indexed.keys());
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    var tfq3_records = false;
    while (try it.next(io)) |e| {
        if (std.mem.endsWith(u8, e.name, ".safetensors") or std.mem.endsWith(u8, e.name, ".tffpw"))
            try names.append(scratch, try scratch.dupe(u8, e.name));
        if (std.mem.endsWith(u8, e.name, ".tffpw")) tfq3_records = true;
    }
    if (tfq3_records) {
        const table = try d.openFile(io, "pw_v3_tables.bin", .{});
        defer table.close(io);
        var bytes: [tfq3_table_bytes]u8 = undefined;
        if (try table.readPositionalAll(io, &bytes, 0) != bytes.len) return error.IdentityTfq3TableTruncated;
        h.update("tfq3-active-tables-v1");
        h.update(&bytes);
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (names.items, 0..) |name, i| {
        if (i > 0 and std.mem.eql(u8, name, names.items[i - 1])) continue;
        const info = d.statFile(io, name, .{}) catch |e|
            return if (e == error.FileNotFound and indexed.contains(name)) error.IndexedShardMissing else e;
        const stamp = [4]i128{ info.size, info.inode, info.mtime.nanoseconds, info.ctime.nanoseconds };
        h.update(name);
        if (!verified) h.update(std.mem.sliceAsBytes(&stamp));
        // Known limit: unchanged descriptor stamps can hide a TFQ3 coefficient edit; loading must not scan payloads.
        const path = try std.fs.path.join(scratch, &.{ dir, name });
        const known_value: ?digests.Result = if (verified) blk: {
            const file = try d.openFile(io, name, .{});
            defer file.close(io);
            break :blk (try digest_cache.checkedFile(scratch, io, file)) orelse return error.UnverifiedPersistence;
        } else try digest_cache.lookup(scratch, io, path);
        if (known_value) |known| {
            if (known.key.size != info.size or known.key.inode != info.inode or
                known.key.mtime != info.mtime.nanoseconds or known.key.ctime != info.ctime.nanoseconds)
                return error.DigestFileChanged;
            field(h, "verified-shard-sha256-v1");
            field(h, &known.value.sha256);
        }
    }
}

/// This Mac's OS build and chip: a macOS update or another chip can compile a kernel to other bits.
pub fn machine(h: anytype, chip: []const u8) void {
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(chip);
}

/// The family/backend's reviewed profile values; tests can supply a registry row and an isolated environment.
pub const Profile = struct {
    family: []const u8,
    backend: []const u8,
    entries: []const api.profile.Entry = &api.profile.entries,
    environment: *const fn ([:0]const u8) ?[:0]const u8 = processEnvironment,

    fn processEnvironment(name: [:0]const u8) ?[:0]const u8 {
        return if (std.c.getenv(name)) |value| std.mem.span(value) else null;
    }
};

/// Only applicable non-identical entries partition kept state; defaults and `on_with` resolve before values are hashed.
pub fn profile(h: anytype, p: Profile) void {
    var began = false;
    for (p.entries) |e| {
        if (e.class == .identical or !e.applies(p.family, p.backend)) continue;
        if (!began) {
            field(h, "profile-values-v1");
            field(h, p.family);
            field(h, p.backend);
            began = true;
        }
        field(h, e.name());
        field(h, @tagName(e.class));
        if (e.members) |members| {
            for (members) |member| {
                if (!member.applies(p.family)) continue;
                field(h, member.name);
                valueField(h, member.syntax, p.environment(member.name));
            }
            continue;
        }
        const value = e.served(p.environment(e.name()), if (e.on_with) |w| p.environment(@tagName(w)) else null) orelse e.stable;
        if (e.report_values != null) {
            field(h, if (value != null) "explicit" else "default");
            field(h, value orelse "");
            continue;
        }
        valueField(h, e.syntax, value);
    }
}

fn valueField(h: anytype, syntax: api.profile.Syntax, value: ?[]const u8) void {
    switch (syntax) {
        .one => field(h, if (value != null and std.mem.eql(u8, value.?, "1")) "1" else "0"),
        .auto => field(h, value orelse "auto"),
        .count => {
            const count = std.fmt.parseInt(u64, value orelse "0", 10) catch {
                field(h, "invalid-count");
                field(h, value.?);
                return;
            };
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, count, .little);
            field(h, &bytes);
        },
        .present => {
            field(h, if (value != null) "explicit" else "default");
            field(h, value orelse "");
        },
        .not_zero => field(h, if (value != null and !std.mem.eql(u8, value.?, "0")) "1" else "0"),
    }
}

fn field(h: anytype, bytes: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, @intCast(bytes.len), .little);
    h.update(&len);
    h.update(bytes);
}

/// A persisted serving namespace extends its probe identity with the checkpoint at the current open.
pub fn scoped(a: Allocator, io: std.Io, dir: []const u8, legacy: u64) !u64 {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var h = std.hash.Wyhash.init(0x70657273);
    h.update("kept-checkpoint-v1");
    h.update(std.mem.asBytes(&legacy));
    try checkpoint(&h, arena.allocator(), io, dir);
    try @import("sys").file_identity.current(&h, arena.allocator(), io);
    return h.final();
}

/// Stable serving namespaces require verified content; unknown copies can serve with optional state kept cold.
pub fn scopedVerified(a: Allocator, io: std.Io, dir: []const u8, legacy: u64) !u64 {
    const root = try digests.environmentRoot(a);
    defer if (root) |r| a.free(r);
    return scopedVerifiedWithCache(a, io, dir, legacy, root);
}
pub fn scopedVerifiedWithCache(a: Allocator, io: std.Io, dir: []const u8, legacy: u64, root: ?[]const u8) !u64 {
    var h = std.hash.Wyhash.init(0x70657273);
    h.update("kept-verified-checkpoint-v2");
    h.update(std.mem.asBytes(&legacy));
    try checkpointMode(&h, a, io, dir, root, true);
    try @import("sys").file_identity.current(&h, a, io);
    return h.final();
}
