//! What a kept state on disk is read under: a checkpoint's files and stamps, the OS build and the chip.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

/// Config/index bytes and stamps of every indexed shard plus top-level safetensors (including auxiliary heads).
pub fn checkpoint(h: *std.hash.Wyhash, a: Allocator, io: std.Io, dir: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    var names: std.ArrayList([]const u8) = .empty;
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
            for (map.values()) |value| try names.append(scratch, value.string);
        }
    }
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors"))
        try names.append(scratch, try scratch.dupe(u8, e.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (names.items, 0..) |name, i| {
        if (i > 0 and std.mem.eql(u8, name, names.items[i - 1])) continue;
        const info = try d.statFile(io, name, .{});
        const stamp = [4]i128{ info.size, info.inode, info.mtime.nanoseconds, info.ctime.nanoseconds };
        h.update(name);
        h.update(std.mem.sliceAsBytes(&stamp));
    }
}

/// This Mac's OS build and chip: a macOS update or another chip can compile a kernel to other bits.
pub fn machine(h: *std.hash.Wyhash, chip: []const u8) void {
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

/// Only applicable non-identical entries partition kept state; defaults resolve before their values are hashed.
pub fn profile(h: *std.hash.Wyhash, p: Profile) void {
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
        const value = p.environment(e.name()) orelse e.stable;
        if (e.report_values != null) {
            field(h, if (value != null) "explicit" else "default");
            field(h, value orelse "");
            continue;
        }
        switch (e.syntax) {
            .one => field(h, if (value != null and std.mem.eql(u8, value.?, "1")) "1" else "0"),
            .auto => field(h, value orelse "auto"),
            .count => {
                const count = std.fmt.parseInt(u64, value orelse "0", 10) catch {
                    field(h, "invalid-count");
                    field(h, value.?);
                    continue;
                };
                var bytes: [8]u8 = undefined;
                std.mem.writeInt(u64, &bytes, count, .little);
                field(h, &bytes);
            },
        }
    }
}

fn field(h: *std.hash.Wyhash, bytes: []const u8) void {
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
