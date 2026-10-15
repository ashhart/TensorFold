//! A rank's checkpoint, digested for the ranks' agreement: config.json, the index, each shard's layout and bytes.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// Blocks of a shard's tensor bytes a whole checkpoint's digest reads, spread evenly from its first byte to its last.
const samples = 64;
const block = 4096;

/// The checkpoint in `dir`: a whole one's sampled bytes, or a split's non-expert tensors (its halves' experts differ).
pub fn digest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, split: bool) ![32]u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var h = Sha256.init(.{});
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 26));
    hashValue(&h, try std.json.parseFromSliceLeaky(std.json.Value, a, config, .{}), false);
    var names: std.ArrayList([]const u8) = .empty;
    const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, index, a, .limited(1 << 26))) |text| {
        h.update(text);
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const map = if (root == .object) root.object.get("weight_map") else null;
        if (map == null or map.? != .object) return error.BadIndex;
        for (map.?.object.values()) |v| if (v == .string) try names.append(a, v.string);
    } else |err| switch (err) {
        error.FileNotFound => try names.append(a, "model.safetensors"),
        else => return err,
    }
    std.mem.sort([]const u8, names.items, {}, lessThan);
    var last: []const u8 = "";
    for (names.items) |name| {
        if (std.mem.eql(u8, name, last)) continue;
        last = name;
        h.update(name);
        try shard(a, io, try std.fs.path.join(a, &.{ dir, name }), split, &h);
    }
    return h.finalResult();
}

/// A JSON value in a fixed form; our split's record (tensorfold_tp) is digested without its rank, which differs.
fn hashValue(h: *Sha256, v: std.json.Value, in_split: bool) void {
    switch (v) {
        .null => h.update("n"),
        .bool => |b| h.update(if (b) "t" else "f"),
        .integer => |i| {
            h.update("i");
            h.update(std.mem.asBytes(&i));
        },
        .float => |f| {
            h.update("d");
            h.update(std.mem.asBytes(&f));
        },
        .number_string, .string => |s| {
            h.update("s");
            h.update(std.mem.asBytes(&s.len));
            h.update(s);
        },
        .array => |x| {
            h.update("[");
            for (x.items) |item| hashValue(h, item, false);
            h.update("]");
        },
        .object => |o| {
            h.update("{");
            for (o.keys(), o.values()) |k, item| {
                if (in_split and std.mem.eql(u8, k, "rank")) continue;
                hashValue(h, .{ .string = k }, false);
                hashValue(h, item, std.mem.eql(u8, k, "tensorfold_tp"));
            }
            h.update("}");
        },
    }
}

/// An expert tensor (routed or shared): a split checkpoint's halves hold different shares of it.
fn expert(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".experts.") != null or std.mem.indexOf(u8, name, ".shared_experts.") != null;
}

/// A shard's size, header and bytes: `samples` blocks (all when fewer), or a split's non-expert tensors' first block.
fn shard(a: std.mem.Allocator, io: std.Io, path: []const u8, split: bool, h: *Sha256) !void {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size = try file.length(io);
    var head: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &head, 0) != 8) return error.BadHeader;
    const n = std.mem.readInt(u64, &head, .little);
    if (n > 1 << 30 or 8 + n > size) return error.BadHeader;
    h.update(std.mem.asBytes(&size));
    const header = try a.alloc(u8, @intCast(n));
    if (try file.readPositionalAll(io, header, 8) != header.len) return error.BadHeader;
    h.update(header);
    const start = 8 + n;
    var buf: [block]u8 = undefined;
    if (split) {
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, header, .{});
        if (root != .object) return error.BadHeader;
        var keys = try a.dupe([]const u8, root.object.keys());
        std.mem.sort([]const u8, keys, {}, lessThan);
        for (keys) |k| {
            if (expert(k) or std.mem.eql(u8, k, "__metadata__")) continue;
            const t = root.object.get(k).?;
            const offs = if (t == .object) t.object.get("data_offsets") else null;
            if (offs == null or offs.? != .array or offs.?.array.items.len != 2) return error.BadHeader;
            const b0: u64 = @intCast(offs.?.array.items[0].integer);
            const b1: u64 = @intCast(offs.?.array.items[1].integer);
            if (b1 < b0 or start + b1 > size) return error.BadHeader;
            const len: usize = @intCast(@min(block, b1 - b0));
            if (try file.readPositionalAll(io, buf[0..len], start + b0) != len) return error.FileShrank;
            h.update(buf[0..len]);
        }
        keys = &.{};
        return;
    }
    const span = size - start;
    if (span <= samples * block) {
        var at = start;
        while (at < size) {
            const got = try file.readPositionalAll(io, buf[0..@intCast(@min(block, size - at))], at);
            if (got == 0) return error.FileShrank;
            h.update(buf[0..got]);
            at += got;
        }
        return;
    }
    for (0..samples) |i| {
        const at = start + @as(u64, i) * (span - block) / (samples - 1);
        if (try file.readPositionalAll(io, &buf, at) != block) return error.FileShrank;
        h.update(&buf);
    }
}

fn lessThan(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

/// A one-shard checkpoint in `sub`: config.json, then a shard with a dense tensor `d` and an expert tensor.
fn write(io: std.Io, dir: std.Io.Dir, sub: []const u8, config: []const u8, dense: u8, expert_byte: u8) !void {
    const header = "{\"m.layers.0.d.weight\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}," ++
        "\"m.layers.1.mixer.experts.0.up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[4,8]}}";
    var bytes: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memcpy(bytes[8 + header.len ..], &[8]u8{ dense, 1, 2, 3, expert_byte, 5, 6, 7 });
    try dir.createDirPath(io, sub);
    var d = try dir.openDir(io, sub, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
}

test "the digest: one split's rank and its experts left out, any other byte or setting changes it" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = "{\"hidden_size\": 8, \"tensorfold_tp\": {\"rank\": 0, \"world\": 2}}";
    try write(io, tmp.dir, "a", cfg, 9, 4);
    try write(io, tmp.dir, "b", "{\"hidden_size\": 8, \"tensorfold_tp\": {\"rank\": 1, \"world\": 2}}", 9, 77);
    try write(io, tmp.dir, "c", cfg, 10, 4);
    try write(io, tmp.dir, "d", "{\"hidden_size\": 16, \"tensorfold_tp\": {\"rank\": 0, \"world\": 2}}", 9, 4);
    const gpa = std.testing.allocator;
    var path: [4][]u8 = undefined;
    for ([_][]const u8{ "a", "b", "c", "d" }, &path) |s, *p| p.* = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, s });
    defer for (path) |p| gpa.free(p);
    try std.testing.expectEqual(try digest(gpa, io, path[0], true), try digest(gpa, io, path[1], true));
    try std.testing.expect(!std.mem.eql(u8, &try digest(gpa, io, path[0], false), &try digest(gpa, io, path[1], false)));
    try std.testing.expect(!std.mem.eql(u8, &try digest(gpa, io, path[0], true), &try digest(gpa, io, path[2], true)));
    try std.testing.expect(!std.mem.eql(u8, &try digest(gpa, io, path[0], true), &try digest(gpa, io, path[3], true)));
}
