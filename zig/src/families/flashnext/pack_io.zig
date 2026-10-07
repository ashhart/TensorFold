//! Pack bytes on disk carry a source identity, and a reader refuses a pack from another checkpoint.
const std = @import("std");
const Io = std.Io;
const st = @import("../../core/safetensors.zig");

/// One file's tensors stay resident until written. put owns the bytes, and each blob keeps its allocation alignment.
pub const Out = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    names: std.ArrayList([]const u8) = .empty,
    dtypes: std.ArrayList([]const u8) = .empty,
    shapes: std.ArrayList([]const usize) = .empty,
    blobs: std.ArrayList([]const u8) = .empty,
    blob_aligns: std.ArrayList(std.mem.Alignment) = .empty,

    pub fn init(gpa: std.mem.Allocator) Out {
        return .{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(out: *Out) void {
        for (out.blobs.items, out.blob_aligns.items) |b, al| out.gpa.rawFree(@constCast(b), al, @returnAddress());
        out.blob_aligns.deinit(out.gpa);
        out.blobs.deinit(out.gpa);
        out.names.deinit(out.gpa);
        out.dtypes.deinit(out.gpa);
        out.shapes.deinit(out.gpa);
        out.arena.deinit();
        out.* = undefined;
    }

    pub fn put(out: *Out, name: []const u8, dtype: []const u8, shape: []const usize, bytes: anytype) !void {
        const a = out.arena.allocator();
        try out.names.append(out.gpa, try a.dupe(u8, name));
        try out.dtypes.append(out.gpa, dtype);
        const shape_copy = try a.alloc(usize, shape.len);
        @memcpy(shape_copy, shape);
        try out.shapes.append(out.gpa, shape_copy);
        try out.blobs.append(out.gpa, std.mem.sliceAsBytes(bytes));
        try out.blob_aligns.append(out.gpa, comptime std.mem.Alignment.fromByteUnits(@alignOf(std.meta.Child(@TypeOf(bytes)))));
    }

    /// Writes the file with `source` recorded under __metadata__ so a reader can refuse a pack from another checkpoint.
    pub fn write(out: *const Out, gpa: std.mem.Allocator, io: Io, path: []const u8, source: []const u8) !void {
        var header: std.Io.Writer.Allocating = .init(gpa);
        defer header.deinit();
        try header.writer.writeAll("{\"__metadata__\":{\"tf_source\":");
        try writeJsonString(&header.writer, source);
        try header.writer.writeAll("}");
        var at: usize = 0;
        for (out.names.items, out.dtypes.items, out.shapes.items, out.blobs.items) |name, dtype, shape, bytes| {
            try header.writer.print(",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, dtype });
            for (shape, 0..) |d, j| try header.writer.print("{s}{d}", .{ if (j == 0) "" else ",", d });
            try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + bytes.len });
            at += bytes.len;
        }
        try header.writer.writeAll("}");
        const head = header.written();
        var file = try Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var wbuf: [64 << 10]u8 = undefined;
        var fw = file.writerStreaming(io, &wbuf);
        const w = &fw.interface;
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, head.len, .little);
        try w.writeAll(&len);
        try w.writeAll(head);
        for (out.blobs.items) |bytes| try w.writeAll(bytes);
        try w.flush();
    }
};

fn writeJsonString(w: *std.Io.Writer, text: []const u8) !void {
    try w.writeByte('"');
    for (text) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        else => if (c < 0x20) try w.print("\\u{x:0>4}", .{c}) else try w.writeByte(c),
    };
    try w.writeByte('"');
}

/// One checkpoint shard's contribution to the pack's source identity.
pub const ShardId = struct {
    name: []const u8,
    size: u64,
    header_sha256: [32]u8,
};

/// A mapped shard's bytes with its true file size (a mapped region can round up to the page).
pub const MappedShard = struct {
    name: []const u8,
    size: u64,
    bytes: []const u8,
};

pub fn hashBytes(bytes: []const u8) [32]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return h;
}

/// The identity's canonical text: the index file's hash, then each shard's name, size and header hash by name.
pub fn identityString(gpa: std.mem.Allocator, index_sha256: ?[32]u8, shards: []const ShardId) ![]u8 {
    const sorted = try gpa.dupe(ShardId, shards);
    defer gpa.free(sorted);
    std.mem.sort(ShardId, sorted, {}, struct {
        fn lt(_: void, a: ShardId, b: ShardId) bool {
            return std.mem.order(u8, a.name, b.name) == .lt;
        }
    }.lt);
    var w: std.Io.Writer.Allocating = .init(gpa);
    errdefer w.deinit();
    if (index_sha256) |h| try w.writer.print("index {s}\n", .{&std.fmt.bytesToHex(h, .lower)}) else try w.writer.writeAll("index absent\n");
    for (sorted) |s| try w.writer.print("shard {s} {d} {s}\n", .{ s.name, s.size, &std.fmt.bytesToHex(s.header_sha256, .lower) });
    return w.toOwnedSlice();
}

fn mapFile(io: Io, path: []const u8) !struct { file: Io.File, map: Io.File.MemoryMap, len: usize } {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);
    const len: usize = @intCast(try file.length(io));
    if (len < 8) return error.BadSafetensors;
    const map = try Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
    errdefer map.destroy(io);
    return .{ .file = file, .map = map, .len = len };
}

fn headerLen(memory: []const u8) !usize {
    const hl: usize = @intCast(std.mem.readInt(u64, memory[0..8], .little));
    if (hl > memory.len - 8) return error.BadSafetensors;
    return hl;
}

/// Canonical identity text comes from the index bytes and each shard's mapped bytes, shared by builder and engine.
pub fn identityFromMapped(gpa: std.mem.Allocator, index_bytes: ?[]const u8, shards: []const MappedShard) ![]u8 {
    var ids: std.ArrayList(ShardId) = .empty;
    defer ids.deinit(gpa);
    for (shards) |s| {
        if (s.size < 8) return error.BadSafetensors;
        const size: usize = @intCast(s.size);
        if (s.bytes.len < size) return error.BadSafetensors;
        const hl = try headerLen(s.bytes[0..size]);
        try ids.append(gpa, .{ .name = s.name, .size = s.size, .header_sha256 = hashBytes(s.bytes[8..][0..hl]) });
    }
    return identityString(gpa, if (index_bytes) |b| hashBytes(b) else null, ids.items);
}

/// A checkpoint identity is the index hash plus each shard size and header hash, and mappings then unmap.
pub fn sourceIdentity(gpa: std.mem.Allocator, io: Io, model_dir: []const u8) ![]u8 {
    const Open = struct { file: Io.File, map: Io.File.MemoryMap };
    var shards: std.ArrayList(MappedShard) = .empty;
    defer shards.deinit(gpa);
    var maps: std.ArrayList(Open) = .empty;
    defer maps.deinit(gpa);
    errdefer for (maps.items) |*m| {
        m.map.destroy(io);
        m.file.close(io);
    };
    const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    var mapped = mapFile(io, index_path) catch |e| switch (e) {
        error.FileNotFound => return identityFromMapped(gpa, null, shards.items),
        else => return e,
    };
    defer mapped.map.destroy(io);
    defer mapped.file.close(io);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, mapped.map.memory, .{});
    defer parsed.deinit();
    const weight_map = parsed.value.object.get("weight_map") orelse return identityFromMapped(gpa, null, shards.items);
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    var it = weight_map.object.iterator();
    while (it.next()) |kv| {
        var dup = false;
        for (names.items) |n| if (std.mem.eql(u8, n, kv.value_ptr.string)) {
            dup = true;
            break;
        };
        if (!dup) try names.append(gpa, try gpa.dupe(u8, kv.value_ptr.string));
    }
    for (names.items) |name| {
        const p = try std.fs.path.join(gpa, &.{ model_dir, name });
        defer gpa.free(p);
        const shard = try mapFile(io, p);
        try maps.append(gpa, .{ .file = shard.file, .map = shard.map });
        try shards.append(gpa, .{ .name = name, .size = shard.len, .bytes = shard.map.memory });
    }
    const out = try identityFromMapped(gpa, mapped.map.memory, shards.items);
    for (maps.items) |*m| {
        m.map.destroy(io);
        m.file.close(io);
    }
    maps.clearRetainingCapacity();
    return out;
}

fn recordedSource(gpa: std.mem.Allocator, header_json: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, header_json, .{});
    defer parsed.deinit();
    const meta = parsed.value.object.get("__metadata__") orelse return error.PackHasNoSource;
    const source = meta.object.get("tf_source") orelse return error.PackHasNoSource;
    if (source != .string) return error.PackHasNoSource;
    return gpa.dupe(u8, source.string);
}

/// A mismatched pack identity is refused, and a pack with no recorded source loads with one warning.
pub fn checkSource(gpa: std.mem.Allocator, io: Io, pack_path: []const u8, model_dir: []const u8) !void {
    var mapped = try mapFile(io, pack_path);
    defer mapped.map.destroy(io);
    defer mapped.file.close(io);
    const hl = try headerLen(mapped.map.memory);
    const recorded = recordedSource(gpa, mapped.map.memory[8..][0..hl]) catch |e| switch (e) {
        error.PackHasNoSource => {
            std.log.warn("pack {s} has no recorded source; skipping the identity check", .{pack_path});
            return;
        },
        else => return e,
    };
    defer gpa.free(recorded);
    const identity = try sourceIdentity(gpa, io, model_dir);
    defer gpa.free(identity);
    if (!std.mem.eql(u8, recorded, identity)) return error.PackSourceMismatch;
}

/// The same identity refusal for pack bytes already mapped, with `name` labelling an unmarked pack.
pub fn checkSourceMapped(gpa: std.mem.Allocator, pack_bytes: []const u8, identity: []const u8, name: []const u8) !void {
    if (pack_bytes.len < 8) return error.BadSafetensors;
    const hl = try headerLen(pack_bytes);
    const recorded = recordedSource(gpa, pack_bytes[8..][0..hl]) catch |e| switch (e) {
        error.PackHasNoSource => {
            std.log.warn("pack {s} has no recorded source; skipping the identity check", .{name});
            return;
        },
        else => return e,
    };
    defer gpa.free(recorded);
    if (!std.mem.eql(u8, recorded, identity)) return error.PackSourceMismatch;
}

/// Compare one built pack with a reference dump's file: dtype, shape and every byte; a line per differing tensor.
pub fn compareFile(gpa: std.mem.Allocator, io: Io, built_path: []const u8, reference_path: []const u8, writer: *std.Io.Writer) !bool {
    var built = try st.File.open(gpa, io, built_path);
    defer built.close(io);
    var reference = try st.File.open(gpa, io, reference_path);
    defer reference.close(io);
    var identical: usize = 0;
    var differ: usize = 0;
    var it = built.names.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const mine = e.value_ptr.*;
        const theirs = reference.names.get(name) orelse {
            try writer.print("DIFF {s}: not in the reference\n", .{name});
            differ += 1;
            continue;
        };
        if (mine.dtype != theirs.dtype or mine.rank != theirs.rank or !std.mem.eql(usize, mine.shape[0..mine.rank], theirs.shape[0..theirs.rank])) {
            try writer.print("DIFF {s}: header differs\n", .{name});
            differ += 1;
            continue;
        }
        const x = built.map.memory[built.data + mine.begin .. built.data + mine.end];
        const y = reference.map.memory[reference.data + theirs.begin .. reference.data + theirs.end];
        if (!std.mem.eql(u8, x, y)) {
            var at: usize = 0;
            while (at < x.len and x[at] == y[at]) at += 1;
            try writer.print("DIFF {s}: first differing byte {d} of {d}\n", .{ name, at, x.len });
            differ += 1;
            continue;
        }
        identical += 1;
    }
    var it2 = reference.names.iterator();
    while (it2.next()) |e| {
        if (built.names.contains(e.key_ptr.*)) continue;
        try writer.print("DIFF {s}: missing from the build\n", .{e.key_ptr.*});
        differ += 1;
    }
    try writer.print("{s}: {d} identical, {d} differ\n", .{ std.fs.path.basename(built_path), identical, differ });
    return differ == 0;
}

test "deinit frees a typed blob at its allocation's alignment" {
    // Each blob is freed with the alignment it was allocated at.
    const t = std.testing.allocator;
    var out = Out.init(t);
    const ids = try t.alloc(u32, 8);
    @memset(ids, 7);
    try out.put("ids", "U32", &.{8}, ids);
    const bytes = try t.alloc(u8, 5);
    @memset(bytes, 0);
    try out.put("plain", "U8", &.{5}, bytes);
    out.deinit();
}

test {
    _ = @import("pack_source_test.zig");
}

/// A no-dump pack cache is complete when both packs are non-empty and their headers match the checkpoint.
pub fn packsReady(gpa: std.mem.Allocator, io: Io, cache_dir: []const u8, identity: []const u8) !bool {
    const pack_path = try std.fmt.allocPrintSentinel(gpa, "{s}/pack.safetensors", .{cache_dir}, 0);
    defer gpa.free(pack_path);
    {
        const mapped = Io.Dir.cwd().openFile(io, pack_path, .{}) catch return false;
        defer mapped.close(io);
        const stat = try mapped.stat(io);
        if (stat.size < 8) return false;
        const head = try gpa.alloc(u8, @intCast(stat.size));
        defer gpa.free(head);
        var buf: [4096]u8 = undefined;
        var reader = mapped.reader(io, &buf);
        try reader.interface.readSliceAll(head);
        checkSourceMapped(gpa, head, identity, pack_path) catch return false;
    }
    const mlx_path = try std.fmt.allocPrintSentinel(gpa, "{s}/pack_mlx.safetensors", .{cache_dir}, 0);
    defer gpa.free(mlx_path);
    const mlx = Io.Dir.cwd().openFile(io, mlx_path, .{}) catch return false;
    defer mlx.close(io);
    const mlx_stat = try mlx.stat(io);
    if (mlx_stat.size < 8) return false;
    const mlx_head = try gpa.alloc(u8, @intCast(mlx_stat.size));
    defer gpa.free(mlx_head);
    var mlx_buf: [4096]u8 = undefined;
    var mlx_reader = mlx.reader(io, &mlx_buf);
    try mlx_reader.interface.readSliceAll(mlx_head);
    checkSourceMapped(gpa, mlx_head, identity, mlx_path) catch return false;
    return true;
}
