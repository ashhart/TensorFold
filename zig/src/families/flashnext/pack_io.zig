//! Pack bytes on disk carry a source identity, and a reader refuses a pack from another checkpoint.
const std = @import("std");
const Io = std.Io;
const st = @import("../../core/safetensors.zig");
const digests = @import("../../core/digest_cache.zig");

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

    /// Takes `bytes` (from `gpa`) even when it fails: then they are freed and every list keeps its length.
    pub fn put(out: *Out, name: []const u8, dtype: []const u8, shape: []const usize, bytes: anytype) !void {
        errdefer out.gpa.free(bytes);
        const a = out.arena.allocator();
        const name_copy = try a.dupe(u8, name);
        const shape_copy = try a.dupe(usize, shape);
        try out.names.ensureUnusedCapacity(out.gpa, 1);
        try out.dtypes.ensureUnusedCapacity(out.gpa, 1);
        try out.shapes.ensureUnusedCapacity(out.gpa, 1);
        try out.blobs.ensureUnusedCapacity(out.gpa, 1);
        try out.blob_aligns.ensureUnusedCapacity(out.gpa, 1);
        out.names.appendAssumeCapacity(name_copy);
        out.dtypes.appendAssumeCapacity(dtype);
        out.shapes.appendAssumeCapacity(shape_copy);
        out.blobs.appendAssumeCapacity(std.mem.sliceAsBytes(bytes));
        out.blob_aligns.appendAssumeCapacity(comptime std.mem.Alignment.fromByteUnits(@alignOf(std.meta.Child(@TypeOf(bytes)))));
    }

    /// Writes the file with `source` recorded under __metadata__ so a reader can refuse a pack from another checkpoint.
    pub fn write(out: *const Out, gpa: std.mem.Allocator, io: Io, path: []const u8, source: []const u8) !void {
        const root = try digests.environmentRoot(gpa);
        defer if (root) |r| gpa.free(r);
        const generated = if (root) |r| try digests.generatedRoot(gpa, r) else null;
        defer if (generated) |r| gpa.free(r);
        return out.writeWithCache(gpa, io, path, source, generated);
    }

    pub fn writeWithCache(out: *const Out, gpa: std.mem.Allocator, io: Io, path: []const u8, source: []const u8, root: ?[]const u8) !void {
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
        var digest = std.crypto.hash.sha2.Sha256.init(.{});
        if (root != null) {
            digest.update(&len);
            digest.update(head);
        }
        for (out.blobs.items) |bytes| {
            try w.writeAll(bytes);
            if (root != null) digest.update(bytes);
        }
        try w.flush();
        if (root != null) {
            try file.sync(io);
            try (digests.Cache{ .root = root }).remember(gpa, io, path, try digests.Key.of(file, io), digest.finalResult());
        }
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
    stamp: ?digests.Key = null,
    payload_sha256: ?[32]u8 = null,
};

/// A mapped shard's bytes with its true file size (a mapped region can round up to the page).
pub const MappedShard = struct {
    name: []const u8,
    size: u64,
    bytes: []const u8,
    stamp: digests.Key,
    payload_sha256: ?[32]u8 = null,
};

pub fn hashBytes(bytes: []const u8) [32]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return h;
}

/// Version 5 separates stable metadata keys from deterministic build-time source-content attestations.
pub const pack_format = 5;

/// The identity's text: pack format, index and config hashes (the config names the widths), each shard's id.
pub fn identityString(gpa: std.mem.Allocator, index_sha256: ?[32]u8, config_sha256: ?[32]u8, shards: []const ShardId) ![]u8 {
    const sorted = try gpa.dupe(ShardId, shards);
    defer gpa.free(sorted);
    std.mem.sort(ShardId, sorted, {}, struct {
        fn lt(_: void, a: ShardId, b: ShardId) bool {
            return std.mem.order(u8, a.name, b.name) == .lt;
        }
    }.lt);
    var w: std.Io.Writer.Allocating = .init(gpa);
    errdefer w.deinit();
    try w.writer.print("flashnext pack {d}\n", .{pack_format});
    if (index_sha256) |h| try w.writer.print("index {s}\n", .{&std.fmt.bytesToHex(h, .lower)}) else try w.writer.writeAll("index absent\n");
    if (config_sha256) |h| try w.writer.print("config {s}\n", .{&std.fmt.bytesToHex(h, .lower)}) else try w.writer.writeAll("config absent\n");
    for (sorted) |s| {
        try w.writer.print("shard {s} {d} {s}\n", .{ s.name, s.size, &std.fmt.bytesToHex(s.header_sha256, .lower) });
        if (s.payload_sha256) |sha| try w.writer.print("payload {s}\n", .{&std.fmt.bytesToHex(sha, .lower)});
    }
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

/// The identity from the index and config bytes and each shard's mapped bytes, shared by builder and engine.
pub fn identityFromMapped(gpa: std.mem.Allocator, index_bytes: ?[]const u8, config_bytes: ?[]const u8, shards: []const MappedShard) ![]u8 {
    var ids: std.ArrayList(ShardId) = .empty;
    defer ids.deinit(gpa);
    for (shards) |s| {
        if (s.size < 8) return error.BadSafetensors;
        const size: usize = @intCast(s.size);
        if (s.bytes.len < size) return error.BadSafetensors;
        const hl = try headerLen(s.bytes[0..size]);
        if (s.stamp.size != s.size) return error.DigestFileChanged;
        try ids.append(gpa, .{ .name = s.name, .size = s.size, .header_sha256 = hashBytes(s.bytes[8..][0..hl]), .stamp = s.stamp, .payload_sha256 = s.payload_sha256 });
    }
    return identityString(gpa, if (index_bytes) |b| hashBytes(b) else null, if (config_bytes) |b| hashBytes(b) else null, ids.items);
}

/// A stable metadata key; local verified-descriptor failures refuse without reading source payloads.
pub fn sourceIdentity(gpa: std.mem.Allocator, io: Io, model_dir: []const u8) ![]u8 {
    const root = try digests.environmentRoot(gpa);
    defer if (root) |r| gpa.free(r);
    return sourceIdentityWithCache(gpa, io, model_dir, root);
}

/// Reuse verified digests when present; a miss stays metadata-only, so warm loads never scan source payloads.
pub fn sourceIdentityWithCache(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, root: ?[]const u8) ![]u8 {
    return sourceIdentityMode(gpa, io, model_dir, root, .metadata);
}
pub fn buildIdentity(gpa: std.mem.Allocator, io: Io, model_dir: []const u8) ![]u8 {
    const root = try digests.environmentRoot(gpa);
    defer if (root) |r| gpa.free(r);
    return sourceIdentityMode(gpa, io, model_dir, root, .build);
}
pub fn identityWithContent(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, root: ?[]const u8, build: bool) ![]u8 {
    return sourceIdentityMode(gpa, io, model_dir, root, if (build) .build else .verified);
}
const IdentityMode = enum { metadata, verified, build };
fn sourceIdentityMode(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, root: ?[]const u8, mode: IdentityMode) ![]u8 {
    const config_path = try std.fs.path.join(gpa, &.{ model_dir, "config.json" });
    defer gpa.free(config_path);
    const config = Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(1 << 24)) catch |e| switch (e) {
        error.FileNotFound => null,
        else => return e,
    };
    defer if (config) |c| gpa.free(c);
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
        error.FileNotFound => return identityFromMapped(gpa, null, config, shards.items),
        else => return e,
    };
    defer mapped.map.destroy(io);
    defer mapped.file.close(io);
    if (mapped.map.memory.len > st.index.max_bytes) return error.BadIndex;
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, mapped.map.memory, .{});
    defer parsed.deinit();
    const weight_map = try st.index.weightMap(parsed.value);
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    var it = weight_map.iterator();
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
        const stamp = try digests.Key.of(shard.file, io);
        const cache: digests.Cache = .{ .root = root };
        var sha: ?[32]u8 = null;
        if (mode == .build) {
            sha = hashBytes(shard.map.memory[0..shard.len]);
            try cache.remember(gpa, io, p, stamp, sha.?);
        } else if (try cache.checkedFile(gpa, io, shard.file)) |known| {
            if (mode == .verified) sha = known.value.sha256;
        }
        if (!stamp.equal(try digests.Key.of(shard.file, io))) return error.DigestFileChanged;
        try shards.append(gpa, .{ .name = name, .size = shard.len, .bytes = shard.map.memory, .stamp = stamp, .payload_sha256 = sha });
    }
    const out = try identityFromMapped(gpa, mapped.map.memory, config, shards.items);
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
    if (parsed.value != .object) return error.BadSafetensors;
    const meta = parsed.value.object.get("__metadata__") orelse return error.PackHasNoSource;
    if (meta != .object) return error.BadSafetensors;
    const source = meta.object.get("tf_source") orelse return error.PackHasNoSource;
    if (source != .string) return error.PackHasNoSource;
    return gpa.dupe(u8, source.string);
}

/// The rebuild advice used by every supplied-pack refusal; arguments remain literal shell words.
pub fn sourceRefusalMessage(w: *Io.Writer, path: []const u8, model_dir: []const u8) !void {
    return refusalMessage(w, path, model_dir, "has no recorded source");
}

fn refusalMessage(w: *Io.Writer, path: []const u8, model_dir: []const u8, reason: []const u8) !void {
    try w.print("pack {s} {s}; rebuild it with: tf-flashnext-pack ", .{ path, reason });
    try shellWord(w, model_dir);
    try w.writeByte(' ');
    try shellWord(w, std.fs.path.dirname(path) orelse ".");
    try w.writeAll(" --draft-vocab 'zig/src/families/flashnext/draft_vocab_default.txt'");
}

fn shellWord(w: *Io.Writer, value: []const u8) !void {
    try w.writeByte('\'');
    for (value) |c| if (c == '\'') try w.writeAll("'\\''") else try w.writeByte(c);
    try w.writeByte('\'');
}

fn namedSource(gpa: std.mem.Allocator, header: []const u8, path: []const u8, model_dir: []const u8) ![]u8 {
    return recordedSource(gpa, header) catch |err| {
        if (err == error.PackHasNoSource and !@import("builtin").is_test) {
            try logRefusal(gpa, path, model_dir, "has no recorded source");
        }
        return err;
    };
}

fn logRefusal(gpa: std.mem.Allocator, path: []const u8, model_dir: []const u8, reason: []const u8) !void {
    if (@import("builtin").is_test) return;
    var message = Io.Writer.Allocating.init(gpa);
    defer message.deinit();
    try refusalMessage(&message.writer, path, model_dir, reason);
    std.log.err("{s}", .{message.written()});
}

/// Both an unmarked pack and a pack from another checkpoint are refused before its tensors are used.
pub fn checkSource(gpa: std.mem.Allocator, io: Io, pack_path: []const u8, model_dir: []const u8) !void {
    const root = try digests.environmentRoot(gpa);
    defer if (root) |r| gpa.free(r);
    return checkSourceWithCache(gpa, io, pack_path, model_dir, root);
}
pub fn checkSourceWithCache(gpa: std.mem.Allocator, io: Io, pack_path: []const u8, model_dir: []const u8, root: ?[]const u8) !void {
    var mapped = try mapFile(io, pack_path);
    defer mapped.map.destroy(io);
    defer mapped.file.close(io);
    const hl = try headerLen(mapped.map.memory);
    const recorded = try namedSource(gpa, mapped.map.memory[8..][0..hl], pack_path, model_dir);
    defer gpa.free(recorded);
    const identity = try identityWithContent(gpa, io, model_dir, root, false);
    defer gpa.free(identity);
    try matchesSource(gpa, recorded, identity, pack_path, model_dir);
}

/// Validate a supplied pack's existing identity policy using only its unpopulated mapping and header.
pub fn checkIdentityPath(gpa: std.mem.Allocator, io: Io, path: []const u8, identity: []const u8, model_dir: []const u8) !void {
    var mapped = try mapFile(io, path);
    defer mapped.map.destroy(io);
    defer mapped.file.close(io);
    const hl = try headerLen(mapped.map.memory);
    const recorded = try namedSource(gpa, mapped.map.memory[8..][0..hl], path, model_dir);
    defer gpa.free(recorded);
    try matchesSource(gpa, recorded, identity, path, model_dir);
    try checkContent(gpa, io, recorded, path, model_dir);
}

/// The same identity refusal for pack bytes already mapped, before the loader consumes their tensors.
pub fn checkSourceMapped(gpa: std.mem.Allocator, io: Io, pack_bytes: []const u8, identity: []const u8, name: []const u8, model_dir: []const u8) !void {
    if (pack_bytes.len < 8) return error.BadSafetensors;
    const hl = try headerLen(pack_bytes);
    const recorded = try namedSource(gpa, pack_bytes[8..][0..hl], name, model_dir);
    defer gpa.free(recorded);
    try matchesSource(gpa, recorded, identity, name, model_dir);
    try checkContent(gpa, io, recorded, name, model_dir);
}

/// Add local verified-content checks without changing the cache API's independently supplied metadata key.
pub fn checkKnownContentPath(gpa: std.mem.Allocator, io: Io, path: []const u8, model_dir: []const u8) !void {
    var mapped = try mapFile(io, path);
    defer mapped.map.destroy(io);
    defer mapped.file.close(io);
    const hl = try headerLen(mapped.map.memory);
    const recorded = try namedSource(gpa, mapped.map.memory[8..][0..hl], path, model_dir);
    defer gpa.free(recorded);
    const root = try digests.environmentRoot(gpa);
    defer if (root) |r| gpa.free(r);
    const current = try identityWithContent(gpa, io, model_dir, root, false);
    defer gpa.free(current);
    if (!source_binding.matchesKnown(recorded, current)) return error.PackSourceMismatch;
}
fn checkContent(gpa: std.mem.Allocator, io: Io, recorded: []const u8, path: []const u8, model_dir: []const u8) !void {
    const root = try digests.environmentRoot(gpa);
    defer if (root) |r| gpa.free(r);
    const current = try identityWithContent(gpa, io, model_dir, root, false);
    defer gpa.free(current);
    try matchesSource(gpa, recorded, current, path, model_dir);
}
pub const source_binding = @import("pack_source_binding.zig");
fn matchesSource(gpa: std.mem.Allocator, recorded: []const u8, identity: []const u8, path: []const u8, model_dir: []const u8) !void {
    if (!source_binding.matches(recorded, identity)) {
        try logRefusal(gpa, path, model_dir, "belongs to a different checkpoint or identity format");
        return error.PackSourceMismatch;
    }
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

/// Normal caches require explicit matching source metadata and valid, non-empty tensor ranges.
pub fn packReady(gpa: std.mem.Allocator, io: Io, path: []const u8, identity: []const u8) !bool {
    const file = Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    defer file.close(io);
    const len = try file.length(io);
    if (len < 8) return false;
    var buf: [4096]u8 = undefined;
    var reader = file.reader(io, &buf);
    var prefix: [8]u8 = undefined;
    reader.interface.readSliceAll(&prefix) catch return false;
    const hl = std.mem.readInt(u64, &prefix, .little);
    if (hl == 0 or hl > len - 8 or hl > 64 << 20) return false;
    const header = try gpa.alloc(u8, @intCast(hl));
    defer gpa.free(header);
    reader.interface.readSliceAll(header) catch return false;
    const recorded = recordedSource(gpa, header) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return false,
    };
    defer gpa.free(recorded);
    if (!source_binding.matches(recorded, identity)) return false;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const tensors = st.parseHeader(arena.allocator(), header, @intCast(len - 8 - hl)) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return false,
    };
    var end: usize = 0;
    for (tensors.values()) |tensor| end = @max(end, tensor.end);
    return tensors.count() > 0 and end > 0 and end == len - 8 - hl;
}

/// The decode and prompt packs have complete headers belonging to this checkpoint.
pub fn packsReady(gpa: std.mem.Allocator, io: Io, cache_dir: []const u8, identity: []const u8) !bool {
    for ([_][]const u8{ "pack.safetensors", "pack_mlx.safetensors" }) |name| {
        const path = try std.fs.path.join(gpa, &.{ cache_dir, name });
        defer gpa.free(path);
        if (!try packReady(gpa, io, path, identity)) return false;
    }
    return true;
}
