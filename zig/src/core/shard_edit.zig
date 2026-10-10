//! Replaced tensors written into a checkpoint's own shards, index and config: in place when sizes hold, else by rename.
const std = @import("std");
const st = @import("safetensors.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Value = std.json.Value;

/// A tensor written in place of `name`; `drop` names companions that leave with it, such as an affine weight's scales.
pub const Replacement = struct {
    name: []const u8,
    dtype: st.DType,
    shape: []const usize,
    bytes: []const u8,
    drop: []const []const u8 = &.{},
};

const index_name = "model.safetensors.index.json";
const single_name = "model.safetensors";

/// The first shard, index or config.json in `dir` that is a symlink or has another hard link (a write in place would reach shared data), or null.
pub fn linked(arena: Allocator, io: Io, dir: []const u8) !?[]const u8 {
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    while (try it.next(io)) |e| {
        const ours = std.mem.endsWith(u8, e.name, ".safetensors") or std.mem.eql(u8, e.name, index_name) or std.mem.eql(u8, e.name, "config.json");
        if (!ours) continue;
        const info = try d.statFile(io, e.name, .{ .follow_symlinks = false });
        if (info.kind == .sym_link or info.nlink > 1) return try arena.dupe(u8, e.name);
    }
    return null;
}

/// Applies `edits` at `dir`: the touched shards, then the index, then config.json marks each module unquantized.
pub fn bake(gpa: Allocator, io: Io, dir: []const u8, edits: []const Replacement) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    for (edits) |r| if (r.bytes.len != numel(r.shape) * r.dtype.size() or !std.mem.endsWith(u8, r.name, ".weight")) return error.BadReplacement;
    const index_path = try std.fs.path.join(arena, &.{ dir, index_name });
    const index: ?Value = readJson(arena, io, index_path) catch |e| if (e == error.FileNotFound) null else return e;
    var shards: std.StringArrayHashMapUnmanaged(std.ArrayList(Replacement)) = .empty;
    for (edits) |r| {
        const shard = if (index) |ix| str((try field(try field(ix, "weight_map"), r.name))) orelse return error.BadIndex else single_name;
        const slot = try shards.getOrPut(arena, shard);
        if (!slot.found_existing) slot.value_ptr.* = .empty;
        try slot.value_ptr.append(arena, r);
    }
    var grown: i64 = 0;
    for (shards.keys(), shards.values()) |shard, list| {
        const path = try std.fs.path.join(arena, &.{ dir, shard });
        if (!try inPlace(gpa, io, path, list.items)) grown += try rewrite(gpa, io, path, list.items);
    }
    if (index) |ix| {
        var root = ix;
        const map = &(root.object.getPtr("weight_map") orelse return error.BadIndex).object;
        for (edits) |r| for (r.drop) |d| {
            _ = map.orderedRemove(d);
        };
        if (root.object.getPtr("metadata")) |meta| if (meta.* == .object) if (meta.object.getPtr("total_size")) |size| {
            if (size.* == .integer) size.integer += grown;
        };
        try writeJson(arena, index_path, root);
    }
    const config_path = try std.fs.path.join(arena, &.{ dir, "config.json" });
    var config = try readJson(arena, io, config_path);
    if (config != .object) return error.BadConfig;
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| if (config.object.getPtr(key)) |q| if (q.* == .object) {
        for (edits) |r| try q.object.put(arena, r.name[0 .. r.name.len - ".weight".len], .{ .bool = false });
    };
    try writeJson(arena, config_path, config);
}

/// `edits` written over their tensors where they lie, if each keeps its dtype and size (else false, nothing written).
fn inPlace(gpa: Allocator, io: Io, path: []const u8, edits: []const Replacement) !bool {
    var places: [512]usize = undefined;
    if (edits.len > places.len) return false;
    {
        var file = try st.File.open(gpa, io, path);
        defer file.close(io);
        for (edits, 0..) |r, i| {
            const e = file.names.get(r.name) orelse return false;
            if (e.dtype != r.dtype or e.end - e.begin != r.bytes.len) return false;
            for (r.drop) |d| if (file.names.contains(d)) return false;
            places[i] = file.data + e.begin;
        }
    }
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const fd = std.c.open(try arena_state.allocator().dupeSentinel(u8, path, 0), .{ .ACCMODE = .WRONLY });
    if (fd < 0) return error.ShardWrite;
    defer _ = std.c.close(fd);
    for (edits, places[0..edits.len]) |r, at| {
        var done: usize = 0;
        while (done < r.bytes.len) {
            const n = std.c.pwrite(fd, r.bytes.ptr + done, @min(r.bytes.len - done, 1 << 30), @intCast(at + done));
            if (n <= 0) return error.ShardWrite;
            done += @intCast(n);
        }
    }
    if (std.c.fsync(fd) != 0) return error.ShardWrite;
    return true;
}

/// One shard rewritten with `edits`, tensors in their old order at contiguous offsets; returns the bytes it gained.
pub fn rewrite(gpa: Allocator, io: Io, path: []const u8, edits: []const Replacement) !i64 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var file = try st.File.open(gpa, io, path);
    defer file.close(io);
    for (edits) |r| if (!file.names.contains(r.name)) return error.MissingTensor;
    const raw = file.map.memory;
    const original = try std.json.parseFromSliceLeaky(Value, arena, raw[8..file.data], .{});
    if (original != .object) return error.BadSafetensors;
    const entries = file.names.values();
    const order = try arena.alloc(usize, entries.len);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, @as([]const st.Entry, entries), struct {
        fn lt(e: []const st.Entry, a: usize, b: usize) bool {
            return e[a].begin < e[b].begin;
        }
    }.lt);
    var header: std.json.ObjectMap = .empty;
    if (original.object.get("__metadata__")) |m| try header.put(arena, "__metadata__", m);
    var pieces: std.ArrayList([]const u8) = .empty;
    var offset: usize = 0;
    var before: usize = 0;
    for (order) |i| {
        const name, const e = .{ file.names.keys()[i], entries[i] };
        before += e.end - e.begin;
        if (dropped(edits, name)) continue;
        const r = find(edits, name);
        const bytes = if (r) |x| x.bytes else raw[file.data + e.begin .. file.data + e.end];
        const shape = if (r) |x| x.shape else e.shape[0..e.rank];
        try header.put(arena, name, try entry(arena, if (r) |x| x.dtype else e.dtype, shape, offset, offset + bytes.len));
        try pieces.append(arena, bytes);
        offset += bytes.len;
    }
    const text = try std.json.Stringify.valueAlloc(arena, Value{ .object = header }, .{});
    const padded = std.mem.alignForward(usize, text.len, 8);
    var head: [8]u8 = undefined;
    std.mem.writeInt(u64, &head, padded, .little);
    const spaces = try arena.alloc(u8, padded - text.len);
    @memset(spaces, ' ');
    try pieces.insertSlice(arena, 0, &.{ &head, text, spaces });
    try writeFile(arena, path, pieces.items);
    return @as(i64, @intCast(offset)) - @as(i64, @intCast(before));
}

fn entry(arena: Allocator, dtype: st.DType, shape: []const usize, begin: usize, end: usize) !Value {
    var dims = std.json.Array.init(arena);
    for (shape) |d| try dims.append(.{ .integer = @intCast(d) });
    var offs = std.json.Array.init(arena);
    try offs.appendSlice(&.{ .{ .integer = @intCast(begin) }, .{ .integer = @intCast(end) } });
    var o: std.json.ObjectMap = .empty;
    try o.put(arena, "dtype", .{ .string = dtypeName(dtype) });
    try o.put(arena, "shape", .{ .array = dims });
    try o.put(arena, "data_offsets", .{ .array = offs });
    return .{ .object = o };
}

/// A dtype as a safetensors header names it: its tag in capitals (bf16 as BF16).
fn dtypeName(d: st.DType) []const u8 {
    switch (d) {
        inline else => |tag| {
            const name = comptime blk: {
                var up: [@tagName(tag).len]u8 = undefined;
                _ = std.ascii.upperString(&up, @tagName(tag));
                const out = up;
                break :blk &out;
            };
            return name;
        },
    }
}

fn numel(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

fn find(edits: []const Replacement, name: []const u8) ?Replacement {
    for (edits) |r| if (std.mem.eql(u8, r.name, name)) return r;
    return null;
}

fn dropped(edits: []const Replacement, name: []const u8) bool {
    for (edits) |r| for (r.drop) |d| if (std.mem.eql(u8, d, name)) return true;
    return false;
}

fn field(v: Value, key: []const u8) !Value {
    if (v != .object) return error.BadIndex;
    return v.object.get(key) orelse error.BadIndex;
}

fn str(v: Value) ?[]const u8 {
    return if (v == .string) v.string else null;
}

fn readJson(arena: Allocator, io: Io, path: []const u8) !Value {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 26));
    return std.json.parseFromSliceLeaky(Value, arena, bytes, .{});
}

fn writeJson(arena: Allocator, path: []const u8, v: Value) !void {
    const text = try std.json.Stringify.valueAlloc(arena, v, .{ .whitespace = .indent_2 });
    try writeFile(arena, path, &.{ text, "\n" });
}

/// `parts` to `path` through `path`.tmp, synced, then renamed over it; a reader sees the old file or the new one.
fn writeFile(arena: Allocator, path: []const u8, parts: []const []const u8) !void {
    const tmp = try std.fmt.allocPrintSentinel(arena, "{s}.tmp", .{path}, 0);
    const dest = try arena.dupeSentinel(u8, path, 0);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.ShardWrite;
    errdefer _ = std.c.unlink(tmp);
    {
        defer _ = std.c.close(fd);
        for (parts) |p| try put(fd, p);
        if (std.c.fsync(fd) != 0) return error.ShardWrite;
    }
    if (std.c.rename(tmp, dest) != 0) return error.ShardWrite;
}

fn put(fd: c_int, b: []const u8) !void {
    var done: usize = 0;
    while (done < b.len) {
        const n = std.c.write(fd, b.ptr + done, @min(b.len - done, 1 << 30));
        if (n <= 0) return error.ShardWrite;
        done += @intCast(n);
    }
}

test "an affine weight becomes bf16 in its shard, its scales and biases leave the index, config marks it unquantized" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const head =
        \\{"__metadata__":{"format":"mlx"},"a.weight":{"dtype":"U32","shape":[2,2],"data_offsets":[0,16]},
        \\"a.scales":{"dtype":"BF16","shape":[2,1],"data_offsets":[16,20]},"a.biases":{"dtype":"BF16","shape":[2,1],
        \\"data_offsets":[20,24]},"b.weight":{"dtype":"F32","shape":[3],"data_offsets":[24,36]}}
    ;
    var shard: [8 + head.len + 36]u8 = undefined;
    std.mem.writeInt(u64, shard[0..8], head.len, .little);
    @memcpy(shard[8..][0..head.len], head);
    for (shard[8 + head.len ..], 0..) |*b, i| b.* = @intCast(i);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = &shard });
    try tmp.dir.writeFile(io, .{ .sub_path = index_name, .data =
        \\{"metadata":{"total_size":36},"weight_map":{"a.weight":"model-00001-of-00001.safetensors",
        \\"a.scales":"model-00001-of-00001.safetensors","a.biases":"model-00001-of-00001.safetensors",
        \\"b.weight":"model-00001-of-00001.safetensors"}}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"x","quantization":{"group_size":64,"bits":4},"quantization_config":{"group_size":64,"bits":4}}
    });
    const dir = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer gpa.free(dir);
    var w: [32]u8 = undefined;
    for (&w, 0..) |*b, i| b.* = @intCast(100 + i);
    const edit: Replacement = .{ .name = "a.weight", .dtype = .bf16, .shape = &.{ 2, 8 }, .bytes = &w, .drop = &.{ "a.scales", "a.biases" } };
    try bake(gpa, io, dir, &.{edit});

    const shard_path = try std.fs.path.join(gpa, &.{ dir, "model-00001-of-00001.safetensors" });
    defer gpa.free(shard_path);
    var f = try st.File.open(gpa, io, shard_path);
    defer f.close(io);
    try std.testing.expectEqual(@as(usize, 2), f.names.count());
    try std.testing.expectEqual(@as(usize, 0), f.data % 8);
    const a = f.get("a.weight").?;
    try std.testing.expect(a.is(.bf16, &.{ 2, 8 }));
    try std.testing.expectEqualSlices(u8, &w, a.bytes);
    try std.testing.expectEqualSlices(u8, shard[8 + head.len + 24 ..], f.get("b.weight").?.bytes);
    try std.testing.expectEqual(@as(usize, 32), f.names.get("b.weight").?.begin);

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const ix = try readJson(arena, io, try std.fs.path.join(arena, &.{ dir, index_name }));
    const map = (try field(ix, "weight_map")).object;
    try std.testing.expect(map.get("a.scales") == null and map.get("a.biases") == null and map.get("a.weight") != null);
    try std.testing.expectEqual(@as(i64, 44), (try field(try field(ix, "metadata"), "total_size")).integer);
    const config = try readJson(arena, io, try std.fs.path.join(arena, &.{ dir, "config.json" }));
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| try std.testing.expectEqual(false, (try field(try field(config, key), "a")).bool);

    try std.testing.expectError(error.BadReplacement, bake(gpa, io, dir, &.{.{ .name = "b.weight", .dtype = .f32, .shape = &.{4}, .bytes = &w }}));
    try std.testing.expectError(error.BadIndex, bake(gpa, io, dir, &.{.{ .name = "c.weight", .dtype = .u8, .shape = &.{4}, .bytes = w[0..4] }}));
}

test "a folder whose shard is a symlink or a hard link is refused for learning; its own files pass" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "blobs");
    for ([_][]const u8{ "config.json", index_name, "model-00001-of-00002.safetensors", "blobs/b2", "notes.txt" }) |name| try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "x" });
    try tmp.dir.symLink(io, "model-00001-of-00002.safetensors", "readme-link", .{});
    const dir = try std.fs.path.join(arena, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    try std.testing.expect((try linked(arena, io, dir)) == null);
    try tmp.dir.symLink(io, "blobs/b2", "model-00002-of-00002.safetensors", .{});
    try std.testing.expectEqualStrings("model-00002-of-00002.safetensors", (try linked(arena, io, dir)).?);
    try tmp.dir.deleteFile(io, "model-00002-of-00002.safetensors");
    try tmp.dir.hardLink("blobs/b2", tmp.dir, "model-00002-of-00002.safetensors", io, .{});
    try std.testing.expectEqualStrings("model-00002-of-00002.safetensors", (try linked(arena, io, dir)).?);
}
