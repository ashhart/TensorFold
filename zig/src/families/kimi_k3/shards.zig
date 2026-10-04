//! The checkpoint's safetensors shards with no copy: a file is mapped whole, one Metal buffer over it, when first used.
const std = @import("std");
const mtl = @import("metal");
const store = @import("store.zig");

const Entry = struct { offset: usize, bytes: usize, dtype: store.DType, shape: [3]u32, rank: u8 };
const File = struct { path: [:0]const u8, map: ?mtl.MappedFile = null, buffer: ?mtl.Buffer = null, entries: std.StringHashMapUnmanaged(Entry) = .empty };

/// Our tensor name for a checkpoint name: the language model's prefixes dropped; null for vision tensors.
pub fn shortName(name: []const u8) ?[]const u8 {
    for ([_][]const u8{ "language_model.model.", "language_model." }) |p| {
        if (std.mem.startsWith(u8, name, p)) return name[p.len..];
    }
    return null;
}

pub const Shards = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    device: mtl.Device,
    files: std.ArrayList(File) = .empty,
    /// Our short name to the index of the file that holds it, from model.safetensors.index.json.
    where: std.StringHashMapUnmanaged(u32) = .empty,

    /// Read `dir`'s index only; files are mapped as their tensors are asked for.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Shards {
        var s = Shards{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa), .device = device };
        errdefer s.deinit();
        const a = s.arena.allocator();
        const index = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), a, .limited(1 << 30));
        const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
        const parsed = try std.json.parseFromSliceLeaky(Index, a, index, .{ .ignore_unknown_fields = true });
        var file_of = std.StringHashMapUnmanaged(u32){};
        for (parsed.weight_map.map.keys(), parsed.weight_map.map.values()) |name, file| {
            const gop = try file_of.getOrPut(a, file);
            if (!gop.found_existing) {
                gop.value_ptr.* = @intCast(s.files.items.len);
                try s.files.append(a, .{ .path = try std.fs.path.joinZ(a, &.{ dir, file }) });
            }
            if (shortName(name)) |short| try s.where.put(a, short, gop.value_ptr.*);
        }
        return s;
    }

    /// Map file `i` and wrap it in a no-copy buffer; its header must put every tensor 8-byte aligned.
    fn load(s: *Shards, i: u32) !*File {
        const f = &s.files.items[i];
        if (f.buffer != null) return f;
        const a = s.arena.allocator();
        const map = try mtl.MappedFile.open(f.path);
        errdefer map.deinit();
        const header_len = std.mem.readInt(u64, map.bytes[0..8], .little);
        const data = 8 + header_len;
        if (data % 8 != 0 or data > map.size) return error.MisalignedShard;
        const header = try std.json.parseFromSliceLeaky(std.json.Value, a, map.bytes[8..data], .{});
        var it = header.object.iterator();
        while (it.next()) |kv| {
            const name = shortName(kv.key_ptr.*) orelse continue;
            const o = kv.value_ptr.object;
            const off = o.get("data_offsets").?.array.items;
            const shape = o.get("shape").?.array.items;
            if (shape.len > 3) return error.RankTooHigh;
            var e = Entry{ .offset = data + @as(usize, @intCast(off[0].integer)), .bytes = @intCast(off[1].integer - off[0].integer), .dtype = try store.DType.parse(o.get("dtype").?.string), .shape = .{ 1, 1, 1 }, .rank = @intCast(shape.len) };
            for (shape, 0..) |d, k| e.shape[k] = @intCast(d.integer);
            if (e.offset % 8 != 0 or e.offset + e.bytes > map.size) return error.MisalignedTensor;
            try f.entries.put(a, name, e);
        }
        f.buffer = try s.device.bufferNoCopy(map.bytes.ptr, map.bytes.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        f.map = map;
        return f;
    }

    pub fn source(s: *Shards) store.Source {
        return .{ .ptr = s, .getFn = getFn };
    }

    fn getFn(ptr: *anyopaque, name: []const u8, dtype: store.DType, shape: []const u32) anyerror!store.Tensor {
        const s: *Shards = @ptrCast(@alignCast(ptr));
        const i = s.where.get(name) orelse {
            std.log.err("the index has no tensor {s}", .{name});
            return error.MissingTensor;
        };
        const f = try s.load(i);
        const e = f.entries.get(name) orelse {
            std.log.err("{s} is not in {s}'s header", .{ name, f.path });
            return error.MissingTensor;
        };
        const t = store.Tensor{ .ref = .{ .buf = f.buffer.?, .off = e.offset }, .dtype = e.dtype, .shape = e.shape, .rank = e.rank };
        if (shape.len == 0) {
            if (t.dtype != dtype) return error.TensorShape;
        } else t.expect(dtype, shape) catch |err| {
            std.log.err("in tensor {s}", .{name});
            return err;
        };
        return t;
    }

    /// The buffers of every file mapped so far, for a residency set.
    pub fn buffers(s: *const Shards, out: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator) !void {
        for (s.files.items) |f| if (f.buffer) |b| try out.append(gpa, b);
    }

    pub fn deinit(s: *Shards) void {
        for (s.files.items) |f| {
            if (f.buffer) |b| b.deinit();
            if (f.map) |m| m.deinit();
        }
        s.arena.deinit();
    }
};

test "checkpoint names map to ours" {
    try std.testing.expectEqualStrings("layers.0.self_attn.q_proj.weight", shortName("language_model.model.layers.0.self_attn.q_proj.weight").?);
    try std.testing.expectEqualStrings("lm_head.weight", shortName("language_model.lm_head.weight").?);
    try std.testing.expect(shortName("vision_tower.patch_embed.proj.weight") == null);
}
