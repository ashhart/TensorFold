//! GLM-5.3-Flash's converted checkpoint: shard headers, files opened once, tensors read whole or as row ranges.

const std = @import("std");
const api = @import("api.zig");

pub const BASE = "model.language_model.";
pub const WORLD = 2;

// -- the checkpoint: shard headers, files opened once, tensors read whole or as row ranges ------------------------

pub const Entry = struct { dtype: []const u8, shape: []const i64, begin: u64, end: u64, file: usize };

pub const Ckpt = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    files: std.ArrayList(std.Io.File) = .empty,
    paths: std.ArrayList([]const u8) = .empty,
    entries: std.StringHashMap(Entry),

    /// ``dir``: the served folder; a shard symlinked under /cache/tf/ resolves against its parent (container or host).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Ckpt {
        var ck: Ckpt = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .entries = .init(gpa) };
        errdefer ck.deinit();
        const a = ck.arena.allocator();
        const index_path = try resolve(a, io, dir, "model.safetensors.index.json");
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(1 << 26));
        const idx = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        var shard_of = std.StringHashMap(usize).init(a);
        var it = idx.object.get("weight_map").?.object.iterator();
        while (it.next()) |e| {
            const shard = e.value_ptr.string;
            if (shard_of.contains(shard)) continue;
            const path = try resolve(a, io, dir, shard);
            try shard_of.put(shard, ck.files.items.len);
            try ck.files.append(gpa, try std.Io.Dir.cwd().openFile(io, path, .{}));
            try ck.paths.append(gpa, path);
            try ck.readHeader(ck.files.items.len - 1);
        }
        return ck;
    }

    pub fn readHeader(ck: *Ckpt, f: usize) !void {
        const a = ck.arena.allocator();
        var head: [8]u8 = undefined;
        _ = try ck.files.items[f].readPositionalAll(ck.io, &head, 0);
        const n = std.mem.readInt(u64, &head, .little);
        const text = try a.alloc(u8, n);
        _ = try ck.files.items[f].readPositionalAll(ck.io, text, 8);
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        var it = v.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const o = e.value_ptr.object;
            const shp = o.get("shape").?.array.items;
            const shape = try a.alloc(i64, shp.len);
            for (shp, 0..) |x, i| shape[i] = x.integer;
            const offs = o.get("data_offsets").?.array.items;
            try ck.entries.put(e.key_ptr.*, .{ .dtype = o.get("dtype").?.string, .shape = shape, .begin = 8 + n + @as(u64, @intCast(offs[0].integer)), .end = 8 + n + @as(u64, @intCast(offs[1].integer)), .file = f });
        }
    }

    pub fn deinit(ck: *Ckpt) void {
        // the shards are read once: their page cache would starve NCCL's first registration (and the n-gram tables)
        for (ck.files.items) |f| {
            _ = std.os.linux.fadvise(f.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
            f.close(ck.io);
        }
        ck.files.deinit(ck.gpa);
        ck.paths.deinit(ck.gpa);
        ck.entries.deinit();
        ck.arena.deinit();
    }

    pub fn has(ck: *const Ckpt, name: []const u8) bool {
        return ck.entries.contains(name);
    }

    pub fn get(ck: *const Ckpt, name: []const u8) !Entry {
        return ck.entries.get(name) orelse {
            std.log.err("glm weights: the checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    /// ``len`` bytes at ``off`` into the tensor's data.
    pub fn readAt(ck: *const Ckpt, e: Entry, off: u64, out: []u8) !void {
        if (off + out.len > e.end - e.begin) return error.ShortRead;
        const got = try ck.files.items[e.file].readPositionalAll(ck.io, out, e.begin + off);
        if (got != out.len) return error.ShortRead;
    }

    pub fn readAll(ck: *const Ckpt, gpa: std.mem.Allocator, name: []const u8) ![]align(16) u8 {
        const e = try ck.get(name);
        const buf = try gpa.alignedAlloc(u8, .@"16", e.end - e.begin);
        errdefer gpa.free(buf);
        try ck.readAt(e, 0, buf);
        return buf;
    }
};

pub fn resolve(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    if (std.Io.Dir.cwd().openFile(io, path, .{})) |f| {
        f.close(io);
        return path;
    } else |_| {}
    var buf: [4096]u8 = undefined;
    const n = std.Io.Dir.cwd().readLink(io, path, &buf) catch return path;
    const target = buf[0..n];
    const container = "/cache/tf/";
    if (std.mem.startsWith(u8, target, container))
        return std.fs.path.join(a, &.{ std.fs.path.dirname(dir) orelse ".", target[container.len..] });
    if (std.fs.path.isAbsolute(target)) return a.dupe(u8, target);
    return std.fs.path.join(a, &.{ dir, target });
}

pub fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

pub fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7FC0;
    return @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

pub fn f16ToBf16(h: u16) u16 {
    return f32ToBf16(@floatCast(@as(f16, @bitCast(h))));
}

pub fn numelOf(shape: []const i64) usize {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    return n;
}

pub fn dtypeOf(s: []const u8) !api.DType {
    if (std.mem.eql(u8, s, "BF16")) return .bf16;
    if (std.mem.eql(u8, s, "F32")) return .f32;
    if (std.mem.eql(u8, s, "F16")) return .f16;
    return error.UnsupportedDType;
}

pub fn bf16Value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn fullName(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, BASE ++ fmt, args);
}
