//! Header-first resident loading: no mapped weights, deferred disk lookup or duplicate Engram projections.
const std = @import("std");
const mtl = @import("metal");
const st = @import("../../core/safetensors.zig");
const cp = @import("../../core/checkpoint_metal.zig");
const paced = @import("../../core/paced_read.zig");

pub const Reference = struct { shard: usize, entry: st.Entry };
pub const Shard = struct {
    path: [:0]const u8,
    file_bytes: usize,
    data_start: usize,
    begin: usize,
    end: usize,
    names: st.Header,
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    shards: std.ArrayList(Shard) = .empty,
    tensors: std.StringHashMapUnmanaged(Reference) = .empty,
    bytes: usize = 0,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, table_dir: []const u8) !Plan {
        var self = Plan{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer self.deinit();
        const a = self.arena.allocator();
        const index_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const index_bytes = try std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(64 << 20));
        const index = try std.json.parseFromSliceLeaky(struct { weight_map: std.json.Value }, a, index_bytes, .{
            .ignore_unknown_fields = true,
        });
        if (index.weight_map != .object or index.weight_map.object.count() == 0) return error.BadCheckpointIndex;
        var files: std.StringHashMapUnmanaged(void) = .empty;
        var entries = index.weight_map.object.iterator();
        while (entries.next()) |item| {
            if (item.value_ptr.* != .string) return error.BadCheckpointIndex;
            const file = item.value_ptr.string;
            if (!safeBasename(file)) return error.BadCheckpointIndex;
            try files.put(a, file, {});
        }
        const ordered = try a.alloc([]const u8, files.count());
        var file_it = files.keyIterator();
        var n: usize = 0;
        while (file_it.next()) |file| : (n += 1) ordered[n] = file.*;
        std.mem.sort([]const u8, ordered, {}, less);
        var owned: std.StringHashMapUnmanaged([]const u8) = .empty;
        entries.reset();
        while (entries.next()) |item| try owned.put(a, item.key_ptr.*, item.value_ptr.string);
        for (ordered) |file| try self.add(try std.fs.path.join(a, &.{ dir, file }), null, &owned);
        if (self.tensors.count() != index.weight_map.object.count()) return error.CheckpointIndexMismatch;
        entries = index.weight_map.object.iterator();
        while (entries.next()) |item| {
            const ref = self.tensors.get(item.key_ptr.*) orelse return error.CheckpointIndexMismatch;
            if (!std.mem.eql(u8, std.fs.path.basename(self.shards.items[ref.shard].path), item.value_ptr.string))
                return error.CheckpointIndexMismatch;
        }
        for ([_]struct { layer: u32, file: []const u8, rows: usize }{
            .{ .layer = 1, .file = "model-00047-of-00048.safetensors", .rows = 384006168 },
            .{ .layer = 14, .file = "model-00048-of-00048.safetensors", .rows = 384016682 },
        }) |table| {
            const prefix = try std.fmt.allocPrint(a, "layers.{d}.engram.embed.", .{table.layer});
            try self.add(try std.fs.path.join(a, &.{ table_dir, table.file }), prefix, null);
            const weight = try std.fmt.allocPrint(a, "{s}weight", .{prefix});
            const scale = try std.fmt.allocPrint(a, "{s}scale", .{prefix});
            try self.require(weight, .f8_e4m3, &.{ table.rows, 256 });
            try self.require(scale, .f8_e8m0, &.{ table.rows, 8 });
            if (self.shards.items[self.shards.items.len - 1].names.count() != 2) return error.UnexpectedEngramTensor;
        }
        return self;
    }

    pub fn deinit(self: *Plan) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn get(self: *const Plan, name: []const u8) !st.Entry {
        return (self.tensors.get(name) orelse return error.MissingTensor).entry;
    }

    pub fn has(self: *const Plan, name: []const u8) bool {
        return self.tensors.contains(name);
    }

    pub fn require(self: *const Plan, name: []const u8, dtype: st.DType, shape: []const usize) !void {
        const e = try self.get(name);
        if (e.dtype != dtype or e.rank != shape.len or !std.mem.eql(usize, e.shape[0..e.rank], shape)) {
            std.log.err("unsupported tensor {s}: dtype {s}, shape {any}", .{ name, @tagName(e.dtype), e.shape[0..e.rank] });
            return error.UnsupportedTensor;
        }
    }

    fn add(self: *Plan, path: []const u8, prefix: ?[]const u8, owned: ?*const std.StringHashMapUnmanaged([]const u8)) !void {
        const a = self.arena.allocator();
        const own_path = try a.dupeSentinel(u8, path, 0);
        const fd = std.c.open(own_path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        const length = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (length < 8) return error.BadSafetensors;
        const file_bytes: usize = @intCast(length);
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const header_bytes = std.mem.readInt(u64, &head, .little);
        if (header_bytes > @min(file_bytes - 8, 64 << 20)) return error.BadSafetensors;
        const json = try a.alloc(u8, @intCast(header_bytes));
        try readAll(fd, json, 8);
        const data_start: usize = 8 + @as(usize, @intCast(header_bytes));
        const names = try st.parseHeaderPrefix(a, json, file_bytes - data_start, prefix);
        if (names.count() == 0) return error.MissingTensor;
        var begin: usize = if (prefix == null) 0 else file_bytes - data_start;
        var end: usize = if (prefix == null) file_bytes - data_start else 0;
        for (names.keys(), names.values()) |name, entry| {
            // A rank-split pack duplicates whole-load tensors beside the shards; the index decides the owner file.
            if (owned) |map| {
                const owner = map.get(name) orelse continue;
                if (!std.mem.eql(u8, owner, std.fs.path.basename(path))) continue;
            }
            begin = @min(begin, entry.begin);
            end = @max(end, entry.end);
            const slot = try self.tensors.getOrPut(a, name);
            if (slot.found_existing) {
                return error.DuplicateTensor;
            }
            slot.value_ptr.* = .{ .shard = self.shards.items.len, .entry = entry };
        }
        if (prefix != null) {
            var selected_bytes: usize = 0;
            for (names.values()) |entry| selected_bytes = try std.math.add(usize, selected_bytes, entry.end - entry.begin);
            if (selected_bytes != end - begin) return error.NoncontiguousEngramTables;
        }
        self.bytes = try std.math.add(usize, self.bytes, end - begin);
        try self.shards.append(a, .{ .path = own_path, .file_bytes = file_bytes, .data_start = data_start, .begin = begin, .end = end, .names = names });
    }

    /// Only called after architecture validation and the complete memory admission calculation.
    pub fn materialize(self: *const Plan, gpa: std.mem.Allocator, device: mtl.Device, residency: mtl.ResidencySet, queue: mtl.Queue) !cp.Checkpoint {
        var out = cp.Checkpoint.init(gpa);
        errdefer out.deinit();
        const stage = try paced.stage(gpa, 32 << 20);
        defer gpa.free(stage);
        var warm = mtl.keepalive.Target{ .queue = queue, .sets = @as(*const [1]mtl.ResidencySet, &residency) };
        var last_tick: f64 = -std.math.inf(f64);
        var short: paced.Memory = undefined;
        for (self.shards.items, 0..) |shard, i| {
            const fd = std.c.open(shard.path, .{ .ACCMODE = .RDONLY });
            if (fd < 0) return error.OpenFailed;
            defer _ = std.c.close(fd);
            if (std.c.fcntl(fd, 48, @as(c_int, 1)) != 0) return error.UncachedReadUnavailable;
            if (std.c.lseek(fd, 0, std.c.SEEK.END) != shard.file_bytes) return error.CheckpointChanged;
            const bytes = shard.end - shard.begin;
            const buffer = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            var owned = false;
            errdefer if (!owned) buffer.deinit();
            try out.own(buffer, bytes);
            owned = true;
            residency.add(buffer);
            residency.commit();
            residency.requestResidency();
            var copied: usize = 0;
            while (copied < bytes) {
                const count = @min(bytes - copied, stage.len - paced.alignment);
                paced.read(fd, buffer.contents()[copied..][0..count], shard.data_start + shard.begin + copied, stage, &short) catch |err| {
                    if (err == error.MemoryShort) {
                        var msg: [512]u8 = undefined;
                        std.log.err("deepseek-v41: {s}", .{paced.refusal(&msg, short, paced.floor())});
                    }
                    return err;
                };
                copied += count;
                const now = mtl.clock.seconds();
                if (now - last_tick >= 0.4) {
                    mtl.keepalive.Target.tick(&warm);
                    last_tick = now;
                }
            }
            for (shard.names.keys(), shard.names.values()) |name, e| {
                try out.set(name, .{ .buffer = buffer, .offset = e.begin - shard.begin, .bytes = e.end - e.begin, .dtype = e.dtype, .shape = e.shape, .rank = e.rank });
            }
            std.log.info("resident checkpoint shard {d}/{d}: {d} bytes, residency set {d} bytes", .{ i + 1, self.shards.items.len, bytes, residency.allocatedSize() });
        }
        residency.commit();
        residency.requestResidency();
        return out;
    }
};

fn readAll(fd: std.c.fd_t, dest: []u8, at: usize) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

fn safeBasename(s: []const u8) bool {
    return s.len > 0 and !std.mem.eql(u8, s, ".") and !std.mem.eql(u8, s, "..") and
        std.mem.indexOfAny(u8, s, "/\\") == null and std.mem.indexOfScalar(u8, s, 0) == null and
        std.mem.endsWith(u8, s, ".safetensors");
}

fn less(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

test "selected table plan excludes projections and preserves tensor byte offsets" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header =
        \\{"padding":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},
        \\ "engram.embed.weight":{"dtype":"U8","shape":[4],"data_offsets":[3,7]},
        \\ "engram.embed.scale":{"dtype":"U8","shape":[2],"data_offsets":[7,9]},
        \\ "engram.wkv":{"dtype":"U8","shape":[1],"data_offsets":[9,10]}}
    ;
    var data: [8 + header.len + 10]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    @memset(data[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "table.safetensors", .data = &data });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/table.safetensors", .{tmp.sub_path});
    defer a.free(path);
    var plan = Plan{ .arena = std.heap.ArenaAllocator.init(a) };
    defer plan.deinit();
    try plan.add(path, "engram.embed.", null);
    try std.testing.expectEqual(@as(usize, 6), plan.bytes);
    try std.testing.expectEqual(@as(usize, 3), plan.shards.items[0].begin);
    try std.testing.expectEqual(@as(usize, 9), plan.shards.items[0].end);
    try std.testing.expectEqual(@as(usize, 7), (try plan.get("engram.embed.scale")).begin);
    try std.testing.expectError(error.MissingTensor, plan.get("engram.wkv"));
    try std.testing.expectError(error.DuplicateTensor, plan.add(path, "engram.embed.", null));
}

test "selected table payload must be contiguous and indexed files cannot escape checkpoint directory" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header =
        \\{"engram.embed.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},
        \\ "projection":{"dtype":"U8","shape":[3],"data_offsets":[4,7]},
        \\ "engram.embed.scale":{"dtype":"U8","shape":[2],"data_offsets":[7,9]}}
    ;
    var data: [8 + header.len + 9]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    @memset(data[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "gap.safetensors", .data = &data });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/gap.safetensors", .{tmp.sub_path});
    defer a.free(path);
    var plan = Plan{ .arena = std.heap.ArenaAllocator.init(a) };
    defer plan.deinit();
    try std.testing.expectError(error.NoncontiguousEngramTables, plan.add(path, "engram.embed.", null));
    for ([_][]const u8{ "../x.safetensors", "/x.safetensors", "a\\x.safetensors", ".", "..", "x.safetensors\x00", "x.json", "" }) |name| try std.testing.expect(!safeBasename(name));
    try std.testing.expect(safeBasename("model-00001-of-00039.safetensors"));
}
