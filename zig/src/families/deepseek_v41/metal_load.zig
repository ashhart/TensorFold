//! Header-first resident loading: no mapped weights, deferred disk lookup or duplicate Engram projections.
const std = @import("std");
const mtl = @import("metal");
const st = @import("../../core/safetensors.zig");
const cp = @import("../../core/checkpoint_metal.zig");
const paced = @import("../../core/paced_read.zig");
const rows = @import("engram_safetensors.zig");

pub const Reference = struct { shard: usize, entry: st.Entry };
pub const Shard = struct {
    path: [:0]const u8,
    file_bytes: usize,
    data_start: usize,
    begin: usize,
    end: usize,
    names: st.Header,
};

/// Where the two Engram tables live: wired beside the backbone, or left in their shards and read per chunk.
pub const Engram = enum { resident, streamed };
pub const Options = struct {
    engram: Engram = .resident,
    /// Stage only tensors the backbone binds (embed, norm, head, layers.*): MTP, vision and image markers stay on disk.
    bound_only: bool = false,
};

pub const tables = [_]struct { layer: u32, file: []const u8, rows: usize }{
    .{ .layer = 1, .file = "model-00047-of-00048.safetensors", .rows = 384006168 },
    .{ .layer = 14, .file = "model-00048-of-00048.safetensors", .rows = 384016682 },
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    shards: std.ArrayList(Shard) = .empty,
    tensors: std.StringHashMapUnmanaged(Reference) = .empty,
    bytes: usize = 0,
    /// Set in streamed mode: each table's shard and offsets, checked like a resident table but never staged.
    engram_files: ?[2]rows.File = null,
    bound_only: bool = false,
    /// Indexed tensors left out by `bound_only`.
    skipped: usize = 0,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, table_dir: []const u8) !Plan {
        return openWith(gpa, io, dir, table_dir, .{});
    }

    pub fn openWith(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, table_dir: []const u8, options: Options) !Plan {
        var self = Plan{ .arena = std.heap.ArenaAllocator.init(gpa), .bound_only = options.bound_only };
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
        if (self.tensors.count() + self.skipped != index.weight_map.object.count()) return error.CheckpointIndexMismatch;
        entries = index.weight_map.object.iterator();
        while (entries.next()) |item| {
            if (self.bound_only and !bound(item.key_ptr.*)) continue;
            const ref = self.tensors.get(item.key_ptr.*) orelse return error.CheckpointIndexMismatch;
            if (!std.mem.eql(u8, std.fs.path.basename(self.shards.items[ref.shard].path), item.value_ptr.string))
                return error.CheckpointIndexMismatch;
        }
        if (options.engram == .streamed) {
            var streamed: [2]rows.File = undefined;
            for (tables, &streamed) |table, *file| file.* = try self.tableFile(try std.fs.path.join(a, &.{ table_dir, table.file }), table.layer, table.rows);
            self.engram_files = streamed;
            return self;
        }
        for (tables) |table| {
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

    const Header = struct { path: [:0]const u8, file_bytes: usize, data_start: usize, names: st.Header };

    fn header(self: *Plan, path: []const u8, prefix: ?[]const u8) !Header {
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
        return .{ .path = own_path, .file_bytes = file_bytes, .data_start = data_start, .names = names };
    }

    /// A streamed table: the same dtype, shape and two-tensor checks as a resident one, nothing staged or counted.
    fn tableFile(self: *Plan, path: []const u8, layer: u32, row_count: usize) !rows.File {
        var buf: [64]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.", .{layer});
        const h = try self.header(path, prefix);
        if (h.names.count() != 2) return error.UnexpectedEngramTensor;
        var wbuf: [80]u8 = undefined;
        var sbuf: [80]u8 = undefined;
        const w = h.names.get(try std.fmt.bufPrint(&wbuf, "{s}weight", .{prefix})) orelse return error.MissingTensor;
        const s = h.names.get(try std.fmt.bufPrint(&sbuf, "{s}scale", .{prefix})) orelse return error.MissingTensor;
        if (w.dtype != .f8_e4m3 or w.rank != 2 or w.shape[0] != row_count or w.shape[1] != 256 or w.end - w.begin != row_count * 256 or
            s.dtype != .f8_e8m0 or s.rank != 2 or s.shape[0] != row_count or s.shape[1] != 8 or s.end - s.begin != row_count * 8)
            return error.UnsupportedTensor;
        return .{ .path = h.path, .file_bytes = h.file_bytes, .data_start = h.data_start, .weight_begin = w.begin, .scale_begin = s.begin, .rows = row_count };
    }

    fn add(self: *Plan, path: []const u8, prefix: ?[]const u8, owned: ?*const std.StringHashMapUnmanaged([]const u8)) !void {
        const a = self.arena.allocator();
        const h = try self.header(path, prefix);
        const own_path = h.path;
        const file_bytes = h.file_bytes;
        const data_start = h.data_start;
        const names = h.names;
        const selective = prefix == null and self.bound_only;
        var begin: usize = if (prefix == null and !selective) 0 else file_bytes - data_start;
        var end: usize = if (prefix == null and !selective) file_bytes - data_start else 0;
        for (names.keys(), names.values()) |name, entry| {
            // A rank-split pack duplicates whole-load tensors beside the shards; the index decides the owner file.
            if (owned) |map| {
                const owner = map.get(name) orelse continue;
                if (!std.mem.eql(u8, owner, std.fs.path.basename(path))) continue;
            }
            if (selective and !bound(name)) {
                self.skipped += 1;
                continue;
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
        // A shard holding only unbound tensors is not staged at all.
        if (selective and end <= begin) return;
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
                var short: paced.Memory = undefined;
                try paced.read(fd, buffer.contents()[copied..][0..count], shard.data_start + shard.begin + copied, stage, &short);
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

/// The tensors weights.bind reads; everything else in a release (mtp.*, vision.*, aligner.*, image markers) is unbound.
pub fn bound(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "layers.") or std.mem.startsWith(u8, name, "head.") or
        std.mem.eql(u8, name, "embed.weight") or std.mem.eql(u8, name, "norm.weight");
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

test "a streamed table is checked like a resident one but stays out of the staged bytes" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header =
        \\{"layers.1.engram.embed.weight":{"dtype":"F8_E4M3","shape":[3,256],"data_offsets":[0,768]},
        \\ "layers.1.engram.embed.scale":{"dtype":"F8_E8M0","shape":[3,8],"data_offsets":[768,792]}}
    ;
    var data: [8 + header.len + 792]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    @memset(data[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "t.safetensors", .data = &data });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/t.safetensors", .{tmp.sub_path});
    defer a.free(path);
    var plan = Plan{ .arena = std.heap.ArenaAllocator.init(a) };
    defer plan.deinit();
    const f = try plan.tableFile(path, 1, 3);
    try std.testing.expectEqual(@as(usize, 8 + header.len), f.data_start);
    try std.testing.expectEqual(@as(usize, 768), f.scale_begin);
    try std.testing.expectEqual(@as(usize, 0), plan.bytes);
    try std.testing.expectEqual(@as(usize, 0), plan.shards.items.len);
    try std.testing.expectError(error.UnsupportedTensor, plan.tableFile(path, 1, 4));
    try std.testing.expectError(error.MissingTensor, plan.tableFile(path, 14, 3));
}

test "bound-only staging leaves MTP, vision and image markers on disk" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header =
        \\{"mtp.0.x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},
        \\ "layers.0.a":{"dtype":"U8","shape":[2],"data_offsets":[4,6]},
        \\ "embed.weight":{"dtype":"U8","shape":[2],"data_offsets":[6,8]},
        \\ "vision.y":{"dtype":"U8","shape":[3],"data_offsets":[8,11]}}
    ;
    var data: [8 + header.len + 11]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    @memset(data[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "s.safetensors", .data = &data });
    const only_mtp =
        \\{"mtp.1.x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}
    ;
    var data2: [8 + only_mtp.len + 4]u8 = undefined;
    std.mem.writeInt(u64, data2[0..8], only_mtp.len, .little);
    @memcpy(data2[8..][0..only_mtp.len], only_mtp);
    @memset(data2[8 + only_mtp.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = &data2 });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/s.safetensors", .{tmp.sub_path});
    defer a.free(path);
    const path2 = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/m.safetensors", .{tmp.sub_path});
    defer a.free(path2);
    var plan = Plan{ .arena = std.heap.ArenaAllocator.init(a), .bound_only = true };
    defer plan.deinit();
    try plan.add(path, null, null);
    try plan.add(path2, null, null);
    try std.testing.expectEqual(@as(usize, 4), plan.bytes);
    try std.testing.expectEqual(@as(usize, 1), plan.shards.items.len);
    try std.testing.expectEqual(@as(usize, 4), plan.shards.items[0].begin);
    try std.testing.expectEqual(@as(usize, 3), plan.skipped);
    try std.testing.expect(!plan.has("mtp.0.x") and !plan.has("vision.y") and plan.has("layers.0.a"));
    for ([_][]const u8{ "aligner.w", "image_start", "mtp.0.e", "vision.blocks.0" }) |n| try std.testing.expect(!bound(n));
    for ([_][]const u8{ "head.trellis", "embed.weight", "norm.weight", "layers.39.ffn.gate.weight" }) |n| try std.testing.expect(bound(n));
}
