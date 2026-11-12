const std = @import("std");
const st = @import("safetensors");
const torch = @import("torch_checkpoint");
const Config = @import("config.zig").Config;
const page_bytes = std.heap.page_size_max;
const Cached = struct { tensor: st.Tensor, storage: []align(page_bytes) u8 };

pub const Store = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    source: union(enum) { safetensors: st.File, pytorch: torch.Archive },
    cached: std.StringHashMapUnmanaged(Cached) = .empty,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Store {
        if (std.mem.endsWith(u8, path, ".safetensors"))
            return .{ .allocator = allocator, .io = io, .source = .{
                .safetensors = try st.File.open(allocator, io, path),
            } };
        if (std.mem.endsWith(u8, path, ".pt") or std.mem.endsWith(u8, path, ".pth"))
            return .{ .allocator = allocator, .io = io, .source = .{
                .pytorch = try torch.Archive.open(allocator, io, path),
            } };
        return error.UnsupportedCheckpointFormat;
    }

    pub fn deinit(self: *Store) void {
        var cached = self.cached.valueIterator();
        while (cached.next()) |entry| self.allocator.free(entry.storage);
        var keys = self.cached.keyIterator();
        while (keys.next()) |key| self.allocator.free(key.*);
        self.cached.deinit(self.allocator);
        switch (self.source) {
            .safetensors => |*file| file.close(self.io),
            .pytorch => |*archive| archive.close(),
        }
        self.* = undefined;
    }

    pub fn has(self: *const Store, name: []const u8) bool {
        return switch (self.source) {
            .safetensors => |file| file.names.contains(name),
            .pytorch => |archive| archive.tensors.contains(name),
        };
    }

    pub fn count(self: *const Store) usize {
        return switch (self.source) {
            .safetensors => |file| file.names.count(),
            .pytorch => |archive| archive.tensors.count(),
        };
    }

    pub fn get(self: *Store, name: []const u8) !st.Tensor {
        switch (self.source) {
            .safetensors => |file| return file.get(name) orelse error.MissingTensor,
            .pytorch => |archive| {
                if (self.cached.get(name)) |entry| return entry.tensor;
                const info = try archive.tensor(name);
                if (info.shape.len > st.max_rank) return error.RankTooHigh;
                var tensor = st.Tensor{
                    .dtype = switch (info.dtype) {
                        .f32 => .f32,
                        .f16 => .f16,
                        .bf16 => .bf16,
                        else => return error.UnsupportedWeightDType,
                    },
                    .rank = @intCast(info.shape.len),
                    .shape = @splat(1),
                    .bytes = undefined,
                };
                for (info.shape, 0..) |size, i|
                    tensor.shape[i] = std.math.cast(usize, size) orelse return error.InvalidTensorShape;
                const bytes = try archive.readBytes(self.allocator, name);
                defer self.allocator.free(bytes);
                tensor.bytes = bytes;
                return self.cacheTensor(name, tensor);
            },
        }
    }

    fn cacheTensor(self: *Store, name: []const u8, source: st.Tensor) !st.Tensor {
        const size = std.mem.alignBackward(usize, try std.math.add(usize, source.bytes.len, page_bytes - 1), page_bytes);
        // Metal may expose complete pages, so cached payloads own their zero-padded pages.
        const storage = try self.allocator.alignedAlloc(u8, .fromByteUnits(page_bytes), @max(page_bytes, size));
        errdefer self.allocator.free(storage);
        @memcpy(storage[0..source.bytes.len], source.bytes);
        @memset(storage[source.bytes.len..], 0);
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        var tensor = source;
        tensor.bytes = storage[0..source.bytes.len];
        try self.cached.putNoClobber(self.allocator, key, .{ .tensor = tensor, .storage = storage });
        return tensor;
    }

    pub fn rotaryPairs(self: *Store, name: []const u8, positions: usize, width: usize) !?st.Tensor {
        if (!self.has(name)) return null;
        const archive = switch (self.source) {
            .pytorch => |*archive| archive,
            .safetensors => return error.UnsupportedSamRotaryDType,
        };
        const info = try archive.tensor(name);
        _ = try rotaryBytes(info, positions, width);
        const key = try std.fmt.allocPrint(self.allocator, "rotary-pairs:{s}", .{name});
        defer self.allocator.free(key);
        if (self.cached.get(key)) |entry| {
            const tensor = entry.tensor;
            if (!tensor.is(.f32, &.{ positions, width })) return error.SamRotaryShape;
            return tensor;
        }
        const bytes = try archive.readBytes(self.allocator, name);
        defer self.allocator.free(bytes);
        const tensor = try rotaryView(info, bytes, positions, width);
        return try self.cacheTensor(key, tensor);
    }

    pub fn config(self: *Store) !Config {
        const patch = try self.get("detector.backbone.vision_backbone.trunk.patch_embed.proj.weight");
        const masks = try self.get(if (self.has("tracker.model.sam_mask_decoder.mask_tokens.weight"))
            "tracker.model.sam_mask_decoder.mask_tokens.weight"
        else
            "tracker.sam_mask_decoder.mask_tokens.weight");
        return inferConfig(patch, masks);
    }
};

fn rotaryBytes(info: torch.Tensor, positions: usize, width: usize) !usize {
    if (info.dtype != .c64) return error.UnsupportedSamRotaryDType;
    if (positions == 0 or width == 0 or width % 2 != 0 or info.shape.len != 2 or
        info.shape[0] != positions or info.shape[1] != width / 2) return error.SamRotaryShape;
    const layout = try info.layout();
    if (!layout.contiguous) return error.NonContiguousRotaryTable;
    return std.math.mul(usize, layout.count, 8);
}

fn rotaryView(info: torch.Tensor, bytes: []const u8, positions: usize, width: usize) !st.Tensor {
    if (bytes.len != try rotaryBytes(info, positions, width)) return error.SamRotaryByteCount;
    for (0..bytes.len / 4) |index| {
        const value: f32 = @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
        if (!std.math.isFinite(value)) return error.NonFiniteSamRotaryTable;
    }
    var tensor = st.Tensor{ .dtype = .f32, .rank = 2, .shape = @splat(1), .bytes = bytes };
    tensor.shape[0] = positions;
    tensor.shape[1] = width;
    return tensor;
}

fn inferConfig(patch: st.Tensor, masks: st.Tensor) !Config {
    if (!floating(patch) or patch.rank != 4 or
        !std.mem.eql(usize, patch.shape[0..4], &.{ 1024, 3, 14, 14 })) return error.UnsupportedSamBackbone;
    if (!floating(masks) or masks.rank != 2 or masks.dim(1) != 256) return error.UnsupportedSamTracker;
    return Config.of(switch (masks.dim(0)) {
        4 => .sam3,
        48 => .sam31,
        else => return error.UnsupportedSamTracker,
    });
}

fn floating(tensor: st.Tensor) bool {
    return tensor.dtype == .f32 or tensor.dtype == .f16 or tensor.dtype == .bf16;
}

test "SAM checkpoint geometry refuses foreign backbones before device allocation" {
    var patch = st.Tensor{ .dtype = .f32, .rank = 4, .shape = .{ 1024, 3, 14, 14, 1 }, .bytes = &.{} };
    var masks = st.Tensor{ .dtype = .f16, .rank = 2, .shape = .{ 48, 256, 1, 1, 1 }, .bytes = &.{} };
    try std.testing.expectEqual(.sam31, (try inferConfig(patch, masks)).version);
    masks.shape[0] = 4;
    try std.testing.expectEqual(.sam3, (try inferConfig(patch, masks)).version);
    masks.shape[0] = 32;
    try std.testing.expectError(error.UnsupportedSamTracker, inferConfig(patch, masks));
    masks.shape[0] = 48;
    masks.dtype = .u8;
    try std.testing.expectError(error.UnsupportedSamTracker, inferConfig(patch, masks));
    patch.shape[0] = 768;
    try std.testing.expectError(error.UnsupportedSamBackbone, inferConfig(patch, masks));
}

test "SAM stored complex rotary pairs retain every real and imaginary bit" {
    var bytes: [32]u8 = undefined;
    const words = [_]u32{ 0x3f800000, 0x80000000, 1, 0x80000001, 0xbf800000, 0x3f000000, 0, 0x3f800000 };
    for (words, 0..) |word, index| std.mem.writeInt(u32, bytes[index * 4 ..][0..4], word, .little);
    const info = torch.Tensor{ .dtype = .c64, .storage = "0", .storage_numel = 4, .offset = 0, .shape = &.{ 2, 2 }, .stride = &.{ 2, 1 } };
    const pairs = try rotaryView(info, &bytes, 2, 4);
    try std.testing.expect(pairs.is(.f32, &.{ 2, 4 }));
    try std.testing.expectEqualSlices(u8, &bytes, pairs.bytes);
    var invalid = info;
    invalid.dtype = .f32;
    try std.testing.expectError(error.UnsupportedSamRotaryDType, rotaryView(invalid, &bytes, 2, 4));
    try std.testing.expectError(error.SamRotaryShape, rotaryView(info, &bytes, 2, 3));
    try std.testing.expectError(error.SamRotaryShape, rotaryView(info, &bytes, 3, 4));
    try std.testing.expectError(error.SamRotaryByteCount, rotaryView(info, bytes[0..31], 2, 4));
    invalid = info;
    invalid.stride = &.{ 1, 2 };
    try std.testing.expectError(error.NonContiguousRotaryTable, rotaryView(invalid, &bytes, 2, 4));
    invalid = info;
    invalid.offset = 1;
    try std.testing.expectError(error.BadArchive, rotaryView(invalid, &bytes, 2, 4));
    std.mem.writeInt(u32, bytes[0..4], 0x7fc00000, .little);
    try std.testing.expectError(error.NonFiniteSamRotaryTable, rotaryView(info, &bytes, 2, 4));
}

fn exerciseRotaryCache(a: std.mem.Allocator) !void {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var payload: [62]u8 = @splat(0);
    std.mem.writeInt(u32, payload[0..4], 0x04034b50, .little);
    const words = [_]u32{ 0x3f800000, 0x80000000, 1, 0x80000001, 0xbf800000, 0x3f000000, 0, 0x3f800000 };
    for (words, 0..) |word, index| std.mem.writeInt(u32, payload[30 + index * 4 ..][0..4], word, .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "weights.pt", .data = &payload });
    var store = Store{ .allocator = a, .io = io, .source = .{ .pytorch = .{ .arena = .init(a), .io = io, .file = try tmp.dir.openFile(io, "weights.pt", .{}), .members = .empty, .tensors = .empty, .prefix = "weights/" } } };
    defer store.deinit();
    const keep = store.source.pytorch.arena.allocator();
    try store.source.pytorch.members.put(keep, "weights/data/0", .{ .local = 0, .size = 32, .crc = std.hash.Crc32.hash(payload[30..]) });
    try store.source.pytorch.tensors.put(keep, "freqs", .{ .dtype = .c64, .storage = "0", .storage_numel = 4, .offset = 0, .shape = &.{ 2, 2 }, .stride = &.{ 2, 1 } });
    const first = (try store.rotaryPairs("freqs", 2, 4)).?;
    const second = (try store.rotaryPairs("freqs", 2, 4)).?;
    try std.testing.expectEqualSlices(u8, payload[30..], first.bytes);
    try std.testing.expect(first.bytes.ptr == second.bytes.ptr);
    const owned = store.cached.get("rotary-pairs:freqs").?;
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(first.bytes.ptr) % page_bytes);
    try std.testing.expect(owned.storage.len >= page_bytes and owned.storage.len % page_bytes == 0);
    try std.testing.expect(std.mem.allEqual(u8, owned.storage[first.bytes.len..], 0));
    try std.testing.expectEqual(@as(usize, 1), store.cached.count());
    try std.testing.expectError(error.UnsupportedWeightDType, store.get("freqs"));
    try std.testing.expectError(error.SamRotaryShape, store.rotaryPairs("freqs", 3, 4));
    try std.testing.expectEqual(null, try store.rotaryPairs("missing", 2, 4));
}

test "SAM rotary cache owns its pages without reinterpreting generic weights" {
    try exerciseRotaryCache(std.testing.allocator);
}

test "SAM cached pages release every owner after allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, exerciseRotaryCache, .{});
}
