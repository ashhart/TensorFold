const std = @import("std");
const st = @import("safetensors");
const torch = @import("torch_checkpoint");
const Config = @import("config.zig").Config;

pub const Store = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    source: union(enum) { safetensors: st.File, pytorch: torch.Archive },
    cached: std.StringHashMapUnmanaged(st.Tensor) = .empty,

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
        while (cached.next()) |tensor| self.allocator.free(tensor.bytes);
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
                if (self.cached.get(name)) |tensor| return tensor;
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
                errdefer self.allocator.free(bytes);
                tensor.bytes = bytes;
                const key = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(key);
                try self.cached.put(self.allocator, key, tensor);
                return tensor;
            },
        }
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
