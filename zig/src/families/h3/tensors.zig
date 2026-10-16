//! Safetensors files mapped once and handed to Metal whole, and a checkpoint's shards searched by tensor name.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");

/// A safetensors file mapped once and handed to Metal whole; a tensor is an offset into that one buffer.
pub const Tensors = struct {
    map: mtl.MappedFile,
    buffer: mtl.Buffer,
    data: usize,
    names: st.Header,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, device: mtl.Device, path: [*:0]const u8) !Tensors {
        const map = try mtl.MappedFile.open(path);
        errdefer map.deinit();
        if (map.size < 8) return error.BadSafetensors;
        const header_len: usize = @intCast(std.mem.readInt(u64, map.bytes[0..8], .little));
        if (header_len > map.size - 8) return error.BadSafetensors;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const names = try st.parseHeader(arena.allocator(), map.bytes[8..][0..header_len], map.size - 8 - header_len);
        const buffer = try device.bufferNoCopy(map.bytes.ptr, map.bytes.len, mtl.ResourceOptions.shared);
        return .{ .map = map, .buffer = buffer, .data = 8 + header_len, .names = names, .arena = arena };
    }

    pub fn deinit(self: *Tensors) void {
        self.buffer.deinit();
        self.arena.deinit();
        self.map.deinit();
    }

    /// The tensor `name` of `dtype` and `count` values: where it starts in the buffer.
    pub fn at(self: *const Tensors, name: []const u8, dtype: st.DType, count: usize) !Ref {
        const e = self.names.get(name) orelse {
            std.debug.print("no tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        const offset = self.data + e.begin;
        if (e.dtype != dtype or e.end - e.begin != count * dtype.size()) {
            std.debug.print("tensor {s}: unexpected type or size\n", .{name});
            return error.BadTensor;
        }
        return .{ .buffer = self.buffer, .offset = offset };
    }

    /// `at` for a tensor a kernel reads in place, which must start on a multiple of its value size.
    pub fn inPlace(self: *const Tensors, name: []const u8, dtype: st.DType, count: usize) !Ref {
        const ref = try self.at(name, dtype, count);
        if (ref.offset % dtype.size() != 0) {
            std.debug.print("tensor {s} is not aligned for the GPU\n", .{name});
            return error.BadTensor;
        }
        return ref;
    }

    pub fn values(self: *const Tensors, comptime T: type, ref: Ref, count: usize) []const T {
        return @as([*]const T, @ptrCast(@alignCast(self.map.bytes.ptr + ref.offset)))[0..count];
    }

    /// The tensor's values copied out, wherever it starts in the file.
    pub fn copy(self: *const Tensors, comptime T: type, gpa: std.mem.Allocator, ref: Ref, count: usize) ![]T {
        const out = try gpa.alloc(T, count);
        @memcpy(std.mem.sliceAsBytes(out), self.map.bytes[ref.offset..][0 .. count * @sizeOf(T)]);
        return out;
    }
};

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };

/// The checkpoint's shards, each mapped once; a tensor is found in whichever holds it.
pub const Checkpoint = struct {
    shards: []Tensors,

    pub fn open(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, count: usize) !Checkpoint {
        const shards = try gpa.alloc(Tensors, count);
        var path: [1024]u8 = undefined;
        for (shards, 1..) |*shard, i| {
            const text = try std.fmt.bufPrint(path[0 .. path.len - 1], "{s}/diffusion_pytorch_model-{d:0>5}-of-{d:0>5}.safetensors", .{ dir, i, count });
            path[text.len] = 0;
            shard.* = try Tensors.open(gpa, device, @ptrCast(text.ptr));
        }
        return .{ .shards = shards };
    }

    pub fn entry(self: *const Checkpoint, name: []const u8) ?st.Entry {
        for (self.shards) |*shard| if (shard.names.get(name)) |e| return e;
        return null;
    }

    /// The bf16 tensor `name` of `count` values, read in place.
    pub fn ref(self: *const Checkpoint, name: []const u8, count: usize) !Ref {
        for (self.shards) |*shard| if (shard.names.get(name) != null) return shard.inPlace(name, .bf16, count);
        std.debug.print("no tensor {s}\n", .{name});
        return error.MissingTensor;
    }
};
