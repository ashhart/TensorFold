//! Ling's bf16 checkpoint tensors, validated by the shared safetensors reader and mapped without weight copies.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");
const config = @import("config.zig");

pub const Tensor = struct {
    buffer: mtl.Buffer,
    offset: usize,
    entry: st.Entry,
};
const File = struct {
    path: [:0]const u8,
    map: ?mtl.MappedFile = null,
    buffer: ?mtl.Buffer = null,
    header: st.Header = .empty,
    data: usize = 0,
};
pub const Checkpoint = struct {
    arena: std.heap.ArenaAllocator,
    device: mtl.Device,
    config: ?config.Config = null,
    files: std.ArrayList(File) = .empty,
    where: std.StringHashMapUnmanaged(u32) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Checkpoint {
        var self = Checkpoint{ .arena = std.heap.ArenaAllocator.init(gpa), .device = device };
        errdefer self.deinit();
        const a = self.arena.allocator();
        const config_path = try std.fs.path.join(a, &.{ dir, "config.json" });
        const config_text = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(16 << 20));
        var why: config.Why = .{};
        self.config = try config.parse(a, config_text, &why);
        const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(st.index.max_bytes));
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const index = try st.index.weightMap(parsed);
        var file_of: std.StringHashMapUnmanaged(u32) = .empty;
        for (index.keys(), index.values()) |name, value| {
            const file = value.string;
            if (std.fs.path.isAbsolute(file) or std.mem.indexOf(u8, file, "..") != null) return error.BadShardPath;
            const slot = try file_of.getOrPut(a, file);
            if (!slot.found_existing) {
                slot.value_ptr.* = @intCast(self.files.items.len);
                try self.files.append(a, .{ .path = try std.fs.path.joinZ(a, &.{ dir, file }) });
            }
            try self.where.put(a, name, slot.value_ptr.*);
        }
        return self;
    }

    pub fn get(self: *Checkpoint, name: []const u8, dtype: st.DType, shape: []const usize) !Tensor {
        const id = self.where.get(name) orelse return error.MissingTensor;
        const f = &self.files.items[id];
        if (f.buffer == null) {
            const map = try mtl.MappedFile.open(f.path);
            errdefer map.deinit();
            if (map.size < 8) return error.BadSafetensors;
            const n = try st.headerSize(std.mem.readInt(u64, map.bytes[0..8], .little), map.size);
            f.data = 8 + n;
            f.header = try st.parseHeader(self.arena.allocator(), map.bytes[8..f.data], map.size - f.data);
            for (f.header.values()) |entry| {
                if (entry.dtype != .bf16 and entry.dtype != .f32) return error.UnsupportedDType;
                if ((f.data + entry.begin) % entry.dtype.size() != 0) return error.MisalignedTensor;
            }
            f.buffer = try self.device.bufferNoCopy(map.bytes.ptr, map.bytes.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            f.map = map;
        }
        const entry = f.header.get(name) orelse return error.MissingTensor;
        if (entry.dtype != dtype or entry.rank != shape.len or !std.mem.eql(usize, entry.shape[0..entry.rank], shape)) return error.TensorShape;
        return .{ .buffer = f.buffer.?, .offset = f.data + entry.begin, .entry = entry };
    }

    pub fn deinit(self: *Checkpoint) void {
        for (self.files.items) |file| {
            if (file.buffer) |b| b.deinit();
            if (file.map) |m| m.deinit();
        }
        self.arena.deinit();
    }
};

test {
    std.testing.refAllDecls(Checkpoint);
}
