//! A model folder's safetensors as one name space (every backend's index): shard files, tensors mapped, leftovers refused.

const std = @import("std");
const Io = std.Io;
const st = @import("safetensors.zig");

pub const Tensor = st.Tensor;
pub const DType = st.DType;

/// The checkpoint's shard paths in name order: model.safetensors.index.json's weight map, else every model*.safetensors.
pub fn shardFiles(gpa: std.mem.Allocator, io: Io, dir: []const u8) ![][:0]u8 {
    var names: std.ArrayList([:0]u8) = .empty;
    errdefer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(index);
    if (Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26))) |text| {
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(gpa);
        var it = parsed.value.object.get("weight_map").?.object.iterator();
        while (it.next()) |e| {
            const file = e.value_ptr.string;
            if ((try seen.getOrPut(gpa, file)).found_existing) continue;
            try names.append(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, file }, 0));
        }
    } else |_| {
        var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| {
            if (e.kind != .file and e.kind != .sym_link) continue;
            if (std.mem.startsWith(u8, e.name, "model") and std.mem.endsWith(u8, e.name, ".safetensors"))
                try names.append(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, e.name }, 0));
        }
    }
    if (names.items.len == 0) return error.NoSafetensors;
    std.mem.sort([:0]u8, names.items, {}, struct {
        fn lt(_: void, a: [:0]u8, b: [:0]u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);
    return names.toOwnedSlice(gpa);
}

pub fn freeShardFiles(gpa: std.mem.Allocator, files: [][:0]u8) void {
    for (files) |f| gpa.free(f);
    gpa.free(files);
}

pub const Checkpoint = struct {
    gpa: std.mem.Allocator,
    io: Io,
    files: std.ArrayList(st.File) = .empty,
    used: std.StringHashMapUnmanaged(void) = .empty,

    /// Every shard of `dir`, mapped.
    pub fn openModel(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Checkpoint {
        return openModelPrefix(gpa, io, dir, null);
    }
    pub fn openModelPrefix(gpa: std.mem.Allocator, io: Io, dir: []const u8, prefix: ?[]const u8) !Checkpoint {
        const paths = try shardFiles(gpa, io, dir);
        defer freeShardFiles(gpa, paths);
        var ck: Checkpoint = .{ .gpa = gpa, .io = io };
        errdefer ck.close();
        for (paths) |p| try ck.files.append(gpa, try st.File.openPrefix(gpa, io, p, prefix));
        return ck;
    }

    /// One more file (an MTP head beside the model's files).
    pub fn add(self: *Checkpoint, dir: []const u8, name: []const u8) !void {
        const path = try std.fs.path.join(self.gpa, &.{ dir, name });
        defer self.gpa.free(path);
        try self.files.append(self.gpa, try st.File.open(self.gpa, self.io, path));
    }

    pub fn close(self: *Checkpoint) void {
        for (self.files.items) |*f| f.close(self.io);
        self.files.deinit(self.gpa);
        self.used.deinit(self.gpa);
        self.* = undefined;
    }

    /// The tensor named `name`, marked as used; a missing name is an error naming it.
    pub fn get(self: *Checkpoint, name: []const u8) !Tensor {
        for (self.files.items) |*f| if (f.get(name)) |t| {
            try self.used.put(self.gpa, f.names.getKey(name).?, {});
            return t;
        };
        std.log.err("checkpoint has no tensor {s}", .{name});
        return error.MissingTensor;
    }

    /// `get` with the dtype and shape checked.
    pub fn expect(self: *Checkpoint, name: []const u8, dtype: DType, shape: []const usize) !Tensor {
        const t = try self.get(name);
        if (!t.is(dtype, shape)) {
            std.log.err("{s}: {t} {any}, expected {t} {any}", .{ name, t.dtype, t.shape[0..t.rank], dtype, shape });
            return error.UnexpectedTensor;
        }
        return t;
    }

    /// Names no `get` took (the loader refuses a checkpoint with leftovers).
    pub fn unused(self: *const Checkpoint) usize {
        var n: usize = 0;
        for (self.files.items) |*f| for (f.names.keys()) |k| {
            if (self.used.contains(k)) continue;
            if (n < 5) std.log.err("unused checkpoint tensor {s}", .{k});
            n += 1;
        };
        return n;
    }
};
