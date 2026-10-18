//! Drama 3's checkpoint layout, shared by the backends: the safetensors shards by tensor name and lm_head's candidates.
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const cfg = @import("config.zig");

const st = core.safetensors;

pub const Shards = struct {
    files: []st.File,
    map: std.StringHashMapUnmanaged(usize),
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Shards {
        var s: Shards = .{ .files = &.{}, .map = .empty, .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer s.arena.deinit();
        const a = s.arena.allocator();
        const ipath = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, ipath, a, .limited(64 << 20));
        const tree = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (tree != .object) return error.BadIndex;
        const map = tree.object.get("weight_map") orelse return error.BadIndex;
        if (map != .object) return error.BadIndex;
        const wm = map.object;
        var names: std.ArrayList([]const u8) = .empty;
        var it = wm.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.BadIndex;
            const fname = e.value_ptr.string;
            if (fname.len == 0 or std.fs.path.isAbsolute(fname) or std.mem.indexOf(u8, fname, "..") != null) return error.BadIndex;
            var idx: ?usize = null;
            for (names.items, 0..) |n, i| if (std.mem.eql(u8, n, fname)) {
                idx = i;
            };
            if (idx == null) {
                try names.append(a, fname);
                idx = names.items.len - 1;
            }
            try s.map.put(a, e.key_ptr.*, idx.?);
        }
        s.files = try a.alloc(st.File, names.items.len);
        var opened: usize = 0;
        errdefer for (s.files[0..opened]) |*f| f.close(io);
        for (names.items, 0..) |n, i| {
            const p = try std.fs.path.join(a, &.{ dir, n });
            s.files[i] = try st.File.open(gpa, io, p);
            opened += 1;
        }
        return s;
    }

    pub fn close(s: *Shards, io: std.Io) void {
        for (s.files) |*f| f.close(io);
        s.arena.deinit();
    }

    /// The shard holding `name` (its map is the whole file), and the tensor's byte offset in that file.
    pub fn locate(s: *const Shards, name: []const u8) !struct { file: usize, offset: usize } {
        const i = s.map.get(name) orelse return error.MissingTensor;
        const e = s.files[i].names.get(name) orelse return error.MissingTensor;
        return .{ .file = i, .offset = s.files[i].data + e.begin };
    }

    pub fn get(s: *const Shards, name: []const u8) !st.Tensor {
        const i = s.map.get(name) orelse {
            std.log.err("drama3: checkpoint has no {s}", .{name});
            return error.MissingTensor;
        };
        return s.files[i].get(name) orelse error.MissingTensor;
    }

    /// Drop a tensor's pages from the page cache once it is on the device (the shards stream from NFS).
    pub fn release(s: *const Shards, name: []const u8) void {
        const i = s.map.get(name) orelse return;
        const f = &s.files[i];
        const e = f.names.get(name) orelse return;
        if (builtin.os.tag == .linux) _ = posix_fadvise(f.file.handle, @intCast(f.data + e.begin), @intCast(e.end - e.begin), 4);
    }
};

extern "c" fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int;

/// The candidate token ids the frame step scores: eos, then the semantic band (sorted, as the sampler's list).
pub fn candidateIds(gpa: std.mem.Allocator, c: cfg.Config) ![]i32 {
    const n = c.candidates();
    const ids = try gpa.alloc(i32, n);
    var j: usize = 0;
    if (c.ids.eos < c.ids.sem_begin) {
        ids[0] = c.ids.eos;
        j = 1;
    }
    var tok = c.ids.sem_begin;
    while (tok <= c.ids.sem_end) : (tok += 1) {
        ids[j] = tok;
        j += 1;
    }
    if (c.ids.eos > c.ids.sem_end) ids[n - 1] = c.ids.eos;
    return ids;
}
