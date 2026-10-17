//! A rank's named weights on the GPU: one device buffer a tensor (bound by name), digested on the way up.

const std = @import("std");
const cuda = @import("cuda");
const named = @import("named.zig");
const ex_stacks = @import("ex_stacks.zig");

pub const Tensor = struct {
    ptr: u64,
    len: usize,
    kind: named.Kind,
    sha256: [32]u8,
    shape: [4]usize = @splat(1),
    rank: u8 = 0,

    pub fn dims(t: *const Tensor) []const usize {
        return t.shape[0..t.rank];
    }
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    map: std.StringHashMapUnmanaged(Tensor) = .empty,
    bytes: usize = 0,
    /// each tensor's SHA-256 taken on the way up (the weights check reads it); off, `sha256` is zeros (served boot)
    digest: bool = true,

    pub fn deinit(w: *Weights) void {
        for (w.bufs.items) |*b| b.free();
        w.bufs.deinit(w.gpa);
        var it = w.map.keyIterator();
        while (it.next()) |k| w.gpa.free(k.*);
        w.map.deinit(w.gpa);
        w.* = undefined;
    }

    /// Uploads every tensor `b` holds (its host images can be freed afterwards); a name already on the GPU is refused.
    pub fn upload(w: *Weights, d: *const cuda.Driver, b: *const named.Builder) !void {
        // TF_DSV41_EX_STACKS: a routed suh / svh and its shared expert's in one buffer (ex_stacks.zig)
        var paired: std.AutoHashMapUnmanaged(usize, void) = .empty;
        defer paired.deinit(w.gpa);
        if (ex_stacks.enabled()) {
            const names = try w.gpa.alloc([]const u8, b.out.items.len);
            defer w.gpa.free(names);
            const lens = try w.gpa.alloc(usize, b.out.items.len);
            defer w.gpa.free(lens);
            for (b.out.items, names, lens) |n, *x, *l| {
                x.* = n.name;
                l.* = n.bytes.len;
            }
            const ps = try ex_stacks.pairs(w.gpa, names, lens);
            defer w.gpa.free(ps);
            for (ps) |p| {
                const r = b.out.items[p[0]];
                const s = b.out.items[p[1]];
                if (w.map.contains(r.name) or w.map.contains(s.name)) return error.DuplicateWeight;
                var buf = try cuda.DeviceBuffer.alloc(d, r.bytes.len + s.bytes.len);
                errdefer buf.free();
                try cuda.DeviceBuffer.upload(buf, 0, r.bytes);
                try cuda.DeviceBuffer.upload(buf, r.bytes.len, s.bytes);
                try w.bufs.append(w.gpa, buf);
                try w.put(r, buf.ptr);
                try w.put(s, buf.ptr + r.bytes.len);
                try paired.put(w.gpa, p[0], {});
                try paired.put(w.gpa, p[1], {});
            }
        }
        for (b.out.items, 0..) |n, i| {
            if (paired.contains(i)) continue;
            if (w.map.contains(n.name)) return error.DuplicateWeight;
            var sha: [32]u8 = @splat(0);
            if (w.digest) std.crypto.hash.sha2.Sha256.hash(n.bytes, &sha, .{});
            var buf = try cuda.DeviceBuffer.fromHost(d, n.bytes);
            errdefer buf.free();
            try w.bufs.append(w.gpa, buf);
            try w.map.put(w.gpa, try w.gpa.dupe(u8, n.name), .{ .ptr = buf.ptr, .len = n.bytes.len, .kind = n.kind, .sha256 = sha, .shape = n.shape, .rank = n.rank });
            w.bytes += n.bytes.len;
        }
        // pageable uploads can still be in flight when cuMemcpyHtoD returns; the weights are whole before any launch
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    fn put(w: *Weights, n: named.Named, ptr: u64) !void {
        var sha: [32]u8 = @splat(0);
        if (w.digest) std.crypto.hash.sha2.Sha256.hash(n.bytes, &sha, .{});
        try w.map.put(w.gpa, try w.gpa.dupe(u8, n.name), .{ .ptr = ptr, .len = n.bytes.len, .kind = n.kind, .sha256 = sha, .shape = n.shape, .rank = n.rank });
        w.bytes += n.bytes.len;
    }

    pub fn get(w: *const Weights, name: []const u8) ?Tensor {
        return w.map.get(name);
    }
};
