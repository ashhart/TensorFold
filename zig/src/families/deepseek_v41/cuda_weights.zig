//! CUDA device resources adapt the one shared named checkpoint representation.
const std = @import("std");
const cuda = @import("cuda");
const named = @import("named.zig");
const stacks = @import("ex_stacks.zig");
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
    digest: bool = true,
    pair_stacks: ?bool = null,
    driver: ?*const cuda.Driver = null,
    source: cuda.copy_source.Stage = .{},
    dirty: bool = false,
    poisoned: bool = false,
    fn finish(w: *Weights) !void {
        if (w.dirty) try w.driver.?.synchronize();
        w.dirty = false;
        try w.source.close();
    }
    /// Failure leaves every owner intact. The caller must retain Weights and Driver and retry close after recovery.
    pub fn close(w: *Weights) !void {
        try w.finish();
        for (w.bufs.items) |*buf| buf.free();
        w.bufs.deinit(w.gpa);
        var keys = w.map.keyIterator();
        while (keys.next()) |key| w.gpa.free(key.*);
        w.map.deinit(w.gpa);
        w.* = undefined;
    }
    /// A failure closes admission. Host copy sources and device buffers remain owned until a successful drain.
    pub fn upload(w: *Weights, d: *const cuda.Driver, b: *const named.Builder) !void {
        if (w.poisoned) return error.UploadPoisoned;
        if (w.driver != null and w.driver.? != d) return error.WrongDriver;
        w.driver = d;
        errdefer {
            w.poisoned = true;
            w.finish() catch {};
        }
        var paired: std.AutoHashMapUnmanaged(usize, void) = .empty;
        defer paired.deinit(w.gpa);
        try paired.ensureTotalCapacity(w.gpa, @intCast(b.out.items.len));
        if (w.pair_stacks orelse stacks.enabled()) {
            const names = try w.gpa.alloc([]const u8, b.out.items.len);
            defer w.gpa.free(names);
            const lengths = try w.gpa.alloc(usize, b.out.items.len);
            defer w.gpa.free(lengths);
            for (b.out.items, names, lengths) |n, *name, *len| {
                name.* = n.name;
                len.* = n.bytes.len;
            }
            const pairs = try stacks.pairs(w.gpa, names, lengths);
            defer w.gpa.free(pairs);
            for (pairs) |pair| {
                if (paired.contains(pair[0]) or paired.contains(pair[1])) return error.DuplicateWeight;
                try w.transfer(d, &.{ b.out.items[pair[0]], b.out.items[pair[1]] });
                paired.putAssumeCapacity(pair[0], {});
                paired.putAssumeCapacity(pair[1], {});
            }
        }
        for (b.out.items, 0..) |n, i| if (!paired.contains(i)) {
            try w.transfer(d, &.{n});
        };
        try w.finish();
    }
    fn transfer(w: *Weights, d: *const cuda.Driver, items: []const named.Named) !void {
        var total: usize = 0;
        for (items, 0..) |n, i| {
            if (w.map.contains(n.name)) return error.DuplicateWeight;
            for (items[0..i]) |old| if (std.mem.eql(u8, old.name, n.name)) return error.DuplicateWeight;
            total = try std.math.add(usize, total, n.bytes.len);
        }
        const held = try std.math.add(usize, w.bytes, total);
        try w.map.ensureUnusedCapacity(w.gpa, @intCast(items.len));
        try w.bufs.ensureUnusedCapacity(w.gpa, 1);
        var keys: [2][]u8 = undefined;
        var made: usize = 0;
        var published = false;
        errdefer if (!published) {
            for (keys[0..made]) |key| w.gpa.free(key);
        };
        for (items, 0..) |n, i| {
            keys[i] = try w.gpa.dupe(u8, n.name);
            made += 1;
        }
        const copy = try w.source.take(d, total);
        const buf = try cuda.DeviceBuffer.alloc(d, total);
        w.bufs.appendAssumeCapacity(buf);
        var offset: usize = 0;
        for (items, 0..) |n, i| {
            @memcpy(copy[offset..][0..n.bytes.len], n.bytes);
            var sha: [32]u8 = @splat(0);
            if (w.digest) std.crypto.hash.sha2.Sha256.hash(n.bytes, &sha, .{});
            w.map.putAssumeCapacity(keys[i], .{ .ptr = buf.ptr + offset, .len = n.bytes.len, .kind = n.kind, .sha256 = sha, .shape = n.shape, .rank = n.rank });
            offset += n.bytes.len;
        }
        w.bytes = held;
        published = true;
        w.dirty = true;
        try w.source.copyTo(buf, 0, .{ .d = d, .handle = null });
    }
    pub fn get(w: *const Weights, name: []const u8) ?Tensor {
        return if (w.poisoned) null else w.map.get(name);
    }
};

pub fn admitWidth(book: @import("core").exl3_load.Codebook, halves: u32) !void {
    if (book != .mul1) return error.UnsupportedCudaCodebook;
    if (cuda.exl3.variantIndex(halves, .mul1, 4) == null) return error.UnsupportedCudaWidth;
}

pub fn admitExpertWidths(lo: u32, hi: u32) !void {
    if (lo > hi) return error.UnsupportedCudaExpertWidth;
    admitWidth(.mul1, lo) catch return error.UnsupportedCudaExpertWidth;
    admitWidth(.mul1, hi) catch return error.UnsupportedCudaExpertWidth;
    _ = @import("block.zig").k2Range(lo, hi) catch return error.UnsupportedCudaExpertWidth;
}

// This adapter executes the chain's mul1 tiled route. The shared loader still represents all T20 codebooks.
pub fn load(w: *Weights, d: *const cuda.Driver, io: std.Io, cfg: *const @import("config.zig").Config, pack: *const @import("pack.zig").Pack, plan: *const @import("plan.zig").Plan) !void {
    try admitPlan(io, cfg, pack, plan);
    var b: named.Builder = .{ .gpa = w.gpa, .io = io, .pack = pack };
    defer b.deinit();
    try b.all(plan, cfg.o_groups);
    try w.upload(d, &b);
}
fn admitGroup(pack: *const @import("pack.zig").Pack, io: std.Io, group: @import("plan.zig").GroupPlan) !void {
    try admitWidth(try pack.codebook(io, group.prefix), group.part.group.k2());
    if (std.mem.indexOfScalar(u32, &cuda.qlinear.block.dense.linear_k2s, group.part.group.k2()) == null)
        return error.UnsupportedCudaWidth;
}

pub fn admitPlan(io: std.Io, cfg: *const @import("config.zig").Config, pack: *const @import("pack.zig").Pack, plan: *const @import("plan.zig").Plan) !void {
    try cfg.validateRelease();
    if (plan.leftovers.len != 0 or pack.missing != 0) return error.UnclaimedTensor;
    for (plan.layers) |layer| {
        for (layer.groups) |group| try admitGroup(pack, io, group);
        for (layer.experts) |projection| for (projection.prefixes, projection.parts) |prefix, part| {
            try admitWidth(try pack.codebook(io, prefix), part.group.k2());
            try admitExpertWidths(part.group.k2(), part.group.k2());
        };
    }
    if (plan.head) |group| try admitGroup(pack, io, group);
    if (plan.dspark) |draft| try admitGroup(pack, io, draft.main_proj);
}
