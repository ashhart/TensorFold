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

    /// Uploads every tensor `b` holds (host images freeable after); a name on the GPU is refused. All or nothing.
    pub fn upload(w: *Weights, d: *const cuda.Driver, b: *const named.Builder) !void {
        return w.uploadAll(d, b.out.items, ex_stacks.enabled());
    }

    fn uploadAll(w: *Weights, d: *const cuda.Driver, items: []const named.Named, stacks: bool) !void {
        const bufs_before = w.bufs.items.len;
        const bytes_before = w.bytes;
        // the keys this call put in the map (the map owns them); a failure takes each back out
        var added: std.ArrayList([]const u8) = .empty;
        defer added.deinit(w.gpa);
        errdefer w.rollback(bufs_before, bytes_before, added.items);
        try added.ensureTotalCapacity(w.gpa, items.len);
        const paired = try w.gpa.alloc(bool, items.len);
        defer w.gpa.free(paired);
        @memset(paired, false);
        // TF_DSV41_EX_STACKS: a routed suh / svh and its shared expert's in one buffer (ex_stacks.zig)
        if (stacks) {
            const names = try w.gpa.alloc([]const u8, items.len);
            defer w.gpa.free(names);
            const lens = try w.gpa.alloc(usize, items.len);
            defer w.gpa.free(lens);
            for (items, names, lens) |n, *x, *l| {
                x.* = n.name;
                l.* = n.bytes.len;
            }
            const ps = try ex_stacks.pairs(w.gpa, names, lens);
            defer w.gpa.free(ps);
            for (ps) |p| {
                const r = items[p[0]];
                const s = items[p[1]];
                if (w.map.contains(r.name) or w.map.contains(s.name)) return error.DuplicateWeight;
                // every host allocation before the buffer: once it is filled, handing it to `w` cannot fail
                try w.reserve(1, 2);
                const rk = try w.gpa.dupe(u8, r.name);
                errdefer w.gpa.free(rk);
                const sk = try w.gpa.dupe(u8, s.name);
                errdefer w.gpa.free(sk);
                var buf = try cuda.DeviceBuffer.alloc(d, r.bytes.len + s.bytes.len);
                errdefer buf.free();
                try cuda.DeviceBuffer.upload(buf, 0, r.bytes);
                try cuda.DeviceBuffer.upload(buf, r.bytes.len, s.bytes);
                w.bufs.appendAssumeCapacity(buf);
                w.commit(&added, rk, r, buf.ptr);
                w.commit(&added, sk, s, buf.ptr + r.bytes.len);
                paired[p[0]] = true;
                paired[p[1]] = true;
            }
        }
        for (items, paired) |n, done| {
            if (done) continue;
            if (w.map.contains(n.name)) return error.DuplicateWeight;
            try w.reserve(1, 1);
            const key = try w.gpa.dupe(u8, n.name);
            errdefer w.gpa.free(key);
            const buf = try cuda.DeviceBuffer.fromHost(d, n.bytes);
            w.bufs.appendAssumeCapacity(buf);
            w.commit(&added, key, n, buf.ptr);
        }
        // pageable uploads can still be in flight when cuMemcpyHtoD returns; the weights are whole before any launch
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    /// Room for `bufs` more buffers and `tensors` more names, so a filled buffer's hand-over cannot fail.
    fn reserve(w: *Weights, bufs: usize, tensors: u32) !void {
        try w.bufs.ensureUnusedCapacity(w.gpa, bufs);
        try w.map.ensureUnusedCapacity(w.gpa, tensors);
    }

    /// Puts `key` (owned, now the map's) for the tensor at `ptr`; `reserve` made the room, so nothing here fails.
    fn commit(w: *Weights, added: *std.ArrayList([]const u8), key: []const u8, n: named.Named, ptr: u64) void {
        var sha: [32]u8 = @splat(0);
        if (w.digest) std.crypto.hash.sha2.Sha256.hash(n.bytes, &sha, .{});
        w.map.putAssumeCapacityNoClobber(key, .{ .ptr = ptr, .len = n.bytes.len, .kind = n.kind, .sha256 = sha, .shape = n.shape, .rank = n.rank });
        added.appendAssumeCapacity(key);
        w.bytes += n.bytes.len;
    }

    /// A failed upload undone: its keys out of the map and freed, its buffers freed, `bufs` and `bytes` restored.
    fn rollback(w: *Weights, bufs: usize, bytes: usize, added: []const []const u8) void {
        for (added) |k| w.gpa.free(w.map.fetchRemove(k).?.key);
        for (w.bufs.items[bufs..]) |*b| b.free();
        w.bufs.shrinkRetainingCapacity(bufs);
        w.bytes = bytes;
    }

    pub fn get(w: *const Weights, name: []const u8) ?Tensor {
        return w.map.get(name);
    }
};

const testing = std.testing;
const abi = cuda.abi;

/// A driver of host-side addresses: it tracks the live allocations and counts a free of an address that is not live.
const Fake = struct {
    var next: abi.DevicePtr = 0x10_0000;
    var live: [16]abi.DevicePtr = @splat(0);
    var count: usize = 0;
    var strays: usize = 0;

    fn memAlloc(p: *abi.DevicePtr, len: usize) callconv(.c) abi.Result {
        std.debug.assert(count < live.len);
        p.* = next;
        next += (len + 255) / 256 * 256;
        live[count] = p.*;
        count += 1;
        return abi.success;
    }

    fn memFree(p: abi.DevicePtr) callconv(.c) abi.Result {
        const i = std.mem.indexOfScalar(abi.DevicePtr, live[0..count], p) orelse {
            strays += 1;
            return abi.success;
        };
        count -= 1;
        live[i] = live[count];
        return abi.success;
    }

    fn copyIn(_: abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) abi.Result {
        return abi.success;
    }

    fn streamSync(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }

    fn ctxSync() callconv(.c) abi.Result {
        return abi.success;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = memAlloc;
        d.api.cuMemFree_v2 = memFree;
        d.api.cuMemcpyHtoD_v2 = copyIn;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuCtxSynchronize = ctxSync;
        return d;
    }
};

fn testTensor(name: []const u8, bytes: []align(16) u8) named.Named {
    return .{ .name = name, .kind = .bf16, .shape = .{ bytes.len / 2, 1, 1, 1 }, .rank = 1, .bytes = bytes };
}

/// Every tensor of `earlier` and `items` held once, digested, its bytes counted; with stacks each pair in one buffer.
fn expectAll(w: *const Weights, earlier: []const named.Named, items: []const named.Named, stacks: bool) !void {
    var total: usize = 0;
    for (earlier) |n| total += n.bytes.len;
    for (items) |n| {
        try testing.expect(w.get(n.name) != null);
        const t = w.get(n.name).?;
        try testing.expectEqual(n.bytes.len, t.len);
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(n.bytes, &sha, .{});
        try testing.expectEqualSlices(u8, &sha, &t.sha256);
        total += n.bytes.len;
    }
    try testing.expectEqual(total, w.bytes);
    try testing.expectEqual(@as(u32, @intCast(earlier.len + items.len)), w.map.count());
    try testing.expectEqual(earlier.len + items.len - @as(usize, if (stacks) 2 else 0), w.bufs.items.len);
    try testing.expectEqual(w.bufs.items.len, Fake.count);
    for ([_][2][]const u8{ .{ "L3.moe.w1.suh", "L3.moe.shared.0.w1.suh" }, .{ "L3.moe.w2.svh", "L3.moe.shared.0.w2.svh" } }) |pair| {
        const routed = w.get(pair[0]).?;
        const shared = w.get(pair[1]).?;
        try testing.expectEqual(stacks, shared.ptr == routed.ptr + routed.len);
    }
}

/// `items` uploaded after `earlier` with its `at`-th allocation failing, then retried; false once it needs fewer.
fn uploadTrial(d: *const cuda.Driver, earlier: []const named.Named, items: []const named.Named, stacks: bool, at: usize) !bool {
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
    var w: Weights = .{ .gpa = failing.allocator() };
    defer w.deinit();
    try w.uploadAll(d, earlier, stacks);
    const bufs = w.bufs.items.len;
    const bytes = w.bytes;
    const live = Fake.count;
    failing.fail_index = failing.alloc_index + at;
    if (w.uploadAll(d, items, stacks)) |_| {
        try testing.expect(!failing.has_induced_failure);
        try expectAll(&w, earlier, items, stacks);
        return false;
    } else |e| {
        try testing.expect(e == error.OutOfMemory);
        try testing.expect(failing.has_induced_failure);
        // as it was: the earlier tensor kept; none of this call's buffers, names or bytes, nothing freed twice
        try testing.expectEqual(bufs, w.bufs.items.len);
        try testing.expectEqual(bytes, w.bytes);
        try testing.expectEqual(live, Fake.count);
        try testing.expectEqual(@as(usize, 0), Fake.strays);
        try testing.expectEqual(@as(u32, @intCast(earlier.len)), w.map.count());
        for (earlier) |n| try testing.expect(w.get(n.name) != null);
        for (items) |n| try testing.expect(w.get(n.name) == null);
    }
    failing.fail_index = std.math.maxInt(usize);
    try w.uploadAll(d, items, stacks);
    try expectAll(&w, earlier, items, stacks);
    return true;
}

test "upload is all or nothing: a failed allocation anywhere, paired or not, leaves the weights to retry or deinit" {
    const d = Fake.driver();
    var data: [256]u8 align(16) = undefined;
    for (&data, 0..) |*x, i| x.* = @truncate(i);
    const earlier = [_]named.Named{testTensor("embed", data[0..16])};
    // two routed / shared pairs (both routed lengths 16-aligned) between unpaired tensors
    const items = [_]named.Named{
        testTensor("L3.moe.shared.0.w1.suh", data[16..32]),
        testTensor("L3.attn.wq_b.trellis", data[32..48]),
        testTensor("L3.moe.w1.suh", data[48..80]),
        testTensor("L3.moe.w2.svh", data[80..112]),
        testTensor("L3.moe.shared.0.w2.svh", data[112..128]),
        testTensor("L3.attn.wkv", data[128..144]),
    };
    for ([_]bool{ false, true }) |stacks| {
        // each allocation in turn: the list's and pairs' room, buffer and map room, each name (a pair's second too)
        var at: usize = 0;
        while (try uploadTrial(&d, &earlier, &items, stacks, at)) : (at += 1) {
            try testing.expectEqual(@as(usize, 0), Fake.count);
            try testing.expectEqual(@as(usize, 0), Fake.strays);
        }
        try testing.expectEqual(@as(usize, 0), Fake.count);
        try testing.expectEqual(@as(usize, 0), Fake.strays);
        try testing.expect(at > 8);
    }
}

test "a name already on the GPU refuses the whole upload and keeps the earlier weights" {
    const d = Fake.driver();
    var data: [64]u8 align(16) = undefined;
    for (&data, 0..) |*x, i| x.* = @truncate(i);
    const live = Fake.count;
    var w: Weights = .{ .gpa = testing.allocator };
    defer w.deinit();
    const earlier = [_]named.Named{testTensor("embed", data[0..16])};
    try w.uploadAll(&d, &earlier, false);
    const again = [_]named.Named{ testTensor("L0.attn.wkv", data[16..32]), testTensor("embed", data[32..48]) };
    try testing.expectError(error.DuplicateWeight, w.uploadAll(&d, &again, false));
    try testing.expectEqual(@as(u32, 1), w.map.count());
    try testing.expectEqual(@as(usize, 1), w.bufs.items.len);
    try testing.expectEqual(@as(usize, 16), w.bytes);
    try testing.expect(w.get("L0.attn.wkv") == null);
    try testing.expectEqual(live + 1, Fake.count);
    try testing.expectEqual(@as(usize, 0), Fake.strays);
}
