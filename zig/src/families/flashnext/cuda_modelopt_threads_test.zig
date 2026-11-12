//! Experts packed on threads against the one-expert loop's packGateUp / packOne words, for every thread count.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const threads = @import("cuda_modelopt_threads.zig");
const ex = cuda.experts.Nvfp4Experts;

const width = 64;
const dims = 64;
const count = 33; // past the 32 threads a pack runs at most, so each count of threads gets a share

test "Flash Next expert loader: experts packed on threads give the one-expert loop's words for 1 to 32 threads" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    const codes_n = width * dims / 2;
    const scales_n = width * dims / 16;
    const planes = try a.alloc(u8, 3 * count * (codes_n + scales_n));
    defer a.free(planes);
    r.bytes(planes);
    var views: [3 * count]threads.View = undefined;
    for (&views, 0..) |*v, j| {
        const at = j * (codes_n + scales_n);
        v.* = .{ .codes = planes[at..][0..codes_n], .scales = planes[at + codes_n ..][0..scales_n], .global = 1.0 };
    }
    const up_words = ex.bytes(1, width, dims, 2) / 4;
    const down_words = ex.bytes(1, dims, width, 1) / 4;
    const want_up = try a.alloc(u32, count * up_words);
    defer a.free(want_up);
    const want_down = try a.alloc(u32, count * down_words);
    defer a.free(want_down);
    for (0..count) |i| {
        const g = views[3 * i];
        const u = views[3 * i + 1];
        const o = views[3 * i + 2];
        ex.packGateUp(want_up[i * up_words ..][0..up_words], .{ g.codes, g.scales }, .{ u.codes, u.scales }, width, dims);
        ex.packOne(want_down[i * down_words ..][0..down_words], o.codes, o.scales, dims, width);
    }
    const up = try a.alloc(u32, count * up_words);
    defer a.free(up);
    const down = try a.alloc(u32, count * down_words);
    defer a.free(down);
    for ([_]usize{ 1, 2, 3, 4, 8, 32 }) |n| {
        @memset(up, 0xdeadbeef);
        @memset(down, 0xdeadbeef);
        try threads.pack(&views, up, down, width, dims, n);
        try std.testing.expectEqualSlices(u32, want_up, up);
        try std.testing.expectEqualSlices(u32, want_down, down);
    }
}

test "Flash Next expert loader: a pack without whole experts or with uneven buffers is refused" {
    const views: [6]threads.View = @splat(.{ .codes = &.{}, .global = 1.0 });
    var up: [7]u32 = undefined;
    var down: [4]u32 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, threads.pack(views[0..0], &up, &down, width, dims, 2));
    try std.testing.expectError(error.UnexpectedTensor, threads.pack(views[0..2], &up, &down, width, dims, 2));
    try std.testing.expectError(error.UnexpectedTensor, threads.pack(views[0..6], &up, &down, width, dims, 2));
}

// ---- the whole loader over a synthetic checkpoint, on a counting fake driver -------------------------------------

const fx = @import("cuda_modelopt_threads_fixture.zig");
const abi = fx.abi;
const wt = fx.wt;
const experts = fx.experts;
const io = fx.io;
const ta = fx.ta;
const env = fx.env;
const Fake = fx.Fake;
const layer_w = fx.layer_w;
const layer_d = fx.layer_d;
const layer_n = fx.layer_n;
const Planes = fx.Planes;
const Layer = fx.Layer;
const csfPlanes = fx.csfPlanes;
const fixture = fx.fixture;
const freePlanes = fx.freePlanes;
const prefixOf = fx.prefixOf;
const expectLoopWords = fx.expectLoopWords;
/// A checkpoint reader without pinned staging (as zig/tests/cuda's sources): loadShape takes the one-expert loop.
const Plain = struct {
    store: *wt.Store,
    pub fn has(self: *Plain, name: []const u8) bool {
        return self.store.has(name);
    }
    pub fn get(self: *Plain, name: []const u8) !core.safetensors.Tensor {
        return self.store.get(name);
    }
    pub fn release(self: *Plain) void {
        self.store.release();
    }
};

/// The weight store with more shards held than one gather takes: a direct load falls back to the mapping.
const Crowded = struct {
    store: *wt.Store,
    pub fn has(self: *Crowded, name: []const u8) bool {
        return self.store.has(name);
    }
    pub fn hold(self: *Crowded, name: []const u8) !core.safetensors.Tensor {
        return self.store.hold(name);
    }
    pub fn holdAhead(self: *Crowded, name: []const u8) !core.safetensors.Tensor {
        return self.store.holdAhead(name);
    }
    pub fn release(self: *Crowded) void {
        self.store.release();
    }
    pub fn pinned(self: *Crowded, d: *const cuda.Driver, len: usize) ![]align(16) u8 {
        return self.store.pinned(d, len);
    }
    pub fn sources(_: *Crowded, _: []core.direct_gather.Source) error{TooManySources}![]core.direct_gather.Source {
        return error.TooManySources;
    }
};

test "Flash Next expert loader: a direct layer whose shards don't fit one gather loads through the mapping, the same words" {
    Fake.reset();
    var d = Fake.driver();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var planes: [2 * 3 * layer_n]Planes = undefined;
    try fixture(tmp.dir, &.{ .{ .index = 0 }, .{ .index = 1, .csf = true } }, null, &planes);
    defer freePlanes(&planes);
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var store = try wt.Store.open(ta, io, path);
    defer store.deinit();
    var crowded: Crowded = .{ .store = &store };
    var buf: [96]u8 = undefined;
    for ([_]usize{ 0, 1 }) |layer| {
        var owned = try threads.loadMode(ta, &d, &crowded, try prefixOf(&buf, layer), layer_d, layer_w, layer_n, 4, threads.staging_bytes, true);
        defer owned.free();
        try expectLoopWords(owned, planes[layer * 3 * layer_n ..][0 .. 3 * layer_n]);
    }
    store.dropStaging();
}

test "Flash Next expert loader: plain and CSF layers over 36 shards load on threads in chunks with the loop's words" {
    Fake.reset();
    var d = Fake.driver();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var planes: [2 * 3 * layer_n]Planes = undefined;
    try fixture(tmp.dir, &.{ .{ .index = 0 }, .{ .index = 1, .csf = true } }, null, &planes);
    defer freePlanes(&planes);
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var store = try wt.Store.open(ta, io, path);
    defer store.deinit();
    var buf: [96]u8 = undefined;
    const per = try experts.packedBytes(layer_w, layer_d, 2) + try experts.packedBytes(layer_d, layer_w, 1);
    // the whole layer in one chunk, then 5 experts a chunk (5, 5, 2), mapped or direct: the same words
    inline for (.{ false, true }) |direct| for ([_]usize{ threads.staging_bytes, 5 * per + per / 2 }) |cap| for ([_]usize{ 0, 1 }) |layer| {
        var owned = try threads.loadMode(ta, &d, &store, try prefixOf(&buf, layer), layer_d, layer_w, layer_n, 4, cap, direct);
        defer owned.free();
        try expectLoopWords(owned, planes[layer * 3 * layer_n ..][0 .. 3 * layer_n]);
    };
    // the staging grew once (the layer, then the smaller chunks reuse it)
    try std.testing.expectEqual(@as(usize, 1), Fake.calls[@backingInt(Fake.Kind.pinned)]);
    try std.testing.expectEqual(@as(usize, 1), Fake.live_pinned);
    store.dropStaging();
    try std.testing.expectEqual(@as(usize, 0), Fake.live_pinned);
    try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
}

test "Flash Next expert loader: loadShape packs on threads from the weight store and in the loop from a plain reader, the same words" {
    _ = env.setenv("TF_FLASHNEXT_LOAD_THREADS", "4", 1);
    defer _ = env.unsetenv("TF_FLASHNEXT_LOAD_THREADS");
    Fake.reset();
    var d = Fake.driver();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var planes: [3 * layer_n]Planes = undefined;
    try fixture(tmp.dir, &.{.{ .index = 0, .csf = true }}, null, &planes);
    defer freePlanes(&planes);
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var store = try wt.Store.open(ta, io, path);
    defer store.deinit();
    var buf: [96]u8 = undefined;
    const prefix = try prefixOf(&buf, 0);
    var threaded = try experts.loadShape(ta, &d, &store, prefix, 1, 0, layer_d, layer_w, layer_n);
    defer threaded.free();
    try std.testing.expectEqual(@as(usize, 1), Fake.calls[@backingInt(Fake.Kind.pinned)]); // the threaded path
    try expectLoopWords(threaded, &planes);
    var plain: Plain = .{ .store = &store };
    var looped = try experts.loadShape(ta, &d, &plain, prefix, 1, 0, layer_d, layer_w, layer_n);
    defer looped.free();
    try std.testing.expectEqual(@as(usize, 1), Fake.calls[@backingInt(Fake.Kind.pinned)]); // no staging in the loop
    try std.testing.expectEqualSlices(u8, @as([*]const u8, @ptrFromInt(threaded.up.ptr))[0..threaded.up.len], @as([*]const u8, @ptrFromInt(looped.up.ptr))[0..looped.up.len]);
    try std.testing.expectEqualSlices(u8, @as([*]const u8, @ptrFromInt(threaded.down.ptr))[0..threaded.down.len], @as([*]const u8, @ptrFromInt(looped.down.ptr))[0..looped.down.len]);
    // TF_FLASHNEXT_LOAD_THREADS=1: the weight store takes the loop too, no staging
    _ = env.setenv("TF_FLASHNEXT_LOAD_THREADS", "1", 1);
    store.dropStaging();
    var one = try experts.loadShape(ta, &d, &store, prefix, 1, 0, layer_d, layer_w, layer_n);
    defer one.free();
    try std.testing.expectEqual(@as(usize, 1), Fake.calls[@backingInt(Fake.Kind.pinned)]);
    try std.testing.expectEqual(@as(usize, 0), Fake.live_pinned);
    try expectLoopWords(one, &planes);
}

test "Flash Next expert loader: a load failing at any step leaves no pinned or device memory once its store closes, and a retry loads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var planes: [2 * 3 * layer_n]Planes = undefined;
    // layer 1's expert 7 lacks its down projection's global: the second layer fails while the staging is live
    try fixture(tmp.dir, &.{ .{ .index = 0 }, .{ .index = 1 } }, .{ .layer = 1, .expert = 7 }, &planes);
    defer freePlanes(&planes);
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var buf: [96]u8 = undefined;
    const old_limit = try cuda.memory.setAllocationLimit(null);
    defer _ = cuda.memory.setAllocationLimit(old_limit) catch unreachable;
    // the load's allocations in order: the up and down words, the pinned staging (one chunk here), both globals
    const up_total = layer_n * try experts.packedBytes(layer_w, layer_d, 2);
    const down_total = layer_n * try experts.packedBytes(layer_d, layer_w, 1);
    const steps = [_]usize{ up_total, down_total, up_total + down_total, 2 * layer_n * 4, layer_n * 4 };
    // pinned buffers live when the refusal comes: the staging is allocated third
    const pinned_live = [_]usize{ 0, 0, 0, 1, 1 };
    // a refusal at each allocation (the allocation limit one byte short of it), then a missing plane on layer 1
    for (0..steps.len + 1) |case| {
        Fake.reset();
        var d = Fake.driver();
        {
            var store = try wt.Store.open(ta, io, path);
            defer store.deinit(); // the engine's errdefer: whatever failed, the store's close frees its staging
            var layer: usize = 0;
            if (case < steps.len) {
                const u = cuda.memory.usage(false);
                var room: u64 = u.device + u.host;
                for (steps[0..case]) |n| room += n;
                _ = try cuda.memory.setAllocationLimit(room + steps[case] - 1);
            } else {
                var ok = try threads.load(ta, &d, &store, try prefixOf(&buf, 0), layer_d, layer_w, layer_n, 4);
                ok.free();
                layer = 1;
            }
            const got = threads.load(ta, &d, &store, try prefixOf(&buf, layer), layer_d, layer_w, layer_n, 4);
            _ = try cuda.memory.setAllocationLimit(null);
            if (got) |owned| {
                var o = owned;
                o.free();
                return error.TestExpectedError;
            } else |err| try std.testing.expectEqual(@as(anyerror, if (case < steps.len) error.OutOfDeviceMemory else error.MissingTensor), err);
            // what the refusal left live is the store's staging alone, until the store closes
            try std.testing.expectEqual(if (case < steps.len) pinned_live[case] else 1, Fake.live_pinned);
            try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
        }
        try std.testing.expectEqual(@as(usize, 0), Fake.live_pinned);
        try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
        // a clean retry: a new store, nothing refused
        var store = try wt.Store.open(ta, io, path);
        defer store.deinit();
        var owned = try threads.load(ta, &d, &store, try prefixOf(&buf, 0), layer_d, layer_w, layer_n, 4);
        defer owned.free();
        try expectLoopWords(owned, planes[0 .. 3 * layer_n]);
        store.dropStaging();
        try std.testing.expectEqual(@as(usize, 0), Fake.live_pinned);
    }
    try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
}

test "Flash Next expert loader: a host allocation failing at any point leaves no device memory, and the next load works" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var planes: [3 * layer_n]Planes = undefined;
    try fixture(tmp.dir, &.{.{ .index = 0, .csf = true }}, null, &planes);
    defer freePlanes(&planes);
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var buf: [96]u8 = undefined;
    var d = Fake.driver();
    const per = try experts.packedBytes(layer_w, layer_d, 2) + try experts.packedBytes(layer_d, layer_w, 1);
    // each allocation fails in turn: mapped's views and globals, plus direct's gathers (chunked: on the side thread)
    inline for (.{ false, true }) |direct| {
        var walked: [2]usize = undefined;
        for ([_]usize{ threads.staging_bytes, 5 * per + per / 2 }, &walked) |cap, *w| {
            var fails: usize = 0;
            while (true) : (fails += 1) {
                Fake.reset();
                var store = try wt.Store.open(ta, io, path);
                defer store.deinit();
                var failing = std.testing.FailingAllocator.init(ta, .{ .fail_index = fails });
                if (threads.loadMode(failing.allocator(), &d, &store, try prefixOf(&buf, 0), layer_d, layer_w, layer_n, 4, cap, direct)) |owned| {
                    var o = owned;
                    defer o.free();
                    try expectLoopWords(o, &planes);
                    break;
                } else |err| try std.testing.expectEqual(@as(anyerror, error.OutOfMemory), err);
                try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
            }
            w.* = fails;
        }
        if (direct) try std.testing.expect(walked[0] > 3 and walked[1] > walked[0]) else try std.testing.expect(walked[0] == 3 and walked[1] == 3);
    }
}

test "Flash Next expert loader: two stores in one process each own and free their staging" {
    Fake.reset();
    var d = Fake.driver();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var planes: [3 * layer_n]Planes = undefined;
    try fixture(tmp.dir, &.{.{ .index = 0 }}, null, &planes);
    defer freePlanes(&planes);
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var buf: [96]u8 = undefined;
    var first = try wt.Store.open(ta, io, path);
    defer first.deinit();
    var second = try wt.Store.open(ta, io, path);
    defer second.deinit();
    var a1 = try threads.load(ta, &d, &first, try prefixOf(&buf, 0), layer_d, layer_w, layer_n, 4);
    defer a1.free();
    var a2 = try threads.load(ta, &d, &second, try prefixOf(&buf, 0), layer_d, layer_w, layer_n, 4);
    defer a2.free();
    try std.testing.expectEqual(@as(usize, 2), Fake.live_pinned);
    first.dropStaging();
    try std.testing.expectEqual(@as(usize, 1), Fake.live_pinned);
    try expectLoopWords(a2, &planes);
    second.dropStaging();
    try std.testing.expectEqual(@as(usize, 0), Fake.live_pinned);
}

test "Flash Next expert loader: uneven geometry is refused before the store or the driver is touched" {
    _ = env.setenv("TF_FLASHNEXT_LOAD_THREADS", "4", 1);
    defer _ = env.unsetenv("TF_FLASHNEXT_LOAD_THREADS");
    Fake.reset();
    var d = Fake.driver();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var planes: [3 * layer_n]Planes = undefined;
    try fixture(tmp.dir, &.{.{ .index = 0 }}, null, &planes);
    defer freePlanes(&planes);
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var store = try wt.Store.open(ta, io, path);
    defer store.deinit();
    var buf: [96]u8 = undefined;
    const prefix = try prefixOf(&buf, 0);
    // a rank past one GPU's world, no width, a width that is not whole 32-column blocks
    try std.testing.expectError(error.UnevenExpertSplit, experts.loadShape(ta, &d, &store, prefix, 1, 1, layer_d, layer_w, layer_n));
    try std.testing.expectError(error.UnevenExpertSplit, experts.loadShape(ta, &d, &store, prefix, 1, 0, layer_d, 0, layer_n));
    try std.testing.expectError(error.UnevenExpertSplit, experts.loadShape(ta, &d, &store, prefix, 1, 0, layer_d, layer_w - 16, layer_n));
    try std.testing.expectEqual(@as(usize, 0), store.held.items.len + store.released.items.len);
    try std.testing.expect(store.staging == null);
    for (Fake.calls) |n| try std.testing.expectEqual(@as(usize, 0), n);
}

/// The weight store with a rendezvous: chunk k + 1's gather start and chunk k's pack end wait for each other.
const Marked = struct {
    store: *wt.Store,
    chunks: usize,
    started: std.atomic.Value(usize) = .init(0),
    ended: std.atomic.Value(usize) = .init(0),
    met: std.atomic.Value(usize) = .init(0),

    pub fn hold(self: *Marked, name: []const u8) !core.safetensors.Tensor {
        return self.store.hold(name);
    }
    pub fn has(self: *Marked, name: []const u8) bool {
        return self.store.has(name);
    }
    pub fn release(self: *Marked) void {
        self.store.release();
    }
    pub fn sources(self: *Marked, out: []core.direct_gather.Source) ![]core.direct_gather.Source {
        return self.store.sources(out);
    }
    pub fn pinned(self: *Marked, d: *const cuda.Driver, len: usize) ![]align(16) u8 {
        return self.store.pinned(d, len);
    }
    /// Only code running both at once meets: run one after the other, either side waits out its 5 s.
    pub fn chunkMark(self: *Marked, gather: bool, k: usize) void {
        if (gather) {
            _ = self.started.fetchAdd(1, .acq_rel);
            if (k > 0 and reach(&self.ended, k)) _ = self.met.fetchAdd(1, .acq_rel);
        } else {
            _ = self.ended.fetchAdd(1, .acq_rel);
            if (k + 1 < self.chunks and reach(&self.started, k + 2)) _ = self.met.fetchAdd(1, .acq_rel);
        }
    }
    fn reach(v: *std.atomic.Value(usize), n: usize) bool {
        var waited: usize = 0;
        while (v.load(.acquire) < n and waited < 500) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
        return v.load(.acquire) >= n;
    }
};

test "Flash Next expert loader: a direct load gathers each next chunk while the chunk before it packs, with the same words" {
    Fake.reset();
    var d = Fake.driver();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var planes: [3 * layer_n]Planes = undefined;
    try fixture(tmp.dir, &.{.{ .index = 0, .csf = true }}, null, &planes);
    defer freePlanes(&planes);
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var store = try wt.Store.open(ta, io, path);
    defer store.deinit();
    var buf: [96]u8 = undefined;
    const per = try experts.packedBytes(layer_w, layer_d, 2) + try experts.packedBytes(layer_d, layer_w, 1);
    // 5 experts a chunk: chunks (5, 5, 2); chunk 0's and chunk 1's packs each meet the next chunk's gather
    var marked: Marked = .{ .store = &store, .chunks = 3 };
    var owned = try threads.loadMode(ta, &d, &marked, try prefixOf(&buf, 0), layer_d, layer_w, layer_n, 4, 5 * per + per / 2, true);
    defer owned.free();
    try expectLoopWords(owned, &planes);
    try std.testing.expectEqual(@as(usize, 3), marked.ended.load(.acquire));
    try std.testing.expectEqual(@as(usize, 3), marked.started.load(.acquire));
    try std.testing.expectEqual(@as(usize, 4), marked.met.load(.acquire));
}
