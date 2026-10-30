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

const abi = cuda.abi;
const wt = @import("cuda_prompt_wt.zig");
const experts = @import("cuda_modelopt_experts.zig");
const io = std.testing.io;
const ta = std.testing.allocator;
const env = @import("sys").env;

/// A driver whose device memory is host memory, counting calls and live allocations; cuda.memory's limit refuses.
const Fake = struct {
    const Kind = enum { pinned, device, copy };
    var calls: [3]usize = @splat(0);
    var live_pinned: usize = 0;
    var live_device: usize = 0;
    var spans: [256][]u8 = undefined;
    var nspans: usize = 0;

    fn reset() void {
        calls = @splat(0);
    }
    fn tally(k: Kind) void {
        calls[@backingInt(k)] += 1;
    }
    fn take(n: usize) ?[]u8 {
        if (nspans == spans.len) @panic("the fake driver holds 256 allocations at most");
        const m = std.heap.page_allocator.alloc(u8, @max(n, 1)) catch return null;
        spans[nspans] = m;
        nspans += 1;
        return m;
    }
    fn give(p: usize) void {
        for (spans[0..nspans], 0..) |m, i| if (@intFromPtr(m.ptr) == p) {
            std.heap.page_allocator.free(m);
            nspans -= 1;
            spans[i] = spans[nspans];
            return;
        };
        @panic("freed memory the fake never gave");
    }
    fn hostAlloc(p: *?*anyopaque, n: usize, _: c_uint) callconv(.c) abi.Result {
        tally(.pinned);
        const m = take(n) orelse return 2;
        p.* = m.ptr;
        live_pinned += 1;
        return abi.success;
    }
    fn hostFree(p: ?*anyopaque) callconv(.c) abi.Result {
        give(@intFromPtr(p));
        live_pinned -= 1;
        return abi.success;
    }
    fn devAlloc(p: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
        tally(.device);
        const m = take(n) orelse return 2;
        p.* = @intFromPtr(m.ptr);
        live_device += 1;
        return abi.success;
    }
    fn devFree(p: abi.DevicePtr) callconv(.c) abi.Result {
        give(p);
        live_device -= 1;
        return abi.success;
    }
    fn copy(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
        tally(.copy);
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        return abi.success;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }
    fn syncCtx() callconv(.c) abi.Result {
        return abi.success;
    }
    fn name(_: abi.Result, s: *?[*:0]const u8) callconv(.c) abi.Result {
        s.* = null;
        return 1;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.func_name = null;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemAlloc_v2 = devAlloc;
        d.api.cuMemFree_v2 = devFree;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = syncCtx;
        d.api.cuGetErrorName = name;
        d.api.cuGetErrorString = name;
        return d;
    }
};

const layer_w = 64; // the experts' intermediate width
const layer_d = 64; // the model width
const layer_n = 12; // experts a layer, each projection in its own shard: 36 shards

/// One expert projection's planes: FP4 codes, row-major E4M3 scales a 16 (decoded, for CSF), the global.
const Planes = struct { codes: []u8, scales: []u8, global: f32 };

const Layer = struct { index: usize, csf: bool = false };

/// A CSF plane of `scales` [rows, cols], as core/nvfp4_csf_test.zig builds one: row bases, offsets, exceptions.
fn csfPlanes(scales: []u8, rows: usize, cols: usize, fixed: *std.ArrayList(u8), exceptions: *std.ArrayList(u8)) !void {
    try fixed.appendNTimes(ta, 0, rows + rows * cols / 2);
    for (0..rows) |row| {
        const base: u8 = 0x30;
        const slab = row / 16 * (16 + cols * 8);
        fixed.items[slab + row % 16] = base;
        for (0..cols) |col| {
            const i = row * cols + col;
            const delta: u8 = if (scales[i] >= base and scales[i] - base < 16) scales[i] - base else 0;
            fixed.items[slab + 16 + row % 16 * (cols / 2) + col / 2] |= delta << @as(u3, if (col % 2 == 0) 0 else 4);
            if (base + delta != scales[i]) try exceptions.appendSlice(ta, &std.mem.toBytes(std.mem.nativeToLittle(u32, @as(u32, @intCast(i)) | (@as(u32, scales[i]) << 24))));
        }
    }
}

/// `layers` with every expert projection in its own shard; `missing` drops one down projection's global.
fn fixture(dir: std.Io.Dir, layers: []const Layer, missing: ?struct { layer: usize, expert: usize }, planes: []Planes) !void {
    var index: std.Io.Writer.Allocating = .init(ta);
    defer index.deinit();
    try index.writer.writeAll("{\"weight_map\":{");
    var prng = std.Random.DefaultPrng.init(0xe1);
    const r = prng.random();
    var first = true;
    var at: usize = 0;
    for (layers) |layer| for (0..layer_n) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        const n: usize = if (pi == 2) layer_d else layer_w;
        const k: usize = if (pi == 2) layer_w else layer_d;
        const p = &planes[at];
        at += 1;
        p.* = .{ .codes = try ta.alloc(u8, n * k / 2), .scales = try ta.alloc(u8, n * k / 16), .global = 0.5 + 0.01 * @as(f32, @floatFromInt(e)) };
        r.bytes(p.codes);
        // finite E4M3 bytes; one in 9 off the CSF slab's 16-byte window (an exception)
        for (p.scales) |*s| s.* = if (r.uintLessThan(u8, 9) == 0) 0x10 + r.uintLessThan(u8, 16) else 0x30 + r.uintLessThan(u8, 16);
        const base = try std.fmt.allocPrint(ta, "language_model.model.layers.{d}.mlp.experts.{d}.{s}", .{ layer.index, e, proj });
        defer ta.free(base);
        const file = try std.fmt.allocPrint(ta, "l{d}-e{d}-{s}.safetensors", .{ layer.index, e, proj });
        defer ta.free(file);
        const drop = if (missing) |m| m.layer == layer.index and m.expert == e and pi == 2 else false;
        var header: std.Io.Writer.Allocating = .init(ta);
        defer header.deinit();
        var data: std.ArrayList(u8) = .empty;
        defer data.deinit(ta);
        var parts: [4][]const u8 = undefined;
        var nparts: usize = 0;
        try data.appendSlice(ta, p.codes);
        try header.writer.print("{{\"{s}.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}", .{ base, n, k / 2, data.items.len });
        parts[nparts] = "weight";
        nparts += 1;
        if (layer.csf) {
            var fixed: std.ArrayList(u8) = .empty;
            defer fixed.deinit(ta);
            var exc: std.ArrayList(u8) = .empty;
            defer exc.deinit(ta);
            try csfPlanes(p.scales, n, k / 16, &fixed, &exc);
            const f0 = data.items.len;
            try data.appendSlice(ta, fixed.items);
            try header.writer.print(",\"{s}.weight_scale.nvfp4_csf_fixed\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ base, n / 16, 16 + 8 * (k / 16), f0, data.items.len });
            const x0 = data.items.len;
            try data.appendSlice(ta, exc.items);
            try header.writer.print(",\"{s}.weight_scale.nvfp4_csf_exceptions\":{{\"dtype\":\"U32\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ base, exc.items.len / 4, x0, data.items.len });
            parts[nparts] = "weight_scale.nvfp4_csf_fixed";
            parts[nparts + 1] = "weight_scale.nvfp4_csf_exceptions";
            nparts += 2;
        } else {
            const s0 = data.items.len;
            try data.appendSlice(ta, p.scales);
            try header.writer.print(",\"{s}.weight_scale\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ base, n, k / 16, s0, data.items.len });
            parts[nparts] = "weight_scale";
            nparts += 1;
        }
        if (!drop) {
            const g0 = data.items.len;
            try data.appendSlice(ta, &std.mem.toBytes(@as(u32, @bitCast(p.global))));
            try header.writer.print(",\"{s}.weight_scale_2\":{{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}", .{ base, g0, data.items.len });
            parts[nparts] = "weight_scale_2";
            nparts += 1;
        }
        try header.writer.writeByte('}');
        var image: std.Io.Writer.Allocating = .init(ta);
        defer image.deinit();
        try image.writer.writeAll(&std.mem.toBytes(std.mem.nativeToLittle(u64, header.written().len)));
        try image.writer.writeAll(header.written());
        try image.writer.writeAll(data.items);
        try dir.writeFile(io, .{ .sub_path = file, .data = image.written() });
        for (parts[0..nparts]) |part| {
            if (!first) try index.writer.writeByte(',');
            first = false;
            try index.writer.print("\"{s}.{s}\":\"{s}\"", .{ base, part, file });
        }
    };
    try index.writer.writeAll("}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
}

fn freePlanes(planes: []Planes) void {
    for (planes) |p| {
        ta.free(p.codes);
        ta.free(p.scales);
    }
}

fn prefixOf(buf: []u8, layer: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.mlp", .{layer});
}

/// The words and globals the one-expert loop packs for a layer's planes, against the device buffers.
fn expectLoopWords(owned: experts.Owned, planes: []const Planes) !void {
    const up_words = ex.bytes(1, layer_w, layer_d, 2) / 4;
    const down_words = ex.bytes(1, layer_d, layer_w, 1) / 4;
    const want_up = try ta.alloc(u32, layer_n * up_words);
    defer ta.free(want_up);
    const want_down = try ta.alloc(u32, layer_n * down_words);
    defer ta.free(want_down);
    for (0..layer_n) |i| {
        const g = planes[3 * i];
        const u = planes[3 * i + 1];
        const o = planes[3 * i + 2];
        ex.packGateUp(want_up[i * up_words ..][0..up_words], .{ g.codes, g.scales }, .{ u.codes, u.scales }, layer_w, layer_d);
        ex.packOne(want_down[i * down_words ..][0..down_words], o.codes, o.scales, layer_d, layer_w);
    }
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_up), @as([*]const u8, @ptrFromInt(owned.up.ptr))[0..owned.up.len]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_down), @as([*]const u8, @ptrFromInt(owned.down.ptr))[0..owned.down.len]);
    const ug: [*]const f32 = @ptrFromInt(owned.up_scale.ptr);
    const dg: [*]const f32 = @ptrFromInt(owned.down_scale.ptr);
    for (0..layer_n) |i| {
        try std.testing.expectEqual(planes[3 * i].global, ug[2 * i]);
        try std.testing.expectEqual(planes[3 * i + 1].global, ug[2 * i + 1]);
        try std.testing.expectEqual(planes[3 * i + 2].global, dg[i]);
    }
}

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
    // the whole layer in one chunk, then 5 experts a chunk (5, 5, 2): the same words
    for ([_]usize{ threads.staging_bytes, 5 * per + per / 2 }) |cap| for ([_]usize{ 0, 1 }) |layer| {
        var owned = try threads.loadChunked(ta, &d, &store, try prefixOf(&buf, layer), layer_d, layer_w, layer_n, 4, cap);
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
    var fails: usize = 0;
    while (true) : (fails += 1) {
        Fake.reset();
        var store = try wt.Store.open(ta, io, path);
        defer store.deinit();
        // the loader's own allocations (the views, both globals' host copies) fail in turn until none is left
        var failing = std.testing.FailingAllocator.init(ta, .{ .fail_index = fails });
        if (threads.load(failing.allocator(), &d, &store, try prefixOf(&buf, 0), layer_d, layer_w, layer_n, 4)) |owned| {
            var o = owned;
            defer o.free();
            try expectLoopWords(o, &planes);
            break;
        } else |err| try std.testing.expectEqual(@as(anyerror, error.OutOfMemory), err);
        try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
    }
    try std.testing.expectEqual(@as(usize, 3), fails);
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
    try std.testing.expectEqual(@as(usize, 0), store.held.items.len + store.nreleased);
    try std.testing.expect(store.staging == null);
    for (Fake.calls) |n| try std.testing.expectEqual(@as(usize, 0), n);
}
