//! One GPU's NVFP4 routed experts packed on threads from the store's mapped planes, through its pinned staging.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const experts = @import("cuda_modelopt_experts.zig");
const ex = cuda.experts.Nvfp4Experts;
const slice = core.nvfp4_slice;

pub const default_threads = 8;
pub const max_threads = 32;

/// Pinned bytes a load stages at most, a whole number of experts' words at a time.
pub const staging_bytes: usize = 192 << 20;

/// TF_FLASHNEXT_LOAD_THREADS: unset, 8; otherwise 1 (loadShape's one-expert loop) to 32 threads, anything else refused.
pub fn parseThreads(v: ?[]const u8) !usize {
    const s = v orelse return default_threads;
    const n = std.fmt.parseInt(usize, s, 10) catch return error.BadLoadThreads;
    if (n == 0 or n > max_threads) return error.BadLoadThreads;
    return n;
}

pub fn configured() !usize {
    return parseThreads(if (std.c.getenv("TF_FLASHNEXT_LOAD_THREADS")) |v| std.mem.span(v) else null);
}

/// A projection's borrowed planes: FP4 codes, plain E4M3 scales or the CSF fixed and exception planes, the global.
pub const View = struct { codes: []const u8, scales: []const u8 = &.{}, fixed: []const u8 = &.{}, exceptions: []const u8 = &.{}, global: f32 };

/// The store as slice.stored reads it, every tensor's pages read ahead (this loader copies all of each).
fn Ahead(comptime S: type) type {
    return struct {
        store: S,

        pub fn get(self: @This(), name: []const u8) !core.safetensors.Tensor {
            return self.store.holdAhead(name);
        }

        pub fn has(self: @This(), name: []const u8) bool {
            return self.store.has(name);
        }
    };
}

/// A projection read and refused as loadShape's loop reads it: weight shape, slice.stored, slice.dimensions, global.
fn viewOf(store: anytype, name: []const u8, n: usize, k: usize) !View {
    var buf: [512]u8 = undefined;
    const w = try store.hold(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}));
    if (!w.is(.u8, &.{ n, k / 2 })) return error.UnexpectedTensor;
    const source = try slice.stored(Ahead(@TypeOf(store)){ .store = store }, name);
    _ = try slice.dimensions(source);
    const global: f32 = @bitCast(std.mem.readInt(u32, source.global.bytes[0..4], .little));
    if (!std.math.isFinite(global) or global <= 0) return error.UnsupportedQuantization;
    return switch (source.scales) {
        .plain => |s| .{ .codes = source.weight.bytes, .scales = s.bytes, .global = global },
        .csf => |s| .{ .codes = source.weight.bytes, .fixed = s.fixed.bytes, .exceptions = s.exceptions.bytes, .global = global },
    };
}

/// A view's row-major E4M3 scales [n, k/16]: the plain plane, or its CSF planes decoded whole as slice.slice does.
fn scalesOf(v: View, n: usize, k: usize, buf: []u8) ![]const u8 {
    if (v.scales.len > 0) return v.scales;
    const plane = try core.nvfp4_csf.Plane.init(n, k / 16, v.fixed, v.exceptions);
    try plane.decode(buf[0 .. n * (k / 16)], 0, n, 0, k / 16);
    return buf[0 .. n * (k / 16)];
}

/// Experts [lo, hi) of `views` (gate, up, down each) packed as loadShape's loop packs them, into `up` and `down`.
const Work = struct {
    views: []const View,
    up: []u32,
    down: []u32,
    width: usize,
    dims: usize,
    lo: usize,
    hi: usize,
    failed: ?(error{OutOfMemory} || core.nvfp4_csf.Error) = null,

    fn run(w: *Work) void {
        w.pack() catch |err| {
            w.failed = err;
        };
    }

    fn pack(w: *Work) !void {
        const count = w.views.len / 3;
        const up_words = w.up.len / count;
        const down_words = w.down.len / count;
        const each = @max(w.width * (w.dims / 16), w.dims * (w.width / 16));
        const bufs = try std.heap.page_allocator.alloc(u8, 3 * each);
        defer std.heap.page_allocator.free(bufs);
        for (w.lo..w.hi) |i| {
            const g = w.views[3 * i];
            const u = w.views[3 * i + 1];
            const o = w.views[3 * i + 2];
            const gs = try scalesOf(g, w.width, w.dims, bufs[0..each]);
            const us = try scalesOf(u, w.width, w.dims, bufs[each..][0..each]);
            const os = try scalesOf(o, w.dims, w.width, bufs[2 * each ..][0..each]);
            ex.packGateUp(w.up[i * up_words ..][0..up_words], .{ g.codes, gs }, .{ u.codes, us }, w.width, w.dims);
            ex.packOne(w.down[i * down_words ..][0..down_words], o.codes, os, w.dims, w.width);
        }
    }
};

/// Every expert's gate / up and down words, split over `n` threads; a thread that cannot start leaves its part here.
pub fn pack(views: []const View, up: []u32, down: []u32, width: usize, dims: usize, n: usize) !void {
    const count = views.len / 3;
    if (count == 0 or views.len != 3 * count or up.len % count != 0 or down.len % count != 0) return error.UnexpectedTensor;
    const t_n = @max(1, @min(n, max_threads, count));
    const per = (count + t_n - 1) / t_n;
    var works: [max_threads]Work = undefined;
    var handles: [max_threads]?std.Thread = @splat(null);
    for (0..t_n) |t| {
        works[t] = .{ .views = views, .up = up, .down = down, .width = width, .dims = dims, .lo = @min(count, t * per), .hi = @min(count, (t + 1) * per) };
        handles[t] = std.Thread.spawn(.{}, Work.run, .{&works[t]}) catch blk: {
            works[t].run();
            break :blk null;
        };
    }
    for (handles[0..t_n]) |h| if (h) |th| th.join();
    for (works[0..t_n]) |w| if (w.failed) |err| return err;
}

/// A layer's planes borrowed with their pages read ahead, then packed on `n` threads a chunk of experts at a time.
pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize) !experts.Owned {
    return loadChunked(a, d, store, prefix, dims, width, count, n, staging_bytes);
}

pub fn loadChunked(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize, stage_cap: usize) !experts.Owned {
    const up_bytes = try experts.packedBytes(width, dims, 2);
    const down_bytes = try experts.packedBytes(dims, width, 1);
    const up_total = std.math.mul(usize, count, up_bytes) catch return error.UnexpectedTensor;
    const down_total = std.math.mul(usize, count, down_bytes) catch return error.UnexpectedTensor;
    if (count == 0 or up_bytes == 0 or down_bytes == 0) return error.UnexpectedTensor;
    const view_count = std.math.mul(usize, 3, count) catch return error.UnexpectedTensor;
    const views = try a.alloc(View, view_count);
    defer a.free(views);
    defer store.release();
    var buf: [512]u8 = undefined;
    for (0..count) |i| {
        views[3 * i] = try viewOf(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.gate_proj", .{ prefix, i }), width, dims);
        views[3 * i + 1] = try viewOf(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.up_proj", .{ prefix, i }), width, dims);
        views[3 * i + 2] = try viewOf(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.down_proj", .{ prefix, i }), dims, width);
    }
    var up = try cuda.DeviceBuffer.alloc(d, up_total);
    errdefer up.free();
    var down = try cuda.DeviceBuffer.alloc(d, down_total);
    errdefer down.free();
    // the store's pinned staging, kept across a load's layers and freed by its dropStaging or deinit on any path
    const chunk = @max(1, @min(count, stage_cap / (up_bytes + down_bytes)));
    const stage = try store.pinned(d, chunk * (up_bytes + down_bytes));
    var lo: usize = 0;
    while (lo < count) : (lo += chunk) {
        const m = @min(chunk, count - lo);
        const up_words = std.mem.bytesAsSlice(u32, stage[0 .. m * up_bytes]);
        const down_words: []u32 = @alignCast(std.mem.bytesAsSlice(u32, stage[m * up_bytes ..][0 .. m * down_bytes]));
        try pack(views[3 * lo ..][0 .. 3 * m], up_words, down_words, width, dims, n);
        // synchronous from pinned memory, as cuda_pocket.zig's staged: done before return, so the next chunk may refill
        try d.check(d.api.cuMemcpyHtoD_v2(try up.at(lo * up_bytes), stage.ptr, m * up_bytes), "cuMemcpyHtoD pinned experts");
        try d.check(d.api.cuMemcpyHtoD_v2(try down.at(lo * down_bytes), stage.ptr + m * up_bytes, m * down_bytes), "cuMemcpyHtoD pinned experts");
    }
    const up_globals = try a.alloc(f32, 2 * count);
    defer a.free(up_globals);
    const down_globals = try a.alloc(f32, count);
    defer a.free(down_globals);
    for (0..count) |i| {
        up_globals[2 * i] = views[3 * i].global;
        up_globals[2 * i + 1] = views[3 * i + 1].global;
        down_globals[i] = views[3 * i + 2].global;
    }
    var us = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_globals));
    errdefer us.free();
    const ds = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_globals));
    return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count };
}

test "Flash Next expert loader: TF_FLASHNEXT_LOAD_THREADS takes 1 to 32 and refuses the rest" {
    try std.testing.expectEqual(@as(usize, default_threads), try parseThreads(null));
    try std.testing.expectEqual(@as(usize, 1), try parseThreads("1"));
    try std.testing.expectEqual(@as(usize, max_threads), try parseThreads("32"));
    for ([_][]const u8{ "0", "33", "-1", "8x", "" }) |bad| try std.testing.expectError(error.BadLoadThreads, parseThreads(bad));
}

test {
    _ = @import("cuda_modelopt_threads_test.zig");
}
