//! Cache.keep on the 27B's shape with a mock driver: no pending copy leaves the log or the buffer, whatever is kept.
const std = @import("std");
const cuda = @import("cuda");
const caches = @import("cuda_cache.zig");
const geometry = @import("cuda_geometry.zig");
const E = @import("cuda_engine.zig").Engine;
const most = @import("cuda_engine.zig").most_window;
const t = std.testing;

/// CUDA_ERROR_INVALID_VALUE: the driver's answer to a copy that leaves its allocation.
const invalid_value: cuda.abi.Result = 1;
/// The keep kernels' mock handles, which index Mock.launches.
const apply = 1;
const apply_pend = 2;

const Mock = struct {
    const Span = struct { at: u64, len: usize };
    var spans: [8]Span = undefined;
    var n: usize = 0;
    var next: u64 = 0x10000;
    var copies: usize = 0;
    var launches: [5]usize = @splat(0);

    fn driver() cuda.Driver {
        n = 0;
        next = 0x10000;
        copies = 0;
        launches = @splat(0);
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuLaunchKernel = launch;
        d.api.cuMemcpy2DAsync_v2 = copy;
        d.api.cuGetErrorName = name;
        d.api.cuGetErrorString = name;
        return d;
    }
    fn alloc(out: *u64, len: usize) callconv(.c) cuda.abi.Result {
        spans[n] = .{ .at = next, .len = len };
        n += 1;
        out.* = next;
        next += len + 4096; // a gap after each allocation, so a copy past its end lands in none
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        launches[@intFromPtr(f)] += 1;
        return 0;
    }
    /// Each side's rows inside one allocation, as the driver checks, or CUDA_ERROR_INVALID_VALUE.
    fn copy(c: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        return if (inside(c.src, c.src_pitch, c.width, c.height) and inside(c.dst, c.dst_pitch, c.width, c.height)) 0 else invalid_value;
    }
    fn inside(at: u64, pitch: usize, width: usize, height: usize) bool {
        const end = at + (height - 1) * pitch + width;
        for (spans[0..n]) |s| if (at >= s.at and end <= s.at + s.len) return true;
        return false;
    }
    fn name(_: cuda.abi.Result, s: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
        s.* = "CUDA_ERROR_INVALID_VALUE";
        return 0;
    }
};

/// The dense 27B's widths: 64 layers, every fourth one attention, so 48 linear layers.
fn engine(d: *cuda.Driver) E {
    var linear: [geometry.max_layers]bool = @splat(false);
    for (linear[0..64], 0..) |*l, i| l.* = i % 4 != 3;
    var e: E = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = null };
    e.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 8, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 4096, .linear = linear };
    e.geo = .{ .hid = 5120, .nk = 16, .nv = 48, .qh = 24, .kvh = 4, .nl = 48 };
    e.kv8 = false;
    e.n_taps = 0;
    const f = struct {
        fn at(dd: *cuda.Driver, i: usize) cuda.Function {
            return .{ .d = dd, .handle = @ptrFromInt(i) };
        }
    }.at;
    e.keeps = .{ .apply = f(d, apply), .apply_pend = f(d, apply_pend), .conv = f(d, 3), .compact = f(d, 4) };
    return e;
}

/// A cache with its own log of a widest window, as a cache outside a pool has.
fn cache(d: *cuda.Driver, g: geometry.Geometry) !caches.Cache {
    return .{ .state = try cuda.DeviceBuffer.alloc(d, g.stateBytes()), .kv = try cuda.DeviceBuffer.alloc(d, g.kvBytes(4096, false)), .log = try cuda.DeviceBuffer.alloc(d, g.logBytes(most)), .cap = 4096, .logits = &.{} };
}

fn release(c: *caches.Cache) void {
    c.state.free();
    c.kv.free();
    c.log.free();
    if (c.pbuf) |*b| b.free();
}

/// A recorded window of `most` rows, then `path` kept.
fn keepAfterWindow(e: *E, c: *caches.Cache, path: []const u32) !void {
    c.advanced(most, true, true);
    try c.keep(e, path);
}

test "Qwen pending keeps: a whole 64-row window kept lands at once, with no copy past the pending buffer" {
    var d = Mock.driver();
    var e = engine(&d);
    var c = try cache(&d, e.g);
    defer release(&c);
    var rows: [most]u32 = undefined;
    for (&rows, 0..) |*r, i| r.* = @intCast(i);
    try keepAfterWindow(&e, &c, &rows);
    try t.expectEqual(@as(usize, 0), Mock.copies);
    try t.expectEqual(@as(usize, 1), Mock.launches[apply]);
    try t.expectEqual(@as(c_int, 0), c.pend);
}

test "Qwen pending keeps: chains and spaced paths of 1 to 64 rows after 0 to 23 pending stay inside their buffers" {
    const late = caches.keepLate(); // the step sets TF_QWEN_KEEP_LATE=1; without it every keep lands at once
    for (1..most + 1) |n| for (0..caches.pend_commit) |before| for ([_]usize{ 1, 2 }) |step| {
        if ((n - 1) * step >= most) continue;
        var d = Mock.driver();
        var e = engine(&d);
        var c = try cache(&d, e.g);
        defer release(&c);
        var path: [most]u32 = undefined;
        for (path[0..@max(n, before)], 0..) |*r, i| r.* = @intCast(i);
        if (before > 0) try keepAfterWindow(&e, &c, path[0..before]);
        try t.expectEqual(@as(c_int, @intCast(if (late) before else 0)), c.pend);
        for (path[0..n], 0..) |*r, i| r.* = @intCast(i * step);
        try keepAfterWindow(&e, &c, path[0..n]);
        const pending = if (!late or n > caches.pend_cap) 0 else if (before + n > caches.pend_cap) n else before + n;
        try t.expectEqual(@as(c_int, @intCast(pending)), c.pend);
        const at_once = if (late) @intFromBool(n > caches.pend_cap) else @as(usize, @intFromBool(before > 0)) + 1;
        try t.expectEqual(@as(usize, at_once), Mock.launches[apply]);
    };
}
