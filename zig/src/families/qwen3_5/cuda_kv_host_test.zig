//! Host page geometry, allocation rollback and KV-only leases through the production constructors.
const std = @import("std");
const cuda = @import("cuda");
const host = @import("cuda_kv_host.zig");
const pages = @import("cuda_pages.zig");
const Storage = @import("cuda_pages_storage.zig").Storage;
const E = @import("cuda_engine.zig").Engine;
const D = cuda.paging_test_device.Device;
const t = std.testing;
const env = @import("sys").env;

fn mapped(out: *u64, ptr: ?*anyopaque, _: c_uint) callconv(.c) c_int {
    out.* = @intFromPtr(ptr.?);
    return 0;
}
fn noMap(_: *u64, _: ?*anyopaque, _: c_uint) callconv(.c) c_int {
    return cuda.abi.error_not_ready;
}
fn noDevice(_: *u64, _: usize) callconv(.c) c_int {
    return cuda.abi.error_not_ready;
}
fn noHost(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) c_int {
    return cuda.abi.error_not_ready;
}

pub fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = D.stream };
    e.kv8 = false;
    e.n_taps = 0;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 64, .linear = @splat(false) };
    e.g.linear[0] = true;
    return e;
}

test "kv safety: Qwen host KV frees pinned storage after failed mapping or device run allocation" {
    for (0..3) |failure| {
        var d = D.driver();
        d.api.cuMemHostGetDevicePointer_v2 = if (failure == 1) noMap else mapped;
        if (failure == 0) d.api.cuMemHostAlloc = noHost;
        if (failure == 2) d.api.cuMemAlloc_v2 = noDevice;
        const before = cuda.usage(false);
        const want = [_]anyerror{ error.QwenKvHostAllocation, error.QwenKvHostMapping, error.QwenKvHostRunAllocation };
        try t.expectError(want[failure], Storage.init(&d, 4096, 1024, true));
        try t.expectEqual(@as(usize, 0), D.hosts);
        try t.expectEqual(before.host, cuda.usage(false).host);
        try t.expectEqual(before.device, cuda.usage(false).device);
    }
}

test "kv safety: Qwen host KV pool contains only KV and maps its separately sized device run" {
    _ = env.setenv("TF_QWEN_KV_HOST", "1", 1);
    defer _ = env.unsetenv("TF_QWEN_KV_HOST");
    var d = D.driver();
    d.api.cuMemHostGetDevicePointer_v2 = mapped;
    var e = engine(&d);
    const l = pages.layout(&e, 64, 16, 0, 0);
    const count = l.pages(64 + pages.window_rows);
    const before = cuda.usage(false);
    const p = try pages.Pages.init(t.allocator, &e, l, 64, 0, 1, count);
    try t.expectEqual(@intFromPtr(p.host.?.bytes.ptr), p.base);
    for (p.host.?.bytes) |byte| try t.expectEqual(@as(u8, 0), byte);
    try t.expectEqual(count * l.pageBytes(), p.host.?.bytes.len);
    try t.expectEqual(l.runPages() * l.pageBytes(), p.run_dev.?.len);
    try t.expectEqual(count, p.pool.total());
    const c = try p.open(64, 0);
    try t.expectEqual(count, p.pool.census().leased);
    try t.expectEqual(@as(usize, 0), p.pool.census().free);
    try t.expect(p.close(c));
    try t.expect(p.pool.idle());
    p.deinit();
    try t.expectEqual(@as(usize, 0), D.hosts);
    try t.expectEqual(before.host, cuda.usage(false).host);
    try t.expectEqual(before.device, cuda.usage(false).device);
    try t.expectEqual(@as(usize, 0), D.freed_pending);
}

test "kv safety: Qwen host KV cap includes lookahead but no run pages and refuses unsupported geometry" {
    try t.expectEqual(@as(u64, 504), try host.bytes(129, 16, 56, null));
    try t.expectError(error.QwenKvHostCap, host.bytes(129, 16, 56, 503));
    try t.expectEqual(@as(u64, 504), try host.bytes(129, 16, 56, 504));
    try t.expectError(error.QwenKvHostNeedsPages, host.bytes(128, 0, 56, null));
    try t.expectError(error.Overflow, host.bytes(129, 16, std.math.maxInt(u64), null));
    try t.expectError(error.QwenKvHostNeedsPages, host.checkDevice(false, 0, false));
    try t.expectError(error.QwenKvHostNeedsPagedEngine, host.checkDevice(false, 16, false));
    try t.expectError(error.QwenKvHostIntegrated, host.checkDevice(true, 16, true));
    try host.checkDevice(false, 16, true);
    try host.checkShared(0);
    try t.expectError(error.QwenKvHostShare, host.checkShared(1));
}
