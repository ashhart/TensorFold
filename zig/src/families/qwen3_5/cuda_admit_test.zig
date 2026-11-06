//! Every stream on pages (TF_QWEN_KV_SHARE, TF_QWEN_KV_PAGES_ALL): the pool holds a lane for each of --parallel's.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const pages = @import("cuda_pages.zig");
const engine_mod = @import("cuda_engine.zig");
const E = engine_mod.Engine;
const D = cuda.paging_test_device.Device;
const t = std.testing;
const env = @import("sys").env;

const page = 256;
const capacity = 3 * 4096;

fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = D.stream };
    e.kv8 = false;
    e.n_taps = 0;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = capacity, .linear = @splat(false) };
    e.g.linear[0] = true;
    return e;
}

/// Streams admitted at once at --parallel `parallel`, the pool attached as the server attaches it.
fn admitted(comptime Lanes: type, parallel: u32, share: usize) !usize {
    var d = D.driver();
    var e = engine(&d);
    const all = share == 0; // TF_QWEN_KV_PAGES_ALL: the engine's own stream takes pages too, as open sets it
    var l: Lanes = .{ .gpa = t.allocator, .io = undefined, .e = &e, .capacity = capacity, .page_tokens = page, .engine_paged = all, .share_snaps = share, .target = undefined };
    const lay = pages.layout(&e, capacity, page, 0, 0);
    // the plan's pages: a window and a run for each stream beside the engine's own (all: for it too)
    const paged = parallel - 1 + @intFromBool(all);
    try native.attach(&l, parallel - 1, paged * (lay.pages(capacity + pages.window_rows) + lay.runPages()));
    try t.expect(l.paged_only);
    var open: std.ArrayList(*engine_mod.Cache) = .empty;
    defer {
        for (open.items) |x| _ = l.pages.?.close(x);
        open.deinit(t.allocator);
        if (l.prefix) |x| x.deinit();
        l.pages.?.deinit();
    }
    while (open.items.len <= parallel) {
        if (!native.fits(&l, 100, 16)) break;
        try open.append(t.allocator, try l.take(100, 16));
    }
    return open.items.len;
}

pub fn admits(comptime Lanes: type) !void {
    for ([_]u32{ 2, 8 }) |parallel| {
        try t.expectEqual(@as(usize, parallel), try admitted(Lanes, parallel, 4 * parallel)); // TF_QWEN_KV_SHARE
        _ = env.setenv("TF_QWEN_KV_PAGES_ALL", "1", 1);
        defer _ = env.unsetenv("TF_QWEN_KV_PAGES_ALL");
        try t.expectEqual(@as(usize, parallel), try admitted(Lanes, parallel, 0)); // the checks
    }
}
