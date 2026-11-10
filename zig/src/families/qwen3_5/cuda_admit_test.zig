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
    l.state_pieces = try pages.statePieces(t.allocator, &e, page);
    const lay = pages.layout(&e, capacity, page, 0, 0, l.state_pieces);
    // the plan's pages: a window and the state's pages for each stream beside the engine's own (all: for it too)
    const paged = parallel - 1 + @intFromBool(all);
    try native.attach(&l, parallel - 1, paged * (lay.pages(capacity + pages.window_rows) + lay.statePages()));
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

/// --parallel `parallel` with pages for `paged` streams: those admit, the rest wait for pages (--parallel a cap).
fn capped(comptime Lanes: type, parallel: u32, paged: u32) !usize {
    var d = D.driver();
    var e = engine(&d);
    var l: Lanes = .{
        .gpa = t.allocator,
        .io = undefined,
        .e = &e,
        .capacity = capacity,
        .page_tokens = page,
        .engine_paged = true,
        .target = undefined,
    };
    l.state_pieces = try pages.statePieces(t.allocator, &e, page);
    const lay = pages.layout(&e, capacity, page, 0, 0, l.state_pieces);
    try native.attach(&l, parallel - 1, paged * (lay.pages(capacity + pages.window_rows) + lay.statePages()));
    var open: std.ArrayList(*engine_mod.Cache) = .empty;
    defer {
        for (open.items) |x| _ = l.pages.?.close(x);
        open.deinit(t.allocator);
        l.pages.?.deinit();
    }
    while (open.items.len <= parallel) {
        if (!native.fits(&l, capacity - pages.window_rows, 16)) break;
        try open.append(t.allocator, try l.take(capacity - pages.window_rows, 16));
    }
    return open.items.len;
}

/// TF_QWEN_KV_PAGES set: --parallel caps the streams and a request waits for pages. Unset: no pool, nothing waits.
pub fn caps(comptime Lanes: type) !void {
    try t.expectEqual(@as(usize, 2), try capped(Lanes, 4, 2));
    try t.expectEqual(@as(usize, 4), try capped(Lanes, 4, 4));
    var d = D.driver();
    var e = engine(&d);
    var l: Lanes = .{ .gpa = t.allocator, .io = undefined, .e = &e, .capacity = capacity, .target = undefined };
    try t.expect(native.pagedLayout(&l, 0) == null); // no pool: the plan reserves a window for each of --parallel's
    try t.expect(native.fits(&l, capacity, 16));
}

/// The refusal `open` gives with these switches: no model at the path, so reaching the load names another error.
fn refused(slide: bool, kv_pages: ?[:0]const u8, share: ?[:0]const u8) !void {
    defer _ = env.unsetenv("TF_QWEN_KV_PAGES");
    defer _ = env.unsetenv("TF_QWEN_KV_SHARE");
    if (kv_pages) |v| _ = env.setenv("TF_QWEN_KV_PAGES", v, 1);
    if (share) |v| _ = env.setenv("TF_QWEN_KV_SHARE", v, 1);
    const o: native.Options = .{ .context = capacity, .drafts = false, .slide = slide };
    _ = try native.open(t.allocator, t.io, undefined, "/nonexistent", null, o);
}

/// A set switch's refusal needs no model: it stops before the weights load, as the other switch checks do.
pub fn refusals() !void {
    try t.expectError(error.QwenPagesNoLearner, refused(true, "256", null));
    try t.expectError(error.QwenShareNeedsPages, refused(false, null, "16"));
    try t.expectError(error.QwenShareNeedsPages, refused(false, "100", "16"));
}
