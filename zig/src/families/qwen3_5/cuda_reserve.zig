//! Target reservation shared with CPU admission controls.
const warm = @import("cuda_warm.zig");
const E = @import("cuda_engine.zig");

/// `streams` windows reserved and warmed; with a page pool (TF_QWEN_KV_PAGES) the first stream's window only, the
/// shared shapes warmed on lanes opened for one row and closed again.
pub fn target(comptime Cache: type, comptime Part: type, l: anytype, streams: u32) !void {
    const started = warm.profile.now(l.io);
    var warm_ns: i96 = 0;
    const paged = comptime @hasField(@TypeOf(l.*), "pages");
    const pooled = if (paged) l.pages != null else false;
    const windows: u32 = if (pooled) @min(streams, 1) else streams;
    try l.pool.ensureTotalCapacity(l.gpa, windows);
    while (l.pool.items.len < windows) {
        const cache = try l.gpa.create(Cache);
        errdefer l.gpa.destroy(cache);
        cache.* = try Cache.init(l.gpa, l.e, l.capacity);
        errdefer cache.deinit(l.gpa);
        // a row through it: kernels have to have referenced a buffer once before the driver's stall is behind it
        const warm_start = warm.profile.now(l.io);
        try l.e.forward(&.{.{ .cache = cache, .rows = 1 }}, &.{0}, .last);
        try l.e.stream.synchronize();
        warm_ns += warm.profile.elapsed(l.io, warm_start);
        try cache.reset(l.e);
        l.pool.appendAssumeCapacity(cache);
    }
    // Warm each shared launch shape once; every cache above is still allocated and touched separately.
    var parts: [E.max_segs]Part = undefined;
    var ids: [@max(E.max_segs, E.most_window)]u32 = @splat(0);
    var picks: [@max(E.max_segs, E.most_window)]E.Draw = undefined;
    var warming: [E.max_segs]*Cache = undefined;
    var opened: usize = 0;
    const most = @min(streams, E.max_segs);
    for (l.pool.items[0..@min(windows, most)], 0..) |cache, i| warming[i] = cache;
    if (comptime paged) if (l.pages) |p| {
        while (windows + opened < most) : (opened += 1) warming[windows + opened] = p.open(1, 0) catch break;
    };
    defer if (comptime paged) if (l.pages) |p| for (warming[windows..][0..opened]) |cache| {
        _ = p.close(cache);
    };
    try warmShapes(Part, l, warming[0 .. @min(windows, most) + opened], &parts, &ids, &picks, &warm_ns);
    // Pending calibration warms on its capacity-safe cache during synchronous open.
    if (l.e.n_taps > 0 and l.calibration == null) for (2..l.e.window_rows + 1) |rows| {
        const shape = E.Engine.Shape.init(rows, 1, true, true, 1);
        if (l.warmed.contains(shape)) continue;
        for (picks[0..rows]) |*pk| pk.* = .{ .rule = null, .position = 1 };
        const warm_start = warm.profile.now(l.io);
        try l.e.forward(&.{.{ .cache = l.pool.items[0], .rows = rows }}, ids[0..rows], .{ .draws = picks[0..rows] });
        try l.e.stream.synchronize();
        warm_ns += warm.profile.elapsed(l.io, warm_start);
        try l.pool.items[0].reset(l.e);
        l.warmed.mark(shape);
    };
    if (comptime @hasField(@TypeOf(l.e.*), "rec_scratch")) E.recurrent.idle(l.e);
    l.startup.warmups += warm_ns;
    l.startup.reserve += warm.profile.elapsed(l.io, started) - warm_ns;
}

/// Each shape of 1..caches.len one-row segments run once on `caches`, then reset.
fn warmShapes(comptime Part: type, l: anytype, caches: anytype, parts: []Part, ids: []u32, picks: []E.Draw, warm_ns: *i96) !void {
    for (caches, 0..) |cache, n| {
        parts[n] = .{ .cache = cache, .rows = 1 };
        picks[n] = .{ .rule = null, .position = 1 };
        const shape = E.Engine.Shape.init(n + 1, n + 1, true, l.e.n_taps > 0, 1);
        if (l.warmed.contains(shape)) continue;
        const warm_start = warm.profile.now(l.io);
        try l.e.forward(parts[0 .. n + 1], ids[0 .. n + 1], .{ .draws = picks[0 .. n + 1] });
        try l.e.stream.synchronize();
        warm_ns.* += warm.profile.elapsed(l.io, warm_start);
        for (caches[0 .. n + 1]) |c| try c.reset(l.e);
        l.warmed.mark(shape);
    }
}

pub const Test = if (@import("builtin").is_test) @import("cuda_reserve_test.zig") else struct {};
