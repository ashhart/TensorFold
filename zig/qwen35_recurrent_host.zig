//! Actual CUDA cache allocation and keep calls with address-only driver stubs, never a GPU or model payload.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const a = std.testing.allocator;
const rec = q.engine.recurrent;

test {
    _ = @import("qwen35_recurrent_budget_host.zig");
    _ = @import("qwen35_recurrent_rows_host.zig");
}

const Driver = struct {
    const stride: u64 = 1 << 32;
    var allocs: usize = 0;
    var frees: usize = 0;
    var bytes: usize = 0;
    var live: usize = 0;
    var sizes: [65536]usize = @splat(0);
    var alive: [65536]bool = @splat(false);
    var pending_src: u64 = 0;
    var pending_dst: u64 = 0;
    var copy_bytes: usize = 0;
    var unsafe_frees: usize = 0;
    var syncs: usize = 0;
    var fail_launch: bool = false;
    var fail_copy: bool = false;
    fn reset() void {
        std.debug.assert(live == 0);
        allocs = 0;
        frees = 0;
        bytes = 0;
        pending_src = 0;
        pending_dst = 0;
        unsafe_frees = 0;
        syncs = 0;
        fail_launch = false;
        fail_copy = false;
    }
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        std.debug.assert(allocs < sizes.len and n < stride);
        out.* = 4096 + @as(u64, @intCast(allocs)) * stride;
        sizes[allocs] = n;
        alive[allocs] = true;
        allocs += 1;
        bytes += n;
        live += n;
        return cuda.abi.success;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        const i: usize = @intCast((ptr - 4096) / stride);
        std.debug.assert(alive[i]);
        if ((pending_src >= ptr and pending_src < ptr + sizes[i]) or (pending_dst >= ptr and pending_dst < ptr + sizes[i])) unsafe_frees += 1;
        alive[i] = false;
        live -= sizes[i];
        frees += 1;
        return cuda.abi.success;
    }
    fn fill(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn fillAsync(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        pending_src = 0;
        pending_dst = 0;
        return cuda.abi.success;
    }
    fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending_src = src;
        pending_dst = dst;
        copy_bytes = n;
        return cuda.abi.success;
    }
    fn copy2D(m: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_copy) return cuda.abi.error_not_ready;
        if (!inside(m.src, (m.height - 1) * m.src_pitch + m.width) or !inside(m.dst, (m.height - 1) * m.dst_pitch + m.width)) return 1;
        pending_src = m.src;
        pending_dst = m.dst;
        copy_bytes = m.width * m.height;
        return cuda.abi.success;
    }
    fn inside(ptr: u64, n: usize) bool {
        for (0..allocs) |i| {
            const at = 4096 + @as(u64, @intCast(i)) * stride;
            if (alive[i] and ptr >= at and ptr + n <= at + sizes[i]) return true;
        }
        return false;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return if (fail_launch) cuda.abi.error_not_ready else cuda.abi.success;
    }
};
fn engine(e: *q.engine.Engine, d: *cuda.Driver) void {
    d.api.cuMemAlloc_v2 = Driver.alloc;
    d.api.cuMemFree_v2 = Driver.free;
    d.api.cuMemsetD8_v2 = Driver.fill;
    d.api.cuMemsetD8Async = Driver.fillAsync;
    d.api.cuStreamSynchronize = Driver.sync;
    d.api.cuMemcpyDtoDAsync_v2 = Driver.copy;
    d.api.cuMemcpy2DAsync_v2 = Driver.copy2D;
    d.api.cuLaunchKernel = Driver.launch;
    e.d = d;
    e.stream = .{ .d = d, .handle = @ptrFromInt(17) };
    e.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
    for (&e.g.linear, 0..) |*linear, i| linear.* = i < 64 and (i + 1) % 4 != 0;
    e.n_taps = 0;
    e.kv8 = false;
    e.geo = .{ .hid = 5120, .nk = 16, .nv = 48, .qh = 24, .kvh = 4, .nl = 48 };
    const f: cuda.Function = .{ .d = d, .handle = @ptrFromInt(1) };
    e.keeps = .{ .apply = f, .apply_pend = f, .conv = f, .compact = f };
    e.rec_scratch = rec.Pool.init(a, @intFromPtr(e));
    if (comptime @hasField(q.engine.Engine, "rec_budget")) e.rec_budget = .{};
}
fn finish(e: *q.engine.Engine, c: *q.engine.Cache) void {
    _ = Driver.sync(e.stream.handle);
    rec.completed(e);
    c.deinit(a);
    rec.shutdown(e);
}
fn forwardDone(e: *q.engine.Engine, c: *q.engine.Cache, rows: usize) !void {
    try rec.mark(c, false);
    if (rows == 1) try rec.mark(c, true);
    c.advanced(rows, true);
    _ = Driver.sync(e.stream.handle);
    rec.completed(e);
}

test "Shared verification: recurrent scratch is absent from an idle real 27B cache" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 1);
    defer finish(&e, &c);
    try std.testing.expectEqual(@as(usize, 2), Driver.allocs);
    try std.testing.expectEqual(e.g.stateBytes() + e.g.kvBytes(1, false), Driver.bytes);
    try std.testing.expect(c.before == null);
    try std.testing.expectEqual(@as(usize, 0), c.log.ptr);
}

test "Shared verification: all window transitions keep log capacity within the original byte bound" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 128);
    defer finish(&e, &c);
    const bound = e.g.stateBytes() + e.g.logBytes(64);
    try std.testing.expectEqual(@as(usize, 153944064), e.g.stateBytes());
    for (1..65) |old| for (1..65) |rows| {
        try c.reset(&e);
        rec.completed(&e);
        try rec.prepare(&e, &c, old, true);
        try forwardDone(&e, &c, old);
        if (old == 1) try c.keep(&e, &.{}) else try c.keep(&e, &.{0});
        try rec.prepare(&e, &c, rows, true);
        try std.testing.expect(c.log.len >= e.g.logBytes(rows));
        try std.testing.expect((c.before != null) == (rows == 1));
        try std.testing.expect(e.rec_scratch.residentBytes() <= bound);
        const sg = c.seg(&e, c.len, 0, rows);
        try std.testing.expectEqual(c.log.ptr, sg.log);
        try std.testing.expectEqual(if (c.before) |b| b.ptr else 0, sg.before);
        try forwardDone(&e, &c, rows);
        if (rows == 1) {
            try c.settle(&e);
            try rec.prepare(&e, &c, 1, true);
            try std.testing.expectEqual(e.g.logBytes(1), c.log.len);
        }
    };
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
}

test "Shared verification: a real empty keep retains its snapshot until the queued restore completes" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 128);
    defer finish(&e, &c);
    try rec.prepare(&e, &c, 1, true);
    try forwardDone(&e, &c, 1);
    const before = c.before.?.ptr;
    try c.keep(&e, &.{});
    try std.testing.expectEqual(before, Driver.pending_src);
    try std.testing.expectEqual(c.state.ptr, Driver.pending_dst);
    try std.testing.expectEqual(e.g.stateBytes(), Driver.copy_bytes);
    rec.release(&c);
    try e.rec_scratch.collect(@intFromPtr(&e), rec.Provider{ .e = &e });
    try std.testing.expectEqual(e.g.stateBytes(), e.rec_scratch.residentBytes());
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    const syncs = Driver.syncs;
    rec.idle(&e);
    try std.testing.expectEqual(syncs + 1, Driver.syncs);
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
}

test "Shared verification: allocation refusal retains a queued keep's old log owner" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 128);
    defer finish(&e, &c);
    var foreign: q.engine.Engine = undefined;
    foreign.kit = null;
    var other_driver: cuda.Driver = undefined;
    engine(&foreign, &other_driver);
    defer rec.shutdown(&foreign);
    try std.testing.expectError(error.ForeignRecurrentQueue, rec.prepare(&foreign, &c, 8, true));
    try std.testing.expectError(error.InvalidRecurrentWindow, rec.prepare(&e, &c, 65, true));
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
    try rec.prepare(&e, &c, 64, true);
    try forwardDone(&e, &c, 64);
    try c.keep(&e, &.{0});
    const usage = cuda.memory.usage(false);
    const prior = try cuda.setAllocationLimit(usage.device + usage.host);
    defer _ = cuda.setAllocationLimit(prior) catch unreachable;
    try std.testing.expectError(error.OutOfDeviceMemory, rec.prepare(&e, &c, 32, true));
    try std.testing.expectEqual(e.g.logBytes(64), e.rec_scratch.residentBytes());
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    _ = Driver.sync(e.stream.handle);
    rec.completed(&e);
    try std.testing.expectEqual(e.g.logBytes(64), e.rec_scratch.residentBytes()); // completed: kept for another window
    rec.idle(&e);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
}

test "Shared verification: streams cycling window widths reuse their scratch instead of freeing it each round" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 1024);
    var o = try q.engine.Cache.init(a, &e, 1024);
    defer {
        _ = Driver.sync(e.stream.handle);
        rec.completed(&e);
        o.deinit(a);
        c.deinit(a);
        rec.shutdown(&e);
    }
    try std.testing.expectEqual(2 * rec.bound(&e), e.rec_scratch.cap);
    const widths = [_]usize{ 1, 8, 3, 16, 1, 5 };
    var settled: usize = 0;
    for (0..4) |cycle| {
        for (widths, 0..) |rows, i| {
            const other = widths[(i + 3) % widths.len];
            try rec.prepare(&e, &c, rows, true);
            try rec.prepare(&e, &o, other, true);
            try forwardDone(&e, &c, rows);
            try rec.mark(&o, false);
            if (other == 1) try rec.mark(&o, true);
            o.advanced(other, true);
            try std.testing.expect(e.rec_scratch.residentBytes() <= e.rec_scratch.cap);
        }
        if (cycle == 0) settled = Driver.allocs;
    }
    try std.testing.expectEqual(settled, Driver.allocs); // after one cycle no round allocates
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
}

fn paged(e: *q.engine.Engine) !q.engine.Cache {
    if (q.cache.keepLate()) return error.SkipZigTest;
    if (!rec.needsFullKeep(e)) try rec.configure(e, 4);
    var c = try q.engine.Cache.initPagedOwned(a, e, 16, 256, 0x80000000);
    errdefer c.deinit(a);
    try c.mapRun(e, 0x90000000, q.engine.Cache.runBytes(e, 0), 0);
    c.cap = 128;
    return c;
}

test "Shared verification: paged runs keep exact state but lease actual-sized before and logs" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try paged(&e);
    defer finish(&e, &c);
    try std.testing.expectEqual(e.g.stateBytes(), q.engine.Cache.runBytes(&e, 0));
    try std.testing.expect(c.owner == &e.rec_scratch and c.pool_run);
    try std.testing.expectEqual(rec.sharedBytes(&e), e.rec_scratch.cap);
    try std.testing.expectEqual(@as(usize, 0), c.scratch);
    try std.testing.expect(c.before == null and c.log.ptr == 0);
    try std.testing.expectEqual(@as(usize, 1), Driver.allocs); // only the page table, no borrowed run allocation
    for ([_]usize{ 1, 8, 3, 16, 64 }) |rows| {
        try c.reset(&e);
        try rec.prepare(&e, &c, rows, true);
        try std.testing.expect(c.log.len >= e.g.logBytes(rows));
        try std.testing.expectEqual(rows == 1, c.before != null);
        try std.testing.expect(c.log.ptr != c.state.ptr);
        const sg = c.seg(&e, 0, 0, rows);
        try std.testing.expectEqual(c.log.ptr, sg.log);
        try std.testing.expectEqual(if (c.before) |b| b.ptr else @as(u64, 0), sg.before);
        try forwardDone(&e, &c, rows);
        try c.keep(&e, if (rows == 1) &.{} else &.{0});
    }
    c.unmapRun(&e);
    try std.testing.expect(c.before == null and c.log.ptr == 0);
    rec.idle(&e);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
}

test "Shared verification: paged empty keep retains the before copy through unmapping until completion" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try paged(&e);
    defer finish(&e, &c);
    try rec.prepare(&e, &c, 1, true);
    try forwardDone(&e, &c, 1);
    const before = c.before.?.ptr;
    try c.keep(&e, &.{});
    try std.testing.expectEqual(before, Driver.pending_src);
    c.unmapRun(&e);
    try e.rec_scratch.collect(@intFromPtr(&e), rec.Provider{ .e = &e });
    try std.testing.expectEqual(e.g.stateBytes(), e.rec_scratch.residentBytes());
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    rec.idle(&e);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
}

test "Shared verification: paged keep failure retains its log and hard budget refuses live overcommit" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try paged(&e);
    defer finish(&e, &c);
    try rec.prepare(&e, &c, 64, true);
    try forwardDone(&e, &c, 64);
    const pointer = c.log.ptr;
    Driver.fail_launch = true;
    try std.testing.expectError(error.NotReady, c.keep(&e, &.{0}));
    Driver.fail_launch = false;
    try std.testing.expect(c.last != null and c.log.ptr == pointer and c.log_lease != null);
    const cap = e.rec_scratch.cap;
    const limit = e.rec_budget.limit;
    e.rec_scratch.cap = c.log.len;
    e.rec_budget.limit = c.log.len;
    defer e.rec_scratch.cap = cap;
    defer e.rec_budget.limit = limit;
    const allocations = Driver.allocs;
    try std.testing.expectError(error.RecurrentScratchBudget, (rec.PagedProvider{ .e = &e }).allocate(1));
    try std.testing.expectEqual(allocations, Driver.allocs);
    try std.testing.expectEqual(pointer, c.log.ptr);
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
}

test "Shared verification: paged size miss drains retired bytes before allocating past its cap" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try paged(&e);
    defer finish(&e, &c);
    try rec.prepare(&e, &c, 1, true);
    try forwardDone(&e, &c, 1);
    try c.keep(&e, &.{});
    const old_cap = e.rec_scratch.cap;
    const old_limit = e.rec_budget.limit;
    e.rec_scratch.cap = e.g.logBytes(64);
    e.rec_budget.limit = e.g.logBytes(64);
    defer e.rec_scratch.cap = old_cap;
    defer e.rec_budget.limit = old_limit;
    const waits = Driver.syncs;
    try rec.prepare(&e, &c, 64, true);
    try std.testing.expect(Driver.syncs > waits);
    try std.testing.expect(e.rec_scratch.residentBytes() <= e.rec_scratch.cap);
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
}

test "Shared verification: one-stream paged mode retains its original fixed run and no leases" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    try rec.configure(&e, 1);
    var c = try q.engine.Cache.initPagedOwned(a, &e, 16, 256, 0x80000000);
    defer finish(&e, &c);
    const before_at = e.g.logBytes(1);
    const run_bytes = e.g.stateBytes() + @max(e.g.logBytes(64), before_at + e.g.stateBytes());
    try std.testing.expectEqual(run_bytes, q.engine.Cache.runBytes(&e, 0));
    try c.mapRun(&e, 0x90000000, run_bytes, 0);
    try std.testing.expect(c.owner == null and !c.pool_run);
    try std.testing.expect(c.before != null and c.before.?.len == e.g.stateBytes());
    try std.testing.expectEqual(c.log.ptr + before_at, c.before.?.ptr);
    try std.testing.expectEqual(e.g.logBytes(64), c.log.len);
    try rec.prepareRoot(&e, &c, 1, true, true);
    try std.testing.expect(c.before != null and !c.before_skipped);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
    try std.testing.expectEqual(@as(usize, 1), Driver.allocs);
}

test "Shared verification: global admission reserves the exact committed-root frame" {
    if (q.cache.keepLate()) return error.SkipZigTest;
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    defer rec.shutdown(&e);
    try rec.configure(&e, 16);
    try std.testing.expectEqual(@as(usize, 1057640448), rec.sharedBytes(&e));
    try std.testing.expect(e.g.stateBytes() + e.g.logBytes(1) <= rec.sharedBytes(&e));
}

test "Shared verification: committed roots bound sixteen windows and the widest log frame" {
    if (q.cache.keepLate()) return error.SkipZigTest;
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    try rec.configure(&e, 16);
    try std.testing.expectEqual(@as(usize, 1057640448), rec.sharedBytes(&e));
    try std.testing.expectEqual(@as(usize, 0), rec.pagedBytes(&e));
    try std.testing.expectError(error.RecurrentBudgetBusy, rec.configure(&e, 8));
    var caches: [16]q.engine.Cache = undefined;
    var made: usize = 0;
    defer {
        _ = Driver.sync(e.stream.handle);
        rec.completed(&e);
        for (caches[0..made]) |*c| c.deinit(a);
        rec.shutdown(&e);
    }
    for (&caches, 0..) |*c, i| {
        c.* = try q.engine.Cache.initPagedOwned(a, &e, 16, 256, 0x80000000);
        made += 1;
        try c.mapRun(&e, 0x10000000 + i * e.g.stateBytes(), q.engine.Cache.runBytes(&e, 0), 0);
        c.cap = 128;
        try rec.prepareRoot(&e, c, 1, true, true);
        try std.testing.expect(c.before == null and c.before_lease == null and c.before_skipped);
        try rec.mark(c, false);
        c.advanced(1, true);
    }
    try std.testing.expectEqual(16 * e.g.logBytes(1), e.rec_scratch.residentBytes());
    _ = Driver.sync(e.stream.handle);
    rec.completed(&e);
    for (&caches) |*c| {
        try c.keep(&e, &.{0});
        try std.testing.expect(!c.before_skipped and c.log_lease == null);
    }
    rec.idle(&e);
    const wide = [_]usize{ 3, 9, 17, 33, 33, 33 };
    for (caches[0..wide.len], wide) |*c, rows| {
        try rec.prepareRoot(&e, c, rows, true, true);
        try rec.mark(c, false);
        c.advanced(rows, true);
    }
    try std.testing.expectEqual(rec.sharedBytes(&e), e.rec_scratch.residentBytes());
    _ = Driver.sync(e.stream.handle);
    rec.completed(&e);
    for (caches[0..wide.len]) |*c| {
        try c.keep(&e, &.{ 0, 1 });
        try std.testing.expect(c.log_lease == null and c.before_lease == null);
    }
    try e.rec_scratch.collect(@intFromPtr(&e), rec.Provider{ .e = &e });
    try std.testing.expect(e.rec_scratch.residentBytes() > 0);
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    rec.idle(&e);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
}

test "Shared verification: only explicit global committed roots skip the before lease" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try paged(&e);
    defer finish(&e, &c);
    try rec.prepare(&e, &c, 1, true);
    try std.testing.expect(c.before != null and !c.before_skipped);
    try rec.prepareRoot(&e, &c, 1, true, true);
    try std.testing.expect(c.before == null and c.before_skipped);
    try std.testing.expectEqual(@as(u64, 0), c.seg(&e, 0, 0, 1).before);
    try forwardDone(&e, &c, 1);
    const len = c.len;
    try std.testing.expectError(error.QwenCommittedRootCannotRollback, c.keep(&e, &.{}));
    try std.testing.expect(c.last != null and c.before_skipped and c.len == len);
    try c.keep(&e, &.{0});
    try std.testing.expect(!c.before_skipped and c.last == null);
    try rec.prepareRoot(&e, &c, 1, true, true);
    try c.reset(&e);
    try std.testing.expect(!c.before_skipped);
    try rec.prepareRoot(&e, &c, 1, true, true);
    c.unmapRun(&e);
    try std.testing.expect(!c.before_skipped);
}

test "Shared verification: global private frames retain rollback and refuse unbudgeted before banks" {
    if (q.cache.keepLate()) return error.SkipZigTest;
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    defer rec.shutdown(&e);
    try rec.configure(&e, 16);
    var parts: [16]q.engine.Part = undefined;
    for (&parts) |*p| p.* = .{ .cache = undefined, .rows = 1 };
    try std.testing.expect(!parts[0].root_committed);
    for (1..65) |rows| {
        parts[0].rows = rows;
        try rec.admitFrame(&e, parts[0..1], true);
    }
    parts[0].rows = 1;
    try std.testing.expectError(error.RecurrentScratchBudget, rec.admitFrame(&e, &parts, true));
    for (&parts) |*p| p.root_committed = true;
    try rec.admitFrame(&e, &parts, true);
    for (parts[0..3]) |*p| p.rows = 64;
    try std.testing.expectError(error.InvalidRecurrentWindow, rec.admitFrame(&e, &parts, true));
    try std.testing.expectEqual(@as(usize, 0), Driver.allocs);
}

test "Shared verification: deferred keeps retain legacy paged allocation and preserve failed copy ownership" {
    if (!q.cache.keepLate()) return error.SkipZigTest;
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    try rec.configure(&e, 16);
    try std.testing.expect(!rec.needsFullKeep(&e));
    try std.testing.expectEqual(@as(usize, 0), rec.sharedBytes(&e));
    const bytes = 2 * e.g.stateBytes() + e.g.logBytes(64);
    try std.testing.expectEqual(bytes, q.engine.Cache.runBytes(&e, 0));
    var run = try cuda.DeviceBuffer.alloc(&d, bytes);
    defer run.free();
    var c = try q.engine.Cache.initPagedOwned(a, &e, 16, 256, 0x80000000);
    defer finish(&e, &c);
    try c.mapRun(&e, run.ptr, run.len, 0);
    try std.testing.expect(c.owner == null and !c.pool_run);
    try rec.prepare(&e, &c, 8, true);
    try forwardDone(&e, &c, 8);
    const log = c.log.ptr;
    Driver.fail_copy = true;
    try std.testing.expectError(error.NotReady, c.keep(&e, &.{ 0, 1 }));
    Driver.fail_copy = false;
    try std.testing.expect(c.last != null and c.log.ptr == log);
    try std.testing.expectEqual(@as(c_int, 0), c.pend);
    try c.keep(&e, &.{ 0, 1 });
    try std.testing.expect(c.last == null and c.log.ptr == log);
    try std.testing.expectEqual(@as(c_int, 2), c.pend);
    try std.testing.expectEqual(log, Driver.pending_src);
    rec.idle(&e);
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
}
