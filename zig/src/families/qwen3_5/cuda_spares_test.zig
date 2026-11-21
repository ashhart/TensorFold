//! Kept states' spare pinned buffers with a mock driver whose device pointers are host memory: copies move real bytes.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const reuse = @import("cuda_reuse.zig");
const disk = @import("cuda_disk.zig");
const spares = @import("cuda_spares.zig");
const native = @import("cuda_native.zig");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const abi = cuda.abi;
const t = std.testing;
const Copy = reuse.Copy;

/// Pinned memory from the testing allocator (a leak fails the test), copies that move bytes, a stream that can stick.
const Mock = struct {
    var live: [8]?[]align(16) u8 = @splat(null);
    var pins: usize = 0;
    var unpins: usize = 0;
    var copies: usize = 0;
    var fail_at: usize = std.math.maxInt(usize); // this copy and every later one fail
    var stuck = false;

    fn reset() void {
        pins = 0;
        unpins = 0;
        copies = 0;
        fail_at = std.math.maxInt(usize);
        stuck = false;
    }
    fn hostAlloc(p: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        const b = t.allocator.alignedAlloc(u8, .@"16", len) catch return 2;
        @memset(b, 0x5a); // new memory reads apart from any state's bytes
        for (&live) |*slot| if (slot.* == null) {
            slot.* = b;
            pins += 1;
            p.* = b.ptr;
            return abi.success;
        };
        t.allocator.free(b);
        return 2;
    }
    fn freeHost(p: ?*anyopaque) callconv(.c) abi.Result {
        for (&live) |*slot| if (slot.*) |b| if (@intFromPtr(b.ptr) == @intFromPtr(p)) {
            t.allocator.free(b);
            slot.* = null;
            unpins += 1;
            return abi.success;
        };
        return 1;
    }
    fn next() bool {
        copies += 1;
        return copies - 1 < fail_at;
    }
    fn dtoh(dst: ?*anyopaque, src: abi.DevicePtr, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (!next()) return abi.error_not_ready;
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..len], @as([*]const u8, @ptrFromInt(src))[0..len]);
        return abi.success;
    }
    fn htod(dst: abi.DevicePtr, src: ?*const anyopaque, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (!next()) return abi.error_not_ready;
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..len], @as([*]const u8, @ptrCast(src.?))[0..len]);
        return abi.success;
    }
    fn copy2d(m: *const abi.Memcpy2D, _: abi.Stream) callconv(.c) abi.Result {
        if (!next()) return abi.error_not_ready;
        const host: c_int = @intCast(abi.memory_type_host);
        const src: [*]const u8 = if (m.src_type == host) @ptrCast(m.src_host.?) else @ptrFromInt(m.src);
        const dst: [*]u8 = if (m.dst_type == host) @ptrCast(m.dst_host.?) else @ptrFromInt(m.dst);
        for (0..m.height) |h| @memcpy(dst[h * m.dst_pitch ..][0..m.width], src[h * m.src_pitch ..][0..m.width]);
        return abi.success;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return if (stuck) abi.error_not_ready else abi.success;
    }
    fn driver() cuda.Driver {
        var a: abi.Api = undefined;
        a.cuMemHostAlloc = hostAlloc;
        a.cuMemFreeHost = freeHost;
        a.cuMemcpyDtoHAsync_v2 = dtoh;
        a.cuMemcpyHtoDAsync_v2 = htod;
        a.cuMemcpy2DAsync_v2 = copy2d;
        a.cuStreamSynchronize = sync;
        return .{ .lib = undefined, .api = a };
    }
};

/// Three linear layers then attention, twice: the 27B's pattern at a test's size.
fn testGeometry() geometry.Geometry {
    var g: geometry.Geometry = undefined;
    g.hidden = 256;
    g.k_heads = 2;
    g.v_heads = 4;
    g.q_heads = 4;
    g.kv_heads = 2;
    g.layers = 8;
    g.first = 0; // one engine, not a pipeline stage
    g.whole = 0;
    @memset(&g.linear, false);
    for (0..g.layers) |i| g.linear[i] = i % 4 != 3;
    return g;
}

/// An engine on the mock driver, a cache whose state and keys/values (contiguous, or pages) are host memory.
const Fixture = struct {
    d: cuda.Driver,
    e: engine.Engine,
    cache: engine.Cache,
    state: []u8 = &.{},
    kv: []u8 = &.{},
    ids: [4]u32 = .{ 3, 0, 2, 1 },
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *engine.Cache) = .empty,
    seen: std.AutoHashMapUnmanaged(*lanes.Stream, []reuse.Seen) = .empty,
    t: reuse.Target = undefined,

    const page_tokens = 16;
    /// The stream the prompt cache names as a save's owner: only its address is read.
    const owner: *lanes.Stream = @ptrFromInt(0x100000);

    fn init(f: *Fixture, kv8: bool, paged: bool) !void {
        Mock.reset();
        f.* = .{ .d = Mock.driver(), .e = undefined, .cache = undefined };
        f.e.kit = null;
        f.e.gpa = t.allocator;
        f.e.d = &f.d;
        f.e.g = testGeometry();
        f.e.stream = .{ .d = &f.d, .handle = null };
        if (comptime @hasField(engine.Engine, "kv8")) f.e.kv8 = kv8 else if (kv8) return error.SkipZigTest;
        const c = &f.cache;
        c.cap = 64;
        c.len = 0;
        c.last = null;
        c.pend = 0;
        c.delta = 0;
        c.paged = null;
        f.state = try t.allocator.alloc(u8, f.e.g.stateBytes());
        f.kv = try t.allocator.alloc(u8, if (paged) f.ids.len * f.e.g.kvBytes(page_tokens, kv8) else f.e.g.kvBytes(c.cap, kv8));
        c.state = .{ .d = &f.d, .ptr = @intFromPtr(f.state.ptr), .len = f.state.len };
        c.kv = .{ .d = &f.d, .ptr = @intFromPtr(f.kv.ptr), .len = f.kv.len };
        if (paged) c.paged = .{ .base = @intFromPtr(f.kv.ptr), .table = .{ .d = &f.d, .ptr = 0, .len = 0 }, .ids = &f.ids, .tokens = page_tokens, .own_ids = false };
        try f.caches.put(t.allocator, owner, c);
        f.t = .{ .e = &f.e, .caches = &f.caches, .seen = &f.seen };
    }

    /// The spares freed with the engine, then the cache's memory: every pinned buffer is gone.
    fn deinit(f: *Fixture) void {
        f.t.spares.deinit();
        f.caches.deinit(t.allocator);
        t.allocator.free(f.state);
        t.allocator.free(f.kv);
    }

    /// The live state at `at` rows, every byte a function of `seed`.
    fn fill(f: *Fixture, at: usize, seed: u8) void {
        for (f.state, 0..) |*b, i| b.* = @truncate(i *% 31 +% seed);
        for (f.kv, 0..) |*b, i| b.* = @truncate(i *% 17 +% seed *% 3);
        f.cache.len = at;
    }

    fn save(f: *Fixture, at: usize) !*Copy {
        f.cache.len = at;
        return @ptrCast(@alignCast(try reuse.save(&f.t, owner, @intCast(at))));
    }

    fn drop(f: *Fixture, c: *Copy) void {
        reuse.drop(&f.t, c);
    }

    fn bytes(f: *Fixture, at: u32) usize {
        return @intCast(reuse.bytes(&f.t, at));
    }
};

test "Qwen kept spares: a drained copy's buffer goes to the next save of its size, which pins nothing" {
    var f: Fixture = undefined;
    try f.init(false, false);
    defer f.deinit();
    f.fill(10, 1);
    const first = try f.save(10);
    try t.expectEqual(@as(usize, 1), Mock.pins);
    try reuse.restoreCache(&f.e, &f.cache, first, &.{}); // the restore's wait drains its copies
    const buffer = first.host.bytes.ptr;
    f.drop(first);
    try t.expectEqual(@as(usize, 0), Mock.unpins);
    try t.expectEqual(@as(u64, f.bytes(10)), spares.spare(&f.t));
    try t.expect(spares.reuses(&f.t, 10) and !spares.reuses(&f.t, 11));
    const again = try f.save(10);
    try t.expectEqual(@as(usize, 1), Mock.pins);
    try t.expectEqual(buffer, again.host.bytes.ptr);
    try t.expectEqual(@as(u64, 0), spares.spare(&f.t));
    f.drop(again);
    f.t.spares.deinit();
    try t.expectEqual(Mock.pins, Mock.unpins);
}

test "Qwen kept spares: a buffer a copy may still use stays out of the list until a wait proves it drained" {
    const level = t.log_level;
    t.log_level = .err;
    defer t.log_level = level;
    var f: Fixture = undefined;
    try f.init(false, false);
    defer f.deinit();
    f.fill(10, 2);
    // a restore whose later copies fail on a stream that will not drain: its drop quarantines the buffer
    const read = try f.save(10);
    Mock.fail_at = Mock.copies + 2;
    Mock.stuck = true;
    try t.expectError(error.NotReady, reuse.restoreCache(&f.e, &f.cache, read, &.{}));
    const held = cuda.copy_source.retained();
    f.drop(read);
    try t.expectEqual(held + 1, cuda.copy_source.retained());
    try t.expect(f.t.spares.n == 0 and !spares.reuses(&f.t, 10));
    Mock.fail_at = std.math.maxInt(usize);
    Mock.stuck = false;
    const next = try f.save(10);
    try t.expectEqual(@as(usize, 2), Mock.pins); // new memory, not the buffer the stuck restore may read
    try f.e.stream.synchronize(); // a wait that drains: quarantine frees the old buffer, not the list
    try t.expectEqual(held, cuda.copy_source.retained());
    try t.expectEqual(@as(usize, 1), Mock.unpins);
    f.drop(next);
    // a save that took the spare and failed on a stuck stream: the buffer goes to quarantine, not back
    Mock.fail_at = Mock.copies + 1;
    Mock.stuck = true;
    try t.expectError(error.NotReady, f.save(10));
    try t.expect(f.t.spares.n == 0 and cuda.copy_source.retained() == held + 1);
    Mock.fail_at = std.math.maxInt(usize);
    Mock.stuck = false;
    try f.e.stream.synchronize();
    try t.expectEqual(@as(usize, 2), Mock.unpins);
    // a failed save whose stream drains gives its buffer back: the wait proved no copy into it is queued
    Mock.fail_at = Mock.copies + 1;
    try t.expectError(error.NotReady, f.save(10));
    Mock.fail_at = std.math.maxInt(usize);
    try t.expect(f.t.spares.n == 1 and spares.reuses(&f.t, 10));
    try t.expectEqual(@as(usize, 3), Mock.pins);
    f.drop(try f.save(10));
    try t.expectEqual(@as(usize, 3), Mock.pins);
}

test "Qwen kept spares: a save of another size pins a new buffer and leaves the spare for its own size" {
    var f: Fixture = undefined;
    try f.init(false, false);
    defer f.deinit();
    f.fill(12, 3);
    const ten = try f.save(10);
    const buffer = ten.host.bytes.ptr;
    f.drop(ten);
    var others: [2]*Copy = undefined;
    for (&others, [_]usize{ 12, 9 }) |*c, at| { // longer and shorter: neither takes a buffer of another size
        c.* = try f.save(at);
        try t.expect(c.*.host.bytes.ptr != buffer and c.*.host.bytes.len == f.bytes(@intCast(at)));
        try t.expect(f.t.spares.n == 1 and f.t.spares.fits(f.bytes(10)));
    }
    try t.expectEqual(@as(usize, 3), Mock.pins);
    const again = try f.save(10);
    try t.expectEqual(buffer, again.host.bytes.ptr);
    try t.expectEqual(@as(usize, 3), Mock.pins);
    for ([_]*Copy{ others[0], others[1], again }) |c| f.drop(c);
}

test "Qwen kept spares: the list keeps the newest two buffers and frees the oldest" {
    var f: Fixture = undefined;
    try f.init(false, false);
    defer f.deinit();
    f.fill(12, 4);
    var kept: [3]*Copy = undefined;
    for (&kept, [_]usize{ 10, 11, 12 }) |*c, at| c.* = try f.save(at);
    for (kept) |c| f.drop(c);
    try t.expectEqual(@as(usize, spares.Spares.cap), f.t.spares.n);
    try t.expectEqual(@as(usize, 1), Mock.unpins);
    try t.expect(!spares.reuses(&f.t, 10) and spares.reuses(&f.t, 11) and spares.reuses(&f.t, 12));
    try t.expectEqual(@as(u64, f.bytes(11) + f.bytes(12)), spares.spare(&f.t));
}

test "Qwen kept spares: trim frees the largest first down to the cache's room, and teardown frees every one" {
    var f: Fixture = undefined;
    try f.init(false, false);
    defer f.deinit();
    f.fill(12, 5);
    const a = try f.save(10);
    const b = try f.save(12);
    f.drop(a);
    f.drop(b);
    spares.trim(&f.t, f.bytes(12)); // both pass the room: the larger goes, the smaller fits it
    try t.expect(spares.reuses(&f.t, 10) and !spares.reuses(&f.t, 12));
    try t.expectEqual(@as(usize, 1), Mock.unpins);
    spares.trim(&f.t, f.bytes(10)); // within the room: nothing goes
    try t.expectEqual(@as(usize, 1), Mock.unpins);
    f.drop(try f.save(12));
    f.t.spares.deinit();
    try t.expectEqual(@as(u64, 0), spares.spare(&f.t));
    try t.expectEqual(Mock.pins, Mock.unpins);
}

test "Qwen kept spares: a save into a reused buffer holds a new buffer's bytes, and restores them, every layout" {
    for ([_][2]bool{ .{ false, false }, .{ true, false }, .{ false, true }, .{ true, true } }) |shape| {
        var f: Fixture = undefined;
        f.init(shape[0], shape[1]) catch |err| if (err == error.SkipZigTest) continue else return err;
        defer f.deinit();
        const at = 40; // two full pages and a part, under pages
        f.fill(at, 6);
        f.drop(try f.save(at));
        @memset(f.t.spares.bufs[0].bytes, 0xc3); // another state's bytes in every place: a byte a save skips shows
        f.fill(at, 7);
        const kept = try f.save(at);
        try t.expectEqual(@as(usize, 1), Mock.pins);
        const fresh = try reuse.saveCache(&f.e, &f.cache, at, &.{}, false, null);
        defer reuse.dropCopy(&f.e, fresh);
        try t.expectEqualSlices(u8, fresh.host.bytes[0..fresh.len], kept.host.bytes[0..kept.len]);
        const other = try f.save(at); // the list gave its one buffer away: a second state never shares it
        try t.expect(other.host.bytes.ptr != kept.host.bytes.ptr);
        f.fill(at, 8);
        f.drop(other);
        try reuse.restoreCache(&f.e, &f.cache, kept, &.{});
        f.cache.len = at;
        const back = try reuse.saveCache(&f.e, &f.cache, at, &.{}, false, null);
        defer reuse.dropCopy(&f.e, back);
        try t.expectEqualSlices(u8, fresh.host.bytes[0..fresh.len], back.host.bytes[0..back.len]);
        f.drop(kept);
    }
}

test "Qwen kept spares: a disk recall's blank takes a spare of its size, and a full read writes every byte" {
    var f: Fixture = undefined;
    f.init(true, false) catch |err| if (err == error.SkipZigTest) try f.init(false, false) else return err;
    defer f.deinit();
    const at = 40;
    f.fill(at, 9);
    const src = try f.save(at);
    defer f.drop(src);
    var parts: std.ArrayList(api.prompt_cache.rows.image.Part) = .empty;
    defer parts.deinit(t.allocator);
    try disk.partsOf(&f.e, at, &parts, t.allocator);
    const img = try t.allocator.alloc(u8, @intCast(api.prompt_cache.rows.image.imageBytes(parts.items)));
    defer t.allocator.free(img);
    try disk.image.gather(&f.t, src, &.{.{ .off = 0, .len = img.len }}, img);
    f.fill(at, 10);
    f.drop(try f.save(at)); // a spare of the recall's size, holding another state's bytes
    const pins = Mock.pins;
    const back: *Copy = @ptrCast(@alignCast(try disk.image.blank(&f.t, at)));
    try t.expectEqual(pins, Mock.pins);
    try disk.image.scatter(&f.t, back, &.{.{ .off = 0, .len = img.len }}, img);
    try t.expectEqualSlices(u8, src.host.bytes[0..src.len], back.host.bytes[0..back.len]);
    f.drop(back);
    const short: *Copy = @ptrCast(@alignCast(try disk.image.blank(&f.t, at - 1)));
    try t.expectEqual(pins + 1, Mock.pins); // another size pins its own
    f.drop(short);
}

test "Qwen kept spares: a full prompt cache's next state takes the buffer of the state it evicts" {
    var f: Fixture = undefined;
    try f.init(false, false);
    defer f.deinit();
    const at = 10;
    const pc = api.prompt_cache;
    const vtable: pc.Snapshots.VTable = .{ .bytes = native.snap_bytes, .save = native.snap_save, .restore = native.snap_restore, .drop = native.snap_drop, .spare = native.snap_spare, .trim = native.snap_trim, .reuses = native.snap_reuses };
    var store = pc.Store.init(t.allocator, .{ .ptr = &f.t, .vtable = &vtable }, .{ .lookahead = native.snap_lookahead }, 2 * f.bytes(at));
    var prompt: [at + 4]u32 = undefined;
    for (0..3) |i| { // three conversations, room for two
        for (&prompt, 0..) |*x, j| x.* = @intCast(1000 * (i + 1) + j);
        f.fill(at, @intCast(i));
        try t.expect(store.keep(&prompt, at, Fixture.owner, &.{}, &.{}));
    }
    try t.expectEqual(@as(usize, 2), store.entries.items.len);
    try t.expectEqual(@as(u64, 1), store.counts.evicted);
    try t.expectEqual(@as(usize, 2), Mock.pins);
    try t.expectEqual(@as(usize, 0), Mock.unpins);
    try t.expect(store.held + store.spare() <= store.budget);
    store.deinit(); // its states' buffers go to the list, freed with the engine
    try t.expectEqual(@as(usize, 0), Mock.unpins);
    f.t.spares.deinit();
    try t.expectEqual(@as(usize, 2), Mock.unpins);
}

test "Qwen kept spares: both Qwen entry points hand the prompt cache every spare hook" {
    inline for (.{ native, @import("cuda_native_moe.zig") }) |F| {
        try t.expect(@hasDecl(F, "snap_spare") and @hasDecl(F, "snap_trim") and @hasDecl(F, "snap_reuses"));
        try t.expect(&F.snap_trim == &spares.trim and &F.snap_reuses == &spares.reuses and &F.snap_spare == &spares.spare);
    }
}
