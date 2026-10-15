//! A stream's state on a runtime over host memory: restores failing part way, restores agreed over loopback, the pool.
const std = @import("std");
const hip = @import("hip");
const cache = @import("cache.zig");
const model = @import("model.zig");
const snapshot = @import("snapshot.zig");

const abi = hip.abi;
const eos: i64 = 99;
const ratio = 4;

/// HIP over host memory; each thread may fail its copies in after a count, its fills, or its syncs.
const Host = struct {
    /// Copies in that go through before one fails (null: none fails).
    threadlocal var copies: ?usize = null;
    threadlocal var fail_fill = false;
    threadlocal var fail_sync = false;
    threadlocal var frees: usize = 0;

    fn htod(dst: abi.DevicePtr, src: [*]const u8, n: usize) callconv(.c) abi.Result {
        if (copies) |*left| {
            if (left.* == 0) return 1;
            left.* -= 1;
        }
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], src[0..n]);
        return 0;
    }
    fn dtoh(dst: [*]u8, src: abi.DevicePtr, n: usize) callconv(.c) abi.Result {
        @memcpy(dst[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn set(dst: abi.DevicePtr, value: c_int, n: usize) callconv(.c) abi.Result {
        if (fail_fill) return 1;
        @memset(@as([*]u8, @ptrCast(dst.?))[0..n], @intCast(value));
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return @intFromBool(fail_sync);
    }
    fn malloc(p: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
        p.* = std.c.malloc(n) orelse return 1;
        return 0;
    }
    fn free(p: abi.DevicePtr) callconv(.c) abi.Result {
        frees += 1;
        std.c.free(p);
        return 0;
    }

    fn runtime() hip.runtime.Runtime {
        var r: hip.runtime.Runtime = undefined;
        r.api.hipMemcpyHtoD = htod;
        r.api.hipMemcpyDtoH = dtoh;
        r.api.hipMemset = set;
        r.api.hipStreamSynchronize = sync;
        r.api.hipMalloc = malloc;
        r.api.hipFree = free;
        return r;
    }
};

/// Two sparse layers of two kv heads around one DeltaNet layer, small enough to check byte by byte.
fn dims() model.Dims {
    return .{
        .hidden = 8,
        .streams = 2,
        .low = 2,
        .vocab = 16,
        .layers = 3,
        .eps = 1e-6,
        .heads = 2,
        .kv_heads = 2,
        .head_dim = 4,
        .rotary = 2,
        .theta = 10000,
        .key_heads = 1,
        .value_heads = 1,
        .key_dim = 2,
        .value_dim = 2,
        .conv = 3,
        .index_heads = 1,
        .index_dim = 4,
        .index_ratio = ratio,
        .index_top = 2,
        .ple_dim = 8,
        .ple_taps = 2,
        .ngram = 2,
        .vocab_rows = 16,
        .gate = .sigmoid,
    };
}

const kinds = [_]bool{ false, true, false };
const cap = 128;
const at = 64;

fn caches(r: *const hip.runtime.Runtime) !cache.Caches {
    return cache.Caches.init(std.testing.allocator, r, dims(), &kinds, cap, .bf16, eos);
}

fn bytes(c: *const cache.Caches) []u8 {
    return @as([*]u8, @ptrCast(c.mem.ptr.?))[0..c.mem.len];
}

/// A stream's main caches and head with `at` positions of patterned state, and its copy.
const Saved = struct {
    main: cache.Caches,
    head: cache.Caches,
    copy: *snapshot.Saved,

    fn init(r: *const hip.runtime.Runtime, seed: u8) !Saved {
        var main = try caches(r);
        errdefer main.deinit(std.testing.allocator);
        var head = try caches(r);
        errdefer head.deinit(std.testing.allocator);
        for (bytes(&main), 0..) |*b, i| b.* = @truncate(i *% 7 +% seed);
        for (bytes(&head), 0..) |*b, i| b.* = @truncate(i *% 5 +% seed);
        main.len = at;
        main.pooled = at / ratio;
        main.history = .{ 5, 6 };
        head.len = at;
        const s: hip.Stream = .{ .r = r, .handle = null };
        const copy = try snapshot.save(std.testing.allocator, r, s, &main, &head, at, ratio);
        return .{ .main = main, .head = head, .copy = copy };
    }

    fn deinit(s: *Saved) void {
        snapshot.drop(std.testing.allocator, s.copy);
        s.head.deinit(std.testing.allocator);
        s.main.deinit(std.testing.allocator);
    }
};

/// Caches are a new stream's: every byte zero, no positions, eos before any row.
fn expectFresh(c: *const cache.Caches) !void {
    for (bytes(c)) |b| try std.testing.expectEqual(@as(u8, 0), b);
    try std.testing.expectEqual(@as(usize, 0), c.len);
    try std.testing.expectEqual(@as(usize, 0), c.pooled);
    try std.testing.expectEqual([2]i64{ eos, eos }, c.history);
}

/// Caches resumed `from`'s state: its first `at` rows of every sparse layer and the rest whole, and its positions.
fn expectRestored(c: *const cache.Caches, from: *const cache.Caches) !void {
    const l = c.layout;
    const tail = l.sparse * (2 * l.kv_bytes + l.raw_bytes + l.pooled_bytes);
    const rows = at * l.head_dim * l.act;
    try std.testing.expectEqualSlices(u8, bytes(from)[tail..], bytes(c)[tail..]);
    try std.testing.expectEqualSlices(u8, bytes(from)[0..rows], bytes(c)[0..rows]);
    try std.testing.expectEqual(@as(usize, at), c.len);
}

test "a lone rank's restore that fails at any copy or allocation leaves fresh caches: the prompt starts from zero" {
    var r = Host.runtime();
    const s: hip.Stream = .{ .r = &r, .handle = null };
    var src = try Saved.init(&r, 3);
    defer src.deinit();
    var failures: usize = 0;
    var k: usize = 0;
    while (true) : (k += 1) {
        var main = try caches(&r);
        defer main.deinit(std.testing.allocator);
        var head = try caches(&r);
        defer head.deinit(std.testing.allocator);
        Host.copies = k;
        defer Host.copies = null;
        snapshot.restoreAgreed(std.testing.allocator, &r, s, src.copy, &main, &head, ratio, eos, null) catch |err| {
            try std.testing.expectEqual(error.RestoreRefused, err);
            try expectFresh(&main);
            try expectFresh(&head);
            failures += 1;
            continue;
        };
        try expectRestored(&main, &src.main);
        try expectRestored(&head, &src.head);
        try std.testing.expectEqual([2]i64{ 5, 6 }, main.history);
        break;
    }
    // the head's copies come after the main state's: some of these failures left the main state restored
    try std.testing.expect(failures > 2);
    k = 0;
    while (true) : (k += 1) {
        var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = k });
        var main = try caches(&r);
        defer main.deinit(std.testing.allocator);
        var head = try caches(&r);
        defer head.deinit(std.testing.allocator);
        const a = failing.allocator();
        snapshot.restoreAgreed(a, &r, s, src.copy, &main, &head, ratio, eos, null) catch {
            try expectFresh(&main);
            try expectFresh(&head);
            continue;
        };
        try expectRestored(&main, &src.main);
        break;
    }
    try std.testing.expect(k > 1);
}

test "a lone rank whose caches cannot be made fresh after a failed restore stops" {
    var r = Host.runtime();
    const s: hip.Stream = .{ .r = &r, .handle = null };
    var src = try Saved.init(&r, 3);
    defer src.deinit();
    var main = try caches(&r);
    defer main.deinit(std.testing.allocator);
    Host.copies = 1;
    Host.fail_fill = true;
    defer Host.fail_fill = false;
    defer Host.copies = null;
    const got = snapshot.restoreAgreed(std.testing.allocator, &r, s, src.copy, &main, null, ratio, eos, null);
    try std.testing.expectError(error.GroupStopped, got);
}

/// Two ranks restoring one state: rank 0 with the head, rank 1 without; `fails` is how rank 1's restore goes wrong.
const Pair = struct {
    fails: enum { none, copy, missing, fill } = .none,
    got: [2]?anyerror = @splat(null),
    fresh: [2]bool = @splat(false),
    restored: [2]bool = @splat(false),

    fn rank(p: *Pair, i: usize, l: hip.collective.Link) void {
        p.run(i, l) catch |err| {
            p.got[i] = err;
        };
    }

    fn run(p: *Pair, i: usize, l: hip.collective.Link) !void {
        var r = Host.runtime();
        const s: hip.Stream = .{ .r = &r, .handle = null };
        var src = try Saved.init(&r, @intCast(i + 1));
        defer src.deinit();
        var main = try caches(&r);
        defer main.deinit(std.testing.allocator);
        var head = try caches(&r);
        defer head.deinit(std.testing.allocator);
        const mine = i == 1;
        if (mine and p.fails == .copy) Host.copies = 2;
        if (mine and p.fails == .fill) {
            Host.copies = 0;
            Host.fail_fill = true;
        }
        defer Host.copies = null;
        defer Host.fail_fill = false;
        const copy: ?*const snapshot.Saved = if (mine and p.fails == .missing) null else src.copy;
        const h: ?*cache.Caches = if (i == 0) &head else null;
        snapshot.restoreAgreed(std.testing.allocator, &r, s, copy, &main, h, ratio, eos, l) catch |err| {
            p.got[i] = err;
            if (err == error.GroupStopped) return;
            expectFresh(&main) catch return;
            if (i == 0) expectFresh(&head) catch return;
            p.fresh[i] = true;
            return;
        };
        expectRestored(&main, &src.main) catch return;
        p.restored[i] = true;
    }
};

test "two ranks restore a state together: both resume, or both start the prompt over, or both stop" {
    var ok: Pair = .{};
    try hip.loopback.run(2, &ok, Pair.rank);
    try std.testing.expectEqual([2]bool{ true, true }, ok.restored);
    // rank 1 fails part way, or holds no copy: rank 0, which restored, starts over too
    for ([_]@TypeOf(ok.fails){ .copy, .missing }) |how| {
        var one: Pair = .{ .fails = how };
        try hip.loopback.run(2, &one, Pair.rank);
        try std.testing.expectEqual([2]?anyerror{ error.RestoreRefused, error.RestoreRefused }, one.got);
        try std.testing.expectEqual([2]bool{ true, true }, one.fresh);
    }
    var stuck: Pair = .{ .fails = .fill };
    try hip.loopback.run(2, &stuck, Pair.rank);
    try std.testing.expectEqual([2]?anyerror{ error.GroupStopped, error.GroupStopped }, stuck.got);
}
