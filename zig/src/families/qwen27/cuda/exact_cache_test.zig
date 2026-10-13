//! Host-only CUDA-driver mocks exercise snapshot ownership and failure paths.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cache_mod = @import("exact_prompt_cache.zig");
const st = @import("state.zig");
const shape = @import("shape.zig");
const kern = @import("kernels.zig");
const dr = @import("draft.zig");
const g = shape.the_27b;
const a = std.testing.allocator;

const Mock = struct {
    var live: usize = 0;
    var freed: usize = 0;
    const Copy = struct { dst: u64, src: u64, bytes: usize };
    var copy_rows: [512]Copy = undefined;
    var copies: usize = 0;
    var fences: usize = 0;
    var fail_copy = false;
    var fail_fence = false;
    var next: u64 = 0x100000000;
    fn alloc(ptr: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        ptr.* = next;
        next += 0x100000000;
        live += 1;
        return cuda.abi.success;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        live -= 1;
        freed += 1;
        return cuda.abi.success;
    }
    fn copy(dst: cuda.abi.DevicePtr, src: cuda.abi.DevicePtr, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_copy) return cuda.abi.error_not_ready;
        std.debug.assert(copies < copy_rows.len);
        copy_rows[copies] = .{ .dst = dst, .src = src, .bytes = bytes };
        copies += 1;
        return cuda.abi.success;
    }
    fn fence(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_fence) return cuda.abi.error_not_ready;
        fences += 1;
        return cuda.abi.success;
    }
    fn reset() void {
        live = 0;
        freed = 0;
        copies = 0;
        fences = 0;
        fail_copy = false;
        fail_fence = false;
    }
};

const Fixture = struct {
    driver: cuda.Driver = undefined,
    kernels: kern.Kernels = undefined,
    rec: [64]u64 = @splat(0x1000),
    conv: [64]u64 = @splat(0x2000),
    keys: [64]u64 = @splat(0x3000),
    values: [64]u64 = @splat(0x4000),
    fn ops(f: *Fixture) kern.Ops {
        f.driver.api.cuMemAlloc_v2 = Mock.alloc;
        f.driver.api.cuMemFree_v2 = Mock.free;
        f.driver.api.cuMemcpyDtoDAsync_v2 = Mock.copy;
        f.driver.api.cuStreamSynchronize = Mock.fence;
        f.kernels.d = &f.driver;
        return .{ .k = &f.kernels, .s = .{ .d = &f.driver, .handle = null } };
    }
    fn seq(f: *Fixture, n: usize) st.Seq {
        return .{ .gpa = a, .arena = .{ .buf = null }, .rec = &f.rec, .conv = &f.conv, .keys = &f.keys, .values = &f.values, .kscale = &.{}, .vscale = &.{}, .kv8 = false, .head_in = 0x5000, .head_xs = 0x6000, .pos = n, .capacity = n + 256 };
    }
};

fn stream(ids: []const u32, drafted: bool) !lanes.Stream {
    return lanes.Stream.init(a, .{ .id = "cache-test", .prompt = ids, .max_new = 256, .drafts = drafted });
}
fn finish(s: *lanes.Stream) void {
    s.finished = true;
    s.reason = .length;
}

test "actual snapshot owner replacement and hit copy never alias a live destination" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    var seed = try stream(&.{ 1, 2, 3 }, false);
    defer seed.deinit(a);
    var seq = f.seq(3);
    try cache.capture(&seed, &seq, null, ops, g);
    try std.testing.expectEqual(@as(usize, 1), Mock.live);
    finish(&seed);
    try cache.release(&seed, true, ops);
    const saved = cache.active.?;
    var hit = try stream(&.{ 1, 2, 3 }, false);
    defer hit.deinit(a);
    var dest = f.seq(0);
    try std.testing.expect(try cache.lookup(&hit, &dest, null, ops, g));
    try std.testing.expectEqual(@as(u32, 3), hit.cached);
    try std.testing.expectEqual(@as(usize, 3), dest.pos);
    try std.testing.expect(saved.head_in != dest.head_in);
    var miss = try stream(&.{ 4, 5, 6 }, false);
    defer miss.deinit(a);
    try cache.capture(&miss, &seq, null, ops, g);
    try std.testing.expectEqual(@as(u64, 1), cache.stats.evictions);
    try std.testing.expectEqual(@as(usize, 1), Mock.live);
    try std.testing.expect(Mock.fences >= 2);
    finish(&miss);
    try cache.release(&miss, true, ops);
    try std.testing.expectEqual(@as(u64, 2), cache.stats.promotions);
}

test "pending is never hit and error or cancellation cannot promote" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    var s = try stream(&.{ 1, 2 }, false);
    defer s.deinit(a);
    var seq = f.seq(2);
    try cache.capture(&s, &seq, null, ops, g);
    var other = try stream(&.{ 1, 2 }, false);
    defer other.deinit(a);
    try std.testing.expect(!try cache.lookup(&other, &seq, null, ops, g));
    try cache.capture(&other, &seq, null, ops, g);
    try std.testing.expectEqual(@as(usize, 1), Mock.live);
    s.finished = true;
    s.reason = .cancelled;
    try cache.release(&s, true, ops);
    try std.testing.expect(cache.active == null and cache.pending == null);
    s.finished = false;
    s.reason = .none;
    try cache.capture(&s, &seq, null, ops, g);
    finish(&s);
    try cache.release(&other, false, ops);
    try std.testing.expect(cache.failed and cache.active == null and Mock.live == 0);
    try cache.release(&s, true, ops);
    try std.testing.expectEqual(@as(u64, 0), cache.stats.promotions);
}

test "copy failure purges and failed fence defers freeing in-flight source" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    var s = try stream(&.{1}, false);
    defer s.deinit(a);
    var seq = f.seq(1);
    Mock.fail_copy = true;
    try std.testing.expectError(error.NotReady, cache.capture(&s, &seq, null, ops, g));
    Mock.fail_copy = false;
    Mock.fail_fence = true;
    try std.testing.expectError(error.NotReady, cache.abandon(&s, ops));
    try std.testing.expectEqual(@as(usize, 1), Mock.live);
    Mock.fail_fence = false;
    try cache.abandon(&s, ops);
    try std.testing.expectEqual(@as(usize, 0), Mock.live);
}

test "allocation refusal and disabled path perform no CUDA copy or fence" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var s = try stream(&.{1}, false);
    defer s.deinit(a);
    var seq = f.seq(1);
    var off = cache_mod.Cache.init(a, false);
    defer off.deinit(ops) catch unreachable;
    try off.capture(&s, &seq, null, ops, g);
    try std.testing.expect(!try off.lookup(&s, &seq, null, ops, g));
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    var refused = cache_mod.Cache.init(failing.allocator(), true);
    defer refused.deinit(ops) catch unreachable;
    try refused.capture(&s, &seq, null, ops, g);
    try std.testing.expectEqual(@as(u64, 1), refused.stats.refusals);
    try std.testing.expect(Mock.copies == 0 and Mock.fences == 0 and Mock.live == 0);
}

test "draft active-bank compact copies preserve metadata and bound maximum snapshot" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    var s = try stream(&.{ 1, 2 }, true);
    defer s.deinit(a);
    var seq = f.seq(2);
    var ctx: dr.Context = .{ .arena = .{ .buf = null }, .keys = @splat(@splat(0x10000)), .values = @splat(@splat(0x20000)), .cur = 1, .len = 2, .end = 2 };
    try cache.capture(&s, &seq, &ctx, ops, g);
    // absorbMany and _block_attention pack each head at len*128 BF16 values, not the allocation window.
    const head_bytes = 2 * 128 * 2;
    const target_copies = 2 + 48 * 2 + 16 * 2;
    const saved = cache.pending.?;
    for (0..5) |layer| for (0..8) |head| {
        const pair = Mock.copy_rows[target_copies + layer * 16 + head * 2 ..][0..2];
        try std.testing.expectEqual(0x10000 + head * head_bytes, pair[0].src);
        try std.testing.expectEqual(0x20000 + head * head_bytes, pair[1].src);
        try std.testing.expectEqual(saved.draft_keys[layer] + head * head_bytes, pair[0].dst);
        try std.testing.expectEqual(saved.draft_values[layer] + head * head_bytes, pair[1].dst);
        try std.testing.expectEqual(head_bytes, pair[0].bytes);
        try std.testing.expectEqual(head_bytes, pair[1].bytes);
    };
    const capture_copies = Mock.copies;
    finish(&s);
    try cache.release(&s, true, ops);
    var hit = try stream(&.{ 1, 2 }, true);
    defer hit.deinit(a);
    ctx.cur = 0;
    ctx.len = 0;
    ctx.end = 0;
    try std.testing.expect(try cache.lookup(&hit, &seq, &ctx, ops, g));
    try std.testing.expect(ctx.cur == 1 and ctx.len == 2 and ctx.end == 2);
    for (0..5) |layer| for (0..8) |head| {
        const pair = Mock.copy_rows[capture_copies + target_copies + layer * 16 + head * 2 ..][0..2];
        try std.testing.expectEqual(0x10000 + head * head_bytes, pair[0].dst);
        try std.testing.expectEqual(0x20000 + head * head_bytes, pair[1].dst);
        try std.testing.expectEqual(saved.draft_keys[layer] + head * head_bytes, pair[0].src);
        try std.testing.expectEqual(saved.draft_values[layer] + head * head_bytes, pair[1].src);
    };
    try std.testing.expectEqual(@as(usize, 197995008), cache_mod.Snapshot.bytes(g, 512, true));
    try std.testing.expect(cache_mod.Snapshot.bytes(g, 512, true) <= cache_mod.policy.max_bytes);
}

test "acquisition hint shares exact eligibility but never copies or reports a hit" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    var seed = try stream(&.{ 1, 2, 3 }, false);
    defer seed.deinit(a);
    var seq = f.seq(3);
    try cache.capture(&seed, &seq, null, ops, g);
    finish(&seed);
    try cache.release(&seed, true, ops);
    var hit = try stream(&.{ 1, 2, 3 }, false);
    defer hit.deinit(a);
    const before = cache.stats;
    const copies = Mock.copies;
    const fences = Mock.fences;
    const active = cache.active;
    try std.testing.expect(cache.canRestore(&hit, false));
    try std.testing.expect(!cache.canRestore(&hit, true));
    hit.cached = 1;
    try std.testing.expect(!cache.canRestore(&hit, false));
    hit.cached = 0;
    cache.failed = true;
    try std.testing.expect(!cache.canRestore(&hit, false));
    cache.failed = false;
    var miss = try stream(&.{ 1, 2, 4 }, false);
    defer miss.deinit(a);
    try std.testing.expect(!cache.canRestore(&miss, false));
    try std.testing.expect(std.meta.eql(before, cache.stats));
    try std.testing.expectEqual(copies, Mock.copies);
    try std.testing.expectEqual(fences, Mock.fences);
    try std.testing.expect(cache.active == active and hit.cached == 0);
}

test "enabled first logits add one bounded row and absent marker prevents promotion" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    cache.logits_enabled = true;
    defer cache.deinit(ops) catch unreachable;
    var seed = try stream(&.{ 1, 2, 3 }, false);
    defer seed.deinit(a);
    var seq = f.seq(3);
    try cache.capture(&seed, &seq, null, ops, g);
    const snap = cache.pending.?;
    try std.testing.expect(!snap.logits_valid and snap.first_logits != 0);
    try std.testing.expectEqual(cache_mod.Snapshot.bytes(g, 3, false) + 496640, snap.device_bytes);
    try std.testing.expectEqual(@as(usize, 198491648), cache_mod.Snapshot.bytesWithLogits(g, 512, true, true));
    try std.testing.expect(cache_mod.Snapshot.bytesWithLogits(g, 512, true, true) < cache_mod.policy.max_bytes);
    finish(&seed);
    try cache.release(&seed, true, ops);
    try std.testing.expect(cache.active == null and cache.pending == null and Mock.live == 0);
    try std.testing.expectEqual(@as(u64, 1), cache.stats.purges);
}

test "actual raw capture and proven current-active copy preserve direction and eviction fallback" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    cache.logits_enabled = true;
    defer cache.deinit(ops) catch unreachable;
    var seed = try stream(&.{ 1, 2, 3 }, false);
    defer seed.deinit(a);
    var seq = f.seq(3);
    try cache.capture(&seed, &seq, null, ops, g);
    const saved_logits = cache.pending.?.first_logits;
    const before = Mock.copies;
    try std.testing.expect(try cache.captureFirstLogits(&seed, &seq, ops, 0x7770));
    try std.testing.expectEqual(before + 1, Mock.copies);
    try std.testing.expectEqual(saved_logits, Mock.copy_rows[before].dst);
    try std.testing.expectEqual(@as(u64, 0x7770), Mock.copy_rows[before].src);
    try std.testing.expectEqual(@as(usize, 496640), Mock.copy_rows[before].bytes);
    finish(&seed);
    try cache.release(&seed, true, ops);
    var hit = try stream(&.{ 1, 2, 3 }, false);
    defer hit.deinit(a);
    var dest = f.seq(0);
    try std.testing.expect(try cache.lookup(&hit, &dest, null, ops, g));
    const copied = Mock.copies;
    try std.testing.expect(!try cache.restoreFirstLogits(&hit, &dest, ops, 0x8880, false));
    try std.testing.expectEqual(copied, Mock.copies);
    dest.pos = 4;
    try std.testing.expect(!try cache.restoreFirstLogits(&hit, &dest, ops, 0x8880, true));
    dest.pos = 3;
    try std.testing.expect(try cache.restoreFirstLogits(&hit, &dest, ops, 0x8880, true));
    try std.testing.expectEqual(saved_logits, Mock.copy_rows[copied].src);
    try std.testing.expectEqual(@as(u64, 0x8880), Mock.copy_rows[copied].dst);
    try std.testing.expectEqual(@as(usize, 496640), Mock.copy_rows[copied].bytes);
    try std.testing.expect(cache.active.?.logits_valid);
    try std.testing.expectEqual(@as(u64, 1), cache.stats.first_logits_hits);
    Mock.copies = 0;
    var miss = try stream(&.{ 4, 5, 6 }, false);
    defer miss.deinit(a);
    try cache.capture(&miss, &seq, null, ops, g);
    try std.testing.expect(cache.active == null and cache.pending != null);
    const after_eviction = Mock.copies;
    try std.testing.expect(!try cache.restoreFirstLogits(&hit, &dest, ops, 0x8880, true));
    try std.testing.expectEqual(after_eviction, Mock.copies);
    try std.testing.expect(Mock.fences > 0);
}

test "raw capture error clears validity and hit copy error disables future use" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    cache.logits_enabled = true;
    defer cache.deinit(ops) catch unreachable;
    var seed = try stream(&.{1}, false);
    defer seed.deinit(a);
    var seq = f.seq(1);
    try cache.capture(&seed, &seq, null, ops, g);
    Mock.fail_copy = true;
    try std.testing.expectError(error.NotReady, cache.captureFirstLogits(&seed, &seq, ops, 0x7770));
    try std.testing.expect(cache.failed and !cache.pending.?.logits_valid);
    Mock.fail_copy = false;
    finish(&seed);
    try cache.release(&seed, true, ops);
    try std.testing.expect(cache.active == null and cache.pending == null);
    cache.failed = false;
    seed.finished = false;
    seed.reason = .none;
    try cache.capture(&seed, &seq, null, ops, g);
    try std.testing.expect(try cache.captureFirstLogits(&seed, &seq, ops, 0x7770));
    finish(&seed);
    try cache.release(&seed, true, ops);
    var hit = try stream(&.{1}, false);
    defer hit.deinit(a);
    var dest = f.seq(0);
    try std.testing.expect(try cache.lookup(&hit, &dest, null, ops, g));
    Mock.fail_copy = true;
    try std.testing.expectError(error.NotReady, cache.restoreFirstLogits(&hit, &dest, ops, 0x8880, true));
    try std.testing.expect(cache.failed);
    Mock.fail_copy = false;
    try std.testing.expect(!try cache.restoreFirstLogits(&hit, &dest, ops, 0x8880, true));
}

test "logits-off capture and first-logit methods add no row copy or fence" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    var seed = try stream(&.{1}, false);
    defer seed.deinit(a);
    var seq = f.seq(1);
    try cache.capture(&seed, &seq, null, ops, g);
    try std.testing.expectEqual(cache_mod.Snapshot.bytes(g, 1, false), cache.pending.?.device_bytes);
    try std.testing.expectEqual(@as(u64, 0), cache.pending.?.first_logits);
    const copied = Mock.copies;
    const fenced = Mock.fences;
    try std.testing.expect(!try cache.captureFirstLogits(&seed, &seq, ops, 0x7770));
    try std.testing.expect(!try cache.restoreFirstLogits(&seed, &seq, ops, 0x8880, true));
    try std.testing.expectEqual(copied, Mock.copies);
    try std.testing.expectEqual(fenced, Mock.fences);
    finish(&seed);
    try cache.release(&seed, true, ops);
    try std.testing.expect(cache.active != null);
}

test "unsupported Seq formats decline active lookup and new capture before any device operation" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    var seed = try stream(&.{ 1, 2 }, false);
    defer seed.deinit(a);
    var seq = f.seq(2);
    try cache.capture(&seed, &seq, null, ops, g);
    finish(&seed);
    try cache.release(&seed, true, ops);
    var hit = try stream(&.{ 1, 2 }, false);
    defer hit.deinit(a);
    var miss = try stream(&.{ 3, 4 }, false);
    defer miss.deinit(a);
    const raw = try cache_mod.Snapshot.init(a, ops, g, &hit);
    defer raw.deinit();
    var media: @import("vision.zig").Media = undefined;
    const before = .{ Mock.live, Mock.copies, Mock.fences };
    for (0..3) |kind| {
        seq.kv8 = kind == 0;
        seq.media = if (kind == 1) &media else null;
        seq.rope_delta = if (kind == 2) 7 else 0;
        try cache.capture(&miss, &seq, null, ops, g);
        try std.testing.expect(!try cache.lookup(&hit, &seq, null, ops, g));
        try std.testing.expectEqual(@as(u32, 0), hit.cached);
        try std.testing.expectError(error.UnsupportedExactCacheState, raw.save(ops, g, &seq, null));
        try std.testing.expectError(error.UnsupportedExactCacheState, raw.restore(ops, g, &seq, null));
        try std.testing.expectEqual(before[0], Mock.live);
        try std.testing.expectEqual(before[1], Mock.copies);
        try std.testing.expectEqual(before[2], Mock.fences);
    }
    seq.kv8 = false;
    seq.media = null;
    seq.rope_delta = 0;
    hit.images = &.{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 0, .tokens = 1 }};
    try std.testing.expect(!try cache.lookup(&hit, &seq, null, ops, g));
    try std.testing.expectEqual(before[1], Mock.copies);
    hit.images = &.{};
    try std.testing.expect(try cache.lookup(&hit, &seq, null, ops, g));
    try std.testing.expectEqual(@as(u32, 2), hit.cached);
}

test "partial host-prefix metadata cannot become full-N exact restoration proof" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    var seed = try stream(&.{ 1, 2, 3 }, false);
    defer seed.deinit(a);
    var seq = f.seq(3);
    try cache.capture(&seed, &seq, null, ops, g);
    finish(&seed);
    try cache.release(&seed, true, ops);
    var hit = try stream(&.{ 1, 2, 3 }, false);
    defer hit.deinit(a);
    const before = Mock.copies;
    hit.reuse.at = 2;
    try std.testing.expect(!cache.canRestore(&hit, false));
    try std.testing.expect(!try cache.lookup(&hit, &seq, null, ops, g));
    hit.reuse.at = 0;
    hit.reuse.marks = &.{2};
    try std.testing.expect(!cache.canRestore(&hit, false));
    hit.reuse.marks = &.{};
    hit.cached = 2;
    try std.testing.expect(!cache.canRestore(&hit, false));
    try std.testing.expect(!try cache.lookup(&hit, &seq, null, ops, g));
    try std.testing.expectEqual(before, Mock.copies);
    try std.testing.expectEqual(@as(u32, 2), hit.cached);
    hit.cached = 0;
    try std.testing.expect(cache.canRestore(&hit, false));
}

test "normal chat assistant-header marker permits full-N snapshot promotion and repeat hit" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var cache = cache_mod.Cache.init(a, true);
    defer cache.deinit(ops) catch unreachable;
    const ids: [31]u32 = @splat(1);
    var seed = try lanes.Stream.init(a, .{ .id = "chat-cold", .prompt = &ids, .max_new = 256, .drafts = false, .decide = 29 });
    defer seed.deinit(a);
    var seq = f.seq(ids.len);
    try std.testing.expect(!cache.canRestore(&seed, false));
    try cache.capture(&seed, &seq, null, ops, g);
    try std.testing.expectEqual(@as(u64, 1), cache.stats.captures);
    finish(&seed);
    try cache.release(&seed, true, ops);
    try std.testing.expectEqual(@as(u64, 1), cache.stats.promotions);
    var repeat = try lanes.Stream.init(a, .{ .id = "chat-repeat", .prompt = &ids, .max_new = 256, .drafts = false, .decide = 29 });
    defer repeat.deinit(a);
    var dest = f.seq(0);
    try std.testing.expect(cache.canRestore(&repeat, false));
    try std.testing.expect(try cache.lookup(&repeat, &dest, null, ops, g));
    try std.testing.expectEqual(@as(u32, 31), repeat.cached);
    try std.testing.expectEqual(@as(usize, 31), dest.pos);
    try std.testing.expectEqual(@as(u64, 1), cache.stats.hits);
}
