//! Nonuniform paged KV and retained features occupy distinct serialized ranges on host and device copies.
const std = @import("std");
const cuda = @import("cuda");
const reuse = @import("cuda_reuse.zig");
const engine = @import("cuda_engine.zig");
const t = std.testing;
const row = 256 * 2;
const rows = 3;
const page_tokens = 2;
const feature_bytes = 4 * 256 * 2;

const Memory = struct {
    var kept: [1 << 20]u8 align(16) = undefined;
    var room: usize = 0;
    var frees: usize = 0;
    var waits: usize = 0;

    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        std.debug.assert(n <= kept.len);
        room = n;
        @memset(kept[0..n], 0xa5);
        out.* = @intFromPtr(&kept);
        return cuda.abi.success;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        var address: u64 = undefined;
        const result = alloc(&address, n);
        out.* = @ptrFromInt(address);
        return result;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        frees += 1;
        return cuda.abi.success;
    }
    fn freeHost(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        return free(0);
    }
    fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        return cuda.abi.success;
    }
    fn htod(dst: u64, src: ?*const anyopaque, n: usize, stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return copy(dst, @intFromPtr(src.?), n, stream);
    }
    fn dtoh(dst: ?*anyopaque, src: u64, n: usize, stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return copy(@intFromPtr(dst.?), src, n, stream);
    }
    fn copy2d(m: *const cuda.abi.Memcpy2D, stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const src = if (m.src_type == cuda.abi.memory_type_host) @intFromPtr(m.src_host.?) else m.src;
        const dst = if (m.dst_type == cuda.abi.memory_type_host) @intFromPtr(m.dst_host.?) else m.dst;
        for (0..m.height) |i| _ = copy(dst + i * m.dst_pitch, src + i * m.src_pitch, m.width, stream);
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        waits += 1;
        return cuda.abi.success;
    }
};

fn buffer(d: *cuda.Driver, data: []u8) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = @intFromPtr(data.ptr), .len = data.len, .borrowed = true };
}

fn fill(data: []u8, seed: u8) void {
    for (data, 0..) |*value, i| value.* = @truncate(seed +% i *% 37 +% (i >> 5));
}

fn check(device: bool, restore_only: bool) !void {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = Memory.alloc;
    d.api.cuMemFree_v2 = Memory.free;
    d.api.cuMemHostAlloc = Memory.hostAlloc;
    d.api.cuMemFreeHost = Memory.freeHost;
    d.api.cuMemcpyDtoDAsync_v2 = Memory.copy;
    d.api.cuMemcpyHtoDAsync_v2 = Memory.htod;
    d.api.cuMemcpyDtoHAsync_v2 = Memory.dtoh;
    d.api.cuMemcpy2DAsync_v2 = Memory.copy2d;
    d.api.cuStreamSynchronize = Memory.sync;
    var e: engine.Engine = undefined;
    e.d = &d;
    e.gpa = t.allocator;
    e.stream = .{ .d = &d, .handle = null };
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 4, .linear = @splat(false) };
    e.g.linear[0] = true;
    e.kv8 = false;
    e.n_taps = 1;
    e.taps[0] = 0;
    e.tap_capacity = 4;
    e.prompt_taps = .{ .cache = true };
    var state_storage: [1 << 18]u8 = undefined;
    const state = state_storage[0..e.g.stateBytes()];
    var kv_storage: [1 << 14]u8 = undefined;
    const page_bytes = e.g.kvBytes(page_tokens, false);
    const kv = kv_storage[0 .. 2 * page_bytes];
    var taps: [feature_bytes]u8 = undefined;
    fill(state, 17);
    fill(kv, 29);
    fill(&taps, 43);
    const want_state = try t.allocator.dupe(u8, state);
    defer t.allocator.free(want_state);
    const want_kv = try t.allocator.dupe(u8, kv);
    defer t.allocator.free(want_kv);
    const want_taps = taps;
    var page_ids = [_]u32{ 1, 0 };
    var cache: engine.Cache = .{
        .state = buffer(&d, state),
        .kv = .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true },
        .paged = .{ .base = @intFromPtr(kv.ptr), .table = .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true }, .ids = &page_ids, .tokens = page_tokens, .own_ids = false },
        .tap = buffer(&d, &taps),
        .log = .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true },
        .cap = 4,
        .len = rows,
        .written = rows,
        .logits = &.{},
    };
    const layout: reuse.Layout = .{ .g = e.g, .format = .bf16, .cap = 4, .at = rows };
    const tail: usize = @intCast(layout.total());
    Memory.frees = 0;
    Memory.waits = 0;
    const snapshot = try reuse.saveCache(&e, &cache, rows, &.{}, device);
    defer reuse.dropCopy(&e, snapshot);
    try t.expectEqual(tail + feature_bytes, snapshot.len);
    try t.expectEqual(@as(usize, 1), Memory.waits);
    const kept = Memory.kept[0..Memory.room];
    if (restore_only) {
        // A separately constructed valid image isolates the restore offset from the old save offset.
        @memcpy(kept[0..state.len], want_state);
        for (0..2) |block| for (0..rows) |position| {
            const src = page_ids[position / page_tokens] * page_bytes + (block * page_tokens + position % page_tokens) * row;
            const dst = state.len + (block * rows + position) * row;
            @memcpy(kept[dst..][0..row], want_kv[src..][0..row]);
        };
        @memcpy(kept[tail..][0..feature_bytes], &want_taps);
    } else {
        try t.expectEqualSlices(u8, want_state, kept[0..state.len]);
        for (0..2) |block| for (0..rows) |position| {
            const src = page_ids[position / page_tokens] * page_bytes + (block * page_tokens + position % page_tokens) * row;
            const dst = state.len + (block * rows + position) * row;
            try t.expectEqualSlices(u8, want_kv[src..][0..row], kept[dst..][0..row]);
        };
        try t.expectEqualSlices(u8, &want_taps, kept[tail..][0..feature_bytes]);
        return;
    }
    @memset(state, 0xcc);
    @memset(kv, 0xdd);
    @memset(&taps, 0xee);
    try reuse.restoreCache(&e, &cache, snapshot, &.{});
    try t.expectEqual(@as(usize, 2), Memory.waits);
    try t.expectEqualSlices(u8, want_state, state);
    for (0..2) |block| for (0..rows) |position| {
        const at = page_ids[position / page_tokens] * page_bytes + (block * page_tokens + position % page_tokens) * row;
        try t.expectEqualSlices(u8, want_kv[at..][0..row], kv[at..][0..row]);
    };
    try t.expectEqualSlices(u8, &want_taps, &taps);
    try t.expectEqual(@as(u64, rows), cache.tap_end);
}

test "retained reuse features: paged BF16 save appends after every KV piece" {
    for ([_]bool{ false, true }) |device| {
        try check(device, false);
        try t.expectEqual(@as(usize, 1), Memory.frees);
    }
}

test "retained reuse features: paged BF16 restore reads the tail after every KV piece" {
    for ([_]bool{ false, true }) |device| {
        try check(device, true);
        try t.expectEqual(@as(usize, 1), Memory.frees);
    }
}

test "retained reuse features: a walker and its epilogue precede exactly one wait, with errors left to the owner" {
    const Protocol = struct {
        const Self = @This();
        steps: [3]u8 = @splat(0),
        count: usize = 0,
        fail: u8 = 0,
        fn record(p: *Self, step: u8) !void {
            p.steps[p.count] = step;
            p.count += 1;
            if (p.fail == step) return error.NotReady;
        }
        pub fn transfer(p: *Self, _: anytype, _: anytype) !void {
            try p.record(1);
        }
        pub fn synchronize(p: *Self) !void {
            try p.record(3);
        }
        const Tail = struct {
            protocol: *Self,
            pub fn transfer(tail: @This(), _: anytype, _: anytype) !void {
                try tail.protocol.record(2);
            }
        };
    };
    for ([_]u8{ 0, 1, 2, 3 }) |failure| {
        var p: Protocol = .{ .fail = failure };
        var e = .{ .stream = &p };
        const complete = @import("cuda_reuse_features.zig").complete(&e, {}, &p, Protocol.Tail{ .protocol = &p });
        if (failure == 0) try complete else try t.expectError(error.NotReady, complete);
        const reached: usize = if (failure == 0) 3 else failure;
        const order = [_]u8{ 1, 2, 3 };
        try t.expectEqualSlices(u8, order[0..reached], p.steps[0..p.count]);
    }
}

fn matrix(device: bool, paged: bool, format: reuse.Format) !void {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = Memory.alloc;
    d.api.cuMemFree_v2 = Memory.free;
    d.api.cuMemHostAlloc = Memory.hostAlloc;
    d.api.cuMemFreeHost = Memory.freeHost;
    d.api.cuMemcpyDtoDAsync_v2 = Memory.copy;
    d.api.cuMemcpyHtoDAsync_v2 = Memory.htod;
    d.api.cuMemcpyDtoHAsync_v2 = Memory.dtoh;
    d.api.cuMemcpy2DAsync_v2 = Memory.copy2d;
    d.api.cuStreamSynchronize = Memory.sync;
    var e: engine.Engine = undefined;
    e.d = &d;
    e.gpa = t.allocator;
    e.stream = .{ .d = &d, .handle = null };
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 4, .linear = @splat(false) };
    e.g.linear[0] = true;
    e.kv8 = format == .fp8;
    e.n_taps = 1;
    e.taps[0] = 0;
    e.tap_capacity = 4;
    e.prompt_taps = .{ .cache = true };
    var state_storage: [1 << 18]u8 = undefined;
    const state = state_storage[0..e.g.stateBytes()];
    var kv_storage: [1 << 14]u8 = undefined;
    const page_bytes = e.g.kvBytes(page_tokens, e.kv8);
    const kv = kv_storage[0 .. 2 * page_bytes];
    var taps: [feature_bytes]u8 = undefined;
    fill(state, 17);
    fill(kv, 29);
    fill(&taps, 43);
    const want_state = try t.allocator.dupe(u8, state);
    defer t.allocator.free(want_state);
    const want_kv = try t.allocator.dupe(u8, kv);
    defer t.allocator.free(want_kv);
    const want_taps = taps;
    var page_ids = [_]u32{ 1, 0 };
    var cache: engine.Cache = .{
        .state = buffer(&d, state),
        .kv = if (paged) .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true } else buffer(&d, kv),
        .paged = if (paged) .{ .base = @intFromPtr(kv.ptr), .table = .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true }, .ids = &page_ids, .tokens = page_tokens, .own_ids = false } else null,
        .tap = buffer(&d, &taps),
        .log = .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true },
        .cap = 4,
        .len = rows,
        .written = rows,
        .logits = &.{},
    };
    const layout: reuse.Layout = .{ .g = e.g, .format = format, .cap = 4, .at = rows };
    const page_layout: reuse.Layout = .{ .g = e.g, .format = format, .cap = page_tokens, .at = page_tokens };
    const tail: usize = @intCast(layout.total());
    Memory.frees = 0;
    Memory.waits = 0;
    const snapshot = try reuse.saveCache(&e, &cache, rows, &.{}, device);
    defer reuse.dropCopy(&e, snapshot);
    const kept = Memory.kept[0..Memory.room];
    try t.expectEqual(tail + feature_bytes, snapshot.len);
    try t.expectEqual(@as(usize, 1), Memory.waits);
    try t.expectEqualSlices(u8, want_state, kept[0..state.len]);
    var off = state.len;
    var expected_kv: [1 << 14]u8 = @splat(0xdd);
    for (1..layout.count()) |i| {
        const piece = layout.piece(i);
        const page_piece = page_layout.piece(i);
        const unit = piece.len / rows;
        for (0..rows) |position| {
            const src = if (paged) page_ids[position / page_tokens] * page_bytes + page_piece.off + (position % page_tokens) * unit else piece.off + position * unit;
            const dst = off + position * unit;
            try t.expectEqualSlices(u8, want_kv[src..][0..unit], kept[dst..][0..unit]);
            @memcpy(expected_kv[src..][0..unit], want_kv[src..][0..unit]);
        }
        off += piece.len;
    }
    try t.expectEqual(tail, off);
    try t.expectEqualSlices(u8, &want_taps, kept[tail..][0..feature_bytes]);
    @memset(state, 0xcc);
    @memset(kv, 0xdd);
    @memset(&taps, 0xee);
    try reuse.restoreCache(&e, &cache, snapshot, &.{});
    try t.expectEqual(@as(usize, 2), Memory.waits);
    try t.expectEqualSlices(u8, want_state, state);
    try t.expectEqualSlices(u8, expected_kv[0..kv.len], kv);
    try t.expectEqualSlices(u8, &want_taps, &taps);
    try t.expectEqual(@as(u64, rows), cache.tap_end);
}

test "retained reuse features: host and device BF16/FP8 images preserve contiguous and shuffled-page bytes" {
    for ([_]bool{ false, true }) |device| for ([_]bool{ false, true }) |paged| for ([_]reuse.Format{ .bf16, .fp8 }) |format| {
        try matrix(device, paged, format);
        try t.expectEqual(@as(usize, 1), Memory.frees);
    };
}
