//! Host tests of the deep prefill: layouts, arenas, window staging, passes, rank masks and buffer ownership.

const std = @import("std");
const cuda = @import("cuda");
const batch = @import("../../cuda_batch.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const history_ = @import("cuda_exl3_prefill_history.zig");
const tp_ = @import("../tp/cuda_exl3_tp.zig");
const window_ = @import("exl3_prefill_window.zig");
const state = @import("cuda_exl3_prefill_state.zig");
const hub = @import("cuda_exl3_prefill.zig");
const chunks = @import("cuda_exl3_prefill_chunk.zig");
const experts = @import("cuda_exl3_prefill_moe.zig");

const dims = wt.dims;
const wide = wt.wide;
const ple_layer = wt.ple_layer;
const table_n = wt.experts_n + 1;
const routed_n_ = wt.experts_n;
const shared_id = wt.experts_n;

const Deep = state.Deep;
const HostLayout = state.HostLayout;
const Layout = state.Layout;
const Options = state.Options;
const Pass = state.Pass;
const Persistent = state.Persistent;
const deviceBytes = state.deviceBytes;
const deviceBytesAt = state.deviceBytesAt;
const release = hub.release;
const releaseFor = hub.releaseFor;
const reserveAt = hub.reserveAt;
const reserveTwoChunk = hub.reserveTwoChunk;
const stepCapacity = hub.stepCapacity;
const compactLast = chunks.compactLast;
const pendingWriteback = chunks.pendingWriteback;
const selectWriteback = chunks.selectWriteback;
const view = chunks.view;
const windowInputs = chunks.windowInputs;
const build = experts.build;
const maskOwned = experts.maskOwned;

test "exl3: reduced prefill writeback preserves slot rounding sub-blocks and tail selection" {
    try @import("cuda_exl3_prefill_writeback_test.zig").run(Deep, Layout.of, view, pendingWriteback, compactLast, selectWriteback);
    try std.testing.expect(!(Options{}).reduced_branch);
}

test "exl3: two-chunk scheduling and reservation contracts" {
    _ = window_;
    try std.testing.expect(!(Options{}).two_chunk);
    try std.testing.expectEqual(@as(usize, 4096), stepCapacity(4096, false));
    try std.testing.expectEqual(@as(usize, 8192), stepCapacity(4096, true));
    try std.testing.expectEqual(@as(usize, 8192), stepCapacity(8192, true));
}

test "exl3: two-chunk arenas retain hidden and deferred HC operands without aliasing" {
    const rows = 4096;
    const layout = Layout.of(rows, 2);
    const persistent = Persistent.of(rows);
    try std.testing.expectEqual(@as(usize, 811285504), layout.total);
    const host_layout = HostLayout.of(rows);
    try std.testing.expectEqual(@as(usize, 160 << 20), 2 * rows * wide * 2);
    var driver: cuda.Driver = undefined;
    var first: Deep = undefined;
    first.world = 2;
    first.rows = rows;
    first.shared = null;
    first.lay = layout;
    first.dev = .{ .d = &driver, .ptr = 0x100000, .len = layout.total };
    first.sum_dev = null;
    var second = first;
    second.dev.ptr += layout.total;
    second.dev.len = persistent.total;
    second.rows = rows;
    second.shared = &first;
    var base: batch.Scratch = undefined;
    base.moe_f32 = 0;
    const fields = .{ "h", "pss", "inj", "mixed", "xsm", "y", "wts", "picks", "ple_emb", "ple_xs", "ids", "pos", "members", "items", "kscratch", "sum" };
    inline for (fields, 0..) |field, index| {
        const start = @field(layout, field);
        const end = if (index + 1 < fields.len) @field(layout, fields[index + 1]) else layout.total;
        try std.testing.expectEqual(@as(usize, 0), start % 256);
        try std.testing.expect(start < end and end <= layout.total);
        try std.testing.expect(first.dev.ptr + end <= second.dev.ptr);
    }
    for ([_]usize{ 0, 1024, 3072 }) |offset| {
        const a = view(&base, &first, offset);
        const b = view(&base, &second, offset);
        inline for (.{ "h", "pss", "inj", "y", "wts" }) |field| {
            const second_start = second.dev.ptr + @field(persistent, field);
            try std.testing.expect(@field(b, field) >= second_start and @field(b, field) < second.dev.ptr + persistent.total);
            try std.testing.expect(@field(a, field) < second.dev.ptr);
        }
        inline for (.{ "mixed", "xsm", "picks", "ple_emb", "ple_xs", "ids", "pos", "moe_f32" }) |field| {
            try std.testing.expectEqual(@field(a, field), @field(b, field));
        }
    }
    inline for (.{ "members", "items", "kscratch", "sum" }) |field| try std.testing.expectEqual(first.dptr(@field(layout, field)), second.dptr(@field(layout, field)));
    try std.testing.expectEqual(@as(usize, 546127872), persistent.total);
    try std.testing.expect(persistent.total < layout.total);
    try std.testing.expect(host_layout.total > rows * dims * 2);
    try std.testing.expect(!window_.admits(layout.total - 1, 0, layout.total));
    try std.testing.expect(window_.admits(layout.total, 0, layout.total));
}

test "exl3: two-chunk memory refusal performs no CUDA allocation" {
    var runtime: state.Runtime = .{};
    const graph = @import("../graph/cuda_exl3_graph.zig");
    const saved_force = graph.force;
    defer {
        graph.force = saved_force;
    }
    graph.force = .off;
    runtime.options = .{ .two_chunk = true };
    var driver: cuda.Driver = undefined;
    var buffers: Deep = undefined;
    buffers.dev = .{ .d = &driver, .ptr = 0x100000, .len = 1 };
    buffers.world = 2;
    buffers.rows = 4096;
    buffers.lay = Layout.of(4096, 2);
    buffers.second = null;
    runtime.deep = &buffers;
    try std.testing.expectEqual(@as(u64, 0), try reserveTwoChunk(&runtime, &driver, Persistent.of(4096).total - 1, 0, 0));
    try std.testing.expect(buffers.second == null);
}

test "exl3: two-chunk PLE input staging advances mutable history only in chronological PLE segments" {
    const Fixture = struct {
        calls: usize = 0,
        before: [2][2]i64 = undefined,

        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }

        fn copy(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }

        pub fn rows(self: *@This(), history: []i64, ids: []const u32, embedding: []u16, sums: []f32) !void {
            self.before[self.calls] = history[0..2].*;
            self.calls += 1;
            for (ids) |id| {
                history[0] = history[1];
                history[1] = id;
            }
            @memset(embedding, 0);
            @memset(sums, 0);
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuStreamSynchronize = Fixture.sync;
    driver.api.cuMemcpyHtoDAsync_v2 = Fixture.copy;
    const host_layout = HostLayout.of(256);
    const host = try std.testing.allocator.alignedAlloc(u8, .@"16", host_layout.total);
    defer std.testing.allocator.free(host);
    var buffers: Deep = undefined;
    buffers.rows = 256;
    buffers.shared = null;
    buffers.lay = Layout.of(256, 2);
    buffers.hl = host_layout;
    buffers.host = .{ .d = &driver, .bytes = host };
    buffers.dev = .{ .d = &driver, .ptr = 0x100000, .len = buffers.lay.total };
    var fixture: Fixture = .{};
    var history: [2]i64 = .{ -2, -1 };
    var pass = .{ .stream = cuda.Stream{ .d = &driver, .handle = null }, .m = .{ .ngram = &fixture }, .seq = .{ .history = &history } };
    var ids: [512]u32 = undefined;
    for (&ids, 1..) |*id, value| id.* = @intCast(value);
    for (0..2) |index| try windowInputs(&buffers, &pass, ids[index * 256 ..][0..256], index * 256, 0);
    try std.testing.expectEqual(@as(usize, 0), fixture.calls);
    try std.testing.expectEqualSlices(i64, &.{ -2, -1 }, &history);
    for (0..2) |index| try windowInputs(&buffers, &pass, ids[index * 256 ..][0..256], index * 256, ple_layer);
    try std.testing.expectEqual(@as(usize, 2), fixture.calls);
    try std.testing.expectEqualSlices(i64, &.{ -2, -1 }, &fixture.before[0]);
    try std.testing.expectEqualSlices(i64, &.{ 255, 256 }, &fixture.before[1]);
    try std.testing.expectEqualSlices(i64, &.{ 511, 512 }, &history);
}

test "exl3: prefill passes keep each expert's pairs together, items relative to their pass" {
    var dp: Deep = undefined;
    // pairs 0..8: experts 5, 512, 2 / 2, 512, 7 / 5, 512, 5 (-1 and out-of-table ids skipped)
    const picks = [_]i32{ 5, 512, 2, 2, 512, 7, 5, 512, 5, -1, 600 };
    @memset(&dp.cnt, 0);
    for (picks) |pk_| {
        if (pk_ < 0 or pk_ >= table_n) continue;
        dp.cnt[@intCast(pk_)] += 1;
    }
    // pass 0: the shared expert and 5; pass 1: 2 and 7
    dp.order[0] = shared_id;
    dp.order[1] = 5;
    dp.order[2] = 2;
    dp.order[3] = 7;
    dp.passes[0] = .{ .id_lo = 0, .id_hi = 2, .m_lo = 0, .m_n = 0, .i_lo = 0, .i_n = 0 };
    dp.passes[1] = .{ .id_lo = 2, .id_hi = 4, .m_lo = 0, .m_n = 0, .i_lo = 0, .i_n = 0 };
    dp.n_passes = 2;
    var members: [16]i32 = undefined;
    var items: [3 * 16]i32 = undefined;
    const c = build(&dp, &picks, &members, &items);
    try std.testing.expectEqual(@as(usize, 9), c.members);
    try std.testing.expectEqual(@as(usize, 4), c.items);
    try std.testing.expectEqualSlices(i32, &.{ 1, 4, 7, 0, 6, 8, 2, 3, 5 }, members[0..9]);
    try std.testing.expectEqual(Pass{ .id_lo = 0, .id_hi = 2, .m_lo = 0, .m_n = 6, .i_lo = 0, .i_n = 2 }, dp.passes[0]);
    try std.testing.expectEqual(Pass{ .id_lo = 2, .id_hi = 4, .m_lo = 6, .m_n = 3, .i_lo = 2, .i_n = 2 }, dp.passes[1]);
    // (expert, first within its pass, count)
    try std.testing.expectEqualSlices(i32, &.{ 512, 0, 3, 5, 3, 3, 2, 0, 2, 7, 2, 1 }, items[0..12]);
}

test "exl3: expert-parallel prefill masks the other rank's picks and keeps the shared expert on rank 1" {
    const r0 = try tp_.Rank.init(2, 0);
    const r1 = try tp_.Rank.init(2, 1);
    const picks = [_]i32{ 5, 300, 512, -1, 255, 256, 511, 0 };
    var a = picks;
    maskOwned(&a, r0);
    try std.testing.expectEqualSlices(i32, &.{ 5, -1, -1, -1, 255, -1, -1, 0 }, &a);
    var b = picks;
    maskOwned(&b, r1);
    try std.testing.expectEqualSlices(i32, &.{ -1, 300, 512, -1, -1, 256, 511, -1 }, &b);
    var c = picks;
    maskOwned(&c, null);
    try std.testing.expectEqualSlices(i32, &picks, &c);
    // every pick lands on exactly one rank
    for (picks, a, b) |p, x, y| if (p >= 0) try std.testing.expect((x == p) != (y == p));
    // two ranks add the gathered partials: one [rows, D] fp32 per rank
    try std.testing.expectEqual(deviceBytesAt(4096, 1) + 2 * 4096 * 2560 * 4, deviceBytesAt(4096, 2));
}

test "exl3: prefetch history follows deep engine ownership reuse replacement and release" {
    const Fixture = struct {
        fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn stream(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn freeHost(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn refuse(_: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
            return cuda.abi.error_not_ready;
        }
        fn install(runtime: *state.Runtime, driver: *const cuda.Driver, owner: *const anyopaque) !*Deep {
            const buffers = try std.testing.allocator.create(Deep);
            buffers.* = .{
                .runtime = runtime,
                .gpa = std.testing.allocator,
                .owner = owner,
                .rows = 4096,
                .lay = undefined,
                .hl = undefined,
                .dev = .{ .d = driver, .ptr = 0, .len = 0 },
                .host = .{ .d = driver, .bytes = @constCast(&.{}) },
                .kern = undefined,
                .aux = .{ .d = driver, .handle = null },
            };
            buffers.kern.mod = .{ .d = driver, .handle = null };
            runtime.deep = buffers;
            return buffers;
        }
    };
    var runtime: state.Runtime = .{};
    try std.testing.expect(runtime.deep == null);
    defer release(&runtime);
    var driver: cuda.Driver = undefined;
    driver.api.cuModuleUnload = Fixture.unload;
    driver.api.cuStreamSynchronize = Fixture.stream;
    driver.api.cuStreamDestroy_v2 = Fixture.stream;
    driver.api.cuMemFreeHost = Fixture.freeHost;
    driver.api.cuMemAlloc_v2 = Fixture.refuse;
    var owners: [2]u8 = .{ 0, 0 };
    const first = try Fixture.install(&runtime, &driver, &owners[0]);
    first.history[3] = .{ .rows = 4096, .used = 80 };
    first.history_shape = .{ .sub = 256, .lo = 0, .hi = 256 };
    releaseFor(&runtime, &owners[1]);
    try std.testing.expect(runtime.deep.? == first);
    try reserveAt(std.testing.allocator, &driver, &runtime, &owners[0], 2048, 1);
    try std.testing.expectEqual(@as(usize, 96), history_.want(runtime.deep.?.history[3], 4096, 400, routed_n_));
    try std.testing.expectError(error.NotReady, reserveAt(std.testing.allocator, &driver, &runtime, &owners[1], 4096, 1));
    try std.testing.expect(runtime.deep == null);
    const second = try Fixture.install(&runtime, &driver, &owners[1]);
    try std.testing.expectEqual(@as(usize, 416), history_.want(second.history[3], 4096, 400, routed_n_));
    try std.testing.expect(second.history_shape == null);
    second.history[3] = .{ .rows = 4096, .used = 80 };
    try std.testing.expectError(error.NotReady, reserveAt(std.testing.allocator, &driver, &runtime, &owners[1], 4096, 2));
    try std.testing.expect(runtime.deep == null);
    const third = try Fixture.install(&runtime, &driver, &owners[1]);
    try std.testing.expectEqual(@as(usize, 416), history_.want(third.history[3], 4096, 400, routed_n_));
    third.history[3] = .{ .rows = 4096, .used = 80 };
    releaseFor(&runtime, &owners[1]);
    try std.testing.expect(runtime.deep == null);
    const fourth = try Fixture.install(&runtime, &driver, &owners[1]);
    try std.testing.expectEqual(@as(usize, 416), history_.want(fourth.history[3], 4096, 400, routed_n_));
    var other_runtime: state.Runtime = .{};
    defer release(&other_runtime);
    const other = try Fixture.install(&other_runtime, &driver, &owners[0]);
    releaseFor(&runtime, &owners[1]);
    try std.testing.expect(runtime.deep == null);
    try std.testing.expect(other_runtime.deep.? == other);
    releaseFor(&other_runtime, &owners[1]);
    try std.testing.expect(other_runtime.deep.? == other);
    releaseFor(&other_runtime, &owners[0]);
    try std.testing.expect(other_runtime.deep == null);
}

test "exl3: the prefill tiles cut K as decode does at Flash Next's shapes, and the two expert groups split at 256" {
    const pk = @import("cuda_exl3_prefill_k.zig");
    try std.testing.expect(pk.sumsAsDecode(dims, wt.moe_w));
    try std.testing.expect(!pk.sumsAsDecode(dims + 64, wt.moe_w + 64)); // no plan of four 8-tile warps: refused
    // decode's plan here: gate/up in 4 splits of 4 warps, down in 4 warps (chains of 10 k tiles)
    try std.testing.expectEqual(@as(u32, 4), try cuda.exl3_experts.plan(dims, wt.moe_w, true));
    try std.testing.expectEqual(@as(u32, 1), try cuda.exl3_experts.plan(wt.moe_w, dims, false));
}

test "exl3: leaves the any-degree reduction tests run with the prefill tests" {
    _ = @import("cuda_exl3_prefill_leaves.zig");
}
