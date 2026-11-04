const std = @import("std");
const cuda = @import("cuda");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const ops_ = @import("cuda_mtp_ops.zig");

const Ops = ops_.Ops;
const Draft = mtp.Draft;
const max_rows = mtp.max_rows;
const Head = mtp.Head;
const vocab = @import("cuda_prompt_wt.zig").vocab;

test "exl3: greedy verification consumes completed records without touching the head or CUDA" {
    var records: [max_rows * 16]u8 align(16) = undefined;
    const values = std.mem.bytesAsSlice(f32, &records);
    const tokens = [_]u32{ 0, 6124, 9564, @intCast(vocab - 1) };
    for (tokens, 0..) |token, row| values[4 * row + 2] = @bitCast(@as(i32, @intCast(token)));
    var window: Draft = undefined;
    window.rows = tokens.len;
    window.fed0 = 37;
    window.verify_cache = .{};
    window.verify_host = .{ .d = undefined, .bytes = &records };
    const ops: Ops(struct {}) = .{ .e = undefined, .tri = .{}, .h = undefined };
    var drawn: [tokens.len]u32 = undefined;
    window.verify_cache.publish(window.verify_cache.generation, window.rows, window.fed0);
    try ops.draws(&window, null, &.{}, &drawn);
    try std.testing.expectEqualSlices(u32, &tokens, &drawn);
    try std.testing.expect(window.verify_cache.ready == null);
    window.verify_cache.publish(window.verify_cache.generation, window.rows, window.fed0);
    try ops.draws(&window, .{ .seed = 0, .temperature = 0 }, &.{}, &drawn);
    try std.testing.expectEqualSlices(u32, &tokens, &drawn);
}

test "exl3: verification tail enqueues the existing top-1 kernel then only its records copy" {
    const Fake = struct {
        var launches: usize = 0;
        var copies: usize = 0;
        var rows: usize = 0;
        var copied: usize = 0;
        var fail_copy: bool = false;

        fn launch(_: cuda.abi.Function, grid: c_uint, _: c_uint, _: c_uint, block: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(block == 1024);
            std.debug.assert(launches == copies);
            const scores: *const u64 = @ptrCast(@alignCast(params.?[0].?));
            const width: *const c_int = @ptrCast(@alignCast(params.?[1].?));
            const stride: *const c_longlong = @ptrCast(@alignCast(params.?[2].?));
            const output: *const u64 = @ptrCast(@alignCast(params.?[3].?));
            std.debug.assert(scores.* == 0x2000 and width.* == vocab and stride.* == vocab and output.* == 0x4000);
            rows = grid;
            launches += 1;
            return 0;
        }

        fn copy(_: ?*anyopaque, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.debug.assert(launches == copies + 1 and src == 0x4000);
            copies += 1;
            copied = bytes;
            if (fail_copy) return cuda.abi.error_not_ready;
            return 0;
        }
    };
    Fake.launches = 0;
    Fake.copies = 0;
    Fake.fail_copy = false;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
    driver.api.cuMemcpyDtoHAsync_v2 = Fake.copy;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var head: Head = undefined;
    head.best_fn = .{ .d = &driver, .handle = null };
    var records: [max_rows * 16]u8 align(16) = undefined;
    var window: Draft = undefined;
    window.logits = .{ .d = &driver, .ptr = 0x2000, .len = max_rows * vocab * 2 };
    window.best = .{ .d = &driver, .ptr = 0x4000, .len = max_rows * 16 };
    window.verify_host = .{ .d = &driver, .bytes = &records };
    window.verify_cache = .{};
    for ([_]usize{ 1, 2, 3, 4, 8, 16 }) |rows| {
        window.rows = rows;
        window.verify_cache.invalidate();
        var tail: Ops(struct {}).VerifyTail = .{ .head = &head, .draft = &window };
        try Ops(struct {}).VerifyTail.enqueue(&tail, stream);
        try std.testing.expectEqual(rows, Fake.rows);
        try std.testing.expectEqual(rows * 16, Fake.copied);
        try std.testing.expect(tail.enqueued);
        try std.testing.expectEqual(window.verify_cache.generation, tail.generation);
        try std.testing.expect(window.verify_cache.ready == null);
    }
    Fake.fail_copy = true;
    var failed: Ops(struct {}).VerifyTail = .{ .head = &head, .draft = &window };
    try std.testing.expectError(error.NotReady, Ops(struct {}).VerifyTail.enqueue(&failed, stream));
    try std.testing.expect(failed.enqueued);
    try std.testing.expect(window.verify_pending != null);
    try std.testing.expect(window.verify_cache.ready == null);
}

test "exl3: common verification scopes TP1 and TP2 validation and delegation without joins" {
    const Tri = @import("cuda_launch.zig").Tri;
    const Fake = @import("exl3/model/cuda_exl3_fake_driver_test.zig").Fake;
    const Phase = @import("exl3/model/exl3_telemetry.zig").Phase;
    const Delegate = struct {
        engine: *prompt.Engine,
        calls: usize = 0,
        fail: bool = false,

        fn verify(ptr: *anyopaque, _: *prompt.Seq, _: *Draft, _: []const u32, _: ?batch.Hooks.OutputTail) !void {
            const delegate: *@This() = @ptrCast(@alignCast(ptr));
            delegate.calls += 1;
            try std.testing.expectEqual(Phase.verify, delegate.engine.telemetry.forward_phase);
            if (delegate.fail) return error.VerificationFailed;
        }

        fn forward(_: *anyopaque, _: *prompt.Seq, _: []const u32, _: batch.Hooks) !void {
            return error.UnexpectedDelegation;
        }

        fn rollback(_: *anyopaque, _: *prompt.Seq, _: *Draft, _: usize) !void {
            return error.UnexpectedDelegation;
        }
    };
    var driver = Fake.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.exl3_prefill = .{};
    engine.exl3_chain = .{};
    engine.stream = &stream;
    engine.round_graph = .{};
    var sequence: prompt.Seq = undefined;
    sequence.cap = max_rows;
    var window: Draft = undefined;
    window.verify_cache = .{};
    var delegate: Delegate = .{ .engine = &engine };
    var ops: Ops(Tri) = .{ .e = &engine, .tri = .{ .s = stream }, .h = undefined };
    const too_wide: [max_rows + 1]u32 = @splat(1);
    const syncs = Fake.sync_calls;
    for ([_]usize{ 1, 2 }) |world| {
        for ([_]bool{ false, true }) |enabled| {
            for ([_]Phase{ .unknown, .prefill, .decode, .verify }) |previous| {
                engine.telemetry = .{ .enabled = enabled, .rank = .{ .rank = 0, .world = world }, .forward_phase = previous };
                sequence.fed = 0;
                ops.main = null;
                try std.testing.expectError(error.WindowTooWide, ops.verifyWithTail(&sequence, &window, &.{}, null));
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                try std.testing.expectError(error.WindowTooWide, ops.verifyWithTail(&sequence, &window, &too_wide, null));
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                sequence.fed = sequence.cap;
                try std.testing.expectError(error.PromptTooLong, ops.verifyWithTail(&sequence, &window, &.{1}, null));
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                ops.main = .{ .ptr = &delegate, .rank = 0, .forward = Delegate.forward, .verify = Delegate.verify, .rollback = Delegate.rollback };
                const calls = delegate.calls;
                delegate.fail = false;
                try ops.verifyWithTail(&sequence, &window, &.{1}, null);
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                delegate.fail = true;
                try std.testing.expectError(error.VerificationFailed, ops.verifyWithTail(&sequence, &window, &.{1}, null));
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                try std.testing.expectEqual(calls + 2, delegate.calls);
                try std.testing.expectEqual(enabled, engine.telemetry.enabled);
                try std.testing.expectEqual(syncs, Fake.sync_calls);
            }
        }
    }
}

test "exl3: paged draft heads enter conditional graph dispatch" {
    const graph = @import("exl3/graph/cuda_exl3_graph.zig");
    const chain = @import("exl3/model/cuda_exl3_chain.zig");
    const Tri = @import("cuda_launch.zig").Tri;
    const Mock = struct {
        e: *prompt.Engine,
        h: *Head = undefined,
        tri: Tri = undefined,

        pub fn absorb(_: @This(), _: *Draft, _: []const u32, _: u64, _: bool) !void {
            return error.UnexpectedGraphAbsorb;
        }
        pub fn pick(_: @This(), _: *Draft, _: u64, _: ?mtp.Sampling) !mtp.Pick {
            return error.UnexpectedGraphPick;
        }
        pub fn forwardHead(_: @This(), _: *Draft, _: []const u32, _: u64, _: bool) !void {
            return error.UnexpectedGraphForward;
        }
    };
    const Pool = @import("exl3/offload/cuda_exl3_pool.zig").Pool;
    var pool: Pool = undefined;
    pool.opt.layer = 0;
    var pools = [_]*Pool{&pool};
    var plan: @import("exl3/dplan/cuda_exl3_dplan.zig").DevPlan = undefined;
    plan.pools = &pools;
    var engine: prompt.Engine = undefined;
    engine.n_layers = 0;
    engine.gpa = std.testing.allocator;
    engine.driver = undefined;
    engine.exl3_dplan = .{ .plan = &plan, .mtp_pool = &pool };
    var d: Draft = undefined;
    d.media = null;
    d.sel = null;
    d.pg = .{ .pt = 0x1000, .ps = 4096, .ptok = 256 };
    var out: mtp.Chain = .{};
    const previous = graph.force;
    graph.force = .graph;
    defer graph.force = previous;
    try std.testing.expectError(error.UnexpectedGraphAbsorb, chain.draft(Tri, Mock{ .e = &engine }, &d, &.{1}, 0, 0, 2, null, 0, &out));
}
