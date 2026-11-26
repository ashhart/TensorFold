//! Host-only async commit failure latching tests for Inkling 2 Mini Metal backend and cache.
const std = @import("std");
const lanes = @import("lanes");
const cfg = @import("../config.zig");
const model_ = @import("model.zig");
const st = @import("state.zig");
const backend_ = @import("backend.zig");

const gpa = std.testing.allocator;
const taps_ids = [_]u32{ 5, 23, 41, 48 };

fn testConfig() cfg.Config {
    return .{
        .text = .{
            .hidden = 2304,
            .layers = 24,
            .vocab = 201024,
            .unpadded_vocab = 200058,
            .local = .{ .q = 24, .kv = 8, .dim = 128 },
            .global = .{ .q = 24, .kv = 8, .dim = 128 },
            .window = 1024,
            .d_rel = 16,
            .rel_extent = 1024,
            .log_floor = 8192,
            .log_alpha = 0.1,
            .eps = 1e-6,
            .qk_clip = 22.627416997969522,
            .rel_clip = 8,
            .block = 6,
            .experts = 128,
            .topk = 8,
            .expert_inter = 1152,
            .route_scale = 8,
            .gate_cap = 4,
            .up_cap = 25,
            .logit_cap = 30,
            .max_context = 1048576,
            .loop = .{ .core_start = 6, .core_end = 18, .passes = 3, .gate_rank = 64 },
        },
        .vision = undefined,
        .mtp = .{
            .layers = 1,
            .q_heads = 24,
            .kv_heads = 8,
            .head_dim = 128,
            .inter = 1152,
            .rope_theta = 10000,
            .window = 1024,
            .local = @splat(true),
            .block = 6,
            .rank = 64,
            .taps = .{ 5, 23, 41, 48 } ++ @as([12]u32, @splat(0)),
            .taps_n = 4,
        },
        .eos = 1,
        .image_token = cfg.default_image_token,
    };
}

fn fakeCache(len: usize, commit_seq: u64, tap_gen: u64) st.Cache {
    return .{
        .kv = undefined,
        .conv = undefined,
        .taps = undefined,
        .logits = undefined,
        .tap = .{ .buf = .{ .id = @ptrFromInt(0x1000) }, .start = len - 1, .rows = 1, .gen = tap_gen },
        .key_at = undefined,
        .value_at = undefined,
        .rings = undefined,
        .globals = undefined,
        .capacity = 128,
        .len = len,
        .commit_seq = commit_seq,
    };
}

test "Cache commit failure latch poisons all transaction operations and clears held taps" {
    var c = fakeCache(12, 3, 5);
    try c.ensureUsable();
    try c.begin(4);
    try std.testing.expectEqual(@as(?usize, 4), c.pending);
    try c.rollback();
    try std.testing.expectEqual(@as(?usize, null), c.pending);

    c.tap = .{ .buf = .{ .id = @ptrFromInt(0x1000) }, .start = 11, .rows = 1, .gen = 5 };
    c.markCommitFailed();
    try std.testing.expect(c.commit_failed);
    try std.testing.expect(c.tap.buf == null);
    try std.testing.expectEqual(@as(usize, 0), c.tap.rows);
    try std.testing.expectError(error.GpuFailed, c.ensureUsable());
    try std.testing.expectError(error.GpuFailed, c.begin(2));
    try std.testing.expectError(error.GpuFailed, c.beginTree(2, &.{ -1, 0 }));
    try std.testing.expectError(error.GpuFailed, c.commit(1));
    try std.testing.expectError(error.GpuFailed, c.finishCommitPath(&.{0}));
    try std.testing.expectError(error.GpuFailed, c.rollback());
}

test "Host backend latches sticky failure across all caches in a failed commit batch when another stream prefill settles it" {
    var fake_model: model_.Model = undefined;
    fake_model.config = testConfig();
    var target: backend_.Metal = .{
        .gpa = gpa,
        .model = &fake_model,
        .options = .{ .capacity = 128, .streams = 4, .budget = 16 * st.cacheBytes(128) },
        .scratch = undefined,
        .prompt = undefined,
        .tokens = &.{},
        .values = &.{},
        .prepared = true,
        .tap_gen = 9,
        .commit_seq = 7,
        .commit_active_seq = 7,
        .test_fail_commit = true,
    };
    defer target.caches.deinit(gpa);

    var s0 = try lanes.Stream.init(gpa, .{ .id = "healthy", .prompt = &.{ 10, 11, 12 }, .max_new = 16 });
    defer s0.deinit(gpa);
    var s1 = try lanes.Stream.init(gpa, .{ .id = "batch_a", .prompt = &.{ 20, 21, 22 }, .max_new = 16 });
    defer s1.deinit(gpa);
    var s2 = try lanes.Stream.init(gpa, .{ .id = "batch_b", .prompt = &.{ 30, 31, 32 }, .max_new = 16 });
    defer s2.deinit(gpa);
    var s3 = try lanes.Stream.init(gpa, .{ .id = "new_prefill", .prompt = &.{ 40, 41, 42 }, .max_new = 16 });
    defer s3.deinit(gpa);

    var c0 = fakeCache(10, 6, 9);
    var c1 = fakeCache(14, 7, 9);
    var c2 = fakeCache(18, 7, 9);
    try target.caches.put(gpa, &s0, &c0);
    try target.caches.put(gpa, &s1, &c1);
    try target.caches.put(gpa, &s2, &c2);

    const b = target.backend();
    // Another stream's prefill observes the async commit failure and clears the pending commit state.
    try std.testing.expectError(error.GpuFailed, b.prefill(&s3));
    try std.testing.expectEqual(@as(u64, 0), target.commit_active_seq);
    try std.testing.expect(!target.caches.contains(&s3));

    // Both caches from commit batch 7 are permanently poisoned; unrelated stream s0 stays usable.
    try std.testing.expect(c1.commit_failed and c2.commit_failed);
    try std.testing.expect(!c0.commit_failed);
    try std.testing.expect(c1.tap.buf == null and c2.tap.buf == null);
    try std.testing.expect(c0.tap.buf != null);

    // Every subsequent operation on BOTH affected streams fails closed with error.GpuFailed.
    for ([_]*lanes.Stream{ &s1, &s2 }, [_]*st.Cache{ &c1, &c2 }) |sm, cache| {
        const len = cache.len;
        try std.testing.expectError(error.GpuFailed, b.queue(sm, .{ .value = 5 }, len + 1));
        var sampled = [_]u32{ 0, 0 };
        var drafts = [_]u32{0};
        const pos = [_]u64{ len + 1, len + 2 };
        const win = [_]lanes.backend.Window{.{ .stream = sm, .pending = 5, .held = 0, .tokens = &.{6}, .parents = null, .positions = &pos }};
        var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &drafts }};
        try std.testing.expectError(error.GpuFailed, b.verify(&win, &out));
        try std.testing.expectError(error.GpuFailed, b.keep(&win, &.{&.{0}}));
        try std.testing.expectError(error.GpuFailed, b.features(sm, &taps_ids, len - 1, 1));
        try std.testing.expectError(error.GpuFailed, backend_.Metal.snapSave(&target, sm, @intCast(len)));
        try std.testing.expectError(error.GpuFailed, b.first(sm, len));
        try std.testing.expectError(error.GpuFailed, b.vtable.first_masked.?(b.ptr, sm, len, &.{}));
        try std.testing.expectError(error.GpuFailed, b.firstRow(sm));
        try std.testing.expectError(error.GpuFailed, b.prefill(sm));
        try std.testing.expectError(error.GpuFailed, target.feed(cache, &.{5}, null, null));
    }

    // Batched verify mixing healthy s0 and poisoned s1 fails closed without opening a pending window on s0.
    var s0_samp = [_]u32{0};
    var s1_samp = [_]u32{0};
    var empty_drafts: [0]u32 = .{};
    const mix_wins = [_]lanes.backend.Window{
        .{ .stream = &s0, .pending = 5, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{11} },
        .{ .stream = &s1, .pending = 5, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{15} },
    };
    var mix_out = [_]lanes.backend.Verified{
        .{ .sampled = &s0_samp, .drafts = &empty_drafts },
        .{ .sampled = &s1_samp, .drafts = &empty_drafts },
    };
    try std.testing.expectError(error.GpuFailed, b.verify(&mix_wins, &mix_out));
    try std.testing.expectEqual(@as(?usize, null), c0.pending);

    // Healthy stream s0 still serves features cleanly.
    const f0 = try b.features(&s0, &taps_ids, 9, 1);
    try std.testing.expectEqual(@as(u32, 1), f0.rows);
}

test "Host backend settles active commit on first and firstRow and latches failure on peers" {
    var fake_model: model_.Model = undefined;
    fake_model.config = testConfig();
    var target: backend_.Metal = .{
        .gpa = gpa,
        .model = &fake_model,
        .options = .{ .capacity = 128, .streams = 4, .budget = 16 * st.cacheBytes(128) },
        .scratch = undefined,
        .prompt = undefined,
        .tokens = &.{},
        .values = &.{},
        .prepared = true,
        .tap_gen = 3,
        .commit_seq = 2,
        .commit_active_seq = 2,
        .test_fail_commit = true,
    };
    defer target.caches.deinit(gpa);

    var s1 = try lanes.Stream.init(gpa, .{ .id = "s1", .prompt = &.{ 1, 2 }, .max_new = 8 });
    defer s1.deinit(gpa);
    var s2 = try lanes.Stream.init(gpa, .{ .id = "s2", .prompt = &.{ 3, 4 }, .max_new = 8 });
    defer s2.deinit(gpa);
    var c1 = fakeCache(5, 2, 3);
    var c2 = fakeCache(6, 2, 3);
    try target.caches.put(gpa, &s1, &c1);
    try target.caches.put(gpa, &s2, &c2);

    const b = target.backend();
    try std.testing.expectError(error.GpuFailed, b.first(&s1, 5));
    try std.testing.expect(c1.commit_failed and c2.commit_failed);
    try std.testing.expectError(error.GpuFailed, b.firstRow(&s2));
    try std.testing.expectError(error.GpuFailed, b.features(&s2, &taps_ids, 5, 1));
    try std.testing.expectError(error.GpuFailed, backend_.Metal.snapSave(&target, &s2, 6));
}
