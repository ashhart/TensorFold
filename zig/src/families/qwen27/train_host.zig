//! Host-only learner ABI, reservation and native cache-epoch regressions.
const std = @import("std");
const to = @import("train_ops.zig");
const config = @import("config.zig");

test "qwen27 learning: Metal argument layouts match the generated wrappers" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(to.Shape));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(to.Shape, "f"));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(to.GdnShape));
    try std.testing.expectEqual(@as(usize, 28), @sizeOf(to.Heads));
    try std.testing.expectEqual(@as(usize, 20), @offsetOf(to.Heads, "scale"));
}

test "qwen27 learning: the reserve covers both native frame widths and every GDN checkpoint" {
    var c: config.Config = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 10000000 };
    for (0..64) |l| c.kinds[l] = if (l % 4 == 3) .attention else .linear;
    const small = try @import("train_memory.zig").bytes(c, 128, true);
    const wide = try @import("train_memory.zig").bytes(c, 2048, true);
    try std.testing.expect(wide > small);
    const checkpoints = (1024 / 4 + 8) * c.v_heads * c.dv * c.dk * 4;
    try std.testing.expect(small > checkpoints * 39);
    try std.testing.expect(wide < 128 << 30);
}

test "qwen27 learning: moving weights invalidates standing, rewind metadata, draft trees and the pending round" {
    const Fake = struct {
        gpa: std.mem.Allocator,
        standing: struct {
            tokens: usize = 17,
            rewind_at: u32 = 11,
            pub fn deinit(s: *@This(), _: std.mem.Allocator) void {
                s.* = .{ .tokens = 0, .rewind_at = 0 };
            }
        } = .{},
        trees: usize = 3,
        round: bool = true,
        pub fn dropTree(f: *@This()) void {
            f.trees = 0;
        }
        pub fn dropRound(f: *@This()) void {
            f.round = false;
        }
    };
    var fake: Fake = .{ .gpa = std.testing.allocator };
    @import("train_epoch.zig").changed(&fake);
    try std.testing.expectEqual(@as(usize, 0), fake.standing.tokens);
    try std.testing.expectEqual(@as(u32, 0), fake.standing.rewind_at);
    try std.testing.expectEqual(@as(usize, 0), fake.trees);
    try std.testing.expect(!fake.round);
}

test "qwen27 learning: detached disk rows retain their server owner until close" {
    const Row = struct { stopped: bool = false };
    const Store = struct {
        disk: ?*Row,
        pub fn clear(s: *@This()) void {
            if (s.disk) |r| r.stopped = true;
            s.disk = null;
        }
    };
    var rows: Row = .{};
    var host: struct { disk_rows: ?*Row } = .{ .disk_rows = &rows };
    var store: Store = .{ .disk = &rows };
    store.clear();
    try std.testing.expect(rows.stopped and store.disk == null);
    const owned = @import("train_epoch.zig").takeDisk(&host);
    try std.testing.expect(owned.? == &rows);
    try std.testing.expect(@import("train_epoch.zig").takeDisk(&host) == null);
}

const schedule = @import("train_schedule.zig");
const mode_api = @import("living").train_mode;

test "qwen27 learning: only answer rows run the vocabulary forward and backward" {
    const fake = struct {
        g: struct { hidden: usize = 64, vocab: usize = 248320 } = .{},
        rows: usize = 12,
        answers: [2]mode_api.Answer = .{ .{ .first = 3, .at = 0, .rows = 2 }, .{ .first = 10, .at = 2, .rows = 2 } },
        nseq: usize = 2,
        dlogits: to.Ref = .{ .buffer = undefined, .offset = 0 },
        head: to.Q4 = undefined,
        dy: to.Ref = .{ .buffer = undefined, .offset = 0 },
    }{};
    const Ops = struct {
        rows: *usize,
        zeros: *usize,
        pub fn q4Back(o: @This(), _: to.Ref, _: to.Q4, _: to.Ref, rows: usize, add: bool) !void {
            try std.testing.expect(!add);
            o.rows.* += rows;
        }
        pub fn zero(o: @This(), _: to.Ref, count: usize) !void {
            o.zeros.* += count;
        }
    };
    var rows: usize = 0;
    var zeros: usize = 0;
    try schedule.headBack(&fake, Ops{ .rows = &rows, .zeros = &zeros });
    try std.testing.expectEqual(@as(usize, 4), rows);
    try std.testing.expectEqual(@as(usize, 12 * 64), zeros);
    try std.testing.expect(!schedule.head(.target, 0, 4));
    try std.testing.expect(!schedule.head(.loss, 2, 4));
    try std.testing.expect(schedule.head(.target, 3, 4));
    try std.testing.expect(!schedule.head(.keys, 3, 4));
}

test "qwen27 learning: one submission per native learner row preserves the serving interval" {
    try std.testing.expectEqual(@as(usize, 0), schedule.commitEvery(true));
    try std.testing.expectEqual(@as(usize, 4), schedule.commitEvery(false));
}

test "qwen27 learning: four independent gradient rows share one affine weight walk" {
    try std.testing.expectEqual(@as(usize, 2 * 64), schedule.q4Threads(7, 64));
    try std.testing.expectEqual(@as(usize, 64), schedule.q4Threads(1, 64));
    try std.testing.expectEqual(@as(usize, 0), schedule.q4Threads(0, 64));
}

const submit = @import("train_submit.zig");
test "qwen27 learning: a native row commits and waits once without a device allocation" {
    const Fake = struct {
        buffers: usize = 0,
        encoders: usize = 0,
        ended: usize = 0,
        commits: usize = 0,
        waits: usize = 0,
        layers: usize = 0,
        encode_fail: bool = false,
        gpu_fail: bool = false,
        const Encoder = struct {
            fake: *ThisFake,
            pub fn end(e: @This()) void {
                e.fake.ended += 1;
            }
        };
        const ThisFake = @This();
        pub fn commandBuffer(f: *@This()) *@This() {
            f.buffers += 1;
            return f;
        }
        pub fn compute(f: *@This(), _: enum { concurrent }) Encoder {
            f.encoders += 1;
            return .{ .fake = f };
        }
        pub fn commit(f: *@This()) void {
            f.commits += 1;
        }
        pub fn wait(f: *@This()) void {
            f.waits += 1;
        }
        pub fn failure(f: *@This()) ?[]const u8 {
            return if (f.gpu_fail) "injected" else null;
        }
        pub fn gpuSeconds(_: *@This()) f64 {
            return 0.25;
        }
        pub fn encode(f: *@This(), _: Encoder) !void {
            for (0..64) |_| f.layers += 1;
            if (f.encode_fail) return error.InjectedEncode;
        }
    };
    var f: Fake = .{};
    var stats: submit.Stats = .{};
    try std.testing.expectEqual(@as(f64, 0.25), try submit.run(&f, &f, &stats, null));
    try std.testing.expectEqual(@as(usize, 64), f.layers);
    try std.testing.expectEqual(@as(usize, 1), f.buffers);
    try std.testing.expectEqual(@as(usize, 1), f.ended);
    try std.testing.expectEqual(@as(usize, 1), f.commits);
    try std.testing.expectEqual(@as(usize, 1), f.waits);
    try std.testing.expectEqual(@as(usize, 1), stats.buffers);
    try std.testing.expectEqual(@as(usize, 1), stats.waits);
    f = .{ .encode_fail = true };
    try std.testing.expectError(error.InjectedEncode, submit.run(&f, &f, &stats, null));
    try std.testing.expectEqual(@as(usize, 1), f.ended);
    try std.testing.expectEqual(@as(usize, 0), f.commits);
    try std.testing.expectEqual(@as(usize, 0), f.waits);
    f = .{ .gpu_fail = true };
    try std.testing.expectError(error.LearningGpuFailure, submit.run(&f, &f, &stats, null));
    try std.testing.expectEqual(@as(usize, 1), f.ended);
    try std.testing.expectEqual(@as(usize, 1), f.commits);
    try std.testing.expectEqual(@as(usize, 1), f.waits);
}

test "qwen27 learning: row metadata is identical without the nine Round allocations" {
    const Round = @import("round_plan.zig").Round;
    const Row = @import("train_row.zig").Row;
    for ([_]u32{ 0, 1, 127, 511 }) |at| for ([_]u32{ 0, 7 }) |slot| {
        var old = try Round.init(std.testing.allocator, &.{.{ .slot = slot, .start = at, .capacity = 513, .ids = &.{17} }}, 4, 8, 32);
        defer old.deinit();
        var row: Row = .{};
        const view = try row.view(std.testing.failing_allocator, 17, slot, at, 513, 4, 8, 32);
        try std.testing.expectEqualSlices(u32, old.ids, view.ids);
        try std.testing.expectEqualSlices(i32, old.feed_positions, view.feed_positions);
        try std.testing.expectEqualSlices(u64, old.draw_positions, view.draw_positions);
        try std.testing.expectEqualSlices(u32, old.slots, view.slots);
        try std.testing.expectEqualSlices(u32, old.firsts, view.firsts);
        try std.testing.expectEqual(old.windows[0].start, view.windows[0].start);
        try std.testing.expectEqual(old.windows[0].taps, view.windows[0].taps);
        try std.testing.expectEqualSlices(i32, old.windows[0].parents, view.windows[0].parents);
        try std.testing.expectEqualSlices(u32, old.windows[0].depths, view.windows[0].depths);
        try std.testing.expectEqualSlices(u32, old.windows[0].conv, view.windows[0].conv);
    };
    try std.testing.expectError(error.OutOfMemory, Round.init(std.testing.failing_allocator, &.{.{ .slot = 0, .start = 0, .capacity = 513, .ids = &.{17} }}, 4, 8, 32));
    var row: Row = .{};
    try std.testing.expectError(error.BadRoundPlan, row.view(std.testing.failing_allocator, 1, 8, 0, 513, 4, 8, 32));
    try std.testing.expectError(error.BadRoundPlan, row.view(std.testing.failing_allocator, 1, 0, 513, 513, 4, 8, 32));
    try std.testing.expectError(error.BadTokenId, row.view(std.testing.failing_allocator, 32, 0, 0, 513, 4, 8, 32));
}

test {
    _ = @import("train_scalars.zig");
}

test "qwen27 learning: server admission keeps the 9B and pair shapes out of the unqualified learner" {
    var c: config.Config = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 10000000 };
    for (0..c.layers) |l| c.kinds[l] = if (l % 4 == 3) .attention else .linear;
    const admits = @import("train_admission.zig").supported;
    try std.testing.expect(admits(c));
    c.hidden = 4096;
    try std.testing.expect(!admits(c));
    c.hidden = 5120;
    c.ranks = 2;
    try std.testing.expect(!admits(c));
    c.ranks = 1;
    c.kinds[3] = .linear;
    try std.testing.expect(!admits(c));
}
