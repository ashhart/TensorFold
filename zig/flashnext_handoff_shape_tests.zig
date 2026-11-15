const std = @import("std");
const mtl = @import("metal");
const Tp2 = @import("src/families/flashnext/tp.zig").Tp2;
const Handoff = @import("src/families/flashnext/tp_handoff.zig").Handoff;
const Job = @import("src/families/flashnext/tp_job.zig").Job;
const testing = std.testing;
const epoch: u64 = 0x1_0000_0003;
const cb_id = 73;
const Exchange = enum { wait, combine, argmax, plain, send };
const Ref = struct { b: mtl.Buffer, off: usize = 0 };

fn expectOp(ops: []const mtl.HandoffOp, at: *usize, expected: mtl.HandoffOp) !void {
    try testing.expect(at.* < ops.len);
    const actual = ops[at.*];
    at.* += 1;
    try testing.expectEqual(expected.kind, actual.kind);
    try testing.expectEqual(expected.cb_id, actual.cb_id);
    switch (expected.kind) {
        .compute, .end => try testing.expectEqual(expected.mode, actual.mode),
        .signal, .wait => {
            try testing.expectEqual(expected.event_id, actual.event_id);
            try testing.expectEqual(expected.seq, actual.seq);
        },
        .dispatch => {
            try testing.expectEqual(expected.mode, actual.mode);
            try testing.expectEqualStrings(expected.pipeline, actual.pipeline);
        },
    }
}

fn expectDispatch(ops: []const mtl.HandoffOp, at: *usize, mode: mtl.DispatchType, pipeline: []const u8) !void {
    try expectOp(ops, at, .{ .kind = .dispatch, .cb_id = cb_id, .mode = mode, .pipeline = pipeline });
}

fn dispatch(enc: mtl.ComputeEncoder, pipeline: []const u8) void {
    enc.setPipeline(.{ .name = pipeline });
    enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
}

fn shape(exchange: Exchange) !void {
    for ([_]mtl.DispatchType{ .serial, .concurrent }) |mode| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        mtl.allocator = arena.allocator();
        mtl.resetHandoffs();
        var tp: Tp2 = undefined;
        tp.rank = 0;
        tp.seq = epoch - 1;
        tp.seqs = .{};
        tp.queued = .init(tp.seq);
        tp.failed = .init(false);
        tp.stop = .init(false);
        tp.last_seq = epoch;
        tp.last_x = .{ .flag = 1, .slot = 0 };
        var jobs: [1024]Job = undefined;
        tp.jobs = &jobs;
        const b = try (mtl.Device{}).buffer(16, mtl.ResourceOptions.shared);
        tp.wbuf = b;
        tp.one = b;
        tp.pick_tmp = b;
        tp.partial = b;
        tp.copy_pipe = .{ .name = "tp_copy" };
        tp.branch_pipe = .{ .name = "tp_branch" };
        tp.argmax_pipe = .{ .name = "tp_argmax_part" };
        tp.pick_pipe = .{ .name = "tp_pick" };
        tp.merge_pipe = .{ .name = "tp_merge" };
        tp.plain_pipe = .{ .name = "tp_plain" };
        tp.handoff = try Handoff.init(.{}, &tp.failed, &tp.stop, tp.rank);
        defer tp.handoff.deinit();
        try testing.expect(tp.handoff.posted.id != tp.handoff.landed.id);
        mtl.record_handoffs = true;
        defer mtl.record_handoffs = false;
        const cb: mtl.CommandBuffer = .{ .id = cb_id };
        var enc = cb.compute(mode);
        dispatch(enc, "shape_producer");
        const ref: Ref = .{ .b = b };
        switch (exchange) {
            .wait => tp.waitWord(&enc, 8, 1),
            .combine => tp.combine(&enc, ref, ref, ref, ref, ref, 2),
            .argmax => tp.argmax(&enc, ref, 32, 0, 16, ref, ref, 2),
            .plain => tp.plain(&enc, ref, ref, ref, ref, ref, 2),
            .send => tp.send(&enc, &.{}, 8, 1),
        }
        dispatch(enc, "shape_consumer");
        const ops = mtl.handoff_ops[0..mtl.handoff_op_count];
        var at: usize = 0;
        try expectOp(ops, &at, .{ .kind = .compute, .cb_id = cb_id, .mode = mode });
        try expectDispatch(ops, &at, mode, "shape_producer");
        switch (exchange) {
            .argmax => try expectDispatch(ops, &at, mode, "tp_argmax_part"),
            .plain => try expectDispatch(ops, &at, mode, "tp_copy"),
            else => {},
        }
        try expectOp(ops, &at, .{ .kind = .end, .cb_id = cb_id, .mode = mode });
        try expectOp(ops, &at, .{
            .kind = .signal,
            .cb_id = cb_id,
            .event_id = tp.handoff.posted.id,
            .seq = epoch,
        });
        try expectOp(ops, &at, .{
            .kind = .wait,
            .cb_id = cb_id,
            .event_id = tp.handoff.landed.id,
            .seq = epoch,
        });
        try expectOp(ops, &at, .{ .kind = .compute, .cb_id = cb_id, .mode = mode });
        switch (exchange) {
            .combine => try expectDispatch(ops, &at, mode, "tp_merge"),
            .argmax => try expectDispatch(ops, &at, mode, "tp_pick"),
            .plain => try expectDispatch(ops, &at, mode, "tp_plain"),
            else => {},
        }
        try expectDispatch(ops, &at, mode, "shape_consumer");
        try testing.expectEqual(ops.len, at);
        try testing.expectEqual(mode, enc.dispatchType());
    }
}

test "pair waitWord handoff preserves serial and concurrent encoder boundaries" {
    try shape(.wait);
}

test "pair combine handoff preserves serial and concurrent encoder boundaries" {
    try shape(.combine);
}

test "pair argmax handoff preserves serial and concurrent encoder boundaries" {
    try shape(.argmax);
}

test "pair plain handoff preserves serial and concurrent encoder boundaries" {
    try shape(.plain);
}

test "pair send handoff preserves serial and concurrent encoder boundaries" {
    try shape(.send);
}
