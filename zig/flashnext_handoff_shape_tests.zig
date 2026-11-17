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

const tp_local = @import("src/families/flashnext/tp_local.zig");
const tp_seq = @import("src/families/flashnext/tp_seq.zig");
const D = 2560;
const Bufs = struct { y: Ref, wts: Ref, lg: Ref, h: Ref, inj: Ref, out: Ref, ssp: Ref, rows: Ref };

fn pair(tp: *Tp2, rank: u32, jobs: *[1024]Job) !void {
    tp.* = undefined;
    tp.rank = rank;
    tp.seq = epoch - 1;
    tp.seqs = .{};
    tp.queued = .init(tp.seq);
    tp.failed = .init(false);
    tp.stop = .init(false);
    tp.jobs = jobs;
    tp.wbuf = try (mtl.Device{}).buffer(64, mtl.ResourceOptions.shared);
    tp.branches = try (mtl.Device{}).buffer(tp_local.BYTES, mtl.ResourceOptions.shared);
    tp_local.arm(tp.branches);
    tp.branch_pipe = .{ .name = "tp_branch" };
    tp.merge_pipe = .{ .name = "tp_merge" };
    inline for (.{ "copy", "argmax", "pick", "plain", "ids" }, .{ "tp_copy", "tp_argmax_part", "tp_pick", "tp_plain", "tp_ids" }) |field, name|
        @field(tp, field ++ "_pipe") = .{ .name = name };
    tp.partial = try (mtl.Device{}).buffer(64, mtl.ResourceOptions.shared);
    tp.pick_tmp = try (mtl.Device{}).buffer(16, mtl.ResourceOptions.shared);
    tp.handoff = try Handoff.init(.{}, &tp.failed, &tp.stop, rank);
}

/// The MoE stage's dispatches for one layer, exchanged (exchange then combine) or local, in recorded order.
fn moe(tp: *Tp2, mode: mtl.DispatchType, bufs: Bufs, rows: usize, local: bool) ![]const mtl.HandoffOp {
    mtl.resetHandoffs();
    mtl.record_handoffs = true;
    defer mtl.record_handoffs = false;
    var enc = (mtl.CommandBuffer{ .id = cb_id }).compute(mode);
    if (local) {
        tp_local.branch(tp, enc, bufs.y, bufs.wts, bufs.lg, rows);
        tp_local.merge(tp, enc, bufs.h, bufs.inj, bufs.out, bufs.ssp, bufs.rows, rows);
    } else {
        tp.exchange(enc, bufs.y, bufs.wts, bufs.lg, rows);
        tp.combine(&enc, bufs.h, bufs.inj, bufs.out, bufs.ssp, bufs.rows, rows);
    }
    return try mtl.allocator.dupe(mtl.HandoffOp, mtl.handoff_ops[0..mtl.handoff_op_count]);
}

fn word(op: mtl.HandoffOp, at: usize, comptime T: type) T {
    return std.mem.bytesToValue(T, op.args[at].bytes[0..@sizeOf(T)]);
}

test "pair decode MoE: local branches and merge reuse the exchange's arguments with no job, seam or event" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    mtl.allocator = arena.allocator();
    var bufs: Bufs = undefined;
    inline for (@typeInfo(Bufs).@"struct".field_names) |name| @field(bufs, name) = .{ .b = try (mtl.Device{}).buffer(16, mtl.ResourceOptions.shared) };
    var jobs: [1024]Job = undefined;
    for ([_]mtl.DispatchType{ .serial, .concurrent }) |mode| for (1..17) |rows| {
        var exchanged: [2][]const mtl.HandoffOp = undefined;
        var ready: [2]usize = undefined;
        for (0..2) |k| {
            var tp: Tp2 = undefined;
            try pair(&tp, @intCast(k), &jobs);
            defer tp.handoff.deinit();
            exchanged[k] = try moe(&tp, mode, bufs, rows, false);
            ready[k] = tp.handoff.ready.id;
        }
        for (0..2) |k| {
            var tp: Tp2 = undefined;
            try pair(&tp, @intCast(k), &jobs);
            defer tp.handoff.deinit();
            const ops = try moe(&tp, mode, bufs, rows, true);
            try testing.expectEqual(epoch - 1, tp.seq);
            try testing.expectEqual(epoch - 1, tp.queued.load(.acquire));
            try testing.expectEqual(tp_seq.Seqs{}, tp.seqs);
            try testing.expect(ops[0].kind == .compute);
            const ranges: usize = if (rows > 1) 2 else 1;
            try testing.expectEqual(1 + ranges + 1, ops.len);
            for (ops[1..]) |op| try testing.expect(op.kind == .dispatch and op.args[if (std.mem.eql(u8, op.pipeline, "tp_merge")) 0 else 5].id == tp.branches.id);
            const merge = ops[ops.len - 1];
            try testing.expectEqualStrings("tp_merge", merge.pipeline);
            const reference = for (exchanged[k]) |op| {
                if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "tp_merge")) break op;
            } else return error.NoExchangedMerge;
            try testing.expectEqual(ready[k], reference.args[0].id); // the exchanged merge reads peer rows only once published
            try testing.expectEqual(tp_local.PASS, word(merge, 1, u64));
            try testing.expectEqual(reference.grid, merge.grid);
            for ([_]usize{ 2, 3, 4, 7, 8, 9 }) |at| try testing.expectEqual(reference.args[at], merge.args[at]);
            const split = word(merge, 7, [4]u32);
            for (ops[1 .. ops.len - 1], 0..) |b, j| {
                try testing.expectEqualStrings("tp_branch", b.pipeline);
                const owner: u32 = if (rows > 1) @intCast(j) else 0;
                const theirs = for (exchanged[owner]) |op| {
                    if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "tp_branch")) break op;
                } else return error.NoExchangedBranch;
                try testing.expectEqual(theirs.grid, b.grid);
                for ([_]usize{ 0, 1, 2, 4 }) |at| try testing.expectEqual(theirs.args[at], b.args[at]);
                try testing.expectEqual(tp_local.PASS, word(b, 6, u64));
                const own = word(b, 4, [2]u32);
                try testing.expectEqual(tp_seq.ownRows(owner, rows), [2]usize{ own[0], own[1] });
                try testing.expectEqual(tp.branches.id, b.args[3].id);
                for (own[0]..own[1]) |r| { // tp_merge's index for row r lands on the byte tp_branch wrote for it
                    const is_mine = r >= split[0] and r < split[1];
                    try testing.expectEqual(owner == k, is_mine);
                    const read = if (is_mine) merge.args[5].off + (r - split[0]) * D * 2 else merge.args[6].off + (r - split[2]) * D * 2;
                    try testing.expectEqual(b.args[3].off + (r - own[0]) * D * 2, read);
                    try testing.expectEqual(tp.branches.id, merge.args[if (is_mine) 5 else 6].id);
                }
            }
        }
    };
}

test "pair decode MoE: the local guard word is the pair's own, and no published or cancelled epoch matches it" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    mtl.allocator = arena.allocator();
    mtl.resetHandoffs();
    var jobs: [1024]Job = undefined;
    var tp: Tp2 = undefined;
    try pair(&tp, 0, &jobs);
    defer tp.handoff.deinit();
    try testing.expectEqual(std.math.maxInt(u64), tp_local.PASS);
    try testing.expectEqual(tp_local.PASS, tp.branches.slice(u64, 1)[0]);
    try testing.expect(tp.branches.id != tp.handoff.ready.id and tp.branches.id != tp.wbuf.id);
    try testing.expect(tp.handoff.publish(epoch));
    try testing.expectEqual(epoch, tp.handoff.ready.slice(u64, 1)[0]);
    tp.handoff.cancel();
    try testing.expectEqual(std.math.maxInt(u64), tp.handoff.landed.value());
    try testing.expect(tp.handoff.ready.slice(u64, 1)[0] != tp_local.PASS);
}

/// One stage's dispatches, exchanged or local, in recorded order.
fn stage(tp: *Tp2, mode: mtl.DispatchType, bufs: Bufs, rows: usize, local: bool, which: enum { plain, pick, ids }) ![]const mtl.HandoffOp {
    mtl.resetHandoffs();
    mtl.record_handoffs = true;
    defer mtl.record_handoffs = false;
    var enc = (mtl.CommandBuffer{ .id = cb_id }).compute(mode);
    const parts = [2][2]usize{ .{ 0, 700 }, .{ 700, 300 } };
    switch (which) {
        .plain => if (local) tp_local.plain(tp, enc, bufs.h, bufs.inj, bufs.out, bufs.ssp, bufs.rows, rows) else tp.plain(&enc, bufs.h, bufs.inj, bufs.out, bufs.ssp, bufs.rows, rows),
        .pick => if (local) tp_local.argmax(tp, enc, bufs.lg, 1024, parts, bufs.y, bufs.rows, rows) else tp.argmax(&enc, bufs.lg, 1024, parts[tp.rank][0], parts[tp.rank][1], bufs.y, bufs.rows, rows),
        .ids => if (local) tp_local.ids(tp, enc, bufs.wts, bufs.out) else tp.mapIds(enc, bufs.wts, bufs.out),
    }
    return try mtl.allocator.dupe(mtl.HandoffOp, mtl.handoff_ops[0..mtl.handoff_op_count]);
}

fn named(ops: []const mtl.HandoffOp, pipeline: []const u8) !mtl.HandoffOp {
    for (ops) |op| if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, pipeline)) return op;
    return error.NoDispatch;
}

test "pair decode reduce and picks: local launches reuse the exchange's arguments and pass word, with no job or seam" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    mtl.allocator = arena.allocator();
    var bufs: Bufs = undefined;
    inline for (@typeInfo(Bufs).@"struct".field_names) |name| @field(bufs, name) = .{ .b = try (mtl.Device{}).buffer(16, mtl.ResourceOptions.shared) };
    var jobs: [1024]Job = undefined;
    for ([_]mtl.DispatchType{ .serial, .concurrent }) |mode| for ([_]usize{ 1, 2, 9, 16 }) |rows| for (0..2) |k| {
        var tp: Tp2 = undefined;
        try pair(&tp, @intCast(k), &jobs);
        defer tp.handoff.deinit();
        const pass: mtl.Arg = .{ .id = tp.branches.id };
        const exchanged = [3][]const mtl.HandoffOp{ try stage(&tp, mode, bufs, rows, false, .plain), try stage(&tp, mode, bufs, rows, false, .pick), try stage(&tp, mode, bufs, rows, false, .ids) };
        const seq = tp.seq;
        const plain = [2][]const mtl.HandoffOp{ exchanged[0], try stage(&tp, mode, bufs, rows, true, .plain) };
        const ref = try named(plain[0], "tp_plain");
        const sum = try named(plain[1], "tp_plain");
        try testing.expectEqual(@as(usize, 2), plain[1].len);
        try testing.expectEqual(ref.grid, sum.grid);
        for ([_]usize{ 2, 5, 6, 7, 8, 9 }) |at| try testing.expectEqual(ref.args[at], sum.args[at]);
        try testing.expectEqual((try named(plain[0], "tp_copy")).args[2], sum.args[3]); // the partial the exchange sends
        try testing.expectEqual(mtl.Arg{ .id = tp.branches.id, .off = tp_local.PARTIAL_AT }, sum.args[4]);
        try testing.expectEqual(pass, sum.args[0]);
        try testing.expectEqual(tp_local.PASS, word(sum, 1, u64));
        const picks = [2][]const mtl.HandoffOp{ exchanged[1], try stage(&tp, mode, bufs, rows, true, .pick) };
        try testing.expectEqual(@as(usize, 4), picks[1].len);
        const mine = picks[1][1];
        try testing.expectEqualStrings("tp_argmax_part", mine.pipeline);
        const theirs = try named(picks[0], "tp_argmax_part");
        try testing.expectEqual(theirs.grid, mine.grid);
        for ([_]usize{ 0, 2 }) |at| try testing.expectEqual(theirs.args[at], mine.args[at]);
        try testing.expectEqualStrings("tp_argmax_part", picks[1][2].pipeline);
        try testing.expectEqual(1 - k, (word(picks[1][2], 2, [4]u32)[1]) / 700);
        for (picks[1][1..3]) |op| {
            try testing.expectEqual(pass, op.args[3]);
            try testing.expectEqual(tp_local.PASS, word(op, 4, u64));
        }
        const pick = try named(picks[1], "tp_pick");
        const ref_pick = try named(picks[0], "tp_pick");
        for ([_]usize{ 2, 5 }) |at| try testing.expectEqual(ref_pick.args[at], pick.args[at]);
        try testing.expectEqual(pass, pick.args[0]);
        try testing.expectEqual(tp_local.PASS, word(pick, 1, u64));
        const ids = [2]mtl.HandoffOp{ try named(exchanged[2], "tp_ids"), try named(try stage(&tp, mode, bufs, rows, true, .ids), "tp_ids") };
        for ([_]usize{ 0, 1, 2 }) |at| try testing.expectEqual(ids[0].args[at], ids[1].args[at]);
        try testing.expectEqual(pass, ids[1].args[3]);
        try testing.expectEqual(tp_local.PASS, word(ids[1], 4, u64));
        for ([_][]const mtl.HandoffOp{ plain[1], picks[1] }) |ops| for (ops) |op| try testing.expect(op.kind == .compute or op.kind == .dispatch);
        try testing.expectEqual(seq, tp.seq); // the local launches queue nothing
    };
}
