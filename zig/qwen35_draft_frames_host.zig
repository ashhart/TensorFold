//! Fixed frame schedules, complete byte records and actual prompt-control routing without device execution.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const lanes = @import("lanes");
const fixture = @import("qwen35_verify_host.zig");
const plan = @import("tests/cuda/qwen_draft_frames_plan.zig");
const control = @import("tests/cuda/qwen_draft_frames_control.zig");
const t = std.testing;

fn fill(c: *plan.Capture, partition: plan.Partition) !void {
    const Fake = struct {
        capture: *plan.Capture,
        count: [plan.streams]usize = @splat(0),
        pub fn frame(f: *@This(), round: usize, group: []const usize) !void {
            var words: [plan.word_bytes]u8 = undefined;
            var selector: [plan.depth * 2]u8 = undefined;
            for (group) |id| {
                try t.expectEqual(round, f.count[id]);
                f.count[id] += 1;
                @memset(&words, @intCast(id));
                @memset(&selector, @intCast(round));
                const proposals: [plan.depth]u32 = @splat(@intCast(100 * id + round));
                try f.capture.write(round, id, &proposals, &words, &selector, (id + round) % plan.block);
            }
        }
    };
    var fake: Fake = .{ .capture = c };
    try plan.schedule(&fake, partition);
    try t.expectEqual(@as([plan.streams]usize, @splat(plan.rounds)), fake.count);
}

test "Draft frames: all partitions execute every stream once a round and canonical records ignore frame order" {
    var base = try plan.Capture.init(t.allocator, 1);
    defer base.deinit();
    try fill(&base, plan.partitions[0]);
    for (plan.partitions[1..]) |partition| {
        var candidate = try plan.Capture.init(t.allocator, 1);
        defer candidate.deinit();
        try fill(&candidate, partition);
        try t.expectEqual(@as(?plan.Difference, null), try plan.compare(base, candidate, false));
    }
    try t.expectEqualSlices(usize, &.{ 37, 64, 129, 200, 300, 511, 700, 1000 }, &plan.lengths);
    try t.expectError(error.BadFramePartition, plan.validate(.{ .mode = .mixed_a, .groups = &.{ &.{ 0, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7 } } }));
    try t.expectError(error.BadFramePartition, plan.validate(.{ .mode = .mixed_a, .groups = &.{&.{ 0, 1, 2, 3, 4, 5, 6, 8 }} }));
    try t.expectError(error.BadFramePartition, plan.validate(.{ .mode = .mixed_a, .groups = &.{&.{ 0, 1, 2 }} }));
}

test "Draft frames: proposals, staged words, selector bytes and acceptance are all compared, with no label oracle" {
    var base = try plan.Capture.init(t.allocator, 1);
    defer base.deinit();
    var candidate = try plan.Capture.init(t.allocator, 1);
    defer candidate.deinit();
    try fill(&base, plan.partitions[0]);
    try fill(&candidate, plan.partitions[1]);
    for ([_]usize{ 0, plan.depth * 4, plan.depth * 4 + plan.word_bytes, base.stride - 1 }) |offset| {
        const record = try candidate.slot(5, 7);
        record[offset] ^= 1;
        try t.expectEqual(plan.Difference{ .round = 5, .stream = 7, .byte = offset }, (try base.difference(candidate)).?);
        try t.expectError(error.DraftFrameDependent, plan.compare(base, candidate, false));
        try t.expect((try plan.compare(base, candidate, true)) != null);
        record[offset] ^= 1;
    }
    try t.expectError(error.FrameControlNotDetected, plan.compare(base, candidate, true));
    try candidate.swapStreams(0, 1);
    try t.expectEqual(@as(usize, 0), (try plan.compare(base, candidate, true)).?.stream);
    try candidate.swapStreams(0, 1);
    try t.expectEqual(@as(?plan.Difference, null), try base.difference(candidate));
    candidate.seen[47] = false;
    try t.expectError(error.MissingFrameRecord, plan.compare(base, candidate, true));
}

test "Draft frames: record writes reject duplicates, invalid sizes and out-of-range identities" {
    var c = try plan.Capture.init(t.allocator, 1);
    defer c.deinit();
    const proposals: [plan.depth]u32 = @splat(3);
    const words: [plan.word_bytes]u8 = @splat(0xa5);
    const selector: [plan.depth * 2]u8 = @splat(0xff);
    try t.expectError(error.BadFrameRecord, c.write(6, 0, &proposals, &words, &selector, 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 8, &proposals, &words, &selector, 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 0, proposals[1..], &words, &selector, 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 0, &proposals, words[1..], &selector, 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 0, &proposals, &words, selector[1..], 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 0, &proposals, &words, &selector, 16));
    try c.write(0, 0, &proposals, &words, &selector, 15);
    try t.expectError(error.BadFrameRecord, c.write(0, 0, &proposals, &words, &selector, 0));
}

test "Draft frames: greedy acceptance stops at the first mismatch and retains a pending or bonus row" {
    var proposals: [plan.depth]u32 = undefined;
    var sampled: [plan.block]u32 = undefined;
    for (&sampled, 0..) |*s, i| s.* = @intCast(i);
    @memcpy(&proposals, sampled[0..plan.depth]);
    try t.expectEqual(@as(usize, plan.depth), try plan.accepted(&proposals, &sampled));
    for (0..plan.depth) |at| {
        proposals[at] += 1;
        try t.expectEqual(at, try plan.accepted(&proposals, &sampled));
        proposals[at] -= 1;
    }
}

test "Draft frames: retained JSON records preserve raw bytes and canonical identity" {
    var c = try plan.Capture.init(t.allocator, 1);
    defer c.deinit();
    const proposals: [plan.depth]u32 = @splat(248319);
    const words: [plan.word_bytes]u8 = @splat(0xff);
    const selector: [plan.depth * 2]u8 = @splat('a');
    try c.write(3, 5, &proposals, &words, &selector, 7);
    var out: std.Io.Writer.Allocating = .init(t.allocator);
    defer out.deinit();
    try c.json(&out.writer, .mixed_b);
    const doc = try std.json.parseFromSlice(std.json.Value, t.allocator, out.written(), .{});
    defer doc.deinit();
    const obj = doc.value.object;
    try t.expectEqualStrings("mixed_b", obj.get("mode").?.string);
    try t.expectEqual(@as(i64, 3), obj.get("round").?.integer);
    try t.expectEqual(@as(i64, 5), obj.get("stream").?.integer);
    try t.expectEqual(@as(i64, 7), obj.get("accepted").?.integer);
    for (obj.get("proposals").?.array.items) |v| try t.expectEqual(@as(i64, 248319), v.integer);
    const w = obj.get("words").?.array.items;
    try t.expectEqual(@as(usize, plan.word_bytes), w.len);
    for (w) |v| try t.expectEqual(@as(i64, 255), v.integer);
    const s = obj.get("selector").?.array.items;
    try t.expectEqual(@as(usize, plan.depth * 2), s.len);
    for (s) |v| try t.expectEqual(@as(i64, 'a'), v.integer);
}

const Witness = struct {
    c: struct { block: usize = plan.block } = .{},
    projection_role: q.draft.precision.Role = .proposal,
    e: *q.engine.Engine,
    fail: bool = false,
    pub fn drafter(w: *Witness) struct {
        w: *Witness,
        pub fn hold(x: @This(), items: []const lanes.drafter.Hold) !void {
            for (items) |item| {
                const shape = item.lanes.?;
                try t.expectEqual(@as(usize, plan.depth), shape.parents.len);
                for (shape.parents, shape.ranks, shape.depths, 0..) |parent, rank, depth, i| {
                    try t.expectEqual(@as(i32, @intCast(i)) - 1, parent);
                    try t.expectEqual(@as(u8, 0), rank);
                    try t.expectEqual(@as(u8, @intCast(i)), depth);
                }
            }
            if (x.w.fail) return error.PlantedHoldFailure;
            const weight: cuda.qlinear.Affine4 = .{ .w = 10, .s = 20, .b = 30, .k = 8192, .n = 4096, .npad = 4096 };
            try q.draft.precision.affine(x.w.e, weight, 0x10000, 0x20000, 0x30000, items.len * plan.block, false, x.w.projection_role);
        }
    } {
        return .{ .w = w };
    }
};

test "Draft frames: the prompt control reaches real projection dispatch only for shared proposal frames" {
    var driver: cuda.Driver = undefined;
    var e = fixture.engine(&driver, true);
    var witness: Witness = .{ .e = &e };
    var items: [plan.streams]lanes.drafter.Hold = @splat(.{ .stream = undefined, .pending = 1, .position = 2, .depth = plan.depth });
    for ([_]bool{ false, true }) |negative| for ([_]usize{ 1, 2, 4, 8 }) |n| {
        fixture.Trace.count = 0;
        fixture.Trace.prompts = 0;
        var stats: control.Stats = .{};
        try control.hold(&witness, items[0..n], negative, &stats);
        const prompt = negative and n > 1;
        try t.expectEqual(@as(usize, @intFromBool(prompt)), fixture.Trace.prompts);
        try t.expectEqual(@as(usize, @intFromBool(prompt)), stats.prompt_frames);
        try t.expectEqual(@as(usize, 1), stats.frames);
        try t.expectEqual(n * plan.block, stats.max_rows);
        try t.expectEqual(q.draft.precision.Role.proposal, witness.projection_role);
    };
    witness.fail = true;
    var stats: control.Stats = .{};
    try t.expectError(error.PlantedHoldFailure, control.hold(&witness, &items, true, &stats));
    try t.expectEqual(q.draft.precision.Role.proposal, witness.projection_role);
    try t.expectEqual(@as(usize, 0), stats.frames);
}

test "Draft frames: a row-limited hold past 17 streams runs frames of one fused tile each, 16 and 17 stay single" {
    const F = q.draft.frame;
    const block = 16;
    const capacity = F.max_streams;
    try t.expect(capacity >= 18); // the admitted widths the failure reached
    try t.expectEqual(@as(usize, 17), F.headStreams(block));
    try t.expectEqual(@as(usize, 36), F.headStreams(8));
    try t.expect(F.rowLimited(.{ .ql = .{ .nvfp4 = cuda.nvfp4.weight(0x10000, 0x20000, 1, 64, 64) } }));
    try t.expect(!F.rowLimited(.{ .a4 = .{ .w = 1, .s = 2, .b = 3, .n = 64, .k = 64, .npad = 64 } }));
    for ([_]usize{ 16, 17 }) |n| try t.expectEqual(n, F.frameLen(n, 0, block, capacity, true)); // one frame, as today
    try t.expectEqual(@as(usize, 0), F.frameLen(16, 16, block, capacity, true)); // and the plan ends there
    try t.expectEqual(@as(usize, 17), F.frameLen(18, 0, block, capacity, true));
    try t.expectEqual(@as(usize, 1), F.frameLen(18, 17, block, capacity, true)); // 18 streams: one frame, then its tail
    // an affine-4 head keeps the whole hold in one frame: 128 streams ran as one before #4190
    try t.expectEqual(@as(usize, 18), F.frameLen(18, 0, block, capacity, false));
    try t.expectEqual(capacity, F.frameLen(capacity + 32, 0, block, capacity, false));
    for (1..capacity + 1) |total| {
        var at: usize = 0;
        var taken: usize = 0;
        while (at < total) : (taken += 1) {
            const n = F.frameLen(total, at, block, capacity, true); // the plan holdFn walks
            try t.expect(n > 0 and n <= F.headStreams(block) and n <= total - at);
            try t.expect(n * (block - 1) < cuda.qmmf.fused_rows); // every frame's rows take the fused tile
            at += n;
        }
        try t.expectEqual(total, at); // consecutive, up to the last stream
        try t.expectEqual((total + F.headStreams(block) - 1) / F.headStreams(block), taken);
    }
    // planted control: the unsplit 18-stream frame alone is at or over the fused tile, so it must split
    try t.expect(18 * (block - 1) >= cuda.qmmf.fused_rows);
}

/// The host probe behind the production hold: embeds, staging copies, waits and head rows, plus one frame failure.
const HoldProbe = struct {
    const Max = 36;
    var embed_rows: [8]u32 = undefined;
    var embeds: usize = 0;
    var uploads: usize = 0;
    var downloads: usize = 0;
    var waits: usize = 0;
    var head_rows: [8]i32 = undefined;
    var heads: usize = 0;
    var fail_embed: usize = 0; // 1-based embed launch to fail; 0 runs every frame

    fn reset() void {
        embeds = 0;
        uploads = 0;
        downloads = 0;
        waits = 0;
        heads = 0;
        fail_embed = 0;
    }

    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id == 1) {
            embeds += 1;
            if (embeds <= embed_rows.len) embed_rows[embeds - 1] = gx;
            if (fail_embed != 0 and fail_embed == embeds) return cuda.abi.error_not_ready;
        }
        if (id == 20) {
            if (heads < head_rows.len) head_rows[heads] = @as(*const i32, @ptrCast(@alignCast(args.?[6].?))).*;
            heads += 1;
        }
        return cuda.abi.success;
    }

    fn launchEx(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return launch(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, args, extra);
    }

    fn upload(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        uploads += 1;
        return cuda.abi.success;
    }

    fn copyDevice(_: u64, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn copy2d(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn download(dst: ?*anyopaque, _: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const data = @as([*]u8, @ptrCast(dst.?))[0..bytes];
        @memset(data, 0);
        const words = std.mem.bytesAsSlice(u32, data);
        for (0..words.len / 43) |row| {
            words[row * 43 + 2] = @intCast(17 + row % 7);
            words[row * 43 + 3] = @intCast(128 + row % 7);
            words[row * 43 + 22] = @bitCast(@as(f32, 1));
            words[row * 43 + 23] = @bitCast(@as(f32, 0));
        }
        downloads += 1;
        return cuda.abi.success;
    }

    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        waits += 1;
        return cuda.abi.success;
    }
};

fn handleFor(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

fn frameWeight() cuda.qlinear.Affine4 {
    return .{ .w = 1, .s = 2, .b = 3, .n = 256, .k = 256, .npad = 256 };
}

fn frameConfig() q.draft.Config {
    return .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 256, .heads = 2, .kv_heads = 1, .head_dim = 128, .block = 16, .mask = 255, .window = 2048, .group = 16, .rank = 2, .topk = 2, .eps = 1e-6, .theta = 10000000, .taps = @splat(0), .n_taps = 1 };
}

test "Draft frames: a ModelOpt hold runs fused-tile frames, and a failed frame clears every context" {
    const F = q.draft.frame;
    const block = 16;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = HoldProbe.launch;
    driver.api.cuLaunchKernelEx = HoldProbe.launchEx;
    driver.api.cuMemcpyHtoDAsync_v2 = HoldProbe.upload;
    driver.api.cuMemcpyDtoHAsync_v2 = HoldProbe.download;
    driver.api.cuMemcpyDtoDAsync_v2 = HoldProbe.copyDevice;
    driver.api.cuMemcpy2DAsync_v2 = HoldProbe.copy2d;
    driver.api.cuStreamSynchronize = HoldProbe.wait;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.d = &driver;
    e.g.hidden = 256;
    e.stream = .{ .d = &driver, .handle = null };
    inline for (@typeInfo(@TypeOf(e.fns)).@"struct".field_names) |name| @field(e.fns, name) = handleFor(&driver, 10);
    e.fns.embed = handleFor(&driver, 1);
    e.embed_bits = 4;
    e.embed_w = 1;
    e.embed_s = 2;
    e.embed_b = 3;
    e.affine = .{
        .group_fn = handleFor(&driver, 10),
        .mlp128_fn = handleFor(&driver, 10),
        .gemv_fn = handleFor(&driver, 7),
        .verify_fn = handleFor(&driver, 10),
        .verify64_fn = handleFor(&driver, 10),
        .verify128_fn = handleFor(&driver, 10),
        .split_fn = handleFor(&driver, 10),
        .seg_fn = handleFor(&driver, 10),
        .many_fn = handleFor(&driver, 10),
        .prefill_fn = handleFor(&driver, 11),
        .prefill22_fn = handleFor(&driver, 10),
        .prefill_many_fn = handleFor(&driver, 10),
        .words_fn = handleFor(&driver, 10),
        .scales_fn = handleFor(&driver, 10),
        .gemv_blocks = 0,
        .split = null,
        .pdl = false,
        .wide_verify = false,
    };
    e.fns.norm = handleFor(&driver, 13);
    // the row-limited head: the ModelOpt lane refuses a projection at the fused tile's rows
    var lane: cuda.qmmf.Lane = undefined;
    lane.mod = .{ .d = &driver, .handle = null };
    for (&lane.fns) |*set| set.* = @splat(handleFor(&driver, 20));
    lane.f32_fns = lane.fns;
    lane.clusters = false;
    lane.pdl = false;
    lane.prompt = null;
    lane.prompt_k32 = null;
    lane.prompt_sms = 0;
    lane.half_n = null;
    lane.fuse_split_n = 0;
    lane.half_below = cuda.qmmf.half_below_default;
    e.mo = .{ .lane = lane, .prompt = undefined, .pack = undefined, .embed = undefined };
    var fp4 = cuda.nvfp4.weight(0x10000, 0x20000, 1, 248320, 256);
    fp4.sk = 1; // one K slice: the frame's rows, not the slice partials, are the limit
    e.head = .{ .ql = .{ .nvfp4 = fp4 } };
    e.draws.logprob_rows_fn = handleFor(&driver, 9);
    e.draws.topk_slices_fn = handleFor(&driver, 9);
    e.draws.topk_merge_fn = handleFor(&driver, 12);
    var d: q.draft.Draft = undefined;
    d.projection_role = .proposal;
    d.frame_streams = 128;
    d.scratch_one = 0;
    d.target_taps = 0;
    d.gpa = t.allocator;
    d.io = std.testing.io;
    d.e = &e;
    d.c = frameConfig();
    d.tree_nodes = try q.draft.tree.nodes(); // as Draft.init reads it
    d.streams = .empty;
    d.trace = null;
    defer d.streams.deinit(t.allocator);
    d.selection_base = 0;
    inline for (@typeInfo(@TypeOf(d.fns)).@"struct".field_names) |name| @field(d.fns, name) = handleFor(&driver, 10);
    d.fns.dyn_conv_batch = handleFor(&driver, 2);
    d.fns.norm_rope_batch = handleFor(&driver, 3);
    d.fns.mask_rows = handleFor(&driver, 4);
    d.fns.attn_partial = handleFor(&driver, 5);
    d.fns.attn_merge = handleFor(&driver, 6);
    var layer: @TypeOf(d.w.layers[0]) = undefined;
    inline for (@typeInfo(@TypeOf(layer)).@"struct".field_names, @typeInfo(@TypeOf(layer)).@"struct".field_types) |name, T| {
        @field(layer, name) = if (T == cuda.qlinear.Affine4) frameWeight() else 1;
    }
    d.w.layers = @as(*[1]@TypeOf(layer), &layer);
    layer.attn_conv.n = 64;
    layer.mlp_conv.n = 64;
    layer.k.n = 128;
    layer.v.n = 128;
    d.w.selector = frameWeight();
    d.w.selector.n = 2;
    d.w.selector.npad = 64;
    d.w.final_norm = 1;
    var book: [512]u16 = @splat(0);
    d.w.predecessor = &book;
    d.w.successor = &book;
    var storage: [F.stagedLen(2)]u8 align(16) = undefined;
    d.staged = .{ .d = &driver, .bytes = &storage };
    const measured = try d.layout(false);
    d.scratch = .{ .d = &driver, .ptr = 0x100000, .len = measured };
    _ = try d.layout(true);
    var contexts: [HoldProbe.Max]q.draft.Ctx = undefined;
    var holds: [HoldProbe.Max]lanes.drafter.Hold = undefined;
    for (&contexts, &holds, 0..) |*ctx, *it, i| {
        ctx.* = .{ .ring = .{ .d = &driver, .ptr = 0x400000 + i * 0x1000, .len = 1 }, .begin = i, .end = 2048 + i };
        it.* = .{ .stream = @ptrFromInt(0x1000 + i * @alignOf(lanes.Stream)), .pending = @intCast(i), .position = 2049 + i, .depth = 1 };
        try d.streams.put(t.allocator, it.stream, ctx);
    }
    // the 256-row boundary: 17 streams are 255 rows, 18 are 270, and 16 and 17 stay one frame
    for ([_]usize{ 16, 17 }) |streams| {
        HoldProbe.reset();
        try d.drafter().hold(holds[0..streams]);
        try t.expectEqual(@as(usize, 1), HoldProbe.embeds);
        try t.expectEqual(@as(u32, @intCast(streams * block)), HoldProbe.embed_rows[0]);
        try t.expectEqual(@as(usize, 2), HoldProbe.uploads);
        try t.expectEqual(@as(usize, 1), HoldProbe.waits);
        try t.expectEqual(@as(usize, 1), HoldProbe.heads);
        try t.expectEqual(@as(i32, @intCast(streams * (block - 1))), HoldProbe.head_rows[0]);
        for (contexts[0..streams]) |ctx| {
            try t.expect(ctx.held_n > 0);
            if (d.tree_nodes > 0) try t.expectEqual(ctx.held_n, ctx.grown) else try t.expectEqual(@as(usize, 0), ctx.grown);
        }
    }
    HoldProbe.reset();
    try d.drafter().hold(holds[0..18]);
    try t.expectEqual(@as(usize, 2), HoldProbe.embeds);
    try t.expectEqualSlices(u32, &.{ 17 * block, block }, HoldProbe.embed_rows[0..2]);
    try t.expectEqual(@as(usize, 4), HoldProbe.uploads);
    try t.expectEqual(@as(usize, 2), HoldProbe.waits);
    try t.expectEqualSlices(i32, &.{ 17 * (block - 1), block - 1 }, HoldProbe.head_rows[0..2]);
    for (contexts[0..18]) |ctx| {
        try t.expect(ctx.held_n > 0);
        if (d.tree_nodes > 0) try t.expectEqual(ctx.held_n, ctx.grown) else try t.expectEqual(@as(usize, 0), ctx.grown);
    }
    // three frames (17, 17, 2); the second fails after the first published and before the third ran
    HoldProbe.reset();
    HoldProbe.fail_embed = 2;
    for (contexts[0..HoldProbe.Max]) |*ctx| {
        ctx.held_n = 3; // a previous hold's drafts must not survive a failed one
        ctx.grown = 2;
    }
    try t.expectError(error.NotReady, d.drafter().hold(holds[0..HoldProbe.Max]));
    try t.expectEqual(@as(usize, 2), HoldProbe.embeds);
    for (contexts[0..HoldProbe.Max]) |ctx| {
        try t.expectEqual(@as(usize, 0), ctx.held_n);
        try t.expectEqual(@as(usize, 0), ctx.grown);
    }
    // the same 18-stream hold on an affine-4 head stays one frame: the split is the ModelOpt head's alone
    e.head = .{ .a4 = .{ .w = 1, .s = 2, .b = 3, .n = 256, .k = 256, .npad = 256 } };
    HoldProbe.reset();
    try d.drafter().hold(holds[0..18]);
    try t.expectEqual(@as(usize, 1), HoldProbe.embeds);
    try t.expectEqual(@as(u32, 18 * block), HoldProbe.embed_rows[0]);
}
