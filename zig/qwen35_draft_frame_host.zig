//! The actual CUDA DFlash frame's staging and launches, with host-only boundaries.
const std = @import("std");
const q = @import("qwen35");
const cuda = @import("cuda");
const lanes = @import("lanes");
const a = std.testing.allocator;
pub fn config() q.draft.Config {
    return .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 256, .heads = 2, .kv_heads = 1, .head_dim = 128, .block = 8, .mask = 255, .window = 2048, .group = 16, .rank = 2, .topk = 2, .eps = 1e-6, .theta = 10000000, .taps = @splat(0), .n_taps = 1 };
}

test "Draft frames: a grown tree's node budget past the block (TF_DRAFT_NODES) stages; a chain keeps one row a depth" {
    const F = q.draft.frame;
    var c = config();
    c.block = 16;
    var storage: [F.stagedLen(2)]u8 align(8) = undefined;
    var ctx: q.draft.Ctx = .{ .ring = .{ .d = undefined, .ptr = 0x1000, .len = 1 }, .begin = 0, .end = 100 };
    var refs = [_]*q.draft.Ctx{&ctx};
    var items = [_]lanes.drafter.Hold{.{ .stream = @ptrFromInt(0x1000), .pending = 1, .position = 101, .depth = 15 }};
    try F.stage(c, &items, &refs, &storage);
    items[0].depth = 16;
    try std.testing.expectError(error.BadDraftFrame, F.stage(c, &items, &refs, &storage));
    _ = setenv("TF_DRAFT_NODES", "24", 1);
    defer _ = unsetenv("TF_DRAFT_NODES");
    items[0].depth = 24;
    try F.stage(c, &items, &refs, &storage);
    items[0].depth = 25;
    try std.testing.expectError(error.BadDraftFrame, F.stage(c, &items, &refs, &storage));
    items[0].depth = 24;
    try std.testing.expectError(error.BadDraftFrame, F.stage(config(), &items, &refs, &storage));
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "Draft frames: the actual drafter advertises batched holds" {
    var d: q.draft.Draft = undefined;
    var target: q.engine.Engine = undefined;
    target.g.hidden = 256; // the drafter's facts read the target's shape: this small one keeps its copy match
    target.g.layers = 1;
    target.g.experts = 0;
    d.e = &target;
    d.projection_role = .proposal;
    d.frame_streams = 128;
    d.scratch_one = 0;
    d.target_taps = 0;
    d.c = config();
    d.step_ms = 1;
    d.n_costs = 0;
    try std.testing.expect(d.drafter().facts().batched);
}

test "Draft frames: a common frame stages each stream's own ring positions and pending id" {
    try std.testing.expect(@hasDecl(q.draft, "frame"));
    if (comptime @hasDecl(q.draft, "frame")) {
        const F = q.draft.frame;
        var storage: [F.stagedLen(2)]u8 align(8) = undefined;
        var contexts: [128]q.draft.Ctx = undefined;
        var refs: [128]*q.draft.Ctx = undefined;
        var items: [128]lanes.drafter.Hold = undefined;
        for (&contexts, &refs, &items, 0..) |*ctx, *ref, *it, i| {
            ctx.* = .{ .ring = .{ .d = undefined, .ptr = 0x1000 + i * 0x1000, .len = 1 }, .begin = i, .end = 2048 + i };
            ref.* = ctx;
            it.* = .{ .stream = @ptrFromInt(0x1000 + i * @alignOf(lanes.Stream)), .pending = @intCast(i), .position = 2049 + i, .depth = 7 };
        }
        try F.stage(config(), &items, &refs, &storage);
        const ids = std.mem.bytesAsSlice(u32, storage[0 .. 128 * 8 * 4]);
        const segments = F.segments(&storage);
        for (items, contexts, segments[0..128], 0..) |it, ctx, sg, i| {
            try std.testing.expectEqual(it.pending, ids[i * 8]);
            try std.testing.expectEqualSlices(u32, &@as([7]u32, @splat(255)), ids[i * 8 + 1 ..][0..7]);
            try std.testing.expectEqual(ctx.ring.ptr, sg.ring);
            try std.testing.expectEqual(@as(i64, @intCast(ctx.begin)), sg.begin);
            try std.testing.expectEqual(@as(i64, @intCast(ctx.end)), sg.position);
        }
        items[127].position -= 1;
        try std.testing.expectError(error.DraftContextBehind, F.stage(config(), &items, &refs, &storage));
        items[127].position += 1;
        refs[127] = refs[0];
        items[127].position = items[0].position;
        try std.testing.expectError(error.OverlappingDraftContexts, F.stage(config(), &items, &refs, &storage));
        try std.testing.expectError(error.BadDraftFrame, F.stage(config(), &.{}, &.{}, &storage));
    }
}

const Record = struct {
    const HeadProjection = struct { columns: c_int, rows: c_int };
    var counts: [16]usize = @splat(0);
    var max_rows: u32 = 0;
    var uploads: usize = 0;
    pub var downloads: usize = 0;
    var waits: usize = 0;
    pub var fail: bool = false;
    var last_norm_input: u64 = 0;
    pub var token_bias: u32 = 0;
    var block: usize = 8;
    var head_projections: [8]HeadProjection = undefined;
    var head_projection_count: usize = 0;
    fn launch(f: cuda.abi.Function, x: c_uint, _: c_uint, z: c_uint, threads: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id < counts.len) counts[id] += 1;
        if (id == 13) last_norm_input = @as(*const u64, @ptrCast(@alignCast(args.?[0].?))).*;
        if (id == 1) max_rows = @max(max_rows, x);
        if (id == 14) {
            const parts = @as(*const cuda.qlinear.Parts, @ptrCast(@alignCast(args.?[2].?))).*;
            const columns = parts.p[0].n;
            if (columns > 256) {
                head_projections[head_projection_count] = .{
                    .columns = columns,
                    .rows = @as(*const c_int, @ptrCast(@alignCast(args.?[3].?))).*,
                };
                head_projection_count += 1;
            }
        }
        if (id == 5) {
            std.debug.assert(z == 9 and threads == 128);
            if (fail) return cuda.abi.error_not_ready;
        }
        return cuda.abi.success;
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
        if (downloads == 0) {
            const words = std.mem.bytesAsSlice(u32, data);
            for (0..words.len / 43) |row| {
                words[row * 43 + 2] = @intCast(17 + Record.token_bias + row / (block - 1));
                words[row * 43 + 3] = @intCast(128 + Record.token_bias + row / (block - 1));
                words[row * 43 + 22] = @bitCast(@as(f32, 1) + @as(f32, @floatFromInt(Record.token_bias)));
                words[row * 43 + 23] = @bitCast(@as(f32, @floatFromInt(Record.token_bias)));
            }
        }
        downloads += 1;
        return cuda.abi.success;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        waits += 1;
        return cuda.abi.success;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn weight() cuda.qlinear.Affine4 {
    return .{ .w = 1, .s = 2, .b = 3, .n = 256, .k = 256, .npad = 256 };
}

test "Draft frames: actual holds group mixed head widths and invalidate drafts on failure" {
    for ([_]usize{ 8, 16 }) |block| try actualFrame(block);
}
fn actualFrame(block: usize) !void {
    return frameCase(block, .{});
}
const ContextCase = struct { active: bool = false, enabled: bool = false, cached: u32 = 0, depth: u32 = 3, want: ?usize = null, cohort: ?usize = null };
pub fn contextHoldFrame(cached: u32, depth: u32, want: usize, enabled: bool) !void {
    return frameCase(16, .{ .active = true, .enabled = enabled, .cached = cached, .depth = depth, .want = want });
}
pub fn contextHoldSelectionBase(cohort: usize) !void {
    return frameCase(16, .{ .active = true, .enabled = true, .cached = 4096, .depth = 3, .want = 8, .cohort = cohort });
}
fn frameCase(block: usize, option: ContextCase) !void {
    const physical = option.want orelse block;
    Record.block = physical;
    const F = q.draft.frame;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Record.launch;
    driver.api.cuMemcpyHtoDAsync_v2 = Record.upload;
    driver.api.cuMemcpyDtoHAsync_v2 = Record.download;
    driver.api.cuMemcpyDtoDAsync_v2 = Record.copyDevice;
    driver.api.cuMemcpy2DAsync_v2 = Record.copy2d;
    driver.api.cuStreamSynchronize = Record.wait;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.d = &driver;
    e.g.hidden = 256;
    e.stream = .{ .d = &driver, .handle = null };
    inline for (@typeInfo(@TypeOf(e.fns)).@"struct".field_names) |name| @field(e.fns, name) = function(&driver, 10);
    e.fns.embed = function(&driver, 1);
    e.embed_bits = 4;
    e.embed_w = 1;
    e.embed_s = 2;
    e.embed_b = 3;
    e.affine = .{
        .group_fn = function(&driver, 10),
        .mlp128_fn = function(&driver, 10),
        .gemv_fn = function(&driver, 7),
        .verify_fn = function(&driver, 10),
        .verify64_fn = function(&driver, 10),
        .verify128_fn = function(&driver, 10),
        .split_fn = function(&driver, 10),
        .seg_fn = function(&driver, 10),
        .many_fn = function(&driver, 10),
        .prefill_fn = function(&driver, 11),
        .prefill22_fn = function(&driver, 10),
        .prefill_many_fn = function(&driver, 10),
        .words_fn = function(&driver, 10),
        .scales_fn = function(&driver, 10),
        .gemv_blocks = 0,
        .split = null,
        .pdl = false,
        .wide_verify = false,
    };
    e.fns.norm = function(&driver, 13);
    e.head = .{ .a4 = weight() };
    e.draws.logprob_rows_fn = function(&driver, 9);
    e.draws.topk_slices_fn = function(&driver, 9);
    e.draws.topk_merge_fn = function(&driver, 12);
    var d: q.draft.Draft = undefined;
    d.projection_role = .proposal;
    d.frame_streams = 128;
    d.scratch_one = 0;
    d.target_taps = 0;
    d.gpa = a;
    d.io = std.testing.io;
    d.e = &e;
    d.c = config();
    d.c.block = block;
    d.tree_nodes = try q.draft.tree.nodes(); // as Draft.init reads it
    d.streams = .empty;
    d.trace = null;
    d.graphs = .{ .on = option.active };
    defer d.streams.deinit(a);
    d.selection_base = 0;
    inline for (@typeInfo(@TypeOf(d.fns)).@"struct".field_names) |name| @field(d.fns, name) = function(&driver, 10);
    d.fns.dyn_conv_batch = function(&driver, 2);
    d.fns.norm_rope_batch = function(&driver, 3);
    d.fns.mask_rows = function(&driver, 4);
    d.fns.attn_partial = function(&driver, 5);
    d.fns.attn_merge = function(&driver, 6);
    var layer: @TypeOf(d.w.layers[0]) = undefined;
    inline for (@typeInfo(@TypeOf(layer)).@"struct".field_names, @typeInfo(@TypeOf(layer)).@"struct".field_types) |name, T| {
        @field(layer, name) = if (T == cuda.qlinear.Affine4) weight() else 1;
    }
    d.w.layers = @as(*[1]@TypeOf(layer), &layer);
    layer.attn_conv.n = 64;
    layer.mlp_conv.n = 64;
    layer.k.n = 128;
    layer.v.n = 128;
    d.w.selector = weight();
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
    try std.testing.expectEqual(measured, try d.layout(true));
    try std.testing.expect(d.hp + F.max_streams * d.c.block * d.c.rank * 2 <= d.scratch.ptr + measured);
    try std.testing.expect(d.po + F.max_streams * d.c.block * d.c.heads * d.c.chunks() * 128 * 4 <= d.scratch.ptr + measured);
    var contexts: [128]q.draft.Ctx = undefined;
    var real_streams: [128]lanes.Stream = undefined;
    var holds: [128]lanes.drafter.Hold = undefined;
    const shape: lanes.shape.Shape = .{ .parents = @constCast(&[_]i32{ -1, -1, 0, 1, 2, 2, 3 }), .depths = @constCast(&[_]u8{ 0, 0, 1, 1, 2, 2, 2 }), .ranks = @constCast(&[_]u8{ 0, 1, 0, 1, 0, 1, 0 }), .expected = 0 };
    for (&contexts, &holds, 0..) |*ctx, *it, i| {
        ctx.* = .{ .ring = .{ .d = &driver, .ptr = 0x400000 + i * 0x1000, .len = 1 }, .begin = i, .end = 2048 + i };
        it.* = .{ .stream = @ptrFromInt(0x1000 + i * @alignOf(lanes.Stream)), .pending = @intCast(i), .position = 2049 + i, .depth = @intCast(1 + i % 7), .lanes = if (i % 2 == 0) &shape else null };
        if (option.active) {
            real_streams[i].cached = 0; // a cold prompt has no restored cache hits, even at a long context
            ctx.end = option.cached;
            it.position = @as(u64, option.cached) + 1;
            it.stream = &real_streams[i];
            it.depth = option.depth;
            it.lanes = null;
        }
        try d.streams.put(a, it.stream, ctx);
    }
    const adapter = q.draft.context_hold.adapt(&d, d.drafter(), option.enabled);
    const single = [_]usize{option.cohort orelse 0};
    const counts: []const usize = if (option.cohort != null) &single else if (option.active) &.{ 1, 4, 8 } else &.{ 1, 2, 4, 8, 16, 128 };
    for (counts) |count| {
        Record.counts = @splat(0);
        Record.max_rows = 0;
        Record.uploads = 0;
        Record.downloads = 0;
        Record.waits = 0;
        Record.fail = false;
        try adapter.hold(holds[0..count]);
        try std.testing.expectEqual(@as(usize, 0), d.selection_base);
        try std.testing.expectEqual(@intFromPtr(storage[F.words_at..].ptr), @intFromPtr(d.candidates(1).words.ptr));
        try std.testing.expectEqual(block, d.c.block);
        try std.testing.expectEqual(option.active, d.graphs.on);
        try std.testing.expectEqual(@as(usize, 0), Record.counts[11]);
        try std.testing.expectEqual(@as(u32, @intCast(count * physical)), Record.max_rows);
        for ([_]usize{ 1, 5, 6, 9, 12 }) |id| try std.testing.expectEqual(@as(usize, 1), Record.counts[id]);
        try std.testing.expectEqual(@as(usize, if (count == 1) 0 else 1), Record.counts[4]);
        try std.testing.expectEqual(if (count == 1) d.h + d.c.hidden * 2 else d.h2, Record.last_norm_input);
        try std.testing.expectEqual(@as(usize, 4), Record.counts[2]);
        try std.testing.expectEqual(@as(usize, 2), Record.counts[3]);
        try std.testing.expectEqual(@as(usize, 2), Record.uploads);
        try std.testing.expectEqual(@as(usize, 2), Record.downloads);
        try std.testing.expectEqual(@as(usize, 1), Record.waits);
        for (contexts[0..count], holds[0..count], 0..) |ctx, it, i| {
            // the production adapter: a grown tree holds the budget it grew, a chain the requested depth
            const limit = if (block == 16) try q.draft.tree.nodes() else 0;
            const held: usize = if (it.lanes != null) 7 else if (limit > 0) ctx.grown else @min(it.depth, block - 1);
            try std.testing.expectEqual(held, ctx.held_n);
            if (limit > 0 and it.lanes == null and it.depth > 0) try std.testing.expect(ctx.grown > 0);
            if (option.active) try std.testing.expectEqual(@as(usize, 15), ctx.grown);
            for (ctx.held[0..ctx.held_n], 0..) |token, row| {
                if (it.lanes == null and ctx.grown > 0) {
                    try std.testing.expect(token == 17 + i or token == 128 + i);
                    try std.testing.expect(ctx.parents[row] >= -1 and ctx.parents[row] < @as(i32, @intCast(row)));
                } else try std.testing.expectEqual(@as(u32, @intCast(if (it.lanes != null and shape.ranks[row] == 1) 128 + i else 17 + i)), token);
            }
        }
    }
    Record.fail = true;
    for (contexts[0..8]) |*ctx| ctx.grown = 2;
    Record.waits = 0;
    try std.testing.expectError(error.NotReady, adapter.hold(holds[0..8]));
    try std.testing.expectEqual(@as(usize, 0), d.selection_base);
    try std.testing.expectEqual(block, d.c.block);
    try std.testing.expectEqual(option.active, d.graphs.on);
    try std.testing.expectEqual(@as(usize, 1), Record.waits);
    for (contexts[0..8]) |ctx| {
        try std.testing.expectEqual(@as(usize, 0), ctx.held_n);
        try std.testing.expectEqual(@as(usize, 0), ctx.grown);
    }

    // The pre-fix hold path sends both contexts through the full head.
    e.head = .{ .a4 = .{ .w = 1, .s = 2, .b = 3, .n = 248320, .k = 256, .npad = 248320 } };
    d.c.vocab = 248320;
    d.scratch.len = try d.layout(false);
    _ = try d.layout(true);
    contexts[0].wide = false;
    contexts[1].wide = true;
    e.affine.group_fn = function(&driver, 14);
    Record.downloads = 0;
    Record.head_projection_count = 0;
    Record.fail = false;
    try d.drafter().hold(holds[0..2]);

    const per = block - 1;
    var narrow_rows: usize = 0;
    var tail_rows: usize = 0;
    var full_rows: usize = 0;
    for (Record.head_projections[0..Record.head_projection_count]) |projection| {
        switch (projection.columns) {
            98304 => narrow_rows += @intCast(projection.rows),
            320 => tail_rows += @intCast(projection.rows),
            248320 => full_rows += @intCast(projection.rows),
            else => unreachable,
        }
    }
    try std.testing.expectEqual(per, full_rows);
    try std.testing.expectEqual(per, narrow_rows);
    try std.testing.expectEqual(per, tail_rows);
    if (block == 16) try traceRounds(&d, contexts[0..2], holds[0..2]);
}

fn traceRounds(d: *q.draft.Draft, contexts: []q.draft.Ctx, source: []lanes.drafter.Hold) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var first = try lanes.Stream.init(a, .{ .id = "first", .prompt = &.{1}, .max_new = 8 });
    defer first.deinit(a);
    var second = try lanes.Stream.init(a, .{ .id = "second", .prompt = &.{1}, .max_new = 8 });
    defer second.deinit(a);
    try d.streams.put(a, &first, &contexts[0]);
    try d.streams.put(a, &second, &contexts[1]);
    defer {
        if (comptime @hasField(q.draft.Ctx, "trace_snapshot")) for (contexts) |*ctx| ctx.trace_snapshot.deinit(a);
    }
    var holds: [2]lanes.drafter.Hold = undefined;
    @memcpy(&holds, source);
    holds[0].stream = &first;
    holds[1].stream = &second;
    holds[0].lanes = null;
    holds[0].depth = 1;
    holds[1].depth = 1;
    holds[0].sampling = .{ .seed = 0x123456789abcdef, .temperature = 0.75 };
    holds[1].sampling = .{ .seed = 0x76543210, .temperature = 1.25 };
    try @import("qwen35_draft_trace_host.zig").run(Record, d, &holds, tmp);
    try traceSaveFailure(d, &holds, tmp);
}

fn traceSaveFailure(d: *q.draft.Draft, holds: *const [2]lanes.drafter.Hold, tmp: std.testing.TmpDir) !void {
    const t = try @import("qwen35_draft_trace_host.zig").enableTrace(d, tmp, "failed.jsonl");
    defer q.draft.trace.drop(&d.trace);
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    d.gpa = failing.allocator();
    defer d.gpa = a;
    Record.fail = false;
    Record.downloads = 0;
    var failed_hold = holds[0];
    failed_hold.position = d.streams.get(failed_hold.stream).?.end + 1;
    try d.drafter().hold(&.{failed_hold});
    if (comptime @hasField(q.draft.trace.Trace, "dropped")) {
        try std.testing.expectEqual(@as(u64, 1), t.dropped);
        try std.testing.expect(t.broken);
    }
    try std.testing.expect(!d.streams.get(holds[0].stream).?.trace_active);
    const accepted = [_]u32{41};
    const item: lanes.drafter.Absorb = .{
        .stream = failed_hold.stream,
        .features = .{ .buffer = 1, .rows = 0, .row_bytes = 0, .space = .device, .dtype = .bf16 },
        .start = failed_hold.position - 1,
        .follow = &accepted,
        .offered = 1,
        .prompt = false,
    };
    try d.drafter().absorb(&.{item});
    if (comptime @hasField(q.draft.trace.Trace, "dropped")) try std.testing.expectEqual(@as(u64, 1), t.dropped);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/failed.jsonl", .{tmp.sub_path});
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1));
    defer a.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), bytes.len);
}

test {
    _ = @import("qwen35_draft_equality.zig");
    _ = @import("qwen35_draft_frame_mixed_host.zig");
    _ = @import("qwen35_draft_graphs_host.zig");
}

test "Draft frames: combined target rows above capacity refuse before even settling an earlier round" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Record.launch;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.verify_rows = 64;
    e.learning = null;
    e.n_taps = 0;
    e.kv8 = false;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 256, .k_heads = 1, .v_heads = 1, .q_heads = 2, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 256, .linear = @splat(true) };
    e.geo = .{ .hid = 256, .nk = 1, .nv = 1, .qh = 1, .kvh = 1, .nl = 1 };
    e.keeps = .{ .apply = function(&driver, 10), .apply_pend = function(&driver, 10), .conv = function(&driver, 10), .compact = function(&driver, 10) };
    const b: cuda.DeviceBuffer = .{ .d = &driver, .ptr = 1, .len = 1 << 20 };
    var caches: [2]q.engine.Cache = @splat(.{ .state = b, .kv = b, .log = b, .cap = 256, .len = 2, .logits = &.{} });
    caches[0].last = .{ .start = 0, .rows = 2 };
    const ids: [65]u32 = @splat(1);
    const parts = [_]q.engine.Part{ .{ .cache = &caches[0], .rows = 64 }, .{ .cache = &caches[1], .rows = 1 } };
    Record.counts = @splat(0);
    try std.testing.expectError(error.InvalidQwenSegments, e.forward(&parts, &ids, .logits));
    try std.testing.expectEqual(@as(usize, 0), Record.counts[10]);
    try std.testing.expect(caches[0].last != null);
    try e.admit(&.{ .{ .cache = &caches[0], .rows = 63 }, .{ .cache = &caches[1], .rows = 1 } }, ids[0..64], .logits);
    try e.admit(&.{.{ .cache = &caches[0], .rows = 64 }}, ids[0..64], .logits);
    try std.testing.expectError(error.InvalidQwenSegments, e.admit(&.{ .{ .cache = &caches[0], .rows = 1 }, .{ .cache = &caches[0], .rows = 1 } }, ids[0..2], .logits));
}

test "Draft frames: an unsupported prompt geometry is refused at admission" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Record.launch;
    var e: q.engine.Engine = undefined;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.verify_rows = 64;
    e.learning = null;
    e.n_taps = 0;
    e.kv8 = false;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 256, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 256, .linear = @splat(true) };
    e.geo = .{ .hid = 256, .nk = 1, .nv = 1, .qh = 1, .kvh = 1, .nl = 1 };
    e.keeps = .{ .apply = function(&driver, 10), .apply_pend = function(&driver, 10), .conv = function(&driver, 10), .compact = function(&driver, 10) };
    const b: cuda.DeviceBuffer = .{ .d = &driver, .ptr = 1, .len = 1 << 20 };
    var caches: [1]q.engine.Cache = @splat(.{ .state = b, .kv = b, .log = b, .cap = 256, .len = 2, .logits = &.{} });
    const ids: [129]u32 = @splat(1);
    const parts = [_]q.engine.Part{.{ .cache = &caches[0], .rows = 129 }};
    // an odd query/KV group has no prompt tiles: the chunk's rows must fit the decode partials
    try std.testing.expectError(error.UnsupportedQwenGeometry, e.admit(&parts, &ids, .last));
    // an even group takes the prompt tiles: the same rows are admitted
    e.g.q_heads = 2;
    try e.admit(&parts, &ids, .last);
    e.g.q_heads = 1;
    // the partials' rows: a verify window past them is refused
    e.verify_rows = 256;
    try std.testing.expectError(error.UnsupportedQwenGeometry, e.admit(&parts, &ids, .logits));
    // the partials' groups: a group outside 2 to 16 is refused, whatever the rows
    const few: [64]u32 = @splat(1);
    const few_parts = [_]q.engine.Part{.{ .cache = &caches[0], .rows = 64 }};
    e.g.q_heads = 1;
    try std.testing.expectError(error.UnsupportedQwenGeometry, e.admit(&few_parts, &few, .logits));
    e.g.q_heads = 32;
    try std.testing.expectError(error.UnsupportedQwenGeometry, e.admit(&few_parts, &few, .logits));
    e.g.q_heads = 2;
    try e.admit(&few_parts, &few, .logits);
}

test "Draft frames: the geometry refusal precedes the request's own state" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Record.launch;
    var e: q.engine.Engine = undefined;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.verify_rows = 64;
    e.learning = null;
    e.n_taps = 0;
    e.kv8 = false;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 256, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 256, .linear = @splat(true) };
    e.geo = .{ .hid = 256, .nk = 1, .nv = 1, .qh = 1, .kvh = 1, .nl = 1 };
    e.keeps = .{ .apply = function(&driver, 10), .apply_pend = function(&driver, 10), .conv = function(&driver, 10), .compact = function(&driver, 10) };
    const b: cuda.DeviceBuffer = .{ .d = &driver, .ptr = 1, .len = 1 << 20 };
    var caches: [1]q.engine.Cache = @splat(.{ .state = b, .kv = b, .log = b, .cap = 256, .len = 2, .logits = &.{} });
    caches[0].last = .{ .start = 0, .rows = 2 };
    const ids: [129]u32 = @splat(1);
    const parts = [_]q.engine.Part{.{ .cache = &caches[0], .rows = 129 }};
    // the cache is full: the geometry is refused before the request's own state is read
    caches[0].cap = 1;
    Record.counts = @splat(0);
    try std.testing.expectError(error.UnsupportedQwenGeometry, e.admit(&parts, &ids, .last));
    // nothing was queued and the earlier round was not settled
    try std.testing.expectEqual(@as(usize, 0), Record.counts[10]);
    try std.testing.expect(caches[0].last != null);
}

test "Draft frames: enlarged frame byte counts reject overflow" {
    try std.testing.expectError(error.Overflow, q.draft.frame.stagedBytes(std.math.maxInt(usize)));
    var d: q.draft.Draft = undefined;
    d.projection_role = .proposal;
    d.frame_streams = 128;
    d.scratch_one = 0;
    d.target_taps = 0;
    d.c = config();
    d.c.rank = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, d.layout(false));
}

test "Draft frames: an EXL3 head's span views plan their K split as the whole head does" {
    const x3 = cuda.exl3;
    const F = q.draft.frame;
    for ([_]bool{ false, true }) |strips| {
        const st = x3.strides(5120, 248320, 12, strips);
        const w: x3.Weight = .{ .trellis = 0x1000, .suh = 0x2000, .svh = 0x3000, .k = 5120, .n = 248320, .halves = 12, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
        const v = try F.x3Views(w);
        const whole = x3.plan(w.k, w.n, 192, 8);
        try std.testing.expectEqual(whole, x3.plan(v.first.k, v.first.n, 192, 8));
        try std.testing.expectEqual(whole, x3.plan(v.tail.k, v.tail.n, 192, 8));
        try std.testing.expect(F.x3_tail_at <= 248032 and F.x3_tail_at + v.tail.n == w.n);
    }
}

test "Draft frames: a ModelOpt head's span views read the whole head's tiles on its K slices" {
    const Q = cuda.qmmf;
    const S = q.draft.frame.ql_span;
    const n = 248320;
    const k = 5120;
    const kg = k / 64;
    const fp4 = cuda.nvfp4.weight(0x10000, 0x8000000000, 2.5, n, k);
    const fp8 = cuda.modelopt.fp8Weight(0x10000, 0.5, n, k);
    for ([_]cuda.qlinear.Weight{ .{ .nvfp4 = fp4 }, .{ .fp8 = fp8 } }) |w| {
        const whole = switch (w) {
            .nvfp4, .fp8 => |x| x,
            else => unreachable,
        };
        const v = S.views(w, 98304, 248000, n).?;
        const first = switch (v.first) {
            .nvfp4, .fp8 => |x| x,
            else => unreachable,
        };
        const tail = switch (v.tail) {
            .nvfp4, .fp8 => |x| x,
            else => unreachable,
        };
        const bytes: usize = if (whole.mode == .fp4) 64 * 64 / 2 else 64 * 64;
        for (0..tail.n / 64) |t| for (0..kg) |g| {
            try std.testing.expectEqual(whole.codes + ((248000 / 64 + t) * kg + g) * bytes, tail.codes + (t * kg + g) * bytes);
            if (whole.mode == .fp4) try std.testing.expectEqual(whole.scales + ((248000 / 64 + t) * kg + g) * 256, tail.scales + (t * kg + g) * 256);
        };
        try std.testing.expectEqual(whole.codes, first.codes);
        try std.testing.expectEqual(whole.scale, tail.scale);
        for (1..Q.fused_rows) |m| for ([_]bool{ false, true }) |cl| {
            try std.testing.expectEqual(Q.planFor(m, whole, cl), Q.planFor(m, first, cl));
            try std.testing.expectEqual(Q.planFor(m, whole, cl), Q.planFor(m, tail, cl));
        };
    }
    try std.testing.expectEqual(@as(u32, 8), Q.splitK(320, k)); // unpinned, the tail would sum eight slices
    var scaled = fp8;
    scaled.act = 1.5;
    try std.testing.expect(S.views(.{ .fp8 = scaled }, 98304, 248000, n) == null);
    try std.testing.expect(S.views(.{ .bf16 = cuda.modelopt.bf16Weight(0x10000, n, k) }, 98304, 248000, n) == null);
    try std.testing.expect(S.views(.{ .nvfp4 = fp4 }, 98304, 248032, n) == null);
    try std.testing.expect(S.views(.{ .nvfp4 = cuda.nvfp4.weight(0x10000, 0x20000, 1, n - 64, k) }, 98304, 248000, n) == null);
    try std.testing.expect(!S.enabled());
}
