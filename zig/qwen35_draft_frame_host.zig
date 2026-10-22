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
    var counts: [16]usize = @splat(0);
    var max_rows: u32 = 0;
    var uploads: usize = 0;
    var downloads: usize = 0;
    var waits: usize = 0;
    var fail: bool = false;
    var last_norm_input: u64 = 0;
    var block: usize = 8;
    fn launch(f: cuda.abi.Function, x: c_uint, _: c_uint, z: c_uint, threads: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id < counts.len) counts[id] += 1;
        if (id == 13) last_norm_input = @as(*const u64, @ptrCast(@alignCast(args.?[0].?))).*;
        if (id == 1) max_rows = @max(max_rows, x);
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
    fn download(dst: ?*anyopaque, _: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const data = @as([*]u8, @ptrCast(dst.?))[0..bytes];
        @memset(data, 0);
        if (downloads == 0) {
            const words = std.mem.bytesAsSlice(u32, data);
            for (0..words.len / 43) |row| {
                words[row * 43 + 2] = @intCast(17 + row / (block - 1));
                words[row * 43 + 3] = @intCast(128 + row / (block - 1));
                words[row * 43 + 22] = @bitCast(@as(f32, 1));
                words[row * 43 + 23] = @bitCast(@as(f32, 0));
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

test "Draft frames: the real hold adapter emits one frame for every stream and invalidates drafts on failure" {
    for ([_]usize{ 8, 16 }) |block| try actualFrame(block);
}
fn actualFrame(block: usize) !void {
    Record.block = block;
    const F = q.draft.frame;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Record.launch;
    driver.api.cuMemcpyHtoDAsync_v2 = Record.upload;
    driver.api.cuMemcpyDtoHAsync_v2 = Record.download;
    driver.api.cuStreamSynchronize = Record.wait;
    var e: q.engine.Engine = undefined;
    e.d = &driver;
    e.g.hidden = 256;
    e.stream = .{ .d = &driver, .handle = null };
    inline for (@typeInfo(@TypeOf(e.fns)).@"struct".field_names) |name| @field(e.fns, name) = function(&driver, 10);
    e.fns.embed = function(&driver, 1);
    e.embed_bits = 4;
    e.embed_w = 1;
    e.embed_s = 2;
    e.embed_b = 3;
    e.affine.prefill_fn = function(&driver, 11);
    e.affine.group_fn = function(&driver, 10);
    e.affine.pdl = false;
    e.affine.wide_verify = false;
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
    d.streams = .empty;
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
    try std.testing.expect(d.po + F.max_streams * d.c.block * d.c.heads * F.chunks * 128 * 4 <= d.scratch.ptr + measured);
    var contexts: [128]q.draft.Ctx = undefined;
    var holds: [128]lanes.drafter.Hold = undefined;
    const shape: lanes.shape.Shape = .{ .parents = @constCast(&[_]i32{ -1, -1, 0, 1, 2, 2, 3 }), .depths = @constCast(&[_]u8{ 0, 0, 1, 1, 2, 2, 2 }), .ranks = @constCast(&[_]u8{ 0, 1, 0, 1, 0, 1, 0 }), .expected = 0 };
    for (&contexts, &holds, 0..) |*ctx, *it, i| {
        ctx.* = .{ .ring = .{ .d = &driver, .ptr = 0x400000 + i * 0x1000, .len = 1 }, .begin = i, .end = 2048 + i };
        it.* = .{ .stream = @ptrFromInt(0x1000 + i * @alignOf(lanes.Stream)), .pending = @intCast(i), .position = 2049 + i, .depth = @intCast(1 + i % 7), .lanes = if (i % 2 == 0) &shape else null };
        try d.streams.put(a, it.stream, ctx);
    }
    for ([_]usize{ 1, 2, 4, 8, 16, 128 }) |count| {
        Record.counts = @splat(0);
        Record.max_rows = 0;
        Record.uploads = 0;
        Record.downloads = 0;
        Record.waits = 0;
        Record.fail = false;
        try d.drafter().hold(holds[0..count]);
        try std.testing.expectEqual(@as(usize, 0), Record.counts[11]);
        try std.testing.expectEqual(@as(u32, @intCast(count * block)), Record.max_rows);
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
            const limit = if (block == 16) q.draft.tree.nodes() else 0;
            const held: usize = if (it.lanes != null) 7 else if (limit > 0) ctx.grown else @min(it.depth, block - 1);
            try std.testing.expectEqual(held, ctx.held_n);
            if (limit > 0 and it.lanes == null and it.depth > 0) try std.testing.expect(ctx.grown > 0);
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
    try std.testing.expectError(error.NotReady, d.drafter().hold(holds[0..8]));
    try std.testing.expectEqual(@as(usize, 1), Record.waits);
    for (contexts[0..8]) |ctx| {
        try std.testing.expectEqual(@as(usize, 0), ctx.held_n);
        try std.testing.expectEqual(@as(usize, 0), ctx.grown);
    }
}

test {
    _ = @import("qwen35_draft_equality.zig");
}

test "Draft frames: combined target rows above capacity refuse before even settling an earlier round" {
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
    e.keeps = .{ .apply = function(&driver, 10), .conv = function(&driver, 10), .compact = function(&driver, 10) };
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
