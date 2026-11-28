//! A recording driver compares DFlash2 graph replays with the eager launches, byte for byte, without a device.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const T = std.testing;
const G = q.draft.graphs;
const abi = cuda.abi;

const taps_id = 20;
const rope_id = 21;
const ring_id = 22;
const rows_id = 23;
const tap_rows_bytes = @sizeOf(q.draft.TapRows);
const wide_id = 24;

const Call = struct {
    f: usize,
    geometry: [6]c_uint,
    data: [512]u8 = @splat(0),

    fn of(f: abi.Function, geometry: [6]c_uint, args: ?[*]?*anyopaque) Call {
        var c: Call = .{ .f = @intFromPtr(f), .geometry = geometry };
        const sizes: []const usize = switch (c.f) {
            taps_id, wide_id => &.{ 8, 4, 8, 4, 8 },
            rows_id => &.{ 8, 4, 8, tap_rows_bytes, 4, 8 },
            rope_id => &.{ 8, 8, 8, 4, 4, 8, 4, 4 },
            ring_id => &.{ 8, 8, 8, 8, 4, 8, 4 },
            else => &.{},
        };
        var at: usize = 0;
        for (sizes, 0..) |n, i| {
            @memcpy(c.data[at..][0..n], @as([*]const u8, @ptrCast(args.?[i].?))[0..n]);
            at += n;
        }
        return c;
    }
};
const Copy = struct { kind: enum { device, host, packing }, dst: u64, src: u64, bytes: usize, packing: ?abi.Memcpy2D = null };

const Recording = struct {
    nodes: [256]Call = undefined,
    node_n: usize = 0,
    copies: [64]Copy = undefined,
    copy_n: usize = 0,
    alive: bool = false,
};

const Fake = struct {
    var eager: [256]Call = undefined;
    var eager_n: usize = 0;
    var ran: [256]Call = undefined;
    var ran_n: usize = 0;
    var recordings: [64]Recording = undefined;
    var recording_n: usize = 0;
    var current: usize = 0;
    var capturing = false;
    var captures: usize = 0;
    var launches: usize = 0;
    var exec_drops: usize = 0;
    var last_dropped: usize = 0;
    var refuse = false;

    fn rec(handle: usize) *Recording {
        return &recordings[handle - 1];
    }
    fn captured() *Recording {
        return &recordings[current];
    }
    fn reset() void {
        eager_n = 0;
        ran_n = 0;
        capturing = false;
        captures = 0;
        launches = 0;
        exec_drops = 0;
        last_dropped = 0;
        refuse = false;
    }
    fn launch(f: abi.Function, x: c_uint, y: c_uint, z: c_uint, bx: c_uint, by: c_uint, bz: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        const c = Call.of(f, .{ x, y, z, bx, by, bz }, args);
        if (capturing) {
            const r = captured();
            r.nodes[r.node_n] = c;
            r.node_n += 1;
        } else {
            eager[eager_n] = c;
            eager_n += 1;
        }
        return abi.success;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        if (recording_n == recordings.len) return 799;
        current = recording_n;
        recording_n += 1;
        recordings[current] = .{};
        capturing = true;
        captures += 1;
        return abi.success;
    }
    fn end(_: abi.Stream, g: *abi.Graph) callconv(.c) abi.Result {
        capturing = false;
        g.* = @ptrFromInt(current + 1);
        return abi.success;
    }
    fn getNodes(g: abi.Graph, out: ?[*]abi.GraphNode, n: *usize) callconv(.c) abi.Result {
        const handle = @intFromPtr(g);
        const r = rec(handle);
        // Node handles carry their recording above the low 16-bit index.
        if (out) |o| for (0..r.node_n) |i| {
            o[i] = @ptrFromInt((handle << 16) | (i + 1));
        };
        n.* = r.node_n;
        return abi.success;
    }
    fn getParams(node: abi.GraphNode, p: *abi.KernelNodeParams) callconv(.c) abi.Result {
        const raw = @intFromPtr(node);
        p.func = @ptrFromInt(rec(raw >> 16).nodes[(raw & 0xffff) - 1].f);
        return abi.success;
    }
    fn instantiate(out: *abi.GraphExec, g: abi.Graph, _: u64) callconv(.c) abi.Result {
        if (refuse) return 800;
        const handle = @intFromPtr(g);
        rec(handle).alive = true;
        out.* = @ptrFromInt(handle);
        return abi.success;
    }
    fn setParams(exec: abi.GraphExec, node: abi.GraphNode, p: *const abi.KernelNodeParams) callconv(.c) abi.Result {
        const handle = @intFromPtr(exec);
        const raw = @intFromPtr(node);
        if (raw >> 16 != handle) return 801;
        rec(handle).nodes[(raw & 0xffff) - 1] = Call.of(p.func, .{ p.grid_x, p.grid_y, p.grid_z, p.block_x, p.block_y, p.block_z }, p.params);
        return abi.success;
    }
    fn graphLaunch(exec: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        const r = rec(@intFromPtr(exec));
        if (!r.alive) return 802;
        @memcpy(ran[0..r.node_n], r.nodes[0..r.node_n]);
        ran_n = r.node_n;
        for (r.copies[0..r.copy_n]) |c| if (c.kind == .host) applyHost(c);
        launches += 1;
        return abi.success;
    }
    fn applyHost(c: Copy) void {
        const dst: [*]u8 = @ptrFromInt(c.dst);
        const src: [*]const u8 = @ptrFromInt(c.src);
        @memcpy(dst[0..c.bytes], src[0..c.bytes]);
    }
    fn record(c: Copy) void {
        if (capturing) {
            const r = captured();
            r.copies[r.copy_n] = c;
            r.copy_n += 1;
        } else if (c.kind == .host) applyHost(c);
    }
    fn copyDevice(dst: u64, src: u64, bytes: usize, _: abi.Stream) callconv(.c) abi.Result {
        record(.{ .kind = .device, .dst = dst, .src = src, .bytes = bytes });
        return abi.success;
    }
    fn copyHost(dst: ?*anyopaque, src: u64, bytes: usize, _: abi.Stream) callconv(.c) abi.Result {
        record(.{ .kind = .host, .dst = @intFromPtr(dst.?), .src = src, .bytes = bytes });
        return abi.success;
    }
    fn copy2d(m: *const abi.Memcpy2D, _: abi.Stream) callconv(.c) abi.Result {
        record(.{ .kind = .packing, .dst = m.dst, .src = m.src, .bytes = m.width * m.height, .packing = m.* });
        return abi.success;
    }
    fn dropExec(exec: abi.GraphExec) callconv(.c) abi.Result {
        const handle = @intFromPtr(exec);
        rec(handle).alive = false;
        last_dropped = handle;
        exec_drops += 1;
        return abi.success;
    }
    fn dropGraph(_: abi.Graph) callconv(.c) abi.Result {
        return abi.success;
    }
};

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn weight() cuda.qlinear.Affine4 {
    return .{ .w = 1, .s = 2, .b = 3, .n = 256, .k = 256, .npad = 256 };
}
fn holdBody(d: anytype, streams: usize, contexts: []const *q.draft.Ctx) !?u64 {
    _ = contexts;
    try d.e.run(function(d.e.d, taps_id), .{ streams, 1, 1 }, .{ 256, 1, 1 }, .{ @as(u64, 7), @as(c_int, 1), @as(c_longlong, 2), @as(c_int, 3), @as(u64, 4) });
    return 0x5000;
}
fn mixedHoldBody(d: anytype, streams: usize, contexts: []const *q.draft.Ctx) !?u64 {
    if (streams != contexts.len) return error.BadDraftGraph;
    for (contexts, 0..) |ctx, i| try d.e.run(function(d.e.d, if (ctx.wide) wide_id else taps_id), .{ streams, i + 1, 1 }, .{ 256, 1, 1 }, .{ @as(u64, 7), @as(c_int, 1), @as(c_longlong, 2), @as(c_int, 3), @as(u64, 4) });
    return null;
}
var mixed_staging: [q.draft.frame.stagedLen(2)]u8 align(8) = undefined;
var mixed_words: [4 * 7 * 43 * 4]u8 = undefined;

fn mixedHeadBody(d: anytype, streams: usize, contexts: []const *q.draft.Ctx) !?u64 {
    if (streams != contexts.len) return error.BadDraftGraph;
    try q.draft.frame.mixedHead(d, contexts, &mixed_staging);
    return null;
}

fn expectMixedWords(contexts: []const *q.draft.Ctx) !void {
    const word_bytes = 7 * 43 * 4;
    var at = [2]usize{ 0, 0 };
    for (contexts, 0..) |ctx, i| {
        const class: usize = @intFromBool(ctx.wide);
        const source = mixed_words[at[class] * word_bytes ..][0..word_bytes];
        const staged = mixed_staging[q.draft.frame.words_at + i * word_bytes ..][0..word_bytes];
        try T.expectEqualSlices(u8, source, staged);
        at[class] += 1;
    }
}

fn setWidthMask(contexts: []q.draft.Ctx, mask: u128) void {
    for (contexts, 0..) |*ctx, i| ctx.wide = mask & (@as(u128, 1) << @intCast(i)) != 0;
}

fn expectWidthMask(contexts: []const q.draft.Ctx) !void {
    try T.expectEqual(contexts.len, Fake.ran_n);
    for (contexts, 0..) |ctx, i| {
        try T.expectEqual(@as(usize, if (ctx.wide) wide_id else taps_id), Fake.ran[i].f);
        try T.expectEqual(@as(c_uint, @intCast(i + 1)), Fake.ran[i].geometry[1]);
    }
}

test "Draft frames: a DFlash2 graph replays the eager launches with this round's arguments" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
    driver.api.cuMemcpyDtoDAsync_v2 = Fake.copyDevice;
    driver.api.cuMemcpyDtoHAsync_v2 = Fake.copyHost;
    driver.api.cuMemcpy2DAsync_v2 = Fake.copy2d;
    driver.api.cuStreamBeginCapture_v2 = Fake.begin;
    driver.api.cuStreamEndCapture = Fake.end;
    driver.api.cuGraphGetNodes = Fake.getNodes;
    driver.api.cuGraphKernelNodeGetParams_v2 = Fake.getParams;
    driver.api.cuGraphInstantiateWithFlags = Fake.instantiate;
    driver.api.cuGraphExecKernelNodeSetParams_v2 = Fake.setParams;
    driver.api.cuGraphLaunch = Fake.graphLaunch;
    driver.api.cuGraphExecDestroy = Fake.dropExec;
    driver.api.cuGraphDestroy = Fake.dropGraph;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.scratch_rows = q.engine.grid_rows;
    e.d = &driver;
    e.g.hidden = 256;
    e.stream = .{ .d = &driver, .handle = null };
    e.learning = null;
    e.pockets.layers = &.{};
    inline for (@typeInfo(@TypeOf(e.fns)).@"struct".field_names) |name| @field(e.fns, name) = function(&driver, 10);
    e.affine = .{
        .group_fn = function(&driver, 10),
        .mlp128_fn = function(&driver, 10),
        .gemv_fn = function(&driver, 10),
        .verify_fn = function(&driver, 10),
        .verify64_fn = function(&driver, 10),
        .verify128_fn = function(&driver, 10),
        .split_fn = function(&driver, 10),
        .seg_fn = function(&driver, 10),
        .many_fn = function(&driver, 10),
        .prefill_fn = function(&driver, 10),
        .prefill22_fn = function(&driver, 10),
        .prefill_many_fn = function(&driver, 10),
        .words_fn = function(&driver, 10),
        .scales_fn = function(&driver, 10),
        .gemv_blocks = 0,
        .split = null,
        .pdl = false,
        .wide_verify = false,
    };
    var d: q.draft.Draft = undefined;
    d.gpa = T.allocator;
    d.e = &e;
    d.projection_role = .proposal;
    d.frame_streams = 1;
    d.graphs = .{ .on = true };
    defer d.graphs.deinit(T.allocator);
    d.c = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 256, .heads = 2, .kv_heads = 1, .head_dim = 128, .block = 8, .mask = 255, .window = 2048, .group = 16, .rank = 2, .topk = 2, .eps = 1e-6, .theta = 10000000, .taps = @splat(0), .n_taps = 1 };
    inline for (@typeInfo(@TypeOf(d.fns)).@"struct".field_names) |name| @field(d.fns, name) = function(&driver, 10);
    d.fns.gather_taps = function(&driver, taps_id);
    d.fns.gather_rows = function(&driver, rows_id);
    d.fns.norm_rope = function(&driver, rope_id);
    d.fns.ring_put = function(&driver, ring_id);
    var layer: @TypeOf(d.w.layers[0]) = undefined;
    inline for (@typeInfo(@TypeOf(layer)).@"struct".field_names, @typeInfo(@TypeOf(layer)).@"struct".field_types) |name, F| {
        @field(layer, name) = if (F == cuda.qlinear.Affine4) weight() else 1;
    }
    layer.k.n = 128;
    layer.v.n = 128;
    var pair = [2]@TypeOf(layer){ layer, layer };
    pair[1].k_norm = 2;
    d.w.layers = &pair;
    d.w.fc = weight();
    d.w.hidden_norm = 1;
    const measured = try d.layout(false);
    d.scratch = .{ .d = &driver, .ptr = 0x100000, .len = measured };
    _ = try d.layout(true);
    var first: q.draft.Ctx = .{ .ring = .{ .d = &driver, .ptr = 0x400000, .len = 1 } };
    var later: q.draft.Ctx = .{ .ring = .{ .d = &driver, .ptr = 0x800000, .len = 1 } };
    try T.expect(setenv("TF_DFLASH_GRAPH", "1", 1) == 0);
    defer _ = unsetenv("TF_DFLASH_GRAPH");
    const kept = [_]u32{ 0, 2, 3 };
    for ([_]?[]const u32{ null, &kept }) |rs| {
        Fake.reset();
        try G.absorbRows(&d, &later, 0x9000, 4242, 3, rs);
        const want = Fake.eager[0..Fake.eager_n];
        try T.expect(try G.absorb(&d, &first, 0x7000, 17, 3, rs, false));
        try T.expect(try G.absorb(&d, &later, 0x9000, 4242, 3, rs, false));
        try T.expectEqual(want.len, Fake.ran_n);
        for (want, Fake.ran[0..Fake.ran_n]) |w, r| {
            try T.expectEqual(w.f, r.f);
            try T.expectEqualSlices(c_uint, &w.geometry, &r.geometry);
            try T.expectEqualSlices(u8, &w.data, &r.data);
        }
        try T.expectEqual(@as(usize, 2), Fake.launches);
    }
    try T.expect(!try G.absorb(&d, &later, 0x9000, 1, 3, null, true));
    try T.expect(!try G.absorb(&d, &later, 0x9000, 1, 17, null, false));
    const contexts = [_]*q.draft.Ctx{&later};
    Fake.reset();
    try T.expectEqual(@as(?u64, 0x5000), try G.hold(&d, 1, &contexts, true, holdBody));
    try T.expectEqual(@as(usize, 0), Fake.eager_n);
    try T.expectEqual(@as(?u64, 0x5000), try G.hold(&d, 1, &contexts, true, holdBody));
    try T.expectEqual(@as(usize, 1), Fake.captured().node_n);
    try T.expectEqual(@as(usize, 2), Fake.launches);
    _ = try G.hold(&d, 1, &contexts, false, holdBody);
    try T.expectEqual(@as(usize, 2), d.graphs.holds.count());
    d.graphs.clear(T.allocator);
    try T.expectEqual(@as(usize, 4), Fake.exec_drops); // two holds and both absorb variants
    try T.expectEqual(@as(usize, 0), d.graphs.holds.count() + d.graphs.absorbs.count());
    try T.expectEqual(@as(usize, 0), d.graphs.stats.counts.count());
    d.graphs.stats.on = true;
    _ = try G.hold(&d, 1, &contexts, true, holdBody);
    _ = try G.hold(&d, 1, &contexts, true, holdBody);
    var counts = d.graphs.stats.counts.iterator();
    while (counts.next()) |entry| if (entry.key_ptr.narrow) {
        try T.expectEqual(@as(usize, 1), entry.value_ptr.captured);
        try T.expectEqual(@as(usize, 1), entry.value_ptr.replayed);
        try T.expectEqual(@as(usize, 1), entry.value_ptr.key_miss);
        try T.expectEqual(@as(usize, 0), entry.value_ptr.eager);
        try T.expectEqual(@as(usize, 0), entry.value_ptr.eviction);
        try T.expectEqual(@as(usize, 0), entry.value_ptr.width_not_graphable);
    };
    var stats_buf: [2048]u8 = undefined;
    var stats_out = std.Io.Writer.fixed(&stats_buf);
    try G.writeStats(&d.graphs, &stats_out);
    try T.expect(std.mem.indexOf(u8, stats_out.buffered(), "width_mask=00000000000000000000000000000000 captured=1 replayed=1 eager=0 key_miss=1 eviction=0 width_not_graphable=0") != null);
    d.graphs.stats.on = false;
    d.graphs.stats.counts.clearRetainingCapacity();
    d.graphs.clear(T.allocator);
    var mixed_contexts: [4]q.draft.Ctx = undefined;
    var mixed_refs: [4]*q.draft.Ctx = undefined;
    for (&mixed_contexts, &mixed_refs) |*ctx, *ref| {
        ctx.* = .{ .ring = .{ .d = &driver, .ptr = 0x400000, .len = 1 } };
        ref.* = ctx;
    }
    Fake.reset();
    const mixed_masks = [_]u4{ 0b0011, 0b0101 };
    for (mixed_masks) |mask| {
        setWidthMask(&mixed_contexts, mask);
        try T.expectEqual(@as(?u64, null), try G.hold(&d, 4, &mixed_refs, false, mixedHoldBody));
        const launches = Fake.launches;
        const captures = Fake.captures;
        try T.expectEqual(@as(?u64, null), try G.hold(&d, 4, &mixed_refs, false, mixedHoldBody));
        try T.expectEqual(launches + 1, Fake.launches);
        try T.expectEqual(captures, Fake.captures);
        try expectWidthMask(&mixed_contexts);
    }
    Fake.reset();
    for ([_]u4{ mixed_masks[0], mixed_masks[1], mixed_masks[0] }) |mask| {
        setWidthMask(&mixed_contexts, mask);
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHoldBody);
        try expectWidthMask(&mixed_contexts);
    }
    try T.expectEqual(@as(usize, 0), Fake.captures);
    d.graphs.clear(T.allocator);
    d.graphs.stats.on = true;
    Fake.reset();
    for (1..9) |mask| {
        setWidthMask(&mixed_contexts, mask);
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHoldBody);
    }
    try T.expectEqual(@as(usize, 8), d.graphs.holds.count());
    for (1..9) |mask| {
        setWidthMask(&mixed_contexts, mask);
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHoldBody);
        try expectWidthMask(&mixed_contexts);
    }
    const drops = Fake.exec_drops;
    setWidthMask(&mixed_contexts, 9);
    _ = try G.hold(&d, 4, &mixed_refs, false, mixedHoldBody);
    try T.expectEqual(drops + 1, Fake.exec_drops);
    try T.expectEqual(@as(usize, 8), d.graphs.holds.count());
    var eviction_total: usize = 0;
    var eviction_it = d.graphs.stats.counts.iterator();
    while (eviction_it.next()) |entry| eviction_total += entry.value_ptr.eviction;
    try T.expectEqual(@as(usize, 1), eviction_total);
    stats_out = std.Io.Writer.fixed(&stats_buf);
    try G.writeStats(&d.graphs, &stats_out);
    try T.expect(std.mem.indexOf(u8, stats_out.buffered(), "eviction=1") != null);
    d.graphs.stats.on = false;
    d.graphs.stats.counts.clearRetainingCapacity();
    for ([_]usize{ 9, 8, 9 }) |mask| {
        setWidthMask(&mixed_contexts, mask);
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHoldBody);
        try expectWidthMask(&mixed_contexts);
    }
    const settled_drops = Fake.exec_drops;
    for (1..9) |mask| {
        for (mixed_refs[0..3], 0..) |ctx, i| ctx.wide = mask & (@as(usize, 1) << @intCast(i)) != 0;
        _ = try G.hold(&d, 3, mixed_refs[0..3], false, mixedHoldBody);
    }
    try T.expectEqual(@as(usize, 16), d.graphs.holds.count());
    d.graphs.clear(T.allocator);
    try T.expectEqual(settled_drops + 16, Fake.exec_drops);

    var high_contexts: [128]q.draft.Ctx = undefined;
    var high_refs: [128]*q.draft.Ctx = undefined;
    for (&high_contexts, &high_refs) |*ctx, *ref| {
        ctx.* = .{ .ring = .{ .d = &driver, .ptr = 0x400000, .len = 1 } };
        ref.* = ctx;
    }
    Fake.reset();
    const masks = [_]u128{ 0, @as(u128, 1) << 63, @as(u128, 1) << 64, @as(u128, 1) << 126, @as(u128, 1) << 127, std.math.maxInt(u128) };
    for (masks) |mask| {
        for (&high_contexts, 0..) |*ctx, i| ctx.wide = mask & (@as(u128, 1) << @intCast(i)) != 0;
        _ = try G.hold(&d, 128, &high_refs, false, mixedHoldBody);
        const launches = Fake.launches;
        _ = try G.hold(&d, 128, &high_refs, false, mixedHoldBody);
        try T.expectEqual(launches + 1, Fake.launches);
        try T.expectEqual(@as(usize, 128), Fake.ran_n);
        for (high_contexts, 0..) |ctx, i| try T.expectEqual(@as(usize, if (ctx.wide) wide_id else taps_id), Fake.ran[i].f);
    }

    d.graphs.clear(T.allocator);
    Fake.reset();
    var head = weight();
    head.n = 248320;
    head.npad = head.n;
    e.head = .{ .a4 = head };
    e.draws.topk_slices_fn = function(&driver, 30);
    e.draws.topk_merge_fn = function(&driver, 31);
    d.c.vocab = 248320;
    d.words = @intFromPtr(&mixed_words);
    for (&mixed_words, 0..) |*byte, i| byte.* = @truncate(i * 31 + 7);
    for (mixed_masks) |mask| {
        setWidthMask(&mixed_contexts, mask);
        @memset(mixed_staging[0..], 0xa5);
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHeadBody);
        try expectMixedWords(&mixed_refs);
        try T.expectEqual(@as(usize, 10), Fake.captured().copy_n);
        @memset(mixed_staging[0..], 0x5a);
        const launches = Fake.launches;
        const captures = Fake.captures;
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHeadBody);
        try T.expectEqual(launches + 1, Fake.launches);
        try T.expectEqual(captures, Fake.captures);
        try expectMixedWords(&mixed_refs);
        try T.expectEqual(@as(usize, 0), Fake.eager_n);
    }
    Fake.reset();
    for ([_]u4{ mixed_masks[0], mixed_masks[1], mixed_masks[0] }) |mask| {
        setWidthMask(&mixed_contexts, mask);
        @memset(mixed_staging[0..], 0x5a);
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHeadBody);
        try expectMixedWords(&mixed_refs);
    }
    try T.expectEqual(@as(usize, 0), Fake.captures);
    @memset(mixed_staging[0..], 0x5a);
    try T.expect(try G.absorb(&d, &later, 0x9000, 4, 4, null, false));
    try T.expectEqual(@as(usize, 0), Fake.captured().copy_n);
    const captures = Fake.captures;
    _ = try G.hold(&d, 4, &mixed_refs, false, mixedHeadBody);
    try T.expectEqual(captures, Fake.captures);
    try expectMixedWords(&mixed_refs);
    for (1..10) |mask| {
        setWidthMask(&mixed_contexts, mask);
        @memset(mixed_staging[0..], 0x5a);
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHeadBody);
        try expectMixedWords(&mixed_refs);
    }
    try T.expectEqual(@as(usize, 8), d.graphs.holds.count());
    for ([_]usize{ 9, 8, 9 }) |mask| {
        setWidthMask(&mixed_contexts, mask);
        @memset(mixed_staging[0..], 0x5a);
        _ = try G.hold(&d, 4, &mixed_refs, false, mixedHeadBody);
        try expectMixedWords(&mixed_refs);
    }
    try T.expect(Fake.last_dropped != 0);
    const launches = Fake.launches;
    const ran_n = Fake.ran_n;
    const dead: abi.GraphExec = @ptrFromInt(Fake.last_dropped);
    @memset(mixed_staging[0..], 0x5a);
    try T.expect(Fake.graphLaunch(dead, null) != abi.success);
    try T.expectEqual(launches, Fake.launches);
    try T.expectEqual(ran_n, Fake.ran_n);
    for (mixed_staging) |byte| try T.expectEqual(@as(u8, 0x5a), byte);
    try T.expectEqual(@as(usize, 0), Fake.eager_n);
    Fake.refuse = true;
    try T.expect(!try G.absorb(&d, &later, 0x9000, 5, 3, null, false));
    try T.expect(!d.graphs.on);
    try T.expectEqual(@as(usize, 1), d.graphs.absorbs.count());
    Fake.reset();
    try T.expectEqual(@as(?u64, 0x5000), try G.hold(&d, 1, &contexts, true, holdBody));
    try T.expectEqual(@as(usize, 1), Fake.eager_n); // off after a refusal: the kernels run eagerly
    d.graphs.stats.on = true;
    _ = try G.hold(&d, 1, &contexts, true, holdBody);
    var eager_total: usize = 0;
    var eager_it = d.graphs.stats.counts.iterator();
    while (eager_it.next()) |entry| {
        eager_total += entry.value_ptr.eager;
        try T.expectEqual(@as(usize, 0), entry.value_ptr.width_not_graphable);
    }
    try T.expectEqual(@as(usize, 1), eager_total);
    stats_out = std.Io.Writer.fixed(&stats_buf);
    try G.writeStats(&d.graphs, &stats_out);
    try T.expect(std.mem.indexOf(u8, stats_out.buffered(), "eager=1 key_miss=0 eviction=0 width_not_graphable=0") != null);
    d.graphs.stats.on = false;
    d.graphs.stats.counts.clearRetainingCapacity();
}
