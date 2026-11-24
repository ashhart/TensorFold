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

const Call = struct {
    f: usize,
    geometry: [6]c_uint,
    data: [512]u8 = @splat(0),

    fn of(f: abi.Function, geometry: [6]c_uint, args: ?[*]?*anyopaque) Call {
        var c: Call = .{ .f = @intFromPtr(f), .geometry = geometry };
        const sizes: []const usize = switch (c.f) {
            taps_id => &.{ 8, 4, 8, 4, 8 },
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

const Fake = struct {
    var eager: [64]Call = undefined;
    var eager_n: usize = 0;
    var nodes: [64]Call = undefined;
    var node_n: usize = 0;
    var ran: [64]Call = undefined;
    var ran_n: usize = 0;
    var capturing = false;
    var launches: usize = 0;
    var exec_drops: usize = 0;
    var refuse = false;

    fn reset() void {
        eager_n = 0;
        node_n = 0;
        ran_n = 0;
        capturing = false;
        launches = 0;
        exec_drops = 0;
        refuse = false;
    }
    fn launch(f: abi.Function, x: c_uint, y: c_uint, z: c_uint, bx: c_uint, by: c_uint, bz: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        const c = Call.of(f, .{ x, y, z, bx, by, bz }, args);
        if (capturing) {
            nodes[node_n] = c;
            node_n += 1;
        } else {
            eager[eager_n] = c;
            eager_n += 1;
        }
        return abi.success;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        capturing = true;
        node_n = 0;
        return abi.success;
    }
    fn end(_: abi.Stream, g: *abi.Graph) callconv(.c) abi.Result {
        capturing = false;
        g.* = @ptrFromInt(1);
        return abi.success;
    }
    fn getNodes(_: abi.Graph, out: ?[*]abi.GraphNode, n: *usize) callconv(.c) abi.Result {
        if (out) |o| for (0..node_n) |i| {
            o[i] = @ptrFromInt(i + 1);
        };
        n.* = node_n;
        return abi.success;
    }
    fn getParams(node: abi.GraphNode, p: *abi.KernelNodeParams) callconv(.c) abi.Result {
        p.func = @ptrFromInt(nodes[@intFromPtr(node) - 1].f);
        return abi.success;
    }
    fn instantiate(out: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        if (refuse) return 800;
        out.* = @ptrFromInt(1);
        return abi.success;
    }
    fn setParams(_: abi.GraphExec, node: abi.GraphNode, p: *const abi.KernelNodeParams) callconv(.c) abi.Result {
        nodes[@intFromPtr(node) - 1] = Call.of(p.func, .{ p.grid_x, p.grid_y, p.grid_z, p.block_x, p.block_y, p.block_z }, p.params);
        return abi.success;
    }
    fn graphLaunch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        @memcpy(ran[0..node_n], nodes[0..node_n]);
        ran_n = node_n;
        launches += 1;
        return abi.success;
    }
    fn dropExec(_: abi.GraphExec) callconv(.c) abi.Result {
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

test "Draft frames: a DFlash2 graph replays the eager launches with this round's arguments" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
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
    try T.expectEqual(@as(usize, 1), Fake.node_n);
    try T.expectEqual(@as(usize, 2), Fake.launches);
    _ = try G.hold(&d, 1, &contexts, false, holdBody);
    try T.expectEqual(@as(usize, 2), d.graphs.holds.count());
    d.graphs.clear(T.allocator);
    try T.expectEqual(@as(usize, 4), Fake.exec_drops); // two holds and both absorb variants
    try T.expectEqual(@as(usize, 0), d.graphs.holds.count() + d.graphs.absorbs.count());
    Fake.refuse = true;
    try T.expect(!try G.absorb(&d, &later, 0x9000, 5, 3, null, false));
    try T.expect(!d.graphs.on);
    try T.expectEqual(@as(usize, 0), d.graphs.absorbs.count());
    Fake.reset();
    try T.expectEqual(@as(?u64, 0x5000), try G.hold(&d, 1, &contexts, true, holdBody));
    try T.expectEqual(@as(usize, 1), Fake.eager_n); // off after a refusal: the kernels run eagerly
}
