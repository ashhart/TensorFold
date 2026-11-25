//! Actual launch traces for the fixed-split NVFP4 candidate; these are not GPU arithmetic receipts.
const std = @import("std");
const cuda = @import("cuda");
const q = cuda.qmmf;
const abi = cuda.abi;
const wide = q.exact_tile;
const contract = @import("contract_qmmf_exact.zig");
const core = @import("core");
extern fn tf_qmmf_wide_scale_offset(tile: usize, groups: c_int, group: c_int, chunk: c_int, wrong: bool) usize;
extern fn tf_qmmf_wide_shared() c_int;
extern fn tf_qmmf_wide_order_witness() c_int;

const Trace = struct {
    var id: usize = 0;
    var grid: u32 = 0;
    var shared: u32 = 0;
    var split: i32 = 0;
    var rows_stride: i32 = 0;
    var z: u32 = 0;
    var cluster: u32 = 0;
    var dependent: bool = false;
    var calls: usize = 0;
    var first_id: usize = 0;
    var first_grid: u32 = 0;
    var first_shared: u32 = 0;
    var first_split: i32 = 0;
    fn reset() void {
        calls = 0;
        cluster = 0;
        dependent = false;
    }
    fn launch(f: abi.Function, gx: c_uint, _: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        id = @intFromPtr(f);
        grid = gx;
        shared = sm;
        z = gz;
        split = @as(*const i32, @ptrCast(@alignCast(args.?[if (bx == 256) 3 else 9].?))).*;
        if (bx == 128) rows_stride = @as(*const i32, @ptrCast(@alignCast(args.?[11].?))).*;
        if (calls == 0) {
            first_id = id;
            first_grid = grid;
            first_shared = shared;
            first_split = split;
        }
        calls += 1;
        return abi.success;
    }
    fn extended(config: *const abi.LaunchConfig, f: abi.Function, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
        if (config.attrs) |attrs| for (attrs[0..config.num_attrs]) |attr| switch (attr.id) {
            .cluster_dimension => cluster = attr.value.cluster_dim.z,
            .programmatic_stream_serialization => dependent = attr.value.int == 1,
            else => {},
        };
        return launch(f, config.grid_x, config.grid_y, config.grid_z, config.block_x, config.block_y, config.block_z, config.shared_bytes, config.stream, args, extra);
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }
};

fn lane(d: *cuda.Driver) q.Lane {
    const old: cuda.Function = .{ .d = d, .handle = @ptrFromInt(10) };
    return .{
        .mod = undefined,
        .fns = @splat(@splat(old)),
        .f32_fns = @splat(@splat(old)),
        .reduce = @splat(old),
        .half = @splat(@splat(@splat(old))),
        .half_below = 0,
        .clusters = false,
    };
}

test "qmmf exact tile: eligible gate projection uses half the CTAs and the original K slice" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Trace.launch;
    Trace.reset();
    var l = lane(&d);
    // The baseline has no candidate set; the same test compiles and records its original dispatch.
    if (comptime @hasField(q.Lane, "fp4_wide")) {
        const fast_fn: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(20) };
        l.fp4_wide = .{ .functions = @splat(@splat(fast_fn)), .sm = 121 };
    }
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const w: q.Weight = .{ .mode = .fp4, .codes = 0x100000, .scales = 0x4000000, .n = 17408, .k = 5120, .npad = 17408 };
    try l.matmul(s, 0x8000000, 5120, 16, w, 0x9000000, 0);
    try std.testing.expectEqual(@as(usize, 20), Trace.id);
    try std.testing.expectEqual(@as(u32, 136), Trace.grid);
    try std.testing.expectEqual(@as(u32, 26624), Trace.shared);
    try std.testing.expectEqual(@as(i32, 1), Trace.split);
}

fn candidate(d: *cuda.Driver) q.Lane {
    var l = lane(d);
    var set: wide.Set = .{ .functions = undefined, .sm = 121 };
    for (&set.functions, 0..) |*outs, oi| {
        for (outs, 0..) |*f, ci| f.* = .{ .d = d, .handle = @ptrFromInt(20 + 2 * oi + ci) };
    }
    l.fp4_wide = set;
    return l;
}

fn gate() q.Weight {
    return .{ .mode = .fp4, .codes = 0x100000, .scales = 0x4000000, .n = 17408, .k = 5120, .npad = 17408 };
}

test "qmmf exact tile: launch recorder retains every pinned split, output type, stride, cluster and dependent flag" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuLaunchKernelEx = Trace.extended;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var l = candidate(&d);
    for ([_]usize{ 1, 4, 8, 15, 16 }) |m| for ([_]bool{ false, true }) |clusters| for ([_]q.Out{ .bf16, .f32 }, 0..) |out, oi| for ([_]u32{ 1, 2, 4, 8 }) |sk| {
        l.clusters = clusters;
        l.pdl = true;
        const w = try q.pinned(gate(), sk);
        Trace.reset();
        try l.matmulOut(s, 0x8000000, 5128, m, w, 0x9000000, out, 0xa000000);
        try std.testing.expectEqual(@as(usize, 20) + 2 * oi + @intFromBool(clusters and sk > 1), Trace.first_id);
        try std.testing.expectEqual(@as(u32, 136), Trace.first_grid);
        try std.testing.expectEqual(@as(u32, 26624), Trace.first_shared);
        try std.testing.expectEqual(@as(i32, @intCast(sk)), Trace.first_split);
        try std.testing.expectEqual(@as(i32, if (m == 1) 5120 else 5128), Trace.rows_stride);
        try std.testing.expectEqual(@as(usize, if (!clusters and sk > 1) 2 else 1), Trace.calls);
        try std.testing.expectEqual(if (clusters and sk > 1) sk else 0, Trace.cluster);
        try std.testing.expect(Trace.dependent);
    };
    var down = gate();
    down.n = 5120;
    down.npad = 5120;
    down.k = 17408;
    l.clusters = false;
    l.pdl = false;
    Trace.reset();
    try l.matmulF32(s, 0x8000000, down.k, 16, down, 0x9000000, 0xa000000);
    try std.testing.expectEqual(@as(usize, 22), Trace.first_id);
    try std.testing.expectEqual(@as(u32, 40), Trace.first_grid);
    try std.testing.expectEqual(@as(i32, 4), Trace.first_split);
    try std.testing.expectEqual(@as(usize, 2), Trace.calls);
    try std.testing.expectEqual(@as(usize, 10), Trace.id);
    try std.testing.expectEqual(@as(usize, 16 * 5120 * 4 * 4), l.partBytes(16, down));
    Trace.reset();
    try std.testing.expectError(error.MissingLanePart, l.matmulF32(s, 0x8000000, down.k, 16, down, 0x9000000, 0));
    try std.testing.expectEqual(@as(usize, 0), Trace.calls);
}

test "qmmf exact tile: unset set, old formats, unsupported SMs, odd columns and row17 keep the old dispatch" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Trace.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var l = lane(&d);
    try l.matmul(s, 0x8000000, 5120, 16, gate(), 0x9000000, 0);
    try std.testing.expectEqual(@as(usize, 10), Trace.id);
    try std.testing.expectEqual(@as(u32, 272), Trace.grid);
    l = candidate(&d);
    for ([_]u32{ 80, 86, 89, 90, 110, 119, 122 }) |sm| {
        l.fp4_wide.?.sm = sm;
        try l.matmul(s, 0x8000000, 5120, 16, gate(), 0x9000000, 0);
        try std.testing.expectEqual(@as(usize, 10), Trace.id);
    }
    l.fp4_wide.?.sm = 121;
    for (std.enums.values(q.Mode)) |mode| {
        if (mode == .fp4) continue;
        var w = gate();
        w.mode = mode;
        try l.matmul(s, 0x8000000, 5120, 16, w, 0x9000000, 0);
        try std.testing.expectEqual(@as(usize, 10), Trace.id);
    }
    try l.matmul(s, 0x8000000, 5120, 17, gate(), 0x9000000, 0);
    try std.testing.expectEqual(@as(usize, 10), Trace.id);
    var odd = gate();
    odd.n += 1;
    odd.npad += 128;
    try l.matmul(s, 0x8000000, 5120, 16, odd, 0x9000000, 0);
    try std.testing.expectEqual(@as(usize, 10), Trace.id);
    var padded = gate();
    padded.npad += 128;
    try std.testing.expect(wide.eligible(16, padded));
    for ([_]usize{ 0, 17, 32, 64, 256 }) |m| try std.testing.expect(!wide.eligible(m, gate()));
    padded.npad = padded.n - 1;
    try std.testing.expect(!wide.eligible(16, padded));
}

test "qmmf exact tile: alias, alignment, stride and partial buffers refuse before any launch" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Trace.launch;
    var l = candidate(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    Trace.reset();
    for ([_]u64{ 0x8000000, 0x100000, 0x4000000 }) |dest|
        try std.testing.expectError(error.WideAlias, l.matmul(s, 0x8000000, 5120, 16, gate(), dest, 0));
    try std.testing.expectError(error.BadWideBuffer, l.matmul(s, 0x8000001, 5120, 16, gate(), 0x9000000, 0));
    try std.testing.expectError(error.BadWideBuffer, l.matmul(s, 0x8000000, 5112, 16, gate(), 0x9000000, 0));
    try std.testing.expectError(error.BadWideBuffer, l.matmul(s, 0x8000000, 5121, 16, gate(), 0x9000000, 0));
    const split = try q.pinned(gate(), 4);
    try std.testing.expectError(error.WideAlias, l.matmulF32(s, 0x8000000, 5120, 16, split, 0x9000000, 0x9000000));
    try std.testing.expectError(error.WideAlias, l.matmulF32(s, 0x8000000, 5120, 16, split, 0x9000000, 0x8000000));
    try std.testing.expectEqual(@as(usize, 0), Trace.calls);
}

test "qmmf exact tile: native two-plane addresses match the original packer and catch the old wide scale stride" {
    const a = std.testing.allocator;
    const n = 193;
    const k = 192;
    const scales = try a.alloc(u8, n * (k / 16));
    defer a.free(scales);
    for (scales, 0..) |*v, i| v.* = @truncate((i / 12) * 37 + (i % 12) * 11);
    var packed_scales: [256 * (k / 16)]u8 = undefined;
    cuda.nvfp4.packScales(&packed_scales, scales, n, k);
    var wrong_caught = false;
    for ([_]usize{ 0, 2 }) |tile| for (0..3) |kg| for (0..32) |chunk| {
        const offset = tf_qmmf_wide_scale_offset(tile, 3, @intCast(kg), @intCast(chunk), false);
        const bad = tf_qmmf_wide_scale_offset(tile, 3, @intCast(kg), @intCast(chunk), true);
        for (0..16) |b| {
            const col = (tile + chunk / 16) * 64 + (chunk % 16) * 4 + b / 4;
            const want = if (col < n) scales[col * 12 + kg * 4 + b % 4] else 0;
            try std.testing.expectEqual(want, packed_scales[offset + b]);
            if (bad + b >= packed_scales.len or packed_scales[bad + b] != want) wrong_caught = true;
        }
    };
    try std.testing.expect(wrong_caught);
    try std.testing.expectEqual(@as(c_int, 26624), tf_qmmf_wide_shared());
    try std.testing.expectEqual(@as(c_int, 1), tf_qmmf_wide_order_witness());
}

test "qmmf exact tile: missing candidate images publish no partial set and unsupported devices do no lookup" {
    const Lookup = struct {
        var calls: usize = 0;
        var missing: usize = 0;
        fn function(f: *abi.Function, _: abi.Module, _: [*:0]const u8) callconv(.c) abi.Result {
            calls += 1;
            if (calls == missing) return abi.error_not_found;
            f.* = @ptrFromInt(20 + calls);
            return abi.success;
        }
        fn attr(_: abi.Function, _: abi.FunctionAttribute, _: c_int) callconv(.c) abi.Result {
            return abi.success;
        }
    };
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleGetFunction = Lookup.function;
    d.api.cuFuncSetAttribute = Lookup.attr;
    const mod: cuda.Module = .{ .d = &d, .handle = null };
    Lookup.calls = 0;
    try std.testing.expectEqual(@as(?wide.Set, null), try wide.load(mod, 89));
    try std.testing.expectEqual(@as(usize, 0), Lookup.calls);
    for (1..5) |missing| {
        Lookup.calls = 0;
        Lookup.missing = missing;
        try std.testing.expectEqual(@as(?wide.Set, null), try wide.load(mod, 121));
        try std.testing.expectEqual(missing, Lookup.calls);
    }
    Lookup.calls = 0;
    Lookup.missing = 0;
    try std.testing.expect((try wide.load(mod, 120)) != null);
    try std.testing.expectEqual(@as(usize, 4), Lookup.calls);
}

test "qmmf exact tile: backend hooks compile and the planted order control launches the mutated FP32 entry" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuStreamSynchronize = Trace.sync;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var p: contract.Product = .{
        .rig = .{ .gpu = undefined, .s = s },
        .lane = candidate(&d),
        .format = .f32,
        .control = .{ .d = &d, .handle = @ptrFromInt(90) },
        .d = .{ .a = std.testing.allocator, .shape = .{ .rows = 1, .n = 129, .k = 512 }, .x = &.{}, .w = &.{} },
        .w = .{ .mode = .fp4, .codes = 0x100000, .scales = 0x4000000, .n = 129, .k = 512, .npad = 256 },
        .x = 0x8000000,
        .out = 0x9000000,
        .ldx = 520,
    };
    const kernel = p.kernel();
    try std.testing.expectEqual(core.contract.spec.Gate.shapes, kernel.control.?);
    try std.testing.expect(kernel.vt.replaced != null and kernel.vt.simulate != null);
    Trace.reset();
    try p.run();
    try std.testing.expectEqual(@as(usize, 90), Trace.id);
    try std.testing.expectEqual(@as(u32, 2), Trace.grid);
    try std.testing.expectEqual(@as(i32, 1), Trace.split);
    try std.testing.expectEqual(@as(u32, 26624), Trace.shared);
    try std.testing.expectEqual(@as(usize, 1), Trace.calls);
}
