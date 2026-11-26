//! The actual candidate launch keeps main's arguments and default-off calls, plus an exact reduction witness.
const std = @import("std");
const cuda = @import("cuda");
const q = cuda.qmmf;
const abi = cuda.abi;
const Seen = struct {
    const Call = struct { id: usize, gx: u32, gz: u32, block: u32, shared: u32, cluster: u32 = 0, pdl: bool = false, x: u64, codes: u64, scales: u64, scale: u32, y: u64, part: u64, rows: i32, n: i32, k: i32, sk: i32, npad: i32, stride: i32, band: i32 };
    var call: Call = undefined;
    var count: usize = 0;
    fn arg(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn run(f: abi.Function, gx: c_uint, _: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        count += 1;
        call = .{ .id = @intFromPtr(f), .gx = gx, .gz = gz, .block = bx, .shared = shared, .x = arg(u64, args, 0), .codes = arg(u64, args, 1), .scales = arg(u64, args, 2), .scale = @bitCast(arg(f32, args, 3)), .y = arg(u64, args, 4), .part = arg(u64, args, 5), .rows = arg(i32, args, 6), .n = arg(i32, args, 7), .k = arg(i32, args, 8), .sk = arg(i32, args, 9), .npad = arg(i32, args, 10), .stride = arg(i32, args, 11), .band = arg(i32, args, 12) };
        return abi.success;
    }
    fn extended(c: *const abi.LaunchConfig, f: abi.Function, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
        _ = run(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, args, extra);
        if (c.attrs) |attrs| for (attrs[0..c.num_attrs]) |at| switch (at.id) {
            .cluster_dimension => call.cluster = at.value.cluster_dim.z,
            .programmatic_stream_serialization => call.pdl = at.value.int == 1,
            else => {},
        };
        return abi.success;
    }
};

test "QMMF SK1: physical schedule retains tile and all logical SK2 arguments at every verify width" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Seen.run;
    d.api.cuLaunchKernelEx = Seen.extended;
    const f: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(10) };
    const fused: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(11) };
    const lane: q.Lane = .{ .mod = undefined, .fns = @splat(@splat(f)), .f32_fns = @splat(@splat(f)), .reduce = @splat(f), .half = @splat(@splat(@splat(f))), .half_below = 0, .clusters = true };
    _ = fused;
    var set: q.sk1.Set = undefined;
    for (&set, 0..) |*outs, oi| {
        for (outs, 0..) |*v, bi| {
            v.* = .{ .d = &d, .handle = @ptrFromInt(11 + oi * 2 + bi) };
        }
    }
    const w: q.Weight = .{ .mode = .fp4, .codes = 0x2000, .scales = 0x3000, .scale = 0.0123, .n = 6144, .k = 5120, .npad = 6144 };
    try std.testing.expectEqual(@as(u32, 2), q.slices(w));
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]q.Out{ .bf16, .f32 }) |out| for ([_]bool{ false, true }) |pdl| {
        var l = lane;
        l.pdl = pdl;
        for (1..65) |rows| {
            try l.matmulOut(stream, 0x1000, 5136, rows, w, 0x4000, out, 0);
            const before = Seen.call;
            const old_count = Seen.count;
            try std.testing.expect(!try q.sk1.matmul(null, l, stream, 0x1000, 5136, rows, w, 0x4000, out));
            try std.testing.expectEqual(old_count, Seen.count);
            try std.testing.expectEqual(before, Seen.call);
            try std.testing.expect(try q.sk1.matmul(set, l, stream, 0x1000, 5136, rows, w, 0x4000, out));
            var want = before;
            want.id = if (rows <= 32) 11 + @as(usize, @backingInt(out)) * 2 + @intFromBool(rows > 16) else 10;
            want.gz = 1;
            want.cluster = 0;
            try std.testing.expectEqual(want, Seen.call);
            try std.testing.expectEqual(@as(i32, 2), Seen.call.sk);
        }
    };
    for ([_]usize{ 65, 127, 128, 129, 255 }) |rows| try std.testing.expect(q.sk1.eligible(rows, w));
    for ([_]usize{ 0, 256, 512, 1024, 2048 }) |rows| try std.testing.expect(!q.sk1.eligible(rows, w));
    for ([_]q.Mode{ .fp8, .mxfp8, .fp8g, .bf16 }) |mode| {
        var other = w;
        other.mode = mode;
        try std.testing.expect(!q.sk1.eligible(1, other));
    }
    var other = w;
    other.n = 6143;
    try std.testing.expect(!q.sk1.eligible(1, other));
    other = w;
    other.k = 5184;
    try std.testing.expect(!q.sk1.eligible(1, other));
    other = w;
    other.sk = 1;
    try std.testing.expect(!q.sk1.eligible(1, other));
}

fn baseline(terms: []const f32) f32 {
    var a: f32 = 0;
    var b: f32 = 0;
    for (terms[0 .. terms.len / 2]) |v| a += v;
    for (terms[terms.len / 2 ..]) |v| b += v;
    return a + b;
}
fn candidate(terms: []const f32, reset: bool) f32 {
    var acc: f32 = 0;
    var first: f32 = 0;
    for (terms, 0..) |v, i| {
        if (i == terms.len / 2 and reset) {
            first = acc;
            acc = 0;
        }
        acc += v;
    }
    return first + acc;
}
fn bits(v: f32) u32 {
    return @bitCast(v);
}

test "QMMF SK1: exact FP32 and BF16 bytes reject an unsplit chain and a missing reset" {
    const terms = [_]f32{ 0x1p30, 1, -0x1p30, 1 };
    const want = baseline(&terms);
    try std.testing.expectEqual(bits(want), bits(candidate(&terms, true)));
    try std.testing.expect(bits(want) != bits(candidate(&terms, false)));
    try std.testing.expectEqual(@as(u32, 0), bits(want));
    const narrowed: u16 = @truncate(bits(candidate(&terms, false)) >> 16);
    try std.testing.expect(narrowed != 0);
    var rng = std.Random.DefaultPrng.init(0x514d4d46);
    var terms_many: [320]f32 = undefined;
    for (0..64) |_| {
        for (&terms_many) |*v| v.* = std.math.ldexp(rng.random().float(f32) * 2 - 1, @as(i32, rng.random().uintLessThan(u8, 40)) - 20);
        const base = bits(baseline(&terms_many));
        const got = bits(candidate(&terms_many, true));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&got));
    }
}

test "QMMF SK1: four new instances retain pending resource qualification" {
    const reg = @import("core").contract.registry;
    var found: usize = 0;
    for (reg.cuda_pending) |r| if (std.mem.startsWith(u8, r.launch.name, "qmmf FP4 physical SK1 ")) {
        try std.testing.expectEqual(@as(usize, 2), try reg.checkCuda(std.testing.allocator, &.{r}, true));
        found += 1;
    };
    try std.testing.expectEqual(@as(usize, 4), found);
}
