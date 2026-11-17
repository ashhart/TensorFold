//! router_gemv.cu's weight denominator sum + eps: 1e-20 (release) keeps zero sums finite, 0 (EXL3) is the sum.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const rg = dsv41.ops.rg;
const Gpu = check.Gpu;

/// Experts and top-k of the backbone (6 of 384) and the DSpark blocks (3 of 128); rt, nw, ew: narrow and wide kernels.
const Case = struct { e: u32, k: u32, rows: u32, rt: u32, nw: u32, ew: u32 };

const cases = [_]Case{
    .{ .e = 384, .k = 6, .rows = 1, .rt = 1, .nw = 2, .ew = 0 },
    .{ .e = 384, .k = 6, .rows = 20, .rt = 8, .nw = 8, .ew = 2 },
    .{ .e = 128, .k = 3, .rows = 1, .rt = 1, .nw = 2, .ew = 0 },
    .{ .e = 128, .k = 3, .rows = 20, .rt = 8, .nw = 4, .ew = 1 },
};

const bf16_one: u16 = 0x3f80;
const bf16_minus_one: u16 = 0xbf80;

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

const Bufs = struct {
    x: cuda.DeviceBuffer,
    w: cuda.DeviceBuffer,
    bias: cuda.DeviceBuffer,
    pick: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    lg: cuda.DeviceBuffer,
    cnt: cuda.DeviceBuffer,

    fn deinit(b: *Bufs) void {
        inline for (.{ "x", "w", "bias", "pick", "wts", "lg", "cnt" }) |f| @field(b, f).free();
    }
};

/// One route of `c` with `eps`: the picks and weights as bytes (caller frees).
fn route(gpu: Gpu, o: anytype, s: cuda.Stream, b: Bufs, c: Case, eps: f32) ![2][]u8 {
    try b.pick.fill8(0xff, s.handle);
    try b.wts.fill8(0x7f, s.handle); // finite (3.4e38): an unwritten weight fails the NaN, +0 and row-sum checks
    try o.route(.{
        .x = b.x.ptr,
        .xs = rg.D,
        .w = b.w.ptr,
        .bias = b.bias.ptr,
        .pick = b.pick.ptr,
        .wts = b.wts.ptr,
        .lg = b.lg.ptr,
        .cnt = b.cnt.ptr,
        .R = @intCast(c.rows),
        .E = @intCast(c.e),
        .K = @intCast(c.k),
        .slots = @intCast(c.k),
        .select = 1,
        .scale = 1.5,
        .eps = eps,
    }, c.rt, c.nw, c.ew, b.lg.len / 4, b.cnt.len / 4);
    try s.synchronize();
    return .{ try check.download(gpu, b.pick), try check.download(gpu, b.wts) };
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const o = k.others(stream);
    const a = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x5eed_e95);
    const rnd = prng.random();
    for (cases) |c| {
        const x = try a.alloc(u16, @as(usize, c.rows) * rg.D);
        defer a.free(x);
        const w = try a.alloc(u16, @as(usize, c.e) * rg.D);
        defer a.free(w);
        const bias = try a.alloc(f32, c.e);
        defer a.free(bias);
        const cnt = try a.alloc(i32, 128);
        defer a.free(cnt);
        @memset(cnt, 0);
        var b: Bufs = .{
            .x = try cuda.DeviceBuffer.alloc(gpu.d, x.len * 2),
            .w = try cuda.DeviceBuffer.alloc(gpu.d, w.len * 2),
            .bias = try cuda.DeviceBuffer.alloc(gpu.d, bias.len * 4),
            .pick = try cuda.DeviceBuffer.alloc(gpu.d, @as(usize, c.rows) * c.k * 4),
            .wts = try cuda.DeviceBuffer.alloc(gpu.d, @as(usize, c.rows) * c.k * 4),
            .lg = try cuda.DeviceBuffer.alloc(gpu.d, @as(usize, c.rows) * c.e * 4),
            .cnt = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(cnt)),
        };
        defer b.deinit();

        // all-zero scores: every logit -5120 (exact), softplus 0, sqrt 0; ties pick ids 0..k-1
        @memset(x, bf16_one);
        @memset(w, bf16_minus_one);
        @memset(bias, 0);
        try b.x.upload(0, std.mem.sliceAsBytes(x));
        try b.w.upload(0, std.mem.sliceAsBytes(w));
        try b.bias.upload(0, std.mem.sliceAsBytes(bias));
        const zero = try route(gpu, o, stream, b, c, 0);
        defer for (zero) |z| a.free(z);
        const kept = try route(gpu, o, stream, b, c, 1e-20);
        defer for (kept) |z| a.free(z);
        const picks0 = std.mem.bytesAsSlice(i32, zero[0]);
        const w0 = std.mem.bytesAsSlice(f32, zero[1]);
        const w1 = std.mem.bytesAsSlice(f32, kept[1]);
        try check.sameBytes("router eps: all-zero picks equal at eps 0 and 1e-20", kept[0], zero[0]);
        for (0..c.rows) |r| for (0..c.k) |t| {
            const i = r * c.k + t;
            try check.expect(picks0[i] == @as(i32, @intCast(t)), "router eps {d}/{d}: all-zero row {d} slot {d} picks {d}, not the lower id", .{ c.k, c.e, r, t, picks0[i] });
            try check.expect(std.math.isNan(w0[i]), "router eps {d}/{d}: eps 0 weight of a zero sum is {d}, not NaN", .{ c.k, c.e, w0[i] });
            try check.expect(@as(u32, @bitCast(w1[i])) == 0, "router eps {d}/{d}: eps 1e-20 weight of a zero sum is {d}, not +0", .{ c.k, c.e, w1[i] });
        };

        // ordinary scores: the sum is far above 1e-13, so sum + 1e-20 is the sum and the bytes are equal
        for (x) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
        for (w) |*v| v.* = bf16((rnd.float(f32) * 2 - 1) * 0.02);
        for (bias) |*v| v.* = (rnd.float(f32) * 2 - 1) * 0.1;
        try b.x.upload(0, std.mem.sliceAsBytes(x));
        try b.w.upload(0, std.mem.sliceAsBytes(w));
        try b.bias.upload(0, std.mem.sliceAsBytes(bias));
        const plain = try route(gpu, o, stream, b, c, 0);
        defer for (plain) |z| a.free(z);
        const guarded = try route(gpu, o, stream, b, c, 1e-20);
        defer for (guarded) |z| a.free(z);
        try check.sameBytes("router eps: ordinary picks equal at eps 0 and 1e-20", guarded[0], plain[0]);
        try check.sameBytes("router eps: ordinary weights equal at eps 0 and 1e-20", guarded[1], plain[1]);
        const wp = std.mem.bytesAsSlice(f32, plain[1]);
        for (0..c.rows) |r| {
            var sum: f64 = 0;
            for (wp[r * c.k ..][0..c.k]) |v| {
                try check.expect(std.math.isFinite(v) and v > 0, "router eps {d}/{d}: an ordinary weight is {d}", .{ c.k, c.e, v });
                sum += v;
            }
            try check.expect(@abs(sum - 1.5) < 1e-5, "router eps {d}/{d}: row {d} weights sum to {d}, not 1.5", .{ c.k, c.e, r, sum });
        }
        check.pass("router eps top-{d} of {d}, {d} rows (rt {d} nw {d} ew {d}): zero sums NaN at 0 and +0 at 1e-20, ordinary bytes equal", .{ c.k, c.e, c.rows, c.rt, c.nw, c.ew });
    }
}
