//! combine-add: ordered FP32 combine plus shared add, BF16 rounding, offset bounds and device time.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const Gpu = check.Gpu;
const Kind = enum { expert, weight, shared };

const Rng = struct {
    s: u64 = 0x9e3779b97f4a7c15,
    fn unit(r: *Rng) f32 {
        r.s ^= r.s << 13;
        r.s ^= r.s >> 7;
        r.s ^= r.s << 17;
        return @as(f32, @floatFromInt(r.s >> 40)) / @as(f32, 1 << 24);
    }
};

fn input(gpu: Gpu, n: usize, kind: Kind, D: usize, slots: usize, ordered: bool) !cuda.DeviceBuffer {
    const h = try gpu.gpa.alloc(f32, n + 8);
    defer gpu.gpa.free(h);
    @memset(h[0..4], 0);
    @memset(h[n + 4 ..], 0);
    var rng: Rng = .{};
    for (h[4..][0..n], 0..) |*v, i| {
        if (ordered) {
            const sign: f32 = if (i % D % 4 < 2) 1 else -1;
            v.* = switch (kind) {
                .expert => sign * ([_]f32{ 0x1p24, 1, -0x1p24, 1, 0, 0 })[(i / D % slots) % 6],
                .weight => 1,
                .shared => ([_]f32{ 0x1p-8, 3 * 0x1p-8, -0x1p-8, -3 * 0x1p-8 })[i % 4],
            };
        } else {
            const u = rng.unit();
            v.* = if (kind == .weight) 0.05 + 0.45 * u else -2 + 4 * u;
        }
    }
    return cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(h));
}

const State = struct {
    o: dsv41.exl3.Ops,
    p: @typeInfo(@TypeOf(dsv41.Kernels.others)).@"fn".return_type.?,
    y: u64,
    w: u64,
    sy: u64,
    routed: u64,
    out: [2]u64,
    rows: usize,
    D: usize,
    slots: usize,

    fn launch(s: *const State, fused: bool) !void {
        if (fused) return s.o.combineAdd(s.y, s.w, s.sy, s.out[1], s.rows, s.D, s.slots);
        try s.o.combine(s.y, s.w, s.routed, s.rows, s.D, s.slots);
        try s.p.addBf16(s.routed, s.sy, s.out[0], s.rows * s.D);
    }

    fn time(s: *const State, stream: cuda.Stream, start: cuda.Event, end: cuda.Event, fused: bool) !f64 {
        for (0..2) |_| try s.launch(fused);
        var ms: [5]f64 = undefined;
        for (&ms) |*v| {
            try start.record(stream);
            try s.launch(fused);
            try end.record(stream);
            try end.synchronize();
            v.* = try cuda.Event.elapsedMs(start, end);
        }
        return check.median(&ms);
    }
};

fn cell(gpu: Gpu, k: *const dsv41.Kernels, rows: usize, D: usize, slots: usize, ordered: bool) !void {
    const n = rows * D;
    var y = try input(gpu, n * slots, .expert, D, slots, ordered);
    defer y.free();
    var w = try input(gpu, rows * slots, .weight, D, slots, ordered);
    defer w.free();
    var sy = try input(gpu, n, .shared, D, slots, ordered);
    defer sy.free();
    var routed = try cuda.DeviceBuffer.alloc(gpu.d, 4 * n + 32);
    defer routed.free();
    var plain = try cuda.DeviceBuffer.alloc(gpu.d, 2 * n + 16);
    defer plain.free();
    var fused = try cuda.DeviceBuffer.alloc(gpu.d, 2 * n + 16);
    defer fused.free();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    try plain.fill8(0xa5, stream.handle);
    try fused.fill8(0xa5, stream.handle);
    const s: State = .{ .o = k.experts(stream), .p = k.others(stream), .y = y.ptr + 16, .w = w.ptr + 16, .sy = sy.ptr + 16, .routed = routed.ptr + 16, .out = .{ plain.ptr + 8, fused.ptr + 8 }, .rows = rows, .D = D, .slots = slots };
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    const before = try s.time(stream, start, end, false);
    const after = try s.time(stream, start, end, true);
    const want = try check.download(gpu, plain);
    defer gpu.gpa.free(want);
    const got = try check.download(gpu, fused);
    defer gpu.gpa.free(got);
    try check.sameBytes("combine-add vs combine plus add_bf16", got, want);
    for (0..8) |i| try check.expect(got[i] == 0xa5 and got[got.len - 8 + i] == 0xa5, "combine-add output guard", .{});
    if (ordered and rows == 1 and D == 4 and slots == 6) {
        const ties = [_]u16{ 0x3f80, 0x3f82, 0xbf80, 0xbf82 };
        try check.sameBytes("ordered FMA and BF16 ties", got[8..16], std.mem.asBytes(&ties));
    }
    check.pass("combine-add rows {d} D {d} slots {d} {s}: {d:.3} -> {d:.3} ms BITEXACT", .{ rows, D, slots, if (ordered) "ordered" else "random", before, after });
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels) !void {
    for ([_]bool{ false, true }) |ordered| {
        for ([_]usize{ 17, 129, 4096, 8192, 16384, 32768 }) |rows| try cell(gpu, k, rows, 5120, 6, ordered);
        for ([_]usize{ 1, 17 }) |rows| for ([_]usize{ 4, 1028 }) |D| for ([_]usize{ 1, 6, 32 }) |slots| {
            try cell(gpu, k, rows, D, slots, ordered);
        };
    }
    check.pass("combine-add: 36 cells, ordered FMA, BF16 ties and offset guards", .{});
}
