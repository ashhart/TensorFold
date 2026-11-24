//! Compare Mini's capped, mixed-bit expert launches with the original grouped traversal.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const check = @import("check.zig");

const experts = 128;
const hidden = 2304;
const inter = 1152;
const slots = 8;
const widths = [3]i32{ 6, 7, 8 };
const ex = cuda.exl3_experts;

fn fill(gpu: check.Gpu, bytes: []const u8) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, bytes.len);
    errdefer b.free();
    try b.upload(0, bytes);
    return b;
}

pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var k = try ink.experts.Kernels.load(gpu.d);
    defer k.unload();

    var prng = std.Random.DefaultPrng.init(7513);
    const random = prng.random();
    var words: [3][widths.len]cuda.DeviceBuffer = undefined;
    var loaded: usize = 0;
    defer for (0..loaded) |i| words[i / widths.len][i % widths.len].free();
    for (&words) |*projection| {
        for (projection, widths) |*b, half_bits| {
            const bytes = try a.alloc(u8, hidden * inter * @as(usize, @intCast(half_bits)) / 16);
            defer a.free(bytes);
            random.bytes(bytes);
            b.* = try fill(gpu, bytes);
            loaded += 1;
        }
    }

    var ptr_table: [3][experts]u64 = undefined;
    var half_table: [3][experts]i32 = undefined;
    for (&ptr_table, &half_table, &words, 0..) |*p, *h, w, m| {
        for (p, h, 0..) |*ptr, *bits, e| {
            const index = (e + m) % widths.len;
            ptr.* = w[index].ptr;
            bits.* = widths[index];
        }
    }
    var ptrs = try fill(gpu, std.mem.asBytes(&ptr_table));
    defer ptrs.free();
    var halves = try fill(gpu, std.mem.asBytes(&half_table));
    defer halves.free();

    const scales = try a.alloc(f16, experts * hidden);
    defer a.free(scales);
    @memset(scales, 0.25);
    var scale = try fill(gpu, std.mem.sliceAsBytes(scales));
    defer scale.free();
    const caps: ink.experts.Caps = .{ .gate = 4, .up = 25 };
    const l: ex.Layer = .{
        .gate = ptrs.ptr,
        .up = ptrs.ptr + experts * 8,
        .down = ptrs.ptr + 2 * experts * 8,
        .gate_k2 = halves.ptr,
        .up_k2 = halves.ptr + experts * 4,
        .down_k2 = halves.ptr + 2 * experts * 4,
        .suh_g = scale.ptr,
        .suh_u = scale.ptr,
        .svh_g = scale.ptr,
        .svh_u = scale.ptr,
        .suh_d = scale.ptr,
        .svh_d = scale.ptr,
        .experts = experts,
        .dims = hidden,
        .width = inter,
        .codebook = .mul1,
        .act_mode = 3,
        .caps = .{ caps.gate, caps.up },
    };

    for ([_]usize{ 1, 17, 512 }) |rows| {
        try compare(gpu, s, &k, l, caps, random, rows);
    }
    std.debug.print("SUMMARY Mini mixed-width EXL3: 3/3 shapes bit-equal, 3/3 planted controls caught\n", .{});
}

fn compare(gpu: check.Gpu, s: cuda.Stream, k: *const ink.experts.Kernels, l: ex.Layer, caps: ink.experts.Caps, random: std.Random, rows: usize) !void {
    const a = gpu.gpa;
    const inputs = try a.alloc(u16, rows * hidden);
    defer a.free(inputs);
    for (inputs) |*x| x.* = @import("core").bf16_round.bits((random.float(f32) - 0.5) * 0.125);
    var x = try fill(gpu, std.mem.sliceAsBytes(inputs));
    defer x.free();
    const picks = try a.alloc(i32, rows * slots);
    defer a.free(picks);
    const weights = try a.alloc(f32, picks.len);
    defer a.free(weights);
    @memset(weights, 1.0 / @as(f32, slots));
    for (picks, 0..) |*e, i| e.* = @intCast((i / slots + i % slots) % experts);
    var pick = try fill(gpu, std.mem.sliceAsBytes(picks));
    defer pick.free();
    var weight = try fill(gpu, std.mem.sliceAsBytes(weights));
    defer weight.free();
    var buf = try cuda.DeviceBuffer.alloc(gpu.d, try ink.experts.scratchBytes(l, rows, slots));
    defer buf.free();
    const sc = try ex.scratch(buf.ptr, l, rows, slots);
    var want = try cuda.DeviceBuffer.alloc(gpu.d, rows * hidden * 4);
    defer want.free();
    var got = try cuda.DeviceBuffer.alloc(gpu.d, want.len);
    defer got.free();
    try k.main.run(s, l, x.ptr, hidden, .bf16, pick.ptr, weight.ptr, rows, want.ptr, sc);
    try ink.experts.run(k, s, l, caps, .{ .lo = 6, .hi = 8 }, x.ptr, hidden, .bf16, pick.ptr, weight.ptr, rows, got.ptr, sc);
    try s.synchronize();
    const wr = try a.alloc(f32, rows * hidden);
    defer a.free(wr);
    const gr = try a.alloc(f32, rows * hidden);
    defer a.free(gr);
    try want.download(0, std.mem.sliceAsBytes(wr));
    try got.download(0, std.mem.sliceAsBytes(gr));
    var nonzero = false;
    for (wr, gr) |w, g| {
        try check.expect(std.math.isFinite(w) and std.math.isFinite(g), "Mini grouped EXL3: nonfinite output", .{});
        nonzero = nonzero or w != 0;
    }
    const expected = std.mem.sliceAsBytes(wr);
    const actual = std.mem.sliceAsBytes(gr);
    try check.expect(nonzero and std.mem.eql(u8, expected, actual), "Mini grouped EXL3 rows={d}: output differs", .{rows});

    // Read every up projection as 3-bit. Its storage is large enough even when its real width is higher.
    const narrow: [experts]i32 = @splat(widths[0]);
    var width_control = try fill(gpu, std.mem.asBytes(&narrow));
    defer width_control.free();
    var bad = l;
    bad.up_k2 = width_control.ptr;
    try ink.experts.run(k, s, bad, caps, .{ .lo = 6, .hi = 8 }, x.ptr, hidden, .bf16, pick.ptr, weight.ptr, rows, got.ptr, sc);
    try s.synchronize();
    try got.download(0, actual);
    try check.expect(!std.mem.eql(u8, expected, actual), "Mini grouped EXL3 rows={d}: wrong width escaped", .{rows});
    check.pass("Mini grouped EXL3 rows={d}, 128 experts, top-8, halves 6/7/8, bit-equal, width control caught", .{rows});
}
