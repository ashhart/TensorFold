//! The pocket's two kernels (tf_pocket_in, then tf_pocket_out) against tf_pocket_rows: every output byte, random rows.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;
const kern = nemotron.kernels;
const gm = nemotron.glue_math;

const width = kern.pocket_width;
const row_mul = 8; // a row's expert slots: the pocket adds into slot 6, the shared expert's first half
const row_add = 6;

/// Rows of `d` inputs, the first `used` of the pocket's neurons set and the rest zero, the output fp32 or bf16.
const Case = struct { d: usize, rows: usize, used: usize, out_f32: bool };

/// The kernels' copy of the inputs: as given, or a control that sums the same terms in another order.
const Order = enum { same, neurons_reversed, inputs_rotated };

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator,
    o: kern.Ops,
    old: cuda.Function,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn uniform(r: *Rig, n: usize, scale: f32) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = (r.rng.float(f32) * 2 - 1) * scale;
        return out;
    }

    fn get(r: *Rig, ptr: u64, n: usize) ![]u8 {
        try r.o.s.synchronize();
        const out = try r.a.alloc(u8, n);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, n), "cuMemcpyDtoH");
        return out;
    }

    /// tf_pocket_rows over all 64 neurons, as the forward launched it.
    fn reference(r: *Rig, c: Case, y: u64, up: u64, down: u64, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(@as(c_int, @intCast(c.d)));
        a.add(up);
        a.add(down);
        a.add(@as(c_int, width));
        a.add(out);
        a.add(@as(c_int, @intFromBool(c.out_f32)));
        a.add(@as(c_int, row_mul));
        a.add(@as(c_int, row_add));
        try cuda.launch.launch(r.old, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 256 } }, r.o.s, &a);
    }
};

/// Inputs of one case on the device, host copies kept for the controls' reordered layouts.
const Inputs = struct { y: []u16, up: []f32, down: []f32, out: []u8 };

fn inputs(r: *Rig, c: Case) !Inputs {
    const y = try r.a.alloc(u16, c.rows * c.d);
    for (y) |*v| v.* = gm.f32ToBf16((r.rng.float(f32) * 2 - 1) * 2);
    const up = try r.uniform(width * c.d, 1.5 / @sqrt(@as(f32, @floatFromInt(c.d))));
    // the first neuron reads the first row's own direction, so at least that row moves; the rest fire at random
    for (up[0..c.d], y[0..c.d]) |*u, v| u.* = gm.bf16ToF32(v) / @as(f32, @floatFromInt(c.d));
    @memset(up[c.used * c.d ..], 0); // the neurons past the count: zero, as the learner and the loader keep them
    const down = try r.uniform(width * c.d, 0.05); // nonzero past the count too: 1.0.8 multiplied them by zero
    const n = c.rows * row_mul * c.d;
    const start = try r.uniform(n, 1);
    const out = try r.a.alloc(u8, n * @as(usize, if (c.out_f32) 4 else 2));
    if (c.out_f32) {
        @memcpy(out, std.mem.sliceAsBytes(start));
    } else for (std.mem.bytesAsSlice(u16, out), start) |*h, v| {
        h.* = gm.f32ToBf16(v);
    }
    return .{ .y = y, .up = up, .down = down, .out = out };
}

/// tf_pocket_in and tf_pocket_out through the forward's launcher, the inputs laid out by `order`; the output's bytes.
fn split(r: *Rig, c: Case, in: Inputs, order: Order) ![]u8 {
    const y = try r.a.dupe(u16, in.y);
    const up = try r.a.dupe(f32, in.up);
    const down = try r.a.dupe(f32, in.down);
    switch (order) {
        .same => {},
        .neurons_reversed => for (0..c.used / 2) |j| for ([_][]f32{ up, down }) |m| for (0..c.d) |i| {
            std.mem.swap(f32, &m[j * c.d + i], &m[(c.used - 1 - j) * c.d + i]);
        },
        .inputs_rotated => {
            for (0..c.rows) |t| std.mem.rotate(u16, y[t * c.d ..][0..c.d], 1);
            for (0..width) |j| std.mem.rotate(f32, up[j * c.d ..][0..c.d], 1);
        },
    }
    const count: u32 = @intCast(c.used);
    const p: kern.Pocket = .{ .up = try r.dev(std.mem.sliceAsBytes(up)), .down = try r.dev(std.mem.sliceAsBytes(down)), .n = width, .used = c.used, .count = try r.dev(std.mem.asBytes(&count)) };
    const act = try r.dev(try r.a.alloc(u8, c.rows * width * 4));
    const out = try r.dev(in.out);
    try r.gpu.ctx.synchronize();
    try r.o.pocketRows(try r.dev(std.mem.sliceAsBytes(y)), c.d, p, act, out, c.out_f32, c.rows, row_mul, row_add);
    return r.get(out, in.out.len);
}

/// Values of `word` bytes that differ between a and b.
fn differing(a: []const u8, b: []const u8, word: usize) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < a.len) : (i += word) n += @intFromBool(!std.mem.eql(u8, a[i..][0..word], b[i..][0..word]));
    return n;
}

/// One case: the reference, then the two kernels; equal bytes, and the reference must have moved the output.
fn compare(r: *Rig, c: Case) !void {
    const in = try inputs(r, c);
    const word: usize = if (c.out_f32) 4 else 2;
    const ref_out = try r.dev(in.out);
    try r.gpu.ctx.synchronize();
    try r.reference(c, try r.dev(std.mem.sliceAsBytes(in.y)), try r.dev(std.mem.sliceAsBytes(in.up)), try r.dev(std.mem.sliceAsBytes(in.down)), ref_out);
    const want = try r.get(ref_out, in.out.len);
    const moved = differing(want, in.out, word);
    // no neuron in use: tf_pocket_rows added zeros, so the forward's skipped launch leaves the same bytes
    if (c.used == 0) try check.expect(moved == 0, "pocket d={d} rows={d} used=0: tf_pocket_rows changed {d} values", .{ c.d, c.rows, moved });
    if (c.used > 0) try check.expect(moved * 2 > c.d, "pocket d={d} rows={d} used={d}: the reference moved {d} of {d} values", .{ c.d, c.rows, c.used, moved, c.rows * c.d });
    const got = try split(r, c, in, .same);
    const differ = differing(got, want, word);
    if (differ > 0) std.debug.print("DIFFER pocket d={d} rows={d} used={d} out_f32={}: {d} of {d} values\n", .{ c.d, c.rows, c.used, c.out_f32, differ, want.len / word });
    try check.sameBytes("pocket", got, want);
}

/// Controls: the same sums in another order must give other bytes, or the comparison above could not see an order.
fn control(r: *Rig, c: Case, order: Order) !usize {
    const in = try inputs(r, c);
    const word: usize = if (c.out_f32) 4 else 2;
    const ref_out = try r.dev(in.out);
    try r.gpu.ctx.synchronize();
    try r.reference(c, try r.dev(std.mem.sliceAsBytes(in.y)), try r.dev(std.mem.sliceAsBytes(in.up)), try r.dev(std.mem.sliceAsBytes(in.down)), ref_out);
    const want = try r.get(ref_out, in.out.len);
    const differ = differing(try split(r, c, in, order), want, word);
    try check.expect(differ > 0, "pocket control {t} d={d} rows={d} used={d}: equal bytes, so the check cannot see the order", .{ order, c.d, c.rows, c.used });
    return differ;
}

/// An exact zero of `word` bytes, little-endian: -0.0 when `negative`, else +0.0.
fn exactZero(bytes: []const u8, negative: bool) bool {
    for (bytes[0 .. bytes.len - 1]) |b| if (b != 0) return false;
    return bytes[bytes.len - 1] == @as(u8, if (negative) 0x80 else 0);
}

/// A layer with no neuron in use whose output holds exact -0.0: 1.0.8's kernel, the launcher, and a captured launch.
fn negativeZero(r: *Rig, d: usize, rows: usize, out_f32: bool) !usize {
    const c: Case = .{ .d = d, .rows = rows, .used = 0, .out_f32 = out_f32 };
    const in = try inputs(r, c);
    const word: usize = if (out_f32) 4 else 2;
    // an exact -0.0 at every third output of the shared expert's slot, and of slot 0, which no pocket adds to
    var placed: usize = 0;
    for (0..rows) |t| for ([_]usize{ 0, row_add }) |slot| {
        var k: usize = 0;
        while (k < d) : (k += 3) {
            const i = (t * row_mul + slot) * d + k;
            if (out_f32) std.mem.bytesAsSlice(u32, in.out)[i] = 0x8000_0000 else std.mem.bytesAsSlice(u16, in.out)[i] = 0x8000;
            placed += @intFromBool(slot == row_add);
        }
    };
    const y = try r.dev(std.mem.sliceAsBytes(in.y));
    const up = try r.dev(std.mem.sliceAsBytes(in.up));
    const down = try r.dev(std.mem.sliceAsBytes(in.down));
    const none: u32 = 0;
    const count = try r.dev(std.mem.asBytes(&none));
    const act = try r.dev(try r.a.alloc(u8, rows * width * 4));
    const outs = [3]u64{ try r.dev(in.out), try r.dev(in.out), try r.dev(in.out) };
    try r.gpu.ctx.synchronize();
    // 1.0.8: tf_pocket_rows over all 64 neurons, every one zero
    try r.reference(c, y, up, down, outs[0]);
    // now: the launcher, given no neuron in use, launches nothing
    try r.o.pocketRows(y, d, .{ .up = up, .down = down, .n = width, .used = 0, .count = count }, act, outs[1], out_f32, rows, row_mul, row_add);
    // a graph captured while the layer held neurons launches both kernels, which read the device's count of 0
    try r.o.pocketRows(y, d, .{ .up = up, .down = down, .n = width, .used = width, .count = count }, act, outs[2], out_f32, rows, row_mul, row_add);
    const old = try r.get(outs[0], in.out.len);
    const skipped = try r.get(outs[1], in.out.len);
    const captured = try r.get(outs[2], in.out.len);
    // 1.0.8 turned each -0.0 it added to into +0.0 and left every other value's bytes, slot 0's -0.0 included
    var turned: usize = 0;
    var i: usize = 0;
    while (i < in.out.len) : (i += word) {
        if (std.mem.eql(u8, old[i..][0..word], in.out[i..][0..word])) continue;
        try check.expect(exactZero(in.out[i..][0..word], true) and exactZero(old[i..][0..word], false), "pocket -0.0 d={d} rows={d} out_f32={}: tf_pocket_rows changed value {d}, not a -0.0 to +0.0", .{ d, rows, out_f32, i / word });
        turned += 1;
    }
    try check.expect(turned == placed, "pocket -0.0 d={d} rows={d} out_f32={}: tf_pocket_rows turned {d} of the {d} -0.0 it added to", .{ d, rows, out_f32, turned, placed });
    try check.sameBytes("pocket -0.0, no neuron in use: the output as it was", skipped, in.out);
    try check.sameBytes("pocket -0.0, a captured launch reading a count of 0: tf_pocket_rows' bytes", captured, old);
    return turned;
}

/// GPU microseconds a launch, medians over `reps` with the L2 cache flushed before each: 1.0.8's kernel, then the two.
fn timing(r: *Rig, c: Case, reps: usize) !void {
    const in = try inputs(r, c);
    const y = try r.dev(std.mem.sliceAsBytes(in.y));
    const up = try r.dev(std.mem.sliceAsBytes(in.up));
    const down = try r.dev(std.mem.sliceAsBytes(in.down));
    const count: u32 = @intCast(c.used);
    const p: kern.Pocket = .{ .up = up, .down = down, .n = width, .used = c.used, .count = try r.dev(std.mem.asBytes(&count)) };
    const act = try r.dev(try r.a.alloc(u8, c.rows * width * 4));
    const out = try r.dev(in.out);
    const flush = try r.dev(try r.a.alloc(u8, 128 << 20));
    var ev: [2]cuda.Event = .{ try cuda.Event.init(r.gpu.d, true), try cuda.Event.init(r.gpu.d, true) };
    defer for (&ev) |*e| e.deinit();
    const us = try r.a.alloc(f64, reps);
    var med: [2]f64 = undefined;
    for (0..2) |which| {
        for (0..reps + 2) |i| {
            try r.o.fill32(flush, @intCast(i), (128 << 20) / 4);
            try ev[0].record(r.o.s);
            if (which == 0) try r.reference(c, y, up, down, out) else try r.o.pocketRows(y, c.d, p, act, out, c.out_f32, c.rows, row_mul, row_add);
            try ev[1].record(r.o.s);
            try r.o.s.synchronize();
            if (i >= 2) us[i - 2] = @as(f64, try ev[0].elapsedMs(ev[1])) * 1000;
        }
        med[which] = check.median(us);
    }
    std.debug.print("RESULT pocket time d={d} rows={d} used={d} out_f32={}: tf_pocket_rows {d:.1} us, tf_pocket_in + tf_pocket_out {d:.1} us\n", .{ c.d, c.rows, c.used, c.out_f32, med[0], med[1] });
}

/// `tf-cuda-test pocket [reps]`: every case byte for byte, the controls, then the launches timed if `reps` > 0.
pub fn run(gpu: Gpu, reps: usize) !void {
    if (!cuda.kernels.available) {
        std.debug.print("SKIP pocket: built without the CUDA kernels (no -Dnvcc or -Dfatbins)\n", .{});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var k = try kern.Kernels.load(gpu.gpa, gpu.io, gpu.ctx, null);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x9e37_0c4e);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .o = .{ .k = &k, .s = s }, .old = try k.mods[7].function("tf_pocket_rows"), .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var cases: usize = 0;
    // 2728 leaves every lane's stride and the last output block a remainder; 70 and 300 rows take the tiled forms
    for ([_]usize{ 2688, 2728 }) |d| for ([_]usize{ 1, 2, 5, 16, 17, 40, 70, 300 }) |rows| for ([_]usize{ 0, 1, 6, 10, 44, 64 }) |used| for ([_]bool{ true, false }) |f| {
        try compare(&r, .{ .d = d, .rows = rows, .used = used, .out_f32 = f });
        cases += 1;
        _ = arena.reset(.retain_capacity);
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    };
    for ([_]usize{ 44, 64 }) |used| {
        try compare(&r, .{ .d = 2688, .rows = 2048, .used = used, .out_f32 = false }); // a whole prompt chunk
        cases += 1;
    }
    check.pass("pocket: {d} cases byte-equal to tf_pocket_rows (rows 1 to 2048, 0 to 64 neurons in use, fp32 and bf16 out)", .{cases});
    var moved: [4]usize = undefined;
    for ([_]usize{ 16, 300 }, 0..) |rows, i| {
        moved[2 * i] = try control(&r, .{ .d = 2688, .rows = rows, .used = 44, .out_f32 = true }, .neurons_reversed);
        moved[2 * i + 1] = try control(&r, .{ .d = 2728, .rows = rows, .used = 10, .out_f32 = true }, .inputs_rotated);
    }
    check.pass("pocket controls: neurons summed in reverse change {d} and {d} values, inputs rotated by one {d} and {d}", .{ moved[0], moved[2], moved[1], moved[3] });
    // the one difference from 1.0.8: a layer with no neuron in use keeps an exact -0.0 that tf_pocket_rows made +0.0
    var turned: usize = 0;
    for ([_]usize{ 1, 16, 300 }) |rows| for ([_]bool{ true, false }) |f| {
        turned += try negativeZero(&r, 2688, rows, f);
    };
    check.pass("pocket -0.0: tf_pocket_rows over no neuron in use turned {d} exact -0.0 to +0.0 and kept every other value; the launcher launches nothing and keeps all of them; a captured launch reading a count of 0 gives tf_pocket_rows' bytes", .{turned});
    if (reps == 0) return;
    for ([_]Case{
        .{ .d = 2688, .rows = 1, .used = 44, .out_f32 = true },
        .{ .d = 2688, .rows = 1, .used = 64, .out_f32 = true },
        .{ .d = 2688, .rows = 4, .used = 44, .out_f32 = true },
        .{ .d = 2688, .rows = 16, .used = 44, .out_f32 = true },
        .{ .d = 2688, .rows = 36, .used = 44, .out_f32 = false },
        .{ .d = 2688, .rows = 128, .used = 44, .out_f32 = false },
        .{ .d = 2688, .rows = 512, .used = 44, .out_f32 = false },
        .{ .d = 2688, .rows = 2048, .used = 44, .out_f32 = false },
        .{ .d = 2688, .rows = 2048, .used = 64, .out_f32 = false },
    }) |c| try timing(&r, c, reps);
}
