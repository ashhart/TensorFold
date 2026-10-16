//! `fn-pocket`: Flash Next's two pocket kernels (fn_back's pocketRows) against tf_fn_pocket_rows, byte for byte.

const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext");
const check = @import("check.zig");
const Gpu = check.Gpu;
const fb = flashnext.fn_back;
const Ops = fb.Ops;

const top_k = 10; // a row's routed experts; the shared expert's gate is its last slot
const slots = top_k + 1;

/// Rows of `d` inputs, `used` neurons, fp32 or bf16 out; the shared gate (`wts`) or 1, row scales from `from` (`rs`).
const Case = struct {
    d: usize,
    rows: usize,
    used: usize,
    out_f32: bool,
    wts: bool = true,
    rs: bool = false,
    from: usize = 0,
};

/// The kernels' copy of the inputs: as given, or a control that sums the same terms in another order.
const Order = enum { same, neurons_reversed, inputs_rotated };

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator,
    o: Ops,
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

    fn reset(r: *Rig, arena: *std.heap.ArenaAllocator) void {
        _ = arena.reset(.retain_capacity);
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    }
};

/// One case's inputs on the device (host copies kept for the controls' reordered layouts).
const Inputs = struct { y: []u16, gate: []f32, up: []f32, down: []f32, wts: []f32, rs: []f32, out: []u8 };

fn bf16(x: f32) u16 {
    const v: u32 = @bitCast(x);
    return @truncate((v + 0x7fff + ((v >> 16) & 1)) >> 16);
}

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

fn inputs(r: *Rig, c: Case) !Inputs {
    const y = try r.a.alloc(u16, c.rows * c.d);
    for (y) |*v| v.* = bf16((r.rng.float(f32) * 2 - 1) * 2);
    const scale = 3 / @sqrt(@as(f32, @floatFromInt(c.d)));
    const gate = try r.uniform(c.used * c.d, scale);
    const up = try r.uniform(c.used * c.d, scale);
    // the first neuron reads the first row's own direction, so at least that row moves; the rest fire at random
    if (c.used > 0) for (gate[0..c.d], up[0..c.d], y[0..c.d]) |*g, *u, v| {
        g.* = bf(v) / @as(f32, @floatFromInt(c.d));
        u.* = g.*;
    };
    const down = try r.uniform(c.used * c.d, 0.05);
    const wts = try alloc01(r, c.rows * slots);
    const rs = try alloc01(r, c.rows);
    const n = c.rows * c.d;
    const start = try r.uniform(n, 1);
    const out = try r.a.alloc(u8, n * @as(usize, if (c.out_f32) 4 else 2));
    if (c.out_f32) {
        @memcpy(out, std.mem.sliceAsBytes(start));
    } else for (std.mem.bytesAsSlice(u16, out), start) |*h, v| {
        h.* = bf16(v);
    }
    return .{ .y = y, .gate = gate, .up = up, .down = down, .wts = wts, .rs = rs, .out = out };
}

fn alloc01(r: *Rig, n: usize) ![]f32 {
    const out = try r.a.alloc(f32, n);
    for (out) |*v| v.* = 0.25 + r.rng.float(f32);
    return out;
}

/// The device pointers of one layout of a case's inputs.
const Dev = struct { y: u64, gate: u64, up: u64, down: u64, wts: u64, rs: u64 };

fn upload(r: *Rig, c: Case, in: Inputs) !Dev {
    return .{
        .y = try r.dev(std.mem.sliceAsBytes(in.y)),
        .gate = try r.dev(std.mem.sliceAsBytes(in.gate)),
        .up = try r.dev(std.mem.sliceAsBytes(in.up)),
        .down = try r.dev(std.mem.sliceAsBytes(in.down)),
        .wts = if (c.wts) try r.dev(std.mem.sliceAsBytes(in.wts)) else 0,
        .rs = if (c.rs) try r.dev(std.mem.sliceAsBytes(in.rs)) else 0,
    };
}

/// tf_fn_pocket_rows over the case's neurons, as pocketRows launched it before.
fn reference(r: *Rig, c: Case, v: Dev, out: u64) !void {
    var a: cuda.Args = .{};
    a.add(v.y);
    a.add(@as(c_int, @intCast(c.d)));
    for ([_]u64{ v.gate, v.up, v.down }) |p| a.add(p);
    a.add(@as(c_int, @intCast(c.used)));
    a.add(v.wts);
    a.add(@as(c_int, slots));
    a.add(@as(c_int, top_k));
    a.add(out);
    a.add(@as(c_int, @intFromBool(c.out_f32)));
    a.add(v.rs);
    a.add(@as(c_int, @intCast(@min(c.from, c.used))));
    try cuda.launch.launch(r.old, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 256 } }, r.o.s, &a);
}

/// The two kernels through pocketRows.
fn split(r: *Rig, c: Case, v: Dev, act: u64, out: u64) !void {
    const pk: Ops.Pocket = .{
        .gate = v.gate,
        .up = v.up,
        .down = v.down,
        .n = c.used,
        .scale = v.rs,
        .from = c.from,
        .act = act,
    };
    try r.o.pocketRows(pk, .{ .x = v.y, .ids = 0, .wts = v.wts }, out, c.out_f32, c.rows, c.d, top_k);
}

/// The reference's bytes and the split kernels' on the inputs laid out by `order`.
fn both(r: *Rig, c: Case, in: Inputs, order: Order) ![2][]u8 {
    const ref_out = try r.dev(in.out);
    try r.gpu.ctx.synchronize();
    try reference(r, c, try upload(r, c, in), ref_out);
    const want = try r.get(ref_out, in.out.len);
    var re = in;
    re.y = try r.a.dupe(u16, in.y);
    re.gate = try r.a.dupe(f32, in.gate);
    re.up = try r.a.dupe(f32, in.up);
    re.down = try r.a.dupe(f32, in.down);
    switch (order) {
        .same => {},
        .neurons_reversed => for (0..c.used / 2) |j| for ([_][]f32{ re.gate, re.up, re.down }) |m| for (0..c.d) |i| {
            std.mem.swap(f32, &m[j * c.d + i], &m[(c.used - 1 - j) * c.d + i]);
        },
        .inputs_rotated => {
            for (0..c.rows) |t| std.mem.rotate(u16, re.y[t * c.d ..][0..c.d], 1);
            for (0..c.used) |j| for ([_][]f32{ re.gate, re.up }) |m| std.mem.rotate(f32, m[j * c.d ..][0..c.d], 1);
        },
    }
    const act = try r.dev(try r.a.alloc(u8, c.rows * fb.max_pocket * 4));
    const out = try r.dev(in.out);
    try r.gpu.ctx.synchronize();
    try split(r, c, try upload(r, c, re), act, out);
    return .{ want, try r.get(out, in.out.len) };
}

/// Values of `word` bytes that differ between a and b.
fn differing(a: []const u8, b: []const u8, word: usize) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < a.len) : (i += word) n += @intFromBool(!std.mem.eql(u8, a[i..][0..word], b[i..][0..word]));
    return n;
}

/// One case: the reference must have moved the output, and the two kernels must give its bytes.
fn compare(r: *Rig, c: Case) !void {
    const in = try inputs(r, c);
    const word: usize = if (c.out_f32) 4 else 2;
    const got = try both(r, c, in, .same);
    const moved = differing(got[0], in.out, word);
    const what = "fn-pocket d={d} rows={d} used={d}: the reference moved {d} of {d} values";
    try check.expect(moved * 2 > c.d, what, .{ c.d, c.rows, c.used, moved, c.rows * c.d });
    const differ = differing(got[1], got[0], word);
    if (differ > 0) std.debug.print("DIFFER fn-pocket {any}: {d} of {d} values\n", .{ c, differ, got[0].len / word });
    try check.sameBytes("fn-pocket", got[1], got[0]);
}

/// Controls: the same sums in another order must give other bytes, or the comparison could not see an order.
fn control(r: *Rig, c: Case, order: Order) !usize {
    const in = try inputs(r, c);
    const got = try both(r, c, in, order);
    const differ = differing(got[1], got[0], if (c.out_f32) 4 else 2);
    const what = "fn-pocket control {t} rows={d} used={d}: equal bytes, so the check cannot see the order";
    try check.expect(differ > 0, what, .{ order, c.rows, c.used });
    return differ;
}

/// No neuron in use: pocketRows launches nothing, so the output keeps every bit, an exact -0.0 too.
fn empty(r: *Rig, rows: usize, out_f32: bool) !void {
    const c: Case = .{ .d = 2560, .rows = rows, .used = 0, .out_f32 = out_f32 };
    const in = try inputs(r, c);
    const word: usize = if (out_f32) 4 else 2;
    var i: usize = 0;
    while (i < in.out.len) : (i += 3 * word) @memset(in.out[i..][0..word], 0);
    i = 0;
    while (i < in.out.len) : (i += 3 * word) in.out[i + word - 1] = 0x80;
    const out = try r.dev(in.out);
    try r.gpu.ctx.synchronize();
    try split(r, c, try upload(r, c, in), try r.dev(try r.a.alloc(u8, rows * fb.max_pocket * 4)), out);
    try check.sameBytes("fn-pocket, no neuron in use: the output as it was", try r.get(out, in.out.len), in.out);
}

/// GPU microseconds a launch, medians over `reps`, the L2 cache flushed before each: tf_fn_pocket_rows, then the two.
fn timing(r: *Rig, c: Case, reps: usize) !void {
    const in = try inputs(r, c);
    const v = try upload(r, c, in);
    const act = try r.dev(try r.a.alloc(u8, c.rows * fb.max_pocket * 4));
    const out = try r.dev(in.out);
    var flush = try cuda.DeviceBuffer.alloc(r.gpu.d, 128 << 20);
    defer flush.free();
    var ev: [2]cuda.Event = .{ try cuda.Event.init(r.gpu.d, true), try cuda.Event.init(r.gpu.d, true) };
    defer for (&ev) |*e| e.deinit();
    const us = try r.a.alloc(f64, reps);
    var med: [2]f64 = undefined;
    for (0..2) |which| {
        for (0..reps + 2) |i| {
            try flush.fill32(@intCast(i), r.o.s.handle);
            try ev[0].record(r.o.s);
            if (which == 0) try reference(r, c, v, out) else try split(r, c, v, act, out);
            try ev[1].record(r.o.s);
            try r.o.s.synchronize();
            if (i >= 2) us[i - 2] = @as(f64, try ev[0].elapsedMs(ev[1])) * 1000;
        }
        med[which] = check.median(us);
    }
    const line = "RESULT fn-pocket time rows={d} used={d}: tf_fn_pocket_rows {d:.1} us, the two kernels {d:.1} us\n";
    std.debug.print(line, .{ c.rows, c.used, med[0], med[1] });
}

/// `tf-cuda-test fn-pocket [reps]`: every case byte for byte, the controls, then the launches timed if `reps` > 0.
pub fn run(gpu: Gpu, reps: usize) !void {
    if (!cuda.kernels.available) {
        std.debug.print("SKIP fn-pocket: built without the CUDA kernels (no -Dnvcc or -Dfatbins)\n", .{});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var k = try fb.Kernels.load(gpu.d);
    defer k.deinit();
    var mix = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_train);
    defer mix.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5717_9a11);
    const old = try mix.function("tf_fn_pocket_rows");
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .o = .{ .k = &k, .s = s }, .old = old, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var cases: usize = 0;
    // 2600 leaves lane strides and output blocks a remainder; rows and neurons cross fn_back.pocketTiled's thresholds
    const row_list = [_]usize{ 1, 2, 3, 5, 8, 16, 17, 40, 63, 64, 70, 127, 128, 255, 256, 300, 683 };
    const used_list = [_]usize{ 1, 6, 13, 33, 64, 128 };
    for ([_]usize{ 2560, 2600 }) |d| for (row_list) |rows| for (used_list) |used| for ([_]bool{ true, false }) |f| {
        try compare(&r, .{ .d = d, .rows = rows, .used = used, .out_f32 = f });
        cases += 1;
        r.reset(&arena);
    };
    // no shared gate (1), and a training pass's row scales from a neuron on (and past the last: none scaled)
    for ([_]usize{ 1, 5, 70, 300 }) |rows| for ([_][2]usize{ .{ 13, 0 }, .{ 13, 7 }, .{ 64, 63 }, .{ 10, 99 } }) |uf| {
        try compare(&r, .{ .d = 2560, .rows = rows, .used = uf[0], .out_f32 = true, .wts = false });
        try compare(&r, .{ .d = 2560, .rows = rows, .used = uf[0], .out_f32 = true, .rs = true, .from = uf[1] });
        cases += 2;
        r.reset(&arena);
    };
    // a whole prompt chunk (batch.default_chunk) and a larger pass
    for ([_]usize{ 1536, 2048 }) |rows| for ([_]usize{ 13, 64 }) |used| {
        try compare(&r, .{ .d = 2560, .rows = rows, .used = used, .out_f32 = true });
        cases += 1;
        r.reset(&arena);
    };
    check.pass("fn-pocket: {d} cases byte-equal to tf_fn_pocket_rows (1 to 2048 rows, 1 to 128 neurons)", .{cases});
    var moved: [4]usize = undefined;
    for ([_]usize{ 16, 300 }, 0..) |rows, i| {
        moved[2 * i] = try control(&r, .{ .d = 2560, .rows = rows, .used = 13, .out_f32 = true }, .neurons_reversed);
        moved[2 * i + 1] = try control(&r, .{ .d = 2600, .rows = rows, .used = 10, .out_f32 = true }, .inputs_rotated);
        r.reset(&arena);
    }
    const said = "fn-pocket controls: neurons in reverse change {d} and {d} values, inputs rotated by one {d} and {d}";
    check.pass(said, .{ moved[0], moved[2], moved[1], moved[3] });
    for ([_]usize{ 1, 16, 300 }) |rows| for ([_]bool{ true, false }) |f| {
        try empty(&r, rows, f);
        r.reset(&arena);
    };
    check.pass("fn-pocket: no neuron in use launches nothing and keeps every output bit, -0.0 included", .{});
    if (reps == 0) return;
    for ([_]usize{ 1, 2, 4, 8, 16, 32, 63, 64, 128, 255, 256, 512, 1536 }) |rows| for ([_]usize{ 4, 13, 64 }) |used| {
        try timing(&r, .{ .d = 2560, .rows = rows, .used = used, .out_f32 = true }, reps);
        r.reset(&arena);
    };
}
