// SPDX-License-Identifier: Apache-2.0

//! EXL3 Level Zero host: one case file in, fp32 outputs out. Python checks the reference.

//! Usage: tf-xpu-exl3-test case.bin out.bin [reps]. Reps print the median call time.

//! Case header: magic EXL3, mode, count, k, n, halves, rows, topk, x_per_slot, formats.

//! Then trellis, suh, svh, ids when routed, then bf16 x. Fused MLP writes y [slots][k].

const std = @import("std");
const xpu = @import("xpu");
const exl3 = xpu.exl3;

const Header = extern struct { magic: u32, mode: u32, count: u32, k: u32, n: u32, halves: u32, rows: u32, topk: u32, x_per_slot: u32, formats: u32 };

const Arena = struct {
    size: usize = 0,
    fn add(a: *Arena, n: usize) usize {
        const at = std.mem.alignForward(usize, a.size, 256);
        a.size = at + n;
        return at;
    }
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.Usage;
    const case = try std.Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .unlimited);
    defer gpa.free(case);
    const h = std.mem.bytesToValue(Header, case[0..@sizeOf(Header)]);
    if (h.magic != 0x334C5845) return error.BadCase;
    const routed = h.mode == 1;
    const fused = h.mode == 2;
    const tile_bytes: usize = 16 * h.halves;
    const tr_bytes = (h.k / 16) * (h.n / 16) * tile_bytes;
    const shared = fused and h.x_per_slot != 0;
    const slots: usize = if (routed or fused) h.rows * h.topk + (if (shared) h.rows else 0) else h.rows;
    const x_rows: usize = if (routed) (if (h.x_per_slot != 0) slots else h.rows) else if (fused) h.rows else h.count * h.rows;
    const y_rows: usize = if (routed or fused) slots else h.count * h.rows;
    const y_n: usize = if (fused) h.k else h.n;
    const output: exl3.Output = std.enums.fromInt(exl3.Output, h.formats & 0xff) orelse return error.BadCase;
    const input: exl3.Input = std.enums.fromInt(exl3.Input, h.formats >> 8) orelse return error.BadCase;
    const y_elem: usize = if (output == .f32) 4 else 2;
    // per group or expert: one matrix [k, n], or (fused) gate and up [k, n] and down [n, k]
    const mats: usize = if (fused) 3 else 1;
    const shapes = [3][2]u32{ .{ h.k, h.n }, .{ h.k, h.n }, .{ h.n, h.k } };

    // one device arena: weights, ids, x, y, scratch, slot table
    var ar: Arena = .{};
    var at: usize = @sizeOf(Header);
    const weights = try gpa.alloc([3]usize, h.count * mats);
    defer gpa.free(weights);
    const src = try gpa.alloc([3]usize, h.count * mats);
    defer gpa.free(src);
    for (weights, src, 0..) |*w, *s, i| {
        const kn = shapes[i % mats];
        for (matBytes(kn, tr_bytes), 0..) |n, j| {
            w[j] = ar.add(n);
            s[j] = at;
            at += n;
        }
    }
    const ids_n: usize = if (routed or fused) h.rows * h.topk * 4 else 0;
    const ids_off = ar.add(ids_n);
    const ids_src = at;
    at += ids_n;
    const x_off = ar.add(x_rows * h.k * 2);
    const x_src = at;
    at += x_rows * h.k * 2;
    if (at != case.len) return error.BadCase;
    const y_off = ar.add(y_rows * y_n * y_elem);
    const scratch_off = ar.add(if (fused) exl3.moeScratchBytes(h.k, h.n, slots) else if (routed) exl3.scratchBytes(h.k, h.n, slots) else h.count * std.mem.alignForward(usize, exl3.scratchBytes(h.k, h.n, h.rows), 256));
    const slots_off = ar.add(exl3.slotTableBytes(slots));

    var drv = try xpu.Driver.open();
    defer drv.close();
    var ctx = try xpu.Context.init(&drv, null);
    defer ctx.deinit();
    var s = try xpu.Stream.init(&ctx);
    defer s.deinit();
    std.debug.print("{s}\n", .{ctx.name()});
    var p = try exl3.Pipelines.load(&ctx);
    defer p.deinit();
    var buf = try xpu.DeviceBuffer.alloc(&ctx, ar.size);
    defer buf.free();
    for (weights, src, 0..) |w, sr, i| for (w, sr, matBytes(shapes[i % mats], tr_bytes)) |d, o, n| try buf.uploadAsync(s, d, case[o..][0..n]);
    try buf.uploadAsync(s, ids_off, case[ids_src..][0..ids_n]);
    try buf.uploadAsync(s, x_off, case[x_src..][0 .. x_rows * h.k * 2]);
    try s.synchronize();
    const base: u64 = @intFromPtr(buf.ptr.?);

    const ms = try gpa.alloc(exl3.Matrix, h.count * mats);
    defer gpa.free(ms);
    for (ms, weights, 0..) |*m, w, i| {
        const kn = shapes[i % mats];
        m.* = .{ .trellis = base + w[0], .suh = base + w[1], .svh = base + w[2], .k = kn[0], .n = kn[1], .halves = h.halves };
    }

    const reps: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 0;
    const weight_bytes: f64 = @floatFromInt((tr_bytes + h.k * 2 + h.n * 2) * mats * (if (routed or fused) slots else h.count));
    if (fused) {
        // tables of gate, up and down: every mats-th matrix
        var plans: [3]exl3.Plan = undefined;
        var made: usize = 0;
        defer for (plans[0..made]) |*pl| pl.deinit();
        for (&plans, 0..) |*pl, j| {
            const one = try gpa.alloc(exl3.Matrix, h.count);
            defer gpa.free(one);
            for (one, 0..) |*m, i| m.* = ms[i * mats + j];
            pl.* = try exl3.Plan.experts(&ctx, s, gpa, one);
            made += 1;
        }
        const r: exl3.MoeRoute = .{ .ids = base + ids_off, .tokens = h.rows, .topk = h.topk, .shared = if (shared) h.count - 1 else null, .x = base + x_off, .x_stride = h.k, .y = base + y_off, .y_stride = h.k, .output = output, .input = input, .scratch = base + scratch_off };
        try exl3.encodeMoe(&p, s, &plans[0], &plans[1], &plans[2], r);
        if (reps > 0) {
            try s.synchronize();
            const t0 = std.Io.Clock.awake.now(io);
            for (0..reps) |_| try exl3.encodeMoe(&p, s, &plans[0], &plans[1], &plans[2], r);
            try s.synchronize();
            report(io, t0, reps, weight_bytes);
        }
    } else if (routed) {
        var experts = try exl3.Plan.experts(&ctx, s, gpa, ms);
        defer experts.deinit();
        const r: exl3.Route = .{ .ids = base + ids_off, .tokens = h.rows, .topk = h.topk, .x = base + x_off, .x_stride = h.k, .x_per_slot = h.x_per_slot != 0, .y = base + y_off, .y_stride = h.n, .output = output, .input = input, .scratch = base + scratch_off, .slots = base + slots_off };
        try exl3.encodeRouted(&p, s, &experts, r);
        if (reps > 0) {
            try s.synchronize();
            const t0 = std.Io.Clock.awake.now(io);
            for (0..reps) |_| try exl3.encodeRouted(&p, s, &experts, r);
            try s.synchronize();
            report(io, t0, reps, weight_bytes);
        }
    } else {
        const groups = try gpa.alloc(exl3.Group, h.count);
        defer gpa.free(groups);
        const per = std.mem.alignForward(usize, exl3.scratchBytes(h.k, h.n, h.rows), 256);
        for (groups, ms, 0..) |*g, m, i| g.* = .{ .m = m, .rows = h.rows, .x = base + x_off + i * h.rows * h.k * 2, .x_stride = h.k, .y = base + y_off + i * h.rows * h.n * y_elem, .y_stride = h.n, .output = output, .input = input, .scratch = base + scratch_off + i * per };
        var plan = try exl3.Plan.init(&ctx, s, gpa, groups);
        defer plan.deinit();
        try exl3.encode(&p, s, &plan);
        if (reps > 0) {
            try s.synchronize();
            const t0 = std.Io.Clock.awake.now(io);
            for (0..reps) |_| try exl3.encode(&p, s, &plan);
            try s.synchronize();
            report(io, t0, reps, weight_bytes);
        }
    }
    const y = try gpa.alloc(u8, y_rows * y_n * y_elem);
    defer gpa.free(y);
    try buf.download(s, y_off, y);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = y });
    return 0;
}

/// Bytes of one matrix [k, n]: trellis (the same for [n, k]), suh, svh.
fn matBytes(kn: [2]u32, tr_bytes: usize) [3]usize {
    return .{ tr_bytes, @as(usize, kn[0]) * 2, @as(usize, kn[1]) * 2 };
}

fn report(io: std.Io, t0: std.Io.Timestamp, reps: usize, weight_bytes: f64) void {
    const dt = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds())) / 1e9 / @as(f64, @floatFromInt(reps));
    std.debug.print("call {d:.1} us, weights {d:.2} MB, {d:.1} GB/s\n", .{ dt * 1e6, weight_bytes / 1e6, weight_bytes / dt / 1e9 });
}
