//! DeepSeek V4.1's EXL3 routed experts end to end on Metal against the serial chain, tf_exl3_matmul_row and hoisted.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const rig_ = @import("rig.zig");
const eref = @import("dsv41_experts_ref.zig");
const bank = @import("dsv41_x3_bank.zig");

const x3 = bank.x3;
const ref = bank.ref;
const xl = tf.exl3_projection;
const fmt = xl.format;
const mx = tf.dsv41_metal.experts;
const topk = bank.topk;
const count_experts = 16;

/// Tokens, hidden and expert width, and the experts' widths: V4.1's sizes at one row, then 2 to 16 rows.
const Case = struct { rows: u32, hidden: u32, width: u32, mix: bank.Mix };
const cases = [_]Case{
    .{ .rows = 1, .hidden = 5120, .width = 2304, .mix = .mul1 },
    .{ .rows = 4, .hidden = 1024, .width = 256, .mix = .mul1 },
    .{ .rows = 2, .hidden = 1024, .width = 256, .mix = .mul1 },
    .{ .rows = 3, .hidden = 1024, .width = 256, .mix = .mul1 },
    .{ .rows = 8, .hidden = 1024, .width = 256, .mix = .mul1 },
    .{ .rows = 16, .hidden = 1024, .width = 256, .mix = .mul1 },
    .{ .rows = 16, .hidden = 1024, .width = 256, .mix = .books },
};

fn words(comptime T: type, rig: *rig_.Rig, r: rig_.Ref, n: usize) []const T {
    return @alignCast(std.mem.bytesAsSlice(T, rig.bytes(r, n * @sizeOf(T))));
}

fn say(ok: bool, cs: Case, comptime f: []const u8, args: anytype) usize {
    std.debug.print("{s} dsv41 x3 exl3 experts lane: {d} rows, {d}x{d}, {t}: ", .{ if (ok) "PASS" else "FAIL", cs.rows, cs.hidden, cs.width, cs.mix });
    std.debug.print(f ++ "\n", args);
    return @intFromBool(!ok);
}

/// One case: the GPU lane's sums, act, outputs and combined rows against the serial chain and the existing kernels.
fn one(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, p: *const x3.Pipelines, m: *const mx.Pipelines, row: *const xl.Pipelines, silu: *const ref.Silu, cs: Case, seed: u64) !usize {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var rig: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();
    var backing: std.ArrayList(mtl.Buffer) = .empty;
    defer backing.deinit(a);
    const plan = try eref.Plan.routed(a, cs.rows, count_experts, rng, .random);
    defer plan.deinit();
    var b = try bank.Bank.init(a, count_experts, cs.hidden, cs.width, cs.mix, &.{ .gate, .up, .down }, .random, rng);
    defer b.deinit();
    const c: x3.Call = .{ .rows = cs.rows, .k = cs.hidden, .n = cs.width, .experts = count_experts };
    const np: usize = cs.rows * topk;
    const x = try a.alloc(u16, cs.rows * cs.hidden);
    defer a.free(x);
    for (x) |*v| v.* = ref.bf16(rng.float(f32) * 2 - 1);

    const u = try b.upload(&rig, &backing);
    try x3.check(c, u.tables);
    const xr = try rig.input(std.mem.sliceAsBytes(x));
    var r = try bank.route(&rig, plan);
    for ([_]*x3.Ref{ &r.slots, &r.nslots, &r.pairs, &r.where }, [_]usize{ 12 * np, 4, 4 * np, 4 * np }) |f, n| f.* = bank.at(try rig.scratch(n));
    const ones: [96]f32 = @splat(1);
    const one_w = try rig.input(std.mem.sliceAsBytes(ones[0..np]));
    const su = try rig.scratch(x3.upLayout(c).end);
    const sd = try rig.scratch(x3.downLayout(c).end);
    const act = try rig.output(np * cs.width * 2);
    const out = try rig.output(np * cs.hidden * 2);
    const y = try rig.output(cs.rows * cs.hidden * 4);
    // tf_exl3_matmul_row on every pair and matrix, and main's routed hoisted kernel on all pairs of each matrix
    const sb1 = try xl.scratchBytes(@max(cs.hidden, cs.width), @max(cs.hidden, cs.width), 1);
    const solo = try rig.scratch(3 * np * sb1);
    const solo_y = try rig.scratch(3 * np * @max(cs.hidden, cs.width) * 2);
    const sbm = try xl.scratchBytes(@max(cs.hidden, cs.width), @max(cs.hidden, cs.width), np);
    const routed = try rig.scratch(3 * sbm);
    const routed_y = try rig.scratch(3 * np * @max(cs.hidden, cs.width) * 2);

    const pass = rig.begin();
    mx.use(pass.e, backing.items);
    mx.plan(m, pass.e, r, c);
    pass.e.barrier();
    x3.up(p, pass.e, bank.at(xr), u.tables, r, bank.at(act), bank.at(su), c, bank.limit);
    x3.down(p, pass.e, bank.at(act), u.tables, r, bank.at(out), bank.at(sd), c);
    var rw = r;
    rw.weights = bank.at(one_w);
    mx.combine(m, pass.e, bank.at(out), rw, rw.weights, bank.at(y), true, c);
    pass.e.barrier();
    for (0..np) |q| for ([_]bank.Kind{ .gate, .up, .down }, 0..) |kind, v| {
        const mat = b.view(u, kind, plan.ids[q]);
        const in: xl.View = if (kind == .down) .{ .buffer = act.buf, .offset = act.off + q * cs.width * 2 } else .{ .buffer = xr.buf, .offset = xr.off + q / topk * cs.hidden * 2 };
        const at_ = (v * np + q);
        try xl.encode(row, pass.e, mat, in, .{ .buffer = solo_y.buf, .offset = solo_y.off + at_ * mat.n * 2 }, .{ .buffer = solo.buf, .offset = solo.off + at_ * sb1 }, .{ .count = 1, .x_stride = mat.k, .y_stride = mat.n });
    };
    for ([_]bank.Kind{ .gate, .up, .down }, [_]x3.Ref{ u.tables.gate, u.tables.up, u.tables.down }, 0..) |kind, t, v| {
        const d = b.dims(kind);
        const in: rig_.Ref = if (kind == .down) act else xr;
        try xl.encodeIndexed(row, pass.e, .{ .descriptors = .{ .buffer = t.buf, .offset = t.off }, .count = count_experts, .k = d[0], .n = d[1] }, .{ .buffer = in.buf, .offset = in.off }, .{ .buffer = routed_y.buf, .offset = routed_y.off + v * np * d[1] * 2 }, .{ .buffer = routed.buf, .offset = routed.off + v * sbm }, .{ .count = np, .x_stride = d[0], .y_stride = d[1], .routes = .{ .ids = .{ .buffer = r.ids.buf, .offset = r.ids.off }, .repeat = if (kind == .down) 1 else topk } });
    }
    try rig.finish(pass);

    // the serial chain
    const want_su = try a.alloc(f32, 2 * np * cs.width);
    defer a.free(want_su);
    const want_act = try a.alloc(u16, np * cs.width);
    defer a.free(want_act);
    const want_sd = try a.alloc(f32, np * cs.hidden);
    defer a.free(want_sd);
    const want_out = try a.alloc(u16, np * cs.hidden);
    defer a.free(want_out);
    const want_y = try a.alloc(f32, cs.rows * cs.hidden);
    defer a.free(want_y);
    try bank.upAll(a, &b, plan, x, silu, want_su, want_act);
    try bank.downAll(a, &b, plan, want_act, want_sd, want_out);
    try bank.combineAll(a, plan, want_out, cs.hidden, count_experts / 8, want_y);

    const lu = x3.upLayout(c);
    const ld = x3.downLayout(c);
    const got_su = words(u32, &rig, .{ .buf = su.buf, .off = su.off + lu.sums }, 2 * np * cs.width);
    const got_sd = words(u32, &rig, .{ .buf = sd.buf, .off = sd.off + ld.sums }, np * cs.hidden);
    const got_out = words(u16, &rig, out, np * cs.hidden);
    const where = words(u32, &rig, .{ .buf = r.where.buf, .off = r.where.off }, np);
    var bad: usize = 0;
    if (bank.differ(u32, got_su, @ptrCast(want_su))) |i| bad += say(false, cs, "gate/up sum {d} differs from the serial chain", .{i});
    if (bank.differ(u16, words(u16, &rig, act, np * cs.width), want_act)) |i| bad += say(false, cs, "act {d} differs from the serial chain", .{i});
    if (bank.differ(u32, got_sd, @ptrCast(want_sd))) |i| bad += say(false, cs, "down sum {d} differs from the serial chain", .{i});
    for (0..np) |q| {
        const g = got_out[where[q] * cs.hidden ..][0..cs.hidden];
        if (bank.differ(u16, g, want_out[plan.where[q] * cs.hidden ..][0..cs.hidden])) |i| {
            bad += say(false, cs, "pair {d}'s output {d} differs from the serial chain", .{ q, i });
            break;
        }
    }
    if (bank.differ(u32, words(u32, &rig, y, cs.rows * cs.hidden), @ptrCast(want_y))) |i| bad += say(false, cs, "combined {d} differs from the serial chain", .{i});

    // row invariance: each pair's sums are P1's one-row kernel's and main's routed kernel's, and so are its outputs
    pairs: for (0..np) |q| for ([_]bank.Kind{ .gate, .up, .down }, 0..) |kind, v| {
        const d = b.dims(kind);
        const at_ = v * np + q;
        const lone = solo.buf.contents()[solo.off + at_ * sb1 ..];
        const lone_sums: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, lone[(d[0] * 2 + 15) & ~@as(usize, 15) ..][0 .. d[1] * 4]));
        const many = routed.buf.contents()[routed.off + v * sbm ..];
        const many_sums: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, many[(np * d[0] * 2 + 15) & ~@as(usize, 15) ..][q * d[1] * 4 ..][0 .. d[1] * 4]));
        const mine = if (kind == .down) got_sd[q * d[1] ..][0..d[1]] else got_su[at_ * d[1] ..][0..d[1]];
        if (bank.differ(u32, mine, lone_sums)) |i| {
            bad += say(false, cs, "pair {d} {t} sum {d} differs from tf_exl3_matmul_row's", .{ q, kind, i });
            break :pairs;
        }
        if (bank.differ(u32, mine, many_sums)) |i| {
            bad += say(false, cs, "pair {d} {t} sum {d} differs from routed tf_exl3_matmul_hoisted's", .{ q, kind, i });
            break :pairs;
        }
        if (kind != .down) continue;
        const lone_y = words(u16, &rig, .{ .buf = solo_y.buf, .off = solo_y.off + at_ * d[1] * 2 }, d[1]);
        const many_y = words(u16, &rig, .{ .buf = routed_y.buf, .off = routed_y.off + at_ * d[1] * 2 }, d[1]);
        const g = got_out[where[q] * d[1] ..][0..d[1]];
        if (bank.differ(u16, g, lone_y) != null or bank.differ(u16, g, many_y) != null) {
            bad += say(false, cs, "pair {d}'s output differs from the one-row or routed path's", .{q});
            break :pairs;
        }
    };
    if (bad == 0) _ = say(true, cs, "{d} pairs' sums, act, outputs and rows bit-equal to the serial chain, tf_exl3_matmul_row and routed tf_exl3_matmul_hoisted", .{np});
    if (cs.rows == 1 and cs.hidden == 5120) try float64(a, &b, plan, x, words(u32, &rig, y, cs.hidden));
    return bad;
}

/// float64 of W_q's dots over `rot`, rotated back and scaled by sv, for the 128-blocks listed; `wq` holds 2 k n.
fn project(m: ref.Matrix, k: usize, n: usize, rot: []const f64, blocks: []const usize, wq: []f16, out: []f64) void {
    bank.columns(m, k, n, wq[0 .. k * n], wq[k * n ..]);
    for (blocks) |blk| {
        const y = out[blk * 128 ..][0..128];
        for (y, blk * 128..) |*s, j| {
            s.* = 0;
            for (rot, wq[k * n + j * k ..][0..k]) |v, w| s.* += v * @as(f64, w);
        }
        fmt.rotate(y, 1, 128, 1);
        for (y, m.sv[blk * 128 ..][0..128]) |*s, sc| s.* *= sc;
    }
}

/// Reported, not a gate: row 0's combined output in float64 from the fp16 rotated row on (the format's one rounding).
fn float64(a: std.mem.Allocator, b: *const bank.Bank, plan: eref.Plan, x: []const u16, y: []const u32) !void {
    const cols = [_]usize{ 0, 1, 127, 128, 2047, 2560, 4095, 5119 };
    const h = b.hidden;
    const n = b.width;
    const wq = try a.alloc(f16, 2 * h * n);
    defer a.free(wq);
    const buf = try a.alloc(f64, 3 * n + 2 * h);
    defer a.free(buf);
    const gate = buf[0..n];
    const up = buf[n .. 2 * n];
    const act = buf[2 * n .. 3 * n];
    const rot = buf[3 * n ..][0..h];
    const down = buf[3 * n + h ..][0..h];
    const rot16 = try a.alloc(f16, h);
    defer a.free(rot16);
    var all: [40]usize = undefined;
    for (&all, 0..) |*v, i| v.* = i;
    var want: [cols.len]f64 = @splat(0);
    var mine: [cols.len]usize = undefined;
    for (cols, &mine) |j, *blk| blk.* = j / 128;
    for (0..topk) |q| {
        const e = plan.ids[q];
        for ([_]bank.Kind{ .gate, .up }, [_][]f64{ gate, up }) |kind, out| {
            const m = b.matrix(kind, e);
            ref.rotIn(x[0..h], m.su, rot16);
            for (rot, rot16) |*r, v| r.* = v;
            project(m, h, n, rot, all[0 .. n / 128], wq, out);
        }
        for (act, gate, up) |*o, gv, uv| {
            const g = @min(gv, bank.limit);
            o.* = g / (1 + @exp(-g)) * std.math.clamp(uv, -bank.limit, bank.limit) * plan.weights[q];
        }
        const m = b.matrix(.down, e);
        for (rot[0..n], act, m.su) |*r, v, sc| r.* = v * sc;
        fmt.rotate(rot[0..n], 1, n, 1);
        project(m, n, h, rot[0..n], &mine, wq, down);
        for (cols, &want) |j, *w| w.* += down[j];
    }
    var worst: f64 = 0;
    var top: f64 = 0;
    for (cols, want) |j, v| {
        worst = @max(worst, @abs(@as(f64, @as(f32, @bitCast(y[j]))) - v));
        top = @max(top, @abs(v));
    }
    std.debug.print("dsv41 x3 exl3 experts lane, reported: row 0 at 8 columns against float64: max |gpu - f64| {e:.3}, max |y| {e:.3}\n", .{ worst, top });
}

/// Reported: the device's silu at every finite bf16 gate against float64 rounded to fp32, and its worst relative error.
fn siluReport(t: *const ref.Silu) void {
    var worst: f64 = 0;
    var off: usize = 0;
    var finite: usize = 0;
    for (t, 0..) |v, i| {
        const g: f64 = ref.fromBf16(@intCast(i));
        if (!std.math.isFinite(g)) continue;
        finite += 1;
        const want = g / (1 + @exp(-g));
        off += @intFromBool(@as(f32, @floatCast(want)) != v);
        if (@abs(want) >= 0x1p-64) worst = @max(worst, @abs(v - want) / @abs(want));
    }
    std.debug.print("dsv41 x3 exl3 experts lane, reported: tf_dsv41_x3_silu differs from float64 rounded to fp32 at {d} of {d} finite gates; worst relative error {e:.3} where |silu| >= 2^-64\n", .{ off, finite, worst });
}

pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, seed: u64) !usize {
    var p = try x3.Pipelines.load(device);
    defer p.deinit();
    var m = try mx.Pipelines.load(a, device);
    defer m.deinit();
    var row = try xl.Pipelines.load(device);
    defer row.deinit();
    const t = try bank.silu(a, device, queue);
    defer a.destroy(t);
    siluReport(t);
    var bad: usize = 0;
    for (cases, 0..) |cs, i| bad += try one(a, device, queue, &p, &m, &row, t, cs, seed + i);
    return bad;
}
