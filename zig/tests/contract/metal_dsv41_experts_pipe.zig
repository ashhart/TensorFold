//! DeepSeek V4.1's routed experts end to end on Metal (plan, up, down, combine at V4.1's sizes), and timing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_experts_ref.zig");

const c = contract.spec;
const ex = tf.dsv41_metal.experts;
const topk = ref.topk;
const opts = mtl.ResourceOptions.shared;
const sample_blocks = [_]usize{ 0, 23, 47, 71 };
const sample_cols = [_]usize{ 0, 1, 31, 32, 777, 2560, 4095, 5119 };

fn at(r: rig_.Ref) ex.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

fn address(r: rig_.Ref) u64 {
    return r.buf.gpuAddress() + r.off;
}

fn u32s(rig: *rig_.Rig, r: rig_.Ref, n: usize) []const u32 {
    return @alignCast(std.mem.bytesAsSlice(u32, rig.bytes(r, 4 * n)));
}

fn bf16s(a: std.mem.Allocator, rig: *rig_.Rig, r: rig_.Ref, n: usize) ![]f32 {
    const h: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, rig.bytes(r, 2 * n)));
    const out = try a.alloc(f32, n);
    for (out, h) |*y, v| y.* = c.fromBf16(v);
    return out;
}

/// sum_i x_i * dequantized weight row, in float64.
fn rawDot(codes: []const u8, sc: []const u8, x: []const f64) f64 {
    var s: f64 = 0;
    for (x, 0..) |v, i| s += v * ref.fp4(if (i % 2 == 0) codes[i / 2] & 15 else codes[i / 2] >> 4) * ref.pow2(@as(i32, sc[i / 32]) - 127);
    return s;
}

/// Row 0's six experts in float64 with no activation rounding at all: the reference's formula, not its roundings.
fn semantic(a: std.mem.Allocator, x: []const u16, plan: ref.Plan, w1: ref.Mats, w3: ref.Mats, w2: ref.Mats, y: []const f32) !f64 {
    const k = w1.k;
    const n = w1.n;
    const xf = try a.alloc(f64, k);
    defer a.free(xf);
    for (xf, x[0..k]) |*v, h| v.* = c.fromBf16(h);
    var want: [sample_cols.len]f64 = @splat(0);
    var mag: [sample_cols.len]f64 = @splat(0);
    const act = try a.alloc(f64, n);
    defer a.free(act);
    for (0..topk) |i| {
        const e = plan.ids[i];
        for (act, 0..) |*v, j| {
            const g = @min(rawDot(w1.row(e, j), w1.srow(e, j), xf), ref.limit);
            const u = std.math.clamp(rawDot(w3.row(e, j), w3.srow(e, j), xf), -ref.limit, ref.limit);
            v.* = g / (1 + @exp(-g)) * u * plan.weights[i];
        }
        for (sample_cols, 0..) |col, s| {
            const d = rawDot(w2.row(e, col), w2.srow(e, col), act);
            want[s] += d;
            for (act, 0..) |v, j| mag[s] += @abs(v * ref.fp4(if (j % 2 == 0) w2.row(e, col)[j / 2] & 15 else w2.row(e, col)[j / 2] >> 4) * ref.pow2(@as(i32, w2.srow(e, col)[j / 32]) - 127));
        }
    }
    var worst: f64 = 0;
    for (sample_cols, 0..) |col, s| worst = @max(worst, @abs(@as(f64, y[col]) - want[s]) / mag[s]);
    return worst;
}

/// Rows 1..16 through the GPU's plan: up inside exp's spread, down and combine byte-equal to the oracle.
pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, seed: u64) !usize {
    var pipes = try ex.Pipelines.load(a, device);
    defer pipes.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const eb = 8;
    const w1 = try ref.Mats.init(a, eb, ex.width, ex.hidden);
    defer w1.deinit();
    const w3 = try ref.Mats.init(a, eb, ex.width, ex.hidden);
    defer w3.deinit();
    const w2 = try ref.Mats.init(a, eb, ex.hidden, ex.width);
    defer w2.deinit();
    for ([_]ref.Mats{ w1, w3, w2 }) |m| m.fill(rng, false);
    var bad: usize = 0;
    var worst_sem: f64 = 0;
    var checked: usize = 0;
    for (1..17) |rows| {
        var rig: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
        defer rig.deinit();
        const plan = try ref.Plan.init(a, rows, ex.experts, rng);
        defer plan.deinit();
        const np = rows * topk;
        const x = try a.alloc(u16, rows * ex.hidden);
        defer a.free(x);
        ref.fillX(x, rng);
        const mats = [3]ref.Mats{ w1, w3, w2 };
        var refs: [6]rig_.Ref = undefined;
        var backing: [6]mtl.Buffer = undefined;
        for (mats, 0..) |m, i| {
            refs[2 * i] = try rig.input(m.codes);
            refs[2 * i + 1] = try rig.input(m.scales);
        }
        for (refs, 0..) |r, i| backing[i] = r.buf;
        var table: [ex.experts]ex.ExpertPtrs = undefined;
        for (&table, 0..) |*t, e| {
            const b = e % eb;
            t.* = .{
                .w1p = address(refs[0]) + b * ex.width * ex.hidden / 2,
                .w1s = address(refs[1]) + b * ex.width * ex.hidden / 32,
                .w3p = address(refs[2]) + b * ex.width * ex.hidden / 2,
                .w3s = address(refs[3]) + b * ex.width * ex.hidden / 32,
                .w2p = address(refs[4]) + b * ex.width * ex.hidden / 2,
                .w2s = address(refs[5]) + b * ex.width * ex.hidden / 32,
            };
        }
        const ones = try a.alloc(f32, np);
        defer a.free(ones);
        @memset(ones, 1);
        const b = ex.bytes(.{ .rows = @intCast(rows) });
        const rt: ex.Route = .{
            .ids = at(try rig.input(std.mem.sliceAsBytes(plan.ids))),
            .weights = at(try rig.input(std.mem.sliceAsBytes(plan.weights))),
            .slots = undefined,
            .nslots = undefined,
            .pairs = undefined,
            .where = undefined,
        };
        const sc = [6]rig_.Ref{ try rig.scratch(b[0]), try rig.scratch(b[1]), try rig.scratch(b[2]), try rig.scratch(b[3]), try rig.scratch(b[4]), try rig.scratch(b[5]) };
        var route = rt;
        route.slots = at(sc[0]);
        route.nslots = at(sc[1]);
        route.pairs = at(sc[2]);
        route.where = at(sc[3]);
        const xr = try rig.input(std.mem.sliceAsBytes(x));
        const tr = try rig.input(std.mem.sliceAsBytes(&table));
        const unit = at(try rig.input(std.mem.sliceAsBytes(ones)));
        const y = try rig.scratch(4 * rows * ex.hidden);
        const call: ex.Call = .{ .rows = @intCast(rows) };
        const pass = rig.begin();
        ex.use(pass.e, &backing);
        ex.plan(&pipes, pass.e, route, call);
        ex.up(&pipes, pass.e, at(xr), at(tr), route, at(sc[4]), call);
        ex.down(&pipes, pass.e, at(sc[4]), at(tr), route, at(sc[5]), .{ .rows = @intCast(rows), .k = ex.hidden, .n = ex.width });
        ex.combine(&pipes, pass.e, at(sc[5]), route, unit, at(y), true, call);
        try rig.finish(pass);
        // The plan: slots exact, every pair inside its expert's range, every position used once.
        var gpu = plan;
        gpu.slots = try a.dupe(u32, u32s(&rig, sc[0], 3 * np));
        defer a.free(gpu.slots);
        gpu.nslots = .{u32s(&rig, sc[1], 1)[0]};
        gpu.pairs = try a.dupe(u32, u32s(&rig, sc[2], np));
        defer a.free(gpu.pairs);
        gpu.where = try a.dupe(u32, u32s(&rig, sc[3], np));
        defer a.free(gpu.where);
        var plan_bad: usize = 0;
        if (gpu.nslots[0] != plan.nslots[0] or !std.mem.eql(u32, gpu.slots[0 .. 3 * gpu.nslots[0]], plan.slots[0 .. 3 * plan.nslots[0]])) plan_bad += 1;
        for (0..np) |p| {
            const pos = gpu.where[p];
            if (pos >= np or gpu.pairs[pos] != p) plan_bad += 1;
        }
        // Up: sampled column blocks inside the spread.
        const act = try a.alloc(f32, np * ex.width);
        defer a.free(act);
        const spread = try a.alloc(f64, np * ex.width);
        defer a.free(spread);
        try ref.upAct(a, x, gpu, w1, w3, &sample_blocks, .{ .act = act, .spread = spread });
        const got_act = try bf16s(a, &rig, sc[4], np * ex.width);
        defer a.free(got_act);
        var act_bad: usize = 0;
        for (0..np) |pos| for (sample_blocks) |blk| for (0..32) |j| {
            const i = pos * ex.width + 32 * blk + j;
            if (!(@abs(@as(f64, got_act[i]) - act[i]) <= spread[i])) act_bad += 1;
        };
        // Down on the GPU's own activations, then combine on the GPU's own outputs: byte-equal.
        const out = try a.alloc(f32, np * ex.hidden);
        defer a.free(out);
        const ex_f = try a.alloc(f64, np * ex.hidden);
        defer a.free(ex_f);
        const mag = try a.alloc(f64, np * ex.hidden);
        defer a.free(mag);
        try ref.downOut(got_act, gpu, w2, &sample_cols, out, ex_f, mag);
        const got_out = try bf16s(a, &rig, sc[5], np * ex.hidden);
        defer a.free(got_out);
        var out_bad: usize = 0;
        for (0..np) |pos| for (sample_cols) |j| {
            if (out[pos * ex.hidden + j] != got_out[pos * ex.hidden + j]) out_bad += 1;
        };
        const gy: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, rig.bytes(y, 4 * rows * ex.hidden)));
        var y_bad: usize = 0;
        for (0..rows) |r| for (sample_cols) |j| {
            if (ref.combineOne(gpu, ones, got_out, ex.hidden, r, j, ex.per_group) != gy[r * ex.hidden + j]) y_bad += 1;
        };
        var sem: f64 = 0;
        if (rows == 1) {
            sem = try semantic(a, x, plan, w1, w3, w2, gy);
            worst_sem = sem;
        }
        const fail = plan_bad + act_bad + out_bad + y_bad + @intFromBool(sem > 0.25);
        bad += fail;
        checked += 1;
        if (fail > 0) std.debug.print("FAIL dsv41_experts pipeline rows {d}: plan {d}, up {d}, down {d}, combine {d}, vs float64 {d:.4}\n", .{ rows, plan_bad, act_bad, out_bad, y_bad, sem });
    }
    const word = if (bad == 0) "PASS" else "FAIL";
    std.debug.print("{s} dsv41_experts pipeline: rows 1..16 at hidden {d}, width {d}, {d} experts, top-{d} ({d} layers): GPU plan consistent, up inside the oracle's spread on 4 of 72 blocks a pair, down and combine byte-equal on 8 columns, rows 1 within {d:.4} of float64's unrounded formula, {d} bad\n", .{ word, ex.hidden, ex.width, ex.experts, topk, checked, worst_sem, bad });
    return bad;
}

/// Median microseconds per layer over 16 layers a command buffer, weights of 96 distinct experts read once a layer set.
pub fn timing(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, io: std.Io) !void {
    var pipes = try ex.Pipelines.load(a, device);
    defer pipes.deinit();
    const eb = 96;
    const sizes = [6]usize{ ex.width * ex.hidden / 2, ex.width * ex.hidden / 32, ex.width * ex.hidden / 2, ex.width * ex.hidden / 32, ex.width * ex.hidden / 2, ex.width * ex.hidden / 32 };
    var bufs: [6]mtl.Buffer = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    for (&bufs, sizes) |*b, sz| {
        b.* = try device.buffer(sz * eb, opts);
        rng.bytes(b.contents()[0..sz]);
        for (1..eb) |e| @memcpy(b.contents()[e * sz ..][0..sz], b.contents()[0..sz]);
    }
    defer for (bufs) |b| b.deinit();
    var table: [ex.experts]ex.ExpertPtrs = undefined;
    for (&table, 0..) |*t, e| {
        const b = e % eb;
        t.* = .{ .w1p = bufs[0].gpuAddress() + b * sizes[0], .w1s = bufs[1].gpuAddress() + b * sizes[1], .w3p = bufs[2].gpuAddress() + b * sizes[2], .w3s = bufs[3].gpuAddress() + b * sizes[3], .w2p = bufs[4].gpuAddress() + b * sizes[4], .w2s = bufs[5].gpuAddress() + b * sizes[5] };
    }
    const tbl = try device.buffer(@sizeOf(@TypeOf(table)), opts);
    defer tbl.deinit();
    @memcpy(tbl.contents()[0..@sizeOf(@TypeOf(table))], std.mem.asBytes(&table));
    const layers = 16;
    const per_expert: f64 = @floatFromInt(3 * (sizes[0] + sizes[1]));
    for ([_]u32{ 1, 4, 16 }) |rows| {
        const np = rows * topk;
        const call: ex.Call = .{ .rows = rows };
        const b = ex.bytes(call);
        const ids = try device.buffer(4 * np * layers, opts);
        defer ids.deinit();
        const idv: []u32 = @alignCast(std.mem.bytesAsSlice(u32, ids.contents()[0 .. 4 * np * layers]));
        for (0..layers) |l| for (0..np) |i| {
            idv[l * np + i] = @intCast((l * np + i) % eb);
        };
        const w = try device.buffer(4 * np, opts);
        defer w.deinit();
        const wv: []f32 = @alignCast(std.mem.bytesAsSlice(f32, w.contents()[0 .. 4 * np]));
        @memset(wv, 1);
        var sc: [6]mtl.Buffer = undefined;
        for (&sc, b) |*s, len| s.* = try device.buffer(len, opts);
        defer for (sc) |s| s.deinit();
        const xs = try device.buffer(2 * rows * ex.hidden, opts);
        defer xs.deinit();
        const xh: []u16 = @alignCast(std.mem.bytesAsSlice(u16, xs.contents()[0 .. 2 * rows * ex.hidden]));
        ref.fillX(xh, rng);
        const y = try device.buffer(4 * rows * ex.hidden, opts);
        defer y.deinit();
        const parts = [_]struct { name: []const u8, up: bool, down: bool, combine: bool }{
            .{ .name = "plan+up+down+combine", .up = true, .down = true, .combine = true },
            .{ .name = "plan+up", .up = true, .down = false, .combine = false },
            .{ .name = "plan+down", .up = false, .down = true, .combine = false },
        };
        for (parts) |part| {
            var samples: [7]f64 = undefined;
            for (&samples) |*out| {
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = queue.commandBuffer();
                const e = cb.compute(.serial);
                ex.use(e, &bufs);
                for (0..layers) |l| {
                    const route: ex.Route = .{ .ids = .{ .buf = ids, .off = 4 * np * l }, .weights = .{ .buf = w }, .slots = .{ .buf = sc[0] }, .nslots = .{ .buf = sc[1] }, .pairs = .{ .buf = sc[2] }, .where = .{ .buf = sc[3] } };
                    ex.plan(&pipes, e, route, call);
                    if (part.up) ex.up(&pipes, e, .{ .buf = xs }, .{ .buf = tbl }, route, .{ .buf = sc[4] }, call);
                    if (part.down) ex.down(&pipes, e, .{ .buf = sc[4] }, .{ .buf = tbl }, route, .{ .buf = sc[5] }, .{ .rows = rows, .k = ex.hidden, .n = ex.width });
                    if (part.combine) ex.combine(&pipes, e, .{ .buf = sc[5] }, route, .{ .buf = w }, .{ .buf = y }, true, call);
                }
                const t0 = std.Io.Clock.awake.now(io);
                e.end();
                cb.commit();
                cb.wait();
                const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
                out.* = ns / 1000.0 / layers;
            }
            std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
            const us = samples[samples.len / 2];
            const share: f64 = if (part.up and part.down) 1.0 else if (part.up) 2.0 / 3.0 else 1.0 / 3.0;
            const mb = per_expert * share * @as(f64, @floatFromInt(np)) / 1e6;
            std.debug.print("TIME dsv41_experts {s}: rows {d} ({d} experts, one pair each): {d:.1} us a layer, {d:.0} GB/s over {d:.1} MB of weights\n", .{ part.name, rows, np, us, mb * 1e6 / us / 1000.0, mb });
        }
    }
}
