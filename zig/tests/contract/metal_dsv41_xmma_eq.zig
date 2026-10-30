//! DeepSeek V4.1 prompt-chunk experts: bit equality with the decode kernels per row, chunk invariance, timing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_experts_ref.zig");

const xm = tf.dsv41_xmma;
const dec = tf.dsv41_experts;
const topk = ref.topk;
const opts = mtl.ResourceOptions.shared;

fn at(r: rig_.Ref) xm.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

fn shifted(r: rig_.Ref, extra: usize) xm.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

fn address(r: rig_.Ref) u64 {
    return r.buf.gpuAddress() + r.off;
}

/// `e` experts of V4.1's expert shapes (up and gate `width` x `k`, down `hid` x `width`), random weights.
const World = struct {
    mats: [3]ref.Mats,
    refs: [6]rig_.Ref,
    table: rig_.Ref,
    e: u32,
    k: u32,
    width: u32,
    hid: u32,

    fn init(a: std.mem.Allocator, rig: *rig_.Rig, e: u32, k: u32, width: u32, hid: u32, rng: std.Random) !World {
        var w: World = undefined;
        w.e = e;
        w.k = k;
        w.width = width;
        w.hid = hid;
        w.mats = .{ try ref.Mats.init(a, e, width, k), try ref.Mats.init(a, e, width, k), try ref.Mats.init(a, e, hid, width) };
        for (w.mats, 0..) |m, i| {
            m.fill(rng, false);
            w.refs[2 * i] = try rig.input(m.codes);
            w.refs[2 * i + 1] = try rig.input(m.scales);
        }
        var table: [64]dec.ExpertPtrs = undefined;
        for (table[0..e], 0..) |*t, i| t.* = .{
            .w1p = address(w.refs[0]) + i * width * k / 2,
            .w1s = address(w.refs[1]) + i * width * k / 32,
            .w3p = address(w.refs[2]) + i * width * k / 2,
            .w3s = address(w.refs[3]) + i * width * k / 32,
            .w2p = address(w.refs[4]) + i * hid * width / 2,
            .w2s = address(w.refs[5]) + i * hid * width / 32,
        };
        w.table = try rig.input(std.mem.sliceAsBytes(table[0..e]));
        return w;
    }

    fn deinit(w: World) void {
        for (w.mats) |m| m.deinit();
    }

    fn backing(w: *const World) [6]mtl.Buffer {
        var b: [6]mtl.Buffer = undefined;
        for (&b, w.refs) |*x, r| x.* = r.buf;
        return b;
    }
};

/// One chunk through the GPU plan, the decode kernels (when `both`) and the MMA kernels; every output kept.
const Job = struct {
    slots: rig_.Ref,
    nslots: rig_.Ref,
    pairs: rig_.Ref,
    where: rig_.Ref,
    act_dec: rig_.Ref,
    out_dec: rig_.Ref,
    act_xm: rig_.Ref,
    out_xm: rig_.Ref,

    fn run(rig: *rig_.Rig, pipes: *const xm.Pipelines, dpipes: *const dec.Pipelines, w: *const World, x: xm.Ref, ids: xm.Ref, wts: xm.Ref, rows: u32, both: bool) !Job {
        const np = @as(usize, rows) * topk;
        var j: Job = undefined;
        const call: xm.Call = .{ .rows = rows, .k = w.k, .n = w.width, .experts = w.e };
        const dcall: dec.Call = .{ .rows = rows, .k = w.k, .n = w.width, .experts = w.e };
        j.slots = try rig.scratch(12 * np);
        j.nslots = try rig.scratch(4);
        j.pairs = try rig.scratch(4 * np);
        j.where = try rig.scratch(4 * np);
        j.act_dec = try rig.scratch(if (both) 2 * np * w.width else 2);
        j.out_dec = try rig.scratch(if (both) 2 * np * w.hid else 2);
        j.act_xm = try rig.output(2 * np * w.width);
        j.out_xm = try rig.output(2 * np * w.hid);
        const b = xm.bytes(call);
        const tiles = try rig.scratch(b[0]);
        const codes = try rig.scratch(b[1]);
        const scales = try rig.scratch(b[2]);
        const route: xm.Route = .{ .ids = ids, .weights = wts, .slots = at(j.slots), .nslots = at(j.nslots), .pairs = at(j.pairs), .where = at(j.where) };
        const back = w.backing();
        const pass = rig.begin();
        xm.use(pass.e, &back);
        dec.plan(dpipes, pass.e, route, dcall);
        if (both) {
            dec.up(dpipes, pass.e, x, at(w.table), route, at(j.act_dec), dcall);
            dec.down(dpipes, pass.e, at(j.act_dec), at(w.table), route, at(j.out_dec), .{ .rows = rows, .k = w.hid, .n = w.width, .experts = w.e });
        }
        xm.tiles(pipes, pass.e, route, at(tiles));
        xm.quant(pipes, pass.e, x, at(codes), at(scales), call);
        xm.up(pipes, pass.e, at(codes), at(scales), at(w.table), route, at(tiles), at(j.act_xm), call);
        xm.down(pipes, pass.e, at(j.act_xm), at(w.table), at(tiles), at(j.out_xm), .{ .rows = rows, .k = w.hid, .n = w.width, .experts = w.e });
        try rig.finish(pass);
        return j;
    }
};

fn u16s(rig: *rig_.Rig, r: rig_.Ref, n: usize) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, rig.bytes(r, 2 * n)));
}

fn u32s(rig: *rig_.Rig, r: rig_.Ref, n: usize) []const u32 {
    return @alignCast(std.mem.bytesAsSlice(u32, rig.bytes(r, 4 * n)));
}

/// bf16 values that differ in bits; two NaNs count as equal.
fn differ(a: []const u16, b: []const u16) usize {
    var n: usize = 0;
    for (a, b) |x, y| {
        const nan = (x & 0x7fff) > 0x7f80 and (y & 0x7fff) > 0x7f80;
        if (x != y and !nan) n += 1;
    }
    return n;
}

/// Random routes and activations of a chunk, on the GPU as inputs.
const Inputs = struct {
    x: []u16,
    plan: ref.Plan,
    xr: rig_.Ref,
    ids: rig_.Ref,
    wts: rig_.Ref,

    fn make(a: std.mem.Allocator, rig: *rig_.Rig, rows: usize, experts: usize, k: usize, rng: std.Random, nonfinite: bool) !Inputs {
        const plan = try ref.Plan.init(a, rows, experts, rng);
        const x = try a.alloc(u16, rows * k);
        ref.fillX(x, rng);
        if (nonfinite) ref.plantNonfinite(x, rows, k);
        return .{ .x = x, .plan = plan, .xr = try rig.input(std.mem.sliceAsBytes(x)), .ids = try rig.input(std.mem.sliceAsBytes(plan.ids)), .wts = try rig.input(std.mem.sliceAsBytes(plan.weights)) };
    }

    fn deinit(i: Inputs, a: std.mem.Allocator) void {
        a.free(i.x);
        i.plan.deinit();
    }
};

/// Both phases of every row equal the decode kernels' bits at V4.1's widths, for several chunk sizes.
fn equality(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pipes: *const xm.Pipelines, dpipes: *const dec.Pipelines, seed: u64) !usize {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var bad: usize = 0;
    var cells: usize = 0;
    for ([_]struct { rows: u32, nonfinite: bool }{ .{ .rows = 1, .nonfinite = false }, .{ .rows = 5, .nonfinite = false }, .{ .rows = 16, .nonfinite = false }, .{ .rows = 17, .nonfinite = false }, .{ .rows = 100, .nonfinite = false }, .{ .rows = 100, .nonfinite = true }, .{ .rows = 512, .nonfinite = false } }) |case| {
        var rig: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
        defer rig.deinit();
        const w = try World.init(a, &rig, 16, ex_hidden, ex_width, 256, rng);
        defer w.deinit();
        const in = try Inputs.make(a, &rig, case.rows, 16, ex_hidden, rng, case.nonfinite);
        defer in.deinit(a);
        const job = try Job.run(&rig, pipes, dpipes, &w, at(in.xr), at(in.ids), at(in.wts), case.rows, true);
        const np = case.rows * topk;
        const up_bad = differ(u16s(&rig, job.act_dec, np * w.width), u16s(&rig, job.act_xm, np * w.width));
        const down_bad = differ(u16s(&rig, job.out_dec, np * w.hid), u16s(&rig, job.out_xm, np * w.hid));
        cells += np * (w.width + w.hid);
        if (up_bad + down_bad > 0) std.debug.print("FAIL dsv41 xmma equality: rows {d}{s}: {d} up and {d} down values differ from decode\n", .{ case.rows, if (case.nonfinite) " (non-finite planted)" else "", up_bad, down_bad });
        bad += up_bad + down_bad;
    }
    std.debug.print("{s} dsv41 xmma equality with the decode kernels: rows 1/5/16/17/100/100 (non-finite)/512 at hidden {d}, width {d}, down columns 256, {d} values, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", ex_hidden, ex_width, cells, bad });
    return bad;
}

const ex_hidden = 5120;
const ex_width = 2304;

/// A chunk of rows [offset, offset + rows) gives each pair the bits the whole 8192-row call gave it.
fn invariance(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pipes: *const xm.Pipelines, dpipes: *const dec.Pipelines, seed: u64) !usize {
    var prng = std.Random.DefaultPrng.init(seed ^ 0x5eed);
    const rng = prng.random();
    const total: u32 = 8192;
    var rig: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();
    const w = try World.init(a, &rig, 24, ex_hidden, 512, 256, rng);
    defer w.deinit();
    const in = try Inputs.make(a, &rig, total, 24, ex_hidden, rng, false);
    defer in.deinit(a);
    const whole = try Job.run(&rig, pipes, dpipes, &w, at(in.xr), at(in.ids), at(in.wts), total, false);
    const np = total * topk;
    const wa = u16s(&rig, whole.act_xm, np * w.width);
    const wo = u16s(&rig, whole.out_xm, np * w.hid);
    const where_w = u32s(&rig, whole.where, np);
    var bad: usize = 0;
    var launches: usize = 0;
    for ([_]u32{ 64, 512, 2048, 8192 }) |rows| for ([_]u32{ 0, 1, 37 }) |off| {
        if (off + rows > total) continue;
        const job = try Job.run(&rig, pipes, dpipes, &w, shifted(in.xr, 2 * off * ex_hidden), shifted(in.ids, 4 * off * topk), shifted(in.wts, 4 * off * topk), rows, false);
        launches += 1;
        const ca = u16s(&rig, job.act_xm, rows * topk * w.width);
        const co = u16s(&rig, job.out_xm, rows * topk * w.hid);
        const where_c = u32s(&rig, job.where, rows * topk);
        var diff: usize = 0;
        for (0..rows * topk) |q| {
            const pw = where_w[off * topk + q];
            const pc = where_c[q];
            diff += differ(wa[pw * w.width ..][0..w.width], ca[pc * w.width ..][0..w.width]);
            diff += differ(wo[pw * w.hid ..][0..w.hid], co[pc * w.hid ..][0..w.hid]);
        }
        if (diff > 0) std.debug.print("FAIL dsv41 xmma chunk invariance: rows {d} offset {d}: {d} values differ\n", .{ rows, off, diff });
        bad += diff;
    };
    std.debug.print("{s} dsv41 xmma chunk invariance: 64/512/2048/8192 rows, offsets 0/1/37 of one 8192-row call, {d} launches, {d} values differ\n", .{ if (bad == 0) "PASS" else "FAIL", launches, bad });
    return bad;
}

pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, seed: u64) !usize {
    var pipes = try xm.Pipelines.load(a, device);
    defer pipes.deinit();
    var dpipes = try dec.Pipelines.load(a, device);
    defer dpipes.deinit();
    return try equality(a, device, queue, &pipes, &dpipes, seed) + try invariance(a, device, queue, &pipes, &dpipes, seed);
}

/// Median time of the stages of one layer at an 8192-row chunk, 384 experts (96 distinct weight sets), top-6.
pub fn timing(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, io: std.Io) !void {
    var pipes = try xm.Pipelines.load(a, device);
    defer pipes.deinit();
    var dpipes = try dec.Pipelines.load(a, device);
    defer dpipes.deinit();
    const rows: u32 = 8192;
    const np = @as(usize, rows) * topk;
    const eb = 96;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const sizes = [6]usize{ ex_width * ex_hidden / 2, ex_width * ex_hidden / 32, ex_width * ex_hidden / 2, ex_width * ex_hidden / 32, ex_width * ex_hidden / 2, ex_width * ex_hidden / 32 };
    var bufs: [6]mtl.Buffer = undefined;
    for (&bufs, sizes) |*b, sz| {
        b.* = try device.buffer(sz * eb, opts);
        rng.bytes(b.contents()[0..sz]);
        for (1..eb) |e| @memcpy(b.contents()[e * sz ..][0..sz], b.contents()[0..sz]);
    }
    defer for (bufs) |b| b.deinit();
    for (0..3) |i| {
        const scales = bufs[2 * i + 1].contents()[0 .. eb * sizes[2 * i + 1]];
        for (scales, 0..) |*v, q| v.* = 119 + @as(u8, @intCast(q % 7));
    }
    var table: [dec.experts]dec.ExpertPtrs = undefined;
    for (&table, 0..) |*t, e| {
        const b = e % eb;
        t.* = .{ .w1p = bufs[0].gpuAddress() + b * sizes[0], .w1s = bufs[1].gpuAddress() + b * sizes[1], .w3p = bufs[2].gpuAddress() + b * sizes[2], .w3s = bufs[3].gpuAddress() + b * sizes[3], .w2p = bufs[4].gpuAddress() + b * sizes[4], .w2s = bufs[5].gpuAddress() + b * sizes[5] };
    }
    const tbl = try device.buffer(@sizeOf(@TypeOf(table)), opts);
    defer tbl.deinit();
    @memcpy(tbl.contents()[0..@sizeOf(@TypeOf(table))], std.mem.asBytes(&table));
    const plan = try ref.Plan.init(a, rows, dec.experts, rng);
    defer plan.deinit();
    const call: xm.Call = .{ .rows = rows };
    const b = xm.bytes(call);
    var sc: [5]mtl.Buffer = undefined;
    for (&sc, [5]usize{ 12 * np, 4, 4 * np, 4 * np, b[0] }) |*s, len| s.* = try device.buffer(len, opts);
    defer for (sc) |s| s.deinit();
    const extra = [4]usize{ b[1], b[2], b[3], b[4] };
    var ex_bufs: [4]mtl.Buffer = undefined;
    for (&ex_bufs, extra) |*s, len| s.* = try device.buffer(len, opts);
    defer for (ex_bufs) |s| s.deinit();
    const xs = try device.buffer(2 * rows * ex_hidden, opts);
    defer xs.deinit();
    ref.fillX(@alignCast(std.mem.bytesAsSlice(u16, xs.contents()[0 .. 2 * rows * ex_hidden])), rng);
    const ids = try device.buffer(4 * np, opts);
    defer ids.deinit();
    @memcpy(ids.contents()[0 .. 4 * np], std.mem.sliceAsBytes(plan.ids));
    const wt = try device.buffer(4 * np, opts);
    defer wt.deinit();
    @memcpy(wt.contents()[0 .. 4 * np], std.mem.sliceAsBytes(plan.weights));
    const where = try device.buffer(4 * np, opts);
    defer where.deinit();
    const route: xm.Route = .{ .ids = .{ .buf = ids }, .weights = .{ .buf = wt }, .slots = .{ .buf = sc[0] }, .nslots = .{ .buf = sc[1] }, .pairs = .{ .buf = sc[2] }, .where = .{ .buf = where } };
    const names = [_][]const u8{ "plan+tiles+quant", "up", "down", "whole layer" };
    var samples: [4][5]f64 = undefined;
    for (0..5) |it| for (0..4) |stage| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        xm.use(e, &bufs);
        const pre = stage == 0 or stage == 3;
        if (pre) {
            dec.plan(&dpipes, e, route, .{ .rows = rows });
            xm.tiles(&pipes, e, route, .{ .buf = sc[4] });
            xm.quant(&pipes, e, .{ .buf = xs }, .{ .buf = ex_bufs[0] }, .{ .buf = ex_bufs[1] }, call);
        }
        if (stage == 1 or stage == 3) xm.up(&pipes, e, .{ .buf = ex_bufs[0] }, .{ .buf = ex_bufs[1] }, .{ .buf = tbl }, route, .{ .buf = sc[4] }, .{ .buf = ex_bufs[2] }, call);
        if (stage == 2 or stage == 3) xm.down(&pipes, e, .{ .buf = ex_bufs[2] }, .{ .buf = tbl }, .{ .buf = sc[4] }, .{ .buf = ex_bufs[3] }, call);
        if (stage == 1 or stage == 2) {
            dec.plan(&dpipes, e, route, .{ .rows = rows });
            xm.tiles(&pipes, e, route, .{ .buf = sc[4] });
        }
        const t0 = std.Io.Clock.awake.now(io);
        e.end();
        cb.commit();
        cb.wait();
        samples[stage][it] = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1000.0;
    };
    const flops = [4]f64{ 0, 2.0 * 2.0 * ex_hidden * ex_width, 2.0 * ex_width * ex_hidden, 3.0 * 2.0 * ex_hidden * ex_width };
    for (&samples, names, flops) |*s, name, f| {
        std.mem.sort(f64, s, {}, std.sort.asc(f64));
        const us = s[2];
        const tf_ = f * @as(f64, @floatFromInt(np)) / us / 1e6;
        std.debug.print("TIME dsv41 xmma {s}: {d} rows, 384 experts, top-{d}: {d:.0} us a layer, {d:.2} TFLOPS\n", .{ name, rows, topk, us, tf_ });
    }
}
