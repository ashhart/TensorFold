//! Tensor- and expert-parallel ranks on our kernels, reduced by the cluster's canon: the same bits as one node.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const cluster = @import("cluster");
const gpu_mod = @import("gpu.zig");
const synth = @import("synth.zig");

const canon = cluster.canon;
const exchange = cluster.exchange;
const Gpu = gpu_mod.Gpu;

/// One command buffer: `begin`, encode on `.e`, then `finish` waits and reports a GPU failure.
const Cmd = struct {
    pool: mtl.objc.Pool,
    cb: mtl.CommandBuffer,
    e: mtl.ComputeEncoder,

    fn begin(g: *Gpu) Cmd {
        const pool = mtl.objc.Pool.push();
        const cb = g.queue.commandBuffer();
        return .{ .pool = pool, .cb = cb, .e = cb.compute(.serial) };
    }

    fn finish(c: Cmd) !void {
        defer c.pool.pop();
        c.e.end();
        c.cb.commit();
        c.cb.wait();
        if (c.cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
    }
};

const RankJob = struct { m: *exchange.Mem, r: u32, mine: canon.Range, slices: []const []const f32, rows: usize, width: usize, out: []f32, err: ?anyerror = null };

fn rankMain(j: *RankJob) void {
    exchange.allReduce(std.heap.page_allocator, j.m.endpoint(j.r), 8, j.mine, j.slices, j.rows, j.width, j.out) catch |err| {
        j.err = err;
    };
}

fn toBf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

/// o_proj split by heads over `tp` ranks: slice partials on the GPU, exchange.allReduce over threads, vs one node.
fn tensorParallel(g: *Gpu, gpa: std.mem.Allocator, src: *synth.Synth, sc: *k3.round.Scratch) !void {
    const cfg = k3.config.Config{};
    const H = cfg.hidden;
    const K = cfg.kdaWidth();
    var w = try src.source().get("layers.1.self_attn.o_proj.weight", .bf16, &.{ H, K });
    var owned = k3.prepare.Owned{ .gpa = gpa };
    defer owned.deinit();
    try runInterleave(g, &w, &owned);
    const P = try g.device.buffer(8 * 128 * @as(usize, H) * 4, mtl.ResourceOptions.shared);
    defer P.deinit();
    for ([_]u32{ 1, 5, 40, 128 }) |R| {
        var name: [64]u8 = undefined;
        for (0..R) |r| synth.rowValues(sc.y.slice(u16, (r + 1) * K)[r * K ..], try std.fmt.bufPrint(&name, "tp.x.{d}", .{r}));
        const a = k3.kernels.RowsArgs{ .K = K, .N = H, .rows = R, .x_stride = K, .y_stride = H };
        var c = Cmd.begin(g);
        g.k.rows(c.e, sc.ref("y"), w.ref, sc.ref("delta"), sc.ref("part"), false, a);
        try c.finish();
        const want = sc.delta.slice(u16, R * H);
        for ([_]u32{ 2, 4, 8 }) |tp| {
            const per = 8 / tp;
            const floats = @as(usize, R) * H;
            const parts = try gpa.alloc(f32, 8 * floats);
            defer gpa.free(parts);
            for (0..tp) |rk| {
                c = Cmd.begin(g);
                g.k.slicePartials(c.e, sc.ref("y"), w.ref, .{ .buf = P }, a, @intCast(rk * per), per);
                try c.finish();
                @memcpy(parts[rk * per * floats ..][0 .. per * floats], P.slice(f32, per * floats));
            }
            const m = try exchange.Mem.init(gpa, tp);
            defer m.deinit();
            var jobs: [8]RankJob = undefined;
            var views: [8][8][]const f32 = undefined;
            var outs = try gpa.alloc(f32, tp * floats);
            defer gpa.free(outs);
            var threads: [8]std.Thread = undefined;
            for (0..tp) |rk| {
                for (0..per) |s| views[rk][s] = parts[(rk * per + s) * floats ..][0..floats];
                jobs[rk] = .{ .m = m, .r = @intCast(rk), .mine = .{ .begin = @intCast(rk * per), .end = @intCast((rk + 1) * per) }, .slices = views[rk][0..per], .rows = R, .width = H, .out = outs[rk * floats ..][0..floats] };
                threads[rk] = try std.Thread.spawn(.{}, rankMain, .{&jobs[rk]});
            }
            for (threads[0..tp]) |t| t.join();
            for (jobs[0..tp], 0..) |j, rk| {
                if (j.err) |err| return err;
                for (outs[rk * floats ..][0..floats], want) |v, ref| if (toBf16(v) != ref) return error.TensorParallelDiffers;
            }
        }
        std.debug.print("TP: o_proj 7168x12288, {d:>3} rows: TP2, TP4 and TP8 ranks give one node's bits on every rank\n", .{R});
    }
}

fn runInterleave(g: *Gpu, w: *k3.store.Tensor, owned: *k3.prepare.Owned) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const buf = try g.device.buffer(w.bytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    try owned.bufs.append(owned.gpa, buf);
    const c = Cmd.begin(g);
    g.k.interleaved(c.e, w.ref, .{ .buf = buf }, w.shape[0], w.shape[1]);
    try c.finish();
    w.ref = .{ .buf = buf };
}

/// Routed experts over `ep` ranks (whole canonical groups each): each rank's subtree, completed by canon.reduce.
fn expertParallel(g: *Gpu, gpa: std.mem.Allocator, src: *synth.Synth, sc: *k3.round.Scratch) !void {
    const cfg = k3.config.Config{};
    const L = cfg.latent;
    var l = try k3.weights.layer(&cfg, 1, src.source(), g.device, gpa, 0, cfg.experts);
    var owned = k3.prepare.Owned{ .gpa = gpa };
    defer owned.deinit();
    try k3.prepare.layer(&g.k, g.device, g.queue, &l, &owned);
    const set = try gpu_mod.resident(g, &.{ src.bufs.items, &.{l.mlp.moe.table} });
    defer {
        g.queue.removeResidencySet(set);
        set.deinit();
    }
    const part = try g.device.buffer(256 * @as(usize, L) * 4, mtl.ResourceOptions.shared);
    defer part.deinit();
    for ([_]u32{ 1, 5, 40, 128 }) |R| {
        sc.rows = R;
        const ids = sc.tids.slice(u32, R * cfg.topk);
        const wts = sc.tw.slice(f32, R * cfg.topk);
        for (0..R) |r| for (0..cfg.topk) |j| {
            ids[r * cfg.topk + j] = (synth.mix(0x5EED, @intCast(r * 16 + j)) % 56) * 16 + @as(u32, @intCast(j));
            wts[r * cfg.topk + j] = 1.0 / 16.0 + @as(f32, @floatFromInt(j)) / 512.0;
        };
        var name: [64]u8 = undefined;
        for (0..R) |r| synth.rowValues(sc.lat.slice(u16, (r + 1) * L)[r * L ..], try std.fmt.bufPrint(&name, "ep.lat.{d}", .{r}));
        var one = k3.experts.Local.all(&g.k, &cfg);
        var c = Cmd.begin(g);
        try one.experts().run(c.e, l.mlp.moe.table, sc);
        try c.finish();
        const want = try gpa.dupe(u16, sc.ysum.slice(u16, R * L));
        defer gpa.free(want);
        for ([_]u32{ 2, 4, 8 }) |ep| {
            const floats = @as(usize, R) * L;
            const datas = try gpa.alloc(f32, ep * floats);
            defer gpa.free(datas);
            var parts: [8]canon.Part = undefined;
            for (0..ep) |rk| {
                var mine = k3.experts.Local{ .k = &g.k, .c = &cfg, .first = @intCast(rk * cfg.experts / ep), .last = @intCast((rk + 1) * cfg.experts / ep), .partial = part };
                c = Cmd.begin(g);
                try mine.experts().run(c.e, l.mlp.moe.table, sc);
                try c.finish();
                @memcpy(datas[rk * floats ..][0..floats], part.slice(f32, floats));
                parts[rk] = .{ .range = .{ .begin = @intCast(rk * 8 / ep), .end = @intCast((rk + 1) * 8 / ep) }, .data = datas[rk * floats ..][0..floats] };
            }
            const out = try gpa.alloc(f32, floats);
            defer gpa.free(out);
            canon.reduce(8, parts[0..ep], out);
            for (out, want) |v, ref| if (toBf16(v) != ref) return error.ExpertParallelDiffers;
        }
        std.debug.print("EP: routed experts, {d:>3} rows: EP2, EP4 and EP8 subtrees completed by canon.reduce give one node's bits\n", .{R});
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    std.debug.print("device: {s}\n", .{g.device.name()});
    const cfg = k3.config.Config{};
    var src = synth.Synth{ .gpa = gpa, .device = g.device, .queue = g.queue, .k = &g.k };
    defer src.deinit();
    var sc = try k3.round.Scratch.init(g.device, &cfg, 128);
    defer sc.deinit();
    try tensorParallel(&g, gpa, &src, &sc);
    try expertParallel(&g, gpa, &src, &sc);
    std.debug.print("PASS\n", .{});
}
