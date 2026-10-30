//! TF_DSV41_TIMING_MV: the bf16 matvec's launch geometries (rows, columns, unroll) at the layer's and head's shapes.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const counters = mtl.counters;

const V = struct { rt: u32, nb: u32, u: u32 };
const variants = blk: {
    var out: [52]V = undefined;
    out[48..52].* = .{ .{ .rt = 1, .nb = 1, .u = 32 }, .{ .rt = 16, .nb = 2, .u = 16 }, .{ .rt = 16, .nb = 1, .u = 8 }, .{ .rt = 16, .nb = 2, .u = 8 } };
    var i: usize = 0;
    for ([_]u32{ 1, 2, 4, 8 }) |rt| for ([_]u32{ 1, 2, 4, 8 }) |nb| for ([_]u32{ 4, 8, 16 }) |u| {
        out[i] = .{ .rt = rt, .nb = nb, .u = u };
        i += 1;
    };
    break :blk out;
};

fn vname(comptime v: V) [:0]const u8 {
    return std.fmt.comptimePrint("sw_{d}_{d}_{d}", .{ v.rt, v.nb, v.u });
}

const text = blk: {
    var s: []const u8 = sources.deepseek_v41_mv[0..std.mem.indexOf(u8, sources.deepseek_v41_mv, "#define TF_MV(").?];
    for (variants) |v| s = s ++ std.fmt.comptimePrint("template [[host_name(\"{s}\")]] [[kernel]] decltype(tf_dsv41_mv<bfloat, float, {d}, {d}, {d}>) tf_dsv41_mv<bfloat, float, {d}, {d}, {d}>;\n", .{ vname(v), v.rt, v.nb, v.u, v.rt, v.nb, v.u });
    break :blk s;
};

const Args = extern struct { k: u32, n: u32, rows: u32 };

fn flush(queue: mtl.Queue, junk: mtl.Buffer) void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const b = cb.blit();
    b.fill(junk, 0, junk.length(), 0x5a);
    b.end();
    cb.commit();
    cb.wait();
}

pub fn run(device: mtl.Device, queue: mtl.Queue, junk: mtl.Buffer, n_runs: usize) !void {
    @setEvalBranchQuota(100000);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: [variants.len]mtl.Pipeline = undefined;
    inline for (variants, 0..) |v, i| pipes[i] = try mtl.Pipeline.init(device, lib, vname(v), false);
    const S = struct { k: u32, n: u32 };
    const shapes = [_]S{ .{ .k = 5120, .n = 512 }, .{ .k = 5120, .n = 384 }, .{ .k = 5120, .n = 32 }, .{ .k = 512, .n = 128 }, .{ .k = 5120, .n = 129280 } };
    const wb = try device.buffer(5120 * 129280 * 2, mtl.ResourceOptions.private);
    defer wb.deinit();
    const xb = try device.buffer(16 * 5120 * 2, mtl.ResourceOptions.private);
    defer xb.deinit();
    const yb = try device.buffer(16 * 129280 * 4, mtl.ResourceOptions.private);
    defer yb.deinit();
    const samples = try counters.SampleBuffer.init(device, 64);
    defer samples.deinit();
    const pass = try counters.ComputePass.init();
    defer pass.deinit();
    var t: [16]f64 = undefined;
    for (shapes) |sh| for ([_]u32{ 1, 2, 4, 5, 8, 16 }) |rows| {
        var best: [variants.len]f64 = undefined;
        for (&pipes, 0..) |p, vi| {
            const v = variants[vi];
            if (v.rt > 2 * rows and v.rt > 1) {
                best[vi] = 1e9;
                continue;
            }
            for (t[0..n_runs]) |*x| {
                flush(queue, junk);
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = queue.commandBuffer();
                const e = try pass.compute(cb, samples, 0, 1);
                e.setPipeline(p);
                e.setBuffer(xb, 0, 0);
                e.setBuffer(wb, 0, 1);
                e.setBuffer(yb, 0, 2);
                e.setValue(Args{ .k = sh.k, .n = sh.n, .rows = rows }, 3);
                e.dispatchGroups(mtl.Size.of((sh.n + v.nb - 1) / v.nb, (rows + v.rt - 1) / v.rt, 1), mtl.Size.of(32, 1, 1));
                e.end();
                const before = counters.sampleClocks(device);
                cb.commit();
                cb.wait();
                const after = counters.sampleClocks(device);
                const res = try samples.resolveCompleted(2);
                defer res.deinit();
                const vals = res.values();
                const spt = @as(f64, @floatFromInt(after.cpu - before.cpu)) / @as(f64, @floatFromInt(after.gpu - before.gpu)) * 1e-9;
                x.* = @as(f64, @floatFromInt(vals[1].timestamp -% vals[0].timestamp)) * spt;
            }
            var m: f64 = 1e9;
            for (t[0..n_runs]) |x| m = @min(m, x);
            best[vi] = m;
        }
        var order: [variants.len]usize = undefined;
        for (&order, 0..) |*o, i| o.* = i;
        const C = struct {
            fn lt(b: *const [variants.len]f64, x: usize, y: usize) bool {
                return b[x] < b[y];
            }
        };
        std.mem.sort(usize, &order, &best, C.lt);
        std.debug.print("mv {d}x{d} rows {d}:", .{ sh.k, sh.n, rows });
        for (order[0..6]) |i| std.debug.print("  {d}/{d}/{d} {d:.1}", .{ variants[i].rt, variants[i].nb, variants[i].u, best[i] * 1e6 });
        std.debug.print("\n", .{});
    };
}
