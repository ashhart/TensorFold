//! Dense hyper-connection normalization, projections and mixing against an independent FP64 chain.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const bf = rig.bf;
const rbf = rig.rbf;
const d = 2560;
const streams = 4;
const wide = d * streams;
const low = 320;
const ndn = low + streams;
const most = 17;

fn tensor(values: []const u16, n: usize, k: usize) core.safetensors.Tensor {
    return .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(values) };
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0xde05e);
    var r = try rig.Rig.init(gpu, arena.allocator(), prng.random());
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    const h = r.bfs(most * wide, 1);
    const gamma = r.f32s(wide, 0.5, 1.5);
    const dw = r.bfs(ndn * wide, 0.01);
    const uw = r.bfs(wide * low, 0.03);
    var down = try flash.projection.dense(r.a, gpu.d, tensor(dw, ndn, wide));
    defer down.free();
    var up = try flash.projection.dense(r.a, gpu.d, tensor(uw, wide, low));
    defer up.free();
    const dh = try r.dev(std.mem.sliceAsBytes(h));
    const dg = try r.dev(std.mem.sliceAsBytes(gamma));
    const pss = try r.out(most * (d / 256) * streams * 4);
    try g.hcWriteback(r.s, dh, dh, pss, 0, 0, 0, 0, most * d, most, d, streams, 0, 1, 1, 1);
    const sums = try r.back(f32, pss, most * (d / 256) * streams);
    const norm = try r.out(most * wide * 2);
    const projected = try r.out(most * ndn * 4);
    const act = try r.out(most * low * 2);
    const xs = try r.out(most * (low / 32) * 4);
    const inj = try r.out(most * streams * 2);
    const up_out = try r.out(most * wide * 2);
    const mixed = try r.out(most * d * 2);
    const xsm = try r.out(most * (d / 32) * 4);
    const part = try r.out(8 * most * ndn * 4);
    var norms: []u16 = undefined;
    var dots: []f32 = undefined;
    var acts: []u16 = undefined;
    var injs: []u16 = undefined;
    var ups: []u16 = undefined;
    var mixes: []u16 = undefined;
    var mix_sums: []f32 = undefined;
    for ([_]usize{ most, 1, 7 }) |rows| {
        try g.hcNormalize(r.s, dh, pss, dg, norm, 1e-6, rows, d, streams);
        try lane.matmulF32(r.s, norm, wide, rows, down.native.?, projected, part);
        try g.hcReduceAct(r.s, projected, act, xs, inj, 1, rows, streams, low, ndn, 1);
        try lane.matmul(r.s, act, low, rows, up.native.?, up_out, part);
        try g.hcDenseMix(r.s, up_out, norm, mixed, xsm, rows, d, streams);
        if (rows == most) {
            norms = try r.back(u16, norm, most * wide);
            dots = try r.back(f32, projected, most * ndn);
            acts = try r.back(u16, act, most * low);
            injs = try r.back(u16, inj, most * streams);
            ups = try r.back(u16, up_out, most * wide);
            mixes = try r.back(u16, mixed, most * d);
            mix_sums = try r.back(f32, xsm, most * (d / 32));
        } else {
            try r.sameRows("dense HC normalize", u16, norm, norms[0 .. rows * wide]);
            try r.sameRows("dense HC down FP32", f32, projected, dots[0 .. rows * ndn]);
            try r.sameRows("dense HC activation", u16, act, acts[0 .. rows * low]);
            try r.sameRows("dense HC inject", u16, inj, injs[0 .. rows * streams]);
            try r.sameRows("dense HC up BF16", u16, up_out, ups[0 .. rows * wide]);
            try r.sameRows("dense HC mix", u16, mixed, mixes[0 .. rows * d]);
            try r.sameRows("dense HC group sums", f32, xsm, mix_sums[0 .. rows * (d / 32)]);
        }
    }
    for (0..most) |row| {
        for (0..streams) |st| {
            var total: f64 = 0;
            for (0..d / 256) |c| total += sums[(row * (d / 256) + c) * streams + st];
            const inv = 1 / @sqrt(total / d + 1e-6);
            for (0..d) |col| {
                const at = row * wide + st * d + col;
                const want = bf(h[at]) * inv * gamma[st * d + col];
                try r.within("dense norm FP32 gamma", bf(norms[at]), want, rig.ulps(want, 0.5) + @abs(want) * 0x1p-20);
            }
        }
        for (0..ndn) |col| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..wide) |i| {
                const v = bf(norms[row * wide + i]) * bf(dw[col * wide + i]);
                sum += v;
                magnitude += @abs(v);
            }
            try r.within("dense down FP32", dots[row * ndn + col], sum, magnitude * 2e-6 + @abs(sum) * 1e-6 + 1e-30);
            const v = rbf(rbf(dots[row * ndn + col]) / streams);
            if (col < low) {
                const want = rbf(v * rig.sigmoid(v));
                try r.within("dense activation", bf(acts[row * low + col]), want, rig.ulps(want, 1));
            } else {
                const want = rbf(2 * rbf(rig.sigmoid(v)));
                try r.within("dense inject", bf(injs[row * streams + col - low]), want, rig.ulps(want, 1));
            }
        }
        for (0..wide) |col| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..low) |i| {
                const v = bf(acts[row * low + i]) * bf(uw[col * low + i]);
                sum += v;
                magnitude += @abs(v);
            }
            try r.within("dense up BF16", bf(ups[row * wide + col]), sum, rig.ulps(sum, 0.5) + magnitude * 2e-6);
        }
        for (0..d / 32) |group| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..32) |i| {
                const col = group * 32 + i;
                var total: f64 = 0;
                var terms: f64 = 0;
                for (0..streams) |st| {
                    const at = row * wide + st * d + col;
                    const term = rbf(rbf(rig.sigmoid(bf(ups[at]))) * bf(norms[at]));
                    total += term;
                    terms += @abs(term);
                }
                const want = rbf(total / streams);
                const got = bf(mixes[row * d + col]);
                try r.within("dense stream mix", got, want, rig.ulps(terms / streams, 1) + rig.ulps(want, 0.5));
                sum += got;
                magnitude += @abs(got);
            }
            try r.within("dense group sums", mix_sums[row * (d / 32) + group], sum, rig.sumTol(magnitude, 32));
        }
    }
    r.report("dense HC: FP32 gamma, BF16 weights, FP32 down and one-round up; rows 1/7/17");
}
