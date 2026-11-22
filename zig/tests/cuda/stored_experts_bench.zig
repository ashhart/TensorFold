//! `stored-experts bench DIR`: one layer's grouped experts (gate-up and down) at 1, 4 and 512 tokens, bytes per time.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const st = @import("stored_experts_tests.zig");

const Gpu = check.Gpu;
const grouped = cuda.grouped;
const ex_ = cuda.experts;
const mx = cuda.experts_mx;

const peak = 273.0; // GB10's unified memory, GB/s
const reps = 200;
const runs = 5;

pub fn bench(gpu: Gpu, dir: []const u8) !void {
    const a = gpu.gpa;
    const path = try std.fmt.allocPrint(a, "{s}/model-00001-of-00001.safetensors", .{dir});
    defer a.free(path);
    var f = try core.safetensors.File.openPrefix(a, gpu.io, path, "model.llm.layers.");
    defer f.close(gpu.io);
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k4 = try ex_.Nvfp4Experts.load(gpu.d, sms);
    defer k4.unload();
    var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
    defer k8.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const gx: ex_.Grouped = .{ .nvfp4 = &k4, .mxfp8 = &k8 };
    for ([_]usize{ 0, 20 }) |layer| {
        const s = try st.Stored.read(&f, layer);
        var bufs: [4]cuda.DeviceBuffer = undefined;
        const l = try s.upload(gpu, &bufs);
        defer for (&bufs) |*b| b.free();
        const per_expert = bufs[0].len / st.experts + bufs[1].len / st.experts;
        for ([_]usize{ 1, 4, 512 }) |tokens| {
            const prompt = tokens >= 64;
            const pairs = tokens * st.top;
            const tile: usize = if (prompt) 64 else 16;
            const xs = try a.alloc(u16, tokens * st.hidden);
            defer a.free(xs);
            var prng = std.Random.DefaultPrng.init(tokens);
            for (xs) |*v| v.* = core.contract.spec.toBf16(prng.random().float(f32) * 2 - 1);
            const ids = try st.picks(a, tokens, 0xbe4c + tokens);
            defer a.free(ids);
            var used: [st.experts]bool = @splat(false);
            for (ids) |e| used[@intCast(e)] = true;
            var touched: usize = 0;
            for (used) |u| touched += @intFromBool(u);
            var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs));
            defer dx.free();
            var dids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids));
            defer dids.free();
            var pb = try st.PlanBufs.init(gpu, pairs, tile);
            defer pb.deinit();
            var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * st.width * 2);
            defer act.free();
            var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * st.hidden * 4);
            defer y.free();
            const up_in: ex_.Rows = .{ .x = dx.ptr, .stride = st.hidden, .slots = st.top };
            const down_in: ex_.Rows = .{ .x = act.ptr, .stride = st.width };
            const Step = struct {
                fn go(g: ex_.Grouped, sm: cuda.Stream, p: bool, ui: ex_.Rows, di: ex_.Rows, ly: ex_.Layer, pl: *const st.PlanBufs, o1: u64, o2: u64, n: usize, t: usize) !void {
                    if (p) {
                        try g.prompt(sm, .up, ui, ly, pl.plan, o1, pl.items, n, t, -1);
                        try g.prompt(sm, .down_bf16, di, ly, pl.plan, o2, pl.items, n, t, -1);
                    } else {
                        try g.decode(sm, .up, ui, ly, pl.plan, o1, pl.items, n, t, -1);
                        try g.decode(sm, .down_f32, di, ly, pl.plan, o2, pl.items, n, t, -1);
                    }
                }
            };
            try router.route(stream, dids.ptr, pairs, st.experts, tile, pb.plan);
            for (0..10) |_| try Step.go(gx, stream, prompt, up_in, down_in, l, &pb, act.ptr, y.ptr, pairs, tile);
            try stream.synchronize();
            var t0 = try cuda.Event.init(gpu.d, true);
            defer t0.deinit();
            var t1 = try cuda.Event.init(gpu.d, true);
            defer t1.deinit();
            var us: [runs]f64 = undefined;
            for (&us) |*u| {
                try t0.record(stream);
                for (0..reps) |_| try Step.go(gx, stream, prompt, up_in, down_in, l, &pb, act.ptr, y.ptr, pairs, tile);
                try t1.record(stream);
                try t1.synchronize();
                u.* = @as(f64, try cuda.Event.elapsedMs(t0, t1)) * 1000 / reps;
            }
            const med = check.median(&us);
            // the weights of every expert the tokens pick, read once; the rows in and out besides
            const bytes = touched * per_expert + tokens * st.hidden * 2 + pairs * st.width * 4 + pairs * st.hidden * (if (prompt) @as(usize, 2) else 4);
            const gbs = @as(f64, @floatFromInt(bytes)) / med / 1000;
            std.debug.print("RESULT stored-experts layer {d} {s} tokens {d} ({s}): {d} experts, {d} bytes, {d:.2} us median of {d}x{d}, {d:.1} GB/s = {d:.0}% of {d:.0}\n", .{ layer, if (s.nv) "NVFP4" else "MXFP8", tokens, if (prompt) "prompt form" else "decode form", touched, bytes, med, runs, reps, gbs, 100 * gbs / peak, peak });
        }
    }
}
