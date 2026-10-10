//! Sliding Weights' cost to serving on CUDA: a decode token's milliseconds without --slide, and with 0 to 512 ranks.

const std = @import("std");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const dims = nemotron.slide_dims;
const tokens = 96;

/// MODEL from IDS_FILE's prompt: one serial decode token's mean milliseconds in each case, through the captured graphs.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |w| try ids.append(gpa, try std.fmt.parseInt(u32, w, 10));
    var base: f64 = 0;
    for ([_]bool{ false, true }) |slide| {
        const e = try nemotron.Engine.init(gpa, io, gpu.ctx, model, null, .{ .context = 2048, .mtp = false, .graphs = true, .sampling = null, .segments = 1, .slide = slide });
        defer e.deinit();
        try e.captureWindows(); // the one-row window's graph, as the server captures it without drafts
        if (!slide) {
            base = try decode(e, ids.items);
            std.debug.print("RESULT without --slide: {d:.3} ms a token\n", .{base});
            continue;
        }
        var prng = std.Random.DefaultPrng.init(9);
        const r = prng.random();
        const s = &e.slide.?;
        for (s.list) |*site| {
            for (site.a.slice(f32, dims.max_rank * site.in)) |*v| v.* = (r.float(f32) - 0.5) * 1e-3;
            for (site.b.slice(f32, dims.max_rank * site.out)) |*v| v.* = (r.float(f32) - 0.5) * 1e-4;
            for (site.tau.slice(f32, dims.max_blocks)) |*v| v.* = -std.math.inf(f32);
        }
        for ([_]usize{ 0, dims.block, 4 * dims.block, dims.max_rank }) |rank| {
            s.rank = rank;
            s.attach(&e.w, true);
            const ms = try decode(e, ids.items);
            std.debug.print("RESULT --slide with {d} ranks: {d:.3} ms a token ({d:.1}% over none)\n", .{ rank, ms, 100 * (ms / base - 1) });
        }
    }
}

/// The change's two window kernels alone at 512 ranks on one row: a and b in mapped host memory, then on the device.
pub fn kernels(gpu: check.Gpu) !void {
    const cuda = @import("cuda");
    var k = try nemotron.kernels.Kernels.load(gpu.gpa, gpu.io, gpu.ctx, null);
    try k.loadTrain();
    defer k.deinit();
    var st = try cuda.Stream.init(gpu.d, true);
    defer st.deinit();
    const t: nemotron.train_ops.Train = .{ .f = &k.train, .s = st, .d = gpu.d };
    const in = 4096;
    const out = 2688;
    var word = try nemotron.sites.Shared.init(gpu.d, u32, 1);
    defer word.free();
    word.slice(u32, 1)[0] = dims.max_rank;
    var ha = try nemotron.sites.Shared.init(gpu.d, f32, dims.max_rank * in);
    defer ha.free();
    var hb = try nemotron.sites.Shared.init(gpu.d, f32, dims.max_rank * out);
    defer hb.free();
    var tau = try nemotron.sites.Shared.init(gpu.d, f32, dims.max_blocks);
    defer tau.free();
    for (tau.slice(f32, dims.max_blocks)) |*v| v.* = -std.math.inf(f32);
    var da = try cuda.DeviceBuffer.alloc(gpu.d, dims.max_rank * in * 4);
    defer da.free();
    var db = try cuda.DeviceBuffer.alloc(gpu.d, dims.max_rank * out * 4);
    defer db.free();
    var x = try cuda.DeviceBuffer.alloc(gpu.d, in * 2);
    defer x.free();
    var y = try cuda.DeviceBuffer.alloc(gpu.d, out * 4);
    defer y.free();
    var xa = try cuda.DeviceBuffer.alloc(gpu.d, 16 * dims.max_rank * 4);
    defer xa.free();
    var xn = try cuda.DeviceBuffer.alloc(gpu.d, 16 * 4);
    defer xn.free();
    for ([2][2]u64{ .{ ha.dev, hb.dev }, .{ da.ptr, db.ptr } }, [2][]const u8{ "mapped host", "device" }) |ab, where| {
        const ad: nemotron.weights.Adapter = .{ .a = ab[0], .b = ab[1], .tau = tau.dev, .rank = word.dev, .xa = xa.ptr, .xn = xn.ptr, .in = in, .out = out };
        for (0..20) |_| try t.adapt(ad, x.ptr, in, y.ptr, true, out, 1, 0, 1);
        try st.synchronize();
        const t0 = check.now(gpu.io);
        for (0..1000) |_| try t.adapt(ad, x.ptr, in, y.ptr, true, out, 1, 0, 1);
        try st.synchronize();
        const us = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e3 / 1000;
        const gbs = @as(f64, @floatFromInt(dims.max_rank * (in + out) * 4)) / us / 1e3;
        std.debug.print("RESULT the change's window kernels, 512 ranks, a and b in {s} memory: {d:.1} us ({d:.0} GB/s)\n", .{ where, us, gbs });
    }
}

/// The prompt, then `tokens` serial rounds of the graph: their mean milliseconds.
fn decode(e: *nemotron.Engine, prompt: []const u32) !f64 {
    var tok = try e.prefill(prompt, null, null);
    for (0..8) |_| tok = try e.step(tok, null);
    const t0 = check.now(e.io);
    for (0..tokens) |_| tok = try e.step(tok, null);
    return @as(f64, @floatFromInt(check.now(e.io) - t0)) / 1e6 / tokens;
}
