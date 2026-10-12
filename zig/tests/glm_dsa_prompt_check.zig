//! A prompt window's experts (every expert once, on the tensor units) against the decode path's 16-row windows, by row.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = tf.glm_dsa;
const ex = tf.exlr8;
const a = std.heap.page_allocator;
const R = g.gpu.max_rows;

const usage = "tf-glm-dsa-prompt-check MODEL_DIR [LAYER 40] [ROWS 256]: rank 0 of 2's experts for ROWS random rows";

fn arg(argv: []const [:0]const u8, i: usize, d: u32) !u32 {
    return if (argv.len > i) try std.fmt.parseInt(u32, argv[i], 10) else d;
}

fn finish(cb: mtl.CommandBuffer) !f64 {
    cb.commit();
    cb.wait();
    if (cb.failure()) |m| {
        std.debug.print("GPU command failed: {s}\n", .{m});
        return error.GpuFailure;
    }
    return (cb.gpuEnd() - cb.gpuStart()) * 1000;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 2) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    const model = argv[1];
    const L_n = try arg(argv, 2, 40);
    const rows = @min(try arg(argv, 3, 256), g.gpu.prompt_rows);
    var c = try g.config.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/config.json", .{model}), a, .limited(1 << 20)));
    try g.config.split(&c, 0, 2);
    var manifest = try ex.manifest.Manifest.load(a, io, try std.fmt.allocPrint(a, "{s}/manifests/exlr8-manifest-home.json", .{model}));
    defer manifest.deinit();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    var xl = try ex.experts.Layer.open(a, io, model, @intCast(L_n));
    defer xl.close(io);
    var pk = ex.pack.Pack.init(a);
    const w = try g.weights.load(a, io, model, &c, &manifest, L_n, &pk, &xl);
    var L = try g.layer.Layer.init(a, device, w, &pk, 1, 64, mtl.ResourceOptions.shared);
    var s = try g.gpu.Scratch.init(device, &c, 64);
    var q = try g.gpu.Prompt.init(device, &c);
    var K = try g.layer.Kernels.load(a, device, &c);
    const hid = c.hidden;
    var prng = std.Random.DefaultPrng.init(5);
    for (q.x.slice(f32, @as(usize, rows) * hid)) |*v| v.* = prng.random().floatNorm(f32);
    const dec = try device.buffer(@as(usize, rows) * hid * 4, mtl.ResourceOptions.shared);
    const out = try device.buffer(@as(usize, rows) * hid * 4, mtl.ResourceOptions.shared);
    // the decode path: route and experts 16 rows at a time (each window's rows copied into the 16-row scratch)
    var dec_ms: f64 = 0;
    var k: u32 = 0;
    while (k < rows) : (k += R) {
        const n = @min(R, rows - k);
        @memcpy(s.x.slice(f32, @as(usize, n) * hid), q.x.slice(f32, @as(usize, rows) * hid)[@as(usize, k) * hid ..][0 .. @as(usize, n) * hid]);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try g.layer.route(&K, &L, &s, e, &c, n, .{ .buf = s.x }, null, null);
        try g.layer.expertsGpu(&K, &L, &s, e, &c, n, .{ .buf = dec, .off = @as(usize, k) * hid * 4 });
        e.end();
        dec_ms += try finish(cb);
        @memcpy(q.pick.slice(i32, @as(usize, rows) * c.topk)[@as(usize, k) * c.topk ..][0 .. @as(usize, n) * c.topk], s.route[0].slice(i32, @as(usize, n) * c.topk));
        @memcpy(q.wts8.slice(f32, @as(usize, rows) * c.topk)[@as(usize, k) * c.topk ..][0 .. @as(usize, n) * c.topk], s.route[1].slice(f32, @as(usize, n) * c.topk));
    }
    // the prompt path: the same picks, every expert once for all rows
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try g.layer.expertsPrompt(&K, &L, &q, e, &c, rows, .{ .buf = q.x }, .{ .buf = out });
    e.end();
    const pp_ms = try finish(cb);
    const tiles = q.dispatch.slice(u32, 8);
    std.debug.print("layer {d}, {d} rows: decode path {d:.2} ms (16-row windows), prompt path {d:.2} ms; {d} gate/up and {d} down jobs\n", .{ L_n, rows, dec_ms, pp_ms, tiles[0], tiles[4] });
    const d = dec.slice(f32, @as(usize, rows) * hid);
    const o = out.slice(f32, @as(usize, rows) * hid);
    var worst: f64 = 0;
    var worst_row: usize = 0;
    var all_sq: f64 = 0;
    var all_ref: f64 = 0;
    for (0..rows) |r| {
        var sq: f64 = 0;
        var ref_sq: f64 = 0;
        for (d[r * hid ..][0..hid], o[r * hid ..][0..hid]) |x, y| {
            sq += (@as(f64, y) - x) * (@as(f64, y) - x);
            ref_sq += @as(f64, x) * x;
        }
        all_sq += sq;
        all_ref += ref_sq;
        const rel = @sqrt(sq / @max(ref_sq, 1e-30));
        if (rel > worst) {
            worst = rel;
            worst_row = r;
        }
        if (r < 4 or r % 64 == 0) std.debug.print("  row {d}: rms(d)/rms {e:.3}\n", .{ r, rel });
    }
    std.debug.print("  all rows rms(d)/rms {e:.3}; worst row {d} at {e:.3}\n", .{ @sqrt(all_sq / all_ref), worst_row, worst });
}
