//! A prompt window's experts (every expert once, on the tensor units) against the decode path's 16-row windows, by row.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = tf.glm_dsa;
const ex = tf.exlr8;
const a = std.heap.page_allocator;
const R = g.gpu.max_rows;

const usage = "tf-glm-dsa-prompt-check MODEL_DIR [LAYER 40] [ROWS 256]: rank 0 of 2's experts (GLM_ATTN=1: attention)";

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
    var xl: ?ex.experts.Layer = if (c.isMoe(L_n)) try ex.experts.Layer.open(a, io, model, @intCast(L_n)) else null;
    defer if (xl) |*x| x.close(io);
    var pk = ex.pack.Pack.init(a);
    const w = try g.weights.load(a, io, model, &c, &manifest, L_n, &pk, if (xl) |*x| x else null);
    const context = rows + 16;
    var L = try g.layer.Layer.init(a, device, w, &pk, 1, context, mtl.ResourceOptions.shared);
    var s = try g.gpu.Scratch.init(device, &c, context);
    var q = try g.gpu.Prompt.init(device, &c, context);
    var K = try g.layer.Kernels.load(a, device, &c);
    const hid = c.hidden;
    var prng = std.Random.DefaultPrng.init(5);
    for (q.x.slice(f32, @as(usize, rows) * hid)) |*v| v.* = prng.random().floatNorm(f32);
    const dec = try device.buffer(@as(usize, rows) * hid * 4, mtl.ResourceOptions.shared);
    const out = try device.buffer(@as(usize, rows) * hid * 4, mtl.ResourceOptions.shared);
    if (platform.c.getenv("GLM_ATTN") != null) return attention(&K, &L, &s, &q, queue, device, &c, rows, w.full);
    if (platform.c.getenv("GLM_NARROW") != null) return narrow(&K, &L, &q, queue, device, &c);
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
    try g.layer_prompt.experts(&K, &L, &q, e, &c, rows, .{ .buf = q.x }, .{ .buf = out });
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

/// Attention both ways over a fresh cache: the decode path 16 rows a window, then the prompt path for all rows at once.
fn attention(K: *g.layer.Kernels, L: *g.layer.Layer, s: *g.gpu.Scratch, q: *g.gpu.Prompt, queue: mtl.Queue, device: mtl.Device, c: *g.config.Config, rows: u32, full: bool) !void {
    const hid = c.hidden;
    const dec = try device.buffer(@as(usize, rows) * hid * 4, mtl.ResourceOptions.shared);
    const out = try device.buffer(@as(usize, rows) * hid * 4, mtl.ResourceOptions.shared);
    const table = q.rows.slice(g.kernels.Row, rows);
    for (table, 0..) |*row, r| {
        row.* = .{ .slot = 0, .pos = @intCast(r), .list = @intCast(r * c.i_topk), .nkeys = @intCast(r + 1) };
        // a shared layer reads the lists a full one left: every key up to the row's own
        for (q.keys.slice(u32, rows * c.i_topk)[r * c.i_topk ..][0 .. r + 1], 0..) |*k, j| k.* = @intCast(j);
    }
    var dec_ms: f64 = 0;
    var k: u32 = 0;
    while (k < rows) : (k += R) {
        const n = @min(R, rows - k);
        @memcpy(s.x.slice(f32, @as(usize, n) * hid), q.x.slice(f32, @as(usize, rows) * hid)[@as(usize, k) * hid ..][0 .. @as(usize, n) * hid]);
        for (s.rows.slice(g.kernels.Row, n), 0..) |*row, r| {
            row.* = .{ .slot = 0, .pos = k + @as(u32, @intCast(r)), .list = @intCast(r * c.i_topk), .nkeys = k + @as(u32, @intCast(r)) + 1 };
            if (!full) for (s.keys.slice(u32, R * c.i_topk)[r * c.i_topk ..][0..row.nkeys], 0..) |*key, j| {
                key.* = @intCast(j);
            };
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try g.layer.attention(K, L, s, e, c, n, k + n - 1, null, .{ .buf = s.x }, .{ .buf = dec, .off = @as(usize, k) * hid * 4 });
        e.end();
        dec_ms += try finish(cb);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try g.layer_prompt.attention(K, L, s, q, e, c, rows, 0, q.rows, .{ .buf = q.x }, .{ .buf = out });
    e.end();
    const pp_ms = try finish(cb);
    std.debug.print("attention ({s} index), {d} rows: decode path {d:.2} ms (16-row windows), prompt path {d:.2} ms\n", .{ if (full) "own" else "shared", rows, dec_ms, pp_ms });
    compare(dec.slice(f32, @as(usize, rows) * hid), out.slice(f32, @as(usize, rows) * hid), rows, hid);
}

fn compare(d: []const f32, o: []const f32, rows: u32, hid: u32) void {
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

/// The layer's dense groups for 1-4 rows: a row a simdgroup against four, byte for byte, and both timed.
fn narrow(K: *g.layer.Kernels, L: *g.layer.Layer, q: *g.gpu.Prompt, queue: mtl.Queue, device: mtl.Device, c: *g.config.Config) !void {
    const Lr = g.layer.Layer;
    const out = [2]mtl.Buffer{ try device.buffer(4 * 8192 * 4, mtl.ResourceOptions.shared), try device.buffer(4 * 8192 * 4, mtl.ResourceOptions.shared) };
    const nh = c.localHeads();
    const cases = [_]struct { name: []const u8, g: usize, in: u32, out: u32 }{
        .{ .name = "x projections", .g = Lr.x_g, .in = c.hidden, .out = g.weights.proj_width },
        .{ .name = "W_uv", .g = Lr.uv_g, .in = nh * c.kv_lora, .out = nh * c.v_dim },
        .{ .name = "router", .g = Lr.router_g, .in = c.hidden, .out = c.experts },
    };
    for (cases) |cs| {
        const grp = L.groups[cs.g];
        if (grp.count == 0) continue;
        for ([_]u32{ 1, 2, 4 }) |rows| {
            var ms: [2]f64 = .{ 0, 0 };
            for (0..2) |v| for (0..20) |_| {
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = queue.commandBuffer();
                const e = cb.compute(.serial);
                const t = if (v == 0) L.k.dense(grp) else L.k.denseFor(grp, rows);
                try tf.exlr8_dense_gpu.encodeProject(&K.dg, e, .{ .buf = q.x }, t, .{ .buf = out[v] }, rows, cs.in, cs.out, true);
                e.end();
                ms[v] += try finish(cb) / 20;
            };
            const same = std.mem.eql(u8, out[0].slice(u8, rows * cs.out * 4), out[1].slice(u8, rows * cs.out * 4));
            std.debug.print("  {s:<14} {d} rows: four a simdgroup {d:.3} ms, one {d:.3} ms, {s}\n", .{ cs.name, rows, ms[0], ms[1], if (same) "byte-equal" else "DIFFERENT" });
        }
    }
}
