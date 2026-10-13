//! One rank's share of a GLM-5.3 layer timed on the GPU, piece by piece, for a window of decode rows over a full cache.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = tf.glm_dsa;
const ex = tf.exlr8;
const a = std.heap.page_allocator;

const usage = "tf-glm-dsa-bench MODEL_DIR LAYER [ROWS 1] [KEYS 4096] [REPS 30]: rank 0 of 2's share, GPU ms a piece";

fn arg(argv: []const [:0]const u8, i: usize, d: u32) !u32 {
    return if (argv.len > i) try std.fmt.parseInt(u32, argv[i], 10) else d;
}

const Bench = struct {
    queue: mtl.Queue,
    reps: u32,

    /// The median GPU ms of `reps` command buffers, each encoding `f` once.
    fn time(b: Bench, name: []const u8, ctx: anytype, comptime f: anytype) !f64 {
        var ms: [256]f64 = undefined;
        for (0..b.reps) |i| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = b.queue.commandBuffer();
            const e = cb.compute(.serial);
            try f(ctx, e);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |m| {
                std.debug.print("{s}: GPU command failed: {s}\n", .{ name, m });
                return error.GpuFailure;
            }
            ms[i] = (cb.gpuEnd() - cb.gpuStart()) * 1000;
        }
        std.mem.sort(f64, ms[0..b.reps], {}, std.sort.asc(f64));
        const med = ms[b.reps / 2];
        std.debug.print("  {s:<22} {d:8.3} ms (min {d:.3})\n", .{ name, med, ms[0] });
        return med;
    }
};

const Ctx = struct {
    K: *g.layer.Kernels,
    L: *g.layer.Layer,
    s: *g.gpu.Scratch,
    c: *g.config.Config,
    rows: u32,
    last: u32,
};

fn attention(x: Ctx, e: mtl.ComputeEncoder) !void {
    try g.layer.attention(x.K, x.L, x.s, e, x.c, x.rows, x.last, null, .{ .buf = x.s.x }, .{ .buf = x.s.y });
}
fn route(x: Ctx, e: mtl.ComputeEncoder) !void {
    try g.layer.route(x.K, x.L, x.s, e, x.c, x.rows, .{ .buf = x.s.x }, null, null);
}
fn mlp(x: Ctx, e: mtl.ComputeEncoder) !void {
    if (x.L.w.mlp == .moe) {
        try g.layer.route(x.K, x.L, x.s, e, x.c, x.rows, .{ .buf = x.s.x }, null, null);
        try g.layer.expertsGpu(x.K, x.L, x.s, e, x.c, x.rows, .{ .buf = x.s.y });
    } else try g.layer.denseMlp(x.K, x.L, x.s, e, x.c, x.rows, .{ .buf = x.s.x }, .{ .buf = x.s.y });
}
const Lr = g.layer.Layer;
fn ref(b: mtl.Buffer) g.kernels.Ref {
    return .{ .buf = b };
}
fn xproj(x: Ctx, e: mtl.ComputeEncoder) !void {
    try tf.exlr8_dense_gpu.encodeProject(&x.K.dg, e, ref(x.s.x), x.L.k.dense(x.L.groups[Lr.x_g]), ref(x.s.proj), x.rows, x.c.hidden, g.weights.proj_width, true);
}
fn qb(x: Ctx, e: mtl.ComputeEncoder) !void {
    const nh = x.c.localHeads();
    try g.layer.linear(x.K, x.L, e, x.L.w.attn.q_b, x.L.groups[Lr.q_b_g], ref(x.s.q_resid), x.c.q_lora, ref(x.s.rot), ref(x.s.q), nh * x.c.qHead(), x.rows);
}
fn uv(x: Ctx, e: mtl.ComputeEncoder) !void {
    const nh = x.c.localHeads();
    try tf.exlr8_dense_gpu.encodeProject(&x.K.dg, e, ref(x.s.olat), x.L.k.dense(x.L.groups[Lr.uv_g]), ref(x.s.o), x.rows, nh * x.c.kv_lora, nh * x.c.v_dim, true);
}
fn oproj(x: Ctx, e: mtl.ComputeEncoder) !void {
    const nh = x.c.localHeads();
    try g.layer.linear(x.K, x.L, e, x.L.w.attn.o, x.L.groups[Lr.o_g], ref(x.s.o), nh * x.c.v_dim, ref(x.s.rot), ref(x.s.y), x.c.hidden, x.rows);
}
fn rproj(x: Ctx, e: mtl.ComputeEncoder) !void {
    try tf.exlr8_dense_gpu.encodeProject(&x.K.dg, e, ref(x.s.x), x.L.k.dense(x.L.groups[Lr.router_g]), ref(x.s.logits), x.rows, x.c.hidden, x.c.experts, true);
}
fn mla(x: Ctx, e: mtl.ComputeEncoder) !void {
    const nh = x.c.localHeads();
    const a_ = x.L.w.attn;
    const ma: g.kernels.MlaArgs = .{ .rows = x.rows, .heads = nh, .slot_keys = x.s.context, .proj_stride = g.weights.proj_width, .q_stride = nh * x.c.qHead(), .kvb = a_.kvb_w, .kvb_scale = a_.kvb_s };
    g.kernels.absorb(&x.K.k, e, ref(x.s.q), x.L.k.bytes, ref(x.s.rows), ma, ref(x.s.qlat), ref(x.s.qrope), ref(x.K.inv));
    e.barrier();
    g.kernels.attend(&x.K.k, e, ref(x.s.qlat), ref(x.s.qrope), ref(x.s.rows), ref(x.s.keys), ref(x.L.cache), ma, ref(x.s.partial), ref(x.s.olat));
}
fn both(x: Ctx, e: mtl.ComputeEncoder) !void {
    try attention(x, e);
    e.barrier();
    try mlp(x, e);
}

fn bf16(f: f32) u16 {
    return @truncate(@as(u32, @bitCast(f)) >> 16);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    const model = argv[1];
    const L_n = try arg(argv, 2, 40);
    const rows = try arg(argv, 3, 1);
    const keys = try arg(argv, 4, 4096);
    const reps = @min(try arg(argv, 5, 30), 256);
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/config.json", .{model}), a, .limited(1 << 20));
    var c = try g.config.parse(a, cfg_text);
    try g.config.split(&c, 0, 2);
    var manifest = try ex.manifest.Manifest.load(a, io, try std.fmt.allocPrint(a, "{s}/manifests/exlr8-manifest-home.json", .{model}));
    defer manifest.deinit();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var xl: ?ex.experts.Layer = if (c.isMoe(L_n)) try ex.experts.Layer.open(a, io, model, @intCast(L_n)) else null;
    defer if (xl) |*x| x.close(io);
    var pk = ex.pack.Pack.init(a);
    const w = try g.weights.load(a, io, model, &c, &manifest, L_n, &pk, if (xl) |*x| x else null);
    const context = keys + rows + 16;
    var L = try g.layer.Layer.init(a, device, w, &pk, 1, context, mtl.ResourceOptions.shared);
    pk.deinit();
    var s = try g.gpu.Scratch.init(device, &c, context);
    var K = try g.layer.Kernels.load(a, device, &c);
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (L.cache.slice(u16, @as(usize, context) * 576)) |*v| v.* = bf16(rnd.floatNorm(f32) * 0.1);
    if (L.index_cache) |ic| for (ic.slice(u16, @as(usize, context) * 128)) |*v| {
        v.* = bf16(rnd.floatNorm(f32) * 0.1);
    };
    for (s.x.slice(f32, rows * c.hidden), s.h.slice(f32, rows * c.hidden)) |*x, *h| {
        x.* = rnd.floatNorm(f32);
        h.* = x.*;
    }
    for (s.rows.slice(g.kernels.Row, rows), 0..) |*r, i| r.* = .{ .slot = 0, .pos = keys + @as(u32, @intCast(i)), .list = @intCast(i * c.i_topk) };
    if (!w.full) for (s.rows.slice(g.kernels.Row, rows), 0..) |*r, i| {
        r.nkeys = @min(r.pos + 1, c.i_topk);
        const list = s.keys.slice(u32, g.gpu.max_rows * c.i_topk)[i * c.i_topk ..][0..r.nkeys];
        for (list, 0..) |*k, j| k.* = @intCast(j + (r.pos + 1 - r.nkeys));
    };
    std.debug.print("layer {d} ({s}, {s} index), rank 0 of 2, {d} rows over {d} keys:\n", .{ L_n, if (c.isMoe(L_n)) "experts" else "dense MLP", if (w.full) "own" else "shared", rows, keys });
    std.debug.print("  formats: q_b {s}, o {s}\n", .{ @tagName(w.attn.q_b), @tagName(w.attn.o) });
    const b: Bench = .{ .queue = queue, .reps = reps };
    const x: Ctx = .{ .K = &K, .L = &L, .s = &s, .c = &c, .rows = rows, .last = keys + rows - 1 };
    const at = try b.time("attention", x, attention);
    _ = try b.time("  x projections", x, xproj);
    _ = try b.time("  q_b", x, qb);
    _ = try b.time("  absorb + attend", x, mla);
    _ = try b.time("  W_uv", x, uv);
    _ = try b.time("  o_proj", x, oproj);
    if (c.isMoe(L_n)) _ = try b.time("  router projection", x, rproj);
    if (c.isMoe(L_n)) _ = try b.time("router", x, route);
    const ml = try b.time(if (c.isMoe(L_n)) "router + experts" else "dense MLP", x, mlp);
    const all = try b.time("layer (no exchanges)", x, both);
    std.debug.print("  attention + MLP {d:.3} ms apart, {d:.3} ms in one buffer\n", .{ at + ml, all });
}
