//! Kimi K3 kernel bandwidth on this GPU: weight bytes each kernel streams over its GPU time, repeated in one buffer.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const gpu_mod = @import("gpu.zig");
const synth = @import("synth.zig");

const Gpu = gpu_mod.Gpu;
const Ref = k3.kernels.Ref;
const reps = 12;

/// GPU seconds per repetition of `encode` (after one warm-up buffer).
fn timed(g: *Gpu, ctx: anytype, comptime encode: fn (@TypeOf(ctx), mtl.ComputeEncoder) void) !f64 {
    var best: f64 = std.math.inf(f64);
    for (0..3) |pass| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..(if (pass == 0) 1 else reps)) |_| encode(ctx, e);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        if (pass > 0) best = @min(best, cb.gpuSeconds() / reps);
    }
    return best;
}

fn report(what: []const u8, rows: u32, bytes: f64, s: f64, peak: f64) void {
    const gbs = bytes / s / 1e9;
    std.debug.print("{s:<34} rows {d:>2}: {d:>8.1} us  {d:>6.1} GB/s  {d:>5.1}% of this GPU's read  {d:>5.1}% of 800 GB/s\n", .{ what, rows, s * 1e6, gbs, 100 * gbs / peak, 100 * gbs / 800 });
}

const Stream = struct { k: *const k3.kernels.Kernels, src: Ref, dst: Ref, n: u32 };

fn streamRead(s: Stream, e: mtl.ComputeEncoder) void {
    s.k.add(e, s.src, s.src.at(s.n * 2), s.dst, s.n);
}

const RowsCase = struct { k: *const k3.kernels.Kernels, x: Ref, w: Ref, y: Ref, part: Ref, a: k3.kernels.RowsArgs, glu: ?Ref = null, mma: bool };

fn rowsRun(c: RowsCase, e: mtl.ComputeEncoder) void {
    if (c.glu) |u| c.k.gluBy(e, c.x, c.w, u, c.y, c.part, c.a, c.mma) else c.k.rowsBy(e, c.x, c.w, c.y, c.part, false, c.a, c.mma);
}

const Peak = struct { k: *const k3.kernels.Kernels, out: Ref, groups: u32, steps: u32 };

fn peakRun(p: Peak, e: mtl.ComputeEncoder) void {
    e.setPipeline(p.k.mma_peak);
    e.setBuffer(p.out.buf, p.out.off, 0);
    e.setValue(p.steps, 1);
    e.dispatchGroups(mtl.Size.of(p.groups, 1, 1), mtl.Size.of(256, 1, 1));
}

const XpCase = struct { k: *const k3.kernels.Kernels, sc: *k3.round.Scratch, local: *k3.experts.Local, table: mtl.Buffer };

fn xpRun(c: XpCase, e: mtl.ComputeEncoder) void {
    c.local.experts().run(e, c.table, c.sc) catch unreachable;
}

const MlaCase = struct { k: *const k3.kernels.Kernels, sc: *k3.round.Scratch, a: k3.kernels.MlaArgs };

fn mlaRun(c: MlaCase, e: mtl.ComputeEncoder) void {
    c.k.mlaAttend(e, c.sc.ref("q"), c.sc.ref("qlat"), c.sc.ref("mrows"), c.sc.ref("partial"), c.a);
    c.k.mlaMerge(e, c.sc.ref("partial"), c.sc.ref("mrows"), c.sc.ref("olat"), c.a);
}

const KdaCase = struct { k: *const k3.kernels.Kernels, sc: *k3.round.Scratch, st: *const k3.state.State, w: k3.kernels.KdaWeights, c: *const k3.config.Config };

fn kdaRun(c: KdaCase, e: mtl.ComputeEncoder) void {
    const W = c.c.kdaWidth();
    c.k.kdaRound(e, c.sc.ref("proj"), c.sc.ref("f"), c.sc.ref("braw"), c.sc.ref("g2"), c.sc.ref("y"), c.sc.ref("kseg"), c.sc.nseg, c.st.kdaState(1), c.w, .{ .heads = c.c.kda_heads, .head0 = 0, .proj_stride = 3 * W, .gate_stride = W, .out_stride = W, .log_rows = c.st.limits.log_rows, .lower_bound = c.c.lower_bound, .eps = c.c.eps });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    const cfg = k3.config.Config{};
    std.debug.print("device: {s} (bandwidth here is this GPU's; the M3 Ultra column is 800 GB/s nominal)\n", .{g.device.name()});
    var src = synth.Synth{ .gpa = gpa, .device = g.device, .queue = g.queue, .k = &g.k };
    defer src.deinit();
    const H = cfg.hidden;
    const W = cfg.kdaWidth();

    const n: u32 = 64 * 1024 * 1024;
    const big = try src.source().get("bench.stream", .bf16, &.{3 * n});
    src.sync();
    const t_read = try timed(&g, Stream{ .k = &g.k, .src = big.ref, .dst = big.ref.at(@as(usize, n) * 4), .n = n }, streamRead);
    const peak = @as(f64, @floatFromInt(n)) * 6 / t_read / 1e9;
    std.debug.print("stream (two 128 MB reads, one 128 MB write): {d:.1} GB/s\n", .{peak});

    var sc = try k3.round.Scratch.init(g.device, &cfg, 256);
    defer sc.deinit();
    @memset(sc.xin.slice(u16, 256 * H), 0x3f80);
    @memset(sc.dact.slice(u16, 256 * cfg.dense_inter), 0x3f80);
    const pk = Peak{ .k = &g.k, .out = sc.ref("logits"), .groups = 1024, .steps = 512 };
    const t_peak = try timed(&g, pk, peakRun);
    const mma_peak = @as(f64, @floatFromInt(@as(u64, pk.groups) * 8 * pk.steps * 16 * 512 * 2)) / t_peak / 1e12;
    std.debug.print("simdgroup MMA peak (bf16 x bf16 -> fp32, registers only): {d:.1} TFLOPS\n", .{mma_peak});
    const shapes = [_]struct { name: []const u8, n: u32, k: u32, glu: bool }{
        .{ .name = "KDA q/k/v/g 12288x7168", .n = W, .k = H, .glu = false },
        .{ .name = "o_proj 7168x12288", .n = H, .k = W, .glu = false },
        .{ .name = "dense GLU 2x33792x7168", .n = cfg.dense_inter, .k = H, .glu = true },
        .{ .name = "lm_head 163840x7168", .n = cfg.vocab, .k = H, .glu = false },
    };
    std.debug.print("{s:<24} {s:>4}  {s:>21}  {s:>34}\n", .{ "bf16 projection", "rows", "scalar: us   GB/s", "MMA: us   GB/s  TFLOPS  % MMA peak" });
    for (shapes) |sh| {
        const wname = try std.fmt.allocPrint(gpa, "bench.{d}x{d}", .{ sh.n, sh.k });
        defer gpa.free(wname);
        const w = try src.source().get(wname, .bf16, &.{ sh.n, sh.k });
        const u = if (sh.glu) try src.source().get("bench.glu.up", .bf16, &.{ sh.n, sh.k }) else w;
        src.sync();
        const x = if (sh.k == W) sc.ref("dact") else sc.ref("xin");
        const y = sc.ref("logits");
        const bytes = @as(f64, @floatFromInt(@as(u64, sh.n) * sh.k * 2 * @as(u64, if (sh.glu) 2 else 1)));
        for ([_]u32{ 1, 2, 4, 8, 16, 32, 64, 128, 256 }) |r| {
            if (sh.n == cfg.vocab and r > 64) continue;
            const a = k3.kernels.RowsArgs{ .K = sh.k, .N = sh.n, .rows = r, .x_stride = sh.k, .y_stride = sh.n, .beta = cfg.situ_beta, .lin = cfg.situ_linear };
            const c = RowsCase{ .k = &g.k, .x = x, .w = w.ref, .y = y, .part = sc.ref("part"), .a = a, .glu = if (sh.glu) u.ref else null, .mma = false };
            const ts = if (r <= 64) try timed(&g, c, rowsRun) else std.math.nan(f64);
            var cm = c;
            cm.mma = true;
            const tm = if (r >= 8) try timed(&g, cm, rowsRun) else std.math.nan(f64);
            const tf = bytes / 2 * 2 * @as(f64, @floatFromInt(r)) / tm / 1e12;
            std.debug.print("{s:<24} {d:>4}  {d:>9.1} {d:>6.0}  {d:>9.1} {d:>6.0} {d:>7.2} {d:>9.0}%\n", .{ sh.name, r, ts * 1e6, bytes / ts / 1e9, tm * 1e6, bytes / tm / 1e9, tf, 100 * tf / mma_peak });
        }
    }
    _ = &peak;

    const l = try k3.weights.layer(&cfg, 1, src.source(), g.device, gpa, 0, cfg.experts);
    src.sync();
    var state = try k3.state.State.init(gpa, g.device, cfg, .{ .slots = 32, .log_rows = 16, .max_ctx = 64 }, &.{1}, cfg.kda_heads);
    defer state.deinit();
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer bufs.deinit(gpa);
    try state.buffers(&bufs, gpa);
    try bufs.append(gpa, l.mlp.moe.table);
    const set = try gpu_mod.resident(&g, &.{ src.bufs.items, bufs.items });
    defer {
        g.queue.removeResidencySet(set);
        set.deinit();
    }
    @memset(sc.lat.slice(u16, 256 * cfg.latent), 0x3f80);
    var local = k3.experts.Local.all(&g.k, &cfg);
    const per_expert = @as(f64, @floatFromInt(3 * (@as(u64, cfg.moe_inter) * cfg.latent / 2 + @as(u64, cfg.moe_inter) * cfg.latent / 32)));
    for ([_]bool{ false, true }) |shared| {
        for ([_]u32{ 1, 2, 4, 8, 16, 32, 64, 128, 256 }) |r| {
            if (shared and r > 16) continue;
            sc.rows = r;
            const ids = sc.tids.slice(u32, r * cfg.topk);
            var seen = std.StaticBitSet(1024).empty;
            for (0..r) |row| for (0..cfg.topk) |j| {
                const e: u32 = if (shared) @intCast(j * 53 % cfg.experts) else synth.mix(0xE7, @intCast(row * 977 + j * 13)) % cfg.experts;
                var pick = e;
                while (blk: {
                    for (ids[row * cfg.topk .. row * cfg.topk + j]) |o| if (o == pick) break :blk true;
                    break :blk false;
                }) pick = (pick + 1) % cfg.experts;
                ids[row * cfg.topk + j] = pick;
                seen.set(pick);
            };
            @memset(sc.tw.slice(f32, r * cfg.topk), 1.0 / 16.0);
            const s = try timed(&g, XpCase{ .k = &g.k, .sc = &sc, .local = &local, .table = l.mlp.moe.table }, xpRun);
            const name = if (shared) "mxfp4 experts: rows share 16" else "mxfp4 experts: 16 distinct a row";
            report(name, r, per_expert * @as(f64, @floatFromInt(seen.count())), s, peak);
        }
    }

    const kw = k3.kernels.KdaWeights{ .conv_q = k3.kernels.addr(l.attn.kda.conv_q.ref), .conv_k = k3.kernels.addr(l.attn.kda.conv_k.ref), .conv_v = k3.kernels.addr(l.attn.kda.conv_v.ref), .A_log = k3.kernels.addr(l.attn.kda.A_log.ref), .dt_bias = k3.kernels.addr(l.attn.kda.dt_bias.ref), .o_norm = k3.kernels.addr(l.attn.kda.o_norm.ref) };
    const st_bytes = 2.0 * @as(f64, @floatFromInt(@as(u64, cfg.kda_heads) * cfg.kda_dim * cfg.kda_dim * 4));
    for ([_]u32{ 1, 4, 8, 16 }) |r| {
        const segs = [_]k3.round.Segment{.{ .slot = 0, .rows = r, .commit = false }};
        var tokens: [16]u32 = @splat(0);
        state.streams[0].kept = r;
        try sc.load(&state, &segs, tokens[0..r]);
        const s = try timed(&g, KdaCase{ .k = &g.k, .sc = &sc, .st = &state, .w = kw, .c = &cfg }, kdaRun);
        std.debug.print("KDA layer, one stream: a {d:>2}-row window after replaying {d:>2} kept rows: {d:>7.1} us (state read+write {d:.1} MB)\n", .{ r, r, s * 1e6, st_bytes / 1e6 });
    }
    for ([_]u32{ 8, 32 }) |streams| {
        var segs: [32]k3.round.Segment = undefined;
        for (0..streams) |j| {
            segs[j] = .{ .slot = @intCast(j), .rows = 1, .commit = false };
            state.streams[j].kept = 1;
        }
        var tokens: [32]u32 = @splat(0);
        try sc.load(&state, segs[0..streams], tokens[0..streams]);
        const s = try timed(&g, KdaCase{ .k = &g.k, .sc = &sc, .st = &state, .w = kw, .c = &cfg }, kdaRun);
        const bytes = st_bytes * @as(f64, @floatFromInt(streams));
        std.debug.print("KDA layer, {d:>2} streams of one row each (one kept row replayed): {d:>7.1} us ({d:.1} MB of state, {d:.0} GB/s)\n", .{ streams, s * 1e6, bytes / 1e6, bytes / s / 1e9 });
    }
    var mst = try k3.state.State.init(gpa, g.device, cfg, .{ .slots = 32, .max_ctx = 4096 }, &.{3}, cfg.kda_heads);
    defer mst.deinit();
    const keys_row = 4096.0 * @as(f64, @floatFromInt(cfg.kv_lora + cfg.rope)) * 2;
    for ([_]u32{ 96, 24 }) |heads| {
        for ([_]u32{ 1, 8, 32 }) |r| {
            const mr = sc.mrows.slice(k3.kernels.MlaRow, r);
            for (mr, 0..) |*m, j| m.* = .{ .slot = @intCast(j), .pos = 4095 };
            const a = k3.kernels.MlaArgs{ .cache = mst.mlaCache(3), .slot_keys = 4096, .heads = heads, .head0 = 0, .rows = r, .q_stride = cfg.mla_heads * cfg.qHead(), .kv_stride = cfg.kv_lora + cfg.rope, .gate_stride = cfg.mla_heads * cfg.v_dim, .out_stride = cfg.mla_heads * cfg.v_dim, .eps = cfg.eps, .scale = 0.07 };
            const s = try timed(&g, MlaCase{ .k = &g.k, .sc = &sc, .a = a }, mlaRun);
            std.debug.print("MLA attention, {d:>2} heads, {d:>2} rows at 4,096 keys: {d:>7.1} us ({d:.0} GB/s of keys, each key read once a row)\n", .{ heads, r, s * 1e6, keys_row * @as(f64, @floatFromInt(r)) / s / 1e9 });
        }
    }
}
