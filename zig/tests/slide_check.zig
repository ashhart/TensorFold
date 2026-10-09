//! Sliding Weights' GPU step against its host reference on synthetic rows, then timed at Nemotron's shapes.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const nm = tf.nemotron;
const sg = nm.slide_gpu;
const ref = tf.slide_rule;
const affine4 = tf.affine4;
const At = nm.prefill_launch.At;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn config(dim: usize, width: usize, vocab: usize) nm.config.Config {
    return .{ .hidden = dim, .vocab = vocab, .layers = 1, .mamba_heads = 2, .mamba_head_dim = 32, .groups = 1, .state = 16, .conv_kernel = 4, .heads = 2, .kv_heads = 1, .head_dim = 32, .experts = 4, .top_k = 2, .expert_width = 64, .shared_width = width, .routed_scaling = 1, .eps = 1e-5 };
}

fn uniform(rng: std.Random, scale: f32) f32 {
    return (rng.float(f32) * 2 - 1) * scale;
}

/// A model's last pieces and one batch of rows, random, in the GPU's buffers and as the host reads them.
const Setup = struct {
    c: nm.config.Config,
    gpu: sg.Gpu,
    batch: sg.Batch,
    head: []f32, // the 4-bit head's values, as the GPU reads them
    norm: []f32,
    start: []f32,
    keys: []f32,
    codes: mtl.Buffer,
    norm_buf: mtl.Buffer,
    start_buf: mtl.Buffer,

    fn init(gpa: std.mem.Allocator, device: mtl.Device, k: *const nm.kernels.Kernels, pk: *const nm.prefill_kernels.Kernels, c: nm.config.Config, rows: usize, seed: u64) !Setup {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const d = c.hidden;
        const w = c.shared_width;
        const n = c.vocab * d;
        const groups = n / affine4.group;
        const head = try gpa.alloc(f32, n);
        for (head) |*v| v.* = uniform(rng, 0.1);
        const codes = try device.buffer(n / 2 + 4 * groups, opts);
        const words = codes.slice(u32, n / 8);
        const sb = @as([*]u16, @ptrCast(@alignCast(codes.contents() + n / 2)))[0 .. 2 * groups];
        affine4.quantize(head, words, sb[0..groups], sb[groups..]);
        affine4.dequantize(words, sb[0..groups], sb[groups..], head);
        const norm_buf = try device.buffer(d * 2, opts);
        const norm = try gpa.alloc(f32, d);
        for (norm, norm_buf.slice(u16, d)) |*v, *b| {
            b.* = affine4.bf16of(1 + uniform(rng, 0.5));
            v.* = affine4.f32of(b.*);
        }
        const start_buf = try device.buffer(d * w * 2, opts);
        const start = try gpa.alloc(f32, d * w);
        for (start, start_buf.slice(u16, d * w)) |*v, *b| {
            b.* = affine4.bf16of(uniform(rng, 0.004));
            v.* = affine4.f32of(b.*);
        }
        const head_at = [3]At{ .{ .b = codes }, .{ .b = codes, .off = n / 2 }, .{ .b = codes, .off = n / 2 + 2 * groups } };
        const gpu = try sg.Gpu.init(device, c, k, pk, head_at, .{ .b = norm_buf }, start_buf);
        var batch = try sg.Batch.init(device, c, (rows + sg.tile - 1) / sg.tile * sg.tile);
        batch.rows = rows;
        const keys = try gpa.alloc(f32, rows * w);
        for (batch.base.slice(f32, rows * d)) |*v| v.* = uniform(rng, 2);
        for (keys, batch.keys.slice(u16, rows * w)) |*v, *b| {
            const x = uniform(rng, 1);
            b.* = affine4.bf16of(x * x);
            v.* = affine4.f32of(b.*);
        }
        for (batch.targets.slice(u32, rows)) |*t| t.* = rng.uintLessThan(u32, @intCast(c.vocab));
        batch.seal(w, d);
        return .{ .c = c, .gpu = gpu, .batch = batch, .head = head, .norm = norm, .start = start, .keys = keys, .codes = codes, .norm_buf = norm_buf, .start_buf = start_buf };
    }
};

/// The head transposed, then `steps` forward and backward passes, each its own command buffer; GPU seconds a step.
fn run(queue: mtl.Queue, s: *Setup, steps: usize) !f64 {
    var total: f64 = 0;
    for (0..steps + 1) |i| {
        const cb = queue.commandBufferUnretained();
        var e: nm.forward.Enc = .{ .e = cb.compute(.serial) };
        if (i == 0) s.gpu.transposeHead(&e) else {
            s.gpu.forward(&e, &s.batch);
            s.gpu.backward(&e, &s.batch);
        }
        e.e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        if (i > 0) total += cb.gpuSeconds();
    }
    return total / @as(f64, @floatFromInt(@max(steps, 1)));
}

fn stage(queue: mtl.Queue, s: *Setup, which: usize) !void {
    const cb = queue.commandBufferUnretained();
    var e: nm.forward.Enc = .{ .e = cb.compute(.serial) };
    const g = &s.gpu;
    const b = &s.batch;
    const c = s.c;
    const n = b.padded();
    const l: nm.prefill_launch.Launch = .{ .k = g.pk, .e = &e };
    switch (which) {
        0 => g.transposeHead(&e),
        1 => sg.mm(l, At.of(b.keys), At.of(g.work), At.of(b.hd), n, c.hidden, c.shared_width, c.shared_width, c.shared_width, c.hidden),
        2 => l.qmm(At.of(b.hn), g.head, At.of(b.logits), At.of(b.logits), n, c.vocab, c.hidden),
        3 => g.forward(&e, b),
        4 => sg.mm(l, At.of(g.head_t), At.of(b.logits), At.of(b.dhn_t), c.hidden, n, c.vocab, c.vocab, c.vocab, n),
        5 => sg.mm(l, At.of(b.dh_t), At.of(b.keys_t), At.of(g.grad), c.hidden, c.shared_width, n, n, n, c.shared_width),
        else => g.backward(&e, b),
    }
    e.e.end();
    cb.commit();
    cb.wait();
    std.debug.print("stage {d}: {s}\n", .{ which, if (cb.failure()) |t| std.mem.span(t) else "ok" });
}

pub fn main() !void {
    const gpa = std.heap.c_allocator;
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const k = try nm.kernels.load(gpa, device);
    const pk = try nm.prefill_kernels.load(gpa, device);
    if (std.c.getenv("TF_SLIDE_QMM")) |spec| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(spec), ',');
        const M = try std.fmt.parseInt(usize, it.next().?, 10);
        const N = try std.fmt.parseInt(usize, it.next().?, 10);
        const K = try std.fmt.parseInt(usize, it.next().?, 10);
        const x = try device.buffer(M * K * 2, opts);
        const w = try device.buffer(N * K / 2, opts);
        const sc = try device.buffer(N * K / 64 * 2, opts);
        const bi = try device.buffer(N * K / 64 * 2, opts);
        const y = try device.buffer(M * N * 2, opts);
        const cb = queue.commandBufferUnretained();
        var e: nm.forward.Enc = .{ .e = cb.compute(.serial) };
        const l: nm.prefill_launch.Launch = .{ .k = &pk, .e = &e };
        l.qmm(At.of(x), .{ At.of(w), At.of(sc), At.of(bi) }, At.of(y), At.of(y), M, N, K);
        e.e.end();
        cb.commit();
        cb.wait();
        std.debug.print("qmm {d}x{d}x{d}: {s}\n", .{ M, N, K, if (cb.failure()) |t| std.mem.span(t) else "ok" });
        return;
    }
    if (std.c.getenv("TF_SLIDE_STAGES") != null) {
        var s = try Setup.init(gpa, device, &k, &pk, config(2688, 512, 1024), 40, 7);
        for (0..7) |i| try stage(queue, &s, i);
        return;
    }

    var small = try Setup.init(gpa, device, &k, &pk, config(2688, 512, 1024), 40, 7);
    small.gpu.rule = .{ .rate = 0.1, .bound = 1 };
    _ = try run(queue, &small, 1);
    const rows = small.batch.rows;
    var gpu_loss: f32 = 0;
    for (small.batch.stats.slice([2]f32, rows)) |st| gpu_loss += st[0];
    gpu_loss /= @floatFromInt(rows);
    const d = small.c.hidden;
    const w = small.c.shared_width;
    const delta = try gpa.dupe(f32, small.start);
    const grad = try gpa.alloc(f32, d * w);
    @memset(grad, 0);
    const host = try ref.measure(gpa, .{ .norm = small.norm, .w = small.head, .eps = small.c.eps }, delta, .{ .base = small.batch.base.slice(f32, rows * d), .keys = small.keys, .targets = small.batch.targets.slice(u32, rows) }, grad);
    if (!ref.step(delta, small.start, grad, .{ .rate = 0.1, .bound = 1 })) return error.HostStepRefused;
    var dot: f64 = 0;
    var gg: f64 = 0;
    var hh: f64 = 0;
    for (small.gpu.master.slice(f32, d * w), delta, small.start) |g, h, s0| {
        dot += (g - s0) * (h - s0);
        gg += (g - s0) * (g - s0);
        hh += (h - s0) * (h - s0);
    }
    const cosine = dot / @sqrt(gg * hh);
    const ratio = @sqrt(gg / hh);
    const loss_gap = @abs(gpu_loss - host.loss) / host.loss;
    std.debug.print("synthetic: loss GPU {d:.5} host {d:.5} (gap {d:.4}), step cosine {d:.5}, length ratio {d:.4}\n", .{ gpu_loss, host.loss, loss_gap, cosine, ratio });
    const matched = loss_gap < 0.01 and cosine > 0.99 and @abs(ratio - 1) < 0.02;

    var big = try Setup.init(gpa, device, &k, &pk, config(2688, 3712, 131072), 512, 9);
    const seconds = try run(queue, &big, 5);
    std.debug.print("Nemotron shapes, 512 rows: {d:.2} ms a step\n", .{seconds * 1e3});
    if (!matched) {
        std.debug.print("MISMATCH\n", .{});
        std.process.exit(1);
    }
    std.debug.print("ok\n", .{});
}
