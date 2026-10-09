//! Sliding Weights' bounded step on the GPU: captured rows through the final norm and LM head, then the move.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const kern = @import("kernels.zig");
const pk = @import("prefill_kernels.zig");
const pl = @import("prefill_launch.zig");
const fwd = @import("forward.zig");

const Buffer = mtl.Buffer;
const At = pl.At;
const Enc = fwd.Enc;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// A batch's rows are padded to whole tiles of this many for the products.
pub const tile = 128;

/// The bounded rule: a step of rate over the gradient's norm (at least 1), every weight within bound of its anchor.
pub const Rule = struct { rate: f32 = 0.1, bound: f32 = 0.01 };

/// Rows to step on: final residuals without the learned change, keys into it, targets, and the step's scratch.
pub const Batch = struct {
    cap: usize,
    rows: usize = 0,
    base: Buffer, // f32 [cap, D]
    keys: Buffer, // bf16 [cap, W]
    keys_t: Buffer, // bf16 [W, padded]
    targets: Buffer, // u32 [cap]
    weights: Buffer, // f32 [cap]: 1 / rows a row, 0 past the rows
    hd: Buffer, // bf16 [cap, D]: keys times the change
    hn: Buffer, // bf16 [cap, D]
    inv: Buffer, // f32 [cap]
    logits: Buffer, // bf16 [cap, V]: the logits, then their gradient
    stats: Buffer, // f32 [cap, 2]: loss, target's probability
    dhn_t: Buffer, // bf16 [D, padded]
    dh_t: Buffer, // bf16 [D, padded]

    pub fn init(device: mtl.Device, c: cfg.Config, cap: usize) !Batch {
        const d = c.hidden;
        const w = c.shared_width;
        var b: Batch = undefined;
        b.cap = cap;
        b.rows = 0;
        const sizes = .{
            .{ "base", cap * d * 4 },         .{ "keys", cap * w * 2 }, .{ "keys_t", cap * w * 2 }, .{ "targets", cap * 4 },
            .{ "weights", cap * 4 },          .{ "hd", cap * d * 2 },   .{ "hn", cap * d * 2 },     .{ "inv", cap * 4 },
            .{ "logits", cap * c.vocab * 2 }, .{ "stats", cap * 8 },    .{ "dhn_t", cap * d * 2 },  .{ "dh_t", cap * d * 2 },
        };
        var made: usize = 0;
        errdefer inline for (sizes, 0..) |f, i| {
            if (i < made) @field(b, f[0]).deinit();
        };
        inline for (sizes) |f| {
            @field(b, f[0]) = try device.buffer(f[1], opts);
            made += 1;
        }
        return b;
    }

    pub fn deinit(b: *Batch) void {
        inline for (.{ "base", "keys", "keys_t", "targets", "weights", "hd", "hn", "inv", "logits", "stats", "dhn_t", "dh_t" }) |f| @field(b, f).deinit();
    }

    pub fn padded(b: Batch) usize {
        return (b.rows + tile - 1) / tile * tile;
    }

    /// Weights 1 / rows on the rows and 0 past them, keys_t from keys, and zeroed padding: call after filling `rows`.
    pub fn seal(b: *Batch, width: usize, dim: usize) void {
        const n = b.padded();
        std.debug.assert(n <= b.cap);
        @memset(b.base.slice(f32, n * dim)[b.rows * dim ..], 0);
        @memset(b.keys.slice(u16, n * width)[b.rows * width ..], 0);
        @memset(b.targets.slice(u32, n)[b.rows..], 0);
        const wt = b.weights.slice(f32, n);
        @memset(wt[0..b.rows], 1 / @as(f32, @floatFromInt(@max(b.rows, 1))));
        @memset(wt[b.rows..], 0);
        const keys = b.keys.slice(u16, n * width);
        const kt = b.keys_t.slice(u16, n * width);
        for (0..n) |r| for (0..width) |i| {
            kt[i * n + r] = keys[r * width + i];
        };
    }
};

/// The learned change with its anchor and the step's own buffers, the LM head transposed for the backward product.
pub const Gpu = struct {
    c: cfg.Config,
    k: *const kern.Kernels,
    pk: *const pk.Kernels,
    head: [3]At, // the 4-bit LM head as the checkpoint keeps it
    norm: At, // the final norm's bf16 weight
    rule: Rule = .{},
    head_t: Buffer, // bf16 [D, V]
    anchor: Buffer, // bf16 [D, W]: the change as loaded; every step stays within bound of it
    master: Buffer, // f32 [D, W]: the change
    work: Buffer, // bf16 [D, W]: the change as the products read it
    grad: Buffer, // bf16 [D, W]
    part: Buffer, // f32 [parts]
    scale: Buffer, // f32 [2]: the step size (NaN: refused) and the gradient's squared norm

    const parts = 1024;

    /// Buffers for a change starting at `start` (bf16 [D, W]; null: zero); encode `transposeHead` before stepping.
    pub fn init(device: mtl.Device, c: cfg.Config, k: *const kern.Kernels, prefill: *const pk.Kernels, head: [3]At, norm: At, start: ?Buffer) !Gpu {
        const n = c.hidden * c.shared_width;
        var g: Gpu = .{ .c = c, .k = k, .pk = prefill, .head = head, .norm = norm, .head_t = undefined, .anchor = undefined, .master = undefined, .work = undefined, .grad = undefined, .part = undefined, .scale = undefined };
        const sizes = .{ .{ "head_t", c.hidden * c.vocab * 2 }, .{ "anchor", n * 2 }, .{ "master", n * 4 }, .{ "work", n * 2 }, .{ "grad", n * 2 }, .{ "part", parts * 4 }, .{ "scale", 16 } };
        var made: usize = 0;
        errdefer inline for (sizes, 0..) |f, i| {
            if (i < made) @field(g, f[0]).deinit();
        };
        inline for (sizes) |f| {
            @field(g, f[0]) = try device.buffer(f[1], opts);
            made += 1;
        }
        const anchor = g.anchor.slice(u16, n);
        if (start) |s| @memcpy(anchor, s.slice(u16, n)) else @memset(anchor, 0);
        g.reset();
        return g;
    }

    pub fn deinit(g: *Gpu) void {
        inline for (.{ "head_t", "anchor", "master", "work", "grad", "part", "scale" }) |f| @field(g, f).deinit();
    }

    /// The change back to `to` (bf16 [D, W]; null: the anchor), dropping whatever steps moved since.
    pub fn resetTo(g: *Gpu, to: ?Buffer) void {
        const n = g.c.hidden * g.c.shared_width;
        const src = (to orelse g.anchor).slice(u16, n);
        @memcpy(g.work.slice(u16, n), src);
        for (g.master.slice(f32, n), src) |*m, b| m.* = @bitCast(@as(u32, b) << 16);
    }

    fn reset(g: *Gpu) void {
        g.resetTo(null);
    }

    /// head_t = the 4-bit LM head dequantized and transposed.
    pub fn transposeHead(g: *const Gpu, e: *Enc) void {
        e.pipe(g.k.get("tf_slide_head_t"));
        for (g.head, 0..) |t, i| e.buf(t.b, t.off, i);
        e.buf(g.head_t, 0, 3);
        e.run(.{ g.c.vocab, g.c.hidden, 1 }, .{ 256, 1, 1 });
    }

    /// base [rows] = h - keys `change`^T for freshly captured rows (bf16 h and keys; `change` null: none applied).
    pub fn settle(g: *const Gpu, e: *Enc, h: At, keys: At, hd: At, base: At, rows: usize, change: ?Buffer) void {
        const d = g.c.hidden;
        if (change) |dw| {
            mm(.{ .k = g.pk, .e = e }, keys, At.of(dw), hd, rows, d, g.c.shared_width, g.c.shared_width, g.c.shared_width, d);
        } else @memset(@as([*]u16, @ptrCast(@alignCast(hd.b.contents() + hd.off)))[0 .. rows * d], 0);
        e.pipe(g.k.get("tf_slide_sub"));
        e.buf(h.b, h.off, 0);
        e.buf(hd.b, hd.off, 1);
        e.buf(base.b, base.off, 2);
        e.run(.{ rows * d, 1, 1 }, .{ 256, 1, 1 });
    }

    /// The batch's loss and target probabilities into stats, and in its logits the loss's gradient where weighted.
    pub fn forward(g: *const Gpu, e: *Enc, b: *const Batch) void {
        const c = g.c;
        const n = b.padded();
        const l: pl.Launch = .{ .k = g.pk, .e = e };
        mm(l, At.of(b.keys), At.of(g.work), At.of(b.hd), n, c.hidden, c.shared_width, c.shared_width, c.shared_width, c.hidden);
        e.pipe(g.k.get("tf_slide_norm"));
        e.buf(b.base, 0, 0);
        e.buf(b.hd, 0, 1);
        e.buf(g.norm.b, g.norm.off, 2);
        e.buf(b.hn, 0, 3);
        e.buf(b.inv, 0, 4);
        e.bytes(@as(u32, @intCast(c.hidden)), 5);
        e.bytes(c.eps, 6);
        e.run(.{ 256 * n, 1, 1 }, .{ 256, 1, 1 });
        std.debug.assert(pl.splitkParts(n, c.vocab, c.hidden) == 1);
        l.qmm(At.of(b.hn), g.head, At.of(b.logits), At.of(b.logits), n, c.vocab, c.hidden);
        e.pipe(g.k.get("tf_slide_softmax"));
        e.buf(b.logits, 0, 0);
        e.buf(b.targets, 0, 1);
        e.buf(b.weights, 0, 2);
        e.buf(b.stats, 0, 3);
        e.bytes(@as(u32, @intCast(c.vocab)), 4);
        e.run(.{ 1024 * n, 1, 1 }, .{ 1024, 1, 1 });
    }

    /// After `forward` on the same batch: the gradient of the change, then one bounded step of master and work.
    pub fn backward(g: *const Gpu, e: *Enc, b: *const Batch) void {
        const c = g.c;
        const d = c.hidden;
        const w = c.shared_width;
        const n = b.padded();
        const l: pl.Launch = .{ .k = g.pk, .e = e };
        mm(l, At.of(g.head_t), At.of(b.logits), At.of(b.dhn_t), d, n, c.vocab, c.vocab, c.vocab, n);
        e.pipe(g.k.get("tf_slide_norm_back"));
        e.buf(b.dhn_t, 0, 0);
        e.buf(b.base, 0, 1);
        e.buf(b.hd, 0, 2);
        e.buf(g.norm.b, g.norm.off, 3);
        e.buf(b.inv, 0, 4);
        e.buf(b.dh_t, 0, 5);
        e.bytes([2]u32{ @intCast(d), @intCast(n) }, 6);
        e.run(.{ 256 * n, 1, 1 }, .{ 256, 1, 1 });
        mm(l, At.of(b.dh_t), At.of(b.keys_t), At.of(g.grad), d, w, n, n, n, w);
        e.pipe(g.k.get("tf_slide_sumsq"));
        e.buf(g.grad, 0, 0);
        e.buf(g.part, 0, 1);
        e.bytes(@as(u32, @intCast(d * w)), 2);
        e.run(.{ 256 * parts, 1, 1 }, .{ 256, 1, 1 });
        e.pipe(g.k.get("tf_slide_scale"));
        e.buf(g.part, 0, 0);
        e.buf(g.scale, 0, 1);
        e.bytes(@as(u32, parts), 2);
        e.bytes(g.rule.rate, 3);
        e.run(.{ 256, 1, 1 }, .{ 256, 1, 1 });
        e.pipe(g.k.get("tf_slide_step"));
        e.buf(g.master, 0, 0);
        e.buf(g.work, 0, 1);
        e.buf(g.anchor, 0, 2);
        e.buf(g.grad, 0, 3);
        e.buf(g.scale, 0, 4);
        e.bytes(g.rule.bound, 5);
        e.run(.{ d * w, 1, 1 }, .{ 256, 1, 1 });
    }

    /// Whether the last step moved: false when the gradient was not finite (read after its command buffer lands).
    pub fn stepped(g: *const Gpu) bool {
        return std.math.isFinite(g.scale.slice(f32, 2)[0]);
    }
};

/// d [m, n] (row stride ldd) = a [m, k] times b [n, k]^T, bf16 on the tensor units (MLX's NAX GEMM, any k).
pub fn mm(l: pl.Launch, a: At, b: At, d: At, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldd: usize) void {
    const whole = m % 64 == 0 and n % 128 == 0;
    const name = if (whole) "custom_kernel_tf_gemm_nax_bf16_n_t_t_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t" else "custom_kernel_tf_gemm_nax_bf16_n_t_n_n_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";
    const tn = (n + 127) / 128;
    const tm = (m + 63) / 64;
    const p = [_]usize{ m, n, k, lda, ldb, ldd, tn, tm, 2, k / 256, 0, 0, m * n, 0, 0 };
    l.go(name, &.{ a, b }, &p, 16, null, &.{d}, .{ (tn << 2) * 32, (tm + 3) / 4 * 4, 2 }, .{ 32, 4, 2 });
}
