//! Prompt chunks over the decode kernels' 16 rows: mlx_lm's backbone launch for launch (tools/zig/prefill_forward.py).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const pk = @import("prefill_kernels.zig");
const pl = @import("prefill_launch.zig");
const mixers = @import("prefill_mixers.zig");
const Model = @import("model.zig").Model;

const Buffer = mtl.Buffer;
const At = pl.At;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// lane_qmm's rows at most: wider chunks take MLX's 4-bit matmuls on the checkpoint's layout.
pub const lane_rows = 128;

/// mlx_lm's SSD step.
pub const ssd_step = 256;

/// A 4-bit linear in MLX's layout: weight [N, K/8] u32, scales and biases [N, K/64] bf16.
pub const Raw = [3]At;

pub const Mamba = struct { in_proj: Raw, out_proj: Raw, conv_w: At, conv_b: At, dt_bias: At, d: At, a: At };
pub const Moe = struct { up: Raw, down: Raw };
pub const Attention = struct { q: Raw, k: Raw, v: Raw, o: Raw };
pub const Layer = union(cfg.Kind) { mamba: Mamba, moe: Moe, attention: Attention };

/// A chunk's activations at `rows` rows at most (names as prefill_forward.py's).
pub const Scratch = struct {
    h: Buffer,
    x: Buffer,
    out: Buffer,
    xs: Buffer,
    proj: Buffer,
    pad: Buffer,
    conv: Buffer,
    act: Buffer,
    dt: Buffer,
    dta: Buffer,
    dtx: Buffer,
    cb: Buffer,
    segin: Buffer,
    seg: Buffer,
    sur: Buffer,
    last: Buffer,
    y: Buffer,
    bh: Buffer,
    dd: Buffer,
    nxt: Buffer,
    ssm_a: Buffer,
    ssm_b: Buffer,
    cs: Buffer,
    e: Buffer,
    cf: Buffer,
    prev: Buffer,
    ya: Buffer,
    yb: Buffer,
    qkv: Buffer,
    logits: Buffer,
    parts: Buffer,
    ids: Buffer,
    wt: Buffer,
    counts: Buffer,
    starts: Buffer,
    offsets: Buffer,
    order: Buffer,
    sorted: Buffer,
    xids: Buffer,
    rowids: Buffer,
    inv: Buffer,
    xr: Buffer,
    y1: Buffer,
    y1r: Buffer,
    y2: Buffer,
    up: Buffer,
    upr: Buffer,
    sh: Buffer,

    fn init(device: mtl.Device, c: cfg.Config, rows: usize) !Scratch {
        @setEvalBranchQuota(20000);
        const d = c.hidden;
        const xd = c.inner();
        const cd = c.convDim();
        const h = c.mamba_heads;
        const pairs = rows * c.top_k;
        const sq = ssd_step * ssd_step;
        const state = h * c.mamba_head_dim * c.state * 4;
        const sizes = .{
            .{ "h", rows * d * 2 },                                .{ "x", rows * d * 2 },                             .{ "out", rows * d * 2 },
            .{ "xs", xd / 64 * lane_rows * 4 },                    .{ "proj", rows * c.projDim() * 2 },                .{ "pad", (rows + c.conv_kernel - 1) * cd * 2 },
            .{ "conv", rows * cd * 2 },                            .{ "act", rows * cd * 2 },                          .{ "dt", rows * h * 4 },
            .{ "dta", rows * h * 4 },                              .{ "dtx", rows * xd * 4 },                          .{ "cb", c.groups * sq * 2 },
            .{ "segin", h * sq * 4 },                              .{ "seg", h * sq * 4 },                             .{ "sur", h * sq * 4 },
            .{ "last", h * ssd_step * 4 },                         .{ "y", h * ssd_step * c.mamba_head_dim * 4 },      .{ "bh", h * c.state * ssd_step * 4 },
            .{ "dd", ssd_step * xd * 4 },                          .{ "nxt", state },                                  .{ "ssm_a", state },
            .{ "ssm_b", state },                                   .{ "cs", ssd_step * h * 4 },                        .{ "e", ssd_step * h * 4 },
            .{ "cf", ssd_step * c.groups * c.state * 4 },          .{ "prev", ssd_step * xd * 4 },                     .{ "ya", rows * xd * 2 },
            .{ "yb", rows * xd * 2 },                              .{ "qkv", rows * c.qkvDim() * 2 },                  .{ "logits", rows * c.experts * 2 },
            .{ "parts", @max(2 * rows * c.experts * 4, 1 << 21) }, .{ "ids", pairs * 4 },                              .{ "wt", pairs * 4 },
            .{ "counts", (pairs + 255) / 256 * c.experts * 4 },    .{ "starts", (pairs + 255) / 256 * c.experts * 4 }, .{ "offsets", c.experts * 4 },
            .{ "order", pairs * 4 },                               .{ "sorted", pairs * 4 },                           .{ "xids", pairs * 4 },
            .{ "rowids", pairs * 4 },                              .{ "inv", pairs * 4 },                              .{ "xr", pairs * d * 2 },
            .{ "y1", pairs * c.expert_width * 2 },                 .{ "y1r", pairs * c.expert_width * 2 },             .{ "y2", pairs * d * 2 },
            .{ "up", rows * c.shared_width * 2 },                  .{ "upr", rows * c.shared_width * 2 },              .{ "sh", rows * d * 2 },
        };
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (sizes, 0..) |f, i| {
            if (i < made) @field(s, f[0]).deinit();
        };
        inline for (sizes) |f| {
            @field(s, f[0]) = try device.buffer(@max(f[1], 64), opts);
            made += 1;
        }
        return s;
    }

    fn deinit(self: *Scratch) void {
        inline for (@typeInfo(Scratch).@"struct".field_names) |name| @field(self, name).deinit();
    }
};

/// Where one chunk runs: its encoder, its stream's caches, the pool slot its Mamba states go to.
pub const Chunk = struct {
    p: *const Prefill,
    e: *fwd.Enc,
    f: fwd.Forward,
    rows: usize,
    cache: *st.Cache,
    pool: *st.Pool,
    fresh: u32,

    pub fn launch(self: Chunk) pl.Launch {
        return .{ .k = self.p.k, .e = self.e };
    }

    /// The stream's caches hold earlier rows (Python's state is not None).
    pub fn carries(self: Chunk) bool {
        return self.cache.len > 0;
    }

    /// x [rows, K] times a projection into `out`: lane_qmm up to 128 rows, else MLX's 4-bit matmul.
    pub fn linear(self: Chunk, comptime key: []const u8, comptime sums: []const u8, lin: wts.Linear, raw: Raw, x: Buffer, out: Buffer) void {
        if (self.rows <= lane_rows) {
            self.f.xsum(self.e, sums, lin.k, x, 0, self.p.s.xs, self.rows);
            self.f.coop(self.e, key, lin, x, 0, self.p.s.xs, out, self.rows);
        } else self.launch().qmm(At.of(x), raw, At.of(out), At.of(self.p.s.parts), self.rows, lin.n, lin.k);
    }

    /// MLX's RMS norm of `rows` rows of `width` (tf_rms_norm_bf16).
    pub fn rms(self: Chunk, x: Buffer, w: At, width: usize, rows: usize, out: Buffer) void {
        const threads = 32 * (((width + 3) / 4 + 31) / 32);
        const e = self.e;
        e.pipe(self.p.k.get("tf_rms_norm_bf16"));
        e.buf(x, 0, 0);
        e.buf(w.b, w.off, 1);
        e.bytes(self.p.c.eps, 2);
        e.bytes(@as(i32, @intCast(width)), 3);
        e.buf(out, 0, 4);
        e.run(.{ threads * rows, 1, 1 }, .{ threads, 1, 1 });
    }

    /// out = a + b over `n` bf16 values (tf_add_bf16; out may be a or b).
    pub fn add(self: Chunk, a: Buffer, b: Buffer, n: usize, out: Buffer) void {
        const e = self.e;
        e.pipe(self.p.k.get("tf_add_bf16"));
        e.buf(a, 0, 0);
        e.buf(b, 0, 1);
        e.bytes(@as(i32, @intCast(n)), 2);
        e.buf(out, 0, 3);
        e.run(.{ n, 1, 1 }, .{ 256, 1, 1 });
    }

    /// relu(x)^2 over `n` bf16 values (tf_relu2_bf16).
    pub fn relu2(self: Chunk, x: Buffer, n: usize, out: Buffer) void {
        const e = self.e;
        e.pipe(self.p.k.get("tf_relu2_bf16"));
        e.buf(x, 0, 0);
        e.bytes(@as(i32, @intCast(n)), 1);
        e.buf(out, 0, 2);
        e.run(.{ n, 1, 1 }, .{ 256, 1, 1 });
    }
};

pub const Prefill = struct {
    c: cfg.Config,
    k: *const pk.Kernels,
    w: *const wts.Weights,
    layers: [cfg.max_layers]Layer = undefined,
    index: [cfg.max_layers]usize = undefined, // a layer's index among its kind's (its state's)
    s: Scratch,
    rows: usize,
    a: Buffer, // f32 [Mamba layers, H]: each Mamba layer's A (mamba_a of A_log)
    ones: Buffer, // bf16 [group width]: the gated norm's weight

    /// Chunks of up to `rows` rows on the model's prefill kernels and its checkpoint's own weight layout.
    pub fn init(m: *Model, rows: usize) !Prefill {
        const c = m.config;
        const group = c.inner() / c.groups;
        var p = Prefill{
            .c = c,
            .k = &m.prefill,
            .w = &m.weights,
            .s = try Scratch.init(m.device, c, rows),
            .rows = rows,
            .a = try m.device.buffer(c.count(.mamba) * c.mamba_heads * 4, opts),
            .ones = try m.device.buffer(group * 2, opts),
        };
        errdefer p.deinit();
        @memset(p.ones.slice(u16, group), 0x3f80);
        var counts: [3]usize = @splat(0);
        for (0..c.layers) |i| {
            p.index[i] = counts[@backingInt(c.kinds[i])];
            counts[@backingInt(c.kinds[i])] += 1;
            p.layers[i] = switch (c.kinds[i]) {
                .mamba => .{ .mamba = .{
                    .in_proj = try rawLinear(m, "backbone.layers.{d}.mixer.in_proj", i),
                    .out_proj = try rawLinear(m, "backbone.layers.{d}.mixer.out_proj", i),
                    .conv_w = try tensor(m, "backbone.layers.{d}.mixer.conv1d.weight", i),
                    .conv_b = try tensor(m, "backbone.layers.{d}.mixer.conv1d.bias", i),
                    .dt_bias = try tensor(m, "backbone.layers.{d}.mixer.dt_bias", i),
                    .d = try tensor(m, "backbone.layers.{d}.mixer.D", i),
                    .a = At.of(p.a).plus(p.index[i] * c.mamba_heads * 4),
                } },
                .moe => .{ .moe = .{
                    .up = try rawLinear(m, "backbone.layers.{d}.mixer.shared_experts.up_proj", i),
                    .down = try rawLinear(m, "backbone.layers.{d}.mixer.shared_experts.down_proj", i),
                } },
                .attention => .{ .attention = .{
                    .q = try rawLinear(m, "backbone.layers.{d}.mixer.q_proj", i),
                    .k = try rawLinear(m, "backbone.layers.{d}.mixer.k_proj", i),
                    .v = try rawLinear(m, "backbone.layers.{d}.mixer.v_proj", i),
                    .o = try rawLinear(m, "backbone.layers.{d}.mixer.o_proj", i),
                } },
            };
        }
        try p.mambaA(m);
        return p;
    }

    pub fn deinit(self: *Prefill) void {
        self.ones.deinit();
        self.a.deinit();
        self.s.deinit();
    }

    /// Each Mamba layer's A from its A_log, once (prefill_forward.py computes it at a layer's first chunk).
    fn mambaA(self: *Prefill, m: *Model) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        var e = fwd.Enc{ .e = cb.compute(.serial) };
        const l = pl.Launch{ .k = self.k, .e = &e };
        const h = self.c.mamba_heads;
        for (0..self.c.layers) |i| {
            if (self.c.kinds[i] != .mamba) continue;
            const a_log = try tensor(m, "backbone.layers.{d}.mixer.A_log", i);
            l.glue("mamba_a_bfloat16_t_int32_t_float", &.{a_log}, &.{h}, &.{self.layers[i].mamba.a}, .{ h, 1, 1 }, .{ h, 1, 1 });
        }
        e.e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |text| {
            std.log.err("mamba_a failed: {s}", .{text});
            return error.GpuFailed;
        }
    }

    /// One prompt chunk of `rows` ids at `ids` + `ids_off`: the caches advance, the final-normed rows land in s.x.
    pub fn chunk(self: *const Prefill, e: *fwd.Enc, f: fwd.Forward, cache: *st.Cache, ids: Buffer, ids_off: usize, rows: usize, fresh: u32) void {
        const x = self.context(e, f, cache, rows, fresh);
        self.embed(x, ids, ids_off);
        for (0..self.c.layers) |i| self.layer(x, i);
        self.final(x);
    }

    /// A chunk of `rows` rows of the stream with `cache`, its Mamba states going to pool slot `fresh`.
    pub fn context(self: *const Prefill, e: *fwd.Enc, f: fwd.Forward, cache: *st.Cache, rows: usize, fresh: u32) Chunk {
        std.debug.assert(rows > 0 and rows <= self.rows);
        return .{ .p = self, .e = e, .f = f, .rows = rows, .cache = cache, .pool = f.pool, .fresh = fresh };
    }

    /// The ids' embedding rows into s.h.
    pub fn embed(self: *const Prefill, x: Chunk, ids: Buffer, ids_off: usize) void {
        const d = self.c.hidden;
        const e = x.e;
        e.pipe(self.k.get("tf_embed_b4_g64"));
        e.buf(ids, ids_off, 0);
        for (self.w.embed, 1..) |t, i| e.buf(t.buffer, t.offset, i);
        e.bytes(@as(i32, @intCast(d)), 4);
        e.buf(self.s.h, 0, 5);
        e.run(.{ d / 2, x.rows, 1 }, .{ 256, 1, 1 });
    }

    /// Layer i on s.h: its input norm into s.x, its mixer into s.out, the residual add.
    pub fn layer(self: *const Prefill, x: Chunk, i: usize) void {
        const s = &self.s;
        const d = self.c.hidden;
        x.rms(s.h, tensorAt(self.w.norms[i]), d, x.rows, s.x);
        switch (self.layers[i]) {
            .mamba => |m| mixers.mamba(x, m, self.w.layers[i].mamba, self.index[i]),
            .attention => |a| mixers.attention(x, a, self.w.layers[i].attention, self.index[i]),
            .moe => |m| mixers.moe(x, m, self.w.layers[i].moe),
        }
        x.add(s.h, s.out, x.rows * d, s.h);
    }

    /// The final norm of s.h into s.x.
    pub fn final(self: *const Prefill, x: Chunk) void {
        x.rms(self.s.h, tensorAt(self.w.norm_f), self.c.hidden, x.rows, self.s.x);
    }
};

pub fn tensorAt(t: anytype) At {
    return .{ .b = t.buffer, .off = t.offset };
}

fn tensor(m: *Model, comptime fmt: []const u8, i: usize) !At {
    var name: [160]u8 = undefined;
    return tensorAt(try m.checkpoint.get(try std.fmt.bufPrint(&name, fmt, .{i})));
}

fn rawLinear(m: *Model, comptime base: []const u8, i: usize) !Raw {
    return .{ try tensor(m, base ++ ".weight", i), try tensor(m, base ++ ".scales", i), try tensor(m, base ++ ".biases", i) };
}
