//! MiniMax-H3-Image's decoder CUDA kernels under the contract, each against its float64 formula.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_h3_image_ref.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

/// The decoder's kernels: each symbol once, h3d_finish at both modes.
pub const Which = enum { post_quant, embed, rms_norm, layer_norm, linear, finish_bias, finish_scale, heads, attention, swiglu };

/// h3_image_decoder.cu's H3dShape, the trailing argument every decoder kernel takes.
const Shape64 = extern struct {
    tiles: u32 = 0,
    h: u32 = 0,
    w: u32 = 0,
    tokens: u32 = 0,
    rows: u32 = 0,
    cols: u32 = 0,
    m: u32 = 0,
    n: u32 = 0,
    k: u32 = 0,
    mode: u32 = 0,
    eps: f32 = 1e-5,
    scale: f32 = 0.125,
};

/// A dot of 24 (post_quant, embed).
fn dot24(_: c.Shape) usize {
    return 24;
}
/// The norms' row of 1024: 4 terms a thread's chain, then 5 warp levels and 8 partials serial.
fn normChain(_: c.Shape) usize {
    return 4 + 5 + 8;
}
/// heads' 64 squares: 2 terms a lane, then the 5 warp levels.
fn headChain(_: c.Shape) usize {
    return 2 + 5 + 2;
}
/// attention's key dots of 64 in a lane's chain, then the value sum over every key.
fn attnChain(s: c.Shape) usize {
    return 64 + s.n + 8;
}
fn none(_: c.Shape) usize {
    return 0;
}

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

/// h3d_linear's dot over k, in increasing K.
fn dotK(s: c.Shape) usize {
    return s.k;
}

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .post_quant => .{ .name = "cuda h3 image decoder h3d_post_quant", .accum = .f32, .depth = dot24, .ops = 1, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 256, .k = 24 }, .{ .rows = 4, .n = 256, .k = 24 }, .{ .rows = 2, .n = 64, .k = 24 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "the 24-term dot's bf16 output hides a bf16 sum's error" } },
        .embed => .{ .name = "cuda h3 image decoder h3d_embed", .accum = .f32, .depth = dot24, .ops = 1, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 256, .k = 24 }, .{ .rows = 2, .n = 64, .k = 24 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "the 24-term dot's bf16 output hides a bf16 sum's error" } },
        .rms_norm => .{ .name = "cuda h3 image decoder h3d_rms_norm", .accum = .f32, .depth = normChain, .ops = 8, .mids = 0, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 1024, .k = 1024 }, .{ .rows = 261, .n = 1024, .k = 1024 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = norm_why } },
        .layer_norm => .{ .name = "cuda h3 image decoder h3d_layer_norm", .accum = .f32, .depth = normChain, .ops = 10, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 1024, .k = 1024 }, .{ .rows = 261, .n = 1024, .k = 1024 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = norm_why } },
        .linear => .{ .name = "cuda h3 image decoder h3d_linear", .accum = .f32, .depth = dotK, .ops = 0, .out = .f32, .served = &.{ .{ .rows = 4, .n = 128, .k = 256 }, .{ .rows = 8, .n = 256, .k = 1024 }, .{ .rows = 8, .n = 128, .k = 4096 }, .{ .rows = 136, .n = 200, .k = 1000 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "each output sums K in increasing order with individually rounded fp32 multiplies and adds" } },
        .finish_bias => .{ .name = "cuda h3 image decoder h3d_finish bias", .accum = .none, .depth = none, .ops = 1, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 1024, .k = 1 }, .{ .rows = 16, .n = 768, .k = 1 } } },
        .finish_scale => .{ .name = "cuda h3 image decoder h3d_finish layerscale", .accum = .none, .depth = none, .ops = 1, .mids = 2, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 8, .n = 1024, .k = 1 }, .{ .rows = 16, .n = 768, .k = 1 } } },
        .heads => .{ .name = "cuda h3 image decoder h3d_heads", .accum = .f32, .depth = headChain, .ops = 8, .mids = 4, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 261, .k = 3072 }, .{ .rows = 2, .n = 64, .k = 3072 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = norm_why } },
        .attention => .{ .name = "cuda h3 image decoder h3d_attention", .accum = .f32, .depth = attnChain, .ops = 8, .mids = 0, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 261, .k = 64 }, .{ .rows = 2, .n = 64, .k = 64 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "softmax weights in [0, 1] with fp32 scores: the bf16 output hides a bf16 sum's error" } },
        .swiglu => .{ .name = "cuda h3 image decoder h3d_swiglu", .accum = .none, .depth = none, .ops = 4, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 4096, .k = 8192 }, .{ .rows = 65, .n = 4096, .k = 8192 } } },
    };
}

pub fn symbol(w: Which) [:0]const u8 {
    return switch (w) {
        .post_quant => "h3d_post_quant",
        .embed => "h3d_embed",
        .rms_norm => "h3d_rms_norm",
        .layer_norm => "h3d_layer_norm",
        .linear => "h3d_linear",
        .finish_bias, .finish_scale => "h3d_finish",
        .heads => "h3d_heads",
        .attention => "h3d_attention",
        .swiglu => "h3d_swiglu",
    };
}

pub const Dec = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    shape: c.Shape = undefined,
    arena: std.heap.ArenaAllocator,
    in: ref.In = .{},
    bufs: [8]u64 = undefined,
    nbufs: usize = 0,
    /// In-place buffers' starting bytes, re-uploaded before every launch (the same order as `bufs`).
    fresh: []const u8 = &.{},
    outs: usize = 0,
    /// Grid geometry the launch derives from the prepared inputs.
    p: Shape64 = .{},

    pub fn deinit(self: *Dec) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Dec) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Dec);

    /// `n` values uniform in [lo, hi), bf16-valued when `half`, some planted non-finite when `plant`.
    fn values(self: *Dec, n: usize, lo: f64, hi: f64, fill: g.Fill, rng: std.Random, half: bool, plant: bool) ![]f64 {
        const v = try self.arena.allocator().alloc(f64, n);
        for (v, 0..) |*x, i| {
            const r = lo + (hi - lo) * rng.float(f64);
            x.* = switch (fill) {
                .stress => @floatFromInt(@as(i64, @intFromFloat(@round(r * 8)))),
                else => r,
            };
            if (i % 61 == 7 and fill != .stress) x.* *= 6;
            x.* = if (half) ref.bf(x.*) else @as(f32, @floatCast(x.*));
        }
        if (fill == .nonfinite and plant) {
            const at = [_]usize{ n / 3, n / 2, n - 1 };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |at_, x| v[at_] = x;
        }
        return v;
    }

    /// Upload `v` as bf16 (half) or fp32, as an input; returns the slot pointer.
    fn upload(self: *Dec, v: []const f64, half: bool) !u64 {
        return self.bytes(try self.encode(v, half));
    }

    fn encode(self: *Dec, v: []const f64, half: bool) ![]const u8 {
        const al = self.arena.allocator();
        if (half) {
            const h = try al.alloc(u16, v.len);
            for (h, v) |*o, x| o.* = c.toBf16(@floatCast(x));
            return std.mem.sliceAsBytes(h);
        }
        const f = try al.alloc(f32, v.len);
        for (f, v) |*o, x| o.* = @floatCast(x);
        return std.mem.sliceAsBytes(f);
    }

    fn bytes(self: *Dec, b: []const u8) !u64 {
        self.bufs[self.nbufs] = try self.rig.input(b);
        self.nbufs += 1;
        return self.bufs[self.nbufs - 1];
    }

    fn ints(self: *Dec, v: []const u32) !u64 {
        return self.bytes(std.mem.sliceAsBytes(v));
    }

    fn output(self: *Dec, len: usize) !u64 {
        self.bufs[self.nbufs] = try self.rig.output(len);
        self.nbufs += 1;
        return self.bufs[self.nbufs - 1];
    }

    /// An in-place buffer: an output slot whose starting bytes are restored before every launch.
    fn inPlace(self: *Dec, v: []const f64, half: bool) !u64 {
        const bytes_ = try self.encode(v, half);
        self.bufs[self.nbufs] = try self.rig.output(bytes_.len);
        self.nbufs += 1;
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.bufs[self.nbufs - 1], .len = bytes_.len };
        try b.upload(0, bytes_);
        return self.bufs[self.nbufs - 1];
    }

    /// The latent tile origins: tile i crops at ((i % 3) * 4, (i % 5) * 4) of a 48x48 plane.
    fn origins(self: *Dec, tiles: usize) ![]u32 {
        const v = try self.arena.allocator().alloc(u32, 2 * tiles);
        for (0..tiles) |i| {
            v[2 * i] = @intCast(i % 3 * @as(usize, 4));
            v[2 * i + 1] = @intCast(i % 5 * @as(usize, 4));
        }
        return v;
    }

    pub fn prepare(self: *Dec, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.nbufs = 0;
        self.shape = s;
        self.p = .{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (self.which) {
            .post_quant => {
                // s.rows tiles of h*w cells over a 48x48 latent plane; cells = s.n needs h*w = s.n.
                if (s.n != 256 and s.n != 64) return false;
                const h: usize = if (s.n == 256) 16 else 8;
                const w: usize = s.n / h;
                const tiles = s.rows;
                const plane = 48 * 48;
                const lat = try self.values(24 * plane, -2, 2, fill, rng, false, true);
                const pw = try self.values(24 * 24, -0.5, 0.5, .random, rng, true, false);
                const pb = try self.values(24, -0.5, 0.5, .random, rng, true, false);
                self.in = .{ .a = lat, .b = pw, .d = pb, .rows = 48, .n = 48, .k = 24, .tiles = tiles, .h = h, .w = w, .ints = try self.origins(tiles) };
                _ = try self.upload(lat, false);
                _ = try self.ints(self.in.ints);
                _ = try self.upload(pw, true);
                _ = try self.upload(pb, true);
                _ = try self.output(tiles * s.n * 24 * 2);
                self.outs = tiles * s.n * 24;
                self.p = .{ .tiles = @intCast(tiles), .h = @intCast(h), .w = @intCast(w), .rows = 48, .cols = 48 };
            },
            .embed => {
                if (s.n != 256 and s.n != 64) return false;
                const h: usize = if (s.n == 256) 16 else 8;
                const w: usize = s.n / h;
                const cells = s.n;
                const tokens = cells + 5;
                const tiles = s.rows;
                const tok = try self.values(tiles * cells * 24, -2, 2, fill, rng, true, true);
                const ew = try self.values(1024 * 24, -0.5, 0.5, .random, rng, true, false);
                const eb = try self.values(1024, -0.5, 0.5, .random, rng, true, false);
                const regs = try self.values(4 * 1024, -1, 1, .random, rng, true, false);
                self.in = .{ .a = tok, .b = ew, .d = eb, .e = regs, .k = 24, .tiles = tiles, .tokens = tokens, .h = h, .w = w };
                _ = try self.upload(tok, true);
                _ = try self.upload(ew, true);
                _ = try self.upload(eb, true);
                _ = try self.upload(regs, true);
                _ = try self.output(tiles * tokens * 1024 * 2);
                self.outs = tiles * tokens * 1024;
                self.p = .{ .tiles = @intCast(tiles), .h = @intCast(h), .w = @intCast(w), .tokens = @intCast(tokens) };
            },
            .rms_norm, .layer_norm => {
                if (s.n != 1024 or s.k != 1024) return false;
                const x = try self.values(s.rows * 1024, -2, 2, fill, rng, true, true);
                // layer norm's stress rows mirror their first half: the mean is 0 and both fp32 passes are exact
                if (fill == .stress and self.which == .layer_norm) for (0..s.rows) |r| for (0..512) |i| {
                    x[r * 1024 + 512 + i] = -x[r * 1024 + i];
                };
                const gg = try self.values(1024, 0.5, 1.5, .random, rng, true, false);
                const bb = if (self.which == .layer_norm) try self.values(1024, -1, 1, .random, rng, true, false) else gg;
                self.in = .{ .a = x, .b = gg, .d = bb, .rows = s.rows, .n = 1024, .k = 1024, .eps = 1e-5 };
                _ = try self.upload(x, true);
                _ = try self.upload(gg, true);
                if (self.which == .layer_norm) _ = try self.upload(bb, true);
                _ = try self.output(s.rows * 1024 * 2);
                self.outs = s.rows * 1024;
                self.p = .{ .m = @intCast(s.rows), .eps = 1e-5 };
            },
            .linear => {
                if (s.k > 8192) return false;
                const x = try self.values(s.rows * s.k, -1, 1, fill, rng, true, true);
                const w = try self.values(s.n * s.k, -1, 1, .random, rng, true, false);
                self.in = .{ .a = x, .b = w, .rows = s.rows, .n = s.n, .k = s.k };
                _ = try self.upload(x, true);
                _ = try self.upload(w, true);
                _ = try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
                self.p = .{ .m = @intCast(s.rows), .n = @intCast(s.n), .k = @intCast(s.k) };
            },
            .finish_bias, .finish_scale => {
                const acc = try self.values(s.rows * s.n, -2, 2, fill, rng, false, true);
                const bias = try self.values(s.n, -0.5, 0.5, .random, rng, true, false);
                const scale = try self.values(s.n, 0.5, 1.5, .random, rng, true, false);
                const y0 = if (self.which == .finish_scale) try self.values(s.rows * s.n, -1, 1, .random, rng, true, false) else acc;
                const mode: u32 = if (self.which == .finish_scale) 1 else 0;
                self.in = .{ .a = acc, .b = bias, .d = scale, .e = y0, .rows = s.rows, .n = s.n, .k = 1, .mode = mode };
                _ = try self.upload(acc, false);
                _ = try self.upload(bias, true);
                _ = try self.upload(scale, true);
                if (self.which == .finish_scale) {
                    _ = try self.inPlace(y0, true);
                    self.fresh = try self.encode(y0, true);
                } else _ = try self.output(s.rows * s.n * 2);
                self.outs = s.rows * s.n;
                self.p = .{ .m = @intCast(s.rows), .n = @intCast(s.n), .mode = mode };
            },
            .heads => {
                // s.rows tiles of s.n tokens; the qkv product's rows [tiles*tokens, 3072].
                if (s.k != 3072 or s.n > 288) return false;
                const tiles = s.rows;
                const tokens = s.n;
                const rows = tiles * tokens;
                const qkv = try self.values(rows * 3072, 1, 2, fill, rng, true, true);
                const cs = try self.values(tokens * 32, 0.9, 1.0, .random, rng, true, false);
                const sn = try self.values(tokens * 32, 0, 0.01, .random, rng, true, false);
                self.in = .{ .a = qkv, .b = cs, .d = sn, .rows = rows, .k = 3072, .tiles = tiles, .tokens = tokens, .eps = 1e-5 };
                _ = try self.upload(qkv, true);
                _ = try self.upload(cs, true);
                _ = try self.upload(sn, true);
                for (0..3) |_| _ = try self.output(tiles * 16 * tokens * 64 * 2);
                self.outs = 3 * tiles * 16 * tokens * 64;
                self.p = .{ .tiles = @intCast(tiles), .tokens = @intCast(tokens), .eps = 1e-5 };
            },
            .attention => {
                if (s.k != 64 or s.n > 288) return false;
                const tiles = s.rows;
                const tokens = s.n;
                const n = tiles * 16 * tokens * 64;
                const q = try self.values(n, -2, 2, fill, rng, true, true);
                const k = try self.values(n, -2, 2, .random, rng, true, false);
                const v = try self.values(n, 0, 2, .random, rng, true, false);
                self.in = .{ .a = q, .b = k, .d = v, .rows = tiles * tokens, .n = s.n, .k = 64, .tiles = tiles, .tokens = tokens, .scale = 0.125 };
                _ = try self.upload(q, true);
                _ = try self.upload(k, true);
                _ = try self.upload(v, true);
                _ = try self.output(tiles * tokens * 1024 * 2);
                self.outs = tiles * tokens * 1024;
                self.p = .{ .tiles = @intCast(tiles), .tokens = @intCast(tokens), .scale = 0.125 };
            },
            .swiglu => {
                if (s.n != 4096 or s.k != 8192) return false;
                const wide = try self.values(s.rows * 8192, -4, 4, fill, rng, true, true);
                self.in = .{ .a = wide, .rows = s.rows, .n = 4096, .k = 8192 };
                _ = try self.upload(wide, true);
                _ = try self.output(s.rows * 4096 * 2);
                self.outs = s.rows * 4096;
                self.p = .{ .m = @intCast(s.rows) };
            },
        }
        return true;
    }

    pub fn run(self: *Dec) !void {
        if (self.fresh.len > 0) {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.bufs[self.nbufs - 1], .len = self.fresh.len };
            try b.upload(0, self.fresh);
        }
        const s = self.shape;
        const b = self.bufs;
        var a: cuda.Args = .{};
        switch (self.which) {
            .post_quant => {
                for (b[0..5]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = 1, .y = @intCast((self.p.tiles * self.p.h * self.p.w + 7) / 8), .z = 1 }, .{ .x = 32, .y = 8 }, &a);
            },
            .embed => {
                for (b[0..5]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = 4, .y = @intCast(self.p.tiles * self.p.tokens), .z = 1 }, .{ .x = 256 }, &a);
            },
            .rms_norm => {
                for (b[0..3]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = @intCast(s.rows) }, .{ .x = 256 }, &a);
            },
            .layer_norm => {
                for (b[0..4]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = @intCast(s.rows) }, .{ .x = 256 }, &a);
            },
            .linear => {
                for (b[0..3]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = @intCast((self.p.n + 127) / 128), .y = @intCast((self.p.m + 127) / 128), .z = 1 }, .{ .x = 256 }, &a);
            },
            .finish_bias, .finish_scale => {
                for (b[0..4]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = @intCast((self.p.n + 31) / 32), .y = @intCast((self.p.m + 7) / 8), .z = 1 }, .{ .x = 32, .y = 8 }, &a);
            },
            .heads => {
                for (b[0..6]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = @intCast(self.p.tiles * self.p.tokens), .y = 16, .z = 1 }, .{ .x = 32 }, &a);
            },
            .attention => {
                for (b[0..4]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = @intCast((self.p.tokens + 7) / 8), .y = 16, .z = @intCast(self.p.tiles) }, .{ .x = 256 }, &a);
            },
            .swiglu => {
                for (b[0..2]) |p| a.add(p);
                a.add(self.p);
                try self.go(.{ .x = 4096 / 32, .y = @intCast((self.p.m + 7) / 8), .z = 1 }, .{ .x = 32, .y = 8 }, &a);
            },
        }
        try self.rig.s.synchronize();
    }

    fn go(self: *Dec, grid: cuda.Dim3, block: cuda.Dim3, args: *cuda.Args) !void {
        try cuda.launch.launch(self.f, .{ .grid = grid, .block = block }, self.rig.s, args);
    }

    pub fn count(self: *Dec) usize {
        return self.outs;
    }

    fn compute(self: *Dec, s: ref.Sum, want: []f64, mag: []f64) !void {
        switch (self.which) {
            .post_quant => ref.postQuant(self.in, s, want, mag),
            .embed => ref.embed(self.in, s, want, mag),
            .rms_norm => ref.rmsNorm(self.in, s, want, mag),
            .layer_norm => ref.layerNorm(self.in, s, want, mag),
            .linear => ref.linear(self.in, s, want, mag),
            .finish_bias, .finish_scale => ref.finish(self.in, want, mag),
            .heads => ref.heads(self.in, s, want, mag),
            .attention => ref.attention(self.in, s, want, mag),
            .swiglu => ref.swiglu(self.in, want, mag),
        }
    }

    pub fn reference(self: *Dec, want: []f64, mag: []f64) !void {
        return self.compute(.{}, want, mag);
    }

    pub fn simulate(self: *Dec, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        return self.compute(.{ .acc = acc, .order = order }, out, mag);
    }
};
