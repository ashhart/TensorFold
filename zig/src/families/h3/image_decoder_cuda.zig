//! MiniMax-H3-Image's image decoder on CUDA, bf16 tiles in batches: K sequential fp32, h3d_finish at Metal's points.
const std = @import("std");
const report = @import("report.zig");
const cuda = @import("cuda");
pub const host = @import("image_decoder.zig");
pub const vae = @import("picture_vae.zig");
pub const reference = @import("image_decoder_reference.zig");
pub const host_decoder = @import("image_decoder_host.zig");

const W = host.width;
const F = host.ffn;

const image align(16) = @embedFile("fatbin_h3_image_decoder").*;
const kernel_names = [_][:0]const u8{ "h3d_post_quant", "h3d_embed", "h3d_rms_norm", "h3d_layer_norm", "h3d_linear", "h3d_finish", "h3d_heads", "h3d_attention", "h3d_swiglu" };

/// h3_image_decoder.cu's H3dShape.
const Shape = extern struct {
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
    eps: f32 = vae.eps,
    scale: f32 = host.scale,
};

pub const selftest = @import("image_decoder_cuda_selftest.zig");

pub const Timing = struct { wall: f64 = 0, gpu: f64 = 0 };

/// What one pass runs: the embedding, blocks [from, to), then norm_out and proj_out.
pub const Stages = struct { embed: bool = true, from: usize = 0, to: usize, final: bool = true };

/// Tiles a pass holds at most; `scratch` is sized at `prepare`.
pub const most_batch = 64;

pub const seconds = @import("clock.zig").seconds;

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    module: cuda.Module,
    functions: [kernel_names.len]cuda.Function,
    queue: cuda.Stream,
    plan: host.Plan,
    theta: f32,
    /// Every parameter in bf16 on the device, laid out as `plan` says.
    weights: cuda.DeviceBuffer,
    /// The raw latent [24, rows, cols] in fp32 that passes read their tiles from.
    latent: ?cuda.DeviceBuffer = null,
    rows: usize = 0,
    cols: usize = 0,
    /// One device buffer of scratch for `batch` tiles of h x w cells, `tokens` each.
    scratch: ?cuda.DeviceBuffer = null,
    /// The tiles the last pass held (what `stream` and `projected` read back).
    passed: usize = 0,
    batch: usize = 0,
    h: usize = 0,
    w: usize = 0,
    tokens: usize = 0,
    view: View = .{},
    /// A download's staging: `stream` or `projected` bytes, reused between reads.
    staged: std.ArrayListUnmanaged(u8) = .empty,

    /// A scratch view's byte offset inside `scratch`.
    const View = struct {
        x: usize = 0,
        xn: usize = 0,
        wide: usize = 0,
        q: usize = 0,
        k: usize = 0,
        v: usize = 0,
        o: usize = 0,
        act: usize = 0,
        out: usize = 0,
        tok: usize = 0,
        cos: usize = 0,
        sin: usize = 0,
        origins: usize = 0,
        acc: usize = 0,
    };

    /// `layers` blocks and `theta`, weights unfilled; heap-made so `DeviceBuffer.d` and driver pointers stay stable.
    pub fn init(gpa: std.mem.Allocator, layers: usize, theta: f32) !*Decoder {
        const plan = try host.Plan.init(layers);
        const d = try gpa.create(Decoder);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .driver = undefined, .ctx = undefined, .module = undefined, .functions = undefined, .queue = undefined, .plan = plan, .theta = theta, .weights = undefined };
        d.driver = try cuda.Driver.open();
        errdefer d.driver.close();
        d.ctx = try cuda.Context.shared(&d.driver);
        errdefer d.ctx.deinit();
        d.module = try cuda.Module.load(&d.driver, &image);
        errdefer d.module.unload();
        inline for (kernel_names, 0..) |name, i| d.functions[i] = try d.module.function(name);
        // the legacy default stream: copies on it stay ordered with this decoder's own launches
        d.queue = .{ .d = &d.driver, .handle = null };
        d.weights = try cuda.DeviceBuffer.alloc(&d.driver, plan.total() * 2);
        errdefer d.weights.free();
        return d;
    }

    pub fn drain(d: *Decoder) !void {
        try d.ctx.makeCurrent();
        try d.driver.synchronize();
    }

    /// A failed drain keeps the decoder live for a retry rather than freeing under pending work.
    pub fn deinit(d: *Decoder) void {
        d.tryDeinit() catch |e| {
            report.say("H3 image decoder retained at close: {t}", .{e});
        };
    }

    fn tryDeinit(d: *Decoder) !void {
        try d.drain();
        d.staged.deinit(d.gpa);
        if (d.scratch) |*s| s.free();
        if (d.latent) |*l| l.free();
        d.weights.free();
        d.module.unload();
        d.ctx.deinit();
        d.driver.close();
        d.gpa.destroy(d);
    }

    fn function(d: *const Decoder, comptime name: []const u8) cuda.Function {
        inline for (kernel_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return d.functions[i];
        @compileError("no kernel " ++ name);
    }

    fn launch(d: *const Decoder, comptime name: []const u8, grid: cuda.Dim3, block: cuda.Dim3, shared: u32, call: *cuda.Args) !void {
        try cuda.launch.launch(d.function(name), .{ .grid = grid, .block = block, .shared = shared }, d.queue, call);
    }

    /// The decoder's tensors of `path` (the VAE's safetensors), checked, each F32 rounded to bf16 in `plan`'s layout.
    pub fn load(d: *Decoder, io: std.Io, a: std.mem.Allocator, path: []const u8) !void {
        const bytes = try host_decoder.read(io, a, path, 8 << 30);
        const w = try host.weights(a, bytes, d.plan);
        const bits = try a.alloc(u16, d.plan.total());
        host.read(w, bits);
        try d.weights.upload(0, std.mem.sliceAsBytes(bits));
    }

    /// Synthetic weights from `seed` for the self-test: matrices by their fan-in, gains near 1, small biases.
    pub fn synthetic(d: *Decoder, a: std.mem.Allocator, seed: u64) !void {
        const bits = try a.alloc(u16, d.plan.total());
        try host.synthetic(a, d.plan, seed, bits);
        try d.weights.upload(0, std.mem.sliceAsBytes(bits));
    }

    /// The raw latent [24, rows, cols] in, and scratch for passes of up to `batch` tiles of h x w cells.
    pub fn prepare(d: *Decoder, raw: []const f32, rows: usize, cols: usize, h: usize, w: usize, batch: usize) !void {
        if (raw.len != try host.latentLength(rows, cols)) return error.H3BadLatentLength;
        if (h > rows or w > cols) return error.H3BadTiles;
        const n = try host.tokens(h, w);
        if (batch == 0 or batch > most_batch) return error.H3BadTiles;
        try d.ctx.makeCurrent();
        try d.drain();
        if (d.latent) |*l| l.free();
        d.latent = null;
        const lat = try cuda.DeviceBuffer.fromHost(&d.driver, std.mem.sliceAsBytes(raw));
        d.latent = lat;
        errdefer {
            if (d.latent) |*l| l.free();
            d.latent = null;
        }
        if (d.scratch) |*s| s.free();
        d.scratch = null;
        const m = batch * n;
        // x, xn, wide, q, k, v, o, act, out (bf16 rows); tok's cells; the rotary tables; origins; acc fp32
        const sizes = [_]usize{ m * W * 2, m * W * 2, m * 2 * F * 2, m * W * 2, m * W * 2, m * W * 2, m * W * 2, m * F * 2, m * host.patch * 2, batch * h * w * host.channels * 2, n * vae.rotated * 2, n * vae.rotated * 2, batch * 8, m * 2 * F * 4 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        const buf = try cuda.DeviceBuffer.alloc(&d.driver, total);
        d.scratch = buf;
        errdefer {
            if (d.scratch) |*s| s.free();
            d.scratch = null;
        }
        var at: usize = 0;
        var offs: [sizes.len]usize = undefined;
        for (&offs, sizes) |*o, s| {
            o.* = at;
            at += std.mem.alignForward(usize, s, 256);
        }
        d.view = .{ .x = offs[0], .xn = offs[1], .wide = offs[2], .q = offs[3], .k = offs[4], .v = offs[5], .o = offs[6], .act = offs[7], .out = offs[8], .tok = offs[9], .cos = offs[10], .sin = offs[11], .origins = offs[12], .acc = offs[13] };
        const cs = try d.gpa.alloc(u16, n * vae.rotated);
        defer d.gpa.free(cs);
        const sn = try d.gpa.alloc(u16, cs.len);
        defer d.gpa.free(sn);
        try host.rotary(h, w, d.theta, cs, sn);
        try buf.upload(d.view.cos, std.mem.sliceAsBytes(cs));
        try buf.upload(d.view.sin, std.mem.sliceAsBytes(sn));
        d.rows = rows;
        d.cols = cols;
        d.h = h;
        d.w = w;
        d.tokens = n;
        d.batch = batch;
        d.passed = 0;
    }

    /// `which` bytes of the scratch buffer to the host, reused staging; the caller copies before the next call.
    fn read(d: *Decoder, off: usize, len: usize) ![]u8 {
        if (d.staged.items.len < len) try d.staged.resize(d.gpa, len);
        try d.scratch.?.download(off, d.staged.items[0..len]);
        return d.staged.items[0..len];
    }

    /// The stream [tiles * tokens, 1,024] (bf16 bits) after the last pass, which the next pass reads.
    pub fn stream(d: *Decoder) ![]u16 {
        const bytes = try d.read(d.view.x, d.passed * d.tokens * W * 2);
        return @as([*]u16, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 2];
    }

    /// proj_out's rows [tiles * tokens, 768] (bf16 bits) after the last pass that ran the final stage.
    pub fn projected(d: *Decoder) ![]const u16 {
        const bytes = try d.read(d.view.out, d.passed * d.tokens * host.patch * 2);
        return @as([*]const u16, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 2];
    }

    fn dev(d: *const Decoder, off: usize) u64 {
        return d.scratch.?.ptr + off;
    }

    fn param(d: *const Decoder, offset: usize) u64 {
        return d.weights.ptr + offset * 2;
    }

    fn args(d: *const Decoder, ptrs: []const u64) cuda.Args {
        _ = d;
        var a: cuda.Args = .{};
        for (ptrs) |p| a.add(p);
        return a;
    }

    /// One pass: `stages` on the tiles whose first cells (row, column) are `origins`.
    pub fn pass(d: *Decoder, origins: []const [2]u32, stages: Stages) !Timing {
        if (d.scratch == null or origins.len == 0 or origins.len > d.batch) return error.H3BadTiles;
        if (stages.from > stages.to or stages.to > d.plan.layers) return error.H3BadTiles;
        for (origins) |o| {
            if (o[0] + d.h > d.rows or o[1] + d.w > d.cols) return error.H3BadTiles;
        }
        try d.ctx.makeCurrent();
        // Drain queued work before destroying events on error.
        var begin = try cuda.Event.init(&d.driver, true);
        defer begin.deinit();
        var end = try cuda.Event.init(&d.driver, true);
        defer end.deinit();
        errdefer cuda.completion.drain(&d.driver, d.queue, "H3 image decoder pass did not drain");
        try d.scratch.?.upload(d.view.origins, std.mem.sliceAsBytes(origins));
        const started = seconds();
        try begin.record(d.queue);
        const s: Shape = .{ .tiles = @intCast(origins.len), .h = @intCast(d.h), .w = @intCast(d.w), .tokens = @intCast(d.tokens), .rows = @intCast(d.rows), .cols = @intCast(d.cols), .m = @intCast(origins.len * d.tokens) };
        if (stages.embed) {
            var a = d.args(&.{ d.latent.?.ptr, d.dev(d.view.origins), d.param(d.plan.head(.post_w)), d.param(d.plan.head(.post_b)), d.dev(d.view.tok) });
            a.add(s);
            try d.launch("h3d_post_quant", .{ .x = 1, .y = @intCast((origins.len * d.h * d.w + 7) / 8), .z = 1 }, .{ .x = 32, .y = 8 }, 0, &a);
            var b = d.args(&.{ d.dev(d.view.tok), d.param(d.plan.head(.embed_w)), d.param(d.plan.head(.embed_b)), d.param(d.plan.head(.registers)), d.dev(d.view.x) });
            b.add(s);
            try d.launch("h3d_embed", .{ .x = 4, .y = s.m, .z = 1 }, .{ .x = 256, .y = 1 }, 0, &b);
        }
        for (stages.from..stages.to) |i| try d.layer(s, i);
        if (stages.final) {
            var a = d.args(&.{ d.dev(d.view.x), d.param(d.plan.tail(.norm_w)), d.param(d.plan.tail(.norm_b)), d.dev(d.view.xn) });
            a.add(s);
            try d.launch("h3d_layer_norm", .{ .x = s.m, .y = 1, .z = 1 }, .{ .x = 256 }, 0, &a);
            try d.gemm(s, d.view.xn, d.plan.tail(.proj_w), d.plan.tail(.proj_b), null, d.view.out, host.patch, W);
        }
        try end.record(d.queue);
        try end.synchronize();
        d.passed = origins.len;
        return .{ .wall = seconds() - started, .gpu = @as(f64, try cuda.Event.elapsedMs(begin, end)) / 1e3 };
    }

    /// y = bf16(x w^T + b) via an ordered fp32 sum, or with `scale` the LayerScale residual.
    fn gemm(d: *Decoder, s: Shape, x: usize, w: usize, b: usize, scale: ?usize, y: usize, n: usize, k: usize) !void {
        var p = s;
        p.n = @intCast(n);
        p.k = @intCast(k);
        p.mode = @intFromBool(scale != null);
        var product = d.args(&.{ d.dev(x), d.param(w), d.dev(d.view.acc) });
        product.add(p);
        try d.launch("h3d_linear", .{ .x = @intCast((n + 127) / 128), .y = @intCast((s.m + 127) / 128), .z = 1 }, .{ .x = 256 }, 0, &product);
        const grid: cuda.Dim3 = .{ .x = @intCast((n + 31) / 32), .y = @intCast((s.m + 7) / 8), .z = 1 };
        const block: cuda.Dim3 = .{ .x = 32, .y = 8 };
        var a = d.args(&.{ d.dev(d.view.acc), d.param(b), d.param(scale orelse b), d.dev(y) });
        a.add(p);
        try d.launch("h3d_finish", grid, block, 0, &a);
    }

    /// Block `i` on the pass's stream in place.
    fn layer(d: *Decoder, s: Shape, i: usize) !void {
        const p = d.plan;
        var a = d.args(&.{ d.dev(d.view.x), d.param(p.block(i, .norm1)), d.dev(d.view.xn) });
        a.add(s);
        try d.launch("h3d_rms_norm", .{ .x = s.m, .y = 1, .z = 1 }, .{ .x = 256 }, 0, &a);
        try d.gemm(s, d.view.xn, p.block(i, .qkv_w), p.block(i, .qkv_b), null, d.view.wide, 3 * W, W);
        var h = d.args(&.{ d.dev(d.view.wide), d.dev(d.view.cos), d.dev(d.view.sin), d.dev(d.view.q), d.dev(d.view.k), d.dev(d.view.v) });
        h.add(s);
        try d.launch("h3d_heads", .{ .x = s.m, .y = host.heads, .z = 1 }, .{ .x = 32 }, 0, &h);
        var t = d.args(&.{ d.dev(d.view.q), d.dev(d.view.k), d.dev(d.view.v), d.dev(d.view.o) });
        t.add(s);
        try d.launch("h3d_attention", .{ .x = @intCast((s.tokens + 7) / 8), .y = host.heads, .z = s.tiles }, .{ .x = 256 }, 0, &t);
        try d.gemm(s, d.view.o, p.block(i, .out_w), p.block(i, .out_b), p.block(i, .scale1), d.view.x, W, W);
        var r = d.args(&.{ d.dev(d.view.x), d.param(p.block(i, .norm2)), d.dev(d.view.xn) });
        r.add(s);
        try d.launch("h3d_rms_norm", .{ .x = s.m, .y = 1, .z = 1 }, .{ .x = 256 }, 0, &r);
        try d.gemm(s, d.view.xn, p.block(i, .w1_w), p.block(i, .w1_b), null, d.view.wide, 2 * F, W);
        var u = d.args(&.{ d.dev(d.view.wide), d.dev(d.view.act) });
        u.add(s);
        try d.launch("h3d_swiglu", .{ .x = @intCast(F / 32), .y = @intCast((s.m + 7) / 8), .z = 1 }, .{ .x = 32, .y = 8 }, 0, &u);
        try d.gemm(s, d.view.act, p.block(i, .w2_w), p.block(i, .w2_b), p.block(i, .scale2), d.view.x, W, F);
    }

    /// The decoded picture [3, 16 rows, 16 cols] (normalised) of a raw latent: tiles in batches, blended in fp32.
    pub fn decode(d: *Decoder, a: std.mem.Allocator, c: *const vae.Config, raw: []const f32, rows: usize, cols: usize, layers: usize, batch: usize, out: []f32) !Timing {
        const t = try host.tiles(a, c, rows, cols);
        if (layers > d.plan.layers) return error.H3BadTiles;
        const per = @min(batch, @min(most_batch, t.total()));
        try d.prepare(raw, rows, cols, t.h, t.w, per);
        var assembly = try vae.Assembler.init(a, t.rows, t.cols, 3, out);
        const tile = try a.alloc(f32, 3 * t.h * t.w * vae.cell_pixels * vae.cell_pixels);
        const origins = try a.alloc([2]u32, per);
        var took: Timing = .{};
        var k: usize = 0;
        while (k < t.total()) {
            const n = @min(per, t.total() - k);
            for (origins[0..n], 0..) |*o, i| {
                const at = t.at(k + i);
                o.* = .{ @intCast(at.row), @intCast(at.col) };
            }
            const one = try d.pass(origins[0..n], .{ .to = layers });
            took.wall += one.wall;
            took.gpu += one.gpu;
            const projected_rows = try d.projected();
            for (0..n) |i| {
                try host.unpatchify(projected_rows[i * d.tokens * host.patch ..][0 .. t.h * t.w * host.patch], t.h, t.w, tile);
                const at = t.at(k + i);
                try assembly.place(at.r, at.c, tile);
            }
            k += n;
        }
        return took;
    }
};
