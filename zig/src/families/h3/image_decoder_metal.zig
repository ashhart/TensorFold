//! MiniMax-H3-Image's image decoder on Metal: the VAE's 2D ViT decoder in bf16 on any Apple GPU, tiles in batches.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");
const rounding = @import("bf16_round");
const kernels = @import("h3_image_decoder_kernels");
pub const host = @import("image_decoder.zig");
pub const vae = @import("picture_vae.zig");
pub const reference = @import("image_decoder_reference.zig");

const W = host.width;
const F = host.ffn;

pub const Kernel = enum { post_quant, embed, rms_norm, layer_norm, gemm, heads, attention, swiglu };
const kernel_names = @typeInfo(Kernel).@"enum".field_names;

/// image_decoder.metal's H3dShape.
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

pub const Timing = struct { wall: f64 = 0, gpu: f64 = 0 };

/// What one pass runs: the embedding, blocks [from, to), then norm_out and proj_out.
pub const Stages = struct { embed: bool = true, from: usize = 0, to: usize, final: bool = true };

/// Tiles a pass holds at most.
pub const most_batch = 64;

const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// A pass's buffers: the stream, normed rows, q | k | v (and later w1's rows), heads, attention, FFN, outputs.
const Views = struct { x: Ref, xn: Ref, wide: Ref, q: Ref, k: Ref, v: Ref, o: Ref, act: Ref, out: Ref, tok: Ref, cos: Ref, sin: Ref, origins: Ref };

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    pipes: [kernel_names.len]mtl.Pipeline,
    plan: host.Plan,
    theta: f32,
    /// Every parameter in bf16, laid out as `plan` says.
    weights: mtl.Buffer,
    /// The raw latent [24, rows, cols] in fp32 that passes read their tiles from.
    latent: ?mtl.Buffer = null,
    rows: usize = 0,
    cols: usize = 0,
    /// Scratch for `batch` tiles of h x w cells, `tokens` each.
    scratch: ?mtl.Buffer = null,
    batch: usize = 0,
    h: usize = 0,
    w: usize = 0,
    tokens: usize = 0,
    views: Views = undefined,

    /// `layers` blocks and the rotary `theta`, the kernels compiled from source, the weights' buffer unfilled.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, layers: usize, theta: f32) !*Decoder {
        const plan = try host.Plan.init(layers);
        const d = try gpa.create(Decoder);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .device = device, .queue = try device.queue(), .lib = undefined, .pipes = undefined, .plan = plan, .theta = theta, .weights = undefined };
        errdefer d.queue.deinit();
        d.lib = try mtl.Library.fromSource(device, kernels.source, mtl.CompileOptions.mlx());
        errdefer d.lib.deinit();
        var made: usize = 0;
        errdefer for (d.pipes[0..made]) |p| p.deinit();
        inline for (kernel_names, 0..) |name, i| {
            d.pipes[i] = try mtl.Pipeline.init(device, d.lib, "h3d_" ++ name, false);
            made += 1;
        }
        d.weights = try device.buffer(plan.total() * 2, mtl.ResourceOptions.shared);
        return d;
    }

    pub fn deinit(d: *Decoder) void {
        if (d.scratch) |s| s.deinit();
        if (d.latent) |l| l.deinit();
        d.weights.deinit();
        for (d.pipes) |p| p.deinit();
        d.lib.deinit();
        d.queue.deinit();
        d.gpa.destroy(d);
    }

    /// The weights' bf16 bits on the host (shared memory).
    pub fn params(d: *const Decoder) []u16 {
        return @as([*]u16, @ptrCast(@alignCast(d.weights.contents())))[0..d.plan.total()];
    }

    /// The decoder's tensors of `vae/diffusion_pytorch_model.safetensors`, checked, each F32 value rounded to bf16.
    pub fn load(d: *Decoder, path: [*:0]const u8) !void {
        const map = try mtl.MappedFile.open(path);
        defer map.deinit();
        const len = try st.headerSize(std.mem.readInt(u64, map.bytes[0..8], .little), map.size);
        var arena = std.heap.ArenaAllocator.init(d.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const header = try st.parseHeader(a, map.bytes[8..][0..len], map.size - 8 - len);
        const items = try d.plan.items(a);
        var culprit: []const u8 = "";
        host.check(a, &header, items, &culprit) catch |err| {
            std.log.err("h3 image decoder: {s}: {s} (the decoder reads F32 tensors at the released shapes)", .{ culprit, @errorName(err) });
            return err;
        };
        const data = map.bytes[8 + len ..];
        const out = d.params();
        for (items) |it| {
            const e = header.get(it.name).?;
            const src = data[e.begin..e.end];
            for (out[it.offset..][0..host.count(it.part)], 0..) |*o, i| {
                o.* = rounding.bits(@bitCast(std.mem.readInt(u32, src[4 * i ..][0..4], .little)));
            }
        }
    }

    /// Synthetic weights from `seed` for the self-test: matrices by their fan-in, gains near 1, small biases.
    pub fn synthetic(d: *Decoder, seed: u64) !void {
        var arena = std.heap.ArenaAllocator.init(d.gpa);
        defer arena.deinit();
        const out = d.params();
        for (try d.plan.items(arena.allocator()), 0..) |it, i| {
            var prng = std.Random.DefaultPrng.init(seed +% i *% 0x9e3779b97f4a7c15);
            const r = prng.random();
            const fan: f32 = @floatFromInt(if (it.part.shape.len > 1) it.part.shape[1] else 1);
            for (out[it.offset..][0..host.count(it.part)]) |*o| {
                const u = r.float(f32) * 2 - 1;
                o.* = rounding.bits(switch (it.part.fill) {
                    .matrix => u * @sqrt(3.0 / fan),
                    .bias => u * 0.1,
                    .gain => 1 + u * 0.25,
                    .layer_scale => 0.3 + u * 0.2,
                    .tokens => u * 1.5,
                });
            }
        }
    }

    /// The raw latent [24, rows, cols] in, and scratch for passes of up to `batch` tiles of h x w cells.
    pub fn prepare(d: *Decoder, raw: []const f32, rows: usize, cols: usize, h: usize, w: usize, batch: usize) !void {
        if (raw.len != try host.latentLength(rows, cols)) return error.H3BadLatentLength;
        if (h > rows or w > cols) return error.H3BadTiles;
        const n = try host.tokens(h, w);
        if (batch == 0 or batch > most_batch) return error.H3BadTiles;
        if (d.latent) |l| l.deinit();
        d.latent = null;
        const lat = try d.device.buffer(raw.len * 4, mtl.ResourceOptions.shared);
        d.latent = lat;
        @memcpy(@as([*]f32, @ptrCast(@alignCast(lat.contents())))[0..raw.len], raw);
        if (d.scratch) |s| s.deinit();
        d.scratch = null;
        const m = batch * n;
        const sizes = [_]usize{ m * W * 2, m * W * 2, m * 2 * F * 2, m * W * 2, m * W * 2, m * W * 2, m * W * 2, m * F * 2, m * host.patch * 2, batch * h * w * host.channels * 2, n * vae.rotated * 2, n * vae.rotated * 2, batch * 8 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        const buf = try d.device.buffer(total, mtl.ResourceOptions.shared);
        d.scratch = buf;
        var refs: [sizes.len]Ref = undefined;
        var at: usize = 0;
        for (&refs, sizes) |*r, s| {
            r.* = .{ .buf = buf, .off = at };
            at += std.mem.alignForward(usize, s, 256);
        }
        d.views = .{ .x = refs[0], .xn = refs[1], .wide = refs[2], .q = refs[3], .k = refs[4], .v = refs[5], .o = refs[6], .act = refs[7], .out = refs[8], .tok = refs[9], .cos = refs[10], .sin = refs[11], .origins = refs[12] };
        try host.rotary(h, w, d.theta, d.bits(d.views.cos, n * vae.rotated), d.bits(d.views.sin, n * vae.rotated));
        d.rows = rows;
        d.cols = cols;
        d.h = h;
        d.w = w;
        d.tokens = n;
        d.batch = batch;
    }

    fn bits(_: *const Decoder, r: Ref, n: usize) []u16 {
        return @as([*]u16, @ptrCast(@alignCast(r.buf.contents() + r.off)))[0..n];
    }

    /// The stream [tiles * tokens, 1,024] (bf16 bits) after the last pass, which the next pass reads.
    pub fn stream(d: *const Decoder) []u16 {
        return d.bits(d.views.x, d.batch * d.tokens * W);
    }

    /// proj_out's rows [tiles * tokens, 768] (bf16 bits) after the last pass that ran the final stage.
    pub fn projected(d: *const Decoder) []const u16 {
        return d.bits(d.views.out, d.batch * d.tokens * host.patch);
    }

    /// One command buffer: `stages` on the tiles whose first cells (row, column) are `origins`.
    pub fn pass(d: *Decoder, origins: []const [2]u32, stages: Stages) !Timing {
        if (d.scratch == null or origins.len == 0 or origins.len > d.batch) return error.H3BadTiles;
        if (stages.from > stages.to or stages.to > d.plan.layers) return error.H3BadTiles;
        const slots = @as([*][2]u32, @ptrCast(@alignCast(d.views.origins.buf.contents() + d.views.origins.off)))[0..origins.len];
        for (slots, origins) |*s, o| {
            if (o[0] + d.h > d.rows or o[1] + d.w > d.cols) return error.H3BadTiles;
            s.* = o;
        }
        const started = mtl.clock.seconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.queue.commandBuffer();
        const e = cb.compute(.serial);
        const s: Shape = .{ .tiles = @intCast(origins.len), .h = @intCast(d.h), .w = @intCast(d.w), .tokens = @intCast(d.tokens), .rows = @intCast(d.rows), .cols = @intCast(d.cols), .m = @intCast(origins.len * d.tokens) };
        const v = d.views;
        if (stages.embed) {
            d.run(e, .post_quant, &.{ .{ .buf = d.latent.? }, v.origins, d.param(d.plan.head(.post_w)), d.param(d.plan.head(.post_b)), v.tok }, s, mtl.Size.of(host.channels, origins.len * d.h * d.w, 1), mtl.Size.of(host.channels, 8, 1), false);
            d.run(e, .embed, &.{ v.tok, d.param(d.plan.head(.embed_w)), d.param(d.plan.head(.embed_b)), d.param(d.plan.head(.registers)), v.x }, s, mtl.Size.of(W, s.m, 1), mtl.Size.of(256, 1, 1), false);
        }
        for (stages.from..stages.to) |i| d.layer(e, s, i);
        if (stages.final) {
            d.run(e, .layer_norm, &.{ v.x, d.param(d.plan.tail(.norm_w)), d.param(d.plan.tail(.norm_b)), v.xn }, s, mtl.Size.of(s.m, 1, 1), mtl.Size.of(256, 1, 1), true);
            d.gemm(e, s, v.xn, d.plan.tail(.proj_w), d.plan.tail(.proj_b), null, v.out, host.patch, W);
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("h3 image decoder: a pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        return .{ .wall = mtl.clock.seconds() - started, .gpu = cb.gpuSeconds() };
    }

    fn param(d: *const Decoder, offset: usize) Ref {
        return .{ .buf = d.weights, .off = offset * 2 };
    }

    fn run(d: *const Decoder, e: mtl.ComputeEncoder, k: Kernel, refs: []const Ref, s: Shape, grid: mtl.Size, group: mtl.Size, whole: bool) void {
        e.setPipeline(d.pipes[@backingInt(k)]);
        for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(s, refs.len);
        if (whole) e.dispatchGroups(grid, group) else e.dispatchThreads(grid, group);
    }

    /// y = bf16(x w^T + b) for w [n, k] at `w`, or with `scale` y += bf16(that * scale) (the LayerScale residual).
    fn gemm(d: *const Decoder, e: mtl.ComputeEncoder, s: Shape, x: Ref, w: usize, b: usize, scale: ?usize, y: Ref, n: usize, k: usize) void {
        var p = s;
        p.n = @intCast(n);
        p.k = @intCast(k);
        p.mode = @intFromBool(scale != null);
        d.run(e, .gemm, &.{ x, d.param(w), d.param(b), d.param(scale orelse b), y }, p, mtl.Size.of(n / 64, (s.m + 63) / 64, 1), mtl.Size.of(128, 1, 1), true);
    }

    /// Block `i` on the pass's stream in place.
    fn layer(d: *const Decoder, e: mtl.ComputeEncoder, s: Shape, i: usize) void {
        const v = d.views;
        const p = d.plan;
        const rows = mtl.Size.of(s.m, 1, 1);
        const row_group = mtl.Size.of(256, 1, 1);
        d.run(e, .rms_norm, &.{ v.x, d.param(p.block(i, .norm1)), v.xn }, s, rows, row_group, true);
        d.gemm(e, s, v.xn, p.block(i, .qkv_w), p.block(i, .qkv_b), null, v.wide, 3 * W, W);
        d.run(e, .heads, &.{ v.wide, v.cos, v.sin, v.q, v.k, v.v }, s, mtl.Size.of(s.m, host.heads, 1), mtl.Size.of(32, 1, 1), true);
        d.run(e, .attention, &.{ v.q, v.k, v.v, v.o }, s, mtl.Size.of((s.tokens + 7) / 8, host.heads, s.tiles), mtl.Size.of(256, 1, 1), true);
        d.gemm(e, s, v.o, p.block(i, .out_w), p.block(i, .out_b), p.block(i, .scale1), v.x, W, W);
        d.run(e, .rms_norm, &.{ v.x, d.param(p.block(i, .norm2)), v.xn }, s, rows, row_group, true);
        d.gemm(e, s, v.xn, p.block(i, .w1_w), p.block(i, .w1_b), null, v.wide, 2 * F, W);
        d.run(e, .swiglu, &.{ v.wide, v.act }, s, mtl.Size.of(F, s.m, 1), mtl.Size.of(256, 1, 1), false);
        d.gemm(e, s, v.act, p.block(i, .w2_w), p.block(i, .w2_b), p.block(i, .scale2), v.x, W, F);
    }

    /// The decoded picture [3, 16 rows, 16 cols] (normalised) of a raw latent: tiles in batches, blended in fp32.
    pub fn decode(d: *Decoder, a: std.mem.Allocator, c: *const vae.Config, raw: []const f32, rows: usize, cols: usize, layers: usize, batch: usize, out: []f32) !Timing {
        const t = try host.tiles(a, c, rows, cols);
        const per = @min(batch, t.total());
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
            const projected_rows = d.projected();
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
