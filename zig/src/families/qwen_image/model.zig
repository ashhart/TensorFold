//! Qwen-Image-2.1's transformer on Metal: the prompt rows once a prompt, then the image rows' velocity at each step.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const host = @import("host.zig");
const wts = @import("weights.zig");
const ks = @import("kernels.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

const H = cfg.hidden;
const D = cfg.head_dim;

/// One buffer's parts for `text` prompt rows and `image` rows, each 256-aligned.
pub const Scratch = struct {
    parts: [18]usize,

    pub fn of(text: usize, image: usize) Scratch {
        const rows = @max(text, image);
        const keys = text + image;
        const wide = rows * H * 2;
        const mlp = rows * cfg.mlp_width * 2;
        const lat = image * cfg.channels * 2;
        const rope = keys * D / 2 * 4;
        return .{ .parts = .{ wide, wide, wide, wide, wide, wide, wide, mlp, mlp, wide, keys * H * 2, keys * H * 2, lat, lat, text * cfg.context * 2, rope, rope, 5 * H * 2 } };
    }

    pub fn bytes(s: Scratch) usize {
        var total: usize = 0;
        for (s.parts) |n| total += std.mem.alignForward(usize, n, 256);
        return total;
    }
};

pub const Timing = struct { wall: f64, gpu: f64, host: f64 };

pub const Model = struct {
    gpa: std.mem.Allocator,
    queue: mtl.Queue,
    k: ks.Kernels,
    w: wts.Weights,
    text: usize,
    image: usize,
    columns: usize,
    scratch: mtl.Buffer,
    /// Each block's prompt keys and values [heads, text, 128] bf16, filled by `prompt`.
    store: mtl.Buffer,
    // bf16 rows: the stream, a norm's output, q k v, the heads side by side, a projection's output, the MLP's halves
    x: Ref,
    xn: Ref,
    qr: Ref,
    kr: Ref,
    vr: Ref,
    att: Ref,
    y: Ref,
    wa: Ref,
    wb: Ref,
    // bf16 heads: queries [heads, rows, 128], keys and values [heads, keys, 128]
    q: Ref,
    k_: Ref,
    v: Ref,
    lat: Ref,
    vel: Ref,
    txt: Ref,
    cos: Ref,
    sin: Ref,
    mods: Ref,
    m: host.Modulation = undefined,

    /// The transformer of checkpoint folder `dir`, with scratch for a `text`-row prompt and a width x height image.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, text: usize, width: usize, height: usize) !*Model {
        const g = try cfg.grid(width, height);
        const t = try gpa.create(Model);
        errdefer gpa.destroy(t);
        t.* = undefined;
        t.gpa = gpa;
        t.text = text;
        t.image = g.rows * g.columns;
        t.columns = g.columns;
        if (text == 0) return error.EmptyPrompt;
        t.queue = try device.queue();
        errdefer t.queue.deinit();
        t.k = try ks.Kernels.load(gpa, device);
        errdefer t.k.deinit();
        t.w = try wts.load(gpa, io, device, dir);
        errdefer t.w.deinit();
        const plan = Scratch.of(text, t.image);
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        t.scratch = try device.buffer(plan.bytes(), opts);
        errdefer t.scratch.deinit();
        t.store = try device.buffer(2 * cfg.layers * text * H * 2, opts);
        errdefer t.store.deinit();
        var at: usize = 0;
        const views = [_]*Ref{ &t.x, &t.xn, &t.qr, &t.kr, &t.vr, &t.att, &t.y, &t.wa, &t.wb, &t.q, &t.k_, &t.v, &t.lat, &t.vel, &t.txt, &t.cos, &t.sin, &t.mods };
        for (views, plan.parts) |view, n| {
            view.* = .{ .buf = t.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
        const half = (text + t.image) * D / 2;
        host.rotary(floats(t.cos, half), floats(t.sin, half), text, g.rows, g.columns);
        return t;
    }

    pub fn deinit(t: *Model) void {
        t.store.deinit();
        t.scratch.deinit();
        t.w.deinit();
        t.k.deinit();
        t.queue.deinit();
        t.gpa.destroy(t);
    }

    fn floats(r: Ref, n: usize) []f32 {
        return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    fn halves(r: Ref, n: usize) []u16 {
        return @as([*]u16, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    fn storeOf(t: *const Model, layer: usize, which: usize) Ref {
        return .{ .buf = t.store, .off = (layer * 2 + which) * t.text * H * 2 };
    }

    /// The modulation at `sigma` into the shared vectors: scale 1, gate 1, scale 2, gate 2, the output's scale.
    fn modulate(t: *Model, sigma: f32) !f64 {
        const started = mtl.clock.seconds();
        try t.m.at(t.gpa, t.w.modulation, sigma);
        const out = halves(t.mods, 5 * H);
        for ([_][]const u16{ &t.m.scale_1, &t.m.gate_1, &t.m.scale_2, &t.m.gate_2, &t.m.scale_out }, 0..) |part, i| @memcpy(out[i * H ..][0..H], part);
        return mtl.clock.seconds() - started;
    }

    fn mod(t: *const Model, i: usize) Ref {
        return t.mods.at(i * H * 2);
    }

    /// The prompt's rows (text, 4096) bf16, the text encoder's last hidden states: each block's keys and values stored.
    pub fn prompt(t: *Model, embeds: []const u16) !Timing {
        if (embeds.len != t.text * cfg.context) return error.PromptRows;
        const started = mtl.clock.seconds();
        const spent = try t.modulate(0);
        @memcpy(halves(t.txt, embeds.len), embeds);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        const T = t.text;
        ks.Kernels.bind(e, t.k.get("qi_text_norm"), &.{ t.txt, t.w.txt_norm, t.xn }, extern struct { w: u32, eps: f32 }{ .w = cfg.context, .eps = cfg.eps });
        e.dispatchGroups(mtl.Size.of(T, 1, 1), mtl.Size.of(256, 1, 1));
        t.k.linear(e, t.xn, t.w.txt_in, t.y, T, H, cfg.context);
        ks.Kernels.bind(e, t.k.get("qi_gelu"), &.{t.y}, @as(u32, H));
        e.dispatchThreads(mtl.Size.of(H, T, 1), mtl.Size.of(256, 1, 1));
        t.k.linear(e, t.y, t.w.txt_out, t.x, T, H, H);
        for (t.w.blocks, 0..) |b, i| t.block(e, b, T, 0, t.storeOf(i, 0), t.storeOf(i, 1), T, 0, true);
        return t.finish(cb, e, started, spent);
    }

    /// The velocity (image rows, 64) at noise level `sigma` for `latents` (image rows, 64), after `prompt`.
    pub fn forward(t: *Model, latents: []const f32, sigma: f32, velocity: []f32) !Timing {
        const R = t.image;
        if (latents.len != R * cfg.channels or velocity.len != latents.len) return error.LatentRows;
        const started = mtl.clock.seconds();
        const spent = try t.modulate(sigma);
        for (halves(t.lat, latents.len), latents) |*o, v| o.* = host.bf16(v);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        const T = t.text;
        const keys = T + R;
        t.k.linear(e, t.lat, t.w.img_in, t.x, R, H, cfg.channels);
        for (t.w.blocks, 0..) |b, i| {
            ks.Kernels.bind(e, t.k.get("qi_prefix"), &.{ t.storeOf(i, 0), t.storeOf(i, 1), t.k_, t.v }, [2]u32{ @intCast(T), @intCast(keys) });
            e.dispatchThreads(mtl.Size.of(D, T, cfg.heads), mtl.Size.of(D, 2, 1));
            t.block(e, b, R, T, t.k_, t.v, keys, T, false);
        }
        t.modNorm(e, R, 4);
        t.k.linear(e, t.xn, t.w.proj_out, t.vel, R, cfg.channels, H);
        const timing = try t.finish(cb, e, started, spent);
        for (velocity, halves(t.vel, velocity.len)) |*o, v| o.* = host.widen(v);
        return timing;
    }

    fn finish(t: *Model, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, started: f64, spent: f64) !Timing {
        _ = t;
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("qwen image: a pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        return .{ .wall = mtl.clock.seconds() - started, .gpu = cb.gpuSeconds(), .host = spent };
    }

    /// xn = modulated layer norm of the stream's `rows` rows, by shared vector `which`.
    fn modNorm(t: *const Model, e: mtl.ComputeEncoder, rows: usize, which: usize) void {
        ks.Kernels.bind(e, t.k.get("qi_mod_norm"), &.{ t.x, t.mod(which), t.xn }, extern struct { w: u32, eps: f32 }{ .w = H, .eps = cfg.eps });
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn gateAdd(t: *const Model, e: mtl.ComputeEncoder, rows: usize, which: usize) void {
        ks.Kernels.bind(e, t.k.get("qi_gate_add"), &.{ t.x, t.y, t.mod(which) }, @as(u32, H));
        e.dispatchThreads(mtl.Size.of(H, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// One block over `rows` rows whose rotary starts at row `first`; keys and values land at row `base` of `kb`, `vb`.
    fn block(t: *const Model, e: mtl.ComputeEncoder, b: wts.Block, rows: usize, first: usize, kb: Ref, vb: Ref, keys: usize, base: usize, causal: bool) void {
        t.modNorm(e, rows, 0);
        t.k.linear(e, t.xn, b.to_q, t.qr, rows, H, H);
        t.k.linear(e, t.xn, b.to_k, t.kr, rows, H, H);
        t.k.linear(e, t.xn, b.to_v, t.vr, rows, H, H);
        const rope = first * D / 2 * 4;
        const hp = extern struct { rows: u32, heads: u32, keys: u32, base: u32, eps: f32 }{ .rows = @intCast(rows), .heads = cfg.heads, .keys = @intCast(keys), .base = @intCast(base), .eps = cfg.eps };
        ks.Kernels.bind(e, t.k.get("qi_qk"), &.{ t.qr, t.kr, t.vr, b.norm_q, b.norm_k, t.cos.at(rope), t.sin.at(rope), t.q, kb, vb }, hp);
        e.dispatchGroups(mtl.Size.of(rows, cfg.heads / 8, 1), mtl.Size.of(256, 1, 1));
        t.k.attention(e, causal, t.q, kb, vb, t.att, rows, keys, cfg.heads);
        t.k.linear(e, t.att, b.to_out, t.y, rows, H, H);
        t.gateAdd(e, rows, 1);
        t.modNorm(e, rows, 2);
        t.k.linear(e, t.xn, b.gate, t.wa, rows, cfg.mlp_width, H);
        t.k.linear(e, t.xn, b.proj, t.wb, rows, cfg.mlp_width, H);
        ks.Kernels.bind(e, t.k.get("qi_swiglu"), &.{ t.wa, t.wb }, @as(u32, cfg.mlp_width));
        e.dispatchThreads(mtl.Size.of(cfg.mlp_width, rows, 1), mtl.Size.of(256, 1, 1));
        t.k.linear(e, t.wa, b.out, t.y, rows, H, cfg.mlp_width);
        t.gateAdd(e, rows, 3);
    }
};

test "scratch for a 1344 x 768 image after a 94-row prompt" {
    const s = Scratch.of(94, 4032);
    try std.testing.expectEqual(@as(usize, 533_978_112), s.bytes()); // the two MLP halves the largest parts
}
