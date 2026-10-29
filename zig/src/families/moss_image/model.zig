//! MOSS-Image's transformer on Metal: a prompt refined once, then image and prompt rows through every joint block.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const host = @import("host.zig");
const wts = @import("weights.zig");
const ks = @import("kernels.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

const H = cfg.hidden;
const F = cfg.ffn;
const D = cfg.head_dim;
const half = D / 2;

/// One buffer's parts for a `text`-row prompt at most and `image` rows, each 256-aligned.
pub const Scratch = struct {
    parts: [17]usize,

    fn mul(a: usize, b: usize) error{ScratchOverflow}!usize {
        return std.math.mul(usize, a, b) catch error.ScratchOverflow;
    }

    fn add(a: usize, b: usize) error{ScratchOverflow}!usize {
        return std.math.add(usize, a, b) catch error.ScratchOverflow;
    }

    pub fn of(text: usize, image: usize) !Scratch {
        if (text == 0) return error.EmptyPrompt;
        if (text > cfg.max_text) return error.PromptTooLong;
        if (image == 0) return error.BadImageSize;
        const rows = try add(text, image);
        const wide = try mul(rows, H * 2);
        const sums = try mul(@max(try mul(image, H), try mul(text, 3 * H)), 4);
        const lat = try mul(image, cfg.channels * 2);
        if (rows > std.math.maxInt(u32) / (2 * F)) return error.ImageTooLarge;
        return .{ .parts = .{
            wide,                 wide, wide, wide, wide, wide, wide, try mul(rows, 2 * F * 2),
            try mul(rows, F * 2), wide, wide, wide, sums, lat,  lat,  try mul(text, cfg.context * 2),
            host.vectors * H * 2,
        } };
    }

    pub fn bytes(s: Scratch) !usize {
        var total: usize = 0;
        for (s.parts) |n| total = try add(total, (try add(n, 255)) & ~@as(usize, 255));
        return total;
    }
};

pub const Timing = struct { wall: f64, gpu: f64, host: f64 };

/// A refined prompt: its rows after the refiner [rows, 3072] bf16, and the rotary tables of the image and its rows.
pub const Prompt = struct {
    rows: usize,
    text: mtl.Buffer,
    /// cos then sin, each (image rows + rows, 64) fp32 holding bf16 values.
    rope: mtl.Buffer,

    pub fn deinit(p: *Prompt) void {
        p.rope.deinit();
        p.text.deinit();
    }

    fn cos(p: *const Prompt) Ref {
        return .{ .buf = p.rope, .off = 0 };
    }

    fn sin(p: *const Prompt, image: usize) Ref {
        return .{ .buf = p.rope, .off = (image + p.rows) * half * 4 };
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: ks.Kernels,
    w: wts.Weights,
    max_text: usize,
    image: usize,
    rows: usize,
    columns: usize,
    scratch: mtl.Buffer,
    // bf16 rows: the stream, a norm's output, q k v, the heads side by side, a projection's output
    x: Ref,
    xn: Ref,
    qr: Ref,
    kr: Ref,
    vr: Ref,
    att: Ref,
    y: Ref,
    /// the MLP's (hidden | gate) rows, and the refiner's packed q k v or its up and gate rows
    ug: Ref,
    /// situ's output rows
    s: Ref,
    // bf16 heads [heads, rows, 128]
    q: Ref,
    k_: Ref,
    v: Ref,
    /// a biased projection's fp32 sums
    sums: Ref,
    lat: Ref,
    vel: Ref,
    ctx: Ref,
    mods: Ref,
    /// The timestep `mods` holds, if any.
    timestep: ?f32 = null,

    /// The transformer of folder `dir`, with scratch for prompts of up to `max_text` rows and a width x height image.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, max_text: usize, width: usize, height: usize) !*Model {
        const g = try cfg.grid(width, height);
        const image = try Scratch.mul(g.rows, g.columns);
        const plan = try Scratch.of(max_text, image);
        const scratch_bytes = try plan.bytes();
        // qualified on the M5's tensor units alone: the projections and attention take another path before them
        if (!device.tensorUnits()) {
            std.log.err("moss image: {s} has no tensor units; this family runs on the M5's alone, the only chip it was checked on", .{device.name()});
            return error.TensorUnitsRequired;
        }
        const t = try gpa.create(Model);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .device = device, .queue = undefined, .k = undefined, .w = undefined, .max_text = max_text, .image = image, .rows = g.rows, .columns = g.columns, .scratch = undefined, .x = undefined, .xn = undefined, .qr = undefined, .kr = undefined, .vr = undefined, .att = undefined, .y = undefined, .ug = undefined, .s = undefined, .q = undefined, .k_ = undefined, .v = undefined, .sums = undefined, .lat = undefined, .vel = undefined, .ctx = undefined, .mods = undefined };
        t.queue = try device.queue();
        errdefer t.queue.deinit();
        t.k = try ks.Kernels.load(gpa, device);
        errdefer t.k.deinit();
        t.w = try wts.load(gpa, io, device, dir);
        errdefer t.w.deinit();
        t.scratch = try device.buffer(scratch_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var at: usize = 0;
        const views = [_]*Ref{ &t.x, &t.xn, &t.qr, &t.kr, &t.vr, &t.att, &t.y, &t.ug, &t.s, &t.q, &t.k_, &t.v, &t.sums, &t.lat, &t.vel, &t.ctx, &t.mods };
        for (views, plan.parts) |view, n| {
            view.* = .{ .buf = t.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
        return t;
    }

    pub fn deinit(t: *Model) void {
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

    fn vec(t: *const Model, which: host.Vector) Ref {
        return t.mods.at(@backingInt(which) * H * 2);
    }

    /// A prompt's rows (rows, 4096) bf16, the text encoder's last hidden states, through the refiner; with its tables.
    pub fn prompt(t: *Model, embeds: []const u16) !Prompt {
        if (embeds.len == 0 or embeds.len % cfg.context != 0) return error.PromptRows;
        const T = embeds.len / cfg.context;
        if (T > t.max_text) return error.PromptTooLong;
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        var p: Prompt = .{ .rows = T, .text = try t.device.buffer(T * H * 2, opts), .rope = undefined };
        errdefer p.text.deinit();
        const table = (t.image + T) * half;
        p.rope = try t.device.buffer(2 * table * 4, opts);
        errdefer p.rope.deinit();
        host.rotary(floats(p.cos(), table), floats(p.sin(t.image), table), &t.w.bands, T, t.rows, t.columns);
        @memcpy(halves(t.ctx, embeds.len), embeds);
        @memset(halves(t.vec(.ones), H), host.bf16(1.0));
        const started = mtl.clock.seconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.k.linearBias(e, t.ctx, t.w.text_in, t.w.text_in_b, t.sums, t.x, T, H, cfg.context);
        const rope = t.image * half * 4;
        for (t.w.text) |b| t.textBlock(e, b, T, p.cos().at(rope), p.sin(t.image).at(rope));
        _ = try finish(cb, e, started, 0);
        @memcpy(halves(.{ .buf = p.text }, T * H), halves(t.x, T * H));
        return p;
    }

    /// The velocity of `latents` (image rows, 128) at `timestep` under prompt `p`, skipping block `skip` if given.
    pub fn forward(t: *Model, p: *const Prompt, latents: []const f32, timestep: f32, skip: ?usize, velocity: []f32) !Timing {
        const R = t.image;
        if (latents.len != R * cfg.channels or velocity.len != latents.len) return error.LatentRows;
        if (skip) |s| if (s >= cfg.layers) return error.BadSkipLayer;
        const started = mtl.clock.seconds();
        var spent: f64 = 0;
        if (t.timestep != timestep) {
            const vectors: *[host.vectors][H]u16 = @ptrCast(@alignCast(t.mods.addr()));
            t.timestep = null;
            try host.modulation(vectors, t.gpa, t.w.modulation, timestep);
            t.timestep = timestep;
            spent = mtl.clock.seconds() - started;
        }
        for (halves(t.lat, latents.len), latents) |*o, v| o.* = host.bf16(v);
        const N = R + p.rows;
        @memcpy(halves(t.x.at(R * H * 2), p.rows * H), halves(.{ .buf = p.text }, p.rows * H));
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.k.linearBias(e, t.lat, t.w.video_in, t.w.video_in_b, t.sums, t.x, R, H, cfg.channels);
        for (t.w.blocks, 0..) |b, i| if (skip != i) t.block(e, b, N, p.cos(), p.sin(R));
        t.norm(e, R, t.w.final_norm, .{ .scale = .scale_out, .shift = .shift_out });
        t.k.linear(e, t.xn, t.w.final_out, t.vel, R, cfg.channels, H);
        const timing = try finish(cb, e, started, spent);
        for (velocity, halves(t.vel, velocity.len)) |*o, v| o.* = host.widen(v);
        return timing;
    }

    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, started: f64, spent: f64) !Timing {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("moss image: a pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        return .{ .wall = mtl.clock.seconds() - started, .gpu = cb.gpuSeconds(), .host = spent };
    }

    const Modulated = struct { scale: host.Vector, shift: host.Vector };

    /// xn = the RMSNorm of the stream's `rows` rows with weight `w`, then modulated when given.
    fn norm(t: *const Model, e: mtl.ComputeEncoder, rows: usize, w: Ref, m: ?Modulated) void {
        const s = if (m) |v| t.vec(v.scale) else t.vec(.ones);
        const b = if (m) |v| t.vec(v.shift) else t.vec(.ones);
        ks.norm(e, t.k.get("mi_norm"), t.x, w, s, b, t.xn, rows, H, cfg.eps, m != null);
    }

    /// x += y gated by shared vector `gate` (ones: plain).
    fn gateAdd(t: *const Model, e: mtl.ComputeEncoder, rows: usize, gate: host.Vector) void {
        ks.gateAdd(e, t.k.get("mi_gate_add"), t.x, t.y, t.vec(gate), rows, H);
    }

    /// q k v rows of `stride` to normed, rotated heads in q, k_, v.
    fn heads(t: *const Model, e: mtl.ComputeEncoder, qr: Ref, kr: Ref, vr: Ref, stride: usize, nq: Ref, nk: Ref, cos: Ref, sin: Ref, rows: usize) void {
        ks.heads(e, t.k.get("mi_qk"), qr, kr, vr, nq, nk, cos, sin, t.q, t.k_, t.v, rows, cfg.heads, stride, cfg.eps);
    }

    /// s = situ(hidden, gate) of rows `stride` apart.
    fn situ(t: *const Model, e: mtl.ComputeEncoder, hidden: Ref, gate: Ref, rows: usize, stride: usize) void {
        ks.situ(e, t.k.get("mi_situ"), hidden, gate, t.s, rows, F, stride);
    }

    /// One joint block over the image and prompt rows (`rows` in all), under the shared modulation.
    fn block(t: *const Model, e: mtl.ComputeEncoder, b: wts.Block, rows: usize, cos: Ref, sin: Ref) void {
        t.norm(e, rows, b.norm1, .{ .scale = .scale_a, .shift = .shift_a });
        t.k.linear(e, t.xn, b.q, t.qr, rows, H, H);
        t.k.linear(e, t.xn, b.k, t.kr, rows, H, H);
        t.k.linear(e, t.xn, b.v, t.vr, rows, H, H);
        t.heads(e, t.qr, t.kr, t.vr, H, b.q_norm, b.k_norm, cos, sin, rows);
        t.k.attention(e, t.q, t.k_, t.v, t.att, rows, cfg.heads);
        t.k.linear(e, t.att, b.o, t.y, rows, H, H);
        t.gateAdd(e, rows, .gate_a);
        t.norm(e, rows, b.norm2, .{ .scale = .scale_f, .shift = .shift_f });
        t.k.linear(e, t.xn, b.up_gate, t.ug, rows, 2 * F, H);
        t.situ(e, t.ug, t.ug.at(F * 2), rows, 2 * F);
        t.k.linear(e, t.s, b.down, t.y, rows, H, F);
        t.gateAdd(e, rows, .gate_f);
    }

    /// One refiner block over the prompt's `rows` rows: plain norms and residuals, biased projections.
    fn textBlock(t: *const Model, e: mtl.ComputeEncoder, b: wts.TextBlock, rows: usize, cos: Ref, sin: Ref) void {
        t.norm(e, rows, b.norm1, null);
        t.k.linearBias(e, t.xn, b.qkv, b.qkv_b, t.sums, t.ug, rows, 3 * H, H);
        t.heads(e, t.ug, t.ug.at(H * 2), t.ug.at(2 * H * 2), 3 * H, b.q_norm, b.k_norm, cos, sin, rows);
        t.k.attention(e, t.q, t.k_, t.v, t.att, rows, cfg.heads);
        t.k.linearBias(e, t.att, b.proj, b.proj_b, t.sums, t.y, rows, H, H);
        t.gateAdd(e, rows, .ones);
        t.norm(e, rows, b.norm2, null);
        const gate = t.ug.at(rows * F * 2);
        t.k.linearBias(e, t.xn, b.up, b.up_b, t.sums, t.ug, rows, F, H);
        t.k.linearBias(e, t.xn, b.gate, b.gate_b, t.sums, gate, rows, F, H);
        t.situ(e, t.ug, gate, rows, F);
        t.k.linearBias(e, t.s, b.down, b.down_b, t.sums, t.y, rows, H, F);
        t.gateAdd(e, rows, .ones);
    }
};

test "scratch for a 1024 x 1024 image after prompts of up to 4,096 rows" {
    const s = try Scratch.of(4096, 4096);
    try std.testing.expectEqual(@as(usize, 1_092_671_488), try s.bytes());
}

test "scratch rejects row products, alignment and total overflow without allocating" {
    const huge = std.math.maxInt(usize);
    try std.testing.expectError(error.ScratchOverflow, Scratch.of(1, huge));
    try std.testing.expectError(error.ScratchOverflow, Scratch.of(1, huge / 2));
    const alignment: Scratch = .{ .parts = @splat(huge) };
    try std.testing.expectError(error.ScratchOverflow, alignment.bytes());
    const total: Scratch = .{ .parts = @splat(huge / 4 + 1) };
    try std.testing.expectError(error.ScratchOverflow, total.bytes());
    try std.testing.expectError(error.EmptyPrompt, Scratch.of(0, 1));
    try std.testing.expectError(error.PromptTooLong, Scratch.of(cfg.max_text + 1, 1));
    try std.testing.expectError(error.BadImageSize, Scratch.of(1, 0));
}

test "model load validates huge dimensions before touching a Metal device or checkpoint" {
    const huge = std.math.maxInt(usize) - (cfg.pixels_per_row - 1);
    try std.testing.expectError(error.ImageTooLarge, Model.load(std.testing.allocator, std.testing.io, undefined, "unused", 1, huge, huge));
    try std.testing.expectError(error.PromptTooLong, Model.load(std.testing.allocator, std.testing.io, undefined, "unused", cfg.max_text + 1, 16, 16));
}
