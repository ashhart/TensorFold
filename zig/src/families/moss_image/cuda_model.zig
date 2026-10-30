//! MOSS-Image's transformer on CUDA: a prompt refined once, then image and prompt rows through every joint block.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const host = @import("host.zig");
const wts = @import("cuda_weights.zig");
const ks = @import("cuda_kernels.zig");

const H = cfg.hidden;
const F = cfg.ffn;
const D = cfg.head_dim;
const half = D / 2;

/// The device scratch for prompts of up to `text` rows and `image` image rows, in `Model`'s field order (Metal's sums).
pub const Scratch = struct {
    parts: [15]usize,

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
        // the kernels' row counts and offsets are c_int
        if (rows > std.math.maxInt(c_int) / (2 * F)) return error.ImageTooLarge;
        const wide = try mul(rows, H * 2);
        const sums = try mul(@max(try mul(image, H), try mul(text, 3 * H)), 4);
        const lat = try mul(image, cfg.channels * 2);
        return .{ .parts = .{
            wide,                 wide, wide, wide, wide,                           try mul(wide, 3),     wide, wide, try mul(rows, 2 * F * 2),
            try mul(rows, F * 2), sums, lat,  lat,  try mul(text, cfg.context * 2), host.vectors * H * 2,
        } };
    }

    pub fn bytes(s: Scratch) !usize {
        var total: usize = 0;
        for (s.parts) |n| total = try add(total, (try add(n, 255)) & ~@as(usize, 255));
        return total;
    }
};

pub const Timing = struct { wall: f64, gpu: f64, host: f64 };

pub fn seconds() f64 {
    const c = @import("sys").c;
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// A refined prompt: its rows after the refiner [rows, 3072] bf16, and the rotary tables of the image and its rows.
pub const Prompt = struct {
    rows: usize,
    text: cuda.DeviceBuffer,
    /// cos then sin, each (image rows + rows, 64) fp32 holding bf16 values.
    rope: cuda.DeviceBuffer,

    pub fn deinit(p: *Prompt) void {
        p.rope.free();
        p.text.free();
    }

    fn cos(p: *const Prompt) u64 {
        return p.rope.ptr;
    }

    fn sin(p: *const Prompt, image: usize) u64 {
        return p.rope.ptr + (image + p.rows) * half * 4;
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: ks.Kernels,
    vk: cuda.vit.Kernels,
    gemms: cuda.vit.Gemms,
    att: cuda.vit.Attention,
    w: *wts.Weights,
    max_text: usize,
    image: usize,
    rows: usize,
    columns: usize,
    scratch: cuda.DeviceBuffer,
    /// A pass's start and end on the stream, for its GPU time.
    marks: [2]cuda.Event,
    // bf16 rows: the stream, a norm's output, q k v, packed q | k | v heads, attention's output, a projection's output
    x: u64 = 0,
    xn: u64 = 0,
    qr: u64 = 0,
    kr: u64 = 0,
    vr: u64 = 0,
    qkv: u64 = 0,
    o: u64 = 0,
    y: u64 = 0,
    /// the MLP's (hidden | gate) rows, and the refiner's packed q k v or its up and gate rows
    ug: u64 = 0,
    /// situ's output rows
    s: u64 = 0,
    /// a biased projection's fp32 sums
    acc: u64 = 0,
    lat: u64 = 0,
    vel: u64 = 0,
    ctx: u64 = 0,
    mods: u64 = 0,
    /// The timestep `mods` holds, if any.
    timestep: ?f32 = null,

    /// The transformer of `dir` for prompts of up to `max_text` rows and one image size, on sm_120 or sm_121 only.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, max_text: usize, width: usize, height: usize) !*Model {
        const d = ctx.d;
        const sm = try ctx.capability();
        if (sm != 120 and sm != 121) {
            std.log.err("moss image: sm_{d} is not a GPU this family was checked on (sm_120, sm_121)", .{sm});
            return error.UncheckedGpu;
        }
        const g = try cfg.grid(width, height);
        const image = try Scratch.mul(g.rows, g.columns);
        const plan = try Scratch.of(max_text, image);
        const t = try gpa.create(Model);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = d, .stream = undefined, .k = undefined, .vk = undefined, .gemms = undefined, .att = undefined, .w = undefined, .max_text = max_text, .image = image, .rows = g.rows, .columns = g.columns, .scratch = undefined, .marks = undefined };
        t.stream = try cuda.Stream.init(d, false);
        errdefer t.stream.deinit();
        t.k = try ks.Kernels.load(d);
        errdefer t.k.unload();
        t.vk = try cuda.vit.Kernels.load(d);
        errdefer t.vk.unload();
        t.gemms = try cuda.vit.Gemms.init(gpa, d);
        errdefer t.gemms.deinit();
        t.att = try cuda.vit.Attention.load(d);
        errdefer t.att.unload();
        t.w = try wts.load(gpa, io, d, dir);
        errdefer {
            t.w.deinit();
            gpa.destroy(t.w);
        }
        t.scratch = try cuda.DeviceBuffer.alloc(d, try plan.bytes());
        errdefer t.scratch.free();
        t.marks[0] = try cuda.Event.init(d, true);
        errdefer t.marks[0].deinit();
        t.marks[1] = try cuda.Event.init(d, true);
        var at: usize = t.scratch.ptr;
        const views = [_]*u64{ &t.x, &t.xn, &t.qr, &t.kr, &t.vr, &t.qkv, &t.o, &t.y, &t.ug, &t.s, &t.acc, &t.lat, &t.vel, &t.ctx, &t.mods };
        for (views, plan.parts) |view, n| {
            view.* = at;
            at += std.mem.alignForward(usize, n, 256);
        }
        return t;
    }

    pub fn deinit(t: *Model) void {
        for (&t.marks) |*m| m.deinit();
        t.scratch.free();
        t.w.deinit();
        t.gpa.destroy(t.w);
        t.att.unload();
        t.gemms.deinit();
        t.vk.unload();
        t.k.unload();
        t.stream.deinit();
        t.gpa.destroy(t);
    }

    fn vec(t: *const Model, which: host.Vector) u64 {
        return t.mods + @backingInt(which) * H * 2;
    }

    fn upload(t: *const Model, at: u64, bytes: []const u8) !void {
        try t.d.check(t.d.api.cuMemcpyHtoDAsync_v2(at, bytes.ptr, bytes.len, t.stream.handle), "cuMemcpyHtoDAsync");
        try t.stream.synchronize();
    }

    /// y [rows][n] bf16 = x [rows][k] . w^T, fp32 sums rounded once.
    fn linear(t: *Model, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        try t.gemms.run(t.stream, x, w, y, rows, n, k, .bf16);
    }

    /// The same with bias b added to the fp32 sums before the one rounding.
    fn linearBias(t: *Model, x: u64, w: u64, b: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        try cuda.vit.linear(&t.gemms, &t.vk, t.stream, t.acc, x, w, b, y, rows, n, k);
    }

    /// A prompt's rows (rows, 4096) bf16, the text encoder's last hidden states, through the refiner; with its tables.
    pub fn prompt(t: *Model, embeds: []const u16) !Prompt {
        if (embeds.len == 0 or embeds.len % cfg.context != 0) return error.PromptRows;
        const T = embeds.len / cfg.context;
        if (T > t.max_text) return error.PromptTooLong;
        var p: Prompt = .{ .rows = T, .text = try cuda.DeviceBuffer.alloc(t.d, T * H * 2), .rope = undefined };
        errdefer p.text.free();
        const table = (t.image + T) * half;
        const tables = try t.gpa.alloc(f32, 2 * table);
        defer t.gpa.free(tables);
        host.rotary(tables[0..table], tables[table..], &t.w.bands, T, t.rows, t.columns);
        p.rope = try cuda.DeviceBuffer.fromHost(t.d, std.mem.sliceAsBytes(tables));
        errdefer p.rope.free();
        try t.upload(t.ctx, std.mem.sliceAsBytes(embeds));
        var ones: [H]u16 = undefined;
        @memset(&ones, host.bf16(1.0));
        try t.upload(t.vec(.ones), std.mem.sliceAsBytes(&ones));
        try t.linearBias(t.ctx, t.w.text_in, t.w.text_in_b, t.x, T, H, cfg.context);
        const rope = t.image * half * 4;
        for (t.w.text) |b| try t.textBlock(b, T, p.cos() + rope, p.sin(t.image) + rope);
        try t.d.check(t.d.api.cuMemcpyDtoDAsync_v2(p.text.ptr, t.x, T * H * 2, t.stream.handle), "cuMemcpyDtoDAsync");
        try t.stream.synchronize();
        return p;
    }

    /// The velocity of `latents` (image rows, 128) at `timestep` under prompt `p`, skipping block `skip` if given.
    pub fn forward(t: *Model, p: *const Prompt, latents: []const f32, timestep: f32, skip: ?usize, velocity: []f32) !Timing {
        const R = t.image;
        if (latents.len != R * cfg.channels or velocity.len != latents.len) return error.LatentRows;
        if (skip) |s| if (s >= cfg.layers) return error.BadSkipLayer;
        const started = seconds();
        var spent: f64 = 0;
        if (t.timestep != timestep) {
            var vectors: [host.vectors][H]u16 = undefined;
            t.timestep = null;
            try host.modulation(&vectors, t.gpa, t.w.modulation, timestep);
            try t.upload(t.mods, std.mem.sliceAsBytes(&vectors));
            t.timestep = timestep;
            spent = seconds() - started;
        }
        const halves = try t.gpa.alloc(u16, latents.len);
        defer t.gpa.free(halves);
        for (halves, latents) |*o, v| o.* = host.bf16(v);
        try t.upload(t.lat, std.mem.sliceAsBytes(halves));
        const N = R + p.rows;
        try t.marks[0].record(t.stream);
        try t.d.check(t.d.api.cuMemcpyDtoDAsync_v2(t.x + R * H * 2, p.text.ptr, p.rows * H * 2, t.stream.handle), "cuMemcpyDtoDAsync");
        try t.linearBias(t.lat, t.w.video_in, t.w.video_in_b, t.x, R, H, cfg.channels);
        for (t.w.blocks, 0..) |b, i| if (skip != i) try t.block(b, N, p.cos(), p.sin(R));
        try t.norm(R, t.w.final_norm, .{ .scale = .scale_out, .shift = .shift_out });
        try t.linear(t.xn, t.w.final_out, t.vel, R, cfg.channels, H);
        try t.marks[1].record(t.stream);
        try t.d.check(t.d.api.cuMemcpyDtoHAsync_v2(halves.ptr, t.vel, halves.len * 2, t.stream.handle), "cuMemcpyDtoHAsync");
        try t.stream.synchronize();
        for (velocity, halves) |*o, v| o.* = host.widen(v);
        return .{ .wall = seconds() - started, .gpu = @as(f64, try cuda.Event.elapsedMs(t.marks[0], t.marks[1])) / 1000, .host = spent };
    }

    const Modulated = struct { scale: host.Vector, shift: host.Vector };

    /// xn = the RMSNorm of the stream's `rows` rows with weight `w`, then modulated when given.
    fn norm(t: *const Model, rows: usize, w: u64, m: ?Modulated) !void {
        const s = if (m) |v| t.vec(v.scale) else t.vec(.ones);
        const b = if (m) |v| t.vec(v.shift) else t.vec(.ones);
        try ks.norm(&t.k, t.stream, t.x, w, s, b, t.xn, rows, H, cfg.eps, m != null);
    }

    /// The heads of q, k, v rows `stride` apart normed, rotated and packed, then full attention into o.
    fn attend(t: *const Model, qr: u64, kr: u64, vr: u64, stride: usize, nq: u64, nk: u64, cos: u64, sin: u64, rows: usize) !void {
        try ks.qk(&t.k, t.stream, qr, kr, vr, nq, nk, cos, sin, t.qkv, rows, cfg.heads, stride, cfg.eps);
        try t.att.full(t.stream, t.qkv, t.o, rows, cfg.heads, D, 1.0 / @sqrt(@as(f32, D)));
    }

    /// One joint block over the image and prompt rows (`rows` in all), under the shared modulation.
    fn block(t: *Model, b: wts.Block, rows: usize, cos: u64, sin: u64) !void {
        try t.norm(rows, b.norm1, .{ .scale = .scale_a, .shift = .shift_a });
        try t.linear(t.xn, b.q, t.qr, rows, H, H);
        try t.linear(t.xn, b.k, t.kr, rows, H, H);
        try t.linear(t.xn, b.v, t.vr, rows, H, H);
        try t.attend(t.qr, t.kr, t.vr, H, b.q_norm, b.k_norm, cos, sin, rows);
        try t.linear(t.o, b.o, t.y, rows, H, H);
        try ks.gateAdd(&t.k, t.stream, t.x, t.y, t.vec(.gate_a), rows, H);
        try t.norm(rows, b.norm2, .{ .scale = .scale_f, .shift = .shift_f });
        try t.linear(t.xn, b.up_gate, t.ug, rows, 2 * F, H);
        try ks.situ(&t.k, t.stream, t.ug, t.ug + F * 2, t.s, rows, F, 2 * F);
        try t.linear(t.s, b.down, t.y, rows, H, F);
        try ks.gateAdd(&t.k, t.stream, t.x, t.y, t.vec(.gate_f), rows, H);
    }

    /// One refiner block over the prompt's `rows` rows: plain norms and residuals, biased projections.
    fn textBlock(t: *Model, b: wts.TextBlock, rows: usize, cos: u64, sin: u64) !void {
        try t.norm(rows, b.norm1, null);
        try t.linearBias(t.xn, b.qkv, b.qkv_b, t.ug, rows, 3 * H, H);
        try t.attend(t.ug, t.ug + H * 2, t.ug + 2 * H * 2, 3 * H, b.q_norm, b.k_norm, cos, sin, rows);
        try t.linearBias(t.o, b.proj, b.proj_b, t.y, rows, H, H);
        try ks.gateAdd(&t.k, t.stream, t.x, t.y, t.vec(.ones), rows, H);
        try t.norm(rows, b.norm2, null);
        const gate = t.ug + rows * F * 2;
        try t.linearBias(t.xn, b.up, b.up_b, t.ug, rows, F, H);
        try t.linearBias(t.xn, b.gate, b.gate_b, gate, rows, F, H);
        try ks.situ(&t.k, t.stream, t.ug, gate, t.s, rows, F, F);
        try t.linearBias(t.s, b.down, b.down_b, t.y, rows, H, F);
        try ks.gateAdd(&t.k, t.stream, t.x, t.y, t.vec(.ones), rows, H);
    }
};

test "scratch: Metal's sums, refusals, and every part inside the c_int range the kernels index" {
    const s = try Scratch.of(34, 4096);
    try std.testing.expectEqual(@as(usize, 4096 * H * 4), s.parts[10]);
    try std.testing.expectEqual(@as(usize, 34 * 3 * H * 4), (try Scratch.of(34, 64)).parts[10]);
    try std.testing.expectError(error.EmptyPrompt, Scratch.of(0, 4096));
    try std.testing.expectError(error.PromptTooLong, Scratch.of(cfg.max_text + 1, 4096));
    try std.testing.expectError(error.ImageTooLarge, Scratch.of(cfg.max_text, 1 << 20));
    const most = std.math.maxInt(c_int) / (2 * F) - cfg.max_text;
    _ = try (try Scratch.of(cfg.max_text, most)).bytes();
}
