//! MiniMax H3's denoising loop around the CUDA block library: rows in, every block, the final norm, rows out, Euler.
const std = @import("std");
const bf = @import("bf16_round");
pub const st = @import("core").safetensors;
pub const geometry = @import("geometry.zig");
pub const euler = @import("euler.zig");
pub const Model = @import("cuda.zig").Model;
pub const host = @import("cuda_host.zig");
pub const Inputs = Model.Inputs;
const Allocator = std.mem.Allocator;

/// The reference's RMSNorm epsilon, as the Metal tool's (h3.eps).
pub const eps: f32 = 1e-5;

/// A case file's tensors as host copies, the block tables in the library's (blocks, lines, 6, hidden) float layout.
pub const Case = struct {
    arena: std.heap.ArenaAllocator,
    g: geometry.Geometry,
    blocks: usize,
    /// Video tiles a video tile keeps (0: dense, every tile sees every tile).
    keep: usize,
    text: []u16,
    condition: []f32,
    tables: []f32,
    adaln: []i32,
    times: []i32,
    final: []u16,
    final_norm: []u16,
    video_in_w: []f32,
    video_in_b: []f32,
    audio_in_w: []f32,
    audio_in_b: []f32,
    video_out_w: []f32,
    video_out_b: []f32,
    audio_out_w: []f32,
    audio_out_b: []f32,
    cos: []f32,
    sin: []f32,
    slot: []i32,
    sizes: []i32,
    video_step: []f32,
    audio_step: []f32,

    /// The case `c` that geometry.read admitted as `g` for `blocks` blocks; `dense` drops the case's keep.
    pub fn read(gpa: Allocator, c: geometry.Case, g: geometry.Geometry, blocks: usize, dense: bool) !Case {
        var self: Case = undefined;
        self.arena = .init(gpa);
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        const r = g.layout;
        self.g = g;
        self.blocks = blocks;
        self.keep = if (dense) 0 else r.keep;
        self.text = try take(u16, a, c, "text", r.text * g.hidden);
        self.condition = if (r.condition > 0) try take(f32, a, c, "condition", r.condition * g.video_in) else &.{};
        self.adaln = try take(i32, a, c, "adaln", g.steps * r.rows);
        self.times = try take(i32, a, c, "times", g.steps * r.rows);
        self.final = try take(u16, a, c, "final", g.times * 2 * g.hidden);
        self.final_norm = try take(u16, a, c, "final_norm.weight", g.hidden);
        inline for (.{ "video", "audio" }) |kind| {
            const width = if (comptime std.mem.eql(u8, kind, "video")) g.video_in else g.audio_in;
            @field(self, kind ++ "_in_w") = try take(f32, a, c, kind ++ "_in.weight", g.hidden * width);
            @field(self, kind ++ "_in_b") = try take(f32, a, c, kind ++ "_in.bias", g.hidden);
            @field(self, kind ++ "_out_w") = try take(f32, a, c, kind ++ "_out.weight", width * g.hidden);
            @field(self, kind ++ "_out_b") = try take(f32, a, c, kind ++ "_out.bias", width);
            @field(self, kind ++ "_step") = try take(f32, a, c, kind ++ "_step", 2 * g.steps);
        }
        self.cos = try take(f32, a, c, "cos", r.rows * g.rot);
        self.sin = try take(f32, a, c, "sin", r.rows * g.rot);
        self.slot = try take(i32, a, c, "tile_slot", r.rows);
        self.sizes = try take(i32, a, c, "tile_sizes", r.tiles);
        // the case's tables are (6, lines, hidden) bf16 a block; the library reads (lines, 6, hidden) floats
        const per = 6 * g.lines * g.hidden;
        self.tables = try a.alloc(f32, try geometry.product(&.{ blocks, per }));
        var name: [32]u8 = undefined;
        for (0..blocks) |b| {
            const raw = try take(u16, a, c, try std.fmt.bufPrint(&name, "tables.{d}", .{b}), per);
            const out = self.tables[b * per ..][0..per];
            for (0..6) |p| for (0..g.lines) |l| for (0..g.hidden) |k| {
                out[(l * 6 + p) * g.hidden + k] = bf.value(raw[(p * g.lines + l) * g.hidden + k]);
            };
        }
        return self;
    }

    pub fn deinit(self: *Case) void {
        self.arena.deinit();
    }
};

/// `count` values of tensor `name` copied out of the case (its bytes need not be aligned there).
pub fn take(comptime T: type, a: Allocator, c: geometry.Case, name: []const u8, count: usize) ![]T {
    const bytes = try c.bytes(name);
    if (bytes.len != count * @sizeOf(T)) return error.H3BadTensorShape;
    const out = try a.alloc(T, count);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

/// The library's forward over every block of the packed stream, in place (tf_h3_forward_host on a real model).
pub const Forward = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, in: *const Inputs) anyerror!void,
};

/// The loop's buffers over one case: the packed stream, its normed rows and a step's velocities.
pub const Loop = struct {
    c: *const Case,
    forward: Forward,
    x: []u16,
    normed: []u16,
    vv: []f32,
    av: []f32,

    pub fn init(a: Allocator, c: *const Case, forward: Forward) !Loop {
        const r = c.g.layout;
        const x = try a.alloc(u16, try geometry.product(&.{ r.rows, c.g.hidden }));
        errdefer a.free(x);
        const normed = try a.alloc(u16, x.len);
        errdefer a.free(normed);
        const vv = try a.alloc(f32, r.video * c.g.video_in);
        errdefer a.free(vv);
        return .{ .c = c, .forward = forward, .x = x, .normed = normed, .vv = vv, .av = try a.alloc(f32, r.audio * c.g.audio_in) };
    }

    pub fn deinit(l: *Loop, a: Allocator) void {
        inline for (.{ "x", "normed", "vv", "av" }) |f| a.free(@field(l, f));
    }

    /// Step `step`'s velocities for the latent rows `video` and `audio`, left in `vv` and `av`.
    pub fn velocities(l: *Loop, video: []const f32, audio: []const f32, step: usize) !void {
        const c = l.c;
        const g = c.g;
        const r = g.layout;
        try geometry.forward(r.rows, r.video * g.video_in, r.audio * g.audio_in, c.adaln.len, c.times.len, step, video.len, audio.len, l.vv.len, l.av.len);
        // the text rows as the case holds them; a first frame's rows enter like video rows and are never stepped
        @memcpy(l.x[0..c.text.len], c.text);
        const media = r.text + r.condition;
        rowsIn(l.x, g.hidden, r.text, c.condition, c.video_in_w, c.video_in_b, g.video_in);
        rowsIn(l.x, g.hidden, media, audio, c.audio_in_w, c.audio_in_b, g.audio_in);
        rowsIn(l.x, g.hidden, media + r.audio, video, c.video_in_w, c.video_in_b, g.video_in);
        const in: Inputs = .{
            .x = @intFromPtr(l.x.ptr),
            .tables = @intFromPtr(c.tables.ptr),
            .line = @intFromPtr(c.adaln[step * r.rows ..].ptr),
            .cos = @intFromPtr(c.cos.ptr),
            .sin = @intFromPtr(c.sin.ptr),
            .slot = @intFromPtr(c.slot.ptr),
            .sizes = @intFromPtr(c.sizes.ptr),
            .rows = @intCast(r.rows),
            .lines = @intCast(g.lines),
            .rot = @intCast(g.rot),
            .tiles = @intCast(r.tiles),
            .prefix_tiles = @intCast(r.prefix_tiles),
            .keep = @intCast(c.keep),
        };
        try l.forward.run(l.forward.ctx, &in);
        finalNorm(l.normed, l.x, c.final_norm, c.final, c.times[step * r.rows ..][0..r.rows], g.hidden);
        rowsOut(l.av, l.normed, media, c.audio_out_w, c.audio_out_b, g.audio_in, g.hidden);
        rowsOut(l.vv, l.normed, media + r.audio, c.video_out_w, c.video_out_b, g.video_in, g.hidden);
    }

    /// Steps [from, to): each step's velocities, then the Euler step of the video rows and the audio rows not held.
    pub fn run(l: *Loop, video: []f32, audio: []f32, from: usize, to: usize) !void {
        const c = l.c;
        try geometry.stepRange(from, c.g.steps);
        if (to > c.g.steps or to < from) return error.H3BadStep;
        const held = c.g.layout.held_audio * c.g.audio_in;
        for (from..to) |step| {
            try l.velocities(video, audio, step);
            euler.advance(video, l.vv, c.video_step[2 * step], c.video_step[2 * step + 1]);
            euler.advance(audio[held..], l.av[held..], c.audio_step[2 * step], c.audio_step[2 * step + 1]);
        }
    }
};

/// X[first + r, n] = bf16(B[n] + sum over k of L[r, k] W[n, k]): latent rows into the stream (Metal's h3_rows_in).
pub fn rowsIn(x: []u16, hidden: usize, first: usize, latents: []const f32, w: []const f32, b: []const f32, width: usize) void {
    var r: usize = 0;
    while (r * width < latents.len) : (r += 1) {
        const l = latents[r * width ..][0..width];
        for (0..hidden) |n| {
            var sum: f32 = 0;
            for (l, w[n * width ..][0..width]) |v, u| sum += v * u;
            x[(first + r) * hidden + n] = bf.bits(sum + b[n]);
        }
    }
}

/// The final layer's modulated RMSNorm to bf16 rows; `table` is (times, 2 hidden) bf16, shift then scale.
pub fn finalNorm(out: []u16, x: []const u16, w: []const u16, table: []const u16, times: []const i32, hidden: usize) void {
    for (times, 0..) |t, row| {
        const xs = x[row * hidden ..][0..hidden];
        var sq: f32 = 0;
        for (xs) |v| sq += bf.value(v) * bf.value(v);
        const inv = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(hidden)) + eps);
        const tab = table[@as(usize, @intCast(t)) * 2 * hidden ..][0 .. 2 * hidden];
        for (xs, 0..) |v, k| out[row * hidden + k] = bf.bits(modulated(bf.value(v), inv, bf.value(w[k]), bf.value(tab[hidden + k]), bf.value(tab[k])));
    }
}

/// The modulated norm of one value, rounded to bf16 where the reference's bf16 arithmetic rounds (Metal's h3_mod).
pub fn modulated(x: f32, inv: f32, w: f32, scale: f32, shift: f32) f32 {
    return bf.round32(bf.round32(bf.round32(x * inv * w) * bf.round32(1.0 + scale)) + shift);
}

/// Y[r, n] = B[n] + sum over k of X[first + r, k] W[n, k] in float: an output head (Metal's h3_rows_out).
pub fn rowsOut(y: []f32, x: []const u16, first: usize, w: []const f32, b: []const f32, width: usize, hidden: usize) void {
    var r: usize = 0;
    while (r * width < y.len) : (r += 1) {
        const xs = x[(first + r) * hidden ..][0..hidden];
        for (0..width) |n| {
            var sum: f32 = 0;
            for (xs, w[n * hidden ..][0..hidden]) |v, u| sum += bf.value(v) * u;
            y[r * width + n] = sum + b[n];
        }
    }
}

test {
    _ = @import("cuda_loop_test.zig");
}
