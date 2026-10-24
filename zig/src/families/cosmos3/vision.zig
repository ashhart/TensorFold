//! The image tower on Metal: SigLIP2 patch embedding plus the learned 16 x 16 position grid resized to the image's,
//! 27 pre-norm encoder blocks with bidirectional attention over the image's patches, then the projector: LayerNorm,
//! four neighbouring patches (block-major order) as one 4608-wide row, two linears with an exact GELU between.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const Buffer = mtl.Buffer;
const Ref = fwd.Ref;

pub const Tower = struct {
    memory: st.Storage,
    patches_most: usize,
    patches: Buffer, // bf16 [P, 768], written by the host
    pos: Buffer, // bf16 [P, 1152], written by the host
    h: Buffer,
    x: Buffer,
    r: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    att: Buffer,
    mlp: Buffer,
    merged: Buffer,
    sums: Buffer,
    /// The learned grid in f32 [16, 16, 1152], read once at load.
    grid: []f32,

    /// Scratch for images of up to `tokens` text rows (4 patches each).
    pub fn init(gpa: std.mem.Allocator, m: *const Model, tokens: usize) !*Tower {
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        const P = tokens * c.merge * c.merge;
        t.* = .{ .memory = .{ .gpa = gpa, .device = m.device }, .patches_most = P, .patches = undefined, .pos = undefined, .h = undefined, .x = undefined, .r = undefined, .q = undefined, .k = undefined, .v = undefined, .att = undefined, .mlp = undefined, .merged = undefined, .sums = undefined, .grid = undefined };
        errdefer t.memory.deinit();
        const mem = &t.memory;
        t.patches = try mem.alloc(P * c.patch_width * 2);
        // q has 32 rows of slack: the last flash tile reads whole 8-row query tiles past the image's rows
        inline for (.{ "pos", "h", "x", "r", "q", "k", "v", "att" }) |name| @field(t, name) = try mem.alloc((P + 32) * c.v_hidden * 2);
        t.mlp = try mem.alloc(P * c.v_intermediate * 2);
        t.merged = try mem.alloc(tokens * c.projector_hidden * 2);
        t.sums = try mem.alloc(@max(P * (c.v_intermediate / 64), tokens * (c.projector_hidden / 64)) * 4);
        t.grid = try gpa.alloc(f32, c.grid * c.grid * c.v_hidden);
        errdefer gpa.free(t.grid);
        const table = m.weights.vision.position;
        const words: [*]const u16 = @ptrCast(@alignCast(table.buffer.contents() + table.offset));
        for (t.grid, words[0..t.grid.len]) |*g, w| g.* = @bitCast(@as(u32, w) << 16);
        return t;
    }

    pub fn deinit(t: *Tower, gpa: std.mem.Allocator) void {
        gpa.free(t.grid);
        t.memory.deinit();
        gpa.destroy(t);
    }

    pub fn setResident(t: *Tower, hook: ?mtl.ResidencySet.Follow) void {
        t.memory.setResident(hook);
    }

    /// One image's patches (f32 [gh * gw, 768] block-major, HWC within a patch) to its text rows ([gh * gw / 4, 2048] at `out`).
    pub fn encode(t: *Tower, m: *Model, pixels: []const f32, gh: u32, gw: u32, out: Ref) !void {
        const P: usize = @as(usize, gh) * gw;
        if (gh % c.merge != 0 or gw % c.merge != 0 or P == 0) return error.CosmosImageGrid;
        if (P > t.patches_most) return error.ImageTooLarge;
        if (pixels.len != P * c.patch_width) return error.CosmosImageGrid;
        for (t.patches.slice(u16, P * c.patch_width), pixels) |*d, x| d.* = bf16(x);
        try positions(m.gpa, t.grid, gh, gw, t.pos.slice(u16, P * c.v_hidden));
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const enc = fwd.Encoder{ .e = cb.compute(.serial), .m = m, .sums = t.sums, .prompt = true };
        const e = enc.e;
        const v = &m.weights.vision;
        const W = c.v_hidden;
        enc.biased(v.patch, t.patches, 0, t.h, 0, P, .none);
        enc.pipe("cos3_add");
        e.setBuffer(t.h, 0, 0);
        e.setBuffer(t.pos, 0, 1);
        e.setValue(@as(u32, @intCast(P * W)), 2);
        enc.threads(.{ P * W, 1, 1 }, .{ 256, 1, 1 });
        const h: Ref = .{ .buf = t.h };
        const r: Ref = .{ .buf = t.r };
        const x: Ref = .{ .buf = t.x };
        for (v.layers, 0..) |l, i| {
            enc.layerNorm(l.ln1, h, if (i == 0) null else r, x, P, W, c.v_eps);
            enc.biased(l.q, t.x, 0, t.q, 0, P, .none);
            enc.biased(l.k, t.x, 0, t.k, 0, P, .none);
            enc.biased(l.v, t.x, 0, t.v, 0, P, .none);
            enc.pipe("cos3_flash_72");
            e.setBuffer(t.q, 0, 0);
            e.setBuffer(t.k, 0, 1);
            e.setBuffer(t.v, 0, 2);
            const args = fwd.AttnArgs{
                .first = 0,
                .keys = @intCast(P),
                .causal = 0,
                .group = 1,
                .q_row = W,
                .k_row = W,
                .k_head = c.v_head_dim,
                .o_row = W,
                .scale = 1.0 / @sqrt(@as(f32, c.v_head_dim)),
            };
            e.setBytes(std.mem.asBytes(&args), 3);
            e.setBuffer(t.att, 0, 4);
            e.setValue(@as(u32, @intCast(P)), 5);
            enc.groups(.{ c.v_heads, (P + 31) / 32, 1 }, .{ 128, 1, 1 });
            enc.biased(l.out, t.att, 0, t.r, 0, P, .none);
            enc.layerNorm(l.ln2, h, r, x, P, W, c.v_eps);
            enc.biased(l.fc1, t.x, 0, t.mlp, 0, P, .gelu_tanh);
            enc.biased(l.fc2, t.mlp, 0, t.r, 0, P, .none);
        }
        enc.layerNorm(v.post, h, r, x, P, W, c.v_eps);
        // the projector: each patch normed on its own, then four neighbours (consecutive in block-major order) one row
        enc.layerNorm(v.merge_norm, x, null, .{ .buf = t.q }, P, W, 1e-6);
        const rows = P / (c.merge * c.merge);
        enc.biased(v.fc1, t.q, 0, t.merged, 0, rows, .gelu_erf);
        enc.biased(v.fc2, t.merged, 0, out.buf, out.off, rows, .none);
        e.end();
        cb.commit();
        cb.wait();
        fwd.gpu_seconds += cb.gpuSeconds();
        if (cb.failure()) |msg| {
            std.debug.print("Cosmos3 image tower: {s}\n", .{msg});
            return error.GpuFailed;
        }
    }
};

/// f32 to bf16, round to nearest even (torch's .to(bfloat16)).
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7fc0;
    const rounded = b +% 0x7fff +% ((b >> 16) & 1);
    return @truncate(rounded >> 16);
}

/// torch's antialiased bilinear weights (F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)) for
/// one axis from `in` to `out` samples: each output's first input and its weights, normalized, in f32 as on the CPU.
const Taps = struct { first: []usize, count: []usize, weights: []f32, width: usize };

fn taps(gpa: std.mem.Allocator, in: usize, out: usize) !Taps {
    const scale: f32 = @as(f32, @floatFromInt(in)) / @as(f32, @floatFromInt(out));
    const support: f32 = if (scale >= 1.0) scale else 1.0;
    const invscale: f32 = if (scale >= 1.0) 1.0 / scale else 1.0;
    const width: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    var t: Taps = .{ .first = try gpa.alloc(usize, out), .count = try gpa.alloc(usize, out), .weights = try gpa.alloc(f32, out * width), .width = width };
    @memset(t.weights, 0);
    for (0..out) |i| {
        const center = scale * (@as(f32, @floatFromInt(i)) + 0.5);
        const lo: i64 = @intFromFloat(@trunc(center - support + 0.5));
        const hi: i64 = @intFromFloat(@trunc(center + support + 0.5));
        const xmin: usize = @intCast(@max(lo, 0));
        const xsize: usize = @intCast(@min(hi, @as(i64, @intCast(in))) - @as(i64, @intCast(xmin)));
        var total: f32 = 0;
        const w = t.weights[i * width ..][0..width];
        for (0..xsize) |j| {
            var d = (@as(f32, @floatFromInt(j + xmin)) - center + 0.5) * invscale;
            if (d < 0) d = -d;
            w[j] = if (d < 1.0) 1.0 - d else 0.0;
            total += w[j];
        }
        if (total != 0) for (w[0..xsize]) |*x| {
            x.* /= total;
        };
        t.first[i] = xmin;
        t.count[i] = xsize;
    }
    return t;
}

fn freeTaps(gpa: std.mem.Allocator, t: Taps) void {
    gpa.free(t.first);
    gpa.free(t.count);
    gpa.free(t.weights);
}

/// The 16 x 16 grid resized to gh x gw (width pass, then height pass, f32), rounded to bf16, in block-major order.
pub fn positions(gpa: std.mem.Allocator, grid: []const f32, gh: u32, gw: u32, out: []u16) !void {
    const W = c.v_hidden;
    const G = c.grid;
    const tx = try taps(gpa, G, gw);
    defer freeTaps(gpa, tx);
    const ty = try taps(gpa, G, gh);
    defer freeTaps(gpa, ty);
    const wide = try gpa.alloc(f32, G * gw * W);
    defer gpa.free(wide);
    @memset(wide, 0);
    for (0..G) |y| for (0..gw) |x| {
        const dst = wide[(y * gw + x) * W ..][0..W];
        for (0..tx.count[x]) |j| {
            const wt = tx.weights[x * tx.width + j];
            const src = grid[(y * G + tx.first[x] + j) * W ..][0..W];
            for (dst, src) |*d, s| d.* += wt * s;
        }
    };
    const acc = try gpa.alloc(f32, W);
    defer gpa.free(acc);
    const M = c.merge;
    for (0..gh) |y| for (0..gw) |x| {
        @memset(acc, 0);
        for (0..ty.count[y]) |j| {
            const wt = ty.weights[y * ty.width + j];
            const src = wide[((ty.first[y] + j) * gw + x) * W ..][0..W];
            for (acc, src) |*d, s| d.* += wt * s;
        }
        // block-major: (block row, block column, row in block, column in block)
        const n = ((y / M) * (gw / M) + x / M) * M * M + (y % M) * M + x % M;
        for (out[n * W ..][0..W], acc) |*d, s| d.* = bf16(s);
    };
}

test "an upsampled grid keeps its corners' mix and a same-size grid is the identity" {
    const gpa = std.testing.allocator;
    const t = try taps(gpa, 16, 16);
    defer freeTaps(gpa, t);
    for (0..16) |i| {
        try std.testing.expectEqual(i, t.first[i]);
        try std.testing.expectEqual(@as(f32, 1), t.weights[i * t.width]);
    }
    const up = try taps(gpa, 16, 40);
    defer freeTaps(gpa, up);
    for (0..40) |i| {
        var s: f32 = 0;
        for (up.weights[i * up.width ..][0..up.count[i]]) |w| s += w;
        try std.testing.expectApproxEqAbs(@as(f32, 1), s, 1e-6);
    }
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.00390625)); // halfway: to even
}
