//! Idol's denoiser tensors (fp32 as stored): the blocks' linears to fp16 at load as idol-sdk does, the rest fp32.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

const H = cfg.hidden;

/// A transformer block: fp16 linears and biases, fp32 head RMSNorm gammas [16, 64]; norms only in the refiner.
pub const Block = struct { qkv: Ref, qkv_b: Ref, out: Ref, out_b: Ref, up: Ref, up_b: Ref, down: Ref, down_b: Ref, q_norm: Ref, k_norm: Ref, norm1: Ref = undefined, norm1_b: Ref = undefined, norm2: Ref = undefined, norm2_b: Ref = undefined };

/// A RePo layer: fp32 norm; gate | content maps over `inv` (a power of two) as fp16 [hi | lo | hi]; final map, freqs.
pub const Repo = struct { norm: Ref, norm_b: Ref, maps: Ref, inv: f32, final: Ref, freqs: Ref };

const repos = 2 * cfg.refiners + cfg.layers;
const map_bytes = 2 * cfg.repo_hidden * 3 * H * 2;

/// The power of two that puts a map's largest magnitude in [2^13, 2^14), so its fp16 halves keep their bits.
fn mapScale(gate: []const f32, content: []const f32) f32 {
    var most: f32 = 0;
    for (gate) |v| most = @max(most, @abs(v));
    for (content) |v| most = @max(most, @abs(v));
    if (most == 0 or !std.math.isFinite(most)) return 1;
    return std.math.ldexp(@as(f32, 1), 13 - std.math.ilogb(most));
}

/// Row `n` of a map as [hi | lo | hi] fp16 of `scale` times its values.
fn split(src: []const f32, scale: f32, dst: []f16) void {
    for (src, 0..) |v, i| {
        const x = v * scale;
        const hi: f16 = @floatCast(x);
        dst[i] = hi;
        dst[H + i] = @floatCast(x - @as(f32, hi));
        dst[2 * H + i] = hi;
    }
}

/// The fp32 vectors and small matrices the host reads each timestep (host.zig, shared with CUDA).
pub const Host = @import("host.zig").Host;

pub const Weights = struct {
    ck: ck.Checkpoint,
    noise: [cfg.refiners]Block,
    context: [cfg.refiners]Block,
    blocks: [cfg.layers]Block,
    noise_repo: [cfg.refiners]Repo,
    context_repo: [cfg.refiners]Repo,
    repo: [cfg.layers]Repo,
    input: Ref,
    input_b: Ref,
    cond: Ref,
    cond_b: Ref,
    cond2: Ref,
    cond2_b: Ref,
    out: Ref,
    out_b: Ref,
    host: Host,

    pub fn deinit(w: *Weights) void {
        w.ck.deinit();
    }
};

const linears = [_][]const u8{ "attn.qkv", "attn.out", "mlp.mlp.0", "mlp.mlp.2" };

/// The blocks' linear weights stay in the file: load reads each once into its fp16 copy.
fn later(name: []const u8, rank: usize) bool {
    if (rank != 2) return false;
    const owned = std.mem.startsWith(u8, name, "blocks.") or std.mem.startsWith(u8, name, "noise_refiner.") or std.mem.startsWith(u8, name, "context_refiner.");
    return owned and std.mem.endsWith(u8, name, ".weight");
}

fn check(t: ck.Tensor, name: []const u8, shape: []const usize) !void {
    if (t.dtype != .f32 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("idol: {s} is {t} {any}; the denoiser reads f32 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
        return error.UnexpectedTensor;
    }
}

fn take(c: *const ck.Checkpoint, name: []const u8, shape: []const usize) !Ref {
    const t = try c.get(name);
    try check(t, name, shape);
    if (t.offset % 16 != 0) return error.UnalignedTensor;
    return .{ .buf = t.buffer, .off = t.offset };
}

fn floats(c: *const ck.Checkpoint, name: []const u8, shape: []const usize) ![]const f32 {
    const t = try c.get(name);
    try check(t, name, shape);
    return t.host(f32);
}

/// fp16 values written into an owned buffer, bump-allocated 256-aligned.
const Halves = struct {
    buf: mtl.Buffer,
    at: usize = 0,
    stage: []f32,

    fn put(h: *Halves, src: []const f32) Ref {
        const dst = @as([*]f16, @ptrCast(@alignCast(h.buf.contents() + h.at)))[0..src.len];
        for (dst, src) |*o, v| o.* = @floatCast(v);
        const r: Ref = .{ .buf = h.buf, .off = h.at };
        h.at += std.mem.alignForward(usize, src.len * 2, ck.alignment);
        return r;
    }

    /// A deferred fp32 weight read in staged pieces, rounded to fp16.
    fn read(h: *Halves, c: *const ck.Checkpoint, name: []const u8, shape: [2]usize) !Ref {
        const d = c.deferred.get(name) orelse return error.MissingTensor;
        if (d.dtype != .f32 or d.rank != 2 or d.shape[0] != shape[0] or d.shape[1] != shape[1]) {
            std.log.err("idol: {s} is {t} {any}; the denoiser reads f32 {any}", .{ name, d.dtype, d.shape[0..d.rank], shape });
            return error.UnexpectedTensor;
        }
        const n = shape[0] * shape[1];
        const r: Ref = .{ .buf = h.buf, .off = h.at };
        const dst = @as([*]f16, @ptrCast(@alignCast(h.buf.contents() + h.at)))[0..n];
        var done: usize = 0;
        while (done < n) {
            const part = @min(h.stage.len, n - done);
            try c.read(name, done * 4, std.mem.sliceAsBytes(h.stage[0..part]));
            for (dst[done..][0..part], h.stage[0..part]) |*o, v| o.* = @floatCast(v);
            done += part;
        }
        h.at += std.mem.alignForward(usize, n * 2, ck.alignment);
        return r;
    }
};

const per_block = blk: {
    var n: usize = 0;
    for ([_]usize{ 3 * H * H, 3 * H, H * H, H, cfg.ffn * H, cfg.ffn, H * cfg.ffn, H }) |v| n += std.mem.alignForward(usize, v * 2, ck.alignment);
    break :blk n;
};

fn block(c: *const ck.Checkpoint, h: *Halves, prefix: []const u8, i: usize, affine: bool) !Block {
    var name: [96]u8 = undefined;
    const P = struct {
        fn n(buf: []u8, p: []const u8, j: usize, s: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}.{d}.{s}", .{ p, j, s });
        }
    };
    var b: Block = undefined;
    b.qkv = try h.read(c, try P.n(&name, prefix, i, "attn.qkv.weight"), .{ 3 * H, H });
    b.qkv_b = h.put(try floats(c, try P.n(&name, prefix, i, "attn.qkv.bias"), &.{3 * H}));
    b.out = try h.read(c, try P.n(&name, prefix, i, "attn.out.weight"), .{ H, H });
    b.out_b = h.put(try floats(c, try P.n(&name, prefix, i, "attn.out.bias"), &.{H}));
    b.up = try h.read(c, try P.n(&name, prefix, i, "mlp.mlp.0.weight"), .{ cfg.ffn, H });
    b.up_b = h.put(try floats(c, try P.n(&name, prefix, i, "mlp.mlp.0.bias"), &.{cfg.ffn}));
    b.down = try h.read(c, try P.n(&name, prefix, i, "mlp.mlp.2.weight"), .{ H, cfg.ffn });
    b.down_b = h.put(try floats(c, try P.n(&name, prefix, i, "mlp.mlp.2.bias"), &.{H}));
    b.q_norm = try take(c, try P.n(&name, prefix, i, "attn.q_norm.gamma"), &.{ cfg.heads, cfg.head_dim });
    b.k_norm = try take(c, try P.n(&name, prefix, i, "attn.k_norm.gamma"), &.{ cfg.heads, cfg.head_dim });
    if (affine) {
        b.norm1 = try take(c, try P.n(&name, prefix, i, "norm1.weight"), &.{H});
        b.norm1_b = try take(c, try P.n(&name, prefix, i, "norm1.bias"), &.{H});
        b.norm2 = try take(c, try P.n(&name, prefix, i, "norm2.weight"), &.{H});
        b.norm2_b = try take(c, try P.n(&name, prefix, i, "norm2.bias"), &.{H});
    }
    return b;
}

/// A RePo layer; its two maps into `pack` at `slot`, and its three frequency vectors as 32 values after every map.
fn repo(c: *const ck.Checkpoint, pack: mtl.Buffer, slot: usize, prefix: []const u8, i: usize) !Repo {
    var name: [96]u8 = undefined;
    var r: Repo = undefined;
    r.norm = try take(c, try std.fmt.bufPrint(&name, "{s}.{d}.norm.weight", .{ prefix, i }), &.{H});
    r.norm_b = try take(c, try std.fmt.bufPrint(&name, "{s}.{d}.norm.bias", .{ prefix, i }), &.{H});
    const gate = try floats(c, try std.fmt.bufPrint(&name, "{s}.{d}.gate_map.weight", .{ prefix, i }), &.{ cfg.repo_hidden, H });
    const content = try floats(c, try std.fmt.bufPrint(&name, "{s}.{d}.content_map.weight", .{ prefix, i }), &.{ cfg.repo_hidden, H });
    const scale = mapScale(gate, content);
    const maps = @as([*]f16, @ptrCast(@alignCast(pack.contents() + slot * map_bytes)))[0 .. 2 * cfg.repo_hidden * 3 * H];
    for (0..2 * cfg.repo_hidden) |n| {
        const src = if (n < cfg.repo_hidden) gate[n * H ..][0..H] else content[(n - cfg.repo_hidden) * H ..][0..H];
        split(src, scale, maps[n * 3 * H ..][0 .. 3 * H]);
    }
    r.maps = .{ .buf = pack, .off = slot * map_bytes };
    r.inv = 1 / scale;
    r.final = try take(c, try std.fmt.bufPrint(&name, "{s}.{d}.final_map.weight", .{ prefix, i }), &.{ 3 * cfg.heads, cfg.repo_hidden });
    const freqs = repos * map_bytes;
    const dst = @as([*]f32, @ptrCast(@alignCast(pack.contents() + freqs)))[slot * cfg.repo_pairs ..][0..cfg.repo_pairs];
    var at: usize = 0;
    inline for (.{ .{ "freqs_0", 10 }, .{ "freqs_1", 10 }, .{ "freqs_2", 12 } }) |f| {
        const src = try floats(c, try std.fmt.bufPrint(&name, "{s}.{d}.{s}", .{ prefix, i, f[0] }), &.{f[1]});
        @memcpy(dst[at..][0..f[1]], src);
        at += f[1];
    }
    r.freqs = .{ .buf = pack, .off = freqs + slot * cfg.repo_pairs * 4 };
    return r;
}

/// The snapshot folder `dir`: config.json checked, then denoiser.safetensors's every tensor, each read once.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    try cfg.check(gpa, config);
    var w: Weights = undefined;
    w.ck = ck.Checkpoint.init(gpa);
    errdefer w.ck.deinit();
    const path = try std.fmt.allocPrintSentinel(arena, "{s}/denoiser.safetensors", .{dir}, 0);
    try w.ck.addFileSplit(device, path, "", null, later);
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const blocks = 2 * cfg.refiners + cfg.layers;
    const fp16 = try device.buffer(blocks * per_block, opts);
    w.ck.own(fp16, blocks * per_block) catch |err| {
        fp16.deinit();
        return err;
    };
    const pack_bytes = repos * (map_bytes + cfg.repo_pairs * 4);
    const pack = try device.buffer(pack_bytes, opts);
    w.ck.own(pack, pack_bytes) catch |err| {
        pack.deinit();
        return err;
    };
    var h: Halves = .{ .buf = fp16, .stage = try arena.alloc(f32, 4 << 20) };
    for (&w.noise, &w.context, &w.noise_repo, &w.context_repo, 0..) |*n, *x, *nr, *xr, i| {
        n.* = try block(&w.ck, &h, "noise_refiner", i, false);
        x.* = try block(&w.ck, &h, "context_refiner", i, true);
        nr.* = try repo(&w.ck, pack, i, "noise_repo_layers", i);
        xr.* = try repo(&w.ck, pack, cfg.refiners + i, "context_repo_layers", i);
    }
    for (&w.blocks, &w.repo, 0..) |*b, *r, i| {
        b.* = try block(&w.ck, &h, "blocks", i, false);
        r.* = try repo(&w.ck, pack, 2 * cfg.refiners + i, "repo_layers", i);
    }
    w.ck.closeFiles();
    w.input = try take(&w.ck, "input_layer.weight", &.{ H, cfg.latent_channels });
    w.input_b = try take(&w.ck, "input_layer.bias", &.{H});
    w.cond = try take(&w.ck, "cond_embedder.weight", &.{ H, cfg.cond_channels });
    w.cond_b = try take(&w.ck, "cond_embedder.bias", &.{H});
    w.cond2 = try take(&w.ck, "cond_embedder2.weight", &.{ H, cfg.cond2_channels });
    w.cond2_b = try take(&w.ck, "cond_embedder2.bias", &.{H});
    w.out = try take(&w.ck, "out_layer.weight", &.{ cfg.latent_channels, H });
    w.out_b = try take(&w.ck, "out_layer.bias", &.{cfg.latent_channels});
    var hw: Host = undefined;
    hw.t1 = try floats(&w.ck, "t_embedder.mlp.0.weight", &.{ H, cfg.time_dim });
    hw.t1_b = try floats(&w.ck, "t_embedder.mlp.0.bias", &.{H});
    hw.t2 = try floats(&w.ck, "t_embedder.mlp.2.weight", &.{ H, H });
    hw.t2_b = try floats(&w.ck, "t_embedder.mlp.2.bias", &.{H});
    hw.ada = try floats(&w.ck, "adaLN_modulation.1.weight", &.{ 6 * H, H });
    hw.ada_b = try floats(&w.ck, "adaLN_modulation.1.bias", &.{6 * H});
    hw.cam1 = try floats(&w.ck, "cam_refiner.mlp.0.weight", &.{ H, cfg.camera_channels });
    hw.cam1_b = try floats(&w.ck, "cam_refiner.mlp.0.bias", &.{H});
    hw.cam2 = try floats(&w.ck, "cam_refiner.mlp.2.weight", &.{ H, H });
    hw.cam2_b = try floats(&w.ck, "cam_refiner.mlp.2.bias", &.{H});
    hw.cam_out = try floats(&w.ck, "cam_out_layer.weight", &.{ cfg.camera_channels, H });
    hw.cam_out_b = try floats(&w.ck, "cam_out_layer.bias", &.{cfg.camera_channels});
    var name: [64]u8 = undefined;
    for (&hw.shifts, 0..) |*s, i| {
        const n = if (i < cfg.refiners) try std.fmt.bufPrint(&name, "noise_refiner.{d}.shift_table", .{i}) else try std.fmt.bufPrint(&name, "blocks.{d}.shift_table", .{i - cfg.refiners});
        s.* = try floats(&w.ck, n, &.{ 1, 6 * H });
    }
    hw.head = try floats(&w.ck, "shift_table", &.{ 1, 2, H });
    hw.pos = try floats(&w.ck, "pos_pe", &.{ 1, cfg.latent_rows, 3 });
    w.host = hw;
    // a block's 4 linears, 4 biases and 2 gammas, the context refiner's 4 norms, the shift tables; 8 a RePo layer
    const all = 2 * cfg.refiners + cfg.layers;
    const read = 10 * all + 4 * cfg.refiners + cfg.modulated + 8 * all + 8 + 12 + 2;
    const total = w.ck.tensors.count() + w.ck.deferred.count();
    if (total != read) {
        std.log.err("idol: denoiser.safetensors has {d} tensors; this family reads {d}", .{ total, read });
        return error.UnexpectedTensors;
    }
    return w;
}

test "a map's split halves sum back to its scaled values within fp32's rounding" {
    var src: [H]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(3);
    for (&src) |*v| v.* = 0.05 * rng.random().floatNorm(f32);
    const scale = mapScale(&src, &src);
    try std.testing.expect(scale >= 1);
    var dst: [3 * H]f16 = undefined;
    split(&src, scale, &dst);
    for (src, 0..) |v, i| {
        const back = (@as(f32, dst[i]) + @as(f32, dst[H + i])) / scale;
        try std.testing.expect(@abs(back - v) <= @abs(v) * 1e-6 + 1e-12);
        try std.testing.expectEqual(dst[i], dst[2 * H + i]);
    }
}

test "the blocks' linear weights, and only they, stay in the file for their fp16 copies" {
    try std.testing.expect(later("blocks.3.attn.qkv.weight", 2));
    try std.testing.expect(later("context_refiner.1.mlp.mlp.2.weight", 2));
    try std.testing.expect(!later("blocks.3.attn.qkv.bias", 1));
    try std.testing.expect(!later("blocks.3.attn.q_norm.gamma", 2));
    try std.testing.expect(!later("context_refiner.0.norm1.weight", 1));
    try std.testing.expect(!later("repo_layers.0.gate_map.weight", 2));
    try std.testing.expect(!later("cond_embedder.weight", 2));
}
