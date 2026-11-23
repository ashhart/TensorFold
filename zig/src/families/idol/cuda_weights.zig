//! Idol's denoiser tensors on CUDA: blocks' linears and biases to fp16 as idol-sdk does, the rest fp32.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const host = @import("host.zig");

const H = cfg.hidden;

/// A block's device addresses: fp16 linears and biases, fp32 head RMSNorm gammas; fp32 norms in the refiner alone.
pub const Block = struct { qkv: u64, qkv_b: u64, out: u64, out_b: u64, up: u64, up_b: u64, down: u64, down_b: u64, q_norm: u64, k_norm: u64, norm1: u64 = 0, norm1_b: u64 = 0, norm2: u64 = 0, norm2_b: u64 = 0 };

/// A RePo layer: fp32 norm, gate | content maps [256, 1024], final map [48, 128] and its 32 frequencies.
pub const Repo = struct { norm: u64, norm_b: u64, maps: u64, final: u64, freqs: u64 };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    ck: core.Checkpoint,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    noise: [cfg.refiners]Block = undefined,
    context: [cfg.refiners]Block = undefined,
    blocks: [cfg.layers]Block = undefined,
    noise_repo: [cfg.refiners]Repo = undefined,
    context_repo: [cfg.refiners]Repo = undefined,
    repo: [cfg.layers]Repo = undefined,
    input: u64 = 0,
    input_b: u64 = 0,
    cond: u64 = 0,
    cond_b: u64 = 0,
    cond2: u64 = 0,
    cond2_b: u64 = 0,
    out: u64 = 0,
    out_b: u64 = 0,
    /// The host's tensors, mapped from the checkpoint (host.modulation, cameraToken, cameraVelocity, positions).
    host: host.Host = undefined,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.ck.close();
    }

    fn typed(w: *Weights, name: []const u8, shape: []const usize) !core.checkpoint.Tensor {
        const t = try w.ck.get(name);
        if (t.dtype != .f32 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("idol: {s} is {t} {any}; the denoiser reads f32 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        if (@intFromPtr(t.bytes.ptr) % 4 != 0) return error.UnalignedTensor;
        return t;
    }

    /// The fp32 tensor `name` of `shape`, its values on the host (mapped from the checkpoint).
    fn floats(w: *Weights, name: []const u8, shape: []const usize) ![]const f32 {
        const t = try w.typed(name, shape);
        return @alignCast(std.mem.bytesAsSlice(f32, t.bytes));
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(d, bytes);
        errdefer b.free();
        try w.buffers.append(w.gpa, b);
        w.bytes += b.len;
        return b.ptr;
    }

    /// The fp32 tensor `name` of `shape` on the device as stored.
    fn take(w: *Weights, d: *const cuda.Driver, name: []const u8, shape: []const usize) !u64 {
        return w.upload(d, std.mem.sliceAsBytes(try w.floats(name, shape)));
    }

    /// The fp32 tensor `name` of `shape` on the device rounded to fp16 (`stage` holds the halves).
    fn half(w: *Weights, d: *const cuda.Driver, stage: []f16, name: []const u8, shape: []const usize) !u64 {
        const src = try w.floats(name, shape);
        for (stage[0..src.len], src) |*o, v| o.* = @floatCast(v);
        return w.upload(d, std.mem.sliceAsBytes(stage[0..src.len]));
    }
};

fn block(w: *Weights, d: *const cuda.Driver, stage: []f16, prefix: []const u8, i: usize, affine: bool) !Block {
    var name: [96]u8 = undefined;
    const N = struct {
        fn n(buf: []u8, p: []const u8, j: usize, s: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}.{d}.{s}", .{ p, j, s });
        }
    };
    var b: Block = undefined;
    b.qkv = try w.half(d, stage, try N.n(&name, prefix, i, "attn.qkv.weight"), &.{ 3 * H, H });
    b.qkv_b = try w.half(d, stage, try N.n(&name, prefix, i, "attn.qkv.bias"), &.{3 * H});
    b.out = try w.half(d, stage, try N.n(&name, prefix, i, "attn.out.weight"), &.{ H, H });
    b.out_b = try w.half(d, stage, try N.n(&name, prefix, i, "attn.out.bias"), &.{H});
    b.up = try w.half(d, stage, try N.n(&name, prefix, i, "mlp.mlp.0.weight"), &.{ cfg.ffn, H });
    b.up_b = try w.half(d, stage, try N.n(&name, prefix, i, "mlp.mlp.0.bias"), &.{cfg.ffn});
    b.down = try w.half(d, stage, try N.n(&name, prefix, i, "mlp.mlp.2.weight"), &.{ H, cfg.ffn });
    b.down_b = try w.half(d, stage, try N.n(&name, prefix, i, "mlp.mlp.2.bias"), &.{H});
    b.q_norm = try w.take(d, try N.n(&name, prefix, i, "attn.q_norm.gamma"), &.{ cfg.heads, cfg.head_dim });
    b.k_norm = try w.take(d, try N.n(&name, prefix, i, "attn.k_norm.gamma"), &.{ cfg.heads, cfg.head_dim });
    b.norm1 = 0;
    b.norm1_b = 0;
    b.norm2 = 0;
    b.norm2_b = 0;
    if (affine) {
        b.norm1 = try w.take(d, try N.n(&name, prefix, i, "norm1.weight"), &.{H});
        b.norm1_b = try w.take(d, try N.n(&name, prefix, i, "norm1.bias"), &.{H});
        b.norm2 = try w.take(d, try N.n(&name, prefix, i, "norm2.weight"), &.{H});
        b.norm2_b = try w.take(d, try N.n(&name, prefix, i, "norm2.bias"), &.{H});
    }
    return b;
}

/// A RePo layer: its gate and content maps stacked [256, 1024] fp32, its three frequency vectors as 32 values.
fn repo(w: *Weights, d: *const cuda.Driver, stage: []f32, prefix: []const u8, i: usize) !Repo {
    var name: [96]u8 = undefined;
    var r: Repo = undefined;
    r.norm = try w.take(d, try std.fmt.bufPrint(&name, "{s}.{d}.norm.weight", .{ prefix, i }), &.{H});
    r.norm_b = try w.take(d, try std.fmt.bufPrint(&name, "{s}.{d}.norm.bias", .{ prefix, i }), &.{H});
    const n = cfg.repo_hidden * H;
    @memcpy(stage[0..n], try w.floats(try std.fmt.bufPrint(&name, "{s}.{d}.gate_map.weight", .{ prefix, i }), &.{ cfg.repo_hidden, H }));
    @memcpy(stage[n .. 2 * n], try w.floats(try std.fmt.bufPrint(&name, "{s}.{d}.content_map.weight", .{ prefix, i }), &.{ cfg.repo_hidden, H }));
    r.maps = try w.upload(d, std.mem.sliceAsBytes(stage[0 .. 2 * n]));
    r.final = try w.take(d, try std.fmt.bufPrint(&name, "{s}.{d}.final_map.weight", .{ prefix, i }), &.{ 3 * cfg.heads, cfg.repo_hidden });
    var freqs: [cfg.repo_pairs]f32 = undefined;
    var at: usize = 0;
    inline for (.{ .{ "freqs_0", 10 }, .{ "freqs_1", 10 }, .{ "freqs_2", 12 } }) |f| {
        @memcpy(freqs[at..][0..f[1]], try w.floats(try std.fmt.bufPrint(&name, "{s}.{d}.{s}", .{ prefix, i, f[0] }), &.{f[1]}));
        at += f[1];
    }
    r.freqs = try w.upload(d, std.mem.sliceAsBytes(&freqs));
    return r;
}

/// The snapshot folder `dir`: config.json checked, then denoiser.safetensors's every tensor, none left unread.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    try cfg.check(gpa, config);
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = .{ .gpa = gpa, .ck = .{ .gpa = gpa, .io = io } };
    errdefer w.deinit();
    try w.ck.add(dir, "denoiser.safetensors");
    const halves = try arena.alloc(f16, cfg.ffn * H);
    const stage = try arena.alloc(f32, 2 * cfg.repo_hidden * H);
    for (&w.noise, &w.context, &w.noise_repo, &w.context_repo, 0..) |*n, *x, *nr, *xr, i| {
        n.* = try block(w, d, halves, "noise_refiner", i, false);
        x.* = try block(w, d, halves, "context_refiner", i, true);
        nr.* = try repo(w, d, stage, "noise_repo_layers", i);
        xr.* = try repo(w, d, stage, "context_repo_layers", i);
    }
    for (&w.blocks, &w.repo, 0..) |*b, *r, i| {
        b.* = try block(w, d, halves, "blocks", i, false);
        r.* = try repo(w, d, stage, "repo_layers", i);
    }
    w.input = try w.take(d, "input_layer.weight", &.{ H, cfg.latent_channels });
    w.input_b = try w.take(d, "input_layer.bias", &.{H});
    w.cond = try w.take(d, "cond_embedder.weight", &.{ H, cfg.cond_channels });
    w.cond_b = try w.take(d, "cond_embedder.bias", &.{H});
    w.cond2 = try w.take(d, "cond_embedder2.weight", &.{ H, cfg.cond2_channels });
    w.cond2_b = try w.take(d, "cond_embedder2.bias", &.{H});
    w.out = try w.take(d, "out_layer.weight", &.{ cfg.latent_channels, H });
    w.out_b = try w.take(d, "out_layer.bias", &.{cfg.latent_channels});
    var hw: host.Host = undefined;
    hw.t1 = try w.floats("t_embedder.mlp.0.weight", &.{ H, cfg.time_dim });
    hw.t1_b = try w.floats("t_embedder.mlp.0.bias", &.{H});
    hw.t2 = try w.floats("t_embedder.mlp.2.weight", &.{ H, H });
    hw.t2_b = try w.floats("t_embedder.mlp.2.bias", &.{H});
    hw.ada = try w.floats("adaLN_modulation.1.weight", &.{ 6 * H, H });
    hw.ada_b = try w.floats("adaLN_modulation.1.bias", &.{6 * H});
    hw.cam1 = try w.floats("cam_refiner.mlp.0.weight", &.{ H, cfg.camera_channels });
    hw.cam1_b = try w.floats("cam_refiner.mlp.0.bias", &.{H});
    hw.cam2 = try w.floats("cam_refiner.mlp.2.weight", &.{ H, H });
    hw.cam2_b = try w.floats("cam_refiner.mlp.2.bias", &.{H});
    hw.cam_out = try w.floats("cam_out_layer.weight", &.{ cfg.camera_channels, H });
    hw.cam_out_b = try w.floats("cam_out_layer.bias", &.{cfg.camera_channels});
    var name: [64]u8 = undefined;
    for (&hw.shifts, 0..) |*s, i| {
        const n = if (i < cfg.refiners) try std.fmt.bufPrint(&name, "noise_refiner.{d}.shift_table", .{i}) else try std.fmt.bufPrint(&name, "blocks.{d}.shift_table", .{i - cfg.refiners});
        s.* = try w.floats(n, &.{ 1, 6 * H });
    }
    hw.head = try w.floats("shift_table", &.{ 1, 2, H });
    hw.pos = try w.floats("pos_pe", &.{ 1, cfg.latent_rows, 3 });
    w.host = hw;
    if (w.ck.unused() != 0) {
        std.log.err("idol: denoiser.safetensors has {d} tensors this family does not read", .{w.ck.unused()});
        return error.UnexpectedTensors;
    }
    return w;
}
