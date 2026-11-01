//! DINOv3 ViT-H+/16's tensors on CUDA from timm's fp32 safetensors, each rounded to bf16 at load as the SDK does.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("dino_config.zig");
const host = @import("dino_host.zig");

const W = cfg.width;
const F = cfg.ffn;

/// A block's bf16 tensors on the device: fused qkv [3840, 1280], gate | up [10240, 1280] as one, LayerScale's gammas.
pub const Block = struct { qkv: u64, proj: u64, proj_b: u64, fc1: u64, fc1_b: u64, fc2: u64, fc2_b: u64, norm1: u64, norm1_b: u64, norm2: u64, norm2_b: u64, gamma1: u64, gamma2: u64 };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    ck: core.Checkpoint,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    blocks: [cfg.layers]Block = undefined,
    /// The conv weight [1280, 768] and bias; the class token then the four registers [5, 1280]; the last norm.
    patch: u64 = 0,
    patch_b: u64 = 0,
    prefix: u64 = 0,
    norm: u64 = 0,
    norm_b: u64 = 0,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.ck.close();
    }

    /// The fp32 tensor `name` holding `n` values (any shape of that count, `rows` leading when given).
    fn floats(w: *Weights, name: []const u8, n: usize, rows: ?usize) ![]align(1) const f32 {
        const t = try w.ck.get(name);
        var count: usize = 1;
        for (t.shape[0..t.rank]) |s| count *= s;
        if (t.dtype != .f32 or count != n or t.bytes.len != 4 * n or (rows != null and (t.rank < 1 or t.shape[0] != rows.?))) {
            std.log.err("idol dino: {s} is {t} {any}; the encoder reads f32 of {d} values", .{ name, t.dtype, t.shape[0..t.rank], n });
            return error.UnexpectedTensor;
        }
        return @as([*]align(1) const f32, @ptrCast(t.bytes.ptr))[0..n];
    }

    /// `stage`'s first `n` halves on the device.
    fn upload(w: *Weights, d: *const cuda.Driver, stage: []const u16) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(stage));
        errdefer b.free();
        try w.buffers.append(w.gpa, b);
        w.bytes += b.len;
        return b.ptr;
    }

    /// The tensors `names` (`n` values each), rounded to bf16 into `stage` one after another, then uploaded as one.
    fn take(w: *Weights, d: *const cuda.Driver, stage: []u16, names: []const []const u8, n: usize, rows: ?usize) !u64 {
        for (names, 0..) |name, i| {
            const src = try w.floats(name, n, rows);
            for (stage[i * n ..][0..n], src) |*o, v| o.* = host.bits(v);
        }
        return w.upload(d, stage[0 .. names.len * n]);
    }
};

/// The bf16 bytes the encoder's weights take on the device.
pub fn deviceBytes() usize {
    const per_block = 3 * W * W + W * W + W + 2 * F * W + 2 * F + W * F + 7 * W;
    return 2 * (W * cfg.patch_k + W + cfg.prefix * W + 2 * W + cfg.layers * per_block);
}

fn block(w: *Weights, d: *const cuda.Driver, stage: []u16, i: usize) !Block {
    var names: [2][96]u8 = undefined;
    const N = struct {
        fn n(buf: *[96]u8, j: usize, s: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "blocks.{d}.{s}", .{ j, s });
        }
    };
    var b: Block = undefined;
    b.qkv = try w.take(d, stage, &.{try N.n(&names[0], i, "attn.qkv.weight")}, 3 * W * W, 3 * W);
    b.proj = try w.take(d, stage, &.{try N.n(&names[0], i, "attn.proj.weight")}, W * W, W);
    b.proj_b = try w.take(d, stage, &.{try N.n(&names[0], i, "attn.proj.bias")}, W, W);
    b.fc1 = try w.take(d, stage, &.{ try N.n(&names[0], i, "mlp.fc1_g.weight"), try N.n(&names[1], i, "mlp.fc1_x.weight") }, F * W, F);
    b.fc1_b = try w.take(d, stage, &.{ try N.n(&names[0], i, "mlp.fc1_g.bias"), try N.n(&names[1], i, "mlp.fc1_x.bias") }, F, F);
    b.fc2 = try w.take(d, stage, &.{try N.n(&names[0], i, "mlp.fc2.weight")}, W * F, W);
    b.fc2_b = try w.take(d, stage, &.{try N.n(&names[0], i, "mlp.fc2.bias")}, W, W);
    const vecs = [_]struct { *u64, []const u8 }{
        .{ &b.norm1, "norm1.weight" }, .{ &b.norm1_b, "norm1.bias" }, .{ &b.norm2, "norm2.weight" },
        .{ &b.norm2_b, "norm2.bias" }, .{ &b.gamma1, "gamma_1" },     .{ &b.gamma2, "gamma_2" },
    };
    for (vecs) |v| v[0].* = try w.take(d, stage, &.{try N.n(&names[0], i, v[1])}, W, W);
    return b;
}

/// The timm snapshot folder `dir`: config.json checked, then model.safetensors's 486 tensors, each read once into bf16.
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
    try w.ck.add(dir, "model.safetensors");
    // the largest tensor group: gate's and up's weights, 2 x 5,120 x 1,280
    const stage = try arena.alloc(u16, 2 * F * W);
    w.patch = try w.take(d, stage, &.{"patch_embed.proj.weight"}, W * cfg.patch_k, W);
    w.patch_b = try w.take(d, stage, &.{"patch_embed.proj.bias"}, W, W);
    // the class token, then the four registers: rows 0 to 4 of the stream
    for (stage[0..W], try w.floats("cls_token", W, null)) |*o, v| o.* = host.bits(v);
    for (stage[W..][0 .. (cfg.prefix - 1) * W], try w.floats("reg_token", (cfg.prefix - 1) * W, null)) |*o, v| o.* = host.bits(v);
    w.prefix = try w.upload(d, stage[0 .. cfg.prefix * W]);
    w.norm = try w.take(d, stage, &.{"norm.weight"}, W, W);
    w.norm_b = try w.take(d, stage, &.{"norm.bias"}, W, W);
    for (&w.blocks, 0..) |*b, i| b.* = try block(w, d, stage, i);
    if (w.ck.unused() != 0) {
        std.log.err("idol dino: model.safetensors has {d} tensors the encoder does not read", .{w.ck.unused()});
        return error.UnexpectedTensors;
    }
    std.debug.assert(w.bytes >= deviceBytes());
    return w;
}

test "the device's bf16 weights: 840.5 million values, under 1.7 GB" {
    try std.testing.expectEqual(@as(usize, 1_681_018_880), deviceBytes());
    try std.testing.expect(deviceBytes() < 1_700_000_000);
}
