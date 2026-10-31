//! DINOv3 ViT-H+/16's tensors from timm's fp32 safetensors, each rounded to bf16 at load as the SDK's dtype does.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const cfg = @import("dino_config.zig");
const host = @import("dino_host.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

const W = cfg.width;

/// A block's bf16 tensors: the fused qkv [3840, 1280], gate | up [10240, 1280] as one, LayerScale's two gammas.
pub const Block = struct { qkv: Ref, proj: Ref, proj_b: Ref, fc1: Ref, fc1_b: Ref, fc2: Ref, fc2_b: Ref, norm1: Ref, norm1_b: Ref, norm2: Ref, norm2_b: Ref, gamma1: Ref, gamma2: Ref };

pub const Weights = struct {
    ck: ck.Checkpoint,
    blocks: [cfg.layers]Block,
    /// The conv weight [1280, 768] and bias; the class token then the four registers [5, 1280]; the last norm.
    patch: Ref,
    patch_b: Ref,
    prefix: Ref,
    norm: Ref,
    norm_b: Ref,
    /// bf16 zeros [3840]: the qkv projection's bias (q and v's are zero, k has none).
    zeros: Ref,

    pub fn deinit(w: *Weights) void {
        w.ck.deinit();
    }
};

fn always(_: []const u8, _: usize) bool {
    return true;
}

/// One tensor's place in the bf16 buffer: its name, the values it holds, and where they go.
const Part = struct { name: []const u8, shape: [2]usize, at: usize };

fn bytes(n: usize) usize {
    return std.mem.alignForward(usize, n * 2, ck.alignment);
}

const per_block = blk: {
    var n: usize = 0;
    for ([_]usize{ 3 * W * W, W * W, W, 2 * cfg.ffn * W, 2 * cfg.ffn, W * cfg.ffn, W, W, W, W, W, W, W }) |v| n += bytes(v);
    break :blk n;
};
const globals = bytes(W * cfg.patch_k) + bytes(W) + bytes(cfg.prefix * W) + 2 * bytes(W) + bytes(3 * W);
pub const total_bytes = globals + cfg.layers * per_block;

/// Reads deferred fp32 tensors into bf16 at their places, through its own staging buffer.
const Reader = struct {
    c: *const ck.Checkpoint,
    base: [*]u8,
    stage: []f32,

    fn put(r: *Reader, p: Part) !void {
        const d = r.c.deferred.get(p.name) orelse {
            std.log.err("idol dino: model.safetensors has no {s}", .{p.name});
            return error.MissingTensor;
        };
        const n = p.shape[0] * p.shape[1];
        if (d.dtype != .f32 or d.bytes != n * 4) {
            std.log.err("idol dino: {s} is {t} {any}; the encoder reads f32 of {d} values", .{ p.name, d.dtype, d.shape[0..d.rank], n });
            return error.UnexpectedTensor;
        }
        var count: usize = 1;
        for (d.shape[0..d.rank]) |s| count *= s;
        if (count != n or (d.rank >= 2 and d.shape[0] != p.shape[0])) return error.UnexpectedTensor;
        const dst = @as([*]u16, @ptrCast(@alignCast(r.base + p.at)))[0..n];
        var done: usize = 0;
        while (done < n) {
            const part = @min(r.stage.len, n - done);
            try r.c.read(p.name, done * 4, std.mem.sliceAsBytes(r.stage[0..part]));
            for (dst[done..][0..part], r.stage[0..part]) |*o, v| o.* = host.bits(v);
            done += part;
        }
    }
};

/// Block `i` placed at `base` in `buf`: its Refs, and its 15 tensors' parts (gate's rows then up's rows make fc1).
fn blockAt(names: *[15][64]u8, i: usize, buf: mtl.Buffer, base: usize, parts: *[15]Part) !Block {
    var at = base;
    var b: Block = undefined;
    const fields = [_]struct { *Ref, usize }{
        .{ &b.qkv, 3 * W * W },   .{ &b.proj, W * W }, .{ &b.proj_b, W }, .{ &b.fc1, 2 * cfg.ffn * W }, .{ &b.fc1_b, 2 * cfg.ffn },
        .{ &b.fc2, W * cfg.ffn }, .{ &b.fc2_b, W },    .{ &b.norm1, W },  .{ &b.norm1_b, W },           .{ &b.norm2, W },
        .{ &b.norm2_b, W },       .{ &b.gamma1, W },   .{ &b.gamma2, W },
    };
    for (fields) |f| {
        f[0].* = .{ .buf = buf, .off = at };
        at += bytes(f[1]);
    }
    std.debug.assert(at - base == per_block);
    const list = [_]struct { []const u8, usize, usize, usize }{
        .{ "attn.qkv.weight", 3 * W, W, b.qkv.off },                      .{ "attn.proj.weight", W, W, b.proj.off },
        .{ "attn.proj.bias", W, 1, b.proj_b.off },                        .{ "mlp.fc1_g.weight", cfg.ffn, W, b.fc1.off },
        .{ "mlp.fc1_x.weight", cfg.ffn, W, b.fc1.off + cfg.ffn * W * 2 }, .{ "mlp.fc1_g.bias", cfg.ffn, 1, b.fc1_b.off },
        .{ "mlp.fc1_x.bias", cfg.ffn, 1, b.fc1_b.off + cfg.ffn * 2 },     .{ "mlp.fc2.weight", W, cfg.ffn, b.fc2.off },
        .{ "mlp.fc2.bias", W, 1, b.fc2_b.off },                           .{ "norm1.weight", W, 1, b.norm1.off },
        .{ "norm1.bias", W, 1, b.norm1_b.off },                           .{ "norm2.weight", W, 1, b.norm2.off },
        .{ "norm2.bias", W, 1, b.norm2_b.off },                           .{ "gamma_1", W, 1, b.gamma1.off },
        .{ "gamma_2", W, 1, b.gamma2.off },
    };
    for (list, parts, names) |e, *p, *n| p.* = .{ .name = try std.fmt.bufPrint(n, "blocks.{d}.{s}", .{ i, e[0] }), .shape = .{ e[1], e[2] }, .at = e[3] };
    return b;
}

/// Blocks t, t + n, t + 2n, ... read on one thread; the first error is kept.
const Job = struct {
    c: *const ck.Checkpoint,
    base: [*]u8,
    parts: *const [cfg.layers][15]Part,
    stage: []f32,
    failed: ?anyerror = null,

    fn run(j: *Job, t: usize, n: usize) void {
        var r: Reader = .{ .c = j.c, .base = j.base, .stage = j.stage };
        var i = t;
        while (i < cfg.layers) : (i += n) for (j.parts[i]) |p| r.put(p) catch |err| {
            j.failed = err;
            return;
        };
    }
};

const threads = 8;

/// The timm snapshot folder `dir`: config.json checked, then model.safetensors's 486 tensors, each read once into bf16.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    try cfg.check(gpa, config);
    var w: Weights = undefined;
    w.ck = ck.Checkpoint.init(gpa);
    errdefer w.ck.deinit();
    const path = try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors", .{dir}, 0);
    try w.ck.addFileSplit(device, path, "", null, always);
    if (w.ck.deferred.count() != cfg.tensors or w.ck.tensors.count() != 0) {
        std.log.err("idol dino: model.safetensors has {d} tensors; DINOv3 ViT-H+/16 has {d}", .{ w.ck.deferred.count() + w.ck.tensors.count(), cfg.tensors });
        return error.UnexpectedTensors;
    }
    const buf = try device.buffer(total_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    w.ck.own(buf, total_bytes) catch |err| {
        buf.deinit();
        return err;
    };
    const base = buf.contents();
    var at: usize = 0;
    const take = struct {
        fn f(b: mtl.Buffer, pos: *usize, n: usize) Ref {
            const r: Ref = .{ .buf = b, .off = pos.* };
            pos.* += bytes(n);
            return r;
        }
    }.f;
    w.patch = take(buf, &at, W * cfg.patch_k);
    w.patch_b = take(buf, &at, W);
    w.prefix = take(buf, &at, cfg.prefix * W);
    w.norm = take(buf, &at, W);
    w.norm_b = take(buf, &at, W);
    w.zeros = take(buf, &at, 3 * W);
    std.debug.assert(at == globals);
    @memset(@as([*]u16, @ptrCast(@alignCast(base + w.zeros.off)))[0 .. 3 * W], 0);
    var r: Reader = .{ .c = &w.ck, .base = base, .stage = try arena.alloc(f32, 1 << 22) };
    const head = [_]Part{
        .{ .name = "patch_embed.proj.weight", .shape = .{ W, cfg.patch_k }, .at = w.patch.off },
        .{ .name = "patch_embed.proj.bias", .shape = .{ W, 1 }, .at = w.patch_b.off },
        .{ .name = "cls_token", .shape = .{ 1, W }, .at = w.prefix.off },
        .{ .name = "reg_token", .shape = .{ 1, (cfg.prefix - 1) * W }, .at = w.prefix.off + W * 2 },
        .{ .name = "norm.weight", .shape = .{ W, 1 }, .at = w.norm.off },
        .{ .name = "norm.bias", .shape = .{ W, 1 }, .at = w.norm_b.off },
    };
    for (head) |p| try r.put(p);
    const names = try arena.create([cfg.layers][15][64]u8);
    const parts = try arena.create([cfg.layers][15]Part);
    for (&w.blocks, 0..) |*b, i| b.* = try blockAt(&names[i], i, buf, globals + i * per_block, &parts[i]);
    var jobs: [threads]Job = undefined;
    var spawned: [threads]?std.Thread = @splat(null);
    for (&jobs, 0..) |*j, t| {
        j.* = .{ .c = &w.ck, .base = base, .parts = parts, .stage = try arena.alloc(f32, 1 << 22) };
        spawned[t] = std.Thread.spawn(.{}, Job.run, .{ j, t, threads }) catch null;
        if (spawned[t] == null) j.run(t, threads);
    }
    for (spawned) |s| if (s) |th| th.join();
    for (jobs) |j| if (j.failed) |err| return err;
    w.ck.closeFiles();
    return w;
}

test "the bf16 layout: every part 256-aligned, the whole under one Metal buffer" {
    try std.testing.expect(total_bytes < 1 << 31);
    try std.testing.expectEqual(@as(usize, 0), per_block % ck.alignment);
    try std.testing.expectEqual(@as(usize, 0), globals % ck.alignment);
}
