//! A Nemotron load retains destinations, staging and the engine's kernel parents through failed completion.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const weights = @import("cuda_weights.zig");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const Source = @import("cuda_source.zig").Source;
const latent = @import("cuda_latent.zig");
const modelopt = @import("cuda_modelopt.zig");
const join = weights.join;
const Block = weights.Block;
pub const Pending = struct {
    gpa: std.mem.Allocator,
    ops: kern.Ops,
    w: weights.Weights,
    src: ?Source = null,
    loader: ?weights.Loader = null,
    pub fn finish(p: *Pending) !void {
        try p.ops.k.d.synchronize();
        if (p.src) |*src| try src.flush();
        try p.ops.s.synchronize();
        try p.ops.k.d.synchronize();
    }
    fn temporary(p: *Pending) void {
        if (p.src) |*src| src.deinit();
        if (p.loader) |*L| {
            L.scratch.free();
            L.host.deinit(p.gpa);
            for (L.held.items) |h| p.gpa.free(h);
            L.held.deinit(p.gpa);
        }
    }
    pub fn close(p: *Pending) !void {
        try p.finish();
        p.temporary();
        p.w.deinit();
        p.gpa.destroy(p);
    }
    pub fn fail(p: *Pending, held: *?*Pending, err: anyerror) anyerror!weights.Weights {
        p.close() catch {
            held.* = p;
            return error.LoadRetained;
        };
        return err;
    }
};
pub fn load(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, dir: []const u8, c: Config, with_mtp: bool, held: *?*Pending) !weights.Weights {
    if (c.format == .mlx) try c.fourOnly("CUDA");
    const p = try gpa.create(Pending);
    p.* = .{ .gpa = gpa, .ops = ops, .w = .{ .gpa = gpa, .config = c } };
    p.src = Source.init(gpa, ops) catch |err| {
        gpa.destroy(p);
        return err;
    };
    const scratch = cuda.DeviceBuffer.alloc(ops.k.d, 1 << 20) catch |err| {
        p.src.?.deinit();
        gpa.destroy(p);
        return err;
    };
    p.loader = .{ .gpa = gpa, .ops = ops, .w = &p.w, .src = &p.src.?, .scratch = scratch };
    fill(p, io, dir, c, with_mtp) catch |err| return p.fail(held, err);
    p.finish() catch |err| return p.fail(held, err);
    const out = p.w;
    p.temporary();
    gpa.destroy(p);
    return out;
}
fn fill(p: *Pending, io: std.Io, dir: []const u8, c: Config, with_mtp: bool) !void {
    const gpa = p.gpa;
    const ops = p.ops;
    const w = &p.w;
    const src = &p.src.?;
    const L = &p.loader.?;
    if (c.latent > 0 and !c.bolt) try latent.checkHeld(c);
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    try src.add(&ck);
    const pre = c.prefix();
    w.blocks = try gpa.alloc(Block, c.layers);
    for (c.kinds[0..c.layers], 0..) |kind, i| {
        var nm: [128]u8 = undefined;
        var nb: [128]u8 = undefined;
        var pb: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&nm, "blocks.{d}", .{i});
        const norm = try ck.expect(try std.fmt.bufPrint(&pb, "{s}backbone.layers.{d}.norm.weight", .{ pre, i }), .bf16, &.{c.hidden});
        var b: Block = .{ .kind = kind, .norm = try L.raw(try join(&nb, name, ".norm"), norm) };
        const mixer = try std.fmt.bufPrint(&pb, "{s}backbone.layers.{d}.mixer.", .{ pre, i });
        switch (kind) {
            .mamba => b.mamba = try L.mamba(&ck, try join(&nb, name, ".mamba"), mixer),
            .attention => b.attn = try L.attention(&ck, try join(&nb, name, ".attn"), mixer),
            .moe => b.moe = if (c.bolt) try modelopt.latent(L, &ck, try join(&nb, name, ".moe"), mixer, c, i) else if (c.latent > 0) try latent.load(L, &ck, try join(&nb, name, ".moe"), mixer, c, @intCast(i)) else try L.moe(&ck, try join(&nb, name, ".moe"), mixer, c),
        }
        w.blocks[i] = b;
    }
    if (c.format == .modelopt) w.embed = try modelopt.embed(L, &ck, c) else {
        const emb = try ck.get("backbone.embeddings.weight");
        w.embed = .{
            .w = try L.raw("embed.weight", emb),
            .s = try L.raw("embed.scales", try ck.get("backbone.embeddings.scales")),
            .b = try L.raw("embed.biases", try ck.get("backbone.embeddings.biases")),
            .n = emb.dim(0),
            .k = emb.dim(1) * 8,
        };
    }
    var tail: [96]u8 = undefined;
    w.norm_f = try L.raw("norm_f", try ck.expect(try std.fmt.bufPrint(&tail, "{s}backbone.norm_f.weight", .{pre}), .bf16, &.{c.hidden}));
    w.head = try L.dense(&ck, "head", &.{try std.fmt.bufPrint(&tail, "{s}lm_head", .{pre})});
    // a ModelOpt checkpoint keeps its MTP head in its own shards
    if (c.format == .modelopt) if (!with_mtp) try ck.skip(try std.fmt.bufPrint(&tail, "{s}mtp.", .{pre})) else if (c.latent > 0) try latent.mtp(L, &ck, c) else try modelopt.mtp(L, &ck, c);
    // the picture tower and its projector load with the image path (an omni checkpoint's other parts)
    if (c.omni) for ([_][]const u8{ "vision_model.", "mlp1.", "vision_projector." }) |part| try ck.skip(part);
    if (ck.unused() != 0) return error.UnusedCheckpointTensors;
    if (L.learned > 0) std.log.info("{d} output projections carry what lessons taught this model", .{L.learned});
    if (with_mtp and c.format == .mlx) try weights.loadMtp(gpa, io, L, &ck, dir, c);
    try src.flush();
    try ops.s.synchronize();
}
