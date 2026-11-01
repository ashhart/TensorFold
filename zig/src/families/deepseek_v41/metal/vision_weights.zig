//! The release's vision tensors in one arena: ViT blocks, aligner, span delimiters and image routing biases.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const weights = @import("weights.zig");
const vision = @import("vision.zig");

const Ref = vision.Ref;
const Pack = host.pack.Pack;
const align_raw: u32 = 8;

/// One ViT block: fp32 norm weights, bf16 projections [n][k] and biases.
pub const Block = struct { norm1: Ref, wqkv: Ref, bqkv: Ref, wo: Ref, bo: Ref, norm2: Ref, w1: Ref, w2: Ref };

pub const Weights = struct {
    a: std.mem.Allocator,
    cfg: vision.Config,
    arena: mtl.Buffer,
    patch_w: Ref,
    patch_b: Ref,
    blocks: []Block,
    norm: Ref,
    al_w1: Ref,
    al_b1: Ref,
    al_w2: Ref,
    al_b2: Ref,
    /// image_start, image_end, image_newline: bf16 [out_dim] each, in that order, one array [3][out_dim]
    delims: Ref,
    /// each text layer's `ffn.gate.bias_vl` [experts] fp32 (the routing bias of image rows)
    bias_vl: []Ref,

    pub fn deinit(w: *Weights) void {
        w.arena.deinit();
        w.delims.buf.deinit();
        w.a.free(w.blocks);
        w.a.free(w.bias_vl);
    }

    /// Whether the pack carries the tower.
    pub fn present(pack: *const Pack) bool {
        return pack.has("vision.patch_embed.proj.weight");
    }

    /// The tower's tensors read from `pack` into one arena; `layers` and `experts` are the text model's.
    pub fn load(a: std.mem.Allocator, io: std.Io, device: mtl.Device, pack: *const Pack, cfg: vision.Config, layers: u32, experts: u32) !Weights {
        try cfg.check();
        var l: weights.Loader = .{ .a = a, .pack = pack };
        defer l.items.deinit(a);
        var buf: [96]u8 = undefined;
        const D = cfg.dim;
        const S = struct {
            fn tensor(ld: *weights.Loader, b: []u8, comptime fmt: []const u8, args: anytype, dtype: host.pack.DType, shape: []const usize, into: weights.Into) !weights.Place {
                const name = try std.fmt.bufPrint(b, fmt, args);
                const info = try ld.pack.need(name);
                if (!info.is(dtype, shape)) return error.BadVisionTensor;
                return ld.add(info, into, align_raw);
            }
        };
        const patch_w = try S.tensor(&l, &buf, "vision.patch_embed.proj.weight", .{}, .bf16, &.{ D, cfg.patchValues() }, .raw);
        const patch_b = try S.tensor(&l, &buf, "vision.patch_embed.proj.bias", .{}, .bf16, &.{D}, .raw);
        const places = try a.alloc([8]weights.Place, cfg.layers);
        defer a.free(places);
        for (places, 0..) |*p, i| {
            const k: u32 = @intCast(i);
            p.* = .{
                try S.tensor(&l, &buf, "vision.blocks.{d}.norm1.weight", .{k}, .bf16, &.{D}, .f32),
                try S.tensor(&l, &buf, "vision.blocks.{d}.attn.wqkv.weight", .{k}, .bf16, &.{ 3 * D, D }, .raw),
                try S.tensor(&l, &buf, "vision.blocks.{d}.attn.wqkv.bias", .{k}, .bf16, &.{3 * D}, .raw),
                try S.tensor(&l, &buf, "vision.blocks.{d}.attn.wo.weight", .{k}, .bf16, &.{ D, D }, .raw),
                try S.tensor(&l, &buf, "vision.blocks.{d}.attn.wo.bias", .{k}, .bf16, &.{D}, .raw),
                try S.tensor(&l, &buf, "vision.blocks.{d}.norm2.weight", .{k}, .bf16, &.{D}, .f32),
                try S.tensor(&l, &buf, "vision.blocks.{d}.mlp.w1.weight", .{k}, .bf16, &.{ 2 * cfg.inter, D }, .raw),
                try S.tensor(&l, &buf, "vision.blocks.{d}.mlp.w2.weight", .{k}, .bf16, &.{ D, cfg.inter }, .raw),
            };
        }
        const norm = try S.tensor(&l, &buf, "vision.norm.weight", .{}, .bf16, &.{D}, .f32);
        const A = cfg.out_dim;
        const al_w1 = try S.tensor(&l, &buf, "aligner.w1.weight", .{}, .bf16, &.{ A, cfg.alignIn() }, .raw);
        const al_b1 = try S.tensor(&l, &buf, "aligner.w1.bias", .{}, .bf16, &.{A}, .raw);
        const al_w2 = try S.tensor(&l, &buf, "aligner.w2.weight", .{}, .bf16, &.{ A, A }, .raw);
        const al_b2 = try S.tensor(&l, &buf, "aligner.w2.bias", .{}, .bf16, &.{A}, .raw);
        // the delimiters are three tensors, so they land in one run of the arena: aligned, consecutive
        var dl: [3]weights.Place = undefined;
        for ([_][]const u8{ "image_start", "image_end", "image_newline" }, 0..) |n, i| {
            const info = try pack.need(n);
            if (!info.is(.bf16, &.{A})) return error.BadVisionTensor;
            dl[i] = try l.add(info, .raw, align_raw);
        }
        const bias = try a.alloc(weights.Place, layers);
        defer a.free(bias);
        for (bias, 0..) |*p, i| p.* = try S.tensor(&l, &buf, "layers.{d}.ffn.gate.bias_vl", .{@as(u32, @intCast(i))}, .f32, &.{experts}, .f32);
        const arena = try device.buffer(@max(l.size, 256), mtl.ResourceOptions.shared);
        errdefer arena.deinit();
        try l.fill(io, arena.contents()[0..l.size]);
        var w: Weights = .{ .a = a, .cfg = cfg, .arena = arena, .patch_w = patch_w.ref(arena), .patch_b = patch_b.ref(arena), .blocks = &.{}, .norm = norm.ref(arena), .al_w1 = al_w1.ref(arena), .al_b1 = al_b1.ref(arena), .al_w2 = al_w2.ref(arena), .al_b2 = al_b2.ref(arena), .delims = undefined, .bias_vl = &.{} };
        // the delimiters are copied side by side (the merge kernel reads them as one [3][out_dim] array)
        const dbuf = try device.buffer(3 * A * 2, mtl.ResourceOptions.shared);
        errdefer dbuf.deinit();
        for (dl, 0..) |p, i| @memcpy(dbuf.contents()[i * A * 2 ..][0 .. A * 2], arena.contents()[p.off..][0 .. A * 2]);
        w.delims = .{ .buf = dbuf };
        w.blocks = try a.alloc(Block, cfg.layers);
        errdefer a.free(w.blocks);
        for (w.blocks, places) |*b, p| b.* = .{ .norm1 = p[0].ref(arena), .wqkv = p[1].ref(arena), .bqkv = p[2].ref(arena), .wo = p[3].ref(arena), .bo = p[4].ref(arena), .norm2 = p[5].ref(arena), .w1 = p[6].ref(arena), .w2 = p[7].ref(arena) };
        w.bias_vl = try a.alloc(Ref, layers);
        for (w.bias_vl, bias) |*r, p| r.* = p.ref(arena);
        return w;
    }
};
