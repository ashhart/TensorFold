//! Unlimited-OCR-2's tensors, bf16 as stored, in shared Metal buffers; every tensor is accounted for.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const index_format = @import("../../core/checkpoint_index.zig");
const cfg = @import("config.zig");
pub const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

/// A SAM block. The relative positions are [len, 64] with len 27 (windowed) or 127 (global).
pub const SamBlock = struct { norm1_w: Ref, norm1_b: Ref, qkv_w: Ref, qkv_b: Ref, proj_w: Ref, proj_b: Ref, rel_h: Ref, rel_w: Ref, rel_len: usize, norm2_w: Ref, norm2_b: Ref, lin1_w: Ref, lin1_b: Ref, lin2_w: Ref, lin2_b: Ref };

/// SAM ViT-B. The convolutions' weights are [out, in x kh x kw] as stored ([out, in, kh, kw]).
pub const Sam = struct {
    patch_w: Ref, // [768, 3 x 16 x 16]
    patch_b: Ref,
    pos: Ref, // [64, 64, 768]
    blocks: [cfg.sam_layers]SamBlock,
    neck0: Ref, // [256, 768] (1 x 1)
    neck1_w: Ref,
    neck1_b: Ref,
    neck2: Ref, // [256, 256 x 3 x 3]
    neck3_w: Ref,
    neck3_b: Ref,
    net2: Ref, // [512, 256 x 3 x 3], stride 2
    net3: Ref, // [1024, 512 x 3 x 3], stride 2
};

pub const ClipBlock = struct { ln1_w: Ref, ln1_b: Ref, qkv_w: Ref, qkv_b: Ref, out_w: Ref, out_b: Ref, ln2_w: Ref, ln2_b: Ref, fc1_w: Ref, fc1_b: Ref, fc2_w: Ref, fc2_b: Ref };

/// CLIP-L's trunk on SAM's map: its own patch embedding is never read.
pub const Clip = struct { cls: Ref, pos: Ref, pre_w: Ref, pre_b: Ref, blocks: [cfg.clip_layers]ClipBlock };

/// The router a layer's attention ranks remote blocks by: its own single-head q and k, Gemma-style norms, head_cls.
pub const Router = struct { q: Ref, k: Ref, q_norm: Ref, k_norm: Ref, head_cls: Ref };

/// A routed expert's three projections, each [rows, cols] as stored.
pub const Expert = struct { gate: Ref, up: Ref, down: Ref };

pub const Layer = struct {
    in_norm: Ref,
    post_norm: Ref,
    q: Ref,
    k: Ref,
    v: Ref,
    o: Ref,
    router: Router,
    /// Layer 0's dense FFN, or a routed layer's shared expert (1,792 wide).
    ffn: Expert,
    /// Routed layers only: the gate [64, 1280] and the 64 experts.
    gate: ?Ref,
    experts: [cfg.experts]Expert,
};

pub const Weights = struct {
    ck: ck.Checkpoint,
    sam: Sam,
    clip: Clip,
    projector_w: Ref, // [1280, 2048]
    projector_b: Ref,
    newline: Ref,
    separator: Ref,
    embed: Ref, // [vocab, 1280]
    layers: [cfg.layers]Layer,
    norm: Ref,
    head: Ref, // [vocab, 1280], not tied

    pub fn deinit(w: *Weights) void {
        w.ck.deinit();
    }
};

/// The release's shard files, from its index's weight map.
fn shards(arena: std.mem.Allocator, io: std.Io, dir: []const u8) ![][:0]const u8 {
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    const index = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index, arena, .limited(16 << 20));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
    var it = (try index_format.weightMap(doc)).iterator();
    while (it.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
    const out = try arena.alloc([:0]const u8, files.count());
    for (files.keys(), out) |name, *path| path.* = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, name }, 0);
    return out;
}

const Taker = struct {
    c: *const ck.Checkpoint,
    read: usize = 0,
    name: [128]u8 = undefined,

    /// The bf16 tensor `name` (formatted), whose dims multiply out to exactly `shape`'s.
    fn take(t: *Taker, comptime fmt: []const u8, args: anytype, shape: []const usize) !Ref {
        const name = try std.fmt.bufPrint(&t.name, fmt, args);
        const x = try t.c.get(name);
        var want: usize = 1;
        for (shape) |s| want *= s;
        var have: usize = 1;
        for (x.shape[0..x.rank]) |s| have *= s;
        if (x.dtype != .bf16 or have != want or x.shape[0] != shape[0]) {
            std.log.err("unlimited ocr: {s} is {t} {any}; this family reads bf16 {any}", .{ name, x.dtype, x.shape[0..x.rank], shape });
            return error.UnexpectedTensor;
        }
        if (x.offset % 16 != 0) return error.UnalignedTensor;
        t.read += 1;
        return .{ .buf = x.buffer, .off = x.offset };
    }

    /// The first dimension of `name` (a relative-position table's length).
    fn rows(t: *Taker, comptime fmt: []const u8, args: anytype) !usize {
        const name = try std.fmt.bufPrint(&t.name, fmt, args);
        return (try t.c.get(name)).shape[0];
    }
};

/// 256-aligned copies of the tensors stored 8 bytes off a 16-byte boundary (the tensor units read 16-byte fragments).
fn realign(c: *ck.Checkpoint, device: mtl.Device) !void {
    var total: usize = 0;
    var it = c.tensors.valueIterator();
    while (it.next()) |t| if (t.offset % 16 != 0) {
        total += std.mem.alignForward(usize, t.bytes, ck.alignment);
    };
    if (total == 0) return;
    const buf = try device.buffer(total, mtl.ResourceOptions.shared);
    c.own(buf, total) catch |err| {
        buf.deinit();
        return err;
    };
    var at: usize = 0;
    it = c.tensors.valueIterator();
    while (it.next()) |t| {
        if (t.offset % 16 == 0) continue;
        @memcpy((buf.contents() + at)[0..t.bytes], (t.buffer.contents() + t.offset)[0..t.bytes]);
        t.buffer = buf;
        t.offset = at;
        at += std.mem.alignForward(usize, t.bytes, ck.alignment);
    }
}

/// Tensors this family never reads: the unused indexer, CLIP's own patch embedding and its position ids.
fn unused(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".self_attn.indexer.") != null or
        std.mem.eql(u8, name, "model.vision_model.embeddings.patch_embedding.weight") or
        std.mem.eql(u8, name, "model.vision_model.embeddings.position_ids");
}

/// The release folder `dir`: config checked, then every tensor this family reads, and none unaccounted for.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config_path = try std.fs.path.join(arena, &.{ dir, "config.json" });
    try cfg.check(gpa, try std.Io.Dir.cwd().readFileAlloc(io, config_path, arena, .limited(1 << 20)));
    var w: Weights = undefined;
    w.ck = ck.Checkpoint.init(gpa);
    errdefer w.ck.deinit();
    for (try shards(arena, io, dir)) |path| try w.ck.addFile(device, path, "");
    try realign(&w.ck, device);
    var t: Taker = .{ .c = &w.ck };
    const H = cfg.hidden;
    const S = cfg.sam_width;
    const C = cfg.clip_width;

    const sp = "model.sam_model.";
    w.sam.patch_w = try t.take(sp ++ "patch_embed.proj.weight", .{}, &.{ S, 3 * cfg.patch * cfg.patch });
    w.sam.patch_b = try t.take(sp ++ "patch_embed.proj.bias", .{}, &.{S});
    w.sam.pos = try t.take(sp ++ "pos_embed", .{}, &.{ 1, cfg.sam_grid * cfg.sam_grid * S });
    for (&w.sam.blocks, 0..) |*b, i| {
        const len = try t.rows(sp ++ "blocks.{d}.attn.rel_pos_h", .{i});
        const want: usize = if (cfg.isGlobal(i)) 2 * cfg.sam_grid - 1 else 2 * cfg.sam_window - 1;
        if (len != want) return error.UnexpectedTensor;
        b.rel_len = len;
        b.norm1_w = try t.take(sp ++ "blocks.{d}.norm1.weight", .{i}, &.{S});
        b.norm1_b = try t.take(sp ++ "blocks.{d}.norm1.bias", .{i}, &.{S});
        b.qkv_w = try t.take(sp ++ "blocks.{d}.attn.qkv.weight", .{i}, &.{ 3 * S, S });
        b.qkv_b = try t.take(sp ++ "blocks.{d}.attn.qkv.bias", .{i}, &.{3 * S});
        b.proj_w = try t.take(sp ++ "blocks.{d}.attn.proj.weight", .{i}, &.{ S, S });
        b.proj_b = try t.take(sp ++ "blocks.{d}.attn.proj.bias", .{i}, &.{S});
        b.rel_h = try t.take(sp ++ "blocks.{d}.attn.rel_pos_h", .{i}, &.{ len, cfg.sam_head_dim });
        b.rel_w = try t.take(sp ++ "blocks.{d}.attn.rel_pos_w", .{i}, &.{ len, cfg.sam_head_dim });
        b.norm2_w = try t.take(sp ++ "blocks.{d}.norm2.weight", .{i}, &.{S});
        b.norm2_b = try t.take(sp ++ "blocks.{d}.norm2.bias", .{i}, &.{S});
        b.lin1_w = try t.take(sp ++ "blocks.{d}.mlp.lin1.weight", .{i}, &.{ cfg.sam_ffn, S });
        b.lin1_b = try t.take(sp ++ "blocks.{d}.mlp.lin1.bias", .{i}, &.{cfg.sam_ffn});
        b.lin2_w = try t.take(sp ++ "blocks.{d}.mlp.lin2.weight", .{i}, &.{ S, cfg.sam_ffn });
        b.lin2_b = try t.take(sp ++ "blocks.{d}.mlp.lin2.bias", .{i}, &.{S});
    }
    const N = cfg.neck_width;
    w.sam.neck0 = try t.take(sp ++ "neck.0.weight", .{}, &.{ N, S });
    w.sam.neck1_w = try t.take(sp ++ "neck.1.weight", .{}, &.{N});
    w.sam.neck1_b = try t.take(sp ++ "neck.1.bias", .{}, &.{N});
    w.sam.neck2 = try t.take(sp ++ "neck.2.weight", .{}, &.{ N, N * 9 });
    w.sam.neck3_w = try t.take(sp ++ "neck.3.weight", .{}, &.{N});
    w.sam.neck3_b = try t.take(sp ++ "neck.3.bias", .{}, &.{N});
    w.sam.net2 = try t.take(sp ++ "net_2.weight", .{}, &.{ cfg.net2_width, N * 9 });
    w.sam.net3 = try t.take(sp ++ "net_3.weight", .{}, &.{ cfg.net3_width, cfg.net2_width * 9 });

    const vp = "model.vision_model.";
    w.clip.cls = try t.take(vp ++ "embeddings.class_embedding", .{}, &.{C});
    w.clip.pos = try t.take(vp ++ "embeddings.position_embedding.weight", .{}, &.{ cfg.clip_positions, C });
    w.clip.pre_w = try t.take(vp ++ "pre_layrnorm.weight", .{}, &.{C});
    w.clip.pre_b = try t.take(vp ++ "pre_layrnorm.bias", .{}, &.{C});
    for (&w.clip.blocks, 0..) |*b, i| {
        const lp = vp ++ "transformer.layers.{d}.";
        b.ln1_w = try t.take(lp ++ "layer_norm1.weight", .{i}, &.{C});
        b.ln1_b = try t.take(lp ++ "layer_norm1.bias", .{i}, &.{C});
        b.qkv_w = try t.take(lp ++ "self_attn.qkv_proj.weight", .{i}, &.{ 3 * C, C });
        b.qkv_b = try t.take(lp ++ "self_attn.qkv_proj.bias", .{i}, &.{3 * C});
        b.out_w = try t.take(lp ++ "self_attn.out_proj.weight", .{i}, &.{ C, C });
        b.out_b = try t.take(lp ++ "self_attn.out_proj.bias", .{i}, &.{C});
        b.ln2_w = try t.take(lp ++ "layer_norm2.weight", .{i}, &.{C});
        b.ln2_b = try t.take(lp ++ "layer_norm2.bias", .{i}, &.{C});
        b.fc1_w = try t.take(lp ++ "mlp.fc1.weight", .{i}, &.{ cfg.clip_ffn, C });
        b.fc1_b = try t.take(lp ++ "mlp.fc1.bias", .{i}, &.{cfg.clip_ffn});
        b.fc2_w = try t.take(lp ++ "mlp.fc2.weight", .{i}, &.{ C, cfg.clip_ffn });
        b.fc2_b = try t.take(lp ++ "mlp.fc2.bias", .{i}, &.{C});
    }
    w.projector_w = try t.take("model.projector.layers.weight", .{}, &.{ H, cfg.fused });
    w.projector_b = try t.take("model.projector.layers.bias", .{}, &.{H});
    w.newline = try t.take("model.image_newline", .{}, &.{H});
    w.separator = try t.take("model.view_seperator", .{}, &.{H});

    w.embed = try t.take("model.embed_tokens.weight", .{}, &.{ cfg.vocab, H });
    w.norm = try t.take("model.norm.weight", .{}, &.{H});
    w.head = try t.take("lm_head.weight", .{}, &.{ cfg.vocab, H });
    const D = cfg.router_dim;
    for (&w.layers, 0..) |*l, i| {
        const lp = "model.layers.{d}.";
        l.in_norm = try t.take(lp ++ "input_layernorm.weight", .{i}, &.{H});
        l.post_norm = try t.take(lp ++ "post_attention_layernorm.weight", .{i}, &.{H});
        l.q = try t.take(lp ++ "self_attn.q_proj.weight", .{i}, &.{ H, H });
        l.k = try t.take(lp ++ "self_attn.k_proj.weight", .{i}, &.{ H, H });
        l.v = try t.take(lp ++ "self_attn.v_proj.weight", .{i}, &.{ H, H });
        l.o = try t.take(lp ++ "self_attn.o_proj.weight", .{i}, &.{ H, H });
        const rp = lp ++ "self_attn.dash_router.";
        l.router = .{
            .q = try t.take(rp ++ "q_proj.weight", .{i}, &.{ D, H }),
            .k = try t.take(rp ++ "k_proj.weight", .{i}, &.{ D, H }),
            .q_norm = try t.take(rp ++ "q_norm.weight", .{i}, &.{D}),
            .k_norm = try t.take(rp ++ "k_norm.weight", .{i}, &.{D}),
            .head_cls = try t.take(rp ++ "head_cls", .{i}, &.{D}),
        };
        if (i == 0) {
            l.ffn = .{
                .gate = try t.take(lp ++ "mlp.gate_proj.weight", .{i}, &.{ cfg.dense_ffn, H }),
                .up = try t.take(lp ++ "mlp.up_proj.weight", .{i}, &.{ cfg.dense_ffn, H }),
                .down = try t.take(lp ++ "mlp.down_proj.weight", .{i}, &.{ H, cfg.dense_ffn }),
            };
            l.gate = null;
            l.experts = undefined;
            continue;
        }
        l.ffn = .{
            .gate = try t.take(lp ++ "mlp.shared_experts.gate_proj.weight", .{i}, &.{ cfg.shared_ffn, H }),
            .up = try t.take(lp ++ "mlp.shared_experts.up_proj.weight", .{i}, &.{ cfg.shared_ffn, H }),
            .down = try t.take(lp ++ "mlp.shared_experts.down_proj.weight", .{i}, &.{ H, cfg.shared_ffn }),
        };
        l.gate = try t.take(lp ++ "mlp.gate.weight", .{i}, &.{ cfg.experts, H });
        for (&l.experts, 0..) |*e, j| e.* = .{
            .gate = try t.take(lp ++ "mlp.experts.{d}.gate_proj.weight", .{ i, j }, &.{ cfg.expert_ffn, H }),
            .up = try t.take(lp ++ "mlp.experts.{d}.up_proj.weight", .{ i, j }, &.{ cfg.expert_ffn, H }),
            .down = try t.take(lp ++ "mlp.experts.{d}.down_proj.weight", .{ i, j }, &.{ H, cfg.expert_ffn }),
        };
    }
    var skipped: usize = 0;
    var it = w.ck.tensors.keyIterator();
    while (it.next()) |name| skipped += @intFromBool(unused(name.*));
    if (t.read + skipped != w.ck.tensors.count()) {
        std.log.err("unlimited ocr: the release has {d} tensors; this family reads {d} and knows {d} unused", .{ w.ck.tensors.count(), t.read, skipped });
        return error.UnexpectedTensors;
    }
    return w;
}
