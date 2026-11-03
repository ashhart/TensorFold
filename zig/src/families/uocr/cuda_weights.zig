//! Unlimited-OCR-2's tensors on CUDA, bf16 as the release stores them, each 256-aligned in one device buffer; every
//! tensor is accounted for (weights.zig's names and shapes). The few the host reads (position tables for a tile, the
//! relative positions, the newline and separator rows) are kept as host copies; the mapped files are closed after.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");

/// A tensor on the device, and its values on the host when `Taker.keep` asked for them.
pub const T = struct {
    dev: u64 = 0,
    host: []const u16 = &.{},
};

pub const SamBlock = struct { norm1_w: T, norm1_b: T, qkv_w: T, qkv_b: T, proj_w: T, proj_b: T, rel_h: T, rel_w: T, rel_len: usize, norm2_w: T, norm2_b: T, lin1_w: T, lin1_b: T, lin2_w: T, lin2_b: T };

pub const Sam = struct { patch_w: T, patch_b: T, pos: T, blocks: [cfg.sam_layers]SamBlock, neck0: T, neck1_w: T, neck1_b: T, neck2: T, neck3_w: T, neck3_b: T, net2: T, net3: T };

pub const ClipBlock = struct { ln1_w: T, ln1_b: T, qkv_w: T, qkv_b: T, out_w: T, out_b: T, ln2_w: T, ln2_b: T, fc1_w: T, fc1_b: T, fc2_w: T, fc2_b: T };

pub const Clip = struct { cls: T, pos: T, pre_w: T, pre_b: T, blocks: [cfg.clip_layers]ClipBlock };

pub const Router = struct { q: T, k: T, q_norm: T, k_norm: T, head_cls: T };

pub const Expert = struct { gate: T, up: T, down: T };

pub const Layer = struct { in_norm: T, post_norm: T, q: T, k: T, v: T, o: T, router: Router, ffn: Expert, gate: ?T, experts: [cfg.experts]Expert };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    buf: cuda.DeviceBuffer,
    /// The host copies' storage.
    arena: std.heap.ArenaAllocator,
    sam: Sam,
    clip: Clip,
    projector_w: T,
    projector_b: T,
    newline: T,
    separator: T,
    embed: T,
    layers: [cfg.layers]Layer,
    norm: T,
    head: T,
    bytes: usize,

    pub fn deinit(w: *Weights) void {
        w.buf.free();
        w.arena.deinit();
        w.gpa.destroy(w);
    }
};

const Taker = struct {
    ck: *core.Checkpoint,
    /// Sizing pass: null. Loading pass: the device buffer the tensors go into, in the sizing pass's order.
    buf: ?*const cuda.DeviceBuffer,
    arena: std.mem.Allocator,
    at: usize = 0,
    name: [160]u8 = undefined,

    fn take(t: *Taker, comptime fmt: []const u8, args: anytype, shape: []const usize) !T {
        return t.takeKept(fmt, args, shape, false);
    }

    /// The bf16 tensor `name` (formatted) whose dims multiply out to exactly `shape`'s, its first dim `shape[0]`.
    fn takeKept(t: *Taker, comptime fmt: []const u8, args: anytype, shape: []const usize, keep: bool) !T {
        const name = try std.fmt.bufPrint(&t.name, fmt, args);
        const x = try t.ck.get(name);
        var want: usize = 1;
        for (shape) |s| want *= s;
        if (x.dtype != .bf16 or x.numel() != want or x.shape[0] != shape[0]) {
            std.log.err("unlimited ocr: {s} is {t} {any}; this family reads bf16 {any}", .{ name, x.dtype, x.shape[0..x.rank], shape });
            return error.UnexpectedTensor;
        }
        const off = t.at;
        t.at += std.mem.alignForward(usize, x.bytes.len, 256);
        var out: T = .{};
        if (t.buf) |b| {
            try b.upload(off, x.bytes);
            out.dev = b.ptr + off;
            if (keep) {
                const h = try t.arena.alloc(u16, want);
                @memcpy(std.mem.sliceAsBytes(h), x.bytes);
                out.host = h;
            }
        }
        return out;
    }

    fn rows(t: *Taker, comptime fmt: []const u8, args: anytype) !usize {
        const name = try std.fmt.bufPrint(&t.name, fmt, args);
        return (try t.ck.get(name)).shape[0];
    }
};

/// The release folder `dir`: config checked, then every tensor this family reads on the device, none unaccounted for.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Weights {
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const config_path = try std.fs.path.join(scratch.allocator(), &.{ dir, "config.json" });
    try cfg.check(gpa, try std.Io.Dir.cwd().readFileAlloc(io, config_path, scratch.allocator(), .limited(1 << 20)));
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.gpa = gpa;
    w.arena = std.heap.ArenaAllocator.init(gpa);
    errdefer w.arena.deinit();
    var sizing: Taker = .{ .ck = &ck, .buf = null, .arena = w.arena.allocator() };
    try walk(&sizing, w);
    w.bytes = sizing.at;
    w.buf = try cuda.DeviceBuffer.alloc(d, w.bytes);
    errdefer w.buf.free();
    var loading: Taker = .{ .ck = &ck, .buf = &w.buf, .arena = w.arena.allocator() };
    try walk(&loading, w);
    for (0..cfg.layers) |i| {
        var name: [64]u8 = undefined;
        try ck.skip(try std.fmt.bufPrint(&name, "model.layers.{d}.self_attn.indexer.", .{i}));
    }
    try ck.skip("model.vision_model.embeddings.patch_embedding.weight");
    try ck.skip("model.vision_model.embeddings.position_ids");
    if (ck.unused() != 0) {
        std.log.err("unlimited ocr: the release has {d} tensors this family does not read", .{ck.unused()});
        return error.UnexpectedTensors;
    }
    return w;
}

fn walk(t: *Taker, w: *Weights) !void {
    const H = cfg.hidden;
    const S = cfg.sam_width;
    const C = cfg.clip_width;
    const sp = "model.sam_model.";
    w.sam.patch_w = try t.take(sp ++ "patch_embed.proj.weight", .{}, &.{ S, 3 * cfg.patch * cfg.patch });
    w.sam.patch_b = try t.take(sp ++ "patch_embed.proj.bias", .{}, &.{S});
    w.sam.pos = try t.takeKept(sp ++ "pos_embed", .{}, &.{ 1, cfg.sam_grid * cfg.sam_grid * S }, true);
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
        b.rel_h = try t.takeKept(sp ++ "blocks.{d}.attn.rel_pos_h", .{i}, &.{ len, cfg.sam_head_dim }, cfg.isGlobal(i));
        b.rel_w = try t.takeKept(sp ++ "blocks.{d}.attn.rel_pos_w", .{i}, &.{ len, cfg.sam_head_dim }, cfg.isGlobal(i));
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
    w.clip.pos = try t.takeKept(vp ++ "embeddings.position_embedding.weight", .{}, &.{ cfg.clip_positions, C }, true);
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
    w.newline = try t.takeKept("model.image_newline", .{}, &.{H}, true);
    w.separator = try t.takeKept("model.view_seperator", .{}, &.{H}, true);

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
}
