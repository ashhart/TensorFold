//! Drama 3 Metal weights: BF16 dense projections and packed FP8/NVFP4 experts with original-ID compaction.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const cfg = @import("../config.zig");
const shards = @import("../shards.zig");
const gpu = @import("gpu.zig");
const quant = @import("quant_format.zig");

const Gpu = gpu.Gpu;
const Shards = shards.Shards;

pub const Gdn = struct { proj: u64, conv: u64, a_log: u64, dt_bias: u64, norm: u64, out: u64 };
pub const Attn = struct { qkv: u64, q_norm: u64, k_norm: u64, o: u64 };

pub const Layer = struct {
    full: bool,
    in_norm: u64,
    post_norm: u64,
    gdn: Gdn = undefined,
    attn: Attn = undefined,
    moe_in: u64, // [1288, 2048]
    shared_down: u64, // [2048, 512]
    experts: u64, // Compact slot table: 6 FP8 pointers or 9 NVFP4 pointers per held expert.
    expert_slots: u64,
    expert_count: usize,
    nvfp4: bool,
};

pub const TowerLayer = struct {
    table: u64,
    enorm: u64,
    hnorm: u64,
    eh: u64,
    attn_norm: u64,
    qkv: u64, // [9216, 2048]: q 4096 | gate 4096 | k 512 | v 512
    q_norm: u64,
    k_norm: u64,
    wo: u64,
    ffn_norm: u64,
    w13: u64, // [12288, 2048]
    w2: u64,
    final_norm: u64,
};

pub const Weights = struct {
    embed: u64,
    cb_emb: u64,
    final_norm: u64,
    cand_head: u64, // [4097, 2048]
    cand_ids: u64, // int32 [4097]
    n_cand: usize,
    layers: [cfg.layers]Layer,
    tower: [cfg.tower_layers]TowerLayer,
    heads: [cfg.codebooks - 1]u64,
    bytes: usize, // copied bf16 bytes
    mapped: usize, // checkpoint bytes read in place
    sh: Shards,

    pub fn deinit(w: *Weights, io: std.Io) void {
        w.sh.close(io);
    }
};

const Loader = struct {
    g: *Gpu,
    sh: *const Shards,
    w: *Weights,

    fn expect(t: core.safetensors.Tensor, dtype: core.safetensors.DType, shape: []const usize, name: []const u8) !void {
        if (!t.is(dtype, shape)) {
            std.log.err("drama3: {s} is {s} {any}, expected {s} {any}", .{ name, @tagName(t.dtype), t.shape[0..t.rank], @tagName(dtype), shape });
            return error.BadTensor;
        }
    }

    fn one(l: *Loader, name: []const u8, shape: []const usize) !u64 {
        const t = try l.sh.get(name);
        try expect(t, .bf16, shape, name);
        const a = try l.g.alloc(t.bytes.len);
        @memcpy(l.g.host(a, t.bytes.len), t.bytes);
        l.w.bytes += t.bytes.len;
        return a;
    }

    /// bf16 [rows_i, K] tensors stacked by rows into one [total_rows, K] matrix (pad rows zero).
    fn stack(l: *Loader, parts: []const []const u8, rows: []const usize, k: usize, total_rows: usize) !u64 {
        const a = try l.g.alloc(total_rows * k * 2);
        var at: usize = 0;
        for (parts, rows) |name, r| {
            const t = try l.sh.get(name);
            try expect(t, .bf16, &.{ r, k }, name);
            @memcpy(l.g.host(a + at * k * 2, t.bytes.len), t.bytes);
            at += r;
        }
        l.w.bytes += total_rows * k * 2;
        return a;
    }
};

fn fmt(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, f, args) catch unreachable;
}

/// Each shard's mapping as a no-copy buffer, by shard index (the experts' pointer tables point into them).
fn mapShards(g: *Gpu, sh: *const Shards) ![]u64 {
    const out = try g.gpa.alloc(u64, sh.files.len);
    errdefer g.gpa.free(out);
    const page = std.heap.pageSize();
    for (sh.files, 0..) |*f, i| {
        const mem = f.map.memory;
        if (@intFromPtr(mem.ptr) % page != 0) return error.ShardNotMapped;
        const len = std.mem.alignForward(usize, mem.len, page);
        const b = try g.device.bufferNoCopy(@ptrCast(@constCast(mem.ptr)), len, mtl.ResourceOptions.shared);
        out[i] = g.register(b, true) catch |err| {
            b.deinit();
            return err;
        };
        // registered: the Gpu owns `b` from here, so a failure below must not release it again
        try g.indirect.append(g.gpa, b);
    }
    return out;
}

/// Compact slots preserve ascending original IDs; each layer has one expert format, selected from tensor headers.
fn expertTable(l: *Loader, li: usize, maps: []const u64, held: *const quant.Held) !void {
    const layer = &l.w.layers[li];
    const ids = held.ids[li][0..held.counts[li]];
    layer.expert_count = ids.len;
    layer.expert_slots = 0;
    if (ids.len != cfg.experts) {
        const slots = held.slots(li);
        layer.expert_slots = try l.g.alloc(@sizeOf(@TypeOf(slots)));
        @memcpy(l.g.host(layer.expert_slots, @sizeOf(@TypeOf(slots))), std.mem.asBytes(&slots));
        l.w.bytes += @sizeOf(@TypeOf(slots));
    }
    const n = cfg.expert_width;
    const k = cfg.hidden;
    const parts = [_]struct { []const u8, [2]usize }{ .{ "gate_proj", .{ n, k } }, .{ "up_proj", .{ n, k } }, .{ "down_proj", .{ k, n } } };
    var nb: [3][192]u8 = undefined;
    const first = try l.sh.get(fmt(&nb[0], "model.language_model.layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ li, ids[0] }));
    layer.nvfp4 = first.dtype == .u8;
    if (!layer.nvfp4 and first.dtype != .f8_e4m3) return error.UnsupportedExpertDtype;
    const stride: usize = if (layer.nvfp4) 9 else 6;
    layer.experts = try l.g.alloc(ids.len * stride * 8);
    const ptrs = std.mem.bytesAsSlice(u64, l.g.host(layer.experts, ids.len * stride * 8));
    for (ids, 0..) |e, slot| {
        for (parts, 0..) |p, j| {
            const wname = fmt(&nb[0], "model.language_model.layers.{d}.mlp.experts.{d}.{s}.weight", .{ li, e, p[0] });
            const sname = fmt(&nb[1], "{s}{s}", .{ wname, if (layer.nvfp4) "_scale" else "_scale_inv" });
            const wt = try l.sh.get(wname);
            const sc = try l.sh.get(sname);
            const wl = try l.sh.locate(wname);
            const sl = try l.sh.locate(sname);
            const base = slot * stride + j * (stride / 3);
            if (layer.nvfp4) {
                const gname = fmt(&nb[2], "{s}_scale_2", .{wname});
                const gl = try l.sh.get(gname);
                try quant.nvfp4(wt, sc, gl, p[1][0], p[1][1]);
                const loc = try l.sh.locate(gname);
                if (loc.offset % 4 != 0) return error.MisalignedScale;
                ptrs[base + 2] = l.g.gpuAddress(maps[loc.file] + loc.offset);
                l.w.mapped += gl.bytes.len;
            } else {
                try Loader.expect(wt, .f8_e4m3, &p[1], wname);
                try Loader.expect(sc, .f32, &.{ p[1][0] / 128, p[1][1] / 128 }, sname);
                if (wl.offset % 8 != 0) return error.MisalignedExpert;
                if (sl.offset % 4 != 0) return error.MisalignedScale;
            }
            ptrs[base] = l.g.gpuAddress(maps[wl.file] + wl.offset);
            ptrs[base + 1] = l.g.gpuAddress(maps[sl.file] + sl.offset);
            l.w.mapped += wt.bytes.len + sc.bytes.len;
        }
    }
    l.w.bytes += ids.len * stride * 8;
}

pub fn load(g: *Gpu, io: std.Io, dir: []const u8, c: cfg.Config) !*Weights {
    const w = try g.gpa.create(Weights);
    errdefer g.gpa.destroy(w);
    w.* = undefined;
    w.bytes = 0;
    w.mapped = 0;
    w.sh = try Shards.open(g.gpa, io, dir);
    errdefer w.sh.close(io);
    const held = try quant.heldMetadata(w.sh.metadata);
    try quant.checkNames(&held, w.sh.map.keyIterator());
    const first = g.bufs.items.len;
    const indirect_first = g.indirect.items.len;
    errdefer {
        g.indirect.shrinkRetainingCapacity(indirect_first);
        g.releaseFrom(first);
    }
    const maps = try mapShards(g, &w.sh);
    defer g.gpa.free(maps);
    var l: Loader = .{ .g = g, .sh = &w.sh, .w = w };
    const H = cfg.hidden;
    var nb: [6][192]u8 = undefined;

    w.embed = try l.one("model.language_model.embed_tokens.weight", &.{ cfg.vocab, H });
    w.cb_emb = try l.one("audio_decoder.codebook_embeddings.weight", &.{ cfg.codebooks * cfg.codebook_size, H });
    w.final_norm = try l.one("model.language_model.norm.weight", &.{H});
    {
        const t = try w.sh.get("lm_head.weight");
        try Loader.expect(t, .bf16, &.{ cfg.vocab, H }, "lm_head.weight");
        const ids = try shards.candidateIds(g.gpa, c);
        defer g.gpa.free(ids);
        w.n_cand = ids.len;
        w.cand_head = try g.alloc(ids.len * H * 2);
        const rows = g.host(w.cand_head, ids.len * H * 2);
        for (ids, 0..) |id, r| @memcpy(rows[r * H * 2 ..][0 .. H * 2], t.bytes[@as(usize, @intCast(id)) * H * 2 ..][0 .. H * 2]);
        w.cand_ids = try g.alloc(ids.len * 4);
        @memcpy(g.host(w.cand_ids, ids.len * 4), std.mem.sliceAsBytes(ids));
        w.bytes += ids.len * (H * 2 + 4);
    }
    const pre = "model.language_model.layers.{d}.";
    for (0..cfg.layers) |li| {
        const L = &w.layers[li];
        L.full = c.full_attention[li];
        L.in_norm = try l.one(fmt(&nb[0], pre ++ "input_layernorm.weight", .{li}), &.{H});
        L.post_norm = try l.one(fmt(&nb[0], pre ++ "post_attention_layernorm.weight", .{li}), &.{H});
        if (L.full) {
            L.attn = .{
                .qkv = try l.stack(&.{ fmt(&nb[0], pre ++ "self_attn.q_proj.weight", .{li}), fmt(&nb[1], pre ++ "self_attn.k_proj.weight", .{li}), fmt(&nb[2], pre ++ "self_attn.v_proj.weight", .{li}) }, &.{ 2 * cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim }, H, cfg.attn_proj),
                .q_norm = try l.one(fmt(&nb[0], pre ++ "self_attn.q_norm.weight", .{li}), &.{cfg.head_dim}),
                .k_norm = try l.one(fmt(&nb[0], pre ++ "self_attn.k_norm.weight", .{li}), &.{cfg.head_dim}),
                .o = try l.one(fmt(&nb[0], pre ++ "self_attn.o_proj.weight", .{li}), &.{ H, cfg.heads * cfg.head_dim }),
            };
        } else {
            L.gdn = .{
                .proj = try l.stack(&.{ fmt(&nb[0], pre ++ "linear_attn.in_proj_qkv.weight", .{li}), fmt(&nb[1], pre ++ "linear_attn.in_proj_z.weight", .{li}), fmt(&nb[2], pre ++ "linear_attn.in_proj_b.weight", .{li}), fmt(&nb[3], pre ++ "linear_attn.in_proj_a.weight", .{li}) }, &.{ cfg.conv_channels, 4096, 32, 32 }, H, cfg.gdn_proj),
                .conv = try l.one(fmt(&nb[0], pre ++ "linear_attn.conv1d.weight", .{li}), &.{ cfg.conv_channels, 1, cfg.conv_kernel }),
                .a_log = try l.one(fmt(&nb[0], pre ++ "linear_attn.A_log", .{li}), &.{cfg.lin_v_heads}),
                .dt_bias = try l.one(fmt(&nb[0], pre ++ "linear_attn.dt_bias", .{li}), &.{cfg.lin_v_heads}),
                .norm = try l.one(fmt(&nb[0], pre ++ "linear_attn.norm.weight", .{li}), &.{cfg.lin_dim}),
                .out = try l.one(fmt(&nb[0], pre ++ "linear_attn.out_proj.weight", .{li}), &.{ H, cfg.lin_v_heads * cfg.lin_dim }),
            };
        }
        L.moe_in = try l.stack(&.{ fmt(&nb[0], pre ++ "mlp.shared_expert.gate_proj.weight", .{li}), fmt(&nb[1], pre ++ "mlp.shared_expert.up_proj.weight", .{li}), fmt(&nb[2], pre ++ "mlp.gate.weight", .{li}), fmt(&nb[3], pre ++ "mlp.shared_expert_gate.weight", .{li}) }, &.{ cfg.shared_width, cfg.shared_width, cfg.experts, 1 }, H, cfg.moe_proj);
        L.shared_down = try l.one(fmt(&nb[0], pre ++ "mlp.shared_expert.down_proj.weight", .{li}), &.{ H, cfg.shared_width });
        try expertTable(&l, li, maps, &held);
    }
    for (0..cfg.tower_layers) |ti| {
        const shared = ti < cfg.shared_layers;
        const idx = if (shared) ti else ti - cfg.shared_layers;
        const p: []const u8 = if (shared) "audio_decoder_mtp_shared.layers." else "audio_decoder_mtp.layers.";
        const T = &w.tower[ti];
        const tn = struct {
            fn f(buf: []u8, pfx: []const u8, i: usize, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "{s}{d}.{s}", .{ pfx, i, suffix }) catch unreachable;
            }
        }.f;
        T.table = try l.one(tn(&nb[0], p, idx, if (shared) "semantic_embedding.weight" else "codebook_embedding.weight"), &.{ cfg.codebook_size, H });
        T.enorm = try l.one(tn(&nb[0], p, idx, "enorm.weight"), &.{H});
        T.hnorm = try l.one(tn(&nb[0], p, idx, "hnorm.weight"), &.{H});
        T.eh = try l.one(tn(&nb[0], p, idx, "eh_proj.weight"), &.{ H, 2 * H });
        T.attn_norm = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention_norm.weight"), &.{H});
        T.qkv = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention.wqkv.weight"), &.{ cfg.attn_proj, H });
        T.q_norm = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention.q_norm.weight"), &.{cfg.head_dim});
        T.k_norm = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention.k_norm.weight"), &.{cfg.head_dim});
        T.wo = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention.wo.weight"), &.{ H, cfg.heads * cfg.head_dim });
        T.ffn_norm = try l.one(tn(&nb[0], p, idx, "transformer_layer.ffn_norm.weight"), &.{H});
        T.w13 = try l.stack(&.{ tn(&nb[0], p, idx, "transformer_layer.feed_forward.w1.weight"), tn(&nb[1], p, idx, "transformer_layer.feed_forward.w3.weight") }, &.{ cfg.tower_ffn, cfg.tower_ffn }, H, 2 * cfg.tower_ffn);
        T.w2 = try l.one(tn(&nb[0], p, idx, "transformer_layer.feed_forward.w2.weight"), &.{ H, cfg.tower_ffn });
        T.final_norm = try l.one(tn(&nb[0], p, idx, "final_layernorm.weight"), &.{H});
    }
    for (0..cfg.codebooks - 1) |hi| w.heads[hi] = try l.one(fmt(&nb[0], "audio_decoder_mtp.heads.{d}.weight", .{hi}), &.{ cfg.codebook_size, H });
    std.log.info("drama3: {d:.2} GB of BF16/copied weights, {d:.2} GB of packed FP8/NVFP4 experts read in place", .{ @as(f64, @floatFromInt(w.bytes)) / 1e9, @as(f64, @floatFromInt(w.mapped)) / 1e9 });
    return w;
}
