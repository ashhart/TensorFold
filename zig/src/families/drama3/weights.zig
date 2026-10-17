//! Drama 3's checkpoint on the device: stacked bf16 projections, FP8 experts repacked for drama3_fp8.cu, a cut lm_head.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shards = @import("shards.zig");

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
    experts: cuda.experts.Layer,
};

pub const TowerLayer = struct {
    table: u64, // [4096, 2048] the code embedding (semantic_embedding / codebook_embedding)
    enorm: u64,
    hnorm: u64,
    eh: u64, // [2048, 4096]
    attn_norm: u64,
    qkv: u64, // [9216, 2048]: q 4096 | gate 4096 | k 512 | v 512
    q_norm: u64,
    k_norm: u64,
    wo: u64, // [2048, 4096]
    ffn_norm: u64,
    w13: u64, // [12288, 2048]
    w2: u64, // [2048, 6144]
    final_norm: u64,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer),
    embed: u64, // [vocab, 2048]
    cb_emb: u64, // [10 * 4096, 2048]
    final_norm: u64,
    cand_head: u64, // [4097, 2048]
    cand_ids: u64, // int32 [4097]
    n_cand: usize,
    layers: [cfg.layers]Layer,
    tower: [cfg.tower_layers]TowerLayer,
    heads: [cfg.codebooks - 1]u64, // [4096, 2048]
    bytes: usize,

    pub fn deinit(w: *Weights) void {
        for (w.bufs.items) |*b| b.free();
        w.bufs.deinit(w.gpa);
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    sh: *const Shards,
    w: *Weights,

    fn alloc(l: *Loader, bytes: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(l.d, bytes);
        try l.w.bufs.append(l.gpa, b);
        l.w.bytes += bytes;
        return b;
    }

    fn expect(t: core.safetensors.Tensor, dtype: core.safetensors.DType, shape: []const usize, name: []const u8) !void {
        if (!t.is(dtype, shape)) {
            std.log.err("drama3: {s} is {s} {any}, expected {s} {any}", .{ name, @tagName(t.dtype), t.shape[0..t.rank], @tagName(dtype), shape });
            return error.BadTensor;
        }
    }

    /// One bf16 tensor on the device as it is.
    fn one(l: *Loader, name: []const u8, shape: []const usize) !u64 {
        const t = try l.sh.get(name);
        try expect(t, .bf16, shape, name);
        const b = try l.alloc(t.bytes.len);
        try b.upload(0, t.bytes);
        l.sh.release(name);
        return b.ptr;
    }

    /// bf16 [rows_i, K] tensors stacked by rows into one [rows_total, K] matrix (pad rows zero).
    fn stack(l: *Loader, parts: []const []const u8, rows: []const usize, k: usize, total_rows: usize) !u64 {
        const b = try l.alloc(total_rows * k * 2);
        try b.fill8(0, null);
        var at: usize = 0;
        for (parts, rows) |name, r| {
            const t = try l.sh.get(name);
            try expect(t, .bf16, &.{ r, k }, name);
            try b.upload(at * k * 2, t.bytes);
            l.sh.release(name);
            at += r;
        }
        return b.ptr;
    }
};

fn tn(buf: []u8, prefix: []const u8, idx: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}{d}.{s}", .{ prefix, idx, suffix }) catch unreachable;
}

fn fmt(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, f, args) catch unreachable;
}

const PackJob = struct {
    sh: *const Shards,
    layer: usize,
    first: usize,
    stride: usize,
    up: []u32, // [E][512*2048*2/4]
    down: []u32, // [E][2048*512/4]
    up_scale: []f32, // [E][2][4][16]
    down_scale: []f32, // [E][16][4]
    err: ?anyerror = null,

    fn run(j: *PackJob) void {
        j.work() catch |e| {
            j.err = e;
        };
    }

    fn work(j: *PackJob) !void {
        const gpa = std.heap.c_allocator;
        const n = cfg.expert_width;
        const k = cfg.hidden;
        var nb: [160]u8 = undefined;
        var e = j.first;
        while (e < cfg.experts) : (e += j.stride) {
            const base = "model.language_model.layers.{d}.mlp.experts.{d}.";
            const g = try j.sh.get(fmt(&nb, base ++ "gate_proj.weight", .{ j.layer, e }));
            const u = try j.sh.get(fmt(&nb, base ++ "up_proj.weight", .{ j.layer, e }));
            const dn = try j.sh.get(fmt(&nb, base ++ "down_proj.weight", .{ j.layer, e }));
            if (!g.is(.f8_e4m3, &.{ n, k }) or !u.is(.f8_e4m3, &.{ n, k }) or !dn.is(.f8_e4m3, &.{ k, n })) return error.BadExpert;
            const per_up = n * k * 2 / 4;
            try cuda.experts.Fp8Experts.packGateUp(gpa, j.up[e * per_up ..][0..per_up], g.bytes, u.bytes, n, k);
            const per_down = k * n / 4;
            cuda.experts.Fp8Experts.packOne(j.down[e * per_down ..][0..per_down], dn.bytes, k, n);
            const gs = try j.sh.get(fmt(&nb, base ++ "gate_proj.weight_scale_inv", .{ j.layer, e }));
            const us = try j.sh.get(fmt(&nb, base ++ "up_proj.weight_scale_inv", .{ j.layer, e }));
            const ds = try j.sh.get(fmt(&nb, base ++ "down_proj.weight_scale_inv", .{ j.layer, e }));
            if (!gs.is(.f32, &.{ n / 128, k / 128 }) or !us.is(.f32, &.{ n / 128, k / 128 }) or !ds.is(.f32, &.{ k / 128, n / 128 })) return error.BadExpertScale;
            const per_us = 2 * (n / 128) * (k / 128);
            const usd = std.mem.sliceAsBytes(j.up_scale[e * per_us ..][0..per_us]);
            @memcpy(usd[0..gs.bytes.len], gs.bytes);
            @memcpy(usd[gs.bytes.len..][0..us.bytes.len], us.bytes);
            const per_ds = (k / 128) * (n / 128);
            @memcpy(std.mem.sliceAsBytes(j.down_scale[e * per_ds ..][0..per_ds]), ds.bytes);
            for ([_][]const u8{ "gate_proj.weight", "up_proj.weight", "down_proj.weight" }) |suffix| {
                var nb2: [160]u8 = undefined;
                const name = std.fmt.bufPrint(&nb2, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ j.layer, e, suffix }) catch unreachable;
                j.sh.release(name);
            }
        }
    }
};

pub const LoadOptions = struct { threads: usize = 16, log: bool = true };

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: cfg.Config, opt: LoadOptions) !*Weights {
    var sh = try Shards.open(gpa, io, dir);
    defer sh.close(io);
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = undefined;
    w.gpa = gpa;
    w.bufs = .empty;
    w.bytes = 0;
    errdefer w.deinit();
    var l: Loader = .{ .gpa = gpa, .d = d, .sh = &sh, .w = w };
    const H = cfg.hidden;
    var nb: [6][192]u8 = undefined;

    w.embed = try l.one("model.language_model.embed_tokens.weight", &.{ cfg.vocab, H });
    w.cb_emb = try l.one("audio_decoder.codebook_embeddings.weight", &.{ cfg.codebooks * cfg.codebook_size, H });
    w.final_norm = try l.one("model.language_model.norm.weight", &.{H});
    // lm_head's candidate rows: eos, then the semantic band (sorted ids, as the sampler's candidate list)
    {
        const t = try sh.get("lm_head.weight");
        try Loader.expect(t, .bf16, &.{ cfg.vocab, H }, "lm_head.weight");
        const ids = try shards.candidateIds(gpa, c);
        defer gpa.free(ids);
        const n = ids.len;
        w.n_cand = n;
        const rows = try gpa.alloc(u8, n * H * 2);
        defer gpa.free(rows);
        for (ids, 0..) |id, r| @memcpy(rows[r * H * 2 ..][0 .. H * 2], t.bytes[@as(usize, @intCast(id)) * H * 2 ..][0 .. H * 2]);
        const hb = try l.alloc(rows.len);
        try hb.upload(0, rows);
        w.cand_head = hb.ptr;
        const ib = try l.alloc(n * 4);
        try ib.upload(0, std.mem.sliceAsBytes(ids));
        w.cand_ids = ib.ptr;
        sh.release("lm_head.weight");
    }

    // routed experts: host staging for one layer, packed by `threads` workers
    const n_up = cfg.experts * cfg.expert_width * H * 2 / 4;
    const n_down = cfg.experts * H * cfg.expert_width / 4;
    const n_us = cfg.experts * 2 * (cfg.expert_width / 128) * (H / 128);
    const n_ds = cfg.experts * (H / 128) * (cfg.expert_width / 128);
    const up_h = try gpa.alloc(u32, n_up);
    defer gpa.free(up_h);
    const down_h = try gpa.alloc(u32, n_down);
    defer gpa.free(down_h);
    const us_h = try gpa.alloc(f32, n_us);
    defer gpa.free(us_h);
    const ds_h = try gpa.alloc(f32, n_ds);
    defer gpa.free(ds_h);
    const nthreads = @max(1, @min(opt.threads, 64));
    const jobs = try gpa.alloc(PackJob, nthreads);
    defer gpa.free(jobs);
    const threads = try gpa.alloc(std.Thread, nthreads);
    defer gpa.free(threads);

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
        // experts
        for (jobs, 0..) |*j, ti| j.* = .{ .sh = &sh, .layer = li, .first = ti, .stride = nthreads, .up = up_h, .down = down_h, .up_scale = us_h, .down_scale = ds_h };
        var spawned: usize = 0;
        defer for (threads[0..spawned]) |t| t.join();
        for (jobs, 0..) |*j, ti| {
            threads[ti] = try std.Thread.spawn(.{}, PackJob.run, .{j});
            spawned += 1;
        }
        for (threads[0..spawned]) |t| t.join();
        spawned = 0;
        for (jobs) |j| if (j.err) |e| return e;
        const ub = try l.alloc(n_up * 4);
        try ub.upload(0, std.mem.sliceAsBytes(up_h));
        const db = try l.alloc(n_down * 4);
        try db.upload(0, std.mem.sliceAsBytes(down_h));
        const usb = try l.alloc(n_us * 4);
        try usb.upload(0, std.mem.sliceAsBytes(us_h));
        const dsb = try l.alloc(n_ds * 4);
        try dsb.upload(0, std.mem.sliceAsBytes(ds_h));
        L.experts = .{ .format = .fp8g, .up = ub.ptr, .down = db.ptr, .up_scale = usb.ptr, .down_scale = dsb.ptr, .width = cfg.expert_width, .dims = H, .experts = cfg.experts };
        if (opt.log and (li % 8 == 7 or li == cfg.layers - 1)) std.log.info("drama3: backbone layer {d} loaded ({d:.1} GB on the device)", .{ li, @as(f64, @floatFromInt(w.bytes)) / 1e9 });
    }

    // the tower: 4 shared layers, then 9 residual ones; 9 codebook heads
    for (0..cfg.tower_layers) |ti| {
        const shared = ti < cfg.shared_layers;
        const idx = if (shared) ti else ti - cfg.shared_layers;
        const p: []const u8 = if (shared) "audio_decoder_mtp_shared.layers." else "audio_decoder_mtp.layers.";
        const T = &w.tower[ti];
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
    if (opt.log) std.log.info("drama3: weights on the device: {d:.2} GB", .{@as(f64, @floatFromInt(w.bytes)) / 1e9});
    return w;
}
