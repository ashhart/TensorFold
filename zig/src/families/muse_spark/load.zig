//! A rank's share of a Muse Spark EXL3 pack on the GPU: the tensor set and its host-side constants (pack.zig reads).

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const ml = @import("pack.zig");
const exl3 = @import("exl3.zig");
const Config = config.Config;

pub const Arena = ml.Arena;
const P = config.prefix;

pub const Layer = struct {
    in_norm: u64, // fp32 [D]: 1 + w (centred RMS before attention)
    ffn_norm: u64, // fp32 [D]: 1 + w (centred RMS before the MoE)
    post_ffn: u64, // fp32 [D]: 1 + w (input-scaled RMS after the MoE)
    attn_ab: u64, // fp32 [2, D]: the post-attention residual gate's alpha, beta
    mlp_ab: u64, // fp32 [2, D]
    q: exl3.Linear,
    k: exl3.Linear,
    v: exl3.Linear,
    g: exl3.Linear, // the full-width output gate
    o: exl3.Linear, // a rank's input columns: partial outputs under TP
    router: u64, // [E, D]: bf16 when every fp32 value is a bf16 (exact), else fp32
    router_bf16: bool,
    bias: u64, // fp32 [E]: e_score_correction_bias
    pre: exl3.Linear, // pre_expert_proj D -> latent (a rank's latent outputs under TP)
    pre_norm: u64, // fp32 [latent]: 1 + w
    post: exl3.Linear, // post_expert_proj latent -> D (a rank's latent inputs under TP)
    expert_norm: u64, // fp32 [latent]: 1 + w (input-scaled RMS of each expert's output)
    experts: exl3.Experts,
};

pub const Weights = struct {
    full: Config,
    cfg: Config, // this rank's view
    rank: usize,
    world: usize,
    embed_host: []const u8 = &.{}, // bf16 [V, D]: read in place from the pack's mapped file, rows staged per window
    embed_file: ?core.safetensors.File = null,
    io: ?std.Io = null,
    layers: []const Layer,
    norm: u64, // fp32 [D]: w (the final norm is weighted by w, not 1 + w)
    head: exl3.Linear, // a rank's 128-row blocks of the padded head
    head_lo: usize, // its first vocabulary row
    head_stride: usize, // columns a rank's slice takes in a gather
    head_rows: usize, // the padded head's rows, every rank's together (202112)
    arena: Arena,
    gpa: std.mem.Allocator,

    pub fn deinit(w: *Weights) void {
        if (w.embed_file) |*f| f.close(w.io.?);
        for (w.layers) |L| {
            w.gpa.free(L.experts.host_ptr);
            w.gpa.free(L.experts.host_k2);
        }
        w.gpa.free(w.layers);
        w.arena.deinit();
    }
};

/// Options for partial packs: the first `max_layers` layers only (a k-layer pack, or a check on fewer).
pub const Options = struct { max_layers: ?usize = null, log: bool = false };

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, model_dir: []const u8, rank: usize, world: usize, opt: Options) !Weights {
    const cfg_text = try readFile(gpa, io, model_dir, "config.json");
    defer gpa.free(cfg_text);
    const gen_text = readFile(gpa, io, model_dir, "generation_config.json") catch null;
    defer if (gen_text) |g| gpa.free(g);
    var full = try Config.parse(gpa, cfg_text, gen_text);
    if (opt.max_layers) |m| full.layers = @min(full.layers, m);
    const view = try full.rankView(world);
    var ck = try core.Checkpoint.openModel(gpa, io, model_dir);
    defer ck.close();
    var w: Weights = .{ .full = full, .cfg = view, .rank = rank, .world = world, .layers = &.{}, .norm = 0, .head = undefined, .head_lo = 0, .head_stride = 0, .head_rows = 0, .arena = .{ .d = d, .gpa = gpa }, .gpa = gpa };
    errdefer w.deinit();
    const ar = &w.arena;
    const emb = try ck.get(P ++ "embed_tokens.weight");
    if (emb.dtype != .bf16 or emb.shape[0] != full.vocab or emb.shape[1] != full.hidden) return error.BadEmbedding;
    // the embedding stays out of GPU memory: token rows are copied from the pack's file mapping into each window
    const name = P ++ "embed_tokens.weight";
    for (ck.files.items) |*f| if (f.get(name) != null) {
        w.embed_file = try core.safetensors.File.open(gpa, io, f.path);
        w.io = io;
        break;
    };
    const et = w.embed_file.?.get(name) orelse return error.BadEmbedding;
    w.embed_host = et.bytes;

    var names: std.ArrayList(u8) = .empty;
    defer names.deinit(gpa);
    try names.ensureTotalCapacity(gpa, 1 << 20);
    var cuts: std.ArrayList(config.Cut) = .empty;
    defer cuts.deinit(gpa);
    var nb: [256]u8 = undefined;
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var layers: std.ArrayList(Layer) = .empty;
    errdefer {
        for (layers.items) |L| {
            gpa.free(L.experts.host_ptr);
            gpa.free(L.experts.host_k2);
        }
        layers.deinit(gpa);
    }
    for (0..full.layers) |i| {
        names.clearRetainingCapacity();
        cuts.clearRetainingCapacity();
        try config.layerCuts(full, i, rank, world, &names, gpa, &cuts);
        var L: Layer = undefined;
        L.in_norm = try plusOne(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.input_layernorm.weight", .{i})), 1.0);
        L.ffn_norm = try plusOne(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.pre_feedforward_layernorm.weight", .{i})), 1.0);
        L.post_ffn = try plusOne(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.post_feedforward_layernorm.weight", .{i})), 1.0);
        L.attn_ab = try gates(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.post_attention_residual_gate.gate", .{i})), full.gate_temp);
        L.mlp_ab = try gates(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.post_feedforward_residual_gate.gate", .{i})), full.gate_temp);
        inline for (.{ "q", "k", "v", "g", "o", "pre", "post" }, 0..) |f, j| @field(L, f) = try ml.linear(gpa, ar, try ml.cut(gpa, try ml.readMat(gpa, &ck, cuts.items[j].prefix), cuts.items[j]));
        const ex = cuts.items[config.attn_mats + config.latent_mats ..];
        L.experts = try experts(gpa, ar, &ck, full, ex);
        errdefer {
            gpa.free(L.experts.host_ptr);
            gpa.free(L.experts.host_k2);
        }
        L.pre_norm = try plusOne(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.mlp.pre_expert_norm.weight", .{i})), 1.0);
        L.expert_norm = try plusOne(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.mlp.experts.post_expert_norm.weight", .{i})), 1.0);
        const r = try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.mlp.gate.weight", .{i}));
        if (r.rank != 2 or r.shape[0] != full.experts or r.shape[1] != full.hidden) return error.BadRouter;
        const rf = try ml.asF32(gpa, r);
        defer gpa.free(rf);
        L.router_bf16 = for (rf) |x| {
            if (ml.bf16Round(x) != x) break false;
        } else true;
        if (L.router_bf16) {
            const rb = try gpa.alloc(u16, rf.len);
            defer gpa.free(rb);
            for (rf, rb) |x, *o| o.* = @intCast(@as(u32, @bitCast(x)) >> 16);
            L.router = try ar.upload(std.mem.sliceAsBytes(rb));
        } else L.router = try ar.upload(std.mem.sliceAsBytes(rf));
        L.bias = try plusOne(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, P ++ "layers.{d}.mlp.gate.e_score_correction_bias", .{i})), 0.0);
        try layers.append(gpa, L);
        ck.release(); // the uploaded layer's pages leave the mapping and the page cache
        if (opt.log and (i % 8 == 7 or i == full.layers - 1)) {
            const s = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9;
            std.debug.print("[tensorfold] muse_spark (zig): {d}/{d} layers in {d:.0}s, {d:.1} GiB\n", .{ i + 1, full.layers, s, @as(f64, @floatFromInt(ar.total)) / (1 << 30) });
        }
    }
    gpa.free(w.layers);
    w.layers = try layers.toOwnedSlice(gpa);
    w.norm = try plusOne(gpa, ar, try ck.get(P ++ "norm.weight"), 0.0);
    const hm = try ml.readMat(gpa, &ck, "lm_head");
    if (hm.n() < full.vocab) return error.BadHead;
    const hc = config.headCut(hm.n(), rank, world);
    w.head_lo = hc.lo;
    w.head_stride = config.headStride(hm.n(), world);
    w.head_rows = hm.n();
    w.head = try ml.linear(gpa, ar, try ml.cut(gpa, hm, hc));
    ck.release();
    return w;
}

/// A norm weight as fp32 `bias + w` (exllamav3's constant_bias: 1 for the centred and input-scaled norms, else 0).
fn plusOne(gpa: std.mem.Allocator, ar: *Arena, t: core.safetensors.Tensor, bias: f32) !u64 {
    const f = try ml.asF32(gpa, t);
    defer gpa.free(f);
    for (f) |*x| x.* = bias + x.*;
    return ar.upload(std.mem.sliceAsBytes(f));
}

/// MuseSparkBlock._alpha_beta: w = g / T, beta = sigmoid(w), alpha = sqrt(max(sigmoid(-w) * (1 + beta), 1e-3)).
fn gates(gpa: std.mem.Allocator, ar: *Arena, t: core.safetensors.Tensor, temp: f64) !u64 {
    const g = try ml.asF32(gpa, t);
    defer gpa.free(g);
    const ab = try gpa.alloc(f32, 2 * g.len);
    defer gpa.free(ab);
    const tf: f32 = @floatCast(temp);
    for (g, 0..) |x, j| {
        const v = x / tf;
        const beta = 1.0 / (1.0 + @exp(-v));
        const sn = 1.0 / (1.0 + @exp(v));
        ab[j] = @sqrt(@max(sn * (1.0 + beta), 1e-3));
        ab[g.len + j] = beta;
    }
    return ar.upload(std.mem.sliceAsBytes(ab));
}

/// experts.prepare: per expert (gate, up, down) cut trellises in the stored layout, referenced by pointer tables.
fn experts(gpa: std.mem.Allocator, ar: *Arena, ck: *core.Checkpoint, full: Config, cuts: []const config.Cut) !exl3.Experts {
    const E = full.experts;
    if (cuts.len != 3 * E) return error.ExpertCuts;
    const ptrs = try gpa.alloc(u64, 3 * E); // kept: Experts.host_ptr
    errdefer gpa.free(ptrs);
    const k2s = try gpa.alloc(i32, 3 * E); // kept: Experts.host_k2
    errdefer gpa.free(k2s);
    var st: [6]std.ArrayList(f16) = @splat(.empty); // suh_g, suh_u, svh_g, svh_u, suh_d, svh_d
    defer for (&st) |*l| l.deinit(gpa);
    var lo: [2]u32 = .{ 99, 99 };
    var hi: [2]u32 = .{ 0, 0 };
    var dims: usize = 0;
    var width: usize = 0;
    for (0..E) |e| for (0..3) |j| {
        const m = try ml.cut(gpa, try ml.readMat(gpa, ck, cuts[3 * e + j].prefix), cuts[3 * e + j]);
        defer m.deinit(gpa);
        ptrs[j * E + e] = try ar.upload(std.mem.sliceAsBytes(m.trellis));
        k2s[j * E + e] = @intCast(m.k2);
        const g: usize = if (j == 2) 1 else 0;
        lo[g] = @min(lo[g], m.k2);
        hi[g] = @max(hi[g], m.k2);
        switch (j) {
            0 => {
                dims = m.k();
                width = m.n();
                try st[0].appendSlice(gpa, m.suh);
                try st[2].appendSlice(gpa, m.svh);
            },
            1 => {
                try st[1].appendSlice(gpa, m.suh);
                try st[3].appendSlice(gpa, m.svh);
            },
            else => {
                if (m.n() != dims or m.k() != width) return error.ExpertShapes;
                try st[4].appendSlice(gpa, m.suh);
                try st[5].appendSlice(gpa, m.svh);
            },
        }
    };
    return .{
        .gate_ptr = try ar.upload(std.mem.sliceAsBytes(ptrs[0..E])),
        .up_ptr = try ar.upload(std.mem.sliceAsBytes(ptrs[E .. 2 * E])),
        .down_ptr = try ar.upload(std.mem.sliceAsBytes(ptrs[2 * E ..])),
        .gate_k2 = try ar.upload(std.mem.sliceAsBytes(k2s[0..E])),
        .up_k2 = try ar.upload(std.mem.sliceAsBytes(k2s[E .. 2 * E])),
        .down_k2 = try ar.upload(std.mem.sliceAsBytes(k2s[2 * E ..])),
        .suh_g = try ar.upload(std.mem.sliceAsBytes(st[0].items)),
        .suh_u = try ar.upload(std.mem.sliceAsBytes(st[1].items)),
        .svh_g = try ar.upload(std.mem.sliceAsBytes(st[2].items)),
        .svh_u = try ar.upload(std.mem.sliceAsBytes(st[3].items)),
        .suh_d = try ar.upload(std.mem.sliceAsBytes(st[4].items)),
        .svh_d = try ar.upload(std.mem.sliceAsBytes(st[5].items)),
        .count = E,
        .dims = dims,
        .width = width,
        .k2_gu = .{ lo[0], hi[0] },
        .k2_d = .{ lo[1], hi[1] },
        .host_ptr = ptrs,
        .host_k2 = k2s,
    };
}

fn readFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
}
