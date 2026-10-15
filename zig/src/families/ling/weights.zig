//! Ling-3.1-flash's weights on CUDA at TP=1, each tensor in the format its checkpoint stores, layer by layer.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const names = @import("names.zig");
pub const proj = @import("weights_proj.zig");

const Config = config.Config;
const Role = names.Role;
pub const Proj = proj.Proj;
pub const Mlp = proj.Mlp;
pub const Routed = proj.Routed;
pub const Store = proj.Store;

/// KDA: full-rank projections, the q|k|v short convs as #804's chain reads them, the decay and output norm tensors.
pub const Kda = struct {
    q: Proj,
    k: Proj,
    v: Proj,
    b: Proj, // beta, a row per head
    f: Proj, // the decay's full-rank projection
    g: Proj, // the output gate's full-rank projection
    o: Proj,
    conv: u64, // fp32 [3 heads dk][taps]: q_conv1d | k_conv1d | v_conv1d
    a_log: u64, // fp32 [heads]
    dt_bias: u64, // fp32 [heads dk]
    o_norm: u64, // bf16 [dk]
};

/// MLA: q through the q_lora pair (or one q_proj), the latent kv pair, the head-wise gate, the output (`dense`).
pub const Mla = struct {
    q: ?Proj = null, // without q_lora_rank
    q_a: ?Proj = null,
    q_a_norm: u64 = 0, // bf16 [q_lora]
    q_b: ?Proj = null,
    kv_a: Proj, // [kv_lora + rope, hidden]
    kv_a_norm: u64, // bf16 [kv_lora]
    kv_b: Proj, // [heads (nope + v), kv_lora]
    gate: ?Proj = null, // head_wise: [heads, hidden]
    o: Proj,
};

pub const Mixer = union(config.Kind) { kda: Kda, mla: Mla };

/// A MoE block: bf16 router and its pick bias, routed experts, the shared expert as projections, SwiGLU limits.
pub const Moe = struct {
    router: u64, // bf16 [experts][hidden], as GLM's router kernel reads it
    bias: u64, // fp32 [experts], 0 without moe_router_enable_expert_bias
    routed: Routed,
    shared: ?Mlp,
    experts: usize, // loaded experts (all but in a test's partial load)
    shared_limit: f32,
};

pub const Ffn = union(enum) { dense: Mlp, moe: Moe };

pub const Layer = struct { index: u32, attn_norm: u64, mlp_norm: u64, mixer: Mixer, ffn: Ffn };

/// The MTP layer's own tensors beside its MLA + MoE layer: the two input norms, eh_proj, its final norm.
pub const Mtp = struct { enorm: u64, hnorm: u64, eh: Proj, norm: u64 };

pub const Options = struct {
    globals: bool = true, // embedding, final norm and head
    experts: ?usize = null, // the first n routed experts of each MoE layer only (tests on a partial checkpoint)
    stage: usize = 512 << 20, // raw expert bytes staged at a time (pinned and on the device), at least one expert
};

pub const Weights = struct {
    config: Config,
    store: Store,
    embed: u64 = 0, // bf16 [vocab][hidden]
    norm: u64 = 0,
    head: ?Proj = null, // null when tied: the embedding's rows
    layers: []Layer = &.{},
    mtp: ?Mtp = null,

    pub fn deinit(w: *Weights) void {
        w.store.gpa.free(w.layers);
        w.store.deinit();
        w.* = undefined;
    }

    /// Layer `i` among the loaded ones.
    pub fn layer(w: *const Weights, i: u32) ?*const Layer {
        for (w.layers) |*l| if (l.index == i) return l;
        return null;
    }
};

const Loader = struct {
    r: proj.Reader,
    c: *const Config,
    opts: Options,
    buf: [192]u8 = undefined,

    fn module(L: *Loader, i: u32, role: Role) ![]const u8 {
        return names.moduleName(&L.buf, i, role, null);
    }

    fn lin(L: *Loader, i: u32, role: Role, n: usize, k: usize) !Proj {
        return L.r.linear(try L.module(i, role), n, k);
    }

    /// A norm (`.weight`) or bare tensor (A_log, dt_bias, expert_bias) of `numel` values, as `to`.
    fn small(L: *Loader, i: u32, role: Role, numel: usize, to: core.checkpoint.DType) !u64 {
        var b: [200]u8 = undefined;
        const m = try L.module(i, role);
        const bare = role == .kda_a_log or role == .kda_dt_bias or role == .router_bias;
        return L.r.small(if (bare) m else try std.fmt.bufPrint(&b, "{s}.weight", .{m}), numel, to);
    }

    fn kda(L: *Loader, i: u32) !Kda {
        const c = L.c;
        const n: usize = c.kda_heads * c.kda_dim;
        const d: usize = c.hidden;
        var b: [64]u8 = undefined;
        var cb: [96]u8 = undefined;
        return .{
            .q = try L.lin(i, .kda_q, n, d),
            .k = try L.lin(i, .kda_k, n, d),
            .v = try L.lin(i, .kda_v, n, d),
            .b = try L.lin(i, .kda_b, c.kda_heads, d),
            .f = try L.lin(i, .kda_f, n, d),
            .g = try L.lin(i, .kda_g, n, d),
            .o = try L.lin(i, .kda_o, d, n),
            .conv = try L.r.convs(try std.fmt.bufPrint(&b, "model.layers.{d}.attention.", .{i}), n, try std.fmt.bufPrint(&cb, "model.layers.{d}.attention.conv", .{i})),
            .a_log = try L.small(i, .kda_a_log, c.kda_heads, .f32),
            .dt_bias = try L.small(i, .kda_dt_bias, n, .f32),
            .o_norm = try L.small(i, .kda_o_norm, c.kda_dim, .bf16),
        };
    }

    fn mla(L: *Loader, i: u32) !Mla {
        const c = L.c;
        const d: usize = c.hidden;
        const h: usize = c.mla_heads;
        var m: Mla = .{
            .kv_a = try L.lin(i, .mla_kv_a, c.kv_lora + c.rope_dim, d),
            .kv_a_norm = try L.small(i, .mla_kv_a_norm, c.kv_lora, .bf16),
            .kv_b = try L.lin(i, .mla_kv_b, h * (c.nope + c.v_dim), c.kv_lora),
            .o = try L.lin(i, .mla_o, d, h * c.v_dim),
        };
        if (c.q_lora > 0) {
            m.q_a = try L.lin(i, .mla_q_a, c.q_lora, d);
            m.q_a_norm = try L.small(i, .mla_q_a_norm, c.q_lora, .bf16);
            m.q_b = try L.lin(i, .mla_q_b, h * c.qkDim(), c.q_lora);
        } else m.q = try L.lin(i, .mla_q, h * c.qkDim(), d);
        if (c.head_gate) m.gate = try L.lin(i, .mla_gate, h, d);
        return m;
    }

    fn moe(L: *Loader, i: u32) !Moe {
        const c = L.c;
        const d: usize = c.hidden;
        const route = try L.module(i, .router);
        if (c.format(route) != .bf16) {
            std.log.err("{s} is {t}: the routing kernels read a bf16 router", .{ route, c.format(route) });
            return error.QuantizedRouter;
        }
        const count = @min(c.experts, L.opts.experts orelse c.experts);
        const lim = @import("cuda.zig").limits(c, i);
        var b: [96]u8 = undefined;
        return .{
            .router = try L.small(i, .router, c.experts * d, .bf16),
            .bias = if (c.expert_bias) try L.small(i, .router_bias, c.experts, .f32) else 0,
            .routed = try L.r.experts(i, count, c.moe_inter, d, lim.routed),
            .shared = if (c.shared_inter > 0) try L.r.mlp(try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.shared_experts.", .{i}), c.shared_inter, d) else null,
            .experts = count,
            .shared_limit = lim.shared,
        };
    }

    fn layer(L: *Loader, i: u32) !Layer {
        const c = L.c;
        var b: [64]u8 = undefined;
        return .{
            .index = i,
            .attn_norm = try L.small(i, .attn_norm, c.hidden, .bf16),
            .mlp_norm = try L.small(i, .mlp_norm, c.hidden, .bf16),
            .mixer = switch (c.kind(i)) {
                .kda => .{ .kda = try L.kda(i) },
                .mla => .{ .mla = try L.mla(i) },
            },
            .ffn = if (c.isMoe(i)) .{ .moe = try L.moe(i) } else .{ .dense = try L.r.mlp(try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.", .{i}), c.dense_inter, c.hidden) },
        };
    }

    fn mtp(L: *Loader, i: u32) !Mtp {
        const d: usize = L.c.hidden;
        return .{
            .enorm = try L.small(i, .mtp_enorm, d, .bf16),
            .hnorm = try L.small(i, .mtp_hnorm, d, .bf16),
            .eh = try L.lin(i, .mtp_eh, d, 2 * d),
            .norm = try L.small(i, .mtp_final_norm, d, .bf16),
        };
    }
};

/// Layers `layers` (index c.layers is the MTP layer) from `ck`, each tensor in its stored format; `stream` blocking.
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, pack: *const cuda.modelopt.Packer, ck: *core.Checkpoint, c: Config, layers: []const u32, opts: Options) !Weights {
    var w: Weights = .{ .config = c, .store = .{ .gpa = gpa, .d = d } };
    errdefer w.store.deinit();
    var L: Loader = .{ .r = .{ .gpa = gpa, .stream = stream, .pack = pack, .ck = ck, .c = &w.config, .store = &w.store, .stage = opts.stage }, .c = &w.config, .opts = opts };
    const out = try gpa.alloc(Layer, layers.len);
    errdefer gpa.free(out);
    for (layers, out) |i, *l| {
        if (i >= c.layers + c.mtp) return error.BadLayer;
        l.* = try L.layer(i);
        if (i >= c.layers) w.mtp = try L.mtp(i);
        ck.release(); // a layer's pages out of the cache before the next: the GPU shares this memory
    }
    w.layers = out;
    if (!opts.globals) return w;
    const e = try ck.get("model.word_embeddings.weight");
    if (!e.is(.bf16, &.{ c.vocab, c.hidden })) {
        std.log.err("model.word_embeddings: {t} {any}, expected bf16 [{d}, {d}]", .{ e.dtype, e.shape[0..e.rank], c.vocab, c.hidden });
        return error.UnexpectedTensor;
    }
    w.embed = try w.store.keep(e.bytes, "model.word_embeddings.weight");
    w.norm = try L.r.small("model.norm.weight", c.hidden, .bf16);
    if (!c.tie) w.head = try L.r.linear("lm_head", c.vocab, c.hidden);
    return w;
}

/// load() from a checkpoint directory: config.json (and hf_quant_config.json, if any) read there too.
pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, why: *config.Why) !struct { ck: core.Checkpoint, c: Config } {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const json = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 22));
    const hf = cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "hf_quant_config.json" }), a, .limited(1 << 22)) catch null;
    const c = try config.parseWith(gpa, json, hf, why);
    return .{ .ck = try core.Checkpoint.openModel(gpa, io, dir), .c = c };
}
