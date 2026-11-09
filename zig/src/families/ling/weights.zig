//! Ling-3.1-flash's weights on CUDA (TP=1 or a TP=2 rank's share): EXL3 packs, or bf16 for checks, layer by layer.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const names = @import("names.zig");
const routing = @import("route.zig");
pub const proj = @import("weights_proj.zig");
pub const cuts = @import("weights_cut.zig");

const Config = config.Config;
const Role = names.Role;
pub const Proj = proj.Proj;
pub const Mlp = proj.Mlp;
pub const Routed = proj.Routed;
pub const Store = proj.Store;
pub const drains = proj.drains;
pub const uploaded = proj.uploaded;

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

pub const Options = struct {
    globals: bool = true, // embedding, final norm and head
    experts: ?usize = null, // the first n routed experts of each MoE layer only (tests on a partial checkpoint)
    rank: u32 = 0, // a TP rank of `world`: heads, widths and the head's vocabulary cut (weights_cut)
    world: u32 = 1,
};

pub const Weights = struct {
    config: Config, // this rank's share (weights_cut.rankConfig): every width the forward runs at
    rank: u32 = 0,
    world: u32 = 1,
    store: Store,
    embed: u64 = 0, // bf16 [vocab][hidden]
    norm: u64 = 0,
    head: ?Proj = null, // null when tied: the embedding's rows
    layers: []Layer = &.{},

    pub fn deinit(w: *Weights) void {
        w.drop(true);
    }

    /// deinit, but the device buffers leaked unless `settled` (nothing queued can still read them).
    pub fn drop(w: *Weights, settled: bool) void {
        w.store.gpa.free(w.layers);
        w.store.drop(settled);
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
    c: *const Config, // the checkpoint's whole shapes
    opts: Options,
    buf: [192]u8 = undefined,

    /// This rank's even share of `total`, or null at TP=1 (the uncut path).
    fn share(L: *const Loader, total: usize) ?cuts.Span {
        return if (L.opts.world == 1) null else cuts.even(total, L.opts.rank, L.opts.world);
    }

    /// Linear `role` of layer i, output rows cut to this rank's share (`out`) or its input columns (else).
    fn cutLin(L: *Loader, i: u32, role: Role, out: bool) !Proj {
        const nk = names.dims(L.c, role).linear;
        const s = L.share(if (out) nk[0] else nk[1]) orelse return L.lin(i, role);
        return cuts.linear(&L.r, try L.module(i, role), nk[0], nk[1], if (out) .{ .out = &.{s} } else .{ .in = s });
    }

    fn module(L: *Loader, i: u32, role: Role) ![]const u8 {
        return names.moduleName(&L.buf, i, role, null);
    }

    /// Linear `role` of layer i at names.dims' [n, k].
    fn lin(L: *Loader, i: u32, role: Role) !Proj {
        const nk = names.dims(L.c, role).linear;
        return L.r.linear(try L.module(i, role), nk[0], nk[1]);
    }

    /// A norm (`.weight`) or bare tensor (A_log, dt_bias, expert_bias) of names.numel values, as `to`.
    fn small(L: *Loader, i: u32, role: Role, to: core.checkpoint.DType) !u64 {
        var b: [200]u8 = undefined;
        const m = try L.module(i, role);
        const bare = role == .kda_a_log or role == .kda_dt_bias or role == .router_bias;
        return L.r.small(if (bare) m else try std.fmt.bufPrint(&b, "{s}.weight", .{m}), names.numel(L.c, role), to);
    }

    /// The MLP of gate, up and down `role`s under `prefix`, at names.dims' widths.
    fn mlp(L: *Loader, prefix: []const u8, gate: Role) !Mlp {
        const nk = names.dims(L.c, gate).linear;
        const s = L.share(nk[0]) orelse return L.r.mlp(prefix, nk[0], nk[1]);
        var m: [3]Proj = undefined;
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, j| {
            var b: [192]u8 = undefined;
            const name = try std.fmt.bufPrint(&b, "{s}{s}", .{ prefix, p });
            m[j] = if (j < 2) try cuts.linear(&L.r, name, nk[0], nk[1], .{ .out = &.{s} }) else try cuts.linear(&L.r, name, nk[1], nk[0], .{ .in = s });
        }
        return .{ .gate = m[0], .up = m[1], .down = m[2] };
    }

    fn kda(L: *Loader, i: u32) !Kda {
        const n = names.dims(L.c, .kda_q_conv).conv[0];
        var b: [64]u8 = undefined;
        var cb: [96]u8 = undefined;
        if (L.c.quant == .exl3) return L.kdaFused(i, n);
        return .{
            .q = try L.lin(i, .kda_q),
            .k = try L.lin(i, .kda_k),
            .v = try L.lin(i, .kda_v),
            .b = try L.lin(i, .kda_b),
            .f = try L.lin(i, .kda_f),
            .g = try L.lin(i, .kda_g),
            .o = try L.lin(i, .kda_o),
            .conv = try L.r.convs(try std.fmt.bufPrint(&b, "model.layers.{d}.attention.", .{i}), n, try std.fmt.bufPrint(&cb, "model.layers.{d}.attention.conv", .{i})),
            .a_log = try L.small(i, .kda_a_log, .f32),
            .dt_bias = try L.small(i, .kda_dt_bias, .f32),
            .o_norm = try L.small(i, .kda_o_norm, .bf16),
        };
    }

    /// exllamav3's KDA as it converts it: q|k|v fused (qkv_proj, viewed per head block by columns), conv1d fused q|k|v.
    fn kdaFused(L: *Loader, i: u32, n: usize) !Kda {
        const d: usize = L.c.hidden;
        var b: [96]u8 = undefined;
        const h = n / L.opts.world; // this rank's heads x 128 of each of q, k, v
        const lo = L.opts.rank * h;
        const sp = [3]cuts.Span{ .{ .lo = lo, .hi = lo + h }, .{ .lo = n + lo, .hi = n + lo + h }, .{ .lo = 2 * n + lo, .hi = 2 * n + lo + h } };
        const qkv_name = try std.fmt.bufPrint(&b, "model.layers.{d}.attention.qkv_proj", .{i});
        const qkv = if (L.opts.world == 1) try L.r.linear(qkv_name, 3 * n, d) else try cuts.linear(&L.r, qkv_name, 3 * n, d, .{ .out = &sp });
        const w = switch (qkv) {
            .exl3 => |x| x,
            else => return error.UnexpectedTensor, // a bf16 pack keeps q, k and v apart (the other path)
        };
        const conv = try std.fmt.bufPrint(&b, "model.layers.{d}.attention.conv1d.weight", .{i});
        const t = try L.r.ck.get(conv);
        if (t.numel() != 3 * n * L.c.conv or (t.dtype != .bf16 and t.dtype != .f32 and t.dtype != .f16)) {
            std.log.err("{s}: {t} {any}, expected {d} x {d} values", .{ conv, t.dtype, t.shape[0..t.rank], 3 * n, L.c.conv });
            return error.UnexpectedTensor;
        }
        if (L.opts.world > 1) return .{
            .q = .{ .exl3 = try w.columns(0, @intCast(h)) },
            .k = .{ .exl3 = try w.columns(@intCast(h), @intCast(h)) },
            .v = .{ .exl3 = try w.columns(@intCast(2 * h), @intCast(h)) },
            .b = try L.cutLin(i, .kda_b, true),
            .f = try L.cutLin(i, .kda_f, true),
            .g = try L.cutLin(i, .kda_g, true),
            .o = try L.cutLin(i, .kda_o, false),
            .conv = try cuts.convs(&L.r, conv, 3 * n, L.c.conv, &sp),
            .a_log = try cuts.small(&L.r, try L.module(i, .kda_a_log), names.numel(L.c, .kda_a_log), .f32, L.share(L.c.kda_heads).?),
            .dt_bias = try cuts.small(&L.r, try L.module(i, .kda_dt_bias), n, .f32, L.share(n).?),
            .o_norm = try L.small(i, .kda_o_norm, .bf16),
        };
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(L.r.gpa);
        try proj.append(L.r.gpa, &host, t, .f32);
        return .{
            .q = .{ .exl3 = try w.columns(0, @intCast(n)) },
            .k = .{ .exl3 = try w.columns(@intCast(n), @intCast(n)) },
            .v = .{ .exl3 = try w.columns(@intCast(2 * n), @intCast(n)) },
            .b = try L.lin(i, .kda_b),
            .f = try L.lin(i, .kda_f),
            .g = try L.lin(i, .kda_g),
            .o = try L.lin(i, .kda_o),
            .conv = try L.r.store.keep(host.items, conv),
            .a_log = try L.small(i, .kda_a_log, .f32),
            .dt_bias = try L.small(i, .kda_dt_bias, .f32),
            .o_norm = try L.small(i, .kda_o_norm, .bf16),
        };
    }

    fn mla(L: *Loader, i: u32) !Mla {
        const c = L.c;
        var m: Mla = .{
            .kv_a = try L.lin(i, .mla_kv_a),
            .kv_a_norm = try L.small(i, .mla_kv_a_norm, .bf16),
            .kv_b = try L.cutLin(i, .mla_kv_b, true),
            .o = try L.cutLin(i, .mla_o, false),
        };
        if (c.q_lora > 0) {
            m.q_a = try L.lin(i, .mla_q_a);
            m.q_a_norm = try L.small(i, .mla_q_a_norm, .bf16);
            m.q_b = try L.cutLin(i, .mla_q_b, true);
        } else m.q = try L.cutLin(i, .mla_q, true);
        if (c.head_gate) m.gate = try L.cutLin(i, .mla_gate, true);
        return m;
    }

    fn moe(L: *Loader, i: u32) !Moe {
        const c = L.c;
        const route = try L.module(i, .router);
        var rb: [128]u8 = undefined;
        const plain_router = if (c.quant == .exl3) !L.r.ck.has(try std.fmt.bufPrint(&rb, "{s}.trellis", .{route})) else c.format(route) == .bf16;
        if (!plain_router) {
            std.log.err("{s} is {t}: the routing kernels read a bf16 router", .{ route, c.format(route) });
            return error.QuantizedRouter;
        }
        const count = @min(c.experts, L.opts.experts orelse c.experts);
        const lim = @import("cuda.zig").limits(c, i);
        var b: [96]u8 = undefined;
        const x = names.dims(c, .expert_gate).linear;
        return .{
            .router = try L.small(i, .router, .bf16),
            .bias = if (c.expert_bias) try routerBias(&L.r, i) else 0,
            .routed = try L.r.experts(i, count, x[0], x[1], lim.routed, L.share(x[0])),
            .shared = if (c.shared_inter > 0) try L.mlp(try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.shared_experts.", .{i}), .shared_gate) else null,
            .experts = count,
            .shared_limit = lim.shared,
        };
    }

    fn layer(L: *Loader, i: u32) !Layer {
        const c = L.c;
        var b: [64]u8 = undefined;
        return .{
            .index = i,
            .attn_norm = try L.small(i, .attn_norm, .bf16),
            .mlp_norm = try L.small(i, .mlp_norm, .bf16),
            .mixer = switch (c.kind(i)) {
                .kda => .{ .kda = try L.kda(i) },
                .mla => .{ .mla = try L.mla(i) },
            },
            .ffn = if (c.isMoe(i)) .{ .moe = try L.moe(i) } else .{ .dense = try L.mlp(try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.", .{i}), .dense_gate) },
        };
    }
};

/// Layer i's expert bias as fp32, refused by name when a routing group holds +inf and -inf (it scores NaN).
pub fn routerBias(r: *proj.Reader, i: u32) !u64 {
    var b: [96]u8 = undefined;
    const m = try names.moduleName(&b, i, .router_bias, null);
    const t = try r.ck.get(m);
    var v: [routing.max_experts]f32 = undefined;
    if (t.numel() == r.c.experts) { // else small refuses it by name
        for (v[0..r.c.experts], 0..) |*x, e| x.* = proj.value(t, e) catch return r.small(m, r.c.experts, .f32);
        routing.checkBias(v[0..r.c.experts], routing.Shape.of(r.c)) catch |err| {
            if (!@import("builtin").is_test) std.log.err("{s}: a routing group holds both a +inf and a -inf bias, so its score is NaN", .{m});
            return err;
        };
    }
    return r.small(m, r.c.experts, .f32);
}

/// Backbone layers `layers` from `ck`, each tensor in its stored format; `stream` blocking. FP8 and NVFP4 are refused.
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, pack: *const cuda.modelopt.Packer, ck: *core.Checkpoint, c: Config, layers: []const u32, opts: Options) !Weights {
    if (c.quant == .fp8_block128 or c.quant == .nvfp4) {
        std.log.err("a {t} checkpoint: Ling on CUDA loads EXL3 packs (and bf16 for checks); FP8 and NVFP4 are not loaded yet", .{c.quant});
        return error.LingFormat;
    }
    var why: config.Why = .{};
    const rc = cuts.rankConfig(c, opts.world, &why) catch |err| {
        std.log.err("{s}", .{why.text()});
        return err;
    };
    if (opts.world == 0 or opts.rank >= opts.world) return error.BadRank;
    var w: Weights = .{ .config = rc, .rank = opts.rank, .world = opts.world, .store = .{ .gpa = gpa, .d = d } };
    const full = c; // the loader reads the file's shapes; the forward runs the rank's
    var L: Loader = .{ .r = .{ .gpa = gpa, .stream = stream, .pack = pack, .ck = ck, .c = &full, .store = &w.store }, .c = &full, .opts = opts };
    errdefer w.store.drop(L.r.settled()); // the packers' outputs freed only once nothing queued writes them
    const out = try gpa.alloc(Layer, layers.len);
    errdefer gpa.free(out);
    for (layers, out) |i, *l| {
        if (i >= c.layers) return error.BadLayer;
        l.* = try L.layer(i);
        ck.release(); // a layer's pages out of the cache before the next: the GPU shares this memory
    }
    w.layers = out;
    if (!opts.globals) return w;
    const e = try ck.get("model.word_embeddings.weight");
    const vd = names.dims(&w.config, .embed).linear;
    if (!e.is(.bf16, &vd)) {
        std.log.err("model.word_embeddings: {t} {any}, expected bf16 {any}", .{ e.dtype, e.shape[0..e.rank], vd });
        return error.UnexpectedTensor;
    }
    w.embed = try w.store.keep(e.bytes, "model.word_embeddings.weight");
    w.norm = try L.r.small("model.norm.weight", names.numel(&w.config, .final_norm), .bf16);
    const hd = names.dims(&w.config, .lm_head).linear;
    if (!c.tie) w.head = if (L.share(hd[0])) |s| try cuts.linear(&L.r, "lm_head", hd[0], hd[1], .{ .out = &.{s} }) else try L.r.linear("lm_head", hd[0], hd[1]);
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
