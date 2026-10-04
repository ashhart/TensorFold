//! Kimi K3's weights by layer, resolved by checkpoint name with the dtype and shape the config implies.
const std = @import("std");
const mtl = @import("metal");
const config = @import("config.zig");
const store = @import("store.zig");
const kernels = @import("kernels.zig");
const Tensor = store.Tensor;
const Config = config.Config;

pub const Kda = struct { q: Tensor, k: Tensor, v: Tensor, g: Tensor, o: Tensor, fa: Tensor, fb: Tensor, b: Tensor, conv_q: Tensor, conv_k: Tensor, conv_v: Tensor, A_log: Tensor, dt_bias: Tensor, o_norm: Tensor };
pub const Mla = struct { qa: Tensor, qa_norm: Tensor, qb: Tensor, kva: Tensor, kva_norm: Tensor, kvb: Tensor, g: Tensor, o: Tensor };
pub const Dense = struct { gate: Tensor, up: Tensor, down: Tensor };
pub const Moe = struct { router: Tensor, bias: Tensor, down: Tensor, norm: Tensor, up: Tensor, sh_gate: Tensor, sh_up: Tensor, sh_down: Tensor, table: mtl.Buffer };

pub const Layer = struct {
    in_norm: Tensor,
    post_norm: Tensor,
    sa_res_norm: Tensor,
    sa_res_proj: Tensor,
    mlp_res_norm: Tensor,
    mlp_res_proj: Tensor,
    attn: union(config.Kind) { kda: Kda, mla: Mla },
    mlp: union(enum) { dense: Dense, moe: Moe },
};

pub const Head = struct { embed: Tensor, lm_head: Tensor, norm: Tensor, res_norm: Tensor, res_proj: Tensor };

/// One tensor the checkpoint must hold, as the config predicts it.
pub const Spec = struct { name: []const u8, dtype: store.DType, shape: [3]u32, rank: u8 };

fn spec(buf: *std.ArrayList(Spec), a: std.mem.Allocator, name: []const u8, dtype: store.DType, shape: []const u32) !void {
    var s = Spec{ .name = name, .dtype = dtype, .shape = .{ 1, 1, 1 }, .rank = @intCast(shape.len) };
    @memcpy(s.shape[0..shape.len], shape);
    try buf.append(a, s);
}

/// Every tensor of layer i (routed experts excluded: `expertSpecs`), named as the checkpoint names them.
pub fn layerSpecs(c: *const Config, i: u32, a: std.mem.Allocator, out: *std.ArrayList(Spec)) !void {
    const H = c.hidden;
    const p = try std.fmt.allocPrint(a, "layers.{d}.", .{i});
    const n = struct {
        fn f(al: std.mem.Allocator, pre: []const u8, rest: []const u8) ![]const u8 {
            return std.mem.concat(al, u8, &.{ pre, rest });
        }
    }.f;
    for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "self_attention_res_norm.weight", "mlp_res_norm.weight" }) |w|
        try spec(out, a, try n(a, p, w), .bf16, &.{H});
    for ([_][]const u8{ "self_attention_res_proj.weight", "mlp_res_proj.weight" }) |w| try spec(out, a, try n(a, p, w), .bf16, &.{ 1, H });
    const W = c.kdaWidth();
    if (c.kind(i) == .kda) {
        for ([_][]const u8{ "q", "k", "v", "g" }) |w| try spec(out, a, try std.fmt.allocPrint(a, "{s}self_attn.{s}_proj.weight", .{ p, w }), .bf16, &.{ W, H });
        try spec(out, a, try n(a, p, "self_attn.o_proj.weight"), .bf16, &.{ H, W });
        try spec(out, a, try n(a, p, "self_attn.f_a_proj.weight"), .bf16, &.{ c.kda_dim, H });
        try spec(out, a, try n(a, p, "self_attn.f_b_proj.weight"), .bf16, &.{ W, c.kda_dim });
        try spec(out, a, try n(a, p, "self_attn.b_proj.weight"), .bf16, &.{ c.kda_heads, H });
        for ([_][]const u8{ "q", "k", "v" }) |w| try spec(out, a, try std.fmt.allocPrint(a, "{s}self_attn.{s}_conv1d.weight", .{ p, w }), .f32, &.{ W, 1, c.conv });
        try spec(out, a, try n(a, p, "self_attn.A_log"), .f32, &.{});
        try spec(out, a, try n(a, p, "self_attn.dt_bias"), .f32, &.{W});
        try spec(out, a, try n(a, p, "self_attn.o_norm.weight"), .f32, &.{c.kda_dim});
    } else {
        const heads = c.mla_heads;
        try spec(out, a, try n(a, p, "self_attn.q_a_proj.weight"), .bf16, &.{ c.q_lora, H });
        try spec(out, a, try n(a, p, "self_attn.q_a_layernorm.weight"), .bf16, &.{c.q_lora});
        try spec(out, a, try n(a, p, "self_attn.q_b_proj.weight"), .bf16, &.{ heads * c.qHead(), c.q_lora });
        try spec(out, a, try n(a, p, "self_attn.kv_a_proj_with_mqa.weight"), .bf16, &.{ c.kv_lora + c.rope, H });
        try spec(out, a, try n(a, p, "self_attn.kv_a_layernorm.weight"), .bf16, &.{c.kv_lora});
        try spec(out, a, try n(a, p, "self_attn.kv_b_proj.weight"), .bf16, &.{ heads * (c.nope + c.v_dim), c.kv_lora });
        try spec(out, a, try n(a, p, "self_attn.g_proj.weight"), .bf16, &.{ heads * c.v_dim, H });
        try spec(out, a, try n(a, p, "self_attn.o_proj.weight"), .bf16, &.{ H, heads * c.v_dim });
    }
    if (!c.isMoe(i)) {
        for ([_][]const u8{ "gate", "up" }) |w| try spec(out, a, try std.fmt.allocPrint(a, "{s}mlp.{s}_proj.weight", .{ p, w }), .bf16, &.{ c.dense_inter, H });
        try spec(out, a, try n(a, p, "mlp.down_proj.weight"), .bf16, &.{ H, c.dense_inter });
        return;
    }
    const m = try n(a, p, "block_sparse_moe.");
    try spec(out, a, try n(a, m, "gate.weight"), .bf16, &.{ c.experts, H });
    try spec(out, a, try n(a, m, "gate.e_score_correction_bias"), .f32, &.{c.experts});
    try spec(out, a, try n(a, m, "routed_expert_down_proj.weight"), .bf16, &.{ c.latent, H });
    try spec(out, a, try n(a, m, "routed_expert_norm.weight"), .bf16, &.{c.latent});
    try spec(out, a, try n(a, m, "routed_expert_up_proj.weight"), .bf16, &.{ H, c.latent });
    for ([_][]const u8{ "gate", "up" }) |w| try spec(out, a, try std.fmt.allocPrint(a, "{s}shared_experts.{s}_proj.weight", .{ m, w }), .bf16, &.{ c.shared_inter, H });
    try spec(out, a, try n(a, m, "shared_experts.down_proj.weight"), .bf16, &.{ H, c.shared_inter });
}

/// The six MXFP4 tensors of routed expert e of layer i: w1 and w3 [inter, latent], w2 [latent, inter].
pub fn expertSpecs(c: *const Config, i: u32, e: u32, a: std.mem.Allocator, out: *std.ArrayList(Spec)) !void {
    for ([_][]const u8{ "w1", "w3", "w2" }) |w| {
        const rows: u32, const cols: u32 = if (w[1] == '2') .{ c.latent, c.moe_inter } else .{ c.moe_inter, c.latent };
        const p = try std.fmt.allocPrint(a, "layers.{d}.block_sparse_moe.experts.{d}.{s}.", .{ i, e, w });
        try spec(out, a, try std.mem.concat(a, u8, &.{ p, "weight_packed" }), .u8, &.{ rows, cols / 2 });
        try spec(out, a, try std.mem.concat(a, u8, &.{ p, "weight_scale" }), .u8, &.{ rows, cols / 32 });
    }
}

pub fn headSpecs(c: *const Config, a: std.mem.Allocator, out: *std.ArrayList(Spec)) !void {
    try spec(out, a, "embed_tokens.weight", .bf16, &.{ c.vocab, c.hidden });
    try spec(out, a, "lm_head.weight", .bf16, &.{ c.vocab, c.hidden });
    try spec(out, a, "norm.weight", .bf16, &.{c.hidden});
    try spec(out, a, "output_attn_res_norm.weight", .bf16, &.{c.hidden});
    try spec(out, a, "output_attn_res_proj.weight", .bf16, &.{ 1, c.hidden });
}

fn fetch(src: store.Source, specs: []const Spec, name_suffix: []const u8) !Tensor {
    for (specs) |s| if (std.mem.endsWith(u8, s.name, name_suffix)) return src.get(s.name, s.dtype, s.shape[0..s.rank]);
    return error.UnknownWeight;
}

/// Layer i's tensors; a MoE layer also gets its expert pointer table (experts [first, last) by address).
pub fn layer(c: *const Config, i: u32, src: store.Source, device: mtl.Device, gpa: std.mem.Allocator, first: u32, last: u32) !Layer {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var specs: std.ArrayList(Spec) = .empty;
    try layerSpecs(c, i, a, &specs);
    const sp = specs.items;
    var l: Layer = undefined;
    const fields = .{ .{ "in_norm", ".input_layernorm.weight" }, .{ "post_norm", "post_attention_layernorm.weight" }, .{ "sa_res_norm", "self_attention_res_norm.weight" }, .{ "sa_res_proj", "self_attention_res_proj.weight" }, .{ "mlp_res_norm", "mlp_res_norm.weight" }, .{ "mlp_res_proj", "mlp_res_proj.weight" } };
    inline for (fields) |f| @field(l, f[0]) = try fetch(src, sp, f[1]);
    if (c.kind(i) == .kda) {
        var k: Kda = undefined;
        const kf = .{ .{ "q", "self_attn.q_proj.weight" }, .{ "k", "self_attn.k_proj.weight" }, .{ "v", "self_attn.v_proj.weight" }, .{ "g", "self_attn.g_proj.weight" }, .{ "o", "self_attn.o_proj.weight" }, .{ "fa", "self_attn.f_a_proj.weight" }, .{ "fb", "self_attn.f_b_proj.weight" }, .{ "b", "self_attn.b_proj.weight" }, .{ "conv_q", "self_attn.q_conv1d.weight" }, .{ "conv_k", "self_attn.k_conv1d.weight" }, .{ "conv_v", "self_attn.v_conv1d.weight" }, .{ "A_log", "self_attn.A_log" }, .{ "dt_bias", "self_attn.dt_bias" }, .{ "o_norm", "self_attn.o_norm.weight" } };
        inline for (kf) |f| @field(k, f[0]) = try fetch(src, sp, f[1]);
        if (k.A_log.count() < c.kda_heads) return error.TensorShape;
        l.attn = .{ .kda = k };
    } else {
        var m: Mla = undefined;
        const mf = .{ .{ "qa", "self_attn.q_a_proj.weight" }, .{ "qa_norm", "self_attn.q_a_layernorm.weight" }, .{ "qb", "self_attn.q_b_proj.weight" }, .{ "kva", "self_attn.kv_a_proj_with_mqa.weight" }, .{ "kva_norm", "self_attn.kv_a_layernorm.weight" }, .{ "kvb", "self_attn.kv_b_proj.weight" }, .{ "g", "self_attn.g_proj.weight" }, .{ "o", "self_attn.o_proj.weight" } };
        inline for (mf) |f| @field(m, f[0]) = try fetch(src, sp, f[1]);
        l.attn = .{ .mla = m };
    }
    if (!c.isMoe(i)) {
        l.mlp = .{ .dense = .{ .gate = try fetch(src, sp, "mlp.gate_proj.weight"), .up = try fetch(src, sp, "mlp.up_proj.weight"), .down = try fetch(src, sp, "mlp.down_proj.weight") } };
        return l;
    }
    var m: Moe = undefined;
    const ef = .{ .{ "router", "moe.gate.weight" }, .{ "bias", "moe.gate.e_score_correction_bias" }, .{ "down", "moe.routed_expert_down_proj.weight" }, .{ "norm", "moe.routed_expert_norm.weight" }, .{ "up", "moe.routed_expert_up_proj.weight" }, .{ "sh_gate", "shared_experts.gate_proj.weight" }, .{ "sh_up", "shared_experts.up_proj.weight" }, .{ "sh_down", "shared_experts.down_proj.weight" } };
    inline for (ef) |f| @field(m, f[0]) = try fetch(src, sp, f[1]);
    m.table = try device.buffer(@as(usize, c.experts) * @sizeOf(kernels.ExpertPtrs), mtl.ResourceOptions.shared);
    const table = m.table.slice(kernels.ExpertPtrs, c.experts);
    @memset(table, std.mem.zeroes(kernels.ExpertPtrs));
    for (first..last) |e| {
        specs.clearRetainingCapacity();
        try expertSpecs(c, i, @intCast(e), a, &specs);
        var t: [6]Tensor = undefined;
        for (specs.items, 0..) |s, j| t[j] = try src.get(s.name, s.dtype, s.shape[0..s.rank]);
        table[e] = .{ .w1p = kernels.addr(t[0].ref), .w1s = kernels.addr(t[1].ref), .w3p = kernels.addr(t[2].ref), .w3s = kernels.addr(t[3].ref), .w2p = kernels.addr(t[4].ref), .w2s = kernels.addr(t[5].ref) };
    }
    l.mlp = .{ .moe = m };
    return l;
}

pub fn head(c: *const Config, src: store.Source, gpa: std.mem.Allocator) !Head {
    var specs: std.ArrayList(Spec) = .empty;
    defer specs.deinit(gpa);
    try headSpecs(c, gpa, &specs);
    var h: Head = undefined;
    const names = .{ "embed", "lm_head", "norm", "res_norm", "res_proj" };
    inline for (names, 0..) |f, j| @field(h, f) = try src.get(specs.items[j].name, specs.items[j].dtype, specs.items[j].shape[0..specs.items[j].rank]);
    return h;
}

/// Bytes a token reads from layer i's weights, and the per-expert MXFP4 bytes (both from the specs).
pub fn layerBytes(c: *const Config, i: u32, a: std.mem.Allocator) !struct { dense: u64, expert: u64 } {
    var specs: std.ArrayList(Spec) = .empty;
    try layerSpecs(c, i, a, &specs);
    var dense: u64 = 0;
    for (specs.items) |s| dense += @as(u64, s.shape[0]) * s.shape[1] * s.shape[2] * s.dtype.size();
    specs.clearRetainingCapacity();
    try expertSpecs(c, i, 0, a, &specs);
    var expert: u64 = 0;
    for (specs.items) |s| expert += @as(u64, s.shape[0]) * s.shape[1];
    return .{ .dense = dense, .expert = expert };
}
