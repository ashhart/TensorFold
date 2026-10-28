//! Ling-3.1-flash's weight map: each tensor name the config implies, by role, checked against an index.
const std = @import("std");
const config = @import("config.zig");

const Config = config.Config;

pub const Role = enum {
    embed,
    final_norm,
    lm_head,
    attn_norm,
    mlp_norm,
    kda_q,
    kda_k,
    kda_v,
    kda_q_conv,
    kda_k_conv,
    kda_v_conv,
    kda_b,
    kda_f,
    kda_g,
    kda_a_log,
    kda_dt_bias,
    kda_o_norm,
    kda_o,
    mla_q,
    mla_q_a,
    mla_q_a_norm,
    mla_q_b,
    mla_kv_a,
    mla_kv_a_norm,
    mla_kv_b,
    mla_gate,
    mla_o,
    dense_gate,
    dense_up,
    dense_down,
    router,
    router_bias,
    expert_gate,
    expert_up,
    expert_down,
    shared_gate,
    shared_up,
    shared_down,
    mtp_enorm,
    mtp_hnorm,
    mtp_eh,
    mtp_final_norm,
};

/// One stored tensor: its role, layer (null for the model's own), routed expert, and whether it is a scale tensor.
pub const Tensor = struct { role: Role, layer: ?u32 = null, expert: ?u32 = null, scale: bool = false };

/// linear: `.weight` plus its format's side tensors (side()); weight: `.weight` only; bare: the module's own name.
const Form = enum { linear, weight, bare };

const Part = struct { role: Role, path: []const u8, form: Form };

const norms = [_]Part{
    .{ .role = .attn_norm, .path = "input_layernorm", .form = .weight },
    .{ .role = .mlp_norm, .path = "post_attention_layernorm", .form = .weight },
};

const kda = [_]Part{
    .{ .role = .kda_q, .path = "attention.q_proj", .form = .linear },
    .{ .role = .kda_k, .path = "attention.k_proj", .form = .linear },
    .{ .role = .kda_v, .path = "attention.v_proj", .form = .linear },
    .{ .role = .kda_q_conv, .path = "attention.q_conv1d", .form = .weight },
    .{ .role = .kda_k_conv, .path = "attention.k_conv1d", .form = .weight },
    .{ .role = .kda_v_conv, .path = "attention.v_conv1d", .form = .weight },
    .{ .role = .kda_b, .path = "attention.b_proj", .form = .linear },
    .{ .role = .kda_f, .path = "attention.f_proj", .form = .linear },
    .{ .role = .kda_g, .path = "attention.g_proj", .form = .linear },
    .{ .role = .kda_a_log, .path = "attention.A_log", .form = .bare },
    .{ .role = .kda_dt_bias, .path = "attention.dt_bias", .form = .bare },
    .{ .role = .kda_o_norm, .path = "attention.o_norm", .form = .weight },
    .{ .role = .kda_o, .path = "attention.o_proj", .form = .linear },
};

const mla_q_lora = [_]Part{
    .{ .role = .mla_q_a, .path = "attention.q_a_proj", .form = .linear },
    .{ .role = .mla_q_a_norm, .path = "attention.q_a_layernorm", .form = .weight },
    .{ .role = .mla_q_b, .path = "attention.q_b_proj", .form = .linear },
};

const mla_q_whole = [_]Part{.{ .role = .mla_q, .path = "attention.q_proj", .form = .linear }};

const mla = [_]Part{
    .{ .role = .mla_kv_a, .path = "attention.kv_a_proj_with_mqa", .form = .linear },
    .{ .role = .mla_kv_a_norm, .path = "attention.kv_a_layernorm", .form = .weight },
    .{ .role = .mla_kv_b, .path = "attention.kv_b_proj", .form = .linear },
    .{ .role = .mla_o, .path = "attention.dense", .form = .linear },
};

const mla_gate = [_]Part{.{ .role = .mla_gate, .path = "attention.g_proj", .form = .linear }};

const dense = [_]Part{
    .{ .role = .dense_gate, .path = "mlp.gate_proj", .form = .linear },
    .{ .role = .dense_up, .path = "mlp.up_proj", .form = .linear },
    .{ .role = .dense_down, .path = "mlp.down_proj", .form = .linear },
};

const router = [_]Part{.{ .role = .router, .path = "mlp.gate", .form = .linear }};
const router_bias = [_]Part{.{ .role = .router_bias, .path = "mlp.gate.expert_bias", .form = .bare }};

const shared = [_]Part{
    .{ .role = .shared_gate, .path = "mlp.shared_experts.gate_proj", .form = .linear },
    .{ .role = .shared_up, .path = "mlp.shared_experts.up_proj", .form = .linear },
    .{ .role = .shared_down, .path = "mlp.shared_experts.down_proj", .form = .linear },
};

const expert = [_]Part{
    .{ .role = .expert_gate, .path = "gate_proj", .form = .linear },
    .{ .role = .expert_up, .path = "up_proj", .form = .linear },
    .{ .role = .expert_down, .path = "down_proj", .form = .linear },
};

const mtp = [_]Part{
    .{ .role = .mtp_enorm, .path = "enorm", .form = .weight },
    .{ .role = .mtp_hnorm, .path = "hnorm", .form = .weight },
    .{ .role = .mtp_eh, .path = "eh_proj", .form = .linear },
    .{ .role = .mtp_final_norm, .path = "final_layernorm", .form = .weight },
};

/// The part's tensors under `prefix`: the name built in `buf`, handed to `sink.put(name, tensor)`.
fn emit(c: *const Config, sink: anytype, prefix: []const u8, p: Part, t: Tensor) !void {
    var buf: [256]u8 = undefined;
    const module = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, p.path });
    var name: [256]u8 = undefined;
    var u = t;
    u.role = p.role;
    switch (p.form) {
        .bare => try sink.put(module, u),
        .weight, .linear => try sink.put(try std.fmt.bufPrint(&name, "{s}.weight", .{module}), u),
    }
    if (p.form != .linear) return;
    u.scale = true;
    for (side(c.format(module))) |tail| try sink.put(try std.fmt.bufPrint(&name, "{s}.{s}", .{ module, tail }), u);
}

/// The tensors a linear stores beside `.weight`: FP8's block scales, ModelOpt NVFP4's group, global and input scales.
pub fn side(f: config.Quant) []const []const u8 {
    return switch (f) {
        .bf16 => &.{},
        .fp8_block128 => &.{"weight_scale_inv"},
        .nvfp4 => &.{ "weight_scale", "weight_scale_2", "input_scale" },
    };
}

/// What the loader reads for a role: a linear [n, k] (`.weight` and its side tensors), `values` values, a conv.
pub const Dims = union(enum) { linear: [2]usize, values: usize, conv: [3]usize };

/// The loader's shape for each role (weights.zig reads every tensor through it).
pub fn dims(c: *const Config, role: Role) Dims {
    const d: usize = c.hidden;
    const n: usize = @as(usize, c.kda_heads) * c.kda_dim;
    const h: usize = c.mla_heads;
    return switch (role) {
        .embed, .lm_head => .{ .linear = .{ c.vocab, d } },
        .final_norm, .attn_norm, .mlp_norm, .mtp_enorm, .mtp_hnorm, .mtp_final_norm => .{ .values = d },
        .kda_q, .kda_k, .kda_v, .kda_f, .kda_g => .{ .linear = .{ n, d } },
        .kda_b => .{ .linear = .{ c.kda_heads, d } },
        .kda_o => .{ .linear = .{ d, n } },
        .kda_q_conv, .kda_k_conv, .kda_v_conv => .{ .conv = .{ n, 1, c.conv } },
        .kda_a_log => .{ .values = c.kda_heads },
        .kda_dt_bias => .{ .values = n },
        .kda_o_norm => .{ .values = c.kda_dim },
        .mla_q => .{ .linear = .{ h * c.qkDim(), d } },
        .mla_q_a => .{ .linear = .{ c.q_lora, d } },
        .mla_q_a_norm => .{ .values = c.q_lora },
        .mla_q_b => .{ .linear = .{ h * c.qkDim(), c.q_lora } },
        .mla_kv_a => .{ .linear = .{ c.kv_lora + c.rope_dim, d } },
        .mla_kv_a_norm => .{ .values = c.kv_lora },
        .mla_kv_b => .{ .linear = .{ h * (c.nope + c.v_dim), c.kv_lora } },
        .mla_gate => .{ .linear = .{ h, d } },
        .mla_o => .{ .linear = .{ d, h * c.v_dim } },
        .dense_gate, .dense_up => .{ .linear = .{ c.dense_inter, d } },
        .dense_down => .{ .linear = .{ d, c.dense_inter } },
        .router => .{ .linear = .{ c.experts, d } },
        .router_bias => .{ .values = c.experts },
        .expert_gate, .expert_up => .{ .linear = .{ c.moe_inter, d } },
        .expert_down => .{ .linear = .{ d, c.moe_inter } },
        .shared_gate, .shared_up => .{ .linear = .{ c.shared_inter, d } },
        .shared_down => .{ .linear = .{ d, c.shared_inter } },
        .mtp_eh => .{ .linear = .{ d, 2 * d } },
    };
}

/// A role's value count: a linear's n k, a small tensor's values, a conv's n taps.
pub fn numel(c: *const Config, role: Role) usize {
    return switch (dims(c, role)) {
        .linear => |x| x[0] * x[1],
        .values => |x| x,
        .conv => |x| x[0] * x[1] * x[2],
    };
}

const tables = [_][]const Part{ &norms, &kda, &mla_q_lora, &mla_q_whole, &mla, &mla_gate, &dense, &router, &router_bias, &shared, &expert, &mtp };

/// Role's module in layer i (`e` for routed experts) as the map names it: the name before `.weight`.
pub fn moduleName(buf: []u8, i: u32, role: Role, e: ?u32) ![]const u8 {
    switch (role) {
        .embed => return "model.word_embeddings",
        .final_norm => return "model.norm",
        .lm_head => return "lm_head",
        .expert_gate, .expert_up, .expert_down => {
            const p = for (expert) |x| (if (x.role == role) break x) else unreachable;
            return std.fmt.bufPrint(buf, "model.layers.{d}.mlp.experts.{d}.{s}", .{ i, e orelse return error.NoExpert, p.path });
        },
        else => {},
    }
    for (tables) |t| for (t) |p| if (p.role == role) return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ i, p.path });
    unreachable;
}

fn emitAll(c: *const Config, sink: anytype, prefix: []const u8, parts: []const Part, t: Tensor) !void {
    for (parts) |p| try emit(c, sink, prefix, p, t);
}

/// Layer i's tensors (an MTP layer when i >= c.layers): norms, the mixer its kind names, the MLP or MoE.
pub fn layer(c: *const Config, i: u32, sink: anytype) !void {
    var buf: [64]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&buf, "model.layers.{d}.", .{i});
    const t: Tensor = .{ .role = .attn_norm, .layer = i };
    try emitAll(c, sink, prefix, &norms, t);
    switch (c.kind(i)) {
        .kda => try emitAll(c, sink, prefix, &kda, t),
        .mla => {
            try emitAll(c, sink, prefix, if (c.q_lora > 0) &mla_q_lora else &mla_q_whole, t);
            try emitAll(c, sink, prefix, &mla, t);
            if (c.head_gate) try emitAll(c, sink, prefix, &mla_gate, t);
        },
    }
    if (!c.isMoe(i)) return emitAll(c, sink, prefix, &dense, t);
    try emitAll(c, sink, prefix, &router, t);
    if (c.expert_bias) try emitAll(c, sink, prefix, &router_bias, t);
    if (c.shared_inter > 0) try emitAll(c, sink, prefix, &shared, t);
    if (i >= c.layers) try emitAll(c, sink, prefix, &mtp, t);
    var ebuf: [96]u8 = undefined;
    for (0..c.experts) |e| {
        const ep = try std.fmt.bufPrint(&ebuf, "{s}mlp.experts.{d}.", .{ prefix, e });
        try emitAll(c, sink, ep, &expert, .{ .role = .expert_gate, .layer = i, .expert = @intCast(e) });
    }
}

/// The model's own tensors: the embeddings, the final norm and, unless tied, the head.
pub fn globals(c: *const Config, sink: anytype) !void {
    try emit(c, sink, "model.", .{ .role = .embed, .path = "word_embeddings", .form = .weight }, .{ .role = .embed });
    try emit(c, sink, "model.", .{ .role = .final_norm, .path = "norm", .form = .weight }, .{ .role = .final_norm });
    if (!c.tie) try emit(c, sink, "", .{ .role = .lm_head, .path = "lm_head", .form = .linear }, .{ .role = .lm_head });
}

/// Every tensor the config implies, backbone then MTP layers.
pub fn all(c: *const Config, sink: anytype) !void {
    try globals(c, sink);
    for (0..c.layers + c.mtp) |i| try layer(c, @intCast(i), sink);
}

/// The map against an index's names: expected tensors (MTP ones counted apart), missing ones and unaccounted ones.
pub const Report = struct {
    expected: usize = 0,
    mtp: usize = 0,
    missing: std.ArrayList([]const u8) = .empty,
    unknown: std.ArrayList([]const u8) = .empty,
};

const Checker = struct {
    arena: std.mem.Allocator,
    seen: *std.StringHashMapUnmanaged(bool),
    c: *const Config,
    r: *Report,

    pub fn put(k: *Checker, name: []const u8, t: Tensor) !void {
        k.r.expected += 1;
        if (t.layer != null and t.layer.? >= k.c.layers) k.r.mtp += 1;
        if (k.seen.getPtr(name)) |hit| {
            if (hit.*) return error.DuplicateName;
            hit.* = true;
        } else try k.r.missing.append(k.arena, try k.arena.dupe(u8, name));
    }
};

/// Every name the map expects is in `index` and every index name is one the map expects (lists in `arena`).
pub fn check(arena: std.mem.Allocator, c: *const Config, index: []const []const u8) !Report {
    var seen: std.StringHashMapUnmanaged(bool) = .empty;
    try seen.ensureTotalCapacity(arena, @intCast(index.len));
    for (index) |n| if (seen.fetchPutAssumeCapacity(n, false) != null) return error.DuplicateName;
    var r: Report = .{};
    var k: Checker = .{ .arena = arena, .seen = &seen, .c = c, .r = &r };
    try all(c, &k);
    for (index) |n| if (!seen.get(n).?) try r.unknown.append(arena, n);
    return r;
}
