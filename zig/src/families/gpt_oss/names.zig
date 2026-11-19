//! The checkpoint's complete tensor namespace, stored dtypes and physical shapes; no data conversion.
const std = @import("std");
const st = @import("safetensors");
const config = @import("config.zig");

pub const Role = enum {
    embed,
    final_norm,
    head,
    attn_norm,
    mlp_norm,
    q,
    q_bias,
    k,
    k_bias,
    v,
    v_bias,
    o,
    o_bias,
    sink,
    router,
    router_bias,
    gate_up_blocks,
    gate_up_scales,
    gate_up_bias,
    down_blocks,
    down_scales,
    down_bias,
};
pub const Spec = struct { role: Role, layer: ?u32, dtype: st.DType, rank: u8, shape: [4]usize };
const Part = struct { key: []const u8, role: Role };
const globals = [_]Part{
    .{ .key = "model.embed_tokens.weight", .role = .embed },
    .{ .key = "model.norm.weight", .role = .final_norm },
    .{ .key = "lm_head.weight", .role = .head },
};
const parts = [_]Part{
    .{ .key = "input_layernorm.weight", .role = .attn_norm },
    .{ .key = "post_attention_layernorm.weight", .role = .mlp_norm },
    .{ .key = "self_attn.q_proj.weight", .role = .q },
    .{ .key = "self_attn.q_proj.bias", .role = .q_bias },
    .{ .key = "self_attn.k_proj.weight", .role = .k },
    .{ .key = "self_attn.k_proj.bias", .role = .k_bias },
    .{ .key = "self_attn.v_proj.weight", .role = .v },
    .{ .key = "self_attn.v_proj.bias", .role = .v_bias },
    .{ .key = "self_attn.o_proj.weight", .role = .o },
    .{ .key = "self_attn.o_proj.bias", .role = .o_bias },
    .{ .key = "self_attn.sinks", .role = .sink },
    .{ .key = "mlp.router.weight", .role = .router },
    .{ .key = "mlp.router.bias", .role = .router_bias },
    .{ .key = "mlp.experts.gate_up_proj_blocks", .role = .gate_up_blocks },
    .{ .key = "mlp.experts.gate_up_proj_scales", .role = .gate_up_scales },
    .{ .key = "mlp.experts.gate_up_proj_bias", .role = .gate_up_bias },
    .{ .key = "mlp.experts.down_proj_blocks", .role = .down_blocks },
    .{ .key = "mlp.experts.down_proj_scales", .role = .down_scales },
    .{ .key = "mlp.experts.down_proj_bias", .role = .down_bias },
};

fn spec(c: *const config.Config, role: Role, layer: ?u32) Spec {
    const h: usize = c.hidden;
    const i: usize = c.intermediate;
    const e: usize = c.experts;
    const dims: []const usize = switch (role) {
        .embed, .head => &.{ c.vocab, h },
        .final_norm, .attn_norm, .mlp_norm, .o_bias => &.{h},
        .q => &.{ c.qWidth(), h },
        .q_bias => &.{c.qWidth()},
        .k, .v => &.{ c.kvWidth(), h },
        .k_bias, .v_bias => &.{c.kvWidth()},
        .o => &.{ h, c.qWidth() },
        .sink => &.{c.heads},
        .router => &.{ e, h },
        .router_bias => &.{e},
        .gate_up_blocks => &.{ e, i * 2, h / 32, 16 },
        .gate_up_scales => &.{ e, i * 2, h / 32 },
        .gate_up_bias => &.{ e, i * 2 },
        .down_blocks => &.{ e, h, i / 32, 16 },
        .down_scales => &.{ e, h, i / 32 },
        .down_bias => &.{ e, h },
    };
    var out: Spec = .{ .role = role, .layer = layer, .dtype = switch (role) {
        .gate_up_blocks, .gate_up_scales, .down_blocks, .down_scales => .u8,
        else => .bf16,
    }, .rank = @intCast(dims.len), .shape = @splat(1) };
    @memcpy(out.shape[0..dims.len], dims);
    return out;
}

pub fn find(c: *const config.Config, name: []const u8) ?Spec {
    for (globals) |p| if (std.mem.eql(u8, name, p.key)) return spec(c, p.role, null);
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const tail = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, tail, '.') orelse return null;
    const digits = tail[0..dot];
    if (digits.len == 0 or (digits.len > 1 and digits[0] == '0')) return null;
    for (digits) |d| if (!std.ascii.isDigit(d)) return null;
    const layer = std.fmt.parseInt(u32, digits, 10) catch return null;
    if (layer >= c.layers) return null;
    for (parts) |p| if (std.mem.eql(u8, tail[dot + 1 ..], p.key)) return spec(c, p.role, layer);
    return null;
}

/// Enumerate all globals and each stored layer tensor. The sink consumes the ephemeral name during put().
pub fn walk(c: *const config.Config, sink: anytype) !void {
    for (globals) |p| try sink.put(p.key, spec(c, p.role, null));
    var buf: [192]u8 = undefined;
    for (0..c.layers) |layer| for (parts) |p| {
        const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ layer, p.key });
        try sink.put(name, spec(c, p.role, @intCast(layer)));
    };
}

fn bad(why: *config.Why, name: []const u8) error{BadGptOssTensor} {
    why.set(name);
    return error.BadGptOssTensor;
}

pub fn check(c: *const config.Config, header: *const st.Header, why: *config.Why) !void {
    for (header.keys(), header.values()) |name, entry| {
        const expected = find(c, name) orelse return bad(why, name);
        if (entry.dtype != expected.dtype or entry.rank != expected.rank or
            !std.mem.eql(usize, entry.shape[0..entry.rank], expected.shape[0..expected.rank])) return bad(why, name);
    }
    var checker: Checker = .{ .header = header, .why = why };
    try walk(c, &checker);
}

const Checker = struct {
    header: *const st.Header,
    why: *config.Why,

    pub fn put(self: *Checker, name: []const u8, _: Spec) !void {
        if (!self.header.contains(name)) return bad(self.why, name);
    }
};
