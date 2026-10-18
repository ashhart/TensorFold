//! Every family Living Weights knows, as data: the learner, the shard writer and the server's view read these.
const std = @import("std");
const family = @import("family.zig");
const Family = family.Family;

/// Nemotron 3.5 Lightning (nemotron_h): relu^2 shared experts at the pattern's MoE layers, 4-bit groups of 64.
pub const nemotron_h: Family = .{
    .model_type = "nemotron_h",
    .names = .{
        .layers = "backbone.layers.",
        .expert = "mixer.shared_experts.",
        .inputs = &.{"up_proj"},
        .mtp = .{ .file = "mtp-4bit.safetensors", .base = "layers.1.mixer.shared_experts." },
    },
    .neuron = .{ .act = .relu2, .margin = 0.25 },
    .quant = .{ .bits = 4, .group = 64 },
    .keys = .{ .width = "moe_shared_expert_intermediate_size" },
    .experts = .pattern,
};

/// Qwen3.8 Flash Next (qwen4_exp): a SwiGLU shared expert at every layer behind its own gate, 4-bit groups of 32.
pub const qwen4_exp: Family = .{
    .model_type = "qwen4_exp",
    .names = .{
        .layers = "language_model.model.layers.",
        .expert = "mlp.shared_expert.",
        .inputs = &.{ "gate_proj", "up_proj" },
        .mtp = .{ .base = "language_model.mtp.layers.0.mlp.shared_expert." },
    },
    // silu(gate) is not zero below it as relu is: a wider margin keeps sigmoid(-16 * 0.5) = 3e-4 of it there
    .neuron = .{ .act = .swiglu, .gated = true, .margin = 0.5, .sharp = 16 },
    .quant = .{ .bits = 4, .group = 32 },
    .keys = .{ .scope = "text_config", .width = "shared_expert_intermediate_size" },
    .experts = .every,
};

/// Dense Qwen3.8-27B: @ZackSample #911, on the owned @BobClawblaw #836 engine.
pub const qwen3_5: Family = .{
    .model_type = "qwen3_5",
    .names = .{ .layers = "language_model.model.layers.", .expert = "mlp.", .inputs = &.{ "gate_proj", "up_proj" } },
    .neuron = .{ .act = .swiglu, .margin = 0.5, .sharp = 16 },
    .quant = .{ .bits = 4, .group = 64 },
    .keys = .{ .scope = "text_config", .width = "intermediate_size" },
    .experts = .every,
};

pub const all = [_]Family{ nemotron_h, qwen4_exp, qwen3_5 };

/// The family of a checkpoint's top-level model_type, or null.
pub fn byModelType(model_type: []const u8) ?Family {
    inline for (all) |f| if (std.mem.eql(u8, f.model_type, model_type)) return f;
    return null;
}

/// A config's layer letters (M mamba, E MoE, * attention, - MLP, ? other); a family with an expert at each: all E.
pub fn kinds(a: std.mem.Allocator, f: Family, o: std.json.ObjectMap) ![]const u8 {
    const n = intField(o, f.keys.layers);
    if (f.experts == .every) {
        const count = n orelse if (o.get("layer_types")) |v| (if (v == .array) v.array.items.len else 0) else 0;
        const out = try a.alloc(u8, count);
        @memset(out, 'E');
        return out;
    }
    if (o.get("hybrid_override_pattern")) |v| if (v == .string and v.string.len > 0) return a.dupe(u8, v.string);
    if (o.get("layers_block_type")) |v| if (v == .array) {
        const out = try a.alloc(u8, v.array.items.len);
        for (v.array.items, out) |b, *k| k.* = letter(if (b == .string) b.string else "");
        return out;
    };
    const out = try a.alloc(u8, n orelse 0);
    @memset(out, '?');
    return out;
}

/// The object a family's model keys sit in (its scope, else the top level).
pub fn scoped(f: Family, root: std.json.ObjectMap) ?std.json.ObjectMap {
    const s = f.keys.scope orelse return root;
    const v = root.get(s) orelse return null;
    return if (v == .object) v.object else null;
}

fn letter(name: []const u8) u8 {
    const names = [_]struct { []const u8, u8 }{ .{ "mamba", 'M' }, .{ "moe", 'E' }, .{ "attention", '*' }, .{ "mlp", '-' } };
    for (names) |n| if (std.mem.eql(u8, name, n[0])) return n[1];
    return '?';
}

pub fn intField(o: std.json.ObjectMap, key: []const u8) ?usize {
    const v = o.get(key) orelse return null;
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else null;
}

test "each family's model type finds it; a pattern family's letters come from its pattern, an every-layer one's are E" {
    try std.testing.expectEqualStrings("nemotron_h", byModelType("nemotron_h").?.model_type);
    try std.testing.expectEqualStrings("qwen4_exp", byModelType("qwen4_exp").?.model_type);
    try std.testing.expect(byModelType("glm") == null);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const n = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"layers_block_type\": [\"mamba\", \"moe\", \"attention\"]}", .{});
    try std.testing.expectEqualStrings("ME*", try kinds(a, nemotron_h, n.object));
    const q = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"text_config\": {\"num_hidden_layers\": 3}}", .{});
    try std.testing.expectEqualStrings("EEE", try kinds(a, qwen4_exp, scoped(qwen4_exp, q.object).?));
}
