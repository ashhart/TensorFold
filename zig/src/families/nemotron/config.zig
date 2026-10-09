//! Nemotron-H dimensions from config.json, one Config for every backend, checked against the shapes our kernels serve.
const std = @import("std");

pub const Kind = enum { mamba, moe, attention };

pub const max_layers = 64;

/// The MTP head's acceptance at depth 1, 2, ... given the ones before it, until a stream has its own (model.py draft_prior).
pub const draft_prior = [_]f64{ 0.8, 0.72, 0.68, 0.62, 0.58, 0.55, 0.5, 0.5 };

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    /// Keep the reason, cut short if it is long.
    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Config = struct {
    hidden: usize,
    vocab: usize,
    layers: usize,
    kinds: [max_layers]Kind = undefined,
    mamba_heads: usize,
    mamba_head_dim: usize,
    groups: usize,
    state: usize,
    conv_kernel: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    experts: usize,
    top_k: usize,
    expert_width: usize,
    shared_width: usize,
    routed_scaling: f32,
    norm_topk: bool = true,
    eps: f32,
    dt_min: f32 = 0, // time_step_limit: the Mamba dt clamp, (0, inf) when the config sets none
    dt_max: f32 = std.math.inf(f32),
    group_size: usize = 64,
    bits: usize = 4,
    eos: [4]u32 = .{ 0, 0, 0, 0 },
    eos_count: usize = 0,

    pub fn inner(self: Config) usize {
        return self.mamba_heads * self.mamba_head_dim;
    }

    pub fn convDim(self: Config) usize {
        return self.inner() + 2 * self.groups * self.state;
    }

    pub fn projDim(self: Config) usize {
        return self.inner() + self.convDim() + self.mamba_heads;
    }

    pub fn qkvDim(self: Config) usize {
        return (self.heads + 2 * self.kv_heads) * self.head_dim;
    }

    /// A token's expert pairs: its routed experts, then the shared expert's two halves.
    pub fn slots(self: Config) usize {
        return self.top_k + 2;
    }

    pub fn count(self: Config, kind: Kind) usize {
        var n: usize = 0;
        for (self.kinds[0..self.layers]) |k| n += @intFromBool(k == kind);
        return n;
    }

    pub fn isEos(self: Config, token: u32) bool {
        for (self.eos[0..self.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// `dir`/config.json.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        var why: Why = .{};
        return parse(gpa, text, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
    }
};

fn int(obj: std.json.ObjectMap, key: []const u8) !usize {
    const v = obj.get(key) orelse {
        std.log.err("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    return @intCast(v.integer);
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn float(obj: std.json.ObjectMap, key: []const u8) !f32 {
    return @floatCast(try number(obj.get(key) orelse return error.BadConfig));
}

fn kindOf(name: []const u8) !Kind {
    if (std.mem.eql(u8, name, "mamba") or std.mem.eql(u8, name, "M")) return .mamba;
    if (std.mem.eql(u8, name, "moe") or std.mem.eql(u8, name, "E")) return .moe;
    if (std.mem.eql(u8, name, "attention") or std.mem.eql(u8, name, "*")) return .attention;
    return error.UnsupportedBlock;
}

/// The quantization block as MLX reads it: "quantization", else the "quantization_config" that mirrors it.
fn quantBlock(o: std.json.ObjectMap) ?std.json.ObjectMap {
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        if (o.get(key)) |v| if (v == .object and v.object.count() > 0) return v.object;
    }
    return null;
}

/// A quantization block's own fields; every other key names a module.
fn blockField(key: []const u8) bool {
    for ([_][]const u8{ "bits", "group_size", "mode", "quant_method" }) |f| if (std.mem.eql(u8, key, f)) return true;
    return false;
}

/// MLX's affine mode, the one an entry has when it names none.
fn affine(o: std.json.ObjectMap) bool {
    const m = o.get("mode") orelse return true;
    return m == .null or (m == .string and (m.string.len == 0 or std.ascii.eqlIgnoreCase(m.string, "affine")));
}

/// A positive integer field, `default` when absent or null, 0 when unreadable.
fn width(o: std.json.ObjectMap, key: []const u8, default: usize) usize {
    const v = o.get(key) orelse return default;
    return switch (v) {
        .null => default,
        .integer => |i| if (i > 0) @intCast(i) else 0,
        else => 0,
    };
}

/// The checkpoint's bits and group size; a per-module entry at any other width is refused, as the kernels read one.
fn quantization(c: *Config, o: std.json.ObjectMap, why: *Why) !void {
    const q = quantBlock(o) orelse {
        why.set("config.json declares no quantization; the native Nemotron kernels read MLX affine checkpoints", .{});
        return error.UnsupportedQuantization;
    };
    if (q.get("quant_method")) |m| if (!(m == .null or (m == .string and (std.ascii.eqlIgnoreCase(m.string, "mlx") or std.ascii.eqlIgnoreCase(m.string, "affine"))))) {
        why.set("config.json's quant_method is {f}; the native Nemotron kernels read MLX affine checkpoints", .{std.json.fmt(m, .{})});
        return error.UnsupportedQuantization;
    };
    if (!affine(q)) {
        why.set("config.json quantizes in mode {f}; the native Nemotron kernels read MLX affine checkpoints", .{std.json.fmt(q.get("mode").?, .{})});
        return error.UnsupportedQuantization;
    }
    c.bits = width(q, "bits", 0);
    c.group_size = width(q, "group_size", 64);
    if (c.bits == 0 or c.group_size == 0) {
        why.set("config.json's quantization has no readable bits or group_size", .{});
        return error.UnsupportedQuantization;
    }
    var it = q.iterator();
    while (it.next()) |e| if (!blockField(e.key_ptr.*)) try module(c.*, e.key_ptr.*, e.value_ptr.*, why);
}

/// A per-module entry as nn.quantize reads it: true, or a dict at the checkpoint's width, passes; anything else is refused.
fn module(c: Config, path: []const u8, v: std.json.Value, why: *Why) !void {
    switch (v) {
        .bool => |on| if (on) return,
        .object => |m| if (m.count() > 0) {
            // a dict's missing bits or group_size take to_quantized's affine defaults, not the top level's
            const bits = width(m, "bits", 4);
            const group = width(m, "group_size", 64);
            if (affine(m) and bits == c.bits and group == c.group_size) return;
            if (!affine(m) or bits == 0 or group == 0) {
                why.set("config.json quantizes {s} as {f}; the native Nemotron kernels read every matrix at {d}-bit in groups of {d}", .{ path, std.json.fmt(v, .{}), c.bits, c.group_size });
            } else {
                why.set("config.json quantizes {s} at {d}-bit in groups of {d}; the native Nemotron kernels read every matrix at the checkpoint's {d}-bit in groups of {d}", .{ path, bits, group, c.bits, c.group_size });
            }
            return error.MixedQuantization;
        },
        .null => {},
        else => return,
    }
    why.set("config.json leaves {s} unquantized; the native Nemotron kernels read every matrix at {d}-bit in groups of {d}", .{ path, c.bits, c.group_size });
    return error.MixedQuantization;
}

/// config.json's text; a refusal's reason goes to `why`.
pub fn parse(allocator: std.mem.Allocator, text: []const u8, why: *Why) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    if (!std.mem.eql(u8, (o.get("model_type") orelse return error.BadConfig).string, "nemotron_h")) return error.NotNemotronH;
    var c = Config{
        .hidden = try int(o, "hidden_size"),
        .vocab = try int(o, "vocab_size"),
        .layers = 0,
        .mamba_heads = try int(o, "mamba_num_heads"),
        .mamba_head_dim = try int(o, "mamba_head_dim"),
        .groups = try int(o, "n_groups"),
        .state = try int(o, "ssm_state_size"),
        .conv_kernel = try int(o, "conv_kernel"),
        .heads = try int(o, "num_attention_heads"),
        .kv_heads = try int(o, "num_key_value_heads"),
        .head_dim = 0,
        .experts = try int(o, "n_routed_experts"),
        .top_k = try int(o, "num_experts_per_tok"),
        .expert_width = try int(o, "moe_intermediate_size"),
        .shared_width = try int(o, "moe_shared_expert_intermediate_size"),
        .routed_scaling = if (o.get("routed_scaling_factor")) |v| @floatCast(try number(v)) else 1.0,
        .eps = if (o.get("layer_norm_epsilon")) |v| @floatCast(try number(v)) else 1e-5,
    };
    c.head_dim = if (o.get("head_dim")) |v| @intCast(v.integer) else c.hidden / c.heads;
    if (o.get("norm_topk_prob")) |v| c.norm_topk = v.bool;
    // hybrid_override_pattern (one character a block) wins over layers_block_type, as the Python loader reads them
    if (o.get("hybrid_override_pattern")) |v| if (v == .string and v.string.len > 0) {
        if (v.string.len > max_layers) return error.BadConfig;
        for (v.string, 0..) |ch, i| c.kinds[i] = try kindOf(&.{ch});
        c.layers = v.string.len;
    };
    if (c.layers == 0) {
        const blocks = (o.get("layers_block_type") orelse return error.BadConfig).array.items;
        if (blocks.len > max_layers) return error.BadConfig;
        for (blocks, 0..) |b, i| c.kinds[i] = try kindOf(b.string);
        c.layers = blocks.len;
    }
    if (o.get("eos_token_id")) |e| switch (e) {
        .integer => |i| {
            c.eos[0] = @intCast(i);
            c.eos_count = 1;
        },
        .array => |a| for (a.items[0..@min(a.items.len, 4)]) |x| {
            c.eos[c.eos_count] = @intCast(x.integer);
            c.eos_count += 1;
        },
        else => {},
    };
    if (o.get("time_step_limit")) |v| if (v == .array and v.array.items.len == 2) {
        c.dt_min = @floatCast(try number(v.array.items[0]));
        c.dt_max = @floatCast(try number(v.array.items[1]));
    };
    try quantization(&c, o, why);
    return c;
}

/// The shapes our kernels serve (Nemotron 3.5 Lightning 30B-A3B, 4-bit groups of 64): Metal's generated sources, CUDA's captured cubins.
pub fn checkShapes(c: Config) !void {
    const ok = c.hidden == 2688 and c.vocab == 131072 and c.mamba_heads == 64 and c.mamba_head_dim == 64 and
        c.groups == 8 and c.state == 128 and c.conv_kernel == 4 and c.heads == 32 and c.kv_heads == 2 and
        c.head_dim == 128 and c.experts == 128 and c.top_k == 6 and c.expert_width == 1856 and c.shared_width == 3712 and
        c.group_size == 64 and c.bits == 4;
    if (!ok) {
        std.log.err("this Nemotron-H's shapes differ from the kernels built for Nemotron 3.5 Lightning 30B-A3B", .{});
        return error.UnsupportedShapes;
    }
}

const lightning =
    \\{"model_type": "nemotron_h", "hidden_size": 2688, "vocab_size": 131072, "mamba_num_heads": 64,
    \\ "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 32,
    \\ "num_key_value_heads": 2, "head_dim": 128, "n_routed_experts": 128, "num_experts_per_tok": 6,
    \\ "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712, "routed_scaling_factor": 2.5,
    \\ "layer_norm_epsilon": 1e-05, "eos_token_id": [2, 11], "layers_block_type": ["mamba", "moe", "attention"],
    \\ "quantization": {"group_size": 64, "bits": 4}}
;

/// The Lightning config with its quantization block's text replaced by `block`.
fn withQuantization(block: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, std.testing.allocator, lightning, "\"quantization\": {\"group_size\": 64, \"bits\": 4}", block);
}

/// parse() must refuse the Lightning config with quantization `block`, its reason naming each of `words`.
fn expectRefused(block: []const u8, err: anyerror, words: []const []const u8) !void {
    const text = try withQuantization(block);
    defer std.testing.allocator.free(text);
    var why: Why = .{};
    try std.testing.expectError(err, parse(std.testing.allocator, text, &why));
    for (words) |word| if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    };
}

test "per-module quantization entries read as MLX reads them, any other width refused" {
    const fc1 = "backbone.layers.1.mixer.switch_mlp.fc1";
    for ([_][]const u8{
        "\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\", \"lm_head\": true}",
        "\"quantization\": {\"bits\": 4, \"" ++ fc1 ++ "\": {\"group_size\": 64, \"bits\": 4}, \"lm_head\": {\"mode\": \"affine\"}}",
        "\"quantization_config\": {\"group_size\": 64, \"bits\": 4, \"quant_method\": \"mlx\", \"oq_note\": \"metadata\"}",
    }) |block| {
        const text = try withQuantization(block);
        defer std.testing.allocator.free(text);
        var why: Why = .{};
        try checkShapes(try parse(std.testing.allocator, text, &why));
    }
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"" ++ fc1 ++ "\": {\"group_size\": 64, \"bits\": 8}}", error.MixedQuantization, &.{ fc1, "8-bit in groups of 64" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"backbone.embeddings\": {\"bits\": 6}}", error.MixedQuantization, &.{ "backbone.embeddings", "6-bit" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {\"group_size\": 32}}", error.MixedQuantization, &.{ "lm_head", "4-bit in groups of 32" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": false}", error.MixedQuantization, &.{ "lm_head", "unquantized" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {}}", error.MixedQuantization, &.{ "lm_head", "unquantized" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {\"mode\": \"mxfp8\"}}", error.MixedQuantization, &.{ "lm_head", "mxfp8" });
    try expectRefused("\"quantization\": {\"group_size\": 32, \"bits\": 4, \"mode\": \"mxfp4\"}", error.UnsupportedQuantization, &.{"mxfp4"});
    try expectRefused("\"quantization_config\": {\"quant_method\": \"fp8\", \"weight_block_size\": [128, 128]}", error.UnsupportedQuantization, &.{"fp8"});
    try expectRefused("\"quantization\": {\"group_size\": 64}", error.UnsupportedQuantization, &.{"bits"});
    try expectRefused("\"tie_word_embeddings\": false", error.UnsupportedQuantization, &.{"no quantization"});
}

test "parse the Lightning config" {
    var why: Why = .{};
    const c = try parse(std.testing.allocator, lightning, &why);
    try std.testing.expectEqual(@as(usize, 3), c.layers);
    try std.testing.expectEqual(Kind.attention, c.kinds[2]);
    try std.testing.expectEqual(@as(usize, 10304), c.projDim());
    try std.testing.expectEqual(@as(usize, 6144), c.convDim());
    try std.testing.expectEqual(@as(usize, 4608), c.qkvDim());
    try std.testing.expectEqual(@as(usize, 8), c.slots());
    try std.testing.expect(c.isEos(11) and !c.isEos(3));
    try std.testing.expectEqual(std.math.inf(f32), c.dt_max);
    try checkShapes(c);
}
