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
        return parse(gpa, text);
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

pub fn parse(allocator: std.mem.Allocator, text: []const u8) !Config {
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
    if (o.get("quantization")) |q| {
        c.group_size = @intCast(q.object.get("group_size").?.integer);
        c.bits = @intCast(q.object.get("bits").?.integer);
    }
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

test "parse the Lightning config" {
    const text =
        \\{"model_type": "nemotron_h", "hidden_size": 2688, "vocab_size": 131072, "mamba_num_heads": 64,
        \\ "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 32,
        \\ "num_key_value_heads": 2, "head_dim": 128, "n_routed_experts": 128, "num_experts_per_tok": 6,
        \\ "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712, "routed_scaling_factor": 2.5,
        \\ "layer_norm_epsilon": 1e-05, "eos_token_id": [2, 11], "layers_block_type": ["mamba", "moe", "attention"],
        \\ "quantization": {"group_size": 64, "bits": 4}}
    ;
    const c = try parse(std.testing.allocator, text);
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
