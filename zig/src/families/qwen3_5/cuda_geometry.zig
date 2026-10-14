//! A Qwen3.5 hybrid's widths (2B, 27B) as the CUDA engine reads them from config.json; what qwen35.cu fixes is checked.
const std = @import("std");
const core = @import("core");

pub const max_layers = 128;
/// qwen35.cu's fixed widths.
pub const dk = 128;
pub const taps = 4;
pub const head_dim = 256;
pub const max_query_heads = 32;

/// A layer's projections, each its own MLX width in a mixed checkpoint.
pub const Slot = enum { gate, up, down, qkv, z, a, b, out, q, k, v, o };

/// Each slot's module under its layer (`layers.{i}.` before it), as the config's quantization table names it.
pub const slot_modules = std.enums.EnumArray(Slot, []const u8).init(.{
    .gate = "mlp.gate_proj",
    .up = "mlp.up_proj",
    .down = "mlp.down_proj",
    .qkv = "linear_attn.in_proj_qkv",
    .z = "linear_attn.in_proj_z",
    .a = "linear_attn.in_proj_a",
    .b = "linear_attn.in_proj_b",
    .out = "linear_attn.out_proj",
    .q = "self_attn.q_proj",
    .k = "self_attn.k_proj",
    .v = "self_attn.v_proj",
    .o = "self_attn.o_proj",
});

/// The MLX code width of every quantized module (2 to 8 bits, groups of 64): a uniform checkpoint's are all one.
pub const Widths = struct {
    embed: u8 = 4,
    head: u8 = 4,
    proj: [max_layers][std.enums.values(Slot).len]u8 = @splat(@splat(4)),

    pub fn of(w: *const Widths, layer: usize, slot: Slot) u8 {
        return w.proj[layer][@backingInt(slot)];
    }

    /// Whether any module is not 4 bits: the loader then packs that module from its own width.
    pub fn mixed(w: *const Widths, layers: usize) bool {
        if (w.embed != 4 or w.head != 4) return true;
        for (w.proj[0..layers]) |row| for (row) |b| if (b != 4) return true;
        return false;
    }
};

pub const Geometry = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    k_heads: usize, // linear-attention key heads
    v_heads: usize, // linear-attention value heads
    q_heads: usize, // attention query heads
    kv_heads: usize, // attention key/value heads
    eps: f32,
    tied: bool,
    context: usize,
    linear: [max_layers]bool,
    bits: Widths = .{},

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Geometry {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// Value-head width of the gated delta rule's rows (and of z and the out projection's input).
    pub fn valueWidth(g: Geometry) usize {
        return g.v_heads * dk;
    }

    /// Channels the causal convolution sees: q, k and v.
    pub fn convWidth(g: Geometry) usize {
        return (2 * g.k_heads + g.v_heads) * dk;
    }

    /// Attention output width (q_heads of head_dim), the o projection's input.
    pub fn attnWidth(g: Geometry) usize {
        return g.q_heads * head_dim;
    }

    pub fn deltas(g: Geometry) usize {
        var n: usize = 0;
        for (g.linear[0..g.layers]) |l| n += @intFromBool(l);
        return n;
    }

    pub fn attentions(g: Geometry) usize {
        return g.layers - g.deltas();
    }

    /// Bytes of one linear layer's recurrence (fp32) and convolution tail (bf16).
    pub fn recBytes(g: Geometry) usize {
        return g.v_heads * dk * dk * 4;
    }

    pub fn convBytes(g: Geometry) usize {
        return (taps - 1) * g.convWidth() * 2;
    }

    /// A stream's recurrences and tails.
    pub fn stateBytes(g: Geometry) usize {
        return g.deltas() * (g.recBytes() + g.convBytes());
    }

    /// A row's record of a linear layer (qwen35.cuh log_bytes): decay and write a value head, key a key head, the tail.
    pub fn logRowBytes(g: Geometry) usize {
        return g.v_heads * 4 + g.v_heads * dk * 4 + g.k_heads * dk * 2 + g.convBytes();
    }

    /// A stream's log of a window's `rows` rows.
    pub fn logBytes(g: Geometry, rows: usize) usize {
        return rows * g.deltas() * g.logRowBytes();
    }

    /// A stream's keys and values at a `cap`-token window.
    pub fn kvBytes(g: Geometry, cap: usize) usize {
        return g.attentions() * 2 * g.kv_heads * cap * head_dim * 2;
    }
};

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadQwenConfig;
}

fn integer(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadQwenConfig;
    if (v != .integer or v.integer < 1) return error.BadQwenConfig;
    return @intCast(v.integer);
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadQwenConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadQwenConfig,
    };
}

fn is(o: std.json.ObjectMap, key: []const u8, comptime T: type, want: T) !void {
    const v = o.get(key) orelse return error.UnsupportedQwenConfig;
    const ok = switch (T) {
        bool => v == .bool and v.bool == want,
        []const u8 => v == .string and std.mem.eql(u8, v.string, want),
        usize => v == .integer and v.integer == want,
        else => @compileError("type"),
    };
    if (!ok) return error.UnsupportedQwenConfig;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Geometry {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    const root = try object(doc.value);
    try is(root, "model_type", []const u8, "qwen3_5");
    const t = try object(root.get("text_config") orelse return error.BadQwenConfig);
    try is(t, "model_type", []const u8, "qwen3_5_text");
    try is(t, "attention_bias", bool, false);
    try is(t, "attn_output_gate", bool, true);
    try is(t, "hidden_act", []const u8, "silu");
    try is(t, "head_dim", usize, head_dim);
    try is(t, "linear_key_head_dim", usize, dk);
    try is(t, "linear_value_head_dim", usize, dk);
    try is(t, "linear_conv_kernel_dim", usize, taps);
    const rope = try object(t.get("rope_parameters") orelse return error.UnsupportedQwenRotary);
    try is(rope, if (rope.get("rope_type") != null) "rope_type" else "type", []const u8, "default"); // "type": the older key
    if (try number(rope, "rope_theta") != 10000000 or try number(rope, "partial_rotary_factor") != 0.25) return error.UnsupportedQwenRotary;
    const q = try object(root.get("quantization") orelse return error.UnsupportedQwenQuantization);
    var g: Geometry = .{
        .hidden = try integer(t, "hidden_size"),
        .intermediate = try integer(t, "intermediate_size"),
        .layers = try integer(t, "num_hidden_layers"),
        .vocab = try integer(t, "vocab_size"),
        .k_heads = try integer(t, "linear_num_key_heads"),
        .v_heads = try integer(t, "linear_num_value_heads"),
        .q_heads = try integer(t, "num_attention_heads"),
        .kv_heads = try integer(t, "num_key_value_heads"),
        .eps = @floatCast(try number(t, "rms_norm_eps")),
        .tied = blk: {
            const v = t.get("tie_word_embeddings") orelse root.get("tie_word_embeddings") orelse break :blk false;
            if (v != .bool) return error.BadQwenConfig;
            break :blk v.bool;
        },
        .context = try integer(t, "max_position_embeddings"),
        .linear = @splat(false),
    };
    if (g.layers > max_layers or g.hidden % 256 != 0 or g.hidden > 8192 or g.intermediate % 64 != 0) return error.UnsupportedQwenGeometry;
    if (g.v_heads % g.k_heads != 0 or g.q_heads % g.kv_heads != 0 or g.q_heads > max_query_heads) return error.UnsupportedQwenGeometry;
    const types = t.get("layer_types") orelse return error.UnsupportedQwenGeometry;
    if (types != .array or types.array.items.len != g.layers) return error.UnsupportedQwenGeometry;
    for (types.array.items, 0..) |kind, i| {
        if (kind != .string) return error.UnsupportedQwenGeometry;
        if (std.mem.eql(u8, kind.string, "linear_attention")) g.linear[i] = true else if (!std.mem.eql(u8, kind.string, "full_attention")) return error.UnsupportedQwenGeometry;
    }
    g.bits = try widths(gpa, root, q, g);
    return g;
}

/// Every module's width from the quantization table, global then per module: MLX affine, groups of 64.
fn widths(gpa: std.mem.Allocator, root: std.json.ObjectMap, q: std.json.ObjectMap, g: Geometry) !Widths {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const mlx = core.quant.mlx;
    const table = (mlx.detect(arena.allocator(), .{ .root = root, .quantization = q }) catch return error.UnsupportedQwenQuantization) orelse
        return error.UnsupportedQwenQuantization;
    const Look = struct {
        fn bits(t: mlx.Config, key: []const u8) !u8 {
            const w = t.width(key) catch return error.UnsupportedQwenQuantization;
            if (w.group != 64) return error.UnsupportedQwenQuantization;
            return w.bits;
        }
    };
    var out: Widths = .{ .embed = try Look.bits(table, "embed_tokens"), .head = undefined };
    out.head = if (g.tied) out.embed else try Look.bits(table, "lm_head");
    var buf: [96]u8 = undefined;
    for (0..g.layers) |i| for (std.enums.values(Slot)) |slot| {
        const used = switch (slot) {
            .gate, .up, .down => true,
            .qkv, .z, .a, .b, .out => g.linear[i],
            .q, .k, .v, .o => !g.linear[i],
        };
        if (used) out.proj[i][@backingInt(slot)] = try Look.bits(table, try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, slot_modules.get(slot) }));
    };
    return out;
}

test "the 27B's widths: 48 value heads over 16 key heads, 24 query heads over 4" {
    const text =
        \\{"model_type":"qwen3_5","quantization":{"group_size":64,"bits":4,"mode":"affine"},"tie_word_embeddings":false,
        \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
        \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,"rms_norm_eps":1e-06,
        \\ "max_position_embeddings":262144,"tie_word_embeddings":false,
        \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
        \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
    ;
    const g = try parse(std.testing.allocator, text);
    try std.testing.expectEqual(@as(usize, 3), g.deltas());
    try std.testing.expectEqual(@as(usize, 10240), g.convWidth());
    try std.testing.expectEqual(@as(usize, 6144), g.attnWidth());
    try std.testing.expectEqual(@as(usize, 48 * 128 * 128 * 4), g.recBytes());
    try std.testing.expect(!g.tied);
    try std.testing.expect(!g.bits.mixed(g.layers)); // uniform 4-bit: every module the global width
    const wide = try std.mem.replaceOwned(u8, std.testing.allocator, text, "\"head_dim\":256", "\"head_dim\":128");
    defer std.testing.allocator.free(wide);
    try std.testing.expectError(error.UnsupportedQwenConfig, parse(std.testing.allocator, wide));
}

test "a mixed checkpoint's widths by module, under any spelling; other groups and modes refused" {
    const text =
        \\{"model_type":"qwen3_5","quantization":{"group_size":64,"bits":2,"mode":"affine",
        \\   "language_model.model.embed_tokens":{"group_size":64,"bits":8},
        \\   "language_model.model.layers.0.linear_attn.in_proj_a":{"group_size":64,"bits":5},
        \\   "model.layers.1.self_attn.v_proj":{"group_size":64,"bits":6},
        \\   "language_model.model.layers.1.mlp.down_proj":{"group_size":64,"bits":3}},
        \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
        \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        \\ "hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":2,"vocab_size":248320,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":16,"num_attention_heads":8,"num_key_value_heads":2,"rms_norm_eps":1e-06,
        \\ "max_position_embeddings":262144,"tie_word_embeddings":true,
        \\ "layer_types":["linear_attention","full_attention"],
        \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
    ;
    const g = try parse(std.testing.allocator, text);
    try std.testing.expectEqual(@as(u8, 8), g.bits.embed);
    try std.testing.expectEqual(@as(u8, 8), g.bits.head); // tied: the embedding's width
    try std.testing.expectEqual(@as(u8, 5), g.bits.of(0, .a));
    try std.testing.expectEqual(@as(u8, 2), g.bits.of(0, .qkv));
    try std.testing.expectEqual(@as(u8, 6), g.bits.of(1, .v));
    try std.testing.expectEqual(@as(u8, 3), g.bits.of(1, .down));
    try std.testing.expectEqual(@as(u8, 2), g.bits.of(1, .gate));
    try std.testing.expect(g.bits.mixed(g.layers));
    const g32 = try std.mem.replaceOwned(u8, std.testing.allocator, text, "\"group_size\":64,\"bits\":3", "\"group_size\":32,\"bits\":3");
    defer std.testing.allocator.free(g32);
    try std.testing.expectError(error.UnsupportedQwenQuantization, parse(std.testing.allocator, g32));
    const gptq = try std.mem.replaceOwned(u8, std.testing.allocator, text, "\"mode\":\"affine\"", "\"mode\":\"gptq\"");
    defer std.testing.allocator.free(gptq);
    try std.testing.expectError(error.UnsupportedQwenQuantization, parse(std.testing.allocator, gptq));
}
