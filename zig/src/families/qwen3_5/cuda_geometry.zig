//! The widths of a Qwen3.5 hybrid text model (the 2B, Qwen3.8-27B) as the CUDA engine reads them from config.json:
//! what varies is read, what qwen35.cu fixes (head widths, taps, rotary layout, 4-bit groups of 64) is checked.
const std = @import("std");

pub const max_layers = 128;
/// qwen35.cu's fixed widths.
pub const dk = 128;
pub const taps = 4;
pub const head_dim = 256;
pub const max_query_heads = 32;

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

    /// One verified row's record of one linear layer (qwen35.cuh's log_bytes): decay and write by value head,
    /// key by key head, the tail after it.
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
    try is(rope, "rope_type", []const u8, "default");
    if (try number(rope, "rope_theta") != 10000000 or try number(rope, "partial_rotary_factor") != 0.25) return error.UnsupportedQwenRotary;
    const q = try object(root.get("quantization") orelse return error.UnsupportedQwenQuantization);
    if (try integer(q, "bits") != 4 or try integer(q, "group_size") != 64) return error.UnsupportedQwenQuantization;
    if (q.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return error.UnsupportedQwenQuantization;
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
    return g;
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
    const wide = try std.mem.replaceOwned(u8, std.testing.allocator, text, "\"head_dim\":256", "\"head_dim\":128");
    defer std.testing.allocator.free(wide);
    try std.testing.expectError(error.UnsupportedQwenConfig, parse(std.testing.allocator, wide));
}
