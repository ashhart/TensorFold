//! Admission of a Gemma 4 E4B checkpoint (NVIDIA Nemotron 3.5 Bolt E4B) for the XPU runner: every geometry key pinned.
//! The family is `gemma4_e4b` (not `bolt`: another model of that name exists in the tree).
const std = @import("std");

pub const hidden = 2560;
pub const layers = 42;
pub const ff = 10240;
pub const heads = 8;
pub const kv_heads = 2;
pub const dim_s = 256; // sliding layers' head
pub const dim_g = 512; // global layers' head
pub const window = 512;
pub const shared_from = 24; // layers 24 .. 41 read the K/V of layer 22 (sliding) or 23 (global)
pub const ple = 256;
pub const ple_all = layers * ple;
pub const vocab = 262144;
pub const softcap: f32 = 30.0;
pub const eps: f32 = 1e-6;
pub const ring = 1536; // sliding K/V slots: the window plus a prefill block of 1024

pub fn isGlobal(layer: usize) bool {
    return layer % 6 == 5;
}

/// The physical K/V cache a layer reads: its own below `shared_from`, else the last layer of its type before it.
pub fn kvSource(layer: usize) usize {
    if (layer < shared_from) return layer;
    return if (isGlobal(layer)) 23 else 22;
}

pub const Error = error{
    BadGemmaConfig,
    NotGemma4E4B,
    UnsupportedGemmaGeometry,
    UnsupportedGemmaRotary,
    UnsupportedGemmaQuantization,
} || std.mem.Allocator.Error;

fn int(o: std.json.ObjectMap, key: []const u8) ?i64 {
    const v = o.get(key) orelse return null;
    return if (v == .integer) v.integer else null;
}

fn num(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn str(v: ?std.json.Value, want: []const u8) bool {
    const x = v orelse return false;
    return x == .string and std.mem.eql(u8, x.string, want);
}

fn flag(o: std.json.ObjectMap, key: []const u8) ?bool {
    const v = o.get(key) orelse return null;
    return if (v == .bool) v.bool else null;
}

/// Checks `config.json` against the one model this runner serves; the first mismatch names the error.
pub fn admit(gpa: std.mem.Allocator, bytes: []const u8) Error!void {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch return error.BadGemmaConfig;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadGemmaConfig;
    const root = parsed.value.object;
    if (!str(root.get("model_type"), "gemma4")) return error.NotGemma4E4B;
    const tv = root.get("text_config") orelse return error.BadGemmaConfig;
    if (tv != .object) return error.BadGemmaConfig;
    const t = tv.object;
    const dims = [_]struct { []const u8, i64 }{
        .{ "hidden_size", hidden },
        .{ "num_hidden_layers", layers },
        .{ "intermediate_size", ff },
        .{ "num_attention_heads", heads },
        .{ "num_key_value_heads", kv_heads },
        .{ "head_dim", dim_s },
        .{ "global_head_dim", dim_g },
        .{ "sliding_window", window },
        .{ "num_kv_shared_layers", layers - shared_from },
        .{ "hidden_size_per_layer_input", ple },
        .{ "vocab_size", vocab },
        .{ "vocab_size_per_layer_input", vocab },
    };
    for (dims) |d| if ((int(t, d[0]) orelse return error.NotGemma4E4B) != d[1]) return error.UnsupportedGemmaGeometry;
    // false: the global layers have their own V; absent keys are the model's defaults, a wrong one is refused
    if ((flag(t, "attention_k_eq_v") orelse false) or (flag(t, "enable_moe_block") orelse false) or (flag(t, "use_double_wide_mlp") orelse false) or (flag(t, "attention_bias") orelse false))
        return error.UnsupportedGemmaGeometry;
    if (!(flag(t, "tie_word_embeddings") orelse false) and !(flag(root, "tie_word_embeddings") orelse false)) return error.UnsupportedGemmaGeometry;
    if (!str(t.get("hidden_activation"), "gelu_pytorch_tanh")) return error.UnsupportedGemmaGeometry;
    if ((num(t.get("rms_norm_eps")) orelse 0) != 1e-6 or (num(t.get("final_logit_softcapping")) orelse 0) != 30.0) return error.UnsupportedGemmaGeometry;
    if (t.get("num_global_key_value_heads")) |v| if (v != .null and (num(v) orelse 0) != kv_heads) return error.UnsupportedGemmaGeometry;
    if (t.get("use_bidirectional_attention")) |v| if (v != .null and v != .bool) return error.UnsupportedGemmaGeometry;
    const lt = t.get("layer_types") orelse return error.UnsupportedGemmaGeometry;
    if (lt != .array or lt.array.items.len != layers) return error.UnsupportedGemmaGeometry;
    for (lt.array.items, 0..) |x, i| if (!str(x, if (isGlobal(i)) "full_attention" else "sliding_attention")) return error.UnsupportedGemmaGeometry;
    const rp = t.get("rope_parameters") orelse return error.UnsupportedGemmaRotary;
    if (rp != .object) return error.UnsupportedGemmaRotary;
    const sl = rp.object.get("sliding_attention") orelse return error.UnsupportedGemmaRotary;
    const gl = rp.object.get("full_attention") orelse return error.UnsupportedGemmaRotary;
    if (sl != .object or gl != .object) return error.UnsupportedGemmaRotary;
    if (!str(sl.object.get("rope_type"), "default") or (num(sl.object.get("rope_theta")) orelse 0) != 10000) return error.UnsupportedGemmaRotary;
    if (!str(gl.object.get("rope_type"), "proportional") or (num(gl.object.get("rope_theta")) orelse 0) != 1000000 or (num(gl.object.get("partial_rotary_factor")) orelse 0) != 0.25) return error.UnsupportedGemmaRotary;
    // weights: MLX affine 4-bit in groups of 64 for every module (8-bit per-layer tables are not read yet)
    const q = root.get("quantization") orelse return error.UnsupportedGemmaQuantization;
    if (q != .object) return error.UnsupportedGemmaQuantization;
    var it = q.object.iterator(); // beside bits, group_size and mode, only modules named `false`: left in bf16
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (std.mem.eql(u8, k, "bits") or std.mem.eql(u8, k, "group_size") or std.mem.eql(u8, k, "mode")) continue;
        if (e.value_ptr.* == .object) { // an attention projection at 8 bits (an entry of the quantization table)
            const o = e.value_ptr.object;
            if (std.mem.indexOf(u8, k, ".self_attn.") == null or (int(o, "bits") orelse 0) != 8 or (int(o, "group_size") orelse 0) != 64 or !str(o.get("mode"), "affine")) return error.UnsupportedGemmaQuantization;
            continue;
        }
        if (e.value_ptr.* != .bool or e.value_ptr.bool) return error.UnsupportedGemmaQuantization;
    }
    if ((int(q.object, "bits") orelse 0) != 4 or (int(q.object, "group_size") orelse 0) != 64 or !str(q.object.get("mode"), "affine")) return error.UnsupportedGemmaQuantization;
}

test "the layer pattern and the K/V sources" {
    try std.testing.expect(isGlobal(5) and isGlobal(41) and !isGlobal(0) and !isGlobal(40));
    try std.testing.expectEqual(@as(usize, 22), kvSource(40));
    try std.testing.expectEqual(@as(usize, 23), kvSource(35));
    try std.testing.expectEqual(@as(usize, 17), kvSource(17));
    var globals: usize = 0;
    for (0..layers) |i| globals += @intFromBool(isGlobal(i));
    try std.testing.expectEqual(@as(usize, 7), globals);
}
