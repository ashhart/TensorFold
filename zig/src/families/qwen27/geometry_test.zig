//! Full-size config and safetensors headers without payload allocations, Metal calls or checkpoint weights.
const std = @import("std");
const cfg = @import("config.zig");
const geometry = @import("geometry.zig");
const projection = @import("projection.zig");
const st = @import("core").safetensors;
const a = std.testing.allocator;

const Case = struct { h: usize, m: usize, layers: usize, heads: usize, vh: usize, qkv: usize, zba: usize, q: usize, kv: usize };
const cases = [_]Case{
    .{ .h = 2048, .m = 6144, .layers = 24, .heads = 8, .vh = 16, .qkv = 6144, .zba = 2080, .q = 4096, .kv = 512 },
    .{ .h = 4096, .m = 12288, .layers = 32, .heads = 16, .vh = 32, .qkv = 8192, .zba = 4160, .q = 8192, .kv = 1024 },
    .{ .h = 5120, .m = 17408, .layers = 64, .heads = 24, .vh = 48, .qkv = 10240, .zba = 6240, .q = 12288, .kv = 1024 },
};

fn config(al: std.mem.Allocator, c: Case) ![]const u8 {
    const kinds = try al.alloc([]const u8, c.layers);
    for (kinds, 0..) |*k, i| k.* = if (i % 4 == 3) "full_attention" else "linear_attention";
    return std.json.Stringify.valueAlloc(al, .{
        .model_type = "qwen3_5",
        .tie_word_embeddings = false,
        .text_config = .{
            .model_type = "qwen3_5_text",
            .tie_word_embeddings = false,
            .hidden_size = c.h,
            .intermediate_size = c.m,
            .num_hidden_layers = c.layers,
            .num_attention_heads = c.heads,
            .num_key_value_heads = c.kv / 256,
            .head_dim = 256,
            .vocab_size = 248320,
            .linear_num_key_heads = 16,
            .linear_num_value_heads = c.vh,
            .linear_key_head_dim = 128,
            .linear_value_head_dim = 128,
            .linear_conv_kernel_dim = 4,
            .eos_token_id = 248044,
            .rms_norm_eps = 1e-6,
            .max_position_embeddings = 262144,
            .full_attention_interval = 4,
            .attention_bias = false,
            .attn_output_gate = true,
            .hidden_act = "silu",
            .mamba_ssm_dtype = "float32",
            .layer_types = kinds,
            .rope_parameters = .{ .rope_type = "default", .rope_theta = 10000000, .partial_rotary_factor = 0.25, .mrope_interleaved = true, .mrope_section = [3]usize{ 11, 11, 10 } },
        },
        .quantization = .{ .bits = 4, .group_size = 64, .mode = "affine" },
    }, .{});
}

const Headers = struct {
    al: std.mem.Allocator,
    text: std.ArrayList(u8) = .empty,
    end: usize = 0,
    fn tensor(h: *Headers, name: []const u8, dtype: []const u8, dims: []const usize) !void {
        const first = h.end;
        var len: usize = if (std.mem.eql(u8, dtype, "U32") or std.mem.eql(u8, dtype, "F32")) 4 else 2;
        for (dims) |d| len *= d;
        h.end += len;
        const shape = try std.json.Stringify.valueAlloc(h.al, dims, .{});
        try h.text.print(h.al, "{s}\"language_model.{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{ if (first == 0) "{" else ",", name, dtype, shape, first, h.end });
    }
    fn linear(h: *Headers, name: []const u8, n: usize, k: usize) !void {
        inline for (.{ "weight", "scales", "biases" }, 0..) |leaf, i| {
            try h.tensor(try std.fmt.allocPrint(h.al, "{s}.{s}", .{ name, leaf }), if (i == 0) "U32" else "BF16", &.{ n, if (i == 0) k / 8 else k / 64 });
        }
    }
    fn build(al: std.mem.Allocator, c: Case) !st.Header {
        var h: Headers = .{ .al = al };
        try h.linear("model.embed_tokens", 248320, c.h);
        try h.linear("lm_head", 248320, c.h);
        try h.tensor("model.norm.weight", "BF16", &.{c.h});
        for (0..c.layers) |i| {
            const p = try std.fmt.allocPrint(al, "model.layers.{d}.", .{i});
            for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |s| try h.tensor(try std.mem.concat(al, u8, &.{ p, s }), "BF16", &.{c.h});
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |s, j| try h.linear(try std.fmt.allocPrint(al, "{s}mlp.{s}", .{ p, s }), if (j == 2) c.h else c.m, if (j == 2) c.m else c.h);
            if (i % 4 != 3) {
                for ([_][]const u8{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj" }, [_]usize{ c.qkv, c.vh * 128, c.vh, c.vh, c.h }, 0..) |s, n, j| try h.linear(try std.fmt.allocPrint(al, "{s}linear_attn.{s}", .{ p, s }), n, if (j == 4) c.vh * 128 else c.h);
                try h.tensor(try std.mem.concat(al, u8, &.{ p, "linear_attn.conv1d.weight" }), "BF16", &.{ c.qkv, 4, 1 });
                for ([_][]const u8{ "A_log", "dt_bias" }) |s| try h.tensor(try std.fmt.allocPrint(al, "{s}linear_attn.{s}", .{ p, s }), "F32", &.{c.vh});
                try h.tensor(try std.mem.concat(al, u8, &.{ p, "linear_attn.norm.weight" }), "BF16", &.{128});
            } else {
                for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" }, [_]usize{ c.q, c.kv, c.kv, c.h }, 0..) |s, n, j| try h.linear(try std.fmt.allocPrint(al, "{s}self_attn.{s}", .{ p, s }), n, if (j == 3) c.q / 2 else c.h);
                for ([_][]const u8{ "q_norm", "k_norm" }) |s| try h.tensor(try std.fmt.allocPrint(al, "{s}self_attn.{s}.weight", .{ p, s }), "BF16", &.{256});
            }
        }
        try h.text.append(al, '}');
        return st.parseHeader(al, h.text.items, h.end);
    }
};

fn checkPacked(header: st.Header, name: []const u8, n: usize, k: usize) !void {
    inline for (.{ "weight", "scales", "biases" }, 0..) |leaf, i| {
        var buf: [200]u8 = undefined;
        const t = header.get(try std.fmt.bufPrint(&buf, "language_model.{s}.{s}", .{ name, leaf })).?;
        try std.testing.expectEqual(if (i == 0) st.DType.u32 else st.DType.bf16, t.dtype);
        try std.testing.expectEqual(@as(u8, 2), t.rank);
        try std.testing.expectEqualSlices(usize, &.{ n, if (i == 0) k / 8 else k / 64 }, t.shape[0..2]);
    }
}

test "dense geometry: 2B 9B and 27B configs and every layer's packed headers agree" {
    for ([_]Case{ cases[1], cases[2], cases[0] }) |c| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const al = arena.allocator();
        const raw = try config(al, c);
        const parsed = try cfg.parse(a, raw, null);
        try geometry.supported(parsed);
        if (c.h == 2048) _ = try @import("../qwen3_5/config.zig").parse(a, raw);
        const header = try Headers.build(al, c);
        try checkPacked(header, "model.embed_tokens", parsed.vocab, parsed.hidden);
        try checkPacked(header, "lm_head", parsed.vocab, parsed.hidden);
        for (0..parsed.layers) |i| {
            const p = try std.fmt.allocPrint(al, "model.layers.{d}.", .{i});
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |s, j| try checkPacked(header, try std.fmt.allocPrint(al, "{s}mlp.{s}", .{ p, s }), if (j == 2) parsed.hidden else parsed.intermediate, if (j == 2) parsed.intermediate else parsed.hidden);
            if (parsed.kind(i) == .linear) {
                for ([_][]const u8{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj" }, [_]usize{ parsed.gdnQkvDim(), parsed.gdnValueDim(), parsed.v_heads, parsed.v_heads, parsed.hidden }, 0..) |s, n, j| try checkPacked(header, try std.fmt.allocPrint(al, "{s}linear_attn.{s}", .{ p, s }), n, if (j == 4) parsed.gdnValueDim() else parsed.hidden);
            } else {
                for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" }, [_]usize{ parsed.qDim(), parsed.kvDim(), parsed.kvDim(), parsed.hidden }, 0..) |s, n, j| try checkPacked(header, try std.fmt.allocPrint(al, "{s}self_attn.{s}", .{ p, s }), n, if (j == 3) parsed.heads * parsed.head_dim else parsed.hidden);
            }
        }
        const shapes = geometry.projections(parsed);
        try std.testing.expectEqual(geometry.Shape{ .n = c.qkv, .k = c.h }, shapes[2]);
        try std.testing.expectEqual(geometry.Shape{ .n = c.zba, .k = c.h }, shapes[3]);
        try std.testing.expectEqual(geometry.Shape{ .n = c.q, .k = c.h }, shapes[5]);
        for (shapes) |p| _ = try projection.split(p.n, p.k);
    }
}

test "dense geometry: fixed head arithmetic and projection alignment still refuse before load" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const c = try cfg.parse(a, try config(arena.allocator(), cases[2]), null);
    inline for (.{ "dk", "dv", "head_dim", "rope_dims", "conv_kernel", "hidden", "intermediate", "vocab" }) |field| {
        var bad = c;
        @field(bad, field) += 1;
        try std.testing.expectError(error.UnqualifiedQwenShape, geometry.supported(bad));
    }
    var bad = c;
    bad.kinds[0] = .attention;
    try std.testing.expectError(error.UnqualifiedLayerSchedule, geometry.supported(bad));
}
