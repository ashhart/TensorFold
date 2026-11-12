//! Independent serialized family fixture; dimensions stay small while DeltaNet and attention retain their head widths.
const std = @import("std");
const core = @import("core");
const g = core.gguf;
const F = core.gguf_test_fixture.Fixture;
pub const json =
    \\{"model_type":"qwen3_5","hidden_size":256,"intermediate_size":256,"num_hidden_layers":4,"vocab_size":32,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":256,"linear_num_key_heads":1,"linear_num_value_heads":16,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rms_norm_eps":0.000001,"partial_rotary_factor":0.25,"rope_theta":10000000,"full_attention_interval":4,"max_position_embeddings":262144,"eos_token_id":2}
;
pub const Record = struct { source: []const u8, native: []const u8, dims: []const u64, matrix: bool = true };
pub const globals = [_]Record{
    .{ .source = "token_embd.weight", .native = "model.embed_tokens.weight", .dims = &.{ 256, 32 } },
    .{ .source = "output.weight", .native = "lm_head.weight", .dims = &.{ 256, 32 } },
    .{ .source = "output_norm.weight", .native = "model.norm.weight", .dims = &.{256}, .matrix = false },
};
pub const common = [_]Record{
    .{ .source = "attn_norm.weight", .native = "input_layernorm.weight", .dims = &.{256}, .matrix = false },
    .{ .source = "post_attention_norm.weight", .native = "post_attention_layernorm.weight", .dims = &.{256}, .matrix = false },
    .{ .source = "ffn_gate.weight", .native = "mlp.gate_proj.weight", .dims = &.{ 256, 256 } },
    .{ .source = "ffn_up.weight", .native = "mlp.up_proj.weight", .dims = &.{ 256, 256 } },
    .{ .source = "ffn_down.weight", .native = "mlp.down_proj.weight", .dims = &.{ 256, 256 } },
};
pub const linear = [_]Record{
    .{ .source = "attn_qkv.weight", .native = "linear_attn.in_proj_qkv.weight", .dims = &.{ 256, 2304 } },
    .{ .source = "attn_gate.weight", .native = "linear_attn.in_proj_z.weight", .dims = &.{ 256, 2048 } },
    .{ .source = "ssm_alpha.weight", .native = "linear_attn.in_proj_a.weight", .dims = &.{ 256, 16 } },
    .{ .source = "ssm_beta.weight", .native = "linear_attn.in_proj_b.weight", .dims = &.{ 256, 16 } },
    .{ .source = "ssm_out.weight", .native = "linear_attn.out_proj.weight", .dims = &.{ 2048, 256 } },
    .{ .source = "ssm_conv1d.weight", .native = "linear_attn.conv1d.weight", .dims = &.{ 4, 2304 }, .matrix = false },
    .{ .source = "ssm_a", .native = "linear_attn.A_log", .dims = &.{16}, .matrix = false },
    .{ .source = "ssm_dt.bias", .native = "linear_attn.dt_bias", .dims = &.{16}, .matrix = false },
    .{ .source = "ssm_norm.weight", .native = "linear_attn.norm.weight", .dims = &.{128}, .matrix = false },
};
pub const attention = [_]Record{
    .{ .source = "attn_q.weight", .native = "self_attn.q_proj.weight", .dims = &.{ 256, 512 } },
    .{ .source = "attn_k.weight", .native = "self_attn.k_proj.weight", .dims = &.{ 256, 256 } },
    .{ .source = "attn_v.weight", .native = "self_attn.v_proj.weight", .dims = &.{ 256, 256 } },
    .{ .source = "attn_output.weight", .native = "self_attn.o_proj.weight", .dims = &.{ 256, 256 } },
    .{ .source = "attn_q_norm.weight", .native = "self_attn.q_norm.weight", .dims = &.{256}, .matrix = false },
    .{ .source = "attn_k_norm.weight", .native = "self_attn.k_norm.weight", .dims = &.{256}, .matrix = false },
};
const count = globals.len + 4 * common.len + 3 * linear.len + attention.len;
const Writer = struct {
    f: F = F.init(count, 0),
    metadata: u64 = 0,
    data: usize = 0,
    kinds: [count]g.TensorKind = undefined,
    lengths: [count]usize = undefined,
    offsets: [count]usize = undefined,
    used: usize = 0,
    fn kv(w: *Writer, key: []const u8, value: g.Value) void {
        w.metadata += 1;
        w.f.kv(key, value);
    }
    fn array(w: *Writer, key: []const u8, kind: g.MetadataKind, n: usize) void {
        w.metadata += 1;
        w.f.array(key, kind, n);
    }
    fn tensor(w: *Writer, layer: ?usize, r: Record, kind: g.TensorKind) !void {
        var name: [128]u8 = undefined;
        const source = if (layer) |l| try std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ l, r.source }) else r.source;
        const k = if (r.matrix) kind else g.TensorKind.f32;
        var elements: usize = 1;
        for (r.dims) |d| elements *= @intCast(d);
        const len = elements / k.block().values * k.block().bytes;
        w.data = std.mem.alignForward(usize, w.data, 32);
        w.offsets[w.used] = w.data;
        w.lengths[w.used] = len;
        w.kinds[w.used] = k;
        w.used += 1;
        w.f.tensor(source, r.dims, k, w.data);
        w.data += len;
    }
};
pub fn image(a: std.mem.Allocator, kind: g.TensorKind, arch: []const u8) ![]u8 {
    var w: Writer = .{};
    w.kv("general.architecture", .{ .string = arch });
    const ints = .{ .{ "block_count", 4 }, .{ "context_length", 262144 }, .{ "embedding_length", 256 }, .{ "feed_forward_length", 256 }, .{ "attention.head_count", 1 }, .{ "attention.head_count_kv", 1 }, .{ "attention.key_length", 256 }, .{ "attention.value_length", 256 }, .{ "rope.dimension_count", 64 }, .{ "ssm.conv_kernel", 4 }, .{ "ssm.state_size", 128 }, .{ "ssm.group_count", 1 }, .{ "ssm.time_step_rank", 16 }, .{ "ssm.inner_size", 2048 }, .{ "full_attention_interval", 4 } };
    inline for (ints) |entry| {
        var buf: [128]u8 = undefined;
        w.kv(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, entry[0] }), .{ .uint32 = entry[1] });
    }
    var key: [128]u8 = undefined;
    w.kv(try std.fmt.bufPrint(&key, "{s}.rope.freq_base", .{arch}), .{ .float32 = 10000000 });
    w.kv(try std.fmt.bufPrint(&key, "{s}.attention.layer_norm_rms_epsilon", .{arch}), .{ .float32 = 1e-6 });
    w.array(try std.fmt.bufPrint(&key, "{s}.rope.dimension_sections", .{arch}), .int32, 4);
    for ([_]i32{ 11, 11, 10, 0 }) |x| w.f.int(i32, x);
    w.array(try std.fmt.bufPrint(&key, "{s}.attention.recurrent_layers", .{arch}), .boolean, 4);
    for ([_]u8{ 1, 1, 1, 0 }) |x| w.f.int(u8, x);
    w.kv("tokenizer.ggml.eos_token_id", .{ .uint32 = 2 });
    w.array("tokenizer.ggml.tokens", .string, 32);
    for (0..32) |_| w.f.string("x");
    w.f.set(u64, 16, w.metadata);
    for (globals) |r| try w.tensor(null, r, kind);
    for (0..4) |l| {
        for (common) |r| try w.tensor(l, r, kind);
        if (l == 3) {
            for (attention) |r| try w.tensor(l, r, kind);
        } else {
            for (linear) |r| try w.tensor(l, r, kind);
        }
    }
    w.f.finish(32, 0);
    const out = try a.alloc(u8, w.f.len + w.data);
    @memcpy(out[0..w.f.len], w.f.slice());
    @memset(out[w.f.len..], 0);
    for (w.kinds, w.offsets, w.lengths, 0..) |k, offset, len, index| {
        const bytes = out[w.f.len + offset ..][0..len];
        if (k == .f32 or k == .f16 or k == .bf16) {
            if (k == .f32) for (0..len / 4) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(@as(f32, 1.0001)), .little) else for (0..len / 2) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], if (k == .f16) 0x3c01 else 0x3f81, .little);
        } else if (k == .q8_1 or k == .q8_k or k == .i8) @memset(bytes, 0x55) else {
            const geo = core.gguf_repack_ref.geometry(k);
            @memset(bytes, 0x55);
            for (0..len / geo.bytes) |block| for (geo.halves, 0..) |at, j| std.mem.writeInt(u16, bytes[block * geo.bytes + at ..][0..2], if (j == 0) 0x3c01 + @as(u16, @intCast(index)) else 0x2801, .little);
        }
    }
    return out;
}
