//! A complete small Q8_0-only Qwen35 container, built from the documented GGUF layouts without a checkpoint.
const std = @import("std");
const gg = @import("../../core/gguf.zig");
const Writer = @import("../../core/gguf_test_fixture.zig").Fixture;
pub fn make(a: std.mem.Allocator) ![]u8 {
    return build(a, false);
}
pub fn makeTokenized(a: std.mem.Allocator) ![]u8 {
    return build(a, true);
}
fn build(a: std.mem.Allocator, tokenized: bool) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const Item = struct { name: []const u8, shape: []const u64, kind: gg.TensorKind, at: usize = 0 };
    var list: std.ArrayList(Item) = .empty;
    const text = @import("../../core/gguf_tokenizer_fixture.zig");
    const vocabulary: u64 = if (tokenized) text.count else 32;
    const embed_shape = [_]u64{ 64, vocabulary };
    for ([_]Item{
        .{ .name = "token_embd.weight", .shape = &embed_shape, .kind = .q8_0 },
        .{ .name = "output.weight", .shape = &embed_shape, .kind = .q8_0 },
        .{ .name = "output_norm.weight", .shape = &.{64}, .kind = .f32 },
    }) |t| try list.append(scratch, t);
    const common = [_]Item{
        .{ .name = "attn_norm.weight", .shape = &.{64}, .kind = .f32 },
        .{ .name = "post_attention_norm.weight", .shape = &.{64}, .kind = .f32 },
        .{ .name = "ffn_gate.weight", .shape = &.{ 64, 128 }, .kind = .q8_0 },
        .{ .name = "ffn_up.weight", .shape = &.{ 64, 128 }, .kind = .q8_0 },
        .{ .name = "ffn_down.weight", .shape = &.{ 128, 64 }, .kind = .q8_0 },
    };
    const linear = [_]Item{
        .{ .name = "attn_qkv.weight", .shape = &.{ 64, 192 }, .kind = .q8_0 },
        .{ .name = "attn_gate.weight", .shape = &.{ 64, 128 }, .kind = .q8_0 },
        .{ .name = "ssm_beta.weight", .shape = &.{ 64, 4 }, .kind = .q8_0 },
        .{ .name = "ssm_alpha.weight", .shape = &.{ 64, 4 }, .kind = .q8_0 },
        .{ .name = "ssm_out.weight", .shape = &.{ 128, 64 }, .kind = .q8_0 },
        .{ .name = "ssm_conv1d.weight", .shape = &.{ 4, 192 }, .kind = .f32 },
        .{ .name = "ssm_a", .shape = &.{4}, .kind = .f32 },
        .{ .name = "ssm_dt.bias", .shape = &.{4}, .kind = .f32 },
        .{ .name = "ssm_norm.weight", .shape = &.{32}, .kind = .f32 },
    };
    const attention = [_]Item{
        .{ .name = "attn_q.weight", .shape = &.{ 64, 128 }, .kind = .q8_0 },
        .{ .name = "attn_k.weight", .shape = &.{ 64, 32 }, .kind = .q8_0 },
        .{ .name = "attn_v.weight", .shape = &.{ 64, 32 }, .kind = .q8_0 },
        .{ .name = "attn_output.weight", .shape = &.{ 64, 64 }, .kind = .q8_0 },
        .{ .name = "attn_q_norm.weight", .shape = &.{32}, .kind = .f32 },
        .{ .name = "attn_k_norm.weight", .shape = &.{32}, .kind = .f32 },
    };
    for (0..4) |i| {
        for (common) |t| try list.append(scratch, .{ .name = try std.fmt.allocPrint(scratch, "blk.{d}.{s}", .{ i, t.name }), .shape = t.shape, .kind = t.kind });
        const extra: []const Item = if (i == 3) &attention else &linear;
        for (extra) |t| try list.append(scratch, .{ .name = try std.fmt.allocPrint(scratch, "blk.{d}.{s}", .{ i, t.name }), .shape = t.shape, .kind = t.kind });
    }
    var h = Writer.init(list.items.len, 0);
    var nk: u64 = 0;
    h.kv("general.architecture", .{ .string = "qwen35" });
    nk += 1;
    const ints = [_]struct { []const u8, u32 }{
        .{ "qwen35.embedding_length", 64 },       .{ "qwen35.feed_forward_length", 128 },   .{ "qwen35.block_count", 4 },
        .{ "qwen35.attention.head_count", 2 },    .{ "qwen35.attention.head_count_kv", 1 }, .{ "qwen35.attention.key_length", 32 },
        .{ "qwen35.attention.value_length", 32 }, .{ "qwen35.ssm.group_count", 2 },         .{ "qwen35.ssm.time_step_rank", 4 },
        .{ "qwen35.ssm.state_size", 16 },         .{ "qwen35.ssm.inner_size", 128 },        .{ "qwen35.ssm.conv_kernel", 4 },
        .{ "qwen35.context_length", 4096 },       .{ "qwen35.rope.dimension_count", 16 },   .{ "tokenizer.ggml.eos_token_id", 1 },
        .{ "general.sampling.top_k", 20 },
    };
    for (ints) |kv| {
        if (tokenized and std.mem.eql(u8, kv[0], "tokenizer.ggml.eos_token_id")) continue;
        h.kv(kv[0], .{ .uint32 = kv[1] });
        nk += 1;
    }
    h.kv("qwen35.attention.layer_norm_rms_epsilon", .{ .float32 = 1e-6 });
    nk += 1;
    h.kv("qwen35.rope.freq_base", .{ .float32 = 10000000 });
    nk += 1;
    h.kv("general.sampling.top_p", .{ .float32 = 0.95 });
    nk += 1;
    h.kv("general.sampling.temp", .{ .float32 = 1 });
    nk += 1;
    if (tokenized) {
        try text.metadata(&h, scratch, false, false);
        nk += text.metadata_count;
    } else {
        h.kv("tokenizer.ggml.model", .{ .string = "gpt2" });
        nk += 1;
        h.kv("tokenizer.ggml.pre", .{ .string = "qwen35" });
        nk += 1;
        h.kv("tokenizer.chat_template", .{ .string = "{{ messages[0].content }}" });
        nk += 1;
        h.array("tokenizer.ggml.tokens", .string, 32);
        nk += 1;
        for (0..32) |i| h.string(try std.fmt.allocPrint(scratch, "t{d}", .{i}));
    }
    h.array("qwen35.rope.dimension_sections", .int32, 4);
    nk += 1;
    for ([_]i32{ 4, 2, 2, 0 }) |v| h.int(i32, v);
    h.array("qwen35.attention.recurrent_layers", .boolean, 4);
    nk += 1;
    h.put(&.{ 1, 1, 1, 0 });
    h.set(u64, 16, nk);
    var total: usize = 0;
    for (list.items) |*t| {
        t.at = std.mem.alignForward(usize, total, 32);
        var count: usize = 1;
        for (t.shape) |d| count *= @intCast(d);
        const b = t.kind.block();
        total = t.at + count / b.values * b.bytes;
        h.tensor(t.name, t.shape, t.kind, t.at);
    }
    h.finish(32, 0);
    const bytes = try a.alloc(u8, h.len + total);
    @memset(bytes, 0);
    @memcpy(bytes[0..h.len], h.slice());
    for (list.items, 0..) |t, ti| {
        const start = h.len + t.at;
        var count: usize = 1;
        for (t.shape) |d| count *= @intCast(d);
        if (t.kind == .q8_0) {
            for (0..count / 32) |g| {
                const block = bytes[start + g * 34 ..][0..34];
                const scale: f16 = @floatCast(0.125 + @as(f32, @floatFromInt(g % 4)) * 0.0001220703125);
                std.mem.writeInt(u16, block[0..2], @bitCast(scale), .little);
                for (block[2..], 0..) |*v, j| v.* = @bitCast(@as(i8, @intCast(@as(i32, @intCast((g * 17 + j + ti) % 256)) - 128)));
            }
        } else for (0..count) |i| {
            const x: f32 = if (std.mem.endsWith(u8, t.name, "ssm_a")) -@exp(@as(f32, 2)) else if (std.mem.endsWith(u8, t.name, "ssm_norm.weight")) 0.75 else 1.25 + @as(f32, @floatFromInt(i % 4)) * 0.0078125;
            std.mem.writeInt(u32, bytes[start + i * 4 ..][0..4], @bitCast(x), .little);
        }
    }
    return bytes;
}
