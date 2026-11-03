//! The observed GGUF key and tensor index pin the host plan without loading a checkpoint during tests.
const std = @import("std");
const core = @import("core");
const adapter = @import("gguf.zig");
const native = @import("config.zig");
const t = std.testing;
const index = @embedFile("testdata/gguf-qwen35-tensors.tsv");
const metadata = @embedFile("testdata/gguf-qwen35-metadata.tsv");
fn value(key: []const u8) ![]const u8 {
    var lines = std.mem.tokenizeScalar(u8, metadata, '\n');
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        if (!std.mem.eql(u8, cols.next().?, key)) continue;
        _ = cols.next().?;
        return cols.next().?;
    }
    return error.MissingObservedMetadata;
}
test "Qwen27 GGUF: all 851 observed tensor names and shapes match the complete hybrid plan" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try native.parse(arena.allocator(), @embedFile("testdata/gguf-qwen35-config.json"), null);
    const plan = try adapter.names.plan(arena.allocator(), c);
    try t.expectEqual(@as(usize, 851), plan.len);
    var seen: [851]bool = @splat(false);
    var lines = std.mem.tokenizeScalar(u8, index, '\n');
    var count: usize = 0;
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        const name = cols.next().?;
        _ = cols.next().?;
        const dims = cols.next().?;
        const pos = for (plan, 0..) |n, i| {
            if (std.mem.eql(u8, n.gguf, name)) break i;
        } else return error.MissingObservedTensor;
        try t.expect(!seen[pos]);
        seen[pos] = true;
        const n = plan[pos];
        const rank = if (n.role == .convolution) 2 else n.rank;
        var shape = std.mem.splitScalar(u8, dims, ',');
        for (0..rank) |i| {
            const actual = try std.fmt.parseInt(usize, shape.next().?, 10);
            const want = if (n.role == .convolution) n.dims[1 - i] else n.dims[rank - 1 - i];
            try t.expectEqual(want, actual);
        }
        try t.expectEqual(null, shape.next());
        count += 1;
    }
    try t.expectEqual(plan.len, count);
    for (seen) |found| try t.expect(found);
}
test "Qwen27 GGUF: all observed computation keys are admitted and config scalar identities remain pinned" {
    const c = try native.parse(t.allocator, @embedFile("testdata/gguf-qwen35-config.json"), null);
    try t.expectEqualStrings("qwen35", try value("general.architecture"));
    var keys: std.StringHashMapUnmanaged(void) = .empty;
    defer keys.deinit(t.allocator);
    var lines = std.mem.tokenizeScalar(u8, metadata, '\n');
    var scoped: usize = 0;
    while (lines.next()) |line| {
        const key = line[0..std.mem.indexOfScalar(u8, line, '\t').?];
        try t.expect(!keys.contains(key));
        try keys.put(t.allocator, key, {});
        if (!std.mem.startsWith(u8, key, "qwen35.")) continue;
        scoped += 1;
        var found = false;
        for (adapter.metadata.fields) |field| found = found or std.mem.eql(u8, field, key[7..]);
        try t.expect(found);
    }
    try t.expectEqual(@as(usize, 40), keys.count());
    try t.expectEqual(adapter.metadata.fields.len, scoped);
    inline for (.{ .{ "qwen35.block_count", c.layers }, .{ "qwen35.context_length", c.max_position }, .{ "qwen35.embedding_length", c.hidden }, .{ "qwen35.feed_forward_length", c.intermediate }, .{ "qwen35.attention.head_count", c.heads }, .{ "qwen35.attention.head_count_kv", c.kv_heads }, .{ "qwen35.attention.key_length", c.head_dim }, .{ "qwen35.attention.value_length", c.head_dim }, .{ "qwen35.rope.dimension_count", c.rope_dims }, .{ "qwen35.ssm.conv_kernel", c.conv_kernel }, .{ "qwen35.ssm.state_size", c.dk }, .{ "qwen35.ssm.group_count", c.k_heads }, .{ "qwen35.ssm.time_step_rank", c.v_heads }, .{ "qwen35.ssm.inner_size", c.gdnValueDim() }, .{ "tokenizer.ggml.eos_token_id", c.eos[0] } }) |pair| {
        try t.expectEqual(@as(usize, pair[1]), try std.fmt.parseInt(usize, try value(pair[0]), 10));
    }
    try t.expectEqual(c.rope_theta, try std.fmt.parseFloat(f64, try value("qwen35.rope.freq_base")));
    try t.expectEqual(c.eps, try std.fmt.parseFloat(f32, try value("qwen35.attention.layer_norm_rms_epsilon")));
    try t.expectEqualStrings("int32:4,11,11,10,0", try value("qwen35.rope.dimension_sections"));
    try t.expectEqualStrings("string:248320", try value("tokenizer.ggml.tokens"));
    try t.expectEqualSlices(usize, &.{ 11, 11, 10 }, &c.mrope_section);
    try t.expectEqual(@as(usize, 4), try std.fmt.parseInt(usize, try value("qwen35.full_attention_interval"), 10));
    const recurrent = try value("qwen35.attention.recurrent_layers");
    try t.expect(std.mem.startsWith(u8, recurrent, "boolean:64,"));
    var flags = std.mem.splitScalar(u8, recurrent[11..], ',');
    for (0..c.layers) |layer| {
        const flag = try std.fmt.parseInt(u8, flags.next().?, 10);
        try t.expectEqual(c.kind(layer), if (flag == 1) native.Kind.linear else native.Kind.attention);
    }
    try t.expectEqual(null, flags.next());
}
test "Qwen27 GGUF: unproven qwen3 architecture is refused by name" {
    const bytes = try @import("gguf_fixture.zig").image(t.allocator, .q4_k, "qwen3");
    defer t.allocator.free(bytes);
    var doc = try core.gguf.Document.parse(t.allocator, bytes, .{});
    defer doc.deinit();
    var problem: adapter.Problem = .{};
    try t.expectError(error.UnsupportedQwenGguf, adapter.metadata.parse(&doc, &problem));
    try t.expectEqualStrings("general.architecture: unsupported qwen3", problem.message());
}

fn roundedBf16(gain: f32) u16 {
    const bits: u32 = @bitCast(gain);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
test "Qwen27 GGUF: six observed norm classes are effective gains matching MLX after BF16 rounding" {
    const bytes = try @import("gguf_fixture.zig").image(t.allocator, .f32, "qwen35");
    defer t.allocator.free(bytes);
    var doc = try core.gguf.Document.parse(t.allocator, bytes, .{});
    defer doc.deinit();
    const samples = @embedFile("testdata/gguf-qwen35-norms.tsv");
    var lines = std.mem.tokenizeScalar(u8, samples, '\n');
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        const src = doc.getTensor(cols.next().?).?;
        _ = cols.next().?;
        const i = try std.fmt.parseInt(usize, cols.next().?, 10);
        const bits = try std.fmt.parseInt(u32, cols.next().?, 16);
        const offset = doc.data_offset + src.offset + i * 4;
        std.mem.writeInt(u32, bytes[offset..][0..4], bits, .little);
    }
    var problem: adapter.Problem = .{};
    var open = try adapter.Adapter.open(t.allocator, bytes, &problem);
    defer open.deinit();
    lines = std.mem.tokenizeScalar(u8, samples, '\n');
    var count: usize = 0;
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        const source = cols.next().?;
        const gain = open.get(cols.next().?).?;
        const i = try std.fmt.parseInt(usize, cols.next().?, 10);
        const bits = try std.fmt.parseInt(u32, cols.next().?, 16);
        const mlx = try std.fmt.parseInt(u16, cols.next().?, 16);
        try t.expectEqualStrings(source, gain.source.name);
        try t.expectEqualStrings("rms_gain", @tagName(gain.name.role));
        const dense = gain.data.dense;
        try t.expect(dense.ptr == gain.source.bytes.ptr);
        const got = std.mem.readInt(u32, dense[i * 4 ..][0..4], .little);
        try t.expectEqual(bits, got);
        const value_f32: f32 = @bitCast(got);
        try t.expectEqual(mlx, roundedBf16(value_f32));
        try t.expect(mlx != roundedBf16(value_f32 + 1));
        try t.expect(mlx != roundedBf16(value_f32 - 1));
        count += 1;
    }
    try t.expectEqual(@as(usize, 96), count);
}
