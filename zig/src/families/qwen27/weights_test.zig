//! Small checkpoint fixtures exercise the target builder, borrowed bytes and failure cleanup.
const std = @import("std");
const Config = @import("config.zig");
const ck = @import("checkpoint.zig");
const Formats = @import("affine.zig").Formats;
const weights = @import("weights.zig");
const gpa = std.testing.allocator;
const json = "{\"model_type\":\"qwen3_5\",\"hidden_size\":64,\"intermediate_size\":128,\"num_hidden_layers\":4,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":16,\"vocab_size\":96,\"linear_num_key_heads\":2,\"linear_num_value_heads\":4,\"linear_key_head_dim\":16,\"linear_value_head_dim\":16,\"linear_conv_kernel_dim\":4,\"eos_token_id\":[2,11],\"quantization\":{\"bits\":4,\"group_size\":32,\"layers.0.linear_attn.in_proj_a\":false}}";
const Fake = struct {
    tensors: std.StringHashMapUnmanaged(ck.Tensor) = .empty,
    prefix: []const u8,

    fn get(ptr: *anyopaque, name: []const u8) anyerror!ck.Tensor {
        const f: *@This() = @ptrCast(@alignCast(ptr));
        return f.tensors.get(name) orelse error.MissingTensor;
    }
    fn source(f: *Fake) ck.Source {
        return .{ .ptr = f, .getFn = get };
    }
    fn tensor(f: *Fake, name: []const u8, dtype: ck.DType, dims: []const usize) !void {
        var t = ck.Tensor{ .dtype = dtype, .rank = @intCast(dims.len), .shape = @splat(1), .bytes = undefined };
        @memcpy(t.shape[0..dims.len], dims);
        var bytes = dtype.size();
        for (dims) |d| bytes *= d;
        t.bytes = try gpa.alloc(u8, bytes);
        @memset(@constCast(t.bytes), 0x59);
        errdefer gpa.free(t.bytes);
        const key = try std.mem.concat(gpa, u8, &.{ f.prefix, name });
        errdefer gpa.free(key);
        try f.tensors.put(gpa, key, t);
    }
    fn linear(f: *Fake, name: []const u8, n: usize, k: usize, dense: bool) !void {
        var key: [160]u8 = undefined;
        try f.tensor(try std.fmt.bufPrint(&key, "{s}.weight", .{name}), if (dense) .f32 else .u32, &.{ n, if (dense) k else k / 8 });
        if (!dense) {
            try f.tensor(try std.fmt.bufPrint(&key, "{s}.scales", .{name}), .bf16, &.{ n, k / 32 });
            try f.tensor(try std.fmt.bufPrint(&key, "{s}.biases", .{name}), .f16, &.{ n, k / 32 });
        }
    }
    fn init(prefix: []const u8) !Fake {
        var f = Fake{ .prefix = prefix };
        errdefer f.deinit();
        try f.linear("model.embed_tokens", 96, 64, false);
        try f.linear("lm_head", 96, 64, false);
        try f.tensor("model.norm.weight", .bf16, &.{64});
        var key: [160]u8 = undefined;
        for (0..4) |i| {
            for ([_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |norm| try f.tensor(try std.fmt.bufPrint(&key, "model.layers.{d}.{s}.weight", .{ i, norm }), .bf16, &.{64});
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, j| try f.linear(try std.fmt.bufPrint(&key, "model.layers.{d}.mlp.{s}", .{ i, p }), if (j == 2) 64 else 128, if (j == 2) 128 else 64, false);
            if (i < 3) {
                for ([_][]const u8{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj" }, [_]usize{ 128, 64, 4, 4, 64 }) |p, n| try f.linear(try std.fmt.bufPrint(&key, "model.layers.{d}.linear_attn.{s}", .{ i, p }), n, 64, i == 0 and std.mem.eql(u8, p, "in_proj_a"));
                try f.tensor(try std.fmt.bufPrint(&key, "model.layers.{d}.linear_attn.conv1d.weight", .{i}), .f16, &.{ 128, 4, 1 });
                for ([_][]const u8{ "A_log", "dt_bias" }) |p| try f.tensor(try std.fmt.bufPrint(&key, "model.layers.{d}.linear_attn.{s}", .{ i, p }), .f32, &.{4});
                try f.tensor(try std.fmt.bufPrint(&key, "model.layers.{d}.linear_attn.norm.weight", .{i}), .bf16, &.{16});
            } else {
                for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" }, [_]usize{ 128, 32, 32, 64 }) |p, n| try f.linear(try std.fmt.bufPrint(&key, "model.layers.{d}.self_attn.{s}", .{ i, p }), n, 64, false);
                for ([_][]const u8{ "q_norm", "k_norm" }) |p| try f.tensor(try std.fmt.bufPrint(&key, "model.layers.{d}.self_attn.{s}.weight", .{ i, p }), .f16, &.{16});
            }
        }
        return f;
    }
    fn deinit(f: *Fake) void {
        var it = f.tensors.iterator();
        while (it.next()) |e| {
            gpa.free(e.key_ptr.*);
            gpa.free(e.value_ptr.bytes);
        }
        f.tensors.deinit(gpa);
    }
};

test "full native weight graph covers scheduled layers without dtype or byte conversion" {
    var src = try Fake.init("");
    defer src.deinit();
    var fmt = try Formats.init(gpa, json);
    defer fmt.deinit();
    const c = try Config.parse(gpa, json, null);
    var w = try weights.load(gpa, src.source(), &fmt, c);
    defer w.deinit();
    try std.testing.expectEqual(@as(usize, 4), w.layers.len);
    const g = w.layers[0].mixer.linear;
    try std.testing.expectEqual(ck.DType.f32, g.a.dense.dtype);
    try std.testing.expectEqual(ck.DType.f16, g.conv.dtype);
    try std.testing.expectEqual(ck.DType.bf16, g.qkv.quantized.scales.dtype);
    try std.testing.expectEqual(ck.DType.f16, g.qkv.quantized.biases.dtype);
    try std.testing.expect(g.a.dense.bytes.ptr == src.tensors.get("model.layers.0.linear_attn.in_proj_a.weight").?.bytes.ptr);
    try std.testing.expect(w.layers[3].mixer.attention.q.quantized.weight.bytes.ptr == src.tensors.get("model.layers.3.self_attn.q_proj.weight").?.bytes.ptr);
    var total: usize = 0;
    var it = src.tensors.valueIterator();
    while (it.next()) |t| total += t.bytes.len;
    try std.testing.expectEqual(total, try w.tensorBytes());
}

test "nested language namespace uses matching head and per-module dense override" {
    var src = try Fake.init("language_model.");
    defer src.deinit();
    var fmt = try Formats.init(gpa, json);
    defer fmt.deinit();
    var w = try weights.load(gpa, src.source(), &fmt, try Config.parse(gpa, json, null));
    defer w.deinit();
    try std.testing.expectEqualStrings("language_model.", w.prefix);
    try std.testing.expectEqual(ck.DType.f32, w.layers[0].mixer.linear.a.dense.dtype);
}

test "query projection includes every gate row and refuses a truncated matrix" {
    var src = try Fake.init("");
    defer src.deinit();
    src.tensors.getPtr("model.layers.3.self_attn.q_proj.weight").?.shape[0] = 64;
    var fmt = try Formats.init(gpa, json);
    defer fmt.deinit();
    try std.testing.expectError(error.ProjectionShape, weights.load(gpa, src.source(), &fmt, try Config.parse(gpa, json, null)));
}

test "conv singleton placement keeps bytes and wrong kernel width refuses" {
    var src = try Fake.init("");
    defer src.deinit();
    const conv = src.tensors.getPtr("model.layers.0.linear_attn.conv1d.weight").?;
    conv.shape = .{ 128, 1, 4, 1, 1 };
    var fmt = try Formats.init(gpa, json);
    defer fmt.deinit();
    const c = try Config.parse(gpa, json, null);
    var w = try weights.load(gpa, src.source(), &fmt, c);
    try std.testing.expect(w.layers[0].mixer.linear.conv.bytes.ptr == conv.bytes.ptr);
    w.deinit();
    conv.shape[2] = 3;
    try std.testing.expectError(error.ProjectionShape, weights.load(gpa, src.source(), &fmt, c));
}

test "normalization width and scalar dtype must match the target operation" {
    var src = try Fake.init("");
    defer src.deinit();
    const norm = src.tensors.getPtr("model.layers.1.linear_attn.norm.weight").?;
    norm.shape[0] = 64;
    var fmt = try Formats.init(gpa, json);
    defer fmt.deinit();
    const c = try Config.parse(gpa, json, null);
    try std.testing.expectError(error.ProjectionShape, weights.load(gpa, src.source(), &fmt, c));
    norm.shape[0] = 16;
    norm.dtype = .u16;
    try std.testing.expectError(error.ProjectionDType, weights.load(gpa, src.source(), &fmt, c));
}

test "missing late-layer tensor releases owned graph records without freeing borrowed bytes" {
    var src = try Fake.init("");
    defer src.deinit();
    const missing = src.tensors.fetchRemove("model.layers.3.self_attn.k_norm.weight").?;
    defer gpa.free(missing.key);
    defer gpa.free(missing.value.bytes);
    var fmt = try Formats.init(gpa, json);
    defer fmt.deinit();
    try std.testing.expectError(error.MissingTensor, weights.load(gpa, src.source(), &fmt, try Config.parse(gpa, json, null)));
}

test "two text namespaces are rejected before choosing an embedding arbitrarily" {
    var src = try Fake.init("");
    defer src.deinit();
    try src.tensor("language_model.model.embed_tokens.weight", .u32, &.{ 96, 8 });
    var fmt = try Formats.init(gpa, json);
    defer fmt.deinit();
    try std.testing.expectError(error.AmbiguousTextNamespace, weights.load(gpa, src.source(), &fmt, try Config.parse(gpa, json, null)));
}

test "wrong layer schedule cannot reuse an attention matrix as recurrent weights" {
    var src = try Fake.init("");
    defer src.deinit();
    var fmt = try Formats.init(gpa, json);
    defer fmt.deinit();
    var c = try Config.parse(gpa, json, null);
    c.kinds[3] = .linear;
    try std.testing.expectError(error.MissingTensor, weights.load(gpa, src.source(), &fmt, c));
}
