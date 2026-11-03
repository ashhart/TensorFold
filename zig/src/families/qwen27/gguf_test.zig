//! Complete small GGUFs check destination identities independently and compare every affine value with source decoding.
const std = @import("std");
const core = @import("core");
const g = core.gguf;
const adapter = @import("gguf.zig");
const f = @import("gguf_fixture.zig");
const t = std.testing;
const a = t.allocator;
fn code(words: []const u32, bits: u8, i: usize) u32 {
    var out: u32 = 0;
    for (0..bits) |j| {
        const at = i * bits + j;
        out |= ((words[at / 32] >> @intCast(at % 32)) & 1) << @intCast(j);
    }
    return out;
}
fn values(tensor: *const adapter.Tensor) !void {
    switch (tensor.data) {
        .dense => |bytes| try t.expect(bytes.ptr == tensor.source.bytes.ptr),
        .affine => |q| {
            const geo = core.gguf_repack_ref.geometry(tensor.source.kind);
            try t.expectEqual(geo.group, q.shape.format.group);
            try t.expectEqual(geo.bits, q.shape.format.bits);
            for (0..q.shape.elements) |i| {
                const block = tensor.source.bytes[i / geo.values * geo.bytes ..][0..geo.bytes];
                const original: f32 = @floatCast(try core.gguf_repack_ref.value(tensor.source.kind, block, i % geo.values));
                const value: f32 = @floatCast(@as(f64, q.scales[i / geo.group]) * @as(f64, @floatFromInt(code(q.words, geo.bits, i))) + @as(f64, q.biases[i / geo.group]));
                try t.expectEqual(@as(u32, @bitCast(original)), @as(u32, @bitCast(value)));
            }
        },
    }
}
fn record(open: *const adapter.Adapter, layer: ?usize, r: f.Record) !void {
    var src: [128]u8 = undefined;
    var dst: [128]u8 = undefined;
    const source = if (layer) |l| try std.fmt.bufPrint(&src, "blk.{d}.{s}", .{ l, r.source }) else r.source;
    const dest = if (layer) |l| try std.fmt.bufPrint(&dst, "model.layers.{d}.{s}", .{ l, r.native }) else r.native;
    const got = open.get(dest) orelse return error.MissingMappedTensor;
    try t.expectEqualStrings(source, got.source.name);
    if (std.mem.eql(u8, r.source, "ssm_conv1d.weight")) {
        try t.expectEqual(@as(u8, 3), got.name.rank);
        try t.expectEqualSlices(usize, &.{ 2304, 4, 1 }, &got.name.dims);
    } else {
        try t.expectEqual(r.dims.len, got.name.rank);
        for (r.dims, 0..) |d, i| try t.expectEqual(d, got.name.dims[r.dims.len - 1 - i]);
    }
    if (std.mem.eql(u8, r.source, "ssm_a")) try t.expectEqual(adapter.names.Role.negative_exp_decay, got.name.role);
    try values(got);
}
fn complete(open: *const adapter.Adapter) !void {
    for (f.globals) |r| try record(open, null, r);
    for (0..4) |layer| {
        for (f.common) |r| try record(open, layer, r);
        if (layer == 3) {
            for (f.attention) |r| try record(open, layer, r);
        } else {
            for (f.linear) |r| try record(open, layer, r);
        }
    }
}
test "Qwen27 GGUF: complete mixed graph matches JSON and every original dequantized value in all ten formats" {
    const want = try @import("config.zig").parse(a, f.json, null);
    for (core.gguf_repack_ref.kinds) |kind| {
        const bytes = try f.image(a, kind, "qwen35");
        defer a.free(bytes);
        const hash = std.hash.Wyhash.hash(0, bytes);
        var p: adapter.Problem = .{};
        var open = try adapter.Adapter.open(a, bytes, &p);
        defer open.deinit();
        try t.expectEqualDeep(want, open.config);
        try complete(&open);
        try t.expectEqual(hash, std.hash.Wyhash.hash(0, bytes));
        try t.expectEqual(@as(usize, 56), open.tensors.len);
        try t.expectEqual(null, open.get("model.layers.4.self_attn.q_proj.weight"));
    }
}
test "Qwen27 GGUF: dense native dtypes stay borrowed and qwen3 scoped metadata uses the same contract" {
    for ([_]g.TensorKind{ .f32, .f16, .bf16 }) |kind| {
        const bytes = try f.image(a, kind, "qwen3");
        defer a.free(bytes);
        var p: adapter.Problem = .{};
        var open = try adapter.Adapter.open(a, bytes, &p);
        defer open.deinit();
        try complete(&open);
        for (open.tensors) |tensor| try t.expect(tensor.data == .dense);
    }
}
fn entry(doc: *g.Document, key: []const u8) *g.Entry {
    for (doc.metadata) |*e| if (std.mem.eql(u8, e.key, key)) return e;
    unreachable;
}
fn bad(doc: *g.Document, key: []const u8, value: g.Value, named: []const u8) !void {
    const e = entry(doc, key);
    const saved = e.*;
    defer e.* = saved;
    e.value = value;
    var p: adapter.Problem = .{};
    try t.expectError(error.UnsupportedQwenGguf, adapter.metadata.parse(doc, &p));
    try t.expect(std.mem.indexOf(u8, p.message(), named) != null);
}
test "Qwen27 GGUF: unknown computation keys, malformed settings, EOS and disagreeing patterns refuse by name" {
    const bytes = try f.image(a, .q4_k, "qwen35");
    defer a.free(bytes);
    var doc = try g.Document.parse(a, bytes, .{});
    defer doc.deinit();
    try bad(&doc, "general.architecture", .{ .string = "qwen35moe" }, "qwen35moe");
    try bad(&doc, "qwen35.block_count", .{ .uint32 = 129 }, "block_count");
    try bad(&doc, "qwen35.attention.key_length", .{ .string = "256" }, "attention.key_length");
    try bad(&doc, "qwen35.attention.value_length", .{ .uint32 = 128 }, "value_length");
    try bad(&doc, "qwen35.ssm.inner_size", .{ .uint32 = 2049 }, "inner_size");
    try bad(&doc, "qwen35.rope.freq_base", .{ .float64 = std.math.nan(f64) }, "rope.freq_base");
    try bad(&doc, "tokenizer.ggml.eos_token_id", .{ .uint32 = 32 }, "tokenizer.ggml");
    try bad(&doc, "qwen35.attention.recurrent_layers", .{ .uint32 = 0 }, "recurrent_layers");
    try bad(&doc, "qwen35.full_attention_interval", .{ .uint32 = 2 }, "conflicts");
    const e = entry(&doc, "qwen35.full_attention_interval");
    const original = e.*;
    defer e.* = original;
    e.key = "qwen35.rope.scaling.factor";
    var p: adapter.Problem = .{};
    try t.expectError(error.UnsupportedQwenGguf, adapter.metadata.parse(&doc, &p));
    try t.expect(std.mem.indexOf(u8, p.message(), "rope.scaling.factor") != null);
}
test "Qwen27 GGUF: unsupported quantization, wrong shape and missing or extra identities refuse before repack" {
    for ([_]g.TensorKind{ .q8_1, .q8_k, .i8 }) |kind| {
        const bytes = try f.image(a, kind, "qwen35");
        defer a.free(bytes);
        var p: adapter.Problem = .{};
        try t.expectError(error.UnsupportedQwenGguf, adapter.Adapter.open(a, bytes, &p));
        try t.expect(std.mem.indexOf(u8, p.message(), @tagName(kind)) != null);
        try t.expect(std.mem.indexOf(u8, p.message(), "token_embd.weight") != null);
    }
    const bytes = try f.image(a, .q4_k, "qwen35");
    defer a.free(bytes);
    const at = std.mem.indexOf(u8, bytes, "ssm_alpha.weight").?;
    @memcpy(bytes[at..][0..16], "ssm_gamma.weight");
    var p: adapter.Problem = .{};
    try t.expectError(error.UnsupportedQwenGguf, adapter.Adapter.open(a, bytes, &p));
    try t.expect(std.mem.indexOf(u8, p.message(), "ssm_alpha.weight") != null);
}
fn allocation(a_: std.mem.Allocator, bytes: []const u8) !void {
    var p: adapter.Problem = .{};
    var open = try adapter.Adapter.open(a_, bytes, &p);
    defer open.deinit();
}
test "Qwen27 GGUF: every allocation failure releases the document index, names and all repacked arrays" {
    const bytes = try f.image(a, .q4_k, "qwen35");
    defer a.free(bytes);
    try t.checkAllAllocationFailures(a, allocation, .{bytes});
}

test "Qwen27 GGUF: unknown GGML type includes the original tensor identity and numeric type" {
    var b = core.gguf_test_fixture.Fixture.init(1, 0);
    b.tensor("blk.0.ffn_gate.weight", &.{ 256, 256 }, .q4_k, 0);
    b.set(u32, b.len - 12, 16);
    b.finish(32, 0);
    var p: adapter.Problem = .{};
    try t.expectError(error.UnsupportedQwenGguf, adapter.Adapter.open(a, b.slice(), &p));
    try t.expect(std.mem.indexOf(u8, p.message(), "blk.0.ffn_gate.weight") != null);
    try t.expect(std.mem.indexOf(u8, p.message(), "type 16") != null);
}
test "Qwen27 GGUF: transposed source rows and a full-attention gate of the wrong width refuse by tensor name" {
    const bytes = try f.image(a, .q4_k, "qwen35");
    defer a.free(bytes);
    const name = "blk.3.attn_q.weight";
    const at = std.mem.indexOf(u8, bytes, name).? + name.len + 4;
    std.mem.writeInt(u64, bytes[at..][0..8], 512, .little);
    std.mem.writeInt(u64, bytes[at + 8 ..][0..8], 256, .little);
    var p: adapter.Problem = .{};
    try t.expectError(error.UnsupportedQwenGguf, adapter.Adapter.open(a, bytes, &p));
    try t.expect(std.mem.indexOf(u8, p.message(), name) != null);
}
