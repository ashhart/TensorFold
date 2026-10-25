//! Actual mapped checkpoints exercise the same placement owner used by native device adapters.
const std = @import("std");
const cfg = @import("config.zig");
const Reader = @import("reader.zig").Reader;
const fixture = @import("reader_fixture.zig");
const Placed = @import("device_weights.zig").Placed;
const a = std.testing.allocator;
const Recording = struct {
    pub const Buffer = []u8;
    live: usize = 0,
    calls: usize = 0,
    fail_at: ?usize = null,
    drain_fail: bool = false,
    pub fn allocationBytes(n: usize) !u64 {
        return std.mem.alignForward(usize, n, 256);
    }
    pub fn upload(r: *Recording, bytes: []const u8) !Buffer {
        r.calls += 1;
        if (r.calls == r.fail_at) return error.PlantedUpload;
        const copy = try a.dupe(u8, bytes);
        r.live += 1;
        return copy;
    }
    pub fn release(r: *Recording, bytes: Buffer) void {
        a.free(bytes);
        r.live -= 1;
    }
    pub fn drain(r: *Recording) !void {
        if (r.drain_fail) return error.PlantedDrain;
    }
};
test "Qwen4 placement: budget before upload mapped tables and failed-drain retention" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var problem: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &problem);
    defer c.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(mem, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try fixture.write(tmp, mem, &c, false);
    var reader = try Reader.open(a, std.testing.io, path, &c, &problem);
    defer reader.close();
    var backend: Recording = .{};
    const Weights = Placed(Recording);
    try std.testing.expectError(error.Qwen4WeightsDoNotFit, Weights.open(a, &backend, &reader, &c, 0, &problem));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    var weights = try Weights.open(a, &backend, &reader, &c, std.math.maxInt(u64), &problem);
    try std.testing.expect(weights.bytes > 0);
    try std.testing.expect(weights.matrices.contains("model.layers.0.mlp.gate_proj"));
    try std.testing.expect(!weights.matrices.contains("model.layers.0.ple.ple_embedding.ngram_embedding"));
    try std.testing.expect(!weights.matrices.contains("model.embed_tokens"));
    const matrix = weights.matrices.get("model.layers.0.mlp.gate_proj").?;
    try std.testing.expectEqual(@as(c_int, 16), matrix.params.k);
    try std.testing.expectEqualSlices(u8, (try reader.get("model.layers.0.mlp.gate_proj.weight")).bytes, matrix.weight);
    var row: [16]u16 = undefined;
    try weights.lookup("model.embed_tokens", 3, &row, &problem);
    try std.testing.expectError(error.TokenOutOfRange, weights.lookup("model.embed_tokens", 32, &row, &problem));
    backend.drain_fail = true;
    const live = backend.live;
    try std.testing.expectError(error.PlantedDrain, weights.close());
    try std.testing.expectEqual(live, backend.live);
    backend.drain_fail = false;
    try weights.close();
    try std.testing.expectEqual(@as(usize, 0), backend.live);
    for (1..live + 1) |fail| {
        backend.calls = 0;
        backend.fail_at = fail;
        try std.testing.expectError(error.PlantedUpload, Weights.open(a, &backend, &reader, &c, std.math.maxInt(u64), &problem));
        try std.testing.expectEqual(@as(usize, 0), backend.live);
    }
}

test "Qwen4 placement: real MLX4 MLX8 FP8 and NVFP4 storage retains A16 maps and row lookup" {
    for ([_][]const u8{
        "\"quantization\":{\"bits\":4,\"group_size\":32}",
        "\"quantization\":{\"bits\":8,\"group_size\":32}",
        "\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"FP8\"}",
        "\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\",\"group_size\":16}",
    }) |recipe| {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const mem = arena.allocator();
        var doc = try std.json.parseFromSliceLeaky(std.json.Value, mem, @embedFile("fixtures/tiny-dense.json"), .{});
        for ([_][]const u8{ "hidden_size", "hc_lowrank", "intermediate_size", "num_attention_heads", "linear_num_value_heads", "ple_embed_dim" }, [_]i64{ 32, 32, 64, 4, 8, 128 }) |key, value| try doc.object.put(mem, key, .{ .integer = value });
        const text = try std.json.Stringify.valueAlloc(mem, doc, .{});
        const json = try std.fmt.allocPrint(mem, "{s},{s}}}", .{ text[0 .. text.len - 1], recipe });
        var problem: cfg.Problem = .{};
        var c = try cfg.parse(a, json, &problem);
        defer c.deinit();
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const path = try std.fmt.allocPrint(mem, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        try fixture.write(tmp, mem, &c, false);
        var reader = try Reader.open(a, std.testing.io, path, &c, &problem);
        defer reader.close();
        var backend: Recording = .{};
        var weights = try Placed(Recording).open(a, &backend, &reader, &c, std.math.maxInt(u64), &problem);
        defer weights.close() catch unreachable;
        var row: [32]u16 = undefined;
        try weights.lookup("model.embed_tokens", 7, &row, &problem);
        for (row) |element| try std.testing.expectEqual(@as(u16, 0x3f00), element);
        try std.testing.expectError(error.Qwen4LookupShape, weights.lookup("model.embed_tokens", 7, row[0..16], &problem));
        var model = try @import("model.zig").Model(Recording).init(a, &weights, &problem);
        defer model.deinit();
        try std.testing.expectEqual(@as(usize, 2), model.layers.len);
        try std.testing.expectEqual(c.intermediate, @as(usize, @intCast(model.layers[0].dense.gate.params.n)));
        try std.testing.expect(!model.layers[0].mixer.linear.norm.centered);
        try std.testing.expect(model.layers[0].attention.norm.centered);
        try std.testing.expect(model.layers[0].attention.norm.prepared);
        try std.testing.expect(model.layers[0].mixer.linear.norm.mid_bf16);
        try std.testing.expect(model.layers[0].mixer.linear.conv_bf16);
        try std.testing.expectEqual(@as(f32, 1), std.mem.bytesToValue(f32, model.layers[0].attention.norm.buffer[0..4]));
        try std.testing.expectEqual(@as(f32, 0), std.mem.bytesToValue(f32, model.layers[0].mixer.linear.norm.buffer[0..4]));
        try std.testing.expect(model.layers[1].mixer.indexed.qn.centered);
        try std.testing.expectEqualStrings("model.layers.0.ple.ple_embedding.ngram_embedding", model.layers[0].ple.?.table);
        try std.testing.expectEqual(@as(usize, c.ngram - 1), model.layers[0].ple.?.policy.ngram - 1);
        const policy = model.layers[0].ple.?.policy;
        try std.testing.expect(!@hasField(@TypeOf(policy), "a"));
        const history = try mem.alloc(u32, c.ngram - 1);
        @memset(history, c.ple_eos);
        const ids = try mem.alloc(usize, policy.sizes.len);
        try policy.step(history, 7, ids);
        try std.testing.expectEqual(@as(u32, 7), history[history.len - 1]);
        const wrong = weights.matrices.getPtr("model.layers.0.mlp.gate_proj").?;
        wrong.params.n -= 1;
        try std.testing.expectError(error.UnsupportedQwen4, @import("model.zig").Model(Recording).init(a, &weights, &problem));
        try std.testing.expectEqualStrings("model.layers.0.mlp.gate_proj", problem.name());
        wrong.params.n += 1;
        const projected = weights.matrices.get("model.layers.0.mlp.gate_proj").?;
        try std.testing.expectEqual(@as(c_int, 32), projected.params.k);
        try std.testing.expectEqual(@as(c_int, @intCast(@backingInt(c.quant.default.format))), @backingInt(projected.params.format));
        try std.testing.expectEqualSlices(u8, (try reader.get("model.layers.0.mlp.gate_proj.weight")).bytes, projected.weight);
    }
}

test "Qwen4 placement: original F32 convolution and norm storage survive exact preparation" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var problem: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &problem);
    defer c.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(mem, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try fixture.writeF32Vectors(tmp, mem, &c);
    var reader = try Reader.open(a, std.testing.io, path, &c, &problem);
    defer reader.close();
    var backend: Recording = .{};
    var weights = try Placed(Recording).open(a, &backend, &reader, &c, std.math.maxInt(u64), &problem);
    defer weights.close() catch unreachable;
    var model = try @import("model.zig").Model(Recording).init(a, &weights, &problem);
    defer model.deinit();
    const linear = model.layers[0].mixer.linear;
    try std.testing.expect(!linear.conv_bf16);
    try std.testing.expect(!linear.norm.mid_bf16);
    try std.testing.expectEqual(@as(u4, 7), @import("cuda_gdn.zig").weightFlags(linear.conv_bf16, linear.norm.mid_bf16));
    for ([_][]const u8{ "model.layers.0.linear_attn.conv1d.weight", "model.layers.0.linear_attn.norm.weight", "model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.dt_bias" }) |name| {
        const tensor = try reader.get(name);
        try std.testing.expectEqual(@TypeOf(tensor.dtype).f32, tensor.dtype);
        try std.testing.expectEqualSlices(u8, tensor.bytes, weights.vectors.get(name).?);
    }
    const hc = try reader.get("model.layers.0.attn_hyper_connection.hc_norm.weight");
    const raw = std.mem.bytesAsSlice(f32, hc.bytes);
    const prepared = std.mem.bytesAsSlice(f32, model.layers[0].attention.norm.buffer);
    for (raw, prepared) |v, actual| try std.testing.expectEqual(@as(f32, 1) + v, actual);
}
