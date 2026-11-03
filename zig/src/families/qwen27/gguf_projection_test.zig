//! The production GGUF projection storage is compared with core's independent serialized-block decoder.
const std = @import("std");
const core = @import("core");
const adapter = @import("gguf.zig");
const projection = @import("gguf_projection.zig");
const fixture = @import("gguf_fixture.zig");
const t = std.testing;
fn fp(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
}
fn decoded(p: projection.Matrix, i: usize) f32 {
    const group: usize = @intCast(p.params.group);
    const c: f32 = @floatFromInt(if (p.params.format == .mlx4) (p.words[i / 2] >> @as(u3, @intCast(i % 2 * 4))) & 15 else p.words[i]);
    return @mulAdd(f32, fp(p.scales, i / group), c, fp(p.biases, i / group));
}
test "Qwen27 GGUF: exact projection matrices retain FP32 tables and lossless six-bit promotion" {
    for ([_]core.gguf.TensorKind{ .q4_k, .q6_k, .q8_0 }) |kind| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const bytes = try fixture.image(a, kind, "qwen35");
        var problem: adapter.Problem = .{};
        var open = try adapter.Adapter.open(a, bytes, &problem);
        defer open.deinit();
        var matrices: usize = 0;
        for (open.tensors) |tensor| {
            if (tensor.name.role != .matrix) continue;
            const got = try projection.prepare(a, open.config, tensor, &problem);
            const q = tensor.data.affine;
            try t.expect(got.scales.ptr == std.mem.sliceAsBytes(q.scales).ptr);
            try t.expect(got.biases.ptr == std.mem.sliceAsBytes(q.biases).ptr);
            try t.expectEqual(@import("core").weight_only.Scale.f32, got.params.scale_kind);
            const geo = core.gguf_repack_ref.geometry(kind);
            const n: usize = @intCast(got.params.n);
            const k: usize = @intCast(got.params.k);
            for (0..n) |row| {
                var want: f64 = 0;
                var actual: f64 = 0;
                for (0..k) |column| {
                    const i = row * k + column;
                    const block = tensor.source.bytes[i / geo.values * geo.bytes ..][0..geo.bytes];
                    const original: f32 = @floatCast(try core.gguf_repack_ref.value(kind, block, i % geo.values));
                    const value = decoded(got, i);
                    try t.expectEqual(@as(u32, @bitCast(original)), @as(u32, @bitCast(value)));
                    const x: f64 = @as(f64, @floatFromInt(@as(i32, @intCast(column % 23)) - 11)) / 32;
                    want += @as(f64, original) * x;
                    actual += @as(f64, value) * x;
                }
                try t.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(actual)));
            }
            matrices += 1;
        }
        try t.expectEqual(@as(usize, 33), matrices);
    }
}
test "Qwen27 GGUF: serving type admission is explicit and every unsupported source type is named" {
    const bytes = try fixture.image(t.allocator, .q4_k, "qwen35");
    defer t.allocator.free(bytes);
    var doc = try core.gguf.Document.parse(t.allocator, bytes, .{});
    defer doc.deinit();
    var p: adapter.Problem = .{};
    try t.expectError(error.UnsupportedQwenGguf, projection.admit(&doc, &p));
    try t.expectEqualStrings("general.file_type: missing", p.message());
    const serving = try fixture.servingImage(t.allocator, .q4_k, "qwen35", 15);
    defer t.allocator.free(serving);
    var positive = try core.gguf.Document.parse(t.allocator, serving, .{});
    defer positive.deinit();
    try projection.admit(&positive, &p);
    for ([_]core.gguf.TensorKind{ .f16, .bf16, .q4_0, .q4_1, .q5_0, .q5_1, .q8_1, .q2_k, .q3_k, .q5_k, .q8_k, .i8, .i16, .i32, .i64, .f64 }) |kind| {
        positive.tensors[0].kind = kind;
        try t.expectError(error.UnsupportedQwenGguf, projection.admit(&positive, &p));
        try t.expect(std.mem.indexOf(u8, p.message(), @tagName(kind)) != null);
    }
    positive.tensors[0].kind = .q5_k;
    try t.expectError(error.UnsupportedQwenGguf, projection.admit(&positive, &p));
    try t.expect(std.mem.indexOf(u8, p.message(), "q5_k") != null);
    for (positive.metadata) |*entry| if (std.mem.eql(u8, entry.key, "general.file_type")) {
        entry.value = .{ .uint32 = 12 };
    };
    try t.expectError(error.UnsupportedQwenGguf, projection.admit(&positive, &p));
    try t.expect(std.mem.startsWith(u8, p.message(), "general.file_type: unsupported 12"));
}

test {
    _ = projection.layout;
}
