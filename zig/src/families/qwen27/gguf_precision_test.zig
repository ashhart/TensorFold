//! A GGUF import either preserves a scalar exactly or refuses with the source tensor or metadata key named.
const std = @import("std");
const t = std.testing;
const gg = @import("../../core/gguf.zig");
const fixture = @import("gguf_fixture.zig");
const config = @import("gguf_config.zig");
const names = @import("gguf_names.zig");
const image = @import("gguf_image.zig");
const source = @import("gguf_source.zig");
fn plan(a: std.mem.Allocator, d: *const gg.Document) ![]names.Entry {
    return names.plan(a, d, try config.parse(a, try config.build(a, d)));
}
fn find(entries: []const names.Entry, name: []const u8) names.Entry {
    for (entries) |entry| if (std.mem.eql(u8, entry.tensor.name, name)) return entry;
    unreachable;
}
test "GGUF Qwen: remaining BF16 scalars refuse inexact values by source tensor and element" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try fixture.make(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    const entries = try plan(a, &d);
    for ([_][]const u8{ "blk.0.ssm_conv1d.weight", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight" }) |name| {
        var entry = find(entries, name);
        entry.heads = null; // This test isolates scalar exactness; full head layouts have their own regression.
        var raw: [4]u8 = undefined;
        entry.tensor.bytes = &raw;
        const exact: f32 = 1.25;
        std.mem.writeInt(u32, &raw, @bitCast(exact), .little);
        var problem: image.Problem = .{};
        try image.validateSmall(entry, &problem);
        const good = try image.small(t.allocator, entry);
        defer t.allocator.free(good);
        try t.expectEqualSlices(u16, &.{@as(u16, 0x3fa0)}, good);
        const inexact: f32 = 1.0009765625;
        std.mem.writeInt(u32, &raw, @bitCast(inexact), .little);
        try t.expectError(error.InexactGgufSmallTensor, image.validateSmall(entry, &problem));
        try t.expect(std.mem.indexOf(u8, problem.text(), name) != null);
        try t.expect(std.mem.indexOf(u8, problem.text(), "[0]") != null);
        try t.expect(std.mem.indexOf(u8, problem.text(), entry.name) != null);
    }
}
test "GGUF Qwen: small-tensor exactness preflight precedes repack and complete-image publication" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try fixture.makeTokenized(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    const norm = d.getTensor("blk.0.ssm_conv1d.weight").?;
    const offset = @intFromPtr(norm.bytes.ptr) - @intFromPtr(bytes.ptr);
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(@as(f32, 1.0009765625)), .little);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const path = try std.fs.path.join(a, &.{ root, "model.gguf" });
    const cache = try std.fs.path.join(a, &.{ root, "images" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.gguf", .data = bytes });
    const Counter = struct {
        calls: usize = 0,
        fn run(ctx: ?*anyopaque, alloc: std.mem.Allocator, tensor: gg.Tensor) !@import("gguf_repack.zig").Affine {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            self.calls += 1;
            return (@import("gguf_repack.zig").Repacker{}).run(null, alloc, tensor);
        }
    };
    var counter: Counter = .{};
    var problem: source.Problem = .{};
    const options: source.Options = .{ .problem = &problem, .repacker = .{ .context = &counter, .run = Counter.run } };
    if (source.prepare(t.allocator, t.io, path, cache, options)) |unexpected| {
        t.allocator.free(unexpected);
        return error.TestExpectedError;
    } else |err| try t.expectEqual(error.InexactGgufSmallTensor, err);
    try t.expectEqual(@as(usize, 0), counter.calls);
    try t.expect(std.mem.indexOf(u8, problem.text(), "blk.0.ssm_conv1d.weight[0]") != null);
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(@as(f32, 1.25)), .little);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.gguf", .data = bytes });
    const good = try source.prepare(t.allocator, t.io, path, cache, options);
    defer t.allocator.free(good);
    try t.expectEqual(@as(usize, 33), counter.calls);
}
fn buildConfig(a: std.mem.Allocator, d: *const gg.Document) !void {
    _ = try config.build(a, d);
}
fn buildTokenizer(a: std.mem.Allocator, d: *const gg.Document) !void {
    _ = try @import("../../core/gguf_tokenizer.zig").assets(a, d);
}
test "GGUF Qwen: every missing required config or tokenizer field names its exact metadata key" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try fixture.makeTokenized(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    inline for (.{ "general.architecture", "qwen35.embedding_length", "qwen35.feed_forward_length", "qwen35.block_count", "qwen35.attention.head_count", "qwen35.attention.head_count_kv", "qwen35.attention.key_length", "qwen35.attention.value_length", "qwen35.ssm.group_count", "qwen35.ssm.time_step_rank", "qwen35.ssm.state_size", "qwen35.ssm.inner_size", "qwen35.ssm.conv_kernel", "qwen35.context_length", "qwen35.rope.dimension_count", "qwen35.rope.dimension_sections", "qwen35.rope.freq_base", "qwen35.attention.layer_norm_rms_epsilon", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.tokens" }) |key| {
        const index = d.keys.get(key).?;
        const old_key = d.metadata[index].key;
        _ = d.keys.remove(key);
        d.metadata[index].key = "omitted";
        try t.expectError(@field(anyerror, "MissingGgufMetadata:" ++ key), buildConfig(a, &d));
        d.metadata[index].key = old_key;
        try d.keys.put(t.allocator, key, index);
    }
    inline for (.{ "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges" }) |key| {
        const index = d.keys.get(key).?;
        _ = d.keys.remove(key);
        try t.expectError(@field(anyerror, "MissingGgufMetadata:" ++ key), buildTokenizer(a, &d));
        try d.keys.put(t.allocator, key, index);
    }
}
