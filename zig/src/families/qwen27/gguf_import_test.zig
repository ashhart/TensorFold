//! Host evidence for complete GGUF-to-native preparation, independent block values and immutable cache publication.
const std = @import("std");
const t = std.testing;
const gg = @import("../../core/gguf.zig");
const src = @import("gguf_source.zig");
const conf = @import("gguf_config.zig");
const image = @import("gguf_image.zig");
const names = @import("gguf_names.zig");
const repack = @import("gguf_repack.zig");
const fixture = @import("gguf_fixture.zig");
const core = @import("core");
const Spy = struct {
    calls: usize = 0,
    fail: bool = false,
    fn run(context: ?*anyopaque, a: std.mem.Allocator, x: gg.Tensor) !repack.Affine {
        const s: *@This() = @ptrCast(@alignCast(context.?));
        s.calls += 1;
        if (s.fail) return error.PlantedRepackFailure;
        return (repack.Repacker{}).run(null, a, x);
    }
    fn tokenizer(_: ?*anyopaque, a: std.mem.Allocator, io: std.Io, d: *const gg.Document, dir: []const u8) !void {
        try t.expectEqual(@as(usize, 32), (try d.tokenizer()).tokens.?.len);
        try image.jsonFile(a, io, dir, "tokenizer.json", .{ .test_fixture = "G3 tokenizer seam" });
        try image.jsonFile(a, io, dir, "tokenizer_config.json", .{ .chat_template = d.get("tokenizer.chat_template").?.string });
    }
    fn options(s: *Spy) src.Options {
        return .{ .repacker = .{ .context = s, .run = run }, .tokenizer = .{ .identity = "host-fixture-only-v1", .write = tokenizer } };
    }
};
test "GGUF Qwen: Q8_0 affine repack equals independent signed-block values and preserves row order" {
    const bytes = try fixture.make(t.allocator);
    defer t.allocator.free(bytes);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    const block = d.getTensor("blk.0.ffn_up.weight").?;
    const p = try (repack.Repacker{}).run(null, t.allocator, block);
    defer p.deinit(t.allocator);
    for (0..p.n) |row| for (0..p.k) |k| {
        const i = row * p.k + k;
        const b = block.bytes[(i / 32) * 34 ..][0..34];
        const scale: f64 = @as(f16, @bitCast(std.mem.readInt(u16, b[0..2], .little)));
        const q: i8 = @bitCast(b[2 + i % 32]);
        const code = (p.words[i / 4] >> @as(u5, @intCast((i % 4) * 8))) & 255;
        const actual = @as(f64, p.scales[i / 32]) * @as(f64, @floatFromInt(code)) + p.biases[i / 32];
        try t.expectEqual(scale * @as(f64, @floatFromInt(q)), actual);
    };
    var bad = block;
    bad.kind = .q3_k;
    try t.expectError(error.UnqualifiedGgufAffine, (repack.Repacker{}).run(null, t.allocator, bad));
    bad = block;
    bad.bytes = bad.bytes[0 .. bad.bytes.len - 1];
    try t.expectError(error.BadGgufProjection, (repack.Repacker{}).run(null, t.allocator, bad));
    try t.checkAllAllocationFailures(t.allocator, allocRepack, .{block});
}
fn allocRepack(a: std.mem.Allocator, x: gg.Tensor) !void {
    const p = try (repack.Repacker{}).run(null, a, x);
    defer p.deinit(a);
}
test "GGUF Qwen: complete native image uses real config, mapped triples, one norm offset and inverse ssm_a" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const a = t.allocator;
    const io = t.io;
    const bytes = try fixture.make(a);
    defer a.free(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.gguf", .data = bytes });
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const path = try std.fs.path.join(a, &.{ root, "model.gguf" });
    defer a.free(path);
    const cache = try std.fs.path.join(a, &.{ root, "images" });
    defer a.free(cache);
    if (@import("builtin").os.tag == .macos) try t.expectError(error.UnqualifiedQwenShape, src.local(a, io, root, path)) else try t.expectError(error.GgufMetalOnly, src.local(a, io, root, path));
    var spy = Spy{};
    const native = try src.prepare(a, io, path, cache, spy.options());
    defer a.free(native);
    try t.expectEqual(@as(usize, 33), spy.calls);
    const config_path = try std.fs.path.join(a, &.{ native, "config.json" });
    defer a.free(config_path);
    const config = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(1 << 20));
    defer a.free(config);
    const c = try @import("config.zig").parse(a, config, null);
    try t.expectEqual(@as(usize, 64), c.hidden);
    try t.expectEqual(@as(usize, 4), c.layers);
    try t.expectEqual(@as(u32, 1), c.eos[0]);
    try t.expectEqual(@import("config.zig").Kind.attention, c.kind(3));
    var formats = try @import("affine.zig").Formats.init(a, config);
    defer formats.deinit();
    const f = (try formats.resolve("language_model.model.layers.0.linear_attn.in_proj_a")).?;
    try t.expectEqual(@as(u8, 8), f.bits);
    try t.expectEqual(@as(usize, 32), f.group_size);
    var ck = try core.Checkpoint.openModel(a, io, native);
    defer ck.close();
    const norm = try ck.get("language_model.model.norm.weight");
    try t.expectEqual(core.safetensors.DType.bf16, norm.dtype);
    try t.expectEqual(@as(u16, 0x3fa0), std.mem.readInt(u16, norm.bytes[0..2], .little));
    const al = try ck.get("language_model.model.layers.0.linear_attn.A_log");
    try t.expectEqual(@as(u16, 0x3f00), std.mem.readInt(u16, al.bytes[0..2], .little));
    const gn = try ck.get("language_model.model.layers.0.linear_attn.norm.weight");
    try t.expectEqual(@as(u16, 0x3f40), std.mem.readInt(u16, gn.bytes[0..2], .little));
    const conv = try ck.get("language_model.model.layers.0.linear_attn.conv1d.weight");
    try t.expectEqual(@as(u8, 3), conv.rank);
    try t.expectEqual(@as(usize, 192), conv.shape[0]);
    try t.expectEqual(@as(usize, 4), conv.shape[1]);
    for ([_][]const u8{ "mlp.gate_proj", "mlp.up_proj", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a" }) |leaf| {
        const name = try std.fmt.allocPrint(a, "language_model.model.layers.0.{s}", .{leaf});
        defer a.free(name);
        const q = (try @import("checkpoint.zig").linear(a, @import("checkpoint.zig").Source.from(&ck), &formats, name, if (std.mem.eql(u8, leaf, "linear_attn.in_proj_b") or std.mem.eql(u8, leaf, "linear_attn.in_proj_a")) 4 else 128, 64)).quantized;
        try t.expectEqual(core.safetensors.DType.f32, q.scales.dtype);
    }
    var parsed = try gg.Document.parse(a, bytes, .{});
    defer parsed.deinit();
    var scratch_arena = std.heap.ArenaAllocator.init(a);
    defer scratch_arena.deinit();
    const plan = try names.plan(scratch_arena.allocator(), &parsed, c);
    for (plan) |entry| if (entry.transform == .affine) {
        const q = (try @import("checkpoint.zig").linear(a, @import("checkpoint.zig").Source.from(&ck), &formats, entry.name, entry.shape[0], entry.shape[1])).quantized;
        for (0..entry.shape[0]) |row| {
            var actual: f64 = 0;
            var expected: f64 = 0;
            for (0..entry.shape[1]) |k| {
                const i = row * entry.shape[1] + k;
                const block = entry.tensor.bytes[(i / 32) * 34 ..][0..34];
                const d: f64 = @as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little)));
                const signed: i8 = @bitCast(block[2 + i % 32]);
                const word = std.mem.readInt(u32, q.weight.bytes[(i / 4) * 4 ..][0..4], .little);
                const code = (word >> @as(u5, @intCast((i % 4) * 8))) & 255;
                const scale: f32 = @bitCast(std.mem.readInt(u32, q.scales.bytes[(i / 32) * 4 ..][0..4], .little));
                const bias: f32 = @bitCast(std.mem.readInt(u32, q.biases.bytes[(i / 32) * 4 ..][0..4], .little));
                const x = @as(f64, @floatFromInt(@as(i32, @intCast(k % 7)) - 3)) / 8;
                expected += x * d * @as(f64, @floatFromInt(signed));
                actual += x * (@as(f64, scale) * @as(f64, @floatFromInt(code)) + bias);
            }
            try t.expectEqual(expected, actual);
        }
    };
    var changed_builder = spy.options();
    changed_builder.repacker.identity = "changed-repacker";
    try t.expect(!std.mem.eql(u8, &src.key(&parsed, spy.options()), &src.key(&parsed, changed_builder)));
    const again = try src.prepare(a, io, path, cache, spy.options());
    defer a.free(again);
    try t.expectEqualStrings(native, again);
    try t.expectEqual(@as(usize, 33), spy.calls);
    // A value-byte edit, with identical metadata and file length, must invalidate the prepared image.
    bytes[bytes.len - 1] ^= 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "model.gguf", .data = bytes });
    const changed = try src.prepare(a, io, path, cache, spy.options());
    defer a.free(changed);
    try t.expect(!std.mem.eql(u8, native, changed));
    try t.expectEqual(@as(usize, 66), spy.calls);
}
test "GGUF Qwen: missing tokenizer and a failed repack never publish a usable image" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const a = t.allocator;
    const io = t.io;
    const bytes = try fixture.make(a);
    defer a.free(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = bytes });
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const path = try std.fs.path.join(a, &.{ root, "m.gguf" });
    defer a.free(path);
    const cache = try std.fs.path.join(a, &.{ root, "images" });
    defer a.free(cache);
    try t.expectError(error.GgufTokenizerPending, src.prepare(a, io, path, cache, .{}));
    var spy = Spy{ .fail = true };
    try t.expectError(error.PlantedRepackFailure, src.prepare(a, io, path, cache, spy.options()));
    spy.fail = false;
    const fixed = try src.prepare(a, io, path, cache, spy.options());
    defer a.free(fixed);
    try t.expectEqual(@as(usize, 34), spy.calls);
    var options = spy.options();
    options.tokenizer.identity = "host-fixture-only-v2";
    const fresh = try src.prepare(a, io, path, cache, options);
    defer a.free(fresh);
    try t.expect(!std.mem.eql(u8, fresh, fixed));
}
test "GGUF Qwen: source selection rejects ambiguous folders and keeps ordinary native checkpoints" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const a = t.allocator;
    const io = t.io;
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try t.expectEqual(@as(?[]u8, null), try src.find(a, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "one.gguf", .data = "stub" });
    const one = (try src.find(a, io, dir)).?;
    defer a.free(one);
    try t.expect(std.mem.endsWith(u8, one, "one.gguf"));
    try tmp.dir.writeFile(io, .{ .sub_path = "two.gguf", .data = "stub" });
    try t.expectError(error.MultipleGgufFiles, src.find(a, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "native.safetensors", .data = "stub" });
    try t.expectEqual(@as(?[]u8, null), try src.find(a, io, dir));
}
test "GGUF Qwen: metadata, tensor shape and ssm_a violations fail closed" {
    const a = t.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const bytes = try fixture.make(a);
    defer a.free(bytes);
    var d = try gg.Document.parse(a, bytes, .{});
    defer d.deinit();
    const value = try conf.build(scratch, &d);
    const c = try conf.parse(scratch, value);
    const plan = try names.plan(scratch, &d, c);
    try t.expectEqual(@as(usize, 56), plan.len);
    var aentry: names.Entry = undefined;
    for (plan) |e| if (e.transform == .a_log) {
        aentry = e;
        break;
    };
    var wrong: [16]u8 = @splat(0);
    aentry.tensor.bytes = &wrong;
    try t.expectError(error.BadGgufSsmA, image.small(a, aentry));
    std.mem.writeInt(u32, wrong[0..4], @bitCast(std.math.inf(f32)), .little);
    try t.expectError(error.NonfiniteGgufTensor, image.small(a, aentry));
    const top_p = d.keys.get("general.sampling.top_p").?;
    d.metadata[top_p].value = .{ .float32 = std.math.nan(f32) };
    try t.expectError(error.BadGgufSampling, image.sampling(scratch, &d, c));
    const i = d.names.get("blk.0.ffn_gate.weight").?;
    d.tensors[i].shape[1] += 1;
    try t.expectError(error.BadGgufTensor, names.plan(scratch, &d, c));
    d.tensors[i].shape[1] -= 1;
    const j = d.keys.get("qwen35.ssm.inner_size").?;
    d.metadata[j].value = .{ .uint32 = 129 };
    try t.expectError(error.BadGgufConfig, conf.build(scratch, &d));
}

test "GGUF Qwen: a corrupt completed image refuses instead of reusing or replacing its bytes" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const a = t.allocator;
    const io = t.io;
    const bytes = try fixture.make(a);
    defer a.free(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = bytes });
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const path = try std.fs.path.join(a, &.{ root, "m.gguf" });
    defer a.free(path);
    const cache = try std.fs.path.join(a, &.{ root, "images" });
    defer a.free(cache);
    var spy = Spy{};
    const native = try src.prepare(a, io, path, cache, spy.options());
    defer a.free(native);
    const tensor = try std.fs.path.join(a, &.{ native, "gguf-000.safetensors" });
    defer a.free(tensor);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tensor, .data = "planted corruption" });
    try t.expectError(error.CorruptGgufImage, src.prepare(a, io, path, cache, spy.options()));
    try t.expectEqual(@as(usize, 33), spy.calls);
}

test "GGUF Qwen: core G2 adapter preserves all eight admitted block types and rejects the two ungated forms" {
    const reference = @import("../../core/gguf_repack_ref.zig");
    for (reference.kinds) |kind| {
        const g = reference.geometry(kind);
        var storage: [420]u8 = undefined;
        const bytes = storage[0 .. 2 * g.bytes];
        for (bytes, 0..) |*v, i| v.* = @truncate(i * 37 + 11);
        for (0..2) |row| for (g.halves, 0..) |at, j| {
            std.mem.writeInt(u16, bytes[row * g.bytes + at ..][0..2], @as(u16, 0x3001) + @as(u16, @intCast(j + row)), .little);
        };
        const tensor = gg.Tensor{ .name = "projection", .kind = kind, .rank = 2, .shape = .{ g.values, 2, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = bytes };
        if (kind == .q2_k or kind == .q3_k) {
            try t.expectError(error.UnqualifiedGgufAffine, (repack.Repacker{}).run(null, t.allocator, tensor));
            continue;
        }
        const p = try (repack.Repacker{}).run(null, t.allocator, tensor);
        defer p.deinit(t.allocator);
        for (0..2) |row| for (0..g.values) |i| {
            const element = row * g.values + i;
            const bit = element * p.bits;
            const lo: u64 = p.words[bit / 32];
            const hi: u64 = if (bit / 32 + 1 < p.words.len) p.words[bit / 32 + 1] else 0;
            const code = ((lo | (hi << 32)) >> @as(u6, @intCast(bit % 32))) & ((@as(u64, 1) << @as(u6, @intCast(p.bits))) - 1);
            const group = element / p.group;
            const actual = @as(f64, p.scales[group]) * @as(f64, @floatFromInt(code)) + p.biases[group];
            const block = bytes[row * g.bytes ..][0..g.bytes];
            const coefficients = try reference.coefficients(kind, block, i / g.group);
            const expected = coefficients.s * @as(f64, @floatFromInt(reference.code(kind, block, i))) + coefficients.b;
            try t.expectEqual(expected, actual);
        };
    }
}
