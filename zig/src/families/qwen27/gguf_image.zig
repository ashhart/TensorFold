//! One layer at a time becomes the native checkpoint image; gpu_weights owns all Metal fusions and layouts.
const std = @import("std");
const gg = @import("../../core/gguf.zig");
const conf = @import("gguf_config.zig");
const names = @import("gguf_names.zig");
const repack = @import("gguf_repack.zig");
const Out = @import("../flashnext/pack_io.zig").Out;
const convert = @import("../../core/quant/convert.zig");
const tokenizer = @import("../../core/gguf_tokenizer.zig");
pub const Tokenizer = struct {
    identity: []const u8 = tokenizer.source,
    context: ?*anyopaque = null,
    write: *const fn (?*anyopaque, std.mem.Allocator, std.Io, *const gg.Document, []const u8) anyerror!void = tokenizer.write,
};
pub const Options = struct { repacker: repack.Repacker = .{}, tokenizer: Tokenizer = .{} };
pub fn write(a: std.mem.Allocator, io: std.Io, d: *const gg.Document, dir: []const u8, identity: []const u8, options: Options) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    var config = try conf.build(scratch, d);
    const c = try conf.parse(scratch, config);
    const plan = try names.plan(scratch, d, c);
    const generation = try sampling(scratch, d, c);
    // The independent tokenizer adapter must complete before any large repack or complete marker is written.
    try options.tokenizer.write(options.tokenizer.context, a, io, d, dir);
    var formats: std.json.Value = .{ .object = .empty };
    try conf.set(scratch, &formats, "bits", .{ .integer = 8 });
    try conf.set(scratch, &formats, "group_size", .{ .integer = 32 });
    var index: std.json.Value = .{ .object = .empty };
    for (0..c.layers + 1) |part| {
        var out = Out.init(a);
        defer out.deinit();
        const shard = try std.fmt.allocPrint(scratch, "gguf-{d:0>3}.safetensors", .{part});
        for (plan) |entry| {
            if (if (entry.layer) |layer| part != layer + 1 else part != 0) continue;
            const start = out.names.items.len;
            if (entry.transform == .affine) {
                const p = try options.repacker.run(options.repacker.context, a, entry.tensor);
                defer p.deinit(a);
                const f = try repack.format(entry.tensor.kind);
                if (p.n != entry.shape[0] or p.k != entry.shape[1] or p.bits != f.bits or p.group != f.group or
                    p.k % p.group != 0 or p.words.len != p.n * p.k * p.bits / 32 or
                    p.scales.len != p.n * (p.k / p.group) or p.biases.len != p.scales.len) return error.BadGgufRepackResult;
                const format = try std.json.Stringify.valueAlloc(scratch, .{ .bits = p.bits, .group_size = p.group }, .{});
                const value = try std.json.parseFromSliceLeaky(std.json.Value, scratch, format, .{});
                try formats.object.put(scratch, entry.name, value);
                try out.put(try std.fmt.allocPrint(scratch, "{s}.weight", .{entry.name}), "U32", &.{ p.n, p.k * p.bits / 32 }, try a.dupe(u32, p.words));
                try out.put(try std.fmt.allocPrint(scratch, "{s}.scales", .{entry.name}), "F32", &.{ p.n, p.k / p.group }, try a.dupe(f32, p.scales));
                try out.put(try std.fmt.allocPrint(scratch, "{s}.biases", .{entry.name}), "F32", &.{ p.n, p.k / p.group }, try a.dupe(f32, p.biases));
            } else {
                try out.put(entry.name, "BF16", entry.shape, try small(a, entry));
            }
            for (out.names.items[start..]) |name| try index.object.put(scratch, try scratch.dupe(u8, name), .{ .string = shard });
        }
        const path = try std.fs.path.join(scratch, &.{ dir, shard });
        try out.write(a, io, path, identity);
    }
    try conf.set(scratch, &config, "quantization", formats);
    try conf.set(scratch, &config, "tensorfold_gguf", .{ .string = identity });
    try jsonFile(a, io, dir, "config.json", config);
    try jsonFile(a, io, dir, "model.safetensors.index.json", @as(std.json.Value, .{ .object = blk: {
        var map: std.json.ObjectMap = .empty;
        try map.put(scratch, "weight_map", index);
        break :blk map;
    } }));
    try jsonFile(a, io, dir, "generation_config.json", generation);
}
pub fn sampling(a: std.mem.Allocator, d: *const gg.Document, c: conf.Config) !std.json.Value {
    var generation: std.json.Value = .{ .object = .empty };
    const fields = [_][2][]const u8{ .{ "top_k", "general.sampling.top_k" }, .{ "top_p", "general.sampling.top_p" }, .{ "temperature", "general.sampling.temp" } };
    for (fields) |pair| if (d.get(pair[1])) |v| {
        const value: std.json.Value = if (std.mem.eql(u8, pair[0], "top_k")) .{ .integer = std.math.cast(i64, try v.unsigned()) orelse return error.BadGgufConfig } else .{ .float = try v.float() };
        if (value == .float and !std.math.isFinite(value.float)) return error.BadGgufSampling;
        try generation.object.put(a, pair[0], value);
    };
    try generation.object.put(a, "eos_token_id", .{ .integer = c.eos[0] });
    return generation;
}
/// Norms use full multipliers here; the real-file gate must check that GGUF stores the +1 offset.
pub fn small(a: std.mem.Allocator, entry: names.Entry) ![]u16 {
    if (entry.tensor.kind != .f32 or entry.tensor.bytes.len % 4 != 0) return error.BadGgufTensor;
    const out = try a.alloc(u16, entry.tensor.bytes.len / 4);
    errdefer a.free(out);
    for (out, 0..) |*v, i| {
        var x: f32 = @bitCast(std.mem.readInt(u32, entry.tensor.bytes[i * 4 ..][0..4], .little));
        if (!std.math.isFinite(x)) return error.NonfiniteGgufTensor;
        if (entry.transform == .a_log) {
            if (x >= 0) return error.BadGgufSsmA;
            x = @log(-x);
        }
        v.* = convert.bf16(x);
        if (!std.math.isFinite(@as(f32, @bitCast(@as(u32, v.*) << 16)))) return error.Bf16GgufOverflow;
    }
    return out;
}
pub fn jsonFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(text);
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
}
