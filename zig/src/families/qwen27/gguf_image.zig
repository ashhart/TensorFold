//! One layer at a time becomes the native checkpoint image; gpu_weights owns all Metal fusions and layouts.
const std = @import("std");
const gg = @import("../../core/gguf.zig");
const conf = @import("gguf_config.zig");
const names = @import("gguf_names.zig");
const repack = @import("gguf_repack.zig");
const Out = @import("../flashnext/pack_io.zig").Out;
const convert = @import("../../core/quant/convert.zig");
const tokenizer = @import("../../core/gguf_tokenizer.zig");
const heads = @import("gguf_heads.zig");
pub const Tokenizer = struct {
    identity: []const u8 = tokenizer.source,
    context: ?*anyopaque = null,
    write: *const fn (?*anyopaque, std.mem.Allocator, std.Io, *const gg.Document, []const u8) anyerror!void = tokenizer.write,
};
pub const Problem = struct {
    bytes: [512]u8 = undefined,
    len: usize = 0,
    pub fn text(p: *const Problem) []const u8 {
        return p.bytes[0..p.len];
    }
    fn inexact(p: *Problem, e: names.Entry, i: usize, x: f32, reason: []const u8) void {
        const message = std.fmt.bufPrint(&p.bytes, "GGUF tensor {s}[{d}] ({s}): {s}; F32 value {e}", .{ e.tensor.name, i, e.name, reason, x }) catch unreachable;
        p.len = message.len;
    }
};
pub const Options = struct { repacker: repack.Repacker = .{}, tokenizer: Tokenizer = .{}, problem: ?*Problem = null };
pub fn write(a: std.mem.Allocator, io: std.Io, d: *const gg.Document, dir: []const u8, identity: []const u8, options: Options) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    var config = try conf.build(scratch, d);
    const c = try conf.parse(scratch, config);
    const plan = try names.plan(scratch, d, c);
    const generation = try sampling(scratch, d, c);
    for (plan) |entry| if (entry.transform != .affine) try validateSmall(entry, options.problem);
    // The tokenizer must complete before any large repack or complete marker is written.
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
                var p = try options.repacker.run(options.repacker.context, a, entry.tensor);
                defer p.deinit(a);
                const f = try repack.format(entry.tensor.kind);
                if (p.n != entry.shape[0] or p.k != entry.shape[1] or p.bits != f.bits or p.group != f.group or
                    p.k % p.group != 0 or p.words.len != p.n * p.k * p.bits / 32 or
                    p.scales.len != p.n * (p.k / p.group) or p.biases.len != p.scales.len) return error.BadGgufRepackResult;
                if (entry.heads) |order| try heads.affine(a, &p, order);
                const format = try std.json.Stringify.valueAlloc(scratch, .{ .bits = p.bits, .group_size = p.group }, .{});
                const value = try std.json.parseFromSliceLeaky(std.json.Value, scratch, format, .{});
                try formats.object.put(scratch, entry.name, value);
                try out.put(try std.fmt.allocPrint(scratch, "{s}.weight", .{entry.name}), "U32", &.{ p.n, p.k * p.bits / 32 }, try a.dupe(u32, p.words));
                try out.put(try std.fmt.allocPrint(scratch, "{s}.scales", .{entry.name}), "F32", &.{ p.n, p.k / p.group }, try a.dupe(f32, p.scales));
                try out.put(try std.fmt.allocPrint(scratch, "{s}.biases", .{entry.name}), "F32", &.{ p.n, p.k / p.group }, try a.dupe(f32, p.biases));
            } else if (entry.transform == .f32 or entry.transform == .ssm_a) {
                try out.put(entry.name, "F32", entry.shape, try rawF32(a, entry));
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
    var stops: std.json.Value = .{ .array = std.json.Array.init(a) };
    for (c.eos[0..c.eos_count]) |id| try stops.array.append(.{ .integer = id });
    try generation.object.put(a, "eos_token_id", stops);
    return generation;
}
/// Raw F32 roles stay exact; roles requiring BF16 refuse an inexact narrowing before repack or publication.
pub fn validateSmall(entry: names.Entry, problem: ?*Problem) !void {
    if (entry.tensor.kind != .f32 or entry.tensor.bytes.len % 4 != 0) return error.BadGgufTensor;
    if (entry.transform == .f32 or entry.transform == .ssm_a) {
        for (0..entry.tensor.bytes.len / 4) |i| {
            const x: f32 = @bitCast(std.mem.readInt(u32, entry.tensor.bytes[i * 4 ..][0..4], .little));
            if (!std.math.isFinite(x)) return error.NonfiniteGgufTensor;
            if (entry.transform == .ssm_a and x >= 0) return error.BadGgufSsmA;
        }
    } else for (0..entry.tensor.bytes.len / 4) |i| _ = try element(entry, i, problem);
}
fn element(entry: names.Entry, i: usize, problem: ?*Problem) !u16 {
    @setFloatMode(.strict);
    const x: f32 = @bitCast(std.mem.readInt(u32, entry.tensor.bytes[i * 4 ..][0..4], .little));
    if (!std.math.isFinite(x)) return error.NonfiniteGgufTensor;
    const value = convert.bf16Exact(x) catch return refusal(entry, i, x, problem, "not exactly representable as BF16");
    return value;
}
fn refusal(entry: names.Entry, i: usize, x: f32, problem: ?*Problem, reason: []const u8) anyerror {
    var local_problem: Problem = .{};
    const detail = problem orelse &local_problem;
    detail.inexact(entry, i, x, reason);
    if (problem == null) std.log.warn("{s}", .{detail.text()});
    return error.InexactGgufSmallTensor;
}
pub fn small(a: std.mem.Allocator, entry: names.Entry) ![]u16 {
    if (entry.tensor.kind != .f32 or entry.tensor.bytes.len % 4 != 0) return error.BadGgufTensor;
    const out = try a.alloc(u16, entry.tensor.bytes.len / 4);
    errdefer a.free(out);
    for (out, 0..) |*v, i| v.* = try element(entry, i, null);
    if (entry.heads) |order| {
        const restored = try heads.matrix(u16, a, out, entry.shape[0], out.len / entry.shape[0], order);
        a.free(out);
        return restored;
    }
    return out;
}
/// Original F32 words, permuted only as whole value heads. No coefficient conversion or inverse decay function.
pub fn rawF32(a: std.mem.Allocator, entry: names.Entry) ![]f32 {
    try validateSmall(entry, null);
    const values = try a.alloc(f32, entry.tensor.bytes.len / 4);
    errdefer a.free(values);
    for (values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, entry.tensor.bytes[i * 4 ..][0..4], .little));
    if (entry.heads) |order| {
        const restored = try heads.matrix(f32, a, values, entry.shape[0], values.len / entry.shape[0], order);
        a.free(values);
        return restored;
    }
    return values;
}
pub fn jsonFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(text);
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
}
