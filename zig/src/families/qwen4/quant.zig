//! Weight-only format declarations. Activations and caches remain BF16/FP32 in every mode.
const std = @import("std");
const f = @import("fields.zig");
const keys = @import("core").quant.keys;
pub const Format = enum { bf16, mlx4, mlx8, fp8, nvfp4 };
pub const Recipe = struct { format: Format = .bf16, group: usize = 0 };
pub const Config = struct {
    block: f.Object = .empty,
    mode: enum { bf16, mlx, modelopt } = .bf16,
    default: Recipe = .{},

    pub fn validateUsed(c: Config, projections: std.StringHashMapUnmanaged(void), p: *f.Problem) !void {
        if (c.mode == .bf16) return;
        const block = if (c.mode == .modelopt) blk: {
            const v = c.block.get("quantized_layers") orelse return;
            break :blk try f.object(v, "quantized_layers", p);
        } else c.block;
        for (block.keys()) |name| {
            if (c.mode == .mlx and isMetadata(name)) continue;
            if (!hasCanonical(projections, name)) return p.reject(name);
        }
    }

    pub fn of(c: Config, name: []const u8, p: *f.Problem) !Recipe {
        if (c.mode == .bf16) return .{};
        const wanted = keys.canonical(name);
        const block = if (c.mode == .modelopt and c.block.get("quantized_layers") != null)
            try f.object(c.block.get("quantized_layers").?, "quantized_layers", p)
        else
            c.block;
        var it = block.iterator();
        var found: ?Recipe = null;
        while (it.next()) |e| {
            if (!std.mem.eql(u8, keys.canonical(e.key_ptr.*), wanted)) continue;
            const recipe = if (c.mode == .mlx) blk: {
                const v = e.value_ptr.*;
                if (v == .bool) break :blk if (v.bool) c.default else Recipe{};
                const o = try f.object(v, e.key_ptr.*, p);
                break :blk if (o.count() == 0) Recipe{} else try mlx(o, 4, p);
            } else try perModule(try f.object(e.value_ptr.*, e.key_ptr.*, p), p);
            if (found) |prior| if (!std.meta.eql(prior, recipe)) return p.reject(e.key_ptr.*);
            found = recipe;
        }
        if (found) |recipe| return recipe;
        if (c.mode == .mlx or c.block.get("quantized_layers") != null) return c.default;
        for ([_][]const u8{ "ignore", "exclude_modules" }) |key| if (c.block.get(key)) |v| {
            if (v != .array) return p.reject(key);
            for (v.array.items) |item| {
                if (item != .string) return p.reject(key);
                const s = keys.canonical(item.string);
                if (std.mem.eql(u8, wanted, s) or (s.len > 0 and s[s.len - 1] == '*' and std.mem.startsWith(u8, wanted, s[0 .. s.len - 1]))) return .{};
            }
        };
        return c.default;
    }
};
fn mlx(o: f.Object, default_bits: ?usize, p: *f.Problem) !Recipe {
    const bits = if (o.get("bits") != null) try f.int(o, "bits", p) else default_bits orelse return p.reject("quantization.bits");
    if (bits != 4 and bits != 8) return p.reject("quantization.bits");
    const group = try f.natural(o, "group_size", 64, p);
    if (group != 32 and group != 64 and group != 128) return p.reject("quantization.group_size");
    if (o.get("mode")) |v| if (v != .null and !f.word(v, "") and !f.word(v, "affine")) return p.reject("quantization.mode");
    return .{ .format = if (bits == 4) .mlx4 else .mlx8, .group = group };
}
fn modelopt(o: f.Object, p: *f.Problem) !Recipe {
    if (o.get("weight_scale_encoding")) |v| if (!f.word(v, "raw")) return p.reject("weight_scale_encoding");

    const algo = o.get("quant_algo") orelse return p.reject("quant_algo");
    if (f.word(algo, "FP8")) {
        if (o.get("group_size")) |v| if (v != .null and (v != .integer or v.integer != 0)) return p.reject("group_size");
        return .{ .format = .fp8 };
    }
    if (f.word(algo, "NVFP4") or f.word(algo, "W4A16_NVFP4")) {
        const group = try f.natural(o, "group_size", 16, p);
        if (group != 16) return p.reject("quantization_config.group_size");
        return .{ .format = .nvfp4, .group = 16 };
    }
    return p.reject("quantization_config.quant_algo");
}
pub fn parse(root: f.Object, text: f.Object, p: *f.Problem) !Config {
    var block: ?f.Object = null;
    for ([_]f.Object{ root, text }) |source| {
        // When text and root are the same object, visit its declarations once.
        if (block != null and source.keys().ptr == root.keys().ptr) continue;
        for ([_][]const u8{ "quantization", "quantization_config" }) |key| if (source.get(key)) |v| {
            if (v == .null) continue;
            const o = try f.object(v, key, p);
            if (o.count() == 0) continue;
            if (block != null) return p.reject("quantization");
            block = o;
        };
    }
    const o = block orelse return .{};
    for ([_][]const u8{ "kv_cache_quant_algo", "activation_quantization", "activation_quant_algo" }) |key| if (o.get(key)) |v| {
        if (v != .null) return p.reject(key);
    };
    const method = o.get("quant_method");
    if (method == null or f.word(method.?, "mlx") or f.word(method.?, "affine")) {
        const default = try mlx(o, null, p);
        return .{ .block = o, .mode = .mlx, .default = default };
    }
    if (!f.word(method.?, "modelopt")) return p.reject("quantization_config.quant_method");
    try f.known(o, &.{ "quant_method", "quant_algo", "quantized_layers", "group_size", "weight_scale_encoding", "kv_cache_quant_algo", "activation_quantization", "activation_quant_algo", "ignore", "exclude_modules" }, p);

    const algo = o.get("quant_algo") orelse return p.reject("quantization_config.quant_algo");
    if (f.word(algo, "MIXED_PRECISION")) {
        const layers = try f.object(o.get("quantized_layers") orelse return p.reject("quantized_layers"), "quantized_layers", p);
        var it = layers.iterator();
        while (it.next()) |e| _ = try perModule(try f.object(e.value_ptr.*, e.key_ptr.*, p), p);
        return .{ .block = o, .mode = .modelopt };
    }
    return .{ .block = o, .mode = .modelopt, .default = try modelopt(o, p) };
}

fn perModule(o: f.Object, p: *f.Problem) !Recipe {
    try f.known(o, &.{ "quant_algo", "group_size", "weight_scale_encoding" }, p);
    return modelopt(o, p);
}

fn isMetadata(name: []const u8) bool {
    for ([_][]const u8{ "bits", "group_size", "mode", "quant_method" }) |metadata|
        if (std.mem.eql(u8, name, metadata)) return true;
    return false;
}
fn hasCanonical(projections: std.StringHashMapUnmanaged(void), name: []const u8) bool {
    var it = projections.keyIterator();
    while (it.next()) |known| if (std.mem.eql(u8, keys.canonical(name), keys.canonical(known.*))) return true;
    return false;
}
