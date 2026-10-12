//! GLM-5.3-Flash's MLX affine width per module: config.json's quantization table, read through core/quant.
const std = @import("std");
const quant = @import("../../core/quant/quant.zig");

pub const Width = quant.mlx.Width;

/// The width every GLM kernel is generated for: 4 bits in groups of 64.
pub const uniform: Width = .{ .bits = 4, .group = 64 };

/// The checkpoint's widths: the core table with its override keys canonical (`layers.N.x`, `embed_tokens`, `lm_head`).
pub const Widths = struct {
    table: quant.mlx.Config,
    mtp_layer: u32,

    /// The width of the module a tensor name names, in either layout; an entry the table refuses fails by name.
    pub fn of(w: *const Widths, name: []const u8) !Width {
        var buf: [256]u8 = undefined;
        return w.table.width(try canonical(&buf, name, w.mtp_layer)) catch |err| {
            say("glm: config.json quantizes {s} in a format the engine does not read (MLX affine, 2 to 8 bits)", .{name});
            return err;
        };
    }

    /// Layer `i`'s routed expert `e`'s `proj`: its own entry, else its layer's stack's (`switch_mlp`), else the global.
    pub fn expert(w: *const Widths, i: usize, e: usize, proj: []const u8) !Width {
        var a: [128]u8 = undefined;
        var b: [128]u8 = undefined;
        const own = try std.fmt.bufPrint(&a, "layers.{d}.mlp.experts.{d}.{s}", .{ i, e, proj });
        const stack = try std.fmt.bufPrint(&b, "layers.{d}.mlp.switch_mlp.{s}", .{ i, proj });
        const x = w.table.overrides.get(own);
        const y = w.table.overrides.get(stack);
        if (x != null and y != null and !std.meta.eql(x.?, y.?)) return conflict(own, stack);
        return w.of(if (x != null) own else stack);
    }
};

/// A refusal's reason, logged as an error; tests check the error itself and keep the log quiet.
pub fn say(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.err(fmt, args);
}

fn conflict(a: []const u8, b: []const u8) error{ConflictingQuantization} {
    say("glm: config.json gives {s} and {s}, one module, different widths", .{ a, b });
    return error.ConflictingQuantization;
}

/// core/quant's module key, then GLM's own: `mtp.0.[block.]` as its layer, `self_attn.forget_gate.` as `self_attn.`.
pub fn canonical(buf: []u8, name: []const u8, mtp_layer: u32) error{NoSpaceLeft}![]const u8 {
    var short = @import("../../core/quant/keys.zig").canonical(name);
    var layer: []const u8 = "";
    var num: [12]u8 = undefined;
    if (std.mem.startsWith(u8, short, "mtp.0.")) {
        short = short["mtp.0.".len..];
        if (std.mem.startsWith(u8, short, "block.")) short = short["block.".len..];
        layer = try std.fmt.bufPrint(&num, "layers.{d}.", .{mtp_layer});
    }
    const gate = ".self_attn.forget_gate.";
    const at = std.mem.indexOf(u8, short, gate) orelse return std.fmt.bufPrint(buf, "{s}{s}", .{ layer, short });
    return std.fmt.bufPrint(buf, "{s}{s}.self_attn.{s}", .{ layer, short[0..at], short[at + gate.len ..] });
}

/// The quantization table of config.json's root, or of its text_config: `quantization`, else `quantization_config`.
fn block(root: std.json.ObjectMap) ?std.json.ObjectMap {
    const text: ?std.json.ObjectMap = if (root.get("text_config")) |t| (if (t == .object) t.object else null) else null;
    for ([_]?std.json.ObjectMap{ root, text }) |o| if (o) |obj| for ([_][]const u8{ "quantization", "quantization_config" }) |k| {
        if (obj.get(k)) |v| if (v == .object) return v.object;
    };
    return null;
}

/// config.json's widths: MLX affine only, overrides keyed by canonical module, aliases that disagree refused.
pub fn parse(a: std.mem.Allocator, json: []const u8, mtp_layer: u32) !Widths {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{ .allocate = .alloc_always });
    if (doc != .object) return error.BadConfig;
    const q = block(doc.object) orelse {
        say("glm: config.json has no quantization table; the engine reads MLX affine checkpoints", .{});
        return error.UnsupportedQuantization;
    };
    const stated = quant.detect(a, .{ .root = doc.object, .quantization = q }) catch |err| {
        say("glm: config.json's quantization is not MLX affine at 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128, one width a module", .{});
        return err;
    };
    const table = switch (stated) {
        .mlx => |t| t,
        else => return error.UnsupportedQuantization,
    };
    var out: Widths = .{ .table = .{ .global = table.global }, .mtp_layer = mtp_layer };
    var keys: std.StringHashMapUnmanaged([]const u8) = .empty; // canonical -> the key that set it
    var it = table.overrides.iterator();
    while (it.next()) |kv| {
        var buf: [256]u8 = undefined;
        const key = try a.dupe(u8, try canonical(&buf, kv.key_ptr.*, mtp_layer));
        const got = try out.table.overrides.getOrPut(a, key);
        if (got.found_existing and !std.meta.eql(got.value_ptr.*, kv.value_ptr.*)) return conflict(keys.get(key).?, kv.key_ptr.*);
        got.value_ptr.* = kv.value_ptr.*;
        try keys.put(a, key, kv.key_ptr.*);
    }
    return out;
}

const testing = std.testing;

test "override keys resolve by module whatever their prefix, the MTP block as its layer" {
    var buf: [256]u8 = undefined;
    const cases = [_][2][]const u8{
        .{ "language_model.model.layers.3.self_attn.q_b_proj", "layers.3.self_attn.q_b_proj" },
        .{ "model.language_model.layers.3.self_attn.q_b_proj", "layers.3.self_attn.q_b_proj" },
        .{ "model.layers.3.self_attn.q_b_proj", "layers.3.self_attn.q_b_proj" },
        .{ "language_model.lm_head", "lm_head" },
        .{ "lm_head", "lm_head" },
        .{ "language_model.model.embed_tokens", "embed_tokens" },
        .{ "language_model.mtp.0.block.self_attn.o_proj", "layers.45.self_attn.o_proj" },
        .{ "language_model.mtp.0.eh_proj", "layers.45.eh_proj" },
        .{ "model.layers.7.self_attn.forget_gate.f_b_proj", "layers.7.self_attn.f_b_proj" },
        .{ "vision_model.blocks.0.attn.qkv", "vision_model.blocks.0.attn.qkv" },
    };
    for (cases) |c| try testing.expectEqualStrings(c[1], try canonical(&buf, c[0], 45));
}

/// An oQ-style table: global 4/64, overrides at 8, 5 and 6 bits under both prefixes, the MTP block's, a stack's.
const mixed =
    \\{"text_config": {}, "quantization": {"group_size": 64, "bits": 4, "mode": "affine",
    \\ "language_model.lm_head": {"group_size": 64, "bits": 8},
    \\ "language_model.model.embed_tokens": {"group_size": 64, "bits": 8},
    \\ "language_model.model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 5},
    \\ "model.layers.0.self_attn.o_proj": {"group_size": 32, "bits": 6},
    \\ "language_model.model.layers.3.mlp.shared_experts.gate_proj": {"group_size": 128, "bits": 3},
    \\ "language_model.model.layers.3.mlp.switch_mlp.down_proj": {"group_size": 64, "bits": 2},
    \\ "language_model.mtp.0.block.self_attn.kv_b_proj": {"group_size": 64, "bits": 6},
    \\ "language_model.model.layers.5.mlp.shared_experts.up_proj": {"group_size": 64, "bits": 7}}}
;

test "each GLM tensor of a mixed oQ table resolves to its module's width, the rest to the global" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const w = try parse(arena.allocator(), mixed, 45);
    const cases = [_]struct { []const u8, Width }{
        .{ "lm_head", .{ .bits = 8, .group = 64 } },
        .{ "model.language_model.embed_tokens", .{ .bits = 8, .group = 64 } },
        .{ "model.language_model.layers.0.self_attn.q_proj", .{ .bits = 5, .group = 64 } },
        .{ "model.language_model.layers.0.self_attn.k_proj", uniform },
        .{ "model.language_model.layers.0.self_attn.o_proj", .{ .bits = 6, .group = 32 } },
        .{ "model.language_model.layers.1.self_attn.o_proj", uniform },
        .{ "model.language_model.layers.3.mlp.shared_experts.gate_proj", .{ .bits = 3, .group = 128 } },
        .{ "model.language_model.layers.45.self_attn.kv_b_proj", .{ .bits = 6, .group = 64 } },
        .{ "model.language_model.layers.44.self_attn.kv_b_proj", uniform },
    };
    for (cases) |c| try testing.expectEqual(c[1], try w.of(c[0]));
    try testing.expectEqual(Width{ .bits = 2, .group = 64 }, try w.expert(3, 17, "down_proj"));
    try testing.expectEqual(uniform, try w.expert(3, 17, "gate_proj"));
    try testing.expectEqual(uniform, try w.expert(4, 0, "down_proj"));
    try testing.expectError(error.UnsupportedQuantization, w.of("model.language_model.layers.5.mlp.shared_experts.up_proj"));
}

test "a uniform table gives every module 4/64; aliases that disagree, no table and other modes are refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const flat = try parse(a, "{\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\"}}", 45);
    try testing.expectEqual(uniform, try flat.of("model.language_model.layers.12.self_attn.q_proj"));
    try testing.expectEqual(uniform, try flat.expert(12, 287, "up_proj"));
    try testing.expectEqual(@as(u32, 0), flat.table.overrides.count());
    const agree = "{\"quantization_config\": {\"group_size\": 64, \"bits\": 4, \"model.layers.1.mlp.down_proj\": {\"group_size\": 64, \"bits\": 8}, \"language_model.model.layers.1.mlp.down_proj\": {\"group_size\": 64, \"bits\": 8}}}";
    try testing.expectEqual(Width{ .bits = 8, .group = 64 }, try (try parse(a, agree, 45)).of("model.language_model.layers.1.mlp.down_proj"));
    const clash = "{\"quantization\": {\"group_size\": 64, \"bits\": 4, \"model.layers.1.mlp.down_proj\": {\"group_size\": 64, \"bits\": 8}, \"language_model.model.layers.1.mlp.down_proj\": {\"group_size\": 64, \"bits\": 6}}}";
    try testing.expectError(error.UnsupportedQuantization, parse(a, clash, 45)); // core/quant refuses the aliases first
    const both = "{\"quantization\": {\"group_size\": 64, \"bits\": 4, \"model.layers.3.mlp.experts.2.up_proj\": {\"group_size\": 64, \"bits\": 8}, \"model.layers.3.mlp.switch_mlp.up_proj\": {\"group_size\": 64, \"bits\": 2}}}";
    const mtp = "{\"quantization\": {\"group_size\": 64, \"bits\": 4, \"language_model.mtp.0.block.self_attn.o_proj\": {\"group_size\": 64, \"bits\": 6}, \"model.layers.45.self_attn.o_proj\": {\"group_size\": 64, \"bits\": 8}}}";
    try testing.expectError(error.ConflictingQuantization, parse(a, mtp, 45)); // GLM's own alias: the MTP block is layer 45
    try testing.expectError(error.ConflictingQuantization, (try parse(a, both, 45)).expert(3, 2, "up_proj"));
    try testing.expectError(error.UnsupportedQuantization, parse(a, "{\"text_config\": {}}", 45));
    try testing.expectError(error.UnsupportedQuantization, parse(a, "{\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"mxfp4\"}}", 45));
}
