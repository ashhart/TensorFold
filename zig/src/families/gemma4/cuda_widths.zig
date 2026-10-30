//! widths.zig for CUDA builds, line for line: MLX affine widths per module (core.quant.mlx imported by name).
const std = @import("std");
const mlx = @import("core").quant.mlx;
const cfg = @import("cuda_config.zig");

pub const Width = mlx.Width;

/// The bits the lane projection and core affine_mm both read; the glue's group sums are 64 wide.
pub const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 };
pub const group = 64;
/// The width the reg lane kernel and the fused one-row prologues take.
pub const fast: Width = .{ .bits = 4, .group = group };

/// A layer's linears, in Layer.mlx's order.
pub const Module = enum { q, k, v, o, gate, up, down };
pub const module_names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };

pub const Widths = struct {
    layers: [cfg.max_layers][module_names.len]Width = @splat(@splat(fast)),
    /// The tied embedding and head.
    embed: Width = fast,
    /// The 31B tower's (and a quantized 12B's) embed_vision projection.
    vision: Width = fast,
    /// The E-series' per-layer input linears and token table; null: stored as BF16 (the table says `false`).
    layer_gate: [cfg.max_layers]?Width = @splat(fast),
    layer_project: [cfg.max_layers]?Width = @splat(fast),
    model_project: ?Width = fast,
    layer_table: ?Width = fast,

    pub fn of(w: *const Widths, layer: usize, m: Module) Width {
        return w.layers[layer][@backingInt(m)];
    }

    /// Every linear at the fast width: the one-row windows may take the fused pipelines.
    pub fn allFast(w: *const Widths, layers: usize) bool {
        if (!std.meta.eql(w.embed, fast) or !std.meta.eql(w.model_project, fast) or !std.meta.eql(w.layer_table, fast)) return false;
        for (w.layer_gate[0..layers], w.layer_project[0..layers]) |g, p| if (!std.meta.eql(g, fast) or !std.meta.eql(p, fast)) return false;
        for (w.layers[0..layers]) |l| for (l) |x| if (!std.meta.eql(x, fast)) return false;
        return true;
    }
};

pub const Error = error{ UnsupportedGemmaQuantization, GemmaWidthNotRead, GemmaModuleUnquantized } || std.mem.Allocator.Error;

/// Whether the family's kernels read `w`.
pub fn readable(w: Width) bool {
    return w.group == group and std.mem.indexOfScalar(u8, &bit_widths, w.bits) != null;
}

/// Whether the key names one of the per-layer input linears or their token table, the modules a table may leave BF16.
fn perLayerInput(key: []const u8) bool {
    for ([_][]const u8{ "per_layer_input_gate", "per_layer_projection", "per_layer_model_projection", "embed_tokens_per_layer" }) |m| {
        if (std.mem.eql(u8, key, m) or (std.mem.endsWith(u8, key, m) and key[key.len - m.len - 1] == '.')) return true;
    }
    return false;
}

/// The width of a per-layer input module, null when the table marks it `false` (or `{}`) under any spelling: BF16.
fn optional(q: mlx.Config, quant: std.json.ObjectMap, module: []const u8) Error!?Width {
    var it = quant.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const v = e.value_ptr.*;
        const unquantized = (v == .bool and !v.bool) or (v == .object and v.object.count() == 0);
        if (!unquantized or !(std.mem.eql(u8, k, module) or (std.mem.endsWith(u8, k, module) and k[k.len - module.len - 1] == '.'))) continue;
        // another spelling of the module has a width
        if (q.modules.get(module)) |w| if (w != null) return error.UnsupportedGemmaQuantization;
        return null;
    }
    return try named(q, module);
}

/// Each module's width from `quant`; `global_values`: global layers have a v_proj; `layer_input`: BF16 linears allowed.
pub fn read(gpa: std.mem.Allocator, quant: std.json.ObjectMap, layers: usize, layer_input: bool, global_values: bool) Error!Widths {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var it = quant.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (std.mem.eql(u8, k, "bits") or std.mem.eql(u8, k, "group_size") or std.mem.eql(u8, k, "mode")) continue;
        if ((e.value_ptr.* != .object or e.value_ptr.object.count() == 0) and !(layer_input and perLayerInput(k))) {
            std.log.warn("gemma4: {s} is not a width table (false or {{}} marks a module unquantized); every linear is affine", .{k});
            return error.GemmaModuleUnquantized;
        }
    }
    const q = (mlx.detect(a, .{ .root = quant, .quantization = quant }) catch |err| return switch (err) {
        error.UnsupportedQuantization => error.UnsupportedGemmaQuantization, // two spellings of one module disagree
        else => |e| e,
    }) orelse return error.UnsupportedGemmaQuantization; // not MLX affine
    var w: Widths = .{};
    var key: [96]u8 = undefined;
    for (0..layers) |i| for (module_names, 0..) |name, mi| {
        if (cfg.global(i) and !global_values and mi == @backingInt(Module.v)) continue;
        w.layers[i][mi] = try named(q, std.fmt.bufPrint(&key, "layers.{d}.{s}", .{ i, name }) catch unreachable);
    };
    w.embed = try named(q, "embed_tokens");
    if (layer_input) {
        for (0..layers) |i| {
            w.layer_gate[i] = try optional(q, quant, std.fmt.bufPrint(&key, "layers.{d}.per_layer_input_gate", .{i}) catch unreachable);
            w.layer_project[i] = try optional(q, quant, std.fmt.bufPrint(&key, "layers.{d}.per_layer_projection", .{i}) catch unreachable);
        }
        w.model_project = try optional(q, quant, "per_layer_model_projection");
        w.layer_table = try optional(q, quant, "embed_tokens_per_layer");
    }
    w.vision = try named(q, "embed_vision.embedding_projection");
    return w;
}

fn named(q: mlx.Config, key: []const u8) Error!Width {
    const w = q.width(key) catch {
        std.log.warn("gemma4: {s}'s width is not MLX affine bits 2-6 or 8", .{key});
        return error.GemmaWidthNotRead;
    };
    if (!readable(w)) {
        std.log.warn("gemma4: {s} is {d}-bit in groups of {d}; the kernels read bits 2-6 and 8 in groups of 64", .{ key, w.bits, w.group });
        return error.GemmaWidthNotRead;
    }
    return w;
}

fn table(text: []const u8) !Widths {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
    defer parsed.deinit();
    return read(std.testing.allocator, parsed.value.object, 12, false, false);
}

fn tableE(text: []const u8) !Widths {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
    defer parsed.deinit();
    return read(std.testing.allocator, parsed.value.object, 12, true, true);
}

test "a uniform table gives every module the global width" {
    const w4 = try table("{\"bits\": 4, \"group_size\": 64, \"mode\": \"affine\"}");
    try std.testing.expect(w4.allFast(12));
    const w8 = try table("{\"bits\": 8, \"group_size\": 64}");
    try std.testing.expect(!w8.allFast(12));
    try std.testing.expectEqual(Width{ .bits = 8, .group = 64 }, w8.of(11, .down));
    try std.testing.expectEqual(Width{ .bits = 8, .group = 64 }, w8.embed);
}

test "per-module entries under any spelling reach their module only" {
    const w = try table(
        \\{"bits": 4, "group_size": 64, "mode": "affine",
        \\ "language_model.model.layers.1.self_attn.v_proj": {"bits": 6, "group_size": 64},
        \\ "model.layers.2.mlp.down_proj": {"bits": 6, "group_size": 64, "mode": "affine"},
        \\ "language_model.model.embed_tokens": {"bits": 8, "group_size": 64}}
    );
    try std.testing.expectEqual(@as(u8, 6), w.of(1, .v).bits);
    try std.testing.expectEqual(@as(u8, 4), w.of(1, .k).bits);
    try std.testing.expectEqual(@as(u8, 4), w.of(0, .v).bits);
    try std.testing.expectEqual(@as(u8, 6), w.of(2, .down).bits);
    try std.testing.expectEqual(@as(u8, 8), w.embed.bits);
    try std.testing.expect(!w.allFast(12));
}

test "widths no kernel reads are refused by name, never guessed" {
    const cases = .{
        .{ "{\"bits\": 7, \"group_size\": 64}", error.UnsupportedGemmaQuantization },
        .{ "{\"bits\": 4, \"group_size\": 32}", error.GemmaWidthNotRead },
        .{ "{\"bits\": 4, \"group_size\": 128}", error.GemmaWidthNotRead },
        .{ "{\"bits\": 4, \"group_size\": 64, \"mode\": \"mxfp4\"}", error.UnsupportedGemmaQuantization },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.mlp.up_proj\": {\"bits\": 7, \"group_size\": 64}}", error.GemmaWidthNotRead },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.mlp.up_proj\": {\"bits\": 4, \"group_size\": 64, \"mode\": \"nvfp4\"}}", error.GemmaWidthNotRead },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.mlp.up_proj\": false}", error.GemmaModuleUnquantized },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.mlp.up_proj\": {}}", error.GemmaModuleUnquantized },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.mlp.up_proj\": {\"bits\": 6, \"group_size\": 64}, \"model.layers.3.mlp.up_proj\": {\"bits\": 8, \"group_size\": 64}}", error.UnsupportedGemmaQuantization },
    };
    inline for (cases) |c| try std.testing.expectError(c[1], table(c[0]));
}

test "a global layer's missing v_proj takes no entry" {
    const w = try table("{\"bits\": 4, \"group_size\": 64, \"layers.5.self_attn.v_proj\": {\"bits\": 4, \"group_size\": 32}}");
    try std.testing.expectEqual(@as(u8, 4), w.of(5, .k).bits);
}

test "an E-series global layer's v_proj takes its entry" {
    const w = try tableE("{\"bits\": 4, \"group_size\": 64, \"layers.5.self_attn.v_proj\": {\"bits\": 8, \"group_size\": 64}}");
    try std.testing.expectEqual(@as(u8, 8), w.of(5, .v).bits);
    try std.testing.expectEqual(@as(u8, 4), w.of(4, .v).bits);
}

test "the per-layer input linears are BF16 when the table says false, or take their own width" {
    const w = try tableE(
        \\{"bits": 4, "group_size": 64, "mode": "affine",
        \\ "language_model.model.layers.2.per_layer_input_gate": false,
        \\ "model.layers.3.per_layer_projection": {},
        \\ "language_model.model.per_layer_model_projection": false,
        \\ "language_model.model.embed_tokens_per_layer": {"bits": 8, "group_size": 64},
        \\ "language_model.model.layers.4.per_layer_projection": {"bits": 8, "group_size": 64},
        \\ "language_model.model.layers.5.self_attn.q_proj": {"bits": 8, "group_size": 64}}
    );
    try std.testing.expectEqual(@as(?Width, null), w.layer_gate[2]);
    try std.testing.expectEqual(@as(?Width, fast), w.layer_gate[3]);
    try std.testing.expectEqual(@as(?Width, null), w.layer_project[3]);
    try std.testing.expectEqual(@as(?Width, .{ .bits = 8, .group = 64 }), w.layer_project[4]);
    try std.testing.expectEqual(@as(?Width, fast), w.layer_project[2]);
    try std.testing.expectEqual(@as(?Width, null), w.model_project);
    try std.testing.expectEqual(@as(?Width, .{ .bits = 8, .group = 64 }), w.layer_table);
    try std.testing.expectEqual(@as(u8, 8), w.of(5, .q).bits);
    try std.testing.expect(!w.allFast(12));
    try std.testing.expect((try tableE("{\"bits\": 4, \"group_size\": 64}")).allFast(12));
}

test "only the per-layer input modules of an E-series may be BF16, and one spelling cannot name both" {
    const cases = .{
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.per_layer_input_gate\": false}", error.GemmaModuleUnquantized, false },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.self_attn.q_proj\": false}", error.GemmaModuleUnquantized, true },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.mlp.up_proj\": {}}", error.GemmaModuleUnquantized, true },
        .{ "{\"bits\": 4, \"group_size\": 64, \"embed_tokens\": false}", error.GemmaModuleUnquantized, true },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.per_layer_projection_norm\": false}", error.GemmaModuleUnquantized, true },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.per_layer_projection\": false, \"model.layers.3.per_layer_projection\": {\"bits\": 8, \"group_size\": 64}}", error.UnsupportedGemmaQuantization, true },
        .{ "{\"bits\": 4, \"group_size\": 64, \"layers.3.per_layer_input_gate\": {\"bits\": 7, \"group_size\": 64}}", error.GemmaWidthNotRead, true },
    };
    inline for (cases) |c| try std.testing.expectError(c[1], if (c[2]) tableE(c[0]) else table(c[0]));
}
