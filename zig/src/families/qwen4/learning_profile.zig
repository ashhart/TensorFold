//! Learning admission is narrower than serving and checks each actual projection before a mutable owner exists.
const std = @import("std");
const cfg = @import("config.zig");
const schema = @import("weights.zig");
const living = @import("living");

pub fn storedFamily(comptime ns: schema.Namespace, comptime wrapped: bool) living.Family {
    var f = if (wrapped) living.families.qwen4 else living.families.qwen4_text;
    f.names.layers = (comptime ns.prefix()) ++ "layers.";
    return f;
}

pub fn check(c: *const cfg.Config, ns: schema.Namespace, p: *cfg.Problem) !void {
    if (c.layers > 64 or c.layers < 2) return p.reject("--slide num_hidden_layers (2 through 64)");
    if (c.ple_count != 0) return p.reject("--slide PLE backward is not implemented");
    if (c.hidden % 64 != 0 or c.intermediate % 64 != 0) return p.reject("--slide group-64 MLP geometry");
    var buffer: [256]u8 = undefined;
    for (0..c.layers) |l| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        const name = try std.fmt.bufPrint(&buffer, "{s}layers.{d}.mlp.{s}", .{ ns.prefix(), l, proj });
        const recipe = try c.quant.of(name, p);
        if (recipe.format != .mlx4 or recipe.group != 64) return p.reject(name);
    };
}

pub fn layout(c: *const cfg.Config) living.Layout {
    var every: living.Layers = .empty;
    for (0..c.layers) |l| every.set(l);
    return .fromExperts(c.hidden, c.intermediate, c.layers, every);
}

pub fn checkpoint(c: *const cfg.Config, reader: *@import("reader.zig").Reader, p: *cfg.Problem) !void {
    try check(c, reader.namespace, p);
    const Sink = struct {
        r: *@import("reader.zig").Reader,
        c: *const cfg.Config,
        p: *cfg.Problem,
        pub fn take(s: *@This(), name: []const u8, want: schema.Want) !void {
            if (want.role != .matrix and !(want.role == .embedding and s.c.tie_embeddings)) return;
            const recipe = want.recipe;
            if (recipe.format == .bf16) return;
            if (recipe.format != .mlx4 or recipe.group != 64) return s.p.reject(name);
            var buffer: [320]u8 = undefined;
            for ([_][]const u8{ "scales", "biases" }) |leaf| {
                const tensor = try s.r.get(try std.fmt.bufPrint(&buffer, "{s}.{s}", .{ name, leaf }));
                if (tensor.dtype != .bf16) return s.p.reject(name);
            }
        }
    };
    var sink: Sink = .{ .r = reader, .c = c, .p = p };
    try schema.walk(c, reader.namespace, &sink, p);
}

test "Qwen4 learning: every stored namespace and config scope has one shared SwiGLU save description" {
    inline for (std.enums.values(schema.Namespace)) |ns| inline for (.{ false, true }) |wrapped| {
        const f = storedFamily(ns, wrapped);
        try f.check();
        try std.testing.expectEqualStrings((comptime ns.prefix()) ++ "layers.", f.names.layers);
        try std.testing.expectEqualStrings("mlp.", f.names.expert);
        try std.testing.expectEqual(@as(usize, 64), f.neurons);
        try std.testing.expectEqual(@as(usize, 64), f.quant.group);
        try std.testing.expectEqual(wrapped, f.keys.scope != null);
        try std.testing.expect(!f.neuron.gated);
        try std.testing.expectEqual(living.family.Activation.swiglu, f.neuron.act);
    };
    try std.testing.expect(living.families.byModelType("qwen4") != null);
    try std.testing.expect(living.families.byModelType("qwen4_text") != null);
    try std.testing.expect(living.families.byModelType("lfm2") == null);
}

test "Qwen4 learning: serving formats do not imply a writable format and each MLP override is checked" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/native-dense.json"), &p);
    defer c.deinit();
    try std.testing.expectError(error.UnsupportedQwen4, check(&c, .causal, &p));
    try std.testing.expectEqualStrings("--slide PLE backward is not implemented", p.name());
    c.ple_count = 0;
    try std.testing.expectError(error.UnsupportedQwen4, check(&c, .causal, &p));
    try std.testing.expectEqualStrings("model.layers.0.mlp.gate_proj", p.name());
    c.quant.mode = .mlx;
    c.quant.default = .{ .format = .mlx4, .group = 64 };
    try check(&c, .causal, &p);
    const lay = layout(&c);
    try std.testing.expect(lay.widen.isSet(0) and lay.widen.isSet(1));
    try std.testing.expect(lay.learn.isSet(0) and !lay.learn.isSet(1));
    const override = try std.json.parseFromSliceLeaky(std.json.Value, mem, "{\"group_size\":128,\"bits\":4}", .{});
    try c.quant.block.put(mem, "model.layers.1.mlp.down_proj", override);
    try std.testing.expectError(error.UnsupportedQwen4, check(&c, .causal, &p));
    try std.testing.expectEqualStrings("model.layers.1.mlp.down_proj", p.name());
    c.layers = 65;
    try std.testing.expectError(error.UnsupportedQwen4, check(&c, .causal, &p));
    try std.testing.expectEqualStrings("--slide num_hidden_layers (2 through 64)", p.name());
}
