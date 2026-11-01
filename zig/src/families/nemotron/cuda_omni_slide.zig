//! Omni uses Nemotron's existing affine learner, with its wrapped language-model widths checked before loading.
const std = @import("std");
const living = @import("living");
const cfg = @import("config.zig");
const family = living.families.nemotron_h_omni;

pub fn refusal(a: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const al = arena.allocator();
    const path = try std.fs.path.join(al, &.{ dir, "config.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, al, .limited(64 << 20)) catch return null;
    var why: cfg.Why = .{};
    const c = cfg.parse(al, text, &why) catch return null;
    if (c.format != .mlx) return "--slide: nemotron_h_omni learns MLX affine checkpoints on CUDA, not ModelOpt FP4/FP8";
    if (c.latent > 0 or c.bolt) return "--slide: nemotron_h_omni latent experts do not learn on CUDA yet";
    if (c.mamba_head_dim != 64 or c.state != 128 or c.top_k > 16 or c.experts > 256)
        return "--slide: nemotron_h_omni needs Mamba head_dim 64, state 128, at most 16 picks and 256 experts on CUDA";
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, al, text, .{})).object;
    const m = (try living.widths.other(al, family, root)) orelse return null;
    return try std.fmt.allocPrint(a, "--slide: {s} learns {d}-bit groups of {d} on CUDA; {s} is {d}-bit groups of {d}", .{ family.model_type, family.quant.bits, family.quant.group, m.name, m.quant.bits, m.quant.group });
}

test "Omni learner: wrapped effective widths come from its family description" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    try std.testing.expectEqualStrings("llm_config", living.families.byModelType(family.model_type).?.keys.scope.?);
    for (0..6) |mode| {
        var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/config.json"), .{});
        var current: std.json.Value = .{ .object = .{} };
        try current.object.put(a, "model_type", .{ .string = family.model_type });
        const q = doc.object.get("quantization").?;
        _ = doc.object.swapRemove("quantization");
        _ = doc.object.swapRemove("quantization_config");
        try current.object.put(a, "llm_config", doc);
        try current.object.put(a, "quantization", q);
        var quant = &current.object.getPtr("quantization").?.object;
        try quant.put(a, "language_model.backbone.layers.1.mixer.shared_experts.up_proj", switch (mode) {
            1 => .{ .object = blk: {
                var o: std.json.ObjectMap = .{};
                try o.put(a, "bits", .{ .integer = 8 });
                break :blk o;
            } },
            else => .{ .bool = true },
        });
        if (mode == 2) try quant.put(a, "quant_method", .{ .string = "modelopt" });
        const llm = &current.object.getPtr("llm_config").?.object;
        if (mode == 3) {
            try llm.put(a, "moe_latent_size", .{ .integer = 1024 });
            try llm.put(a, "moe_intermediate_size", .{ .integer = 2048 });
            try llm.put(a, "moe_shared_expert_intermediate_size", .{ .integer = 4096 });
        }
        if (mode == 4) try llm.put(a, "num_experts_per_tok", .{ .integer = 17 });
        if (mode == 5) try llm.put(a, "num_experts_per_tok", .{ .integer = 6 });
        const text = try std.json.Stringify.valueAlloc(a, current, .{});
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = text });
        const got = try refusal(a, std.testing.io, dir);
        if (mode == 0 or mode == 5) try std.testing.expect(got == null) else {
            const words = [_][]const u8{ "", "shared_experts.up_proj is 8-bit", "ModelOpt", "latent experts", "16 picks" };
            try std.testing.expectStringStartsWith(got orelse "", "--slide: nemotron_h_omni");
            try std.testing.expect(std.mem.indexOf(u8, got.?, words[mode]) != null);
        }
    }
}
