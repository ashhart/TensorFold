const std = @import("std");
const mx = @import("mlx.zig");
const weights = @import("weights.zig");
const lanes = @import("lanes.zig");

const Module = struct { path: []const u8, block: i32, embedding: bool, dtype: []const u8 };
const Transform = struct {
    @"prism.hadamard.version": i32,
    @"prism.hadamard.block_size": i32,
    @"prism.hadamard.transform": []const u8,
    @"prism.hadamard.axis": []const u8,
    @"prism.hadamard.sign_mode": []const u8,
    @"prism.hadamard.gdn_v_grouped": bool,
    @"prism.hadamard.sign_widths": []const usize,
    @"prism.hadamard.sign_values": []const f32,
    @"prism.hadamard.inverse_weight_names": []const []const u8,

    fn validate(t: Transform) !void {
        if (t.@"prism.hadamard.version" != 1 or t.@"prism.hadamard.block_size" != 1024 or
            !std.mem.eql(u8, t.@"prism.hadamard.transform", "normalized-sylvester-walsh-hadamard") or
            !std.mem.eql(u8, t.@"prism.hadamard.axis", "input-last-dimension") or
            !std.mem.eql(u8, t.@"prism.hadamard.sign_mode", "explicit") or !t.@"prism.hadamard.gdn_v_grouped") return error.UnsupportedHadamardTransform;
        if (t.@"prism.hadamard.inverse_weight_names".len != 1 or !std.mem.eql(u8, t.@"prism.hadamard.inverse_weight_names"[0], "language_model.model.embed_tokens.weight")) return error.UnsupportedHadamardTransform;
        var count: usize = 0;
        for (t.@"prism.hadamard.sign_widths", 0..) |width, i| {
            if (width == 0 or width % 1024 != 0) return error.InvalidHadamardSigns;
            for (t.@"prism.hadamard.sign_widths"[0..i]) |other| if (other == width) return error.InvalidHadamardSigns;
            count = std.math.add(usize, count, width) catch return error.InvalidHadamardSigns;
        }
        if (count != t.@"prism.hadamard.sign_values".len) return error.InvalidHadamardSigns;
        for (t.@"prism.hadamard.sign_values") |value| if (value != -1 and value != 1) return error.InvalidHadamardSigns;
    }
    fn signs(t: Transform, width: usize) ![]const f32 {
        var offset: usize = 0;
        for (t.@"prism.hadamard.sign_widths") |w| {
            if (w == width) return t.@"prism.hadamard.sign_values"[offset..][0..w];
            offset += w;
        }
        return error.InvalidHadamardSigns;
    }
};

pub fn load(w: *weights.Weights, io: std.Io, dir: []const u8, config: std.json.Value) !void {
    var path: [4096]u8 = undefined;
    const bytes = try weights.readFile(io, try std.fmt.bufPrint(&path, "{s}/hadamard.json", .{dir}));
    defer mx.allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(Transform, mx.allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try parsed.value.validate();
    const modules = try std.json.parseFromValue([]const Module, mx.allocator, config.object.get("modules") orelse return error.MissingHadamardModules, .{});
    defer modules.deinit();
    var store = @import("checkpoint.zig").Store.init(128);
    defer store.deinit();
    try store.load(io, dir, "language_model.");
    try @import("schema.zig").validateConfig(.qwen, &store.arrays, false, config);
    var used = std.StringHashMap(void).init(mx.allocator);
    defer {
        var keys = used.keyIterator();
        while (keys.next()) |key| mx.allocator.free(key.*);
        used.deinit();
    }
    for (modules.value) |module| {
        if (module.block != 1024 or !std.mem.eql(u8, module.dtype, "float16")) return error.UnsupportedHadamardTransform;
        if (module.embedding != std.mem.eql(u8, module.path, "model.embed_tokens")) return error.InvalidHadamardEmbedding;
        var s = mx.Scope{};
        defer s.deinit();
        const q = try store.triple(module.path);
        const geometry = try (@import("quantization.zig").Spec{ .bits = 2, .group_size = 128 }).shape(mx.shape(q[0]), mx.shape(q[1]), mx.shape(q[2]));
        const values = try parsed.value.signs(@intCast(geometry.k));
        const signs = try s.cast(try store.field(module.path, "signs"), mx.f32t);
        if (mx.c.mlx_array_size(signs) != values.len) return error.InvalidHadamardSigns;
        try mx.eval(signs);
        if (!std.mem.eql(f32, values, mx.c.mlx_array_data_float32(signs)[0..values.len])) return error.InvalidHadamardSigns;
        if (module.embedding) {
            if (w.embedding_signs.ctx != null) return error.DuplicateWeight;
            w.embedding_signs = try mx.retain(signs);
            for ([_][]const u8{ "weight", "scales", "biases" }, q) |suffix, value| try w.putArray(try std.fmt.bufPrint(&path, "{s}.{s}", .{ module.path, suffix }), try mx.retain(value));
        } else {
            var scales = q[1];
            var biases = q[2];
            if (mx.tensor_units) {
                scales = try s.cast(try s.reshape(try s.stack(&.{ scales, scales }, -1), &.{ geometry.n, @divExact(geometry.k, 64) }), mx.bf16);
                biases = try s.cast(try s.reshape(try s.stack(&.{ biases, biases }, -1), &.{ geometry.n, @divExact(geometry.k, 64) }), mx.bf16);
            }
            var linear = try lanes.Linear.initFormat(&s, q[0], scales, biases, .{ .bits = 2, .group_size = if (mx.tensor_units) 64 else 128 });
            linear.signs = mx.retain(signs) catch |err| {
                linear.deinit();
                return err;
            };
            try w.putLinear(module.path, linear);
        }
        for ([_][]const u8{ "weight", "scales", "biases", "signs" }) |suffix| {
            const key = try std.fmt.allocPrint(mx.allocator, "{s}.{s}", .{ module.path, suffix });
            errdefer mx.allocator.free(key);
            if (used.contains(key)) return error.DuplicateWeight;
            try used.put(key, {});
        }
    }
    if (w.embedding_signs.ctx == null) return error.MissingHadamardEmbedding;
    var tensors = store.arrays.iterator();
    while (tensors.next()) |entry| {
        const key = entry.key_ptr.*;
        if (used.contains(key)) continue;
        var s = mx.Scope{};
        defer s.deinit();
        if (std.mem.endsWith(u8, key, ".in_proj_a.weight") or std.mem.endsWith(u8, key, ".in_proj_b.weight")) {
            var linear = try lanes.Linear.initFormat(&s, try s.cast(entry.value_ptr.*, mx.f32t), mx.empty, mx.empty, null);
            linear.prism_dense = true;
            try w.putLinear(key[0 .. key.len - 7], linear);
        } else {
            if (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases") or std.mem.endsWith(u8, key, ".signs")) return error.UnmatchedHadamardModule;
            try w.putArray(key, try mx.retain(try s.cast(entry.value_ptr.*, mx.bf16)));
        }
    }
}

test "Bonsai transform rejects incompatible transforms and malformed sign tables" {
    const signs: [1024]f32 = @splat(1);
    const base = Transform{
        .@"prism.hadamard.version" = 1,
        .@"prism.hadamard.block_size" = 1024,
        .@"prism.hadamard.transform" = "normalized-sylvester-walsh-hadamard",
        .@"prism.hadamard.axis" = "input-last-dimension",
        .@"prism.hadamard.sign_mode" = "explicit",
        .@"prism.hadamard.gdn_v_grouped" = true,
        .@"prism.hadamard.sign_widths" = &.{1024},
        .@"prism.hadamard.sign_values" = &signs,
        .@"prism.hadamard.inverse_weight_names" = &.{"language_model.model.embed_tokens.weight"},
    };
    try base.validate();
    try std.testing.expectEqualSlices(f32, &signs, try base.signs(1024));
    try std.testing.expectError(error.InvalidHadamardSigns, base.signs(2048));
    inline for (.{ .{ "prism.hadamard.version", 2 }, .{ "prism.hadamard.block_size", 512 }, .{ "prism.hadamard.transform", "other" }, .{ "prism.hadamard.axis", "output" }, .{ "prism.hadamard.sign_mode", "implicit" }, .{ "prism.hadamard.gdn_v_grouped", false } }) |field| {
        var config = base;
        @field(config, field[0]) = field[1];
        try std.testing.expectError(error.UnsupportedHadamardTransform, config.validate());
    }
    for ([_][]const usize{ &.{0}, &.{512}, &.{2048}, &.{ 1024, 1024 }, &.{ std.math.maxInt(usize) - 1023, 1024 } }) |widths| {
        var config = base;
        config.@"prism.hadamard.sign_widths" = widths;
        try std.testing.expectError(error.InvalidHadamardSigns, config.validate());
    }
    for ([_]f32{ 0, 0.5, std.math.nan(f32), std.math.inf(f32) }) |invalid| {
        var bad = signs;
        bad[511] = invalid;
        var config = base;
        config.@"prism.hadamard.sign_values" = &bad;
        try std.testing.expectError(error.InvalidHadamardSigns, config.validate());
    }
    for ([_][]const []const u8{ &.{}, &.{"lm_head.weight"}, &.{ "language_model.model.embed_tokens.weight", "lm_head.weight" } }) |names| {
        var config = base;
        config.@"prism.hadamard.inverse_weight_names" = names;
        try std.testing.expectError(error.UnsupportedHadamardTransform, config.validate());
    }
}
