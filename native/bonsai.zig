const std = @import("std");
const mx = @import("mlx.zig");
const weights = @import("weights.zig");
pub var form_override: ?Form = null;
pub var memory_limit: ?[]const u8 = null;

pub const Form = union(enum) {
    lanes,
    widened,
    @"packed",
    partial: usize,

    pub fn parse(text: []const u8) !Form {
        if (std.mem.eql(u8, text, "lanes")) return .lanes;
        if (std.mem.eql(u8, text, "widened")) return .widened;
        if (std.mem.eql(u8, text, "packed")) return .@"packed";
        if (std.mem.startsWith(u8, text, "widened:")) {
            const count = text[8..];
            if (count.len == 0) return error.InvalidBonsaiForm;
            for (count) |c| if (!std.ascii.isDigit(c)) return error.InvalidBonsaiForm;
            return .{ .partial = std.fmt.parseInt(usize, count, 10) catch return error.InvalidBonsaiForm };
        }
        return error.InvalidBonsaiForm;
    }

    pub fn name(form: Form, a: std.mem.Allocator) ![]u8 {
        return switch (form) {
            .partial => |n| std.fmt.allocPrint(a, "widened:{d}", .{n}),
            else => a.dupe(u8, @tagName(form)),
        };
    }

    pub fn module(form: Form, path: []const u8) Form {
        if (form != .partial) return form;
        return if (layerIndex(path)) |i| if (i < form.partial) .widened else .@"packed" else .@"packed";
    }
};

fn layerIndex(path: []const u8) ?usize {
    if (!std.mem.startsWith(u8, path, "model.layers.")) return null;
    const rest = path[13..];
    const end = std.mem.indexOfScalar(u8, rest, '.') orelse rest.len;
    if (end == 0) return null;
    for (rest[0..end]) |c| if (!std.ascii.isDigit(c)) return null;
    return std.fmt.parseInt(usize, rest[0..end], 10) catch null;
}

pub const Widening = struct {
    model: u64,
    layers: []u64,
    rest: u64,

    pub fn choose(plan: Widening, budget: u64) Form {
        var room = @as(i128, budget) - plan.model - 12 * @as(i128, 1 << 30);
        var total: i128 = plan.rest;
        for (plan.layers) |cost| total += cost;
        if (total <= room) return .widened;
        var count: usize = 0;
        while (count < plan.layers.len and plan.layers[count] <= room) : (count += 1) room -= plan.layers[count];
        return if (count > 0) .{ .partial = count } else .@"packed";
    }
};

pub fn widening(a: std.mem.Allocator, io: std.Io, dir: []const u8, modules: []const Module) !Widening {
    var path: [4096]u8 = undefined;
    var file = try @import("safetensors.zig").File.open(a, io, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{dir}));
    defer file.deinit();
    var plan = Widening{ .model = 0, .layers = &.{}, .rest = 0 };
    var tensors = file.header.tensors.iterator();
    while (tensors.next()) |entry| if (std.mem.startsWith(u8, entry.key_ptr.*, "language_model.")) {
        plan.model = try std.math.add(u64, plan.model, entry.value_ptr.len);
    };
    var layers: std.AutoHashMapUnmanaged(usize, u64) = .empty;
    defer layers.deinit(a);
    for (modules) |mod| {
        if (mod.embedding) continue;
        var added: u64 = 0;
        for ([_][]const u8{ "weight", "scales", "biases" }) |part| {
            const key = try std.fmt.bufPrint(&path, "language_model.{s}.{s}", .{ mod.path, part });
            const tensor = file.header.tensors.get(key) orelse return error.MissingWeight;
            added = try std.math.add(u64, added, tensor.len);
        }
        if (layerIndex(mod.path)) |index| {
            const entry = try layers.getOrPut(a, index);
            if (!entry.found_existing) entry.value_ptr.* = 0;
            entry.value_ptr.* = try std.math.add(u64, entry.value_ptr.*, added);
        } else plan.rest = try std.math.add(u64, plan.rest, added);
    }
    const indices = try a.alloc(usize, layers.count());
    defer a.free(indices);
    var keys = layers.keyIterator();
    for (indices) |*i| i.* = keys.next().?.*;
    std.mem.sort(usize, indices, {}, std.sort.asc(usize));
    plan.layers = try a.alloc(u64, indices.len);
    for (plan.layers, indices) |*cost, index| cost.* = layers.get(index).?;
    return plan;
}

pub fn widen(s: *mx.Scope, weight: mx.Array) !mx.Array {
    if (mx.shape(weight).len != 2 or mx.dtype(weight) != mx.c.MLX_UINT32 or mx.dim(weight, 0) <= 0 or mx.dim(weight, 1) <= 0) return error.InvalidTensorShape;
    var parts: std.ArrayList(mx.Array) = .empty;
    defer {
        for (parts.items) |part| mx.free(part);
        parts.deinit(mx.allocator);
    }
    const rows = mx.dim(weight, 0);
    const words = mx.dim(weight, 1);
    var start: i32 = 0;
    while (start < rows) : (start += @min(8192, rows - start)) {
        var chunk = mx.Scope{};
        defer chunk.deinit();
        const count = @min(8192, rows - start);
        var shifts: [16]u32 = undefined;
        for (&shifts, 0..) |*shift, i| shift.* = @intCast(2 * i);
        const codes = try chunk.reshape(try chunk.binary(mx.c.mlx_bitwise_and, try chunk.binary(mx.c.mlx_right_shift, try chunk.reshape(try chunk.slice(weight, 0, start, start + count), &.{ count, words, 1 }), try chunk.data(&shifts, &.{16}, mx.c.MLX_UINT32)), try chunk.data(&@as(u32, 3), &.{}, mx.c.MLX_UINT32)), &.{ count, words * 2, 8 });
        var word = try chunk.reshape(try chunk.slice(codes, 2, 0, 1), &.{ count, words * 2 });
        for (1..8) |j| {
            const shift: u32 = @intCast(4 * j);
            const value = try chunk.reshape(try chunk.slice(codes, 2, @intCast(j), @intCast(j + 1)), &.{ count, words * 2 });
            word = try chunk.binary(mx.c.mlx_bitwise_or, word, try chunk.binary(mx.c.mlx_left_shift, value, try chunk.data(&shift, &.{}, mx.c.MLX_UINT32)));
        }
        try mx.eval(word);
        const owned = try mx.retain(word);
        parts.append(mx.allocator, owned) catch |err| {
            mx.free(owned);
            return err;
        };
    }
    return s.cat(parts.items, 0);
}

pub fn check(io: std.Io, directory: []const u8, fixtures: []const u8) !void {
    const a = mx.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const temp = arena.allocator();
    const bytes = try weights.readFile(io, try std.fs.path.join(temp, &.{ fixtures, "widening.json" }));
    defer a.free(bytes);
    const Fixture = struct {
        model: u64,
        layers: []u64,
        rest: u64,
        budgets: []struct { budget: u64, form: []const u8 },
        modules: []struct { form: []const u8, path: []const u8, result: []const u8 },
        invalid: [][]const u8,
        weights: [][]const u8,
    };
    const reference = (try std.json.parseFromSlice(Fixture, temp, bytes, .{})).value;
    const config_bytes = try weights.readFile(io, try std.fs.path.join(temp, &.{ directory, "config.json" }));
    defer a.free(config_bytes);
    const config = (try std.json.parseFromSlice(std.json.Value, temp, config_bytes, .{})).value;
    const modules = (try std.json.parseFromValue([]const Module, temp, config.object.get("modules").?, .{})).value;
    const plan = try widening(temp, io, directory, modules);
    try std.testing.expectEqual(reference.model, plan.model);
    try std.testing.expectEqual(reference.rest, plan.rest);
    try std.testing.expectEqualSlices(u64, reference.layers, plan.layers);
    for (reference.budgets) |case| try std.testing.expectEqualStrings(case.form, try plan.choose(case.budget).name(temp));
    for (reference.modules) |case| try std.testing.expectEqualStrings(case.result, try (try Form.parse(case.form)).module(case.path).name(temp));
    for (reference.invalid) |form| try std.testing.expectError(error.InvalidBonsaiForm, Form.parse(form));
    try mx.init();
    defer mx.shutdown();
    for (reference.weights) |name| {
        var store = @import("checkpoint.zig").Store.init(64);
        defer store.deinit();
        try store.loadFile(io, try std.fs.path.join(temp, &.{ fixtures, name }), "", "");
        var s = mx.Scope{};
        defer s.deinit();
        const actual = try widen(&s, try store.get("input"));
        const expected = try store.get("output");
        try mx.evalMany(&.{ actual, expected }, false);
        try std.testing.expectEqualSlices(i32, mx.shape(expected), mx.shape(actual));
        const size = mx.c.mlx_array_size(actual);
        try std.testing.expectEqualSlices(u32, mx.c.mlx_array_data_uint32(expected)[0..size], mx.c.mlx_array_data_uint32(actual)[0..size]);
    }
    std.debug.print("PASS: Bonsai header accounting, {d} budget boundaries, {d} module forms and {d} bit-exact widening cases\n", .{ reference.budgets.len, reference.modules.len, reference.weights.len });
}
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
    const form = form_override orelse if (mx.tensor_units) Form.lanes else blk: {
        const plan = try widening(mx.allocator, io, dir, modules.value);
        defer mx.allocator.free(plan.layers);
        var ram: u64 = 0;
        var size: usize = @sizeOf(u64);
        if (std.c.sysctlbyname("hw.memsize", &ram, &size, null, 0) != 0 or ram == 0) return error.PhysicalMemoryUnavailable;
        const budget = try @import("memory_budget.zig").limit(ram, try @import("memory_runtime.zig").recommendedBytes(), 0.70, memory_limit);
        break :blk plan.choose(budget);
    };
    if ((form == .lanes) != mx.tensor_units) return error.BonsaiFormBackendMismatch;
    w.bonsai_form = form;
    const form_name = try form.name(mx.allocator);
    defer mx.allocator.free(form_name);
    std.debug.print("Bonsai projections: {s}\n", .{form_name});
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
            const selected = form.module(module.path);
            const weight = if (selected == .widened) try widen(&s, q[0]) else q[0];
            var scales = q[1];
            var biases = q[2];
            if (selected != .@"packed") {
                scales = try s.cast(try s.reshape(try s.stack(&.{ scales, scales }, -1), &.{ geometry.n, @divExact(geometry.k, 64) }), mx.bf16);
                biases = try s.cast(try s.reshape(try s.stack(&.{ biases, biases }, -1), &.{ geometry.n, @divExact(geometry.k, 64) }), mx.bf16);
            }
            var linear = try lanes.Linear.initFormat(&s, weight, scales, biases, .{ .bits = if (selected == .widened) 4 else 2, .group_size = if (selected != .@"packed") 64 else 128 });
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
