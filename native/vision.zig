const std = @import("std");
const mx = @import("mlx.zig");
const c = mx.c;
const A = mx.Array;
const Store = @import("checkpoint.zig").Store;
const quant = @import("quantization.zig");
pub const Grid = @import("vision_positions.zig").Grid;
const vp = @import("vision_positions.zig");
const input = @import("image_input.zig");

pub const Prompt = struct {
    scope: mx.Scope = .{},
    embeddings: A = mx.empty,
    positions: vp.Positions,
    pub fn deinit(p: *Prompt) void {
        p.scope.deinit();
        p.positions.deinit(mx.allocator);
    }
    pub fn prepare(io: std.Io, dir: []const u8, paths: []const []const u8, tokens: *std.ArrayList(i32), allocator: std.mem.Allocator, weights: *@import("weights.zig").Weights) !Prompt {
        if (paths.len == 0 or paths.len > 4) return error.InvalidImageCount;
        var tower = try Tower.init(io, dir);
        defer tower.deinit();
        var grids: [4]Grid = undefined;
        var features: [4]A = undefined;
        var scope = mx.Scope{};
        errdefer scope.deinit();
        const ids = try tower.tokenIds();
        const limits = try processorLimits(io, dir);
        var decoded_pixels: usize = 0;
        var encoded_bytes: usize = 0;
        for (paths, 0..) |path, i| {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, mx.allocator, .limited(10 * 1024 * 1024));
            defer mx.allocator.free(bytes);
            encoded_bytes += bytes.len;
            if (encoded_bytes > 20 * 1024 * 1024) return error.ImageByteLimitExceeded;
            const image = try input.Image.decode(bytes);
            defer image.deinit();
            decoded_pixels += image.width * image.height;
            if (decoded_pixels > 32 * 1024 * 1024) return error.ImagePixelLimitExceeded;
            const cap = @min(limits.max, 4096 / paths.len * 1024);
            grids[i] = try input.resizeGrid(image.height, image.width, @min(limits.min, cap), cap);
            const pixels = try input.patches(image, grids[i]);
            defer mx.allocator.free(pixels);
            var image_scope = mx.Scope{};
            defer image_scope.deinit();
            const array = try image_scope.data(pixels.ptr, &.{ @intCast(pixels.len / 1536), 1536 }, mx.f32t);
            features[i] = try scope.own(try mx.retain(try tower.encode(&image_scope, array, grids[i])));
        }
        var expanded: std.ArrayList(i32) = .empty;
        defer expanded.deinit(allocator);
        if (std.mem.indexOfScalar(i32, tokens.items, ids.image) == null) {
            for (grids[0..paths.len]) |grid| {
                try expanded.append(allocator, ids.start);
                try expanded.appendNTimes(allocator, ids.image, try grid.count());
                try expanded.append(allocator, ids.end);
            }
            try expanded.appendSlice(allocator, tokens.items);
        }
        var index: usize = 0;
        if (expanded.items.len == 0) {
            for (tokens.items) |token| {
                if (token == ids.image) {
                    if (index >= paths.len) return error.UnmatchedImageTokens;
                    const count = try grids[index].count();
                    try expanded.appendNTimes(allocator, token, count);
                    index += 1;
                } else try expanded.append(allocator, token);
            }
        } else index = paths.len;
        if (index != paths.len) return error.MissingImageTokens;
        const positions = try vp.Positions.init(mx.allocator, expanded.items, grids[0..paths.len], ids);
        errdefer positions.deinit(mx.allocator);
        const text = try weights.embed(&scope, expanded.items);
        var parts: [9]A = undefined;
        var length: usize = 0;
        var cursor: usize = 0;
        for (positions.spans, features[0..paths.len]) |span, feature| {
            if (span.begin > cursor) {
                parts[length] = try scope.slice(text, 1, @intCast(cursor), @intCast(span.begin));
                length += 1;
            }
            parts[length] = try scope.reshape(try scope.cast(feature, mx.dtype(text)), &.{ 1, @intCast(span.end - span.begin), 5120 });
            length += 1;
            cursor = span.end;
        }
        if (cursor < expanded.items.len) {
            parts[length] = try scope.slice(text, 1, @intCast(cursor), @intCast(expanded.items.len));
            length += 1;
        }
        const embeddings = try scope.cat(parts[0..length], 1);
        try mx.eval(embeddings);
        std.mem.swap(std.ArrayList(i32), tokens, &expanded);
        return .{ .scope = scope, .embeddings = embeddings, .positions = positions };
    }
};

const Limits = struct { min: usize, max: usize };
fn processorLimits(io: std.Io, dir: []const u8) !Limits {
    var result: Limits = .{ .min = 65536, .max = 4096 * 1024 };
    var buf: [4096]u8 = undefined;
    for ([_][]const u8{ "preprocessor_config.json", "processor_config.json" }) |file| {
        const bytes = @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, file })) catch |err| {
            if (err == error.FileNotFound) continue;
            return err;
        };
        defer mx.allocator.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer parsed.deinit();
        var raw = parsed.value;
        if (raw != .object) return error.InvalidImageProcessor;
        if (std.mem.eql(u8, file, "processor_config.json")) raw = raw.object.get("image_processor") orelse continue;
        if (raw != .object) return error.InvalidImageProcessor;
        inline for (.{ .{ "patch_size", 16 }, .{ "temporal_patch_size", 2 }, .{ "merge_size", 2 } }) |field| if (raw.object.get(field[0])) |v| {
            if (v != .integer or v.integer != field[1]) return error.InvalidImageProcessor;
        };
        for ([_][]const u8{ "image_mean", "image_std" }) |key| if (raw.object.get(key)) |v| {
            if (v != .array or v.array.items.len != 3) return error.InvalidImageProcessor;
            for (v.array.items) |component| if (component != .float or component.float != 0.5) return error.UnsupportedImageNormalization;
        };
        for ([_][]const u8{ "do_rescale", "do_normalize", "do_convert_rgb" }) |key| if (raw.object.get(key)) |v| {
            if (v != .bool or !v.bool) return error.UnsupportedImageNormalization;
        };
        if (raw.object.get("rescale_factor")) |v| if (v != .float or v.float != 1.0 / 255.0) return error.UnsupportedImageNormalization;
        const size = raw.object.get("size") orelse .null;
        inline for (.{ .{ "min_pixels", "shortest_edge", "min" }, .{ "max_pixels", "longest_edge", "max" } }) |field| {
            const value = raw.object.get(field[0]) orelse (if (size == .object) size.object.get(field[1]) else null);
            if (value) |v| {
                if (v != .integer or v.integer < 1024) return error.InvalidImageProcessor;
                @field(result, field[2]) = std.math.cast(usize, v.integer) orelse return error.InvalidImageProcessor;
            }
        }
    }
    if (result.min > result.max) return error.InvalidImageProcessor;
    // TensorFold uses getattr(processor, "min_pixels"/"max_pixels"). Newer
    // Transformers stores these only in `size`, selecting the frontend fallbacks.
    if (!@import("native_runtime").vision_legacy_pixel_limits) return .{ .min = 1024, .max = 4096 * 1024 };
    return result;
}

pub fn checkImage(io: std.Io, path: []const u8, fixture: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, mx.allocator, .limited(10 * 1024 * 1024));
    defer mx.allocator.free(bytes);
    const image = try input.Image.decode(bytes);
    defer image.deinit();
    const grid = try input.resizeGrid(image.height, image.width, 65536, 4096 * 1024);
    const pixels = try input.patches(image, grid);
    defer mx.allocator.free(pixels);
    var s = mx.Scope{};
    defer s.deinit();
    const array = try s.data(pixels.ptr, &.{ @intCast(pixels.len / 1536), 1536 }, mx.f32t);
    var buf: [4096]u8 = undefined;
    try mx.eval(array);
    try mx.check(c.mlx_save(try std.fmt.bufPrintSentinel(&buf, "{s}/pixels-native.npy", .{fixture}, 0), array));
}

pub fn check(io: std.Io, dir: []const u8, fixture: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var tower = try Tower.init(io, dir);
    defer tower.deinit();
    var buf: [4096]u8 = undefined;
    const output = try std.fmt.bufPrint(&buf, "{s}/native", .{fixture});
    try std.Io.Dir.cwd().createDirPath(io, output);
    const saved = try mx.allocator.dupe(u8, output);
    defer mx.allocator.free(saved);
    tower.trace_dir = saved;
    const grid_bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/grid.json", .{fixture}));
    defer mx.allocator.free(grid_bytes);
    const parsed = try std.json.parseFromSlice(Grid, mx.allocator, grid_bytes, .{});
    defer parsed.deinit();
    var s = mx.Scope{};
    defer s.deinit();
    var pixels = c.mlx_array_new();
    const cpu = c.mlx_default_cpu_stream_new();
    defer _ = c.mlx_stream_free(cpu);
    const rc = c.mlx_load(&pixels, try std.fmt.bufPrintSentinel(&buf, "{s}/pixels.npy", .{fixture}, 0), cpu);
    _ = try s.result(rc, pixels);
    _ = try tower.encode(&s, pixels, parsed.value);
    std.debug.print("Vision encoder completed; compare stage arrays against the upstream oracle.\n", .{});
}

pub const Tower = struct {
    weights: Store,
    config: std.json.Parsed(std.json.Value),
    ops: @import("prefill_ops.zig").Ops = .{},
    prefix: []const u8,
    trace_dir: ?[]const u8 = null,

    pub fn init(io: std.Io, dir: []const u8) !Tower {
        var buf: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const config = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{ .allocate = .alloc_always });
        errdefer config.deinit();
        const root = config.value;
        if (root != .object) return error.InvalidVisionConfig;
        const kind = root.object.get("model_type") orelse return error.InvalidVisionConfig;
        if (kind != .string or !std.mem.eql(u8, kind.string, "qwen3_5")) return error.UnsupportedVisionModel;
        const vision = root.object.get("vision_config") orelse return error.InvalidVisionConfig;
        if (vision != .object) return error.InvalidVisionConfig;
        const text = root.object.get("text_config") orelse return error.InvalidVisionConfig;
        if (text != .object) return error.InvalidVisionConfig;
        const rope = text.object.get("rope_parameters") orelse return error.InvalidVisionConfig;
        if (rope != .object) return error.InvalidVisionConfig;
        if (rope.object.get("mrope_section")) |sections| {
            if (sections != .array or sections.array.items.len != 3) return error.InvalidVisionConfig;
            for (sections.array.items, [_]i64{ 11, 11, 10 }) |section, expected| if (section != .integer or section.integer != expected) return error.UnsupportedVisionRotary;
        }
        inline for (.{ .{ "depth", 27 }, .{ "hidden_size", 1152 }, .{ "num_heads", 16 }, .{ "intermediate_size", 4304 }, .{ "out_hidden_size", 5120 }, .{ "patch_size", 16 }, .{ "temporal_patch_size", 2 }, .{ "spatial_merge_size", 2 }, .{ "num_position_embeddings", 2304 } }) |field| {
            const value = vision.object.get(field[0]) orelse return error.InvalidVisionConfig;
            if (value != .integer or value.integer != field[1]) return error.UnsupportedVisionGeometry;
        }
        if (vision.object.get("deepstack_visual_indexes")) |value| if (value != .array or value.array.items.len != 0) return error.UnsupportedVisionDeepstack;
        var weights = Store.init(64);
        errdefer weights.deinit();
        const prefixes = [_][]const u8{ "model.language_model.visual.", "model.visual.", "vision_tower.", "visual." };
        for (prefixes) |prefix| {
            weights.load(io, dir, prefix) catch |err| {
                if (err == error.MissingWeights) continue;
                return err;
            };
            if (weights.arrays.count() != 0) return .{ .weights = weights, .config = config, .prefix = prefix };
        }
        return error.MissingVisionWeights;
    }
    pub fn deinit(t: *Tower) void {
        t.weights.deinit();
        t.config.deinit();
        t.ops.deinit();
    }
    pub fn tokenIds(t: *Tower) !vp.Tokens {
        var out: vp.Tokens = undefined;
        inline for (.{ .{ "image", "image_token_id" }, .{ "start", "vision_start_token_id" }, .{ "end", "vision_end_token_id" }, .{ "video", "video_token_id" } }) |field| {
            const value = t.config.value.object.get(field[1]) orelse return error.InvalidVisionConfig;
            if (value != .integer) return error.InvalidVisionConfig;
            @field(out, field[0]) = std.math.cast(i32, value.integer) orelse return error.InvalidVisionConfig;
            if (@field(out, field[0]) < 0 or @field(out, field[0]) >= 248320) return error.InvalidVisionConfig;
        }
        const values = [_]i32{ out.image, out.start, out.end, out.video };
        for (values, 0..) |value, i| for (values[i + 1 ..]) |other| if (value == other) return error.InvalidVisionConfig;
        return out;
    }
    fn format(t: *Tower, name: []const u8) !?quant.Spec {
        var buf: [256]u8 = undefined;
        if (!t.weights.has(try std.fmt.bufPrint(&buf, "{s}.scales", .{name}))) return null;
        const root = t.config.value.object;
        const raw = root.get("quantization") orelse root.get("quantization_config") orelse return error.MissingQuantization;
        if (raw != .object) return error.InvalidQuantization;
        var bits = raw.object.get("bits");
        var group = raw.object.get("group_size");
        var mode = raw.object.get("mode");
        if (raw.object.get(try std.fmt.bufPrint(&buf, "{s}{s}", .{ t.prefix, name }))) |override| {
            if (override == .bool and !override.bool) return error.InvalidQuantization;
            if (override == .object) {
                bits = override.object.get("bits") orelse bits;
                group = override.object.get("group_size") orelse group;
                mode = override.object.get("mode") orelse mode;
            }
        }
        if (mode) |v| if (v != .string or !std.mem.eql(u8, v.string, "affine")) return error.UnsupportedQuantization;
        const b = bits orelse return error.MissingQuantization;
        const g = group orelse return error.MissingQuantization;
        if (b != .integer or g != .integer) return error.InvalidQuantization;
        const spec = quant.Spec{ .bits = std.math.cast(i32, b.integer) orelse return error.InvalidQuantization, .group_size = std.math.cast(i32, g.integer) orelse return error.InvalidQuantization };
        try spec.validate();
        return spec;
    }
    fn linear(t: *Tower, s: *mx.Scope, name: []const u8, x: A) !A {
        const w = try t.weights.field(name, "weight");
        const b = try t.weights.field(name, "bias");
        var out = c.mlx_array_new();
        if (try t.format(name)) |f| {
            const rc = c.mlx_quantized_matmul(&out, x, w, try t.weights.field(name, "scales"), try t.weights.field(name, "biases"), true, mx.opt(f.group_size), mx.opt(f.bits), "affine", mx.stream);
            return s.binary(c.mlx_add, try s.result(rc, out), b);
        }
        const rc = c.mlx_addmm(&out, b, x, try s.transpose(w, &.{ 1, 0 }), 1, 1, mx.stream);
        return s.result(rc, out);
    }
    fn norm(t: *Tower, s: *mx.Scope, name: []const u8, x: A) !A {
        var out = c.mlx_array_new();
        const rc = c.mlx_fast_layer_norm(&out, x, try t.weights.field(name, "weight"), try t.weights.field(name, "bias"), 1e-6, mx.stream);
        return s.result(rc, out);
    }
    fn trace(t: *Tower, s: *mx.Scope, name: []const u8, x: A) !void {
        const dir = t.trace_dir orelse return;
        var buf: [4096]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}.npy", .{ dir, name }, 0);
        const value = try s.cast(x, mx.f32t);
        try mx.eval(value);
        try mx.check(c.mlx_save(path, value));
    }
    fn positionEmbedding(t: *Tower, s: *mx.Scope, grid: Grid) !A {
        var coords: [2]A = undefined;
        for ([_]i32{ grid.height, grid.width }, 0..) |n, axis| {
            var value = c.mlx_array_new();
            const rc = c.mlx_linspace(&value, 0, 47, n, mx.f32t, mx.stream);
            coords[axis] = try s.result(rc, value);
        }
        const hf = try s.cast(coords[0], mx.i32t);
        const wf = try s.cast(coords[1], mx.i32t);
        const hc = try s.binary(c.mlx_minimum, try s.binary(c.mlx_add, hf, try s.ints(&.{1})), try s.ints(&.{47}));
        const wc = try s.binary(c.mlx_minimum, try s.binary(c.mlx_add, wf, try s.ints(&.{1})), try s.ints(&.{47}));
        const dh = try s.reshape(try s.binary(c.mlx_subtract, coords[0], try s.cast(hf, mx.f32t)), &.{ grid.height, 1 });
        const dw = try s.reshape(try s.binary(c.mlx_subtract, coords[1], try s.cast(wf, mx.f32t)), &.{ 1, grid.width });
        const ih = try s.binary(c.mlx_subtract, try s.scalar(1), dh);
        const iw = try s.binary(c.mlx_subtract, try s.scalar(1), dw);
        var parts: [4]A = undefined;
        for (0..4) |j| {
            const rows = try s.reshape(try s.binary(c.mlx_multiply, if (j < 2) hf else hc, try s.ints(&.{48})), &.{ grid.height, 1 });
            const cols = try s.reshape(if (j % 2 == 0) wf else wc, &.{ 1, grid.width });
            const indices = try s.reshape(try s.binary(c.mlx_add, rows, cols), &.{grid.height * grid.width});
            var embedding = try s.take(try t.weights.get("pos_embed.weight"), indices, 0);
            if (try t.format("pos_embed")) |f| {
                var value = c.mlx_array_new();
                const rc = c.mlx_dequantize(&value, embedding, try s.take(try t.weights.get("pos_embed.scales"), indices, 0), try s.take(try t.weights.get("pos_embed.biases"), indices, 0), mx.opt(f.group_size), mx.opt(f.bits), "affine", mx.empty, .{ .has_value = false, .value = mx.bf16 }, mx.stream);
                embedding = try s.result(rc, value);
            }
            const fraction = try s.reshape(try s.binary(c.mlx_multiply, if (j < 2) ih else dh, if (j % 2 == 0) iw else dw), &.{ grid.height * grid.width, 1 });
            parts[j] = try s.binary(c.mlx_multiply, embedding, try s.cast(fraction, mx.dtype(embedding)));
        }
        const sum = try s.binary(c.mlx_add, try s.binary(c.mlx_add, try s.binary(c.mlx_add, parts[0], parts[1]), parts[2]), parts[3]);
        return s.reshape(try s.transpose(try s.reshape(sum, &.{ @divExact(grid.height, 2), 2, @divExact(grid.width, 2), 2, 1152 }), &.{ 0, 2, 1, 3, 4 }), &.{ grid.height * grid.width, 1152 });
    }
    fn frequencies(s: *mx.Scope, grid: Grid) !A {
        const n: usize = @intCast(grid.height * grid.width);
        const ids = try mx.allocator.alloc(i32, 2 * n);
        defer mx.allocator.free(ids);
        var j: usize = 0;
        for (0..@intCast(@divExact(grid.height, 2))) |row| for (0..@intCast(@divExact(grid.width, 2))) |col| for (0..2) |dy| for (0..2) |dx| {
            ids[j] = @intCast(row * 2 + dy);
            ids[n + j] = @intCast(col * 2 + dx);
            j += 1;
        };
        const powers: [18]f32 = .{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34 };
        const exp = try s.binary(c.mlx_divide, try s.data(&powers, &.{18}, mx.f32t), try s.scalar(36));
        const inv = try s.binary(c.mlx_divide, try s.scalar(1), try s.binary(c.mlx_power, try s.scalar(10000), exp));
        const h = try s.binary(c.mlx_multiply, try s.reshape(try s.cast(try s.ints(ids[0..n]), mx.f32t), &.{ @intCast(n), 1 }), inv);
        const w = try s.binary(c.mlx_multiply, try s.reshape(try s.cast(try s.ints(ids[n..]), mx.f32t), &.{ @intCast(n), 1 }), inv);
        return s.reshape(try s.cat(&.{ h, w, h, w }, 1), &.{ 1, @intCast(n), 1, 72 });
    }
    fn rotary(s: *mx.Scope, x: A, cos: A, sin: A) !A {
        const half = try s.cat(&.{ try s.unary(c.mlx_negative, try s.slice(x, 3, 36, 72)), try s.slice(x, 3, 0, 36) }, 3);
        return s.cast(try s.binary(c.mlx_add, try s.binary(c.mlx_multiply, x, cos), try s.binary(c.mlx_multiply, half, sin)), mx.dtype(x));
    }
    pub fn encode(t: *Tower, s: *mx.Scope, pixels: A, grid: Grid) !A {
        const count = try grid.count();
        if (count > 4096 or mx.shape(pixels).len != 2 or mx.dim(pixels, 0) != count * 4 or mx.dim(pixels, 1) != 1536) return error.InvalidImagePatches;
        const n: i32 = @intCast(count * 4);
        var conv = try t.weights.get("patch_embed.proj.weight");
        if (std.mem.eql(i32, mx.shape(conv), &.{ 1152, 3, 2, 16, 16 })) conv = try s.transpose(conv, &.{ 0, 2, 3, 4, 1 });
        if (!std.mem.eql(i32, mx.shape(conv), &.{ 1152, 2, 16, 16, 3 })) return error.InvalidVisionConvolution;
        const patches = try s.transpose(try s.reshape(try s.cast(pixels, mx.dtype(conv)), &.{ n, 3, 2, 16, 16 }), &.{ 0, 2, 3, 4, 1 });
        var out = c.mlx_array_new();
        const rc = c.mlx_conv3d(&out, patches, conv, 2, 16, 16, 0, 0, 0, 1, 1, 1, 1, mx.stream);
        var h = try s.reshape(try s.binary(c.mlx_add, try s.result(rc, out), try t.weights.get("patch_embed.proj.bias")), &.{ n, 1152 });
        try t.trace(s, "patch", h);
        const pos = try t.positionEmbedding(s, grid);
        try t.trace(s, "position", pos);
        h = try s.binary(c.mlx_add, h, pos);
        const freq = try frequencies(s, grid);
        try t.trace(s, "frequencies", freq);
        const cos = try s.unary(c.mlx_cos, freq);
        const sin = try s.unary(c.mlx_sin, freq);
        for (0..27) |layer| {
            var ls = mx.Scope{};
            defer ls.deinit();
            var buf: [128]u8 = undefined;
            const x = try t.norm(&ls, try std.fmt.bufPrint(&buf, "blocks.{d}.norm1", .{layer}), h);
            const qkv = try ls.transpose(try ls.reshape(try t.linear(&ls, try std.fmt.bufPrint(&buf, "blocks.{d}.attn.qkv", .{layer}), x), &.{ n, 3, 16, 72 }), &.{ 1, 0, 2, 3 });
            var q = try rotary(&ls, try ls.slice(qkv, 0, 0, 1), cos, sin);
            var k = try rotary(&ls, try ls.slice(qkv, 0, 1, 2), cos, sin);
            var v = try ls.slice(qkv, 0, 2, 3);
            const pad = try ls.zeros(&.{ 1, n, 16, 8 }, mx.dtype(q));
            q = try ls.transpose(try ls.cat(&.{ q, pad }, 3), &.{ 0, 2, 1, 3 });
            k = try ls.transpose(try ls.cat(&.{ k, pad }, 3), &.{ 0, 2, 1, 3 });
            v = try ls.transpose(try ls.cat(&.{ v, pad }, 3), &.{ 0, 2, 1, 3 });
            var attn = c.mlx_array_new();
            const arc = c.mlx_fast_scaled_dot_product_attention(&attn, q, k, v, 0.11785113019775792, "", mx.empty, mx.empty, false, mx.stream);
            _ = try ls.result(arc, attn);
            const rows = try ls.reshape(try ls.transpose(try ls.slice(attn, 3, 0, 72), &.{ 0, 2, 1, 3 }), &.{ n, 1152 });
            h = try ls.binary(c.mlx_add, h, try t.linear(&ls, try std.fmt.bufPrint(&buf, "blocks.{d}.attn.proj", .{layer}), rows));
            const normed = try t.norm(&ls, try std.fmt.bufPrint(&buf, "blocks.{d}.norm2", .{layer}), h);
            const fc = try t.linear(&ls, try std.fmt.bufPrint(&buf, "blocks.{d}.mlp.linear_fc1", .{layer}), normed);
            const act = try t.ops.call(&ls, .gelu_tanh, &.{fc});
            h = try ls.binary(c.mlx_add, h, try t.linear(&ls, try std.fmt.bufPrint(&buf, "blocks.{d}.mlp.linear_fc2", .{layer}), act));
            try mx.eval(h);
            h = try s.own(try mx.retain(h));
            try t.trace(s, try std.fmt.bufPrint(&buf, "block-{d}", .{layer}), h);
        }
        const normed = try s.reshape(try t.norm(s, "merger.norm", h), &.{ @intCast(count), 4608 });
        const act = try t.ops.call(s, .gelu, &.{try t.linear(s, "merger.linear_fc1", normed)});
        const embeddings = try t.linear(s, "merger.linear_fc2", act);
        try mx.eval(embeddings);
        try t.trace(s, "embeddings", embeddings);
        return embeddings;
    }
};
