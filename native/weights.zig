const std = @import("std");
const mx = @import("mlx.zig");
const lanes = @import("lanes.zig");
pub fn readFile(io: std.Io, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var buffer: [8192]u8 = undefined;
    var reader = f.reader(io, &buffer);
    return reader.interface.allocRemaining(mx.allocator, .limited(128 * 1024 * 1024));
}
pub const Weights = struct {
    arrays: std.StringHashMap(mx.Array),
    linears: std.StringHashMap(lanes.Linear),
    pub fn init() Weights {
        return .{ .arrays = std.StringHashMap(mx.Array).init(mx.allocator), .linears = std.StringHashMap(lanes.Linear).init(mx.allocator) };
    }
    pub fn deinit(w: *Weights) void {
        var ls = w.linears.iterator();
        while (ls.next()) |e| {
            e.value_ptr.deinit();
            mx.allocator.free(e.key_ptr.*);
        }
        w.linears.deinit();
        var it = w.arrays.iterator();
        while (it.next()) |e| {
            mx.free(e.value_ptr.*);
            mx.allocator.free(e.key_ptr.*);
        }
        w.arrays.deinit();
    }
    pub fn get(w: *const Weights, name: []const u8) !mx.Array {
        return w.arrays.get(name) orelse {
            std.debug.print("Missing weight: {s}\n", .{name});
            return error.MissingWeight;
        };
    }
    pub fn linear(w: *const Weights, name: []const u8) !lanes.Linear {
        return w.linears.get(name) orelse error.MissingLinear;
    }
    // Both helpers take ownership, including when map insertion fails.
    fn putArray(w: *Weights, name: []const u8, value: mx.Array) !void {
        errdefer mx.free(value);
        if (w.arrays.contains(name)) return error.DuplicateWeight;
        const key = try mx.allocator.dupe(u8, name);
        errdefer mx.allocator.free(key);
        try w.arrays.put(key, value);
    }
    fn putLinear(w: *Weights, name: []const u8, value: lanes.Linear) !void {
        var owned = value;
        errdefer owned.deinit();
        if (w.linears.contains(name)) return error.DuplicateWeight;
        const key = try mx.allocator.dupe(u8, name);
        errdefer mx.allocator.free(key);
        try w.linears.put(key, owned);
    }
    fn releaseLinearSources(w: *Weights) !void {
        var it = w.linears.keyIterator();
        var buffer: [256]u8 = undefined;
        while (it.next()) |name| {
            for ([_][]const u8{ ".weight", ".scales", ".biases" }) |suffix| {
                const key = try std.fmt.bufPrint(&buffer, "{s}{s}", .{ name.*, suffix });
                if (w.arrays.fetchRemove(key)) |entry| {
                    mx.free(entry.value);
                    mx.allocator.free(entry.key);
                }
            }
        }
    }
    pub fn loadDraft(w: *Weights, io: std.Io, dir: []const u8) !void {
        var pathbuf: [4096]u8 = undefined;
        const bytes = try readFile(io, try std.fmt.bufPrint(&pathbuf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer cfg.deinit();
        try @import("config.zig").draft(cfg.value);
        const path = try std.fmt.bufPrintSentinel(&pathbuf, "{s}/model.safetensors", .{dir}, 0);
        try @import("safetensors.zig").validateFile(io, path);
        var map = mx.c.mlx_map_string_to_array_new();
        defer _ = mx.c.mlx_map_string_to_array_free(map);
        var meta = mx.c.mlx_map_string_to_string_new();
        defer _ = mx.c.mlx_map_string_to_string_free(meta);
        const cpu = mx.c.mlx_default_cpu_stream_new();
        defer _ = mx.c.mlx_stream_free(cpu);
        try mx.check(mx.c.mlx_load_safetensors(&map, &meta, path, cpu));
        const iter = mx.c.mlx_map_string_to_array_iterator_new(map);
        defer _ = mx.c.mlx_map_string_to_array_iterator_free(iter);
        while (true) {
            var key: [*c]const u8 = null;
            var value = mx.c.mlx_array_new();
            const rc = mx.c.mlx_map_string_to_array_iterator_next(&key, &value, iter);
            if (rc != 0 or key == null) {
                mx.free(value);
                break;
            }
            try w.putArray(std.mem.span(key), value);
        }
        var it = w.arrays.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            const value = e.value_ptr.*;
            if (!std.mem.endsWith(u8, name, ".weight") or mx.shape(value).len != 2 or std.mem.indexOf(u8, name, "codebook") != null) continue;
            var s = mx.Scope{};
            defer s.deinit();
            var quant = mx.c.mlx_vector_array_new();
            defer _ = mx.c.mlx_vector_array_free(quant);
            try mx.check(mx.c.mlx_quantize(&quant, value, mx.opt(64), mx.opt(4), "affine", mx.empty, mx.stream));
            var arrays: [3]mx.Array = undefined;
            for (0..3) |j| {
                var a = mx.c.mlx_array_new();
                const rc = mx.c.mlx_vector_array_get(&a, quant, j);
                arrays[j] = try s.result(rc, a);
            }
            const l = try lanes.Linear.init(&s, arrays[0], arrays[1], arrays[2]);
            try w.putLinear(name[0 .. name.len - 7], l);
        }
        try w.releaseLinearSources();
    }
    pub fn load(w: *Weights, io: std.Io, dir: []const u8) !void {
        var pathbuf: [4096]u8 = undefined;
        const config = try readFile(io, try std.fmt.bufPrint(&pathbuf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(config);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, config, .{});
        defer cfg.deinit();
        try @import("config.zig").target(cfg.value);
        const index = try readFile(io, try std.fmt.bufPrint(&pathbuf, "{s}/model.safetensors.index.json", .{dir}));
        defer mx.allocator.free(index);
        const parsed = try std.json.parseFromSlice(std.json.Value, mx.allocator, index, .{});
        defer parsed.deinit();
        var shards = std.StringHashMap(void).init(mx.allocator);
        defer shards.deinit();
        if (parsed.value != .object) return error.InvalidWeightIndex;
        const weight_map = parsed.value.object.get("weight_map") orelse return error.InvalidWeightIndex;
        if (weight_map != .object) return error.InvalidWeightIndex;
        var it = weight_map.object.iterator();
        while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, "language_model.")) {
            if (e.value_ptr.* != .string) return error.InvalidWeightIndex;
            try @import("safetensors.zig").shardName(e.value_ptr.string);
            try shards.put(e.value_ptr.string, {});
        };
        var files = shards.keyIterator();
        if (shards.count() == 0) return error.MissingWeights;
        while (files.next()) |name| {
            const path = try std.fmt.bufPrintSentinel(&pathbuf, "{s}/{s}", .{ dir, name.* }, 0);
            try @import("safetensors.zig").validateFile(io, path);
            std.debug.print("Loading {s}\n", .{name.*});
            var map = mx.c.mlx_map_string_to_array_new();
            defer _ = mx.c.mlx_map_string_to_array_free(map);
            var meta = mx.c.mlx_map_string_to_string_new();
            defer _ = mx.c.mlx_map_string_to_string_free(meta);
            const cpu = mx.c.mlx_default_cpu_stream_new();
            defer _ = mx.c.mlx_stream_free(cpu);
            try mx.check(mx.c.mlx_load_safetensors(&map, &meta, path, cpu));
            const iter = mx.c.mlx_map_string_to_array_iterator_new(map);
            defer _ = mx.c.mlx_map_string_to_array_iterator_free(iter);
            while (true) {
                var key: [*c]const u8 = null;
                var value = mx.c.mlx_array_new();
                const rc = mx.c.mlx_map_string_to_array_iterator_next(&key, &value, iter);
                if (rc != 0 or key == null) {
                    mx.free(value);
                    break;
                }
                const n = std.mem.span(key);
                if (!std.mem.startsWith(u8, n, "language_model.") or std.mem.indexOf(u8, n, ".mtp.") != null) {
                    mx.free(value);
                    continue;
                }
                try w.putArray(n[15..], value);
            }
        }
        // MLX-format checkpoints already carry shifted RMS weights and [C,4,1] convs.
        // Refuse raw HF tensors rather than silently applying the wrong normalization.
        if (mx.dim(try w.get("model.layers.0.linear_attn.conv1d.weight"), -1) != 1) return error.UnsanitizedCheckpoint;
        var entries = w.arrays.iterator();
        while (entries.next()) |e| {
            if (!std.mem.endsWith(u8, e.key_ptr.*, ".scales") or std.mem.indexOf(u8, e.key_ptr.*, "embed_tokens") != null) continue;
            const name = e.key_ptr.*[0 .. e.key_ptr.len - 7];
            var s = mx.Scope{};
            defer s.deinit();
            const weight = try w.get(try std.fmt.bufPrint(&pathbuf, "{s}.weight", .{name}));
            const biases = try w.get(try std.fmt.bufPrint(&pathbuf, "{s}.biases", .{name}));
            const linear_ = try lanes.Linear.init(&s, weight, e.value_ptr.*, biases);
            try w.putLinear(name, linear_);
        }
        try w.releaseLinearSources();
    }
    pub fn embed(w: *const Weights, s: *mx.Scope, tokens: []const i32) !mx.Array {
        const ids = try s.ints(tokens);
        const out = try s.dequant(try s.take(try w.get("model.embed_tokens.weight"), ids, 0), try s.take(try w.get("model.embed_tokens.scales"), ids, 0), try s.take(try w.get("model.embed_tokens.biases"), ids, 0));
        return s.reshape(out, &.{ 1, @intCast(tokens.len), 5120 });
    }
};
