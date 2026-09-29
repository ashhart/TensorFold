//! Shared native checkpoint reader for the additional Metal families.
const std = @import("std");
const mx = @import("mlx.zig");
const lanes = @import("lanes.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const Quant = @import("quantization.zig");
pub const Store = struct {
    arrays: std.StringHashMap(A),
    dense: std.StringHashMap(lanes.Linear),
    group: i32,
    flash_drafts: ?bool = null,
    quant_config: ?std.json.Parsed(std.json.Value) = null,
    formats: std.StringHashMap(Quant.Spec),
    pub fn init(group: i32) Store {
        return .{ .arrays = std.StringHashMap(A).init(mx.allocator), .dense = std.StringHashMap(lanes.Linear).init(mx.allocator), .formats = std.StringHashMap(Quant.Spec).init(mx.allocator), .group = group };
    }
    pub fn deinit(w: *Store) void {
        var it = w.arrays.iterator();
        while (it.next()) |e| {
            mx.free(e.value_ptr.*);
            mx.allocator.free(e.key_ptr.*);
        }
        w.arrays.deinit();
        var ls = w.dense.iterator();
        while (ls.next()) |e| {
            e.value_ptr.deinit();
            mx.allocator.free(e.key_ptr.*);
        }
        w.dense.deinit();
        var formats = w.formats.keyIterator();
        while (formats.next()) |key| mx.allocator.free(key.*);
        w.formats.deinit();
        if (w.quant_config) |cfg| cfg.deinit();
    }
    pub fn configure(w: *Store, bytes: []const u8) !void {
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{ .allocate = .alloc_always });
        errdefer cfg.deinit();
        _ = try Quant.resolve(cfg.value, null);
        if (w.quant_config) |old| old.deinit();
        w.quant_config = cfg;
    }
    pub fn format(w: *const Store, name: []const u8) !?Quant.Spec {
        if (w.formats.get(name)) |value| return value;
        if (w.quant_config) |cfg| return if (w.flash_drafts != null) Quant.resolveFlash(cfg.value, name) else Quant.resolve(cfg.value, name);
        return Quant.Spec{ .bits = 4, .group_size = w.group };
    }
    pub fn affine(w: *Store, name: []const u8) !@import("flash_ops.zig").Weight {
        return .{ .arrays = try w.triple(name), .format = (try w.format(name)) orelse return error.UnsupportedQuantization };
    }
    pub fn putAffine(w: *Store, name: []const u8, value: @import("flash_ops.zig").Weight) !void {
        var buffer: [512]u8 = undefined;
        inline for (.{ "weight", "scales", "biases" }, 0..) |suffix, i| try w.put(try std.fmt.bufPrint(&buffer, "{s}.{s}", .{ name, suffix }), value.arrays[i]);
        if (w.formats.getPtr(name)) |f| f.* = value.format else {
            const key = try mx.allocator.dupe(u8, name);
            errdefer mx.allocator.free(key);
            try w.formats.put(key, value.format);
        }
    }
    pub fn put(w: *Store, key: []const u8, value: A) !void {
        const own = try mx.retain(value);
        errdefer mx.free(own);
        if (w.arrays.getPtr(key)) |p| {
            mx.free(p.*);
            p.* = own;
            return;
        }
        const name = try mx.allocator.dupe(u8, key);
        errdefer mx.allocator.free(name);
        try w.arrays.put(name, own);
    }
    pub fn get(w: *Store, key: []const u8) !A {
        return w.arrays.get(key) orelse {
            @import("server_live.zig").print("Missing tensor: {s}\n", .{key});
            return error.MissingWeight;
        };
    }
    pub fn field(w: *Store, name: []const u8, suffix: []const u8) !A {
        var buf: [512]u8 = undefined;
        return w.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, suffix }));
    }
    pub fn has(w: *Store, key: []const u8) bool {
        return w.arrays.contains(key);
    }
    fn accepts(w: *const Store, key: []const u8, strip: []const u8) bool {
        return if (w.flash_drafts) |drafts| @import("flash_names.zig").accepts(key, drafts) else std.mem.startsWith(u8, key, strip);
    }
    pub fn loadFile(w: *Store, io: std.Io, path: []const u8, prefix: []const u8, strip: []const u8) !void {
        try @import("safetensors.zig").validateFile(io, path);
        const z = try mx.allocator.dupeSentinel(u8, path, 0);
        defer mx.allocator.free(z);
        var map = mx.c.mlx_map_string_to_array_new();
        defer _ = mx.c.mlx_map_string_to_array_free(map);
        var meta = mx.c.mlx_map_string_to_string_new();
        defer _ = mx.c.mlx_map_string_to_string_free(meta);
        const cpu = mx.c.mlx_default_cpu_stream_new();
        defer _ = mx.c.mlx_stream_free(cpu);
        try mx.check(mx.c.mlx_load_safetensors(&map, &meta, z, cpu));
        const it = mx.c.mlx_map_string_to_array_iterator_new(map);
        defer _ = mx.c.mlx_map_string_to_array_iterator_free(it);
        while (true) {
            var key: [*c]const u8 = null;
            var value = mx.c.mlx_array_new();
            const rc = mx.c.mlx_map_string_to_array_iterator_next(&key, &value, it);
            defer mx.free(value);
            if (rc != 0 or key == null) break;
            const raw = std.mem.span(key);
            if (!w.accepts(raw, strip)) continue;
            var buf: [512]u8 = undefined;
            const name = if (w.flash_drafts != null) try @import("flash_names.zig").normalize(&buf, raw) else try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, raw[strip.len..] });
            if (w.flash_drafts != null and std.mem.endsWith(u8, name, "ngram_embedding.weight_scale")) {
                var scope = mx.Scope{};
                defer scope.deinit();
                const scale = try scope.cast(value, mx.f32t);
                try mx.eval(scale);
                for (mx.c.mlx_array_data_float32(scale)[0..mx.c.mlx_array_size(scale)]) |v|
                    if (v != 1.0) return error.UnsupportedPLEScale;
                continue;
            }
            if (w.has(name)) return error.DuplicateWeight;
            try w.put(name, value);
        }
    }
    pub fn load(w: *Store, io: std.Io, dir: []const u8, strip: []const u8) !void {
        var buf: [4096]u8 = undefined;
        const bytes = @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/model.safetensors.index.json", .{dir})) catch |err| {
            if (err == error.FileNotFound) return w.loadUnindexed(io, dir, strip);
            return err;
        };
        defer mx.allocator.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidWeightIndex;
        const map = parsed.value.object.get("weight_map") orelse return error.InvalidWeightIndex;
        if (map != .object) return error.InvalidWeightIndex;
        var shards = std.StringHashMap(void).init(mx.allocator);
        defer shards.deinit();
        var it = map.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.InvalidWeightIndex;
            try @import("safetensors.zig").shardName(e.value_ptr.string);
            if (w.accepts(e.key_ptr.*, strip)) try shards.put(e.value_ptr.string, {});
        }
        var files = shards.keyIterator();
        if (shards.count() == 0) return error.MissingWeights;
        while (files.next()) |file| {
            std.debug.print("Loading {s}\n", .{file.*});
            try w.loadFile(io, try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, file.* }), "", strip);
        }
    }
    fn loadUnindexed(w: *Store, io: std.Io, dir: []const u8, strip: []const u8) !void {
        if (w.flash_drafts != null) {
            var checkpoint = try @import("safetensors.zig").Checkpoint.open(mx.allocator, io, dir);
            defer checkpoint.deinit();
            for (checkpoint.files.items) |file| try w.loadFile(io, file.path, "", strip);
            return;
        }
        var directory = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer directory.close(io);
        var it = directory.iterate();
        var files: std.ArrayList([]const u8) = .empty;
        defer {
            for (files.items) |name| mx.allocator.free(name);
            files.deinit(mx.allocator);
        }
        while (try it.next(io)) |entry| if (std.mem.startsWith(u8, entry.name, "model") and std.mem.endsWith(u8, entry.name, ".safetensors")) {
            const name = try mx.allocator.dupe(u8, entry.name);
            errdefer mx.allocator.free(name);
            try files.append(mx.allocator, name);
        };
        if (files.items.len == 0) return error.MissingWeights;
        std.mem.sort([]const u8, files.items, {}, struct {
            fn less(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.less);
        if (files.items.len > 1 or !std.mem.eql(u8, files.items[0], "model.safetensors")) for (files.items, 0..) |name, i| {
            if (name.len != 32 or !std.mem.eql(u8, name[11..15], "-of-")) return error.InvalidShardName;
            const number = try std.fmt.parseInt(usize, name[6..11], 10);
            const total = try std.fmt.parseInt(usize, name[15..20], 10);
            if (number != i + 1 or total != files.items.len) return error.IncompleteCheckpoint;
        };
        var buf: [4096]u8 = undefined;
        for (files.items) |name| {
            std.debug.print("Loading {s}\n", .{name});
            try w.loadFile(io, try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name }), "", strip);
        }
    }
    pub fn triple(w: *Store, name: []const u8) ![3]A {
        return .{ try w.field(name, "weight"), try w.field(name, "scales"), try w.field(name, "biases") };
    }
    pub fn dequant(w: *Store, s: *mx.Scope, name: []const u8) !A {
        const fmt = (try w.format(name)) orelse return w.field(name, "weight");
        const t = try w.triple(name);
        return dequantizeFormat(s, t, fmt);
    }
    pub fn embed(w: *Store, s: *mx.Scope, name: []const u8, ids: []const i32) !A {
        return w.embedArray(s, name, try s.ints(ids));
    }
    pub fn embedArray(w: *Store, s: *mx.Scope, name: []const u8, ix: A) !A {
        const fmt = (try w.format(name)) orelse return s.take(try w.field(name, "weight"), ix, 0);
        const t = try w.triple(name);
        return dequantizeFormat(s, .{ try s.take(t[0], ix, 0), try s.take(t[1], ix, 0), try s.take(t[2], ix, 0) }, fmt);
    }
    pub fn linear(w: *Store, k: *mx.Kernels, s: *mx.Scope, name: []const u8, x: A, exact: bool) !A {
        const fmt = (try w.format(name)) orelse {
            const weight = try w.field(name, "weight");
            return s.binary(mx.c.mlx_matmul, x, try s.transpose(weight, &.{ 1, 0 }));
        };
        const t = try w.triple(name);
        if (exact and w.flash_drafts != null) return @import("flash_ops.zig").project(k, s, x, .{ .arrays = t, .format = fmt }, mx.gpu_generation, 4);
        const n = mx.dim(t[0], 0);
        const dims = mx.dim(x, -1);
        const rows: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(dims)));
        if (exact and fmt.bits == 4 and fmt.group_size == 64 and mx.tensor_units and @mod(n, 32) == 0) {
            if (!w.dense.contains(name)) {
                var linear_ = try lanes.Linear.init(s, t[0], t[1], t[2]);
                errdefer linear_.deinit();
                const key = try mx.allocator.dupe(u8, name);
                errdefer mx.allocator.free(key);
                try w.dense.put(key, linear_);
            }
            return s.reshape(try w.dense.get(name).?.apply(k, s, .{ .x = x }), &.{ rows, n });
        }
        if (exact and fmt.bits == 4 and rows <= 16 and @mod(n, 8) == 0 and @mod(dims, 64) == 0) {
            return (try k.run(s, src.nemotron_rows_qmv, &.{ try s.reshape(x, &.{ rows, dims }), t[0], t[1], t[2] }, &.{ mx.ti("K", dims), mx.ti("N", n), mx.ti("GS", fmt.group_size), mx.ti("RPS", 4) }, .{ 32 * rows, @divExact(n, 4), 1 }, .{ 32 * rows, if (rows <= 8) 2 else 1, 1 }, &.{.{ .shape = &.{ rows, n } }}))[0];
        }
        var out = mx.c.mlx_array_new();
        const rc = mx.c.mlx_quantized_matmul(&out, x, t[0], t[1], t[2], true, mx.opt(fmt.group_size), mx.opt(fmt.bits), "affine", mx.stream);
        return s.result(rc, out);
    }
};
pub fn dequantize(s: *mx.Scope, t: [3]A, group: i32) !A {
    return dequantizeFormat(s, t, .{ .bits = 4, .group_size = group });
}
pub fn dequantizeFormat(s: *mx.Scope, t: [3]A, format: Quant.Spec) !A {
    try format.validate();
    var out = mx.c.mlx_array_new();
    const rc = mx.c.mlx_dequantize(&out, t[0], t[1], t[2], mx.opt(format.group_size), mx.opt(format.bits), "affine", mx.empty, .{ .value = mx.bf16, .has_value = true }, mx.stream);
    return s.result(rc, out);
}
pub fn norm(s: *mx.Scope, x: A, w: A, eps: f32) !A {
    var out = mx.c.mlx_array_new();
    const rc = mx.c.mlx_fast_rms_norm(&out, x, w, eps, mx.stream);
    return s.result(rc, out);
}
