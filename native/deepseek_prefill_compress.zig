const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const ds = @import("deepseek.zig");

pub fn forward(m: *ds.Model, s: *mx.Scope, layer: usize, x: mx.Array, cache: *ds.Cache, past: i32) !mx.Array {
    const g = m.config.value;
    const ratio = m.layerRatio(layer);
    if (ratio != 4 and ratio != 128) return error.UnsupportedCompressionRatio;
    if (mx.shape(x).len != 2 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != g.hidden_size) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    if (rows < 1 or rows > 2048 or past < 0 or past > 1048576 - rows) return error.ContextLimitExceeded;
    const limit = ratio * @as(i32, if (ratio == 4) 2 else 1);
    const width = if (ratio == 4) 4 * (g.head_dim + g.index_head_dim) else 2 * g.head_dim;
    const kept = @min(past, limit);
    if ((cache.proj.ctx != null) != (kept > 0)) return error.InvalidTensorShape;
    if (kept > 0 and (!std.mem.eql(i32, mx.shape(cache.proj), &.{ kept, width }) or mx.dtype(cache.proj) != mx.f32t)) return error.InvalidTensorShape;
    const first = @divTrunc(past, ratio);
    inline for (.{ "pool", "ipool" }) |key| {
        const value = @field(cache, key);
        const present = first > 0 and (ratio == 4 or comptime std.mem.eql(u8, key, "pool"));
        if ((value.ctx != null) != present) return error.InvalidTensorShape;
        const dims = if (comptime std.mem.eql(u8, key, "pool")) g.head_dim else g.index_head_dim;
        if (present and (!std.mem.eql(i32, mx.shape(value), &.{ first, dims }) or mx.dtype(value) != mx.bf16)) return error.InvalidTensorShape;
    }
    const projected = try ds.Model.stock(s, try s.cast(x, mx.f32t), try m.triple(layer, "attn.cproj"));
    const span = if (kept > 0) try s.cat(&.{ cache.proj, projected }, 0) else projected;
    var next = cache.*;
    const count = @divTrunc(past + rows, ratio) - first;
    if (count > 0) {
        const pool = try m.poolBlocks(s, layer, span, past - kept, first, count, false);
        next.pool = if (first > 0) try s.cat(&.{ cache.pool, pool }, 0) else pool;
        if (ratio == 4) {
            const ipool = try m.poolBlocks(s, layer, span, past - kept, first, count, true);
            next.ipool = if (first > 0) try s.cat(&.{ cache.ipool, ipool }, 0) else ipool;
        }
    }
    next.proj = try s.contiguous(try s.slice(span, 0, @max(0, mx.dim(span, 0) - limit), mx.dim(span, 0)));
    cache.* = next;
    return projected;
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/compress.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, past: i32 };
    const Group = struct { checkpoint: []const u8, cases: []const Case };
    const groups = try std.json.parseFromSlice([]const Group, mx.allocator, bytes, .{});
    defer groups.deinit();
    if (groups.value.len == 0) return error.EmptyFixtures;
    var count: usize = 0;
    for (groups.value) |group| {
        var m = try ds.Model.init(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, group.checkpoint }));
        defer m.deinit();
        var cache = ds.Cache{};
        defer cache.deinit();
        if (group.cases.len == 0) return error.EmptyFixtures;
        for (group.cases) |case| {
            errdefer std.debug.print("DeepSeek prefill compressor fixture failed: {s}/{s}\n", .{ group.checkpoint, case.name });
            var store = cp.Store.init(16);
            defer store.deinit();
            try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
            var s = mx.Scope{};
            defer s.deinit();
            var next = cache;
            const projected = try forward(&m, &s, 0, try store.get("input"), &next, case.past);
            try @import("variant_checks.zig").equalBits(&s, projected, try store.get("projection"));
            inline for (.{ "proj", "pool", "ipool" }) |key| {
                errdefer std.debug.print("Mismatch in {s}\n", .{key});
                const actual = @field(next, key);
                try std.testing.expectEqual(store.has(key), actual.ctx != null);
                if (actual.ctx != null) try @import("variant_checks.zig").equalBits(&s, actual, try store.get(key));
            }
            const owned = try next.clone();
            cache.deinit();
            cache = owned;
            count += 1;
        }
    }
    std.debug.print("PASS: {d} DeepSeek batched compressor cases, ratio-4 overlap, ratio-128 blocks, projection tails and pooled caches\n", .{count});
}
