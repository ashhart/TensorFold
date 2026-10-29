//! Geometry-aware launches reuse Flash Next's maintained QSA kernels, without changing their arithmetic or images.
const std = @import("std");
const cfg = @import("config.zig");
const wo = @import("core").weight_only;
pub const Geometry = struct {
    heads: usize,
    kv: usize,
    index_heads: usize,
    index_dim: usize,
    rotary: usize,
    ratio: usize,
    top: usize,
    pub fn of(c: *const cfg.Config, p: *cfg.Problem) !Geometry {
        if (c.head_dim != 256) return p.reject("head_dim");
        if (c.heads / c.kv_heads > 16) return p.reject("num_key_value_heads");
        if (c.index_dim != 128) return p.reject("indexer_head_dim");
        if (c.index_heads > 8) return p.reject("indexer_n_heads");
        if (c.index_ratio > 1024) return p.reject("indexer_compress_ratio");
        const out: Geometry = .{ .heads = c.heads, .kv = c.kv_heads, .index_heads = c.index_heads, .index_dim = c.index_dim, .rotary = c.rotary, .ratio = c.index_ratio, .top = c.index_budget / c.index_ratio };
        try out.validate();
        return out;
    }
    pub fn validate(g: Geometry) !void {
        if (g.heads == 0 or g.kv == 0 or g.heads % g.kv != 0 or g.heads / g.kv > 16 or g.index_heads == 0 or g.index_heads > 8 or g.index_dim != 128 or g.rotary > 128 or g.rotary % 2 != 0 or g.ratio == 0 or g.ratio > 1024 or g.top == 0) return error.Qwen4AttentionShape;
        if (g.heads > std.math.maxInt(c_int) / 512 or g.kv > std.math.maxInt(c_int) / 256) return error.Qwen4AttentionShape;
        _ = try std.math.add(usize, try std.math.mul(usize, g.top, g.ratio), g.ratio);
    }
    pub fn projectionWidth(g: Geometry) usize {
        return 2 * (g.heads + g.kv) * 256 + (g.index_heads + 1) * g.index_dim;
    }
    pub fn ids(g: Geometry) !usize {
        return std.math.add(usize, try std.math.mul(usize, g.top, g.ratio), g.ratio);
    }
};
pub const Call = struct { rows: usize, start: usize, capacity: usize, eps: f32, gate: bool = true };
const shared: u32 = (16 + 32) * (256 + 8) * 2 + 16 * (32 + 8) * 2;
pub const Sizes = struct { bytes: [23]usize, score_stride: usize, chunks: usize };
fn mul(values: []const usize) !usize {
    var n: usize = 1;
    for (values) |v| n = try std.math.mul(usize, n, v);
    return n;
}
pub fn sizes(g: Geometry, p: Call) !Sizes {
    try g.validate();
    if (p.rows == 0 or p.rows > 128 or p.capacity == 0 or p.start > p.capacity or p.rows > p.capacity - p.start or p.capacity > std.math.maxInt(c_int) or !std.math.isFinite(p.eps) or p.eps <= 0) return error.Qwen4AttentionShape;
    const stride = @max(@as(usize, 1), p.capacity / g.ratio);
    if (try mul(&.{ p.rows, g.projectionWidth() }) > std.math.maxInt(c_int)) return error.Qwen4AttentionShape;
    if (try mul(&.{ p.rows, g.heads, 256 }) > std.math.maxInt(c_int)) return error.Qwen4AttentionShape;
    const limit = try std.math.add(usize, try std.math.mul(usize, g.top, g.ratio), g.ratio - 1);
    const chunks = (try std.math.add(usize, @min(p.capacity, limit), 511)) / 512;
    const idw = try g.ids();
    if (idw > std.math.maxInt(c_int) or g.projectionWidth() > std.math.maxInt(c_int) or g.top > std.math.maxInt(c_int)) return error.Qwen4AttentionShape;
    const query = try mul(&.{ p.rows, g.heads, 256, 2 });
    const key = try mul(&.{ p.capacity, g.kv, 256, 2 });
    const part = try mul(&.{ p.rows, g.heads, chunks, 4 });
    return .{ .score_stride = stride, .chunks = chunks, .bytes = .{
        try mul(&.{ p.rows, g.projectionWidth(), 2 }), 4,                                    1024,       1024,                                                 g.index_dim * 4,                           g.index_dim * 4,                                                          @max(@as(usize, 4), g.rotary * 2),
        query,                                         key,                                  key,        try mul(&.{ p.rows, g.index_heads, g.index_dim, 2 }), try mul(&.{ p.capacity, g.index_dim, 2 }), @max(@as(usize, 2), try mul(&.{ p.capacity / g.ratio, g.index_dim, 2 })), try mul(&.{ p.rows, stride, 4 }),
        try mul(&.{ p.rows, idw, 4 }),                 p.rows * 4,                           p.rows * 4, try std.math.mul(usize, part, 256),                   part,                                      part,                                                                     query,
        query,                                         try mul(&.{ p.rows, g.heads, 8, 4 }),
    } };
}
pub fn Kernels(comptime cuda: type) type {
    return struct {
        const Self = @This();
        pub const View = struct { buffer: *const cuda.DeviceBuffer, offset: usize = 0, len: usize };
        pub const names = [_][:0]const u8{ "fn_attn_prep", "fn_qsa_pool", "fn_qsa_scores", "fn_qsa_select", "fn_attn_chunks_one", "fn_attn_merge", "fn_attn_gate" };
        attention: cuda.Module,
        batch: cuda.Module,
        f: [names.len]cuda.Function = undefined,
        pub fn load(d: *const cuda.Driver, attention: []const u8, batch: []const u8) !Self {
            var a = try cuda.Module.load(d, attention);
            errdefer a.unload();
            var b = try cuda.Module.load(d, batch);
            errdefer b.unload();
            var out: Self = .{ .attention = a, .batch = b };
            const threads = [_]usize{ 256, 128, 256, 1024, 128, 256, 256 };
            for (names, threads, 0..) |name, n, i| {
                out.f[i] = try (if (i == 3) b else a).function(name);
                if (try out.f[i].attribute(.max_threads_per_block) < n) return error.Qwen4AttentionThreads;
            }
            try out.f[4].allowDynamicShared(shared);
            return out;
        }
        pub fn close(k: *Self, s: cuda.Stream) !void {
            if (s.d != k.attention.d) return error.Qwen4AttentionContext;
            try s.synchronize();
            k.batch.unload();
            k.attention.unload();
            k.* = undefined;
        }
        fn go(k: *const Self, s: cuda.Stream, index: usize, grid: [3]usize, threads: u32, args: *cuda.Args) !void {
            try cuda.launch.launch(k.f[index], .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = if (index == 4) shared else 0 }, s, args);
        }
        fn ints(a: *cuda.Args, values: []const usize) void {
            for (values) |v| a.add(@as(c_int, @intCast(v)));
        }
        fn ptrs(a: *cuda.Args, pointers: []const u64) void {
            for (pointers) |v| a.add(v);
        }
        pub fn encode(k: *const Self, s: cuda.Stream, g: Geometry, p: Call, views: [23]View) !void {
            if (s.d != k.attention.d) return error.Qwen4AttentionContext;
            const z = try sizes(g, p);
            var pointers: [23]u64 = undefined;
            var ranges: [23]wo.Range = undefined;
            for (views, z.bytes, 0..) |v, need, i| {
                if (v.buffer.d != s.d) return error.Qwen4AttentionContext;
                if (v.offset > v.buffer.len or v.len > v.buffer.len - v.offset or v.len < need) return error.Qwen4AttentionBuffer;
                const ptr = try std.math.add(usize, @intCast(v.buffer.ptr), v.offset);
                const align_to: usize = if (i >= 7 and i <= 12) 16 else if (i == 17) 8 else if (i == 0 or i == 20 or i == 21) 2 else 4;
                if (ptr == 0 or ptr % align_to != 0) return error.Qwen4AttentionAlignment;
                _ = try std.math.add(usize, ptr, v.len);
                ranges[i] = .{ .address = ptr, .len = v.len };
                pointers[i] = ptr;
            }
            for (7..23) |i| for (0..i) |j| {
                const x = ranges[i];
                const y = ranges[j];
                if (if (x.address <= y.address) y.address - x.address < x.len else x.address - y.address < y.len) return error.Qwen4AttentionAlias;
            };
            var a: cuda.Args = .{};
            ptrs(&a, &.{ pointers[0], pointers[1], pointers[2], pointers[3], pointers[4], pointers[6], pointers[7], pointers[8], pointers[9], pointers[10], pointers[11] });
            a.add(p.eps);
            ints(&a, &.{ g.projectionWidth(), g.heads, g.kv, g.index_heads, g.index_dim, g.rotary / 2 });
            try k.go(s, 0, .{ p.rows, g.heads + g.kv + g.index_heads + 1, 1 }, 256, &a);
            a = .{};
            ptrs(&a, &.{ pointers[11], pointers[12], pointers[1], pointers[5], pointers[6] });
            a.add(p.eps);
            ints(&a, &.{ p.rows, g.index_dim, g.rotary / 2, g.ratio });
            try k.go(s, 1, .{ p.rows / g.ratio + 2, 1, 1 }, 128, &a);
            a = .{};
            ptrs(&a, &.{ pointers[10], pointers[12], pointers[1], pointers[13] });
            ints(&a, &.{ z.score_stride, g.index_heads, g.ratio, g.top });
            try k.go(s, 2, .{ p.rows, (z.score_stride + 63) / 64, 1 }, 256, &a);
            a = .{};
            ptrs(&a, &.{ pointers[13], pointers[1], pointers[14], pointers[15], pointers[16] });
            ints(&a, &.{ z.score_stride, try g.ids(), g.top, g.ratio });
            try k.go(s, 3, .{ p.rows, 1, 1 }, 1024, &a);
            a = .{};
            ptrs(&a, &.{ pointers[7], pointers[8], pointers[9], pointers[1], pointers[17], pointers[18], pointers[19], pointers[14], pointers[15], pointers[16] });
            ints(&a, &.{ g.heads, g.kv, g.heads / g.kv, 512, z.chunks });
            a.add(@as(f32, 0.0625));
            ints(&a, &.{ try g.ids(), 1 });
            try k.go(s, 4, .{ p.rows, g.kv, z.chunks }, 128, &a);
            a = .{};
            ptrs(&a, &.{ pointers[17], pointers[18], pointers[19], pointers[1], pointers[20], pointers[15], pointers[16] });
            ints(&a, &.{ g.heads, g.kv, g.heads / g.kv, 512, z.chunks, 1 });
            try k.go(s, 5, .{ p.rows, g.heads, 1 }, 256, &a);
            if (!p.gate) return;
            a = .{};
            ptrs(&a, &.{ pointers[20], pointers[0], pointers[21], pointers[22] });
            ints(&a, &.{ g.projectionWidth(), g.heads });
            try k.go(s, 6, .{ p.rows, (g.heads * 256 + 255) / 256, 1 }, 256, &a);
        }
    };
}
