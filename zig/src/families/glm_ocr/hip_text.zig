//! The four-norm GLM decoder, with per-stream BF16 KV ownership and caller-supplied image/text rows and positions.
const std = @import("std");
const hip = @import("hip");
const c = @import("config.zig");
const schema = @import("schema.zig");
const Weights = @import("hip_weights.zig").Weights;
const Ops = @import("hip_ops.zig").Ops;
const Observer = @import("hip_vision.zig").Observer;
const attention = hip.attention_aotriton;

pub const Cache = struct {
    stream: hip.Stream,
    buffer: hip.DeviceBuffer,
    capacity: usize,
    len: usize = 0,

    pub fn init(stream: hip.Stream, capacity: usize, budget: usize) !Cache {
        if (capacity == 0 or capacity > c.context) return error.Invalid;
        const bytes = capacity * c.layers * c.kv * 2 * 2;
        if (bytes > budget) return error.GlmOcrCacheDoesNotFit;
        return .{ .stream = stream, .buffer = try .alloc(stream.r, bytes), .capacity = capacity };
    }
    pub fn deinit(cache: *Cache) void {
        cache.stream.synchronize() catch {};
        cache.buffer.free();
        cache.* = undefined;
    }
    pub fn truncate(cache: *Cache, len: usize) !void {
        if (len > cache.len) return error.Invalid;
        cache.len = len;
    }
    /// A failed device copy leaves the destination empty; invalid spans leave it unchanged.
    pub fn copyPrefix(cache: *Cache, source: *const Cache, len: usize) !void {
        if (!cache.admitted(cache.stream) or !source.admitted(cache.stream) or len > source.len or len > cache.capacity or
            cache.stream.r != source.stream.r or cache.stream.handle != source.stream.handle) return error.Invalid;
        if (cache == source) {
            cache.len = len;
            return;
        }
        if (cache.buffer.ptr == source.buffer.ptr) return error.Invalid;
        cache.len = 0;
        if (len == 0) return;
        for (0..c.layers) |layer| {
            const key_view: hip.DeviceBuffer = .{ .r = cache.stream.r, .ptr = @ptrFromInt(try cache.key(layer)), .len = cache.capacity * c.kv * 2 };
            const value_view: hip.DeviceBuffer = .{ .r = cache.stream.r, .ptr = @ptrFromInt(try cache.value(layer)), .len = cache.capacity * c.kv * 2 };
            try key_view.copyFromAsync(0, try source.key(layer), len * c.kv * 2, cache.stream);
            try value_view.copyFromAsync(0, try source.value(layer), len * c.kv * 2, cache.stream);
        }
        try cache.stream.synchronize();
        cache.len = len;
    }
    fn admitted(cache: *const Cache, stream: hip.Stream) bool {
        if (cache.capacity == 0 or cache.capacity > c.context or cache.len > cache.capacity or
            cache.buffer.ptr == null or cache.buffer.len != cache.capacity * c.layers * c.kv * 4 or
            cache.buffer.r != stream.r or cache.stream.r != stream.r or cache.stream.handle != stream.handle) return false;
        _ = std.math.add(u64, cache.buffer.base(), cache.buffer.len) catch return false;
        return true;
    }
    fn key(cache: *const Cache, layer: usize) !u64 {
        return cache.buffer.address(layer * cache.capacity * c.kv * 2);
    }
    fn value(cache: *const Cache, layer: usize) !u64 {
        return cache.buffer.address((c.layers + layer) * cache.capacity * c.kv * 2);
    }
};

test "GLM-OCR HIP decoder: cache and text budgets refuse before device allocation" {
    const r: hip.Runtime = undefined;
    const stream: hip.Stream = .{ .r = &r, .handle = null };
    const w: Weights = undefined;
    const ops: Ops = undefined;
    const blas: hip.blas.Blas = undefined;
    const flash: attention.Library = undefined;
    var graph: Graph = undefined;
    graph.logit_rows = 32;
    try std.testing.expectError(error.Invalid, graph.logitsRows(0, 0));
    try std.testing.expectError(error.Invalid, graph.logitsRows(0, 33));
    try std.testing.expectError(error.Invalid, Cache.init(stream, 0, 0));
    try std.testing.expectError(error.Invalid, Cache.init(stream, c.context + 1, 0));
    try std.testing.expectError(error.GlmOcrCacheDoesNotFit, Cache.init(stream, 512, 0));
    try std.testing.expectError(error.Invalid, Graph.init(stream, &w, &ops, &blas, &flash, 0, 0));
    try std.testing.expectError(error.GlmOcrTextDoesNotFit, Graph.init(stream, &w, &ops, &blas, &flash, 512, 0));
    var cache: Cache = undefined;
    cache.len = 5;
    try std.testing.expectError(error.Invalid, cache.truncate(6));
    try cache.truncate(3);
    try std.testing.expectEqual(@as(usize, 3), cache.len);
}

test "GLM-OCR HIP decoder: all row segments are validated before a shared forward" {
    const r: hip.Runtime = undefined;
    const stream: hip.Stream = .{ .r = &r, .handle = null };
    var graph: Graph = undefined;
    graph.stream = stream;
    graph.max_rows = 8;
    const bytes = 8 * c.layers * c.kv * 4;
    var first: Cache = .{ .stream = stream, .buffer = .{ .r = &r, .ptr = @ptrFromInt(0x3000), .len = bytes }, .capacity = 8, .len = 3 };
    var second: Cache = .{ .stream = stream, .buffer = .{ .r = &r, .ptr = @ptrFromInt(0x4000), .len = bytes }, .capacity = 8 };
    const one: Segment = .{ .cache = &first, .embeddings = 0x1000, .positions = 0x100000, .rows = 2 };
    const two: Segment = .{ .cache = &second, .embeddings = 0x2000, .positions = 0x200000, .rows = 3 };
    try std.testing.expectEqual(@as(usize, 5), try graph.validate(&.{ one, two }));
    try std.testing.expectError(error.Invalid, graph.validate(&.{}));
    try std.testing.expectError(error.Invalid, graph.validate(&.{ one, one }));
    var bad = two;
    bad.rows = 7;
    try std.testing.expectError(error.Invalid, graph.validate(&.{ one, bad }));
    var resumed = one;
    resumed.rows = 6;
    resumed.prefix_rows = 3;
    try std.testing.expectEqual(@as(usize, 6), try graph.validate(&.{resumed}));
    resumed.phase = .decode;
    try std.testing.expectError(error.Invalid, graph.validate(&.{resumed}));
    resumed.phase = .prompt;
    resumed.prefix_rows = 2;
    try std.testing.expectError(error.Invalid, graph.validate(&.{resumed}));
    resumed.prefix_rows = resumed.rows;
    try std.testing.expectError(error.Invalid, graph.validate(&.{resumed}));
    var alias = first;
    bad = two;
    bad.cache = &alias;
    try std.testing.expectError(error.Invalid, graph.validate(&.{ one, bad }));
    bad = two;
    bad.positions = 0;
    try std.testing.expectError(error.Invalid, graph.validate(&.{ one, bad }));
    bad = two;
    bad.embeddings = std.math.maxInt(u64) - 1;
    try std.testing.expectError(error.Invalid, graph.validate(&.{ one, bad }));
    try std.testing.expectEqual(@as(usize, 3), first.len);
    try std.testing.expectEqual(@as(usize, 0), second.len);
    try std.testing.expectError(error.Invalid, first.copyPrefix(&second, 1));
    try std.testing.expectError(error.Invalid, first.copyPrefix(&alias, 2));
    var damaged = second;
    damaged.buffer.len -= 2;
    try std.testing.expectError(error.Invalid, first.copyPrefix(&damaged, 0));
    try std.testing.expectEqual(@as(usize, 3), first.len);
    try first.copyPrefix(&first, 2);
    try std.testing.expectEqual(@as(usize, 2), first.len);
    try first.copyPrefix(&second, 0);
    try std.testing.expectEqual(@as(usize, 0), first.len);
}

pub fn inverseFrequencies() [64]f32 {
    var inverse: [64]f32 = undefined;
    for (&inverse, 0..) |*v, i| {
        const power: f32 = @floatCast(std.math.pow(f64, 10000, @as(f64, @floatFromInt(i)) / 64));
        v.* = 1 / power;
    }
    return inverse;
}

pub const Segment = struct {
    cache: *Cache,
    embeddings: u64,
    positions: u64,
    rows: usize,
    phase: Ops.Phase = .prompt,
    /// Keep this prefix's K/V while running the original full prompt row shape.
    prefix_rows: usize = 0,
};

pub const Graph = struct {
    stream: hip.Stream,
    weights: *const Weights,
    ops: *const Ops,
    blas: *const hip.blas.Blas,
    flash: *const attention.Library,
    fixed: hip.blas_fixed.Fixed,
    buffer: hip.DeviceBuffer,
    slots: [9]u64,
    inverse: u64,
    lse: u64,
    counter: u64,
    logits: u64,
    logit_rows: usize,
    positions: u64,
    max_rows: usize,

    const Layout = struct { offsets: [9]usize, lse: usize, positions: usize, inverse: usize, counter: usize, logits: usize, logit_rows: usize, bytes: usize };
    fn layout(max_rows: usize) !Layout {
        if (max_rows == 0 or max_rows > c.context) return error.Invalid;
        const widths = [_]usize{ c.hidden, c.hidden, c.query, c.kv, c.kv, c.query, c.hidden, c.ffn * 2, c.ffn };
        var offsets: [9]usize = undefined;
        var bytes: usize = 0;
        for (widths, &offsets) |width, *offset| {
            offset.* = bytes;
            bytes += max_rows * width * 2;
        }
        const lse_offset = bytes;
        bytes += max_rows * c.heads * 4;
        const positions_offset = bytes;
        bytes += max_rows * 3 * 4;
        bytes = std.mem.alignForward(usize, bytes, 256);
        const inverse_offset = bytes;
        const counter_offset = bytes + 256;
        const logits_offset = bytes + 512;
        const logit_rows: usize = @min(max_rows, 32);
        bytes += 512 + logit_rows * c.vocab * 2;
        return .{ .offsets = offsets, .lse = lse_offset, .positions = positions_offset, .inverse = inverse_offset, .counter = counter_offset, .logits = logits_offset, .logit_rows = logit_rows, .bytes = bytes };
    }
    pub fn memoryBytes(max_rows: usize) !usize {
        return (try layout(max_rows)).bytes;
    }

    pub fn init(stream: hip.Stream, weights: *const Weights, ops: *const Ops, blas: *const hip.blas.Blas, flash: *const attention.Library, max_rows: usize, budget: usize) !Graph {
        const p = try layout(max_rows);
        if (p.bytes > budget) return error.GlmOcrTextDoesNotFit;
        if (weights.buffer.r != stream.r or ops.module.r != stream.r or blas.stream.r != stream.r or blas.stream.handle != stream.handle) return error.Invalid;
        var fixed = try hip.blas_fixed.Fixed.open(stream, &.{ .{ .n = c.query, .k = c.hidden }, .{ .n = c.kv, .k = c.hidden }, .{ .n = c.hidden, .k = c.query }, .{ .n = c.ffn * 2, .k = c.hidden }, .{ .n = c.hidden, .k = c.ffn }, .{ .n = c.vocab, .k = c.hidden } });
        errdefer fixed.close();
        var buffer = try hip.DeviceBuffer.alloc(stream.r, p.bytes);
        errdefer buffer.free();
        const inverse = inverseFrequencies();
        try buffer.upload(p.inverse, std.mem.asBytes(&inverse));
        var slots: [9]u64 = undefined;
        for (&slots, p.offsets) |*slot, offset| slot.* = try buffer.address(offset);
        return .{ .stream = stream, .weights = weights, .ops = ops, .blas = blas, .flash = flash, .fixed = fixed, .buffer = buffer, .slots = slots, .inverse = try buffer.address(p.inverse), .lse = try buffer.address(p.lse), .counter = try buffer.address(p.counter), .logits = try buffer.address(p.logits), .logit_rows = p.logit_rows, .positions = try buffer.address(p.positions), .max_rows = max_rows };
    }
    pub fn deinit(graph: *Graph) void {
        graph.stream.synchronize() catch {};
        graph.fixed.close();
        graph.buffer.free();
        graph.* = undefined;
    }
    fn weight(graph: *const Graph, stem: []const u8, name: []const u8) !u64 {
        var buf: [192]u8 = undefined;
        return graph.weights.get(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ stem, name }));
    }
    fn norm(graph: *const Graph, segments: []const Segment, stem: []const u8, name: []const u8, x: u64, out: u64, rows: usize) !void {
        const w = try graph.weight(stem, name);
        const phase = segments[0].phase;
        const same = for (segments) |s| {
            if (s.phase != phase) break false;
        } else true;
        if (same) return graph.ops.normText(graph.stream, x, w, out, rows, phase);
        var offset: usize = 0;
        for (segments) |s| {
            try graph.ops.normText(graph.stream, x + offset, w, out + offset, s.rows, s.phase);
            offset += s.rows * c.hidden * 2;
        }
    }
    fn linear(graph: *const Graph, segments: []const Segment, stem: []const u8, name: []const u8, x: u64, out: u64, rows: usize, input: usize, output: usize) !void {
        const w = try graph.weight(stem, name);
        const phase = segments[0].phase;
        const same = for (segments) |s| {
            if (s.phase != phase) break false;
        } else true;
        if (same) {
            if (phase == .decode) return graph.fixed.project(x, w, out, rows, output, input);
            return graph.blas.project(x, w, out, rows, output, input);
        }
        var offset: usize = 0;
        for (segments) |s| {
            if (s.phase == .decode) try graph.fixed.project(x + offset * input * 2, w, out + offset * output * 2, s.rows, output, input) else try graph.blas.project(x + offset * input * 2, w, out + offset * output * 2, s.rows, output, input);
            offset += s.rows;
        }
    }

    fn emitStage(observer: ?Observer, layer: usize, suffix: []const u8, address: u64, values: usize) !void {
        if (observer == null) return;
        var buf: [96]u8 = undefined;
        const name = if (suffix.len == 0) try std.fmt.bufPrint(&buf, "text-block{d}", .{layer}) else try std.fmt.bufPrint(&buf, "text-block{d}-{s}", .{ layer, suffix });
        try Observer.emit(observer, name, address, values);
    }

    /// Append rows after the cache's committed prefix; a failed pass never advances the visible cache length.
    pub fn forward(graph: *const Graph, cache: *Cache, embeddings: u64, positions: u64, rows: usize, observer: ?Observer) !u64 {
        return graph.forwardMany(&.{.{ .cache = cache, .embeddings = embeddings, .positions = positions, .rows = rows, .phase = if (cache.len == 0) .prompt else .decode }}, observer);
    }

    pub fn forwardMany(graph: *const Graph, segments: []const Segment, observer: ?Observer) !u64 {
        const rows = try graph.validate(segments);
        const x, const normed, const q, const k, const val, const attended, const work, const gate_up, const gate = graph.slots;
        const input: hip.DeviceBuffer = .{ .r = graph.stream.r, .ptr = @ptrFromInt(x), .len = rows * c.hidden * 2 };
        const positions: hip.DeviceBuffer = .{ .r = graph.stream.r, .ptr = @ptrFromInt(graph.positions), .len = rows * 3 * 4 };
        var offset: usize = 0;
        for (segments) |segment| {
            try input.copyFromAsync(offset * c.hidden * 2, segment.embeddings, segment.rows * c.hidden * 2, graph.stream);
            try positions.copyFromAsync(offset * 3 * 4, segment.positions, segment.rows * 3 * 4, graph.stream);
            offset += segment.rows;
        }
        for (0..c.layers) |layer| {
            var stem_buf: [96]u8 = undefined;
            const stem = try std.fmt.bufPrint(&stem_buf, schema.text ++ "layers.{d}.", .{layer});
            try graph.norm(segments, stem, "input_layernorm", x, normed, rows);
            try emitStage(observer, layer, "norm1", normed, rows * c.hidden);
            try graph.linear(segments, stem, "self_attn.q_proj", normed, q, rows, c.hidden, c.query);
            try graph.linear(segments, stem, "self_attn.k_proj", normed, k, rows, c.hidden, c.kv);
            try graph.linear(segments, stem, "self_attn.v_proj", normed, val, rows, c.hidden, c.kv);
            {
                try emitStage(observer, layer, "q", q, rows * c.query);
                try emitStage(observer, layer, "k", k, rows * c.kv);
                try emitStage(observer, layer, "v", val, rows * c.kv);
            }
            try graph.ops.textRope(graph.stream, q, graph.inverse, graph.positions, rows, c.heads);
            try graph.ops.textRope(graph.stream, k, graph.inverse, graph.positions, rows, c.kv_heads);
            {
                try emitStage(observer, layer, "qrope", q, rows * c.query);
                try emitStage(observer, layer, "krope", k, rows * c.kv);
            }
            offset = 0;
            for (segments) |segment| {
                const cache = segment.cache;
                const key = try cache.key(layer);
                const value = try cache.value(layer);
                const key_view: hip.DeviceBuffer = .{ .r = graph.stream.r, .ptr = @ptrFromInt(key), .len = cache.capacity * c.kv * 2 };
                const value_view: hip.DeviceBuffer = .{ .r = graph.stream.r, .ptr = @ptrFromInt(value), .len = cache.capacity * c.kv * 2 };
                const skip = segment.prefix_rows;
                const added = segment.rows - skip;
                try key_view.copyFromAsync(cache.len * c.kv * 2, k + (offset + skip) * c.kv * 2, added * c.kv * 2, graph.stream);
                try value_view.copyFromAsync(cache.len * c.kv * 2, val + (offset + skip) * c.kv * 2, added * c.kv * 2, graph.stream);
                if (segment.phase == .decode) {
                    // Shared products retain one-token attention arithmetic at every verification row.
                    for (0..segment.rows) |i| {
                        const keys = cache.len + i + 1;
                        const at = offset + i;
                        try graph.flash.forward(graph.stream, attention.rows(q + at * c.query * 2, c.heads, 1, c.head_dim), attention.rows(key, c.kv_heads, keys, c.head_dim), attention.rows(value, c.kv_heads, keys, c.head_dim), attention.rows(attended + at * c.query * 2, c.heads, 1, c.head_dim), graph.lse + at * c.heads * 4, graph.counter, false);
                    }
                } else {
                    const keys = cache.len + added;
                    try graph.flash.forward(graph.stream, attention.rows(q + offset * c.query * 2, c.heads, segment.rows, c.head_dim), attention.rows(key, c.kv_heads, keys, c.head_dim), attention.rows(value, c.kv_heads, keys, c.head_dim), attention.rows(attended + offset * c.query * 2, c.heads, segment.rows, c.head_dim), graph.lse + offset * c.heads * 4, graph.counter, segment.rows > 1);
                }
                offset += segment.rows;
            }
            try emitStage(observer, layer, "attended", attended, rows * c.query);
            try graph.linear(segments, stem, "self_attn.o_proj", attended, work, rows, c.query, c.hidden);
            try emitStage(observer, layer, "proj", work, rows * c.hidden);
            try graph.norm(segments, stem, "post_self_attn_layernorm", work, normed, rows);
            try emitStage(observer, layer, "attn-norm", normed, rows * c.hidden);
            try graph.ops.add(graph.stream, x, normed, x, rows * c.hidden);
            try graph.norm(segments, stem, "post_attention_layernorm", x, normed, rows);
            try emitStage(observer, layer, "norm2", normed, rows * c.hidden);
            try graph.linear(segments, stem, "mlp.gate_up_proj", normed, gate_up, rows, c.hidden, c.ffn * 2);
            try emitStage(observer, layer, "gate-up", gate_up, rows * c.ffn * 2);
            try graph.ops.siluMulPacked(graph.stream, gate_up, gate, rows);
            try graph.linear(segments, stem, "mlp.down_proj", gate, work, rows, c.ffn, c.hidden);
            try emitStage(observer, layer, "down", work, rows * c.hidden);
            try graph.norm(segments, stem, "post_mlp_layernorm", work, normed, rows);
            try emitStage(observer, layer, "mlp-norm", normed, rows * c.hidden);
            try graph.ops.add(graph.stream, x, normed, x, rows * c.hidden);
            try emitStage(observer, layer, "", x, rows * c.hidden);
        }
        try Observer.emit(observer, "text-last", x, rows * c.hidden);
        try graph.norm(segments, schema.text, "norm", x, normed, rows);
        try Observer.emit(observer, "text-norm", normed, rows * c.hidden);
        try graph.stream.synchronize();
        for (segments) |segment| segment.cache.len += segment.rows - segment.prefix_rows;
        return normed;
    }

    fn validate(graph: *const Graph, segments: []const Segment) !usize {
        if (segments.len == 0 or segments.len > 32) return error.Invalid;
        var rows: usize = 0;
        for (segments, 0..) |segment, i| {
            const cache = segment.cache;
            if (rows > graph.max_rows or segment.rows == 0 or segment.rows > graph.max_rows - rows or
                !cache.admitted(graph.stream) or segment.prefix_rows >= segment.rows or
                (segment.prefix_rows != 0 and (segment.phase != .prompt or segment.prefix_rows != cache.len)) or
                segment.rows - segment.prefix_rows > cache.capacity - cache.len or
                segment.embeddings == 0 or segment.embeddings % 2 != 0 or segment.positions == 0 or segment.positions % 4 != 0) return error.Invalid;
            _ = std.math.add(u64, segment.embeddings, segment.rows * c.hidden * 2) catch return error.Invalid;
            _ = std.math.add(u64, segment.positions, segment.rows * 3 * 4) catch return error.Invalid;
            for (segments[0..i]) |previous| if (previous.cache == cache or previous.cache.buffer.ptr == cache.buffer.ptr) return error.Invalid;
            rows += segment.rows;
        }
        return rows;
    }

    /// The caller consumes these logits before another forward reuses this scratch.
    pub fn lastLogits(graph: *const Graph, normed: u64, rows: usize) !u64 {
        if (rows == 0 or rows > graph.max_rows) return error.Invalid;
        try graph.blas.project(normed + (rows - 1) * c.hidden * 2, try graph.weights.get("lm_head.weight"), graph.logits, 1, c.vocab, c.hidden);
        return graph.logits;
    }

    /// All verification logits share the retained single-row algorithm and stay on the GPU for the shared drawer.
    pub fn logitsRows(graph: *const Graph, normed: u64, rows: usize) !u64 {
        if (rows == 0 or rows > graph.logit_rows) return error.Invalid;
        try graph.fixed.project(normed, try graph.weights.get("lm_head.weight"), graph.logits, rows, c.vocab, c.hidden);
        return graph.logits;
    }
};
