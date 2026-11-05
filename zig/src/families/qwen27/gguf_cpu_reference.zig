//! Diagnostic binary64 forward over independently decoded GGUF weights, retaining the family's BF16 activations.
const std = @import("std");
pub const source = @import("gguf_cpu_weights.zig");
const mixers = @import("gguf_cpu_mixers.zig");
const A = std.mem.Allocator;
const bf = source.rounded;
pub const Session = struct {
    weights: *const source.Weights,
    layers: []mixers.State,
    position: usize = 0,
    capacity: usize,
    pub fn init(a: A, weights: *const source.Weights, capacity: usize) !Session {
        const c = weights.config;
        const layers = try a.alloc(mixers.State, c.layers);
        for (layers, 0..) |*layer, i| switch (c.kind(i)) {
            .linear => {
                const state = try a.alloc(f64, c.v_heads * c.dv * c.dk);
                const history = try a.alloc(f64, c.gdnQkvDim() * (c.conv_kernel - 1));
                @memset(state, 0);
                @memset(history, 0);
                layer.* = .{ .linear = .{ .state = state, .history = history } };
            },
            .attention => {
                const keys = try a.alloc(f64, capacity * c.kvDim());
                const values = try a.alloc(f64, capacity * c.kvDim());
                @memset(keys, 0);
                @memset(values, 0);
                layer.* = .{ .attention = .{ .keys = keys, .values = values } };
            },
        };
        return .{ .weights = weights, .layers = layers, .capacity = capacity };
    }
    /// Returns final-row logits; each layer's scratch is freed before the following layer.
    pub fn forward(s: *Session, a: A, ids: []const u32) ![]f64 {
        const w = s.weights;
        const c = w.config;
        if (ids.len == 0 or s.position + ids.len > s.capacity) return error.CpuReferenceContext;
        const hidden = try a.alloc(f64, ids.len * c.hidden);
        defer a.free(hidden);
        const embedding = try w.matrix("model.embed_tokens.weight");
        for (ids, 0..) |id, row| {
            if (id >= c.vocab) return error.CpuReferenceToken;
            for (hidden[row * c.hidden ..][0..c.hidden], embedding.values[@as(usize, id) * c.hidden ..][0..c.hidden]) |*dst, value| dst.* = bf(value);
        }
        for (s.layers, 0..) |*state, i| {
            var arena = std.heap.ArenaAllocator.init(a);
            defer arena.deinit();
            const temp = arena.allocator();
            const norm = try mixers.rms(temp, hidden, ids.len, c.hidden, try w.vector(try mixers.name(temp, i, "input_layernorm.weight")), c.eps, false);
            const mixed = switch (state.*) {
                .linear => |*g| try mixers.gdn(temp, w, g, i, norm, ids.len),
                .attention => |*attn| try mixers.attention(temp, w, attn, i, norm, ids.len, s.position),
            };
            for (hidden, mixed) |*dst, value| dst.* = bf(dst.* + value);
            const input = try mixers.rms(temp, hidden, ids.len, c.hidden, try w.vector(try mixers.name(temp, i, "post_attention_layernorm.weight")), c.eps, false);
            const gate = try w.project(temp, try mixers.name(temp, i, "mlp.gate_proj.weight"), input, ids.len);
            const up = try w.project(temp, try mixers.name(temp, i, "mlp.up_proj.weight"), input, ids.len);
            for (gate, up) |*value, u| value.* = bf(value.* / (1 + @exp(-value.*)) * u);
            const output = try w.project(temp, try mixers.name(temp, i, "mlp.down_proj.weight"), gate, ids.len);
            for (hidden, output) |*dst, value| dst.* = bf(dst.* + value);
            std.debug.print("CPU_LAYER start={d} rows={d} layer={d}\n", .{ s.position, ids.len, i });
        }
        s.position += ids.len;
        const norm = try mixers.rms(a, hidden[(ids.len - 1) * c.hidden ..][0..c.hidden], 1, c.hidden, try w.vector("model.norm.weight"), c.eps, false);
        defer a.free(norm);
        return w.project(a, "lm_head.weight", norm, 1);
    }
};
pub const Pick = struct { token: u32, runner_up: u32, logit: f64, margin: f64 };
pub fn pick(logits: []const f64) !Pick {
    var first: usize = 0;
    var second: usize = 0;
    for (logits, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonfiniteCpuLogit;
        if (v > logits[first]) first = i;
    }
    if (first == 0) second = 1;
    for (logits, 0..) |v, i| if (i != first and v > logits[second]) {
        second = i;
    };
    return .{ .token = @intCast(first), .runner_up = @intCast(second), .logit = logits[first], .margin = logits[first] - logits[second] };
}

test "Qwen27 GGUF CPU: binary64 projection retains cancellation lost by float32 sums" {
    const t = std.testing;
    var values = [_]f32{ 100000000, 1, -100000000, 3, 0.125, -0.5, 0.25, 0.125 };
    var weights: source.Weights = .{ .file = undefined, .arena = undefined, .config = undefined, .names = &.{} };
    defer weights.matrices.deinit(t.allocator);
    try weights.matrices.put(t.allocator, "test", .{ .values = &values, .n = 2, .k = 4 });
    const output = try weights.project(t.allocator, "test", &.{ 1, 1, 1, 1, 2, 2, 2, 2 }, 2);
    defer t.allocator.free(output);
    try t.expectEqualSlices(f64, &.{ 4, 0, 8, 0 }, output);
    var narrowed: f32 = 0;
    for (values[0..4]) |v| narrowed += v;
    try t.expect(narrowed != output[0]);
}
test "Qwen27 GGUF CPU: greedy ties choose the first id and nonfinite logits refuse" {
    const t = std.testing;
    const choice = try pick(&.{ -1, 4, 4, 2 });
    try t.expectEqual(@as(u32, 1), choice.token);
    try t.expectEqual(@as(f64, 0), choice.margin);
    try t.expectError(error.NonfiniteCpuLogit, pick(&.{ 0, std.math.nan(f64) }));
}
test {
    _ = source;
    _ = mixers;
}
