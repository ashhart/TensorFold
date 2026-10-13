//! Independent wide-sum qualification with explicit FP32/BF16 boundaries; not GPU self-reference or CUDA emulation.

// Equations: pinned DeepSeek AI inference/{model,engram,kernel}.py; provenance in THIRD_PARTY_NOTICES.md.
const std = @import("std");
const tf = @import("tensorfold");
const weights = tf.deepseek_v41_weights;
const config = tf.deepseek_v41_config;
const cp = tf.checkpoint_metal;
const exl3 = tf.exl3_projection;
const projection = @import("projection.zig");
const math = @import("math.zig");
const attention = @import("attention_math.zig");
const engram = @import("engram_math.zig");
const Stage = tf.deepseek_v41_profiler.Stage;

/// Local routing isolation also uses these source equations; no native values enter qualification.
pub const MoeRouting = struct {
    logits: [384]f32 = undefined,
    scores: [384]f32 = undefined,
    ids: [6]u32 = undefined,
    weights: [6]f32 = undefined,
};

pub fn moeSelect(moe_weights: weights.Moe, c: config.Text, logits: *const [384]f32, scores: *[384]f32, ids: *[6]u32, routing_weights: *[6]f32) void {
    @setFloatMode(.strict);
    for (logits, scores) |z, *score| {
        const softplus = if (z > 20) z else std.math.log1p(@exp(z));
        score.* = @sqrt(softplus);
    }
    var used: [384]bool = @splat(false);
    for (0..6) |pick| {
        var best: usize = 0;
        var high: f32 = -std.math.inf(f32);
        for (0..384) |expert| {
            const biased = scores[expert] + math.value(moe_weights.bias, expert);
            if (!used[expert] and biased > high) {
                best = expert;
                high = biased;
            }
        }
        ids[pick] = @intCast(best);
        routing_weights[pick] = scores[best];
        used[best] = true;
    }
    var normalization: f64 = 0;
    for (routing_weights) |v| normalization += v;
    const denominator = @as(f32, @floatCast(normalization)) + 1e-20;
    for (routing_weights) |*v| v.* = (v.* / denominator) * c.routed_scaling_factor;
}

pub fn moeActivation(gate: []const u16, up: []const u16, route: f32, c: config.Text, activation: []u16) void {
    @setFloatMode(.strict);
    for (activation, gate, up) |*v, g_value, u_value| {
        const g = @min(math.f(g_value), c.swiglu_limit);
        const u = @max(-c.swiglu_limit, @min(math.f(u_value), c.swiglu_limit));
        v.* = math.b(route * ((g / (1 + @exp(-g))) * u));
    }
}

pub const Reference = struct {
    gpa: std.mem.Allocator,
    c: config.Text,
    w: *const weights.Model,
    checkpoint: *const cp.Checkpoint,
    capacity: u32,
    position: u32 = 0,
    projection_scratch: projection.Scratch,
    attention_state: attention.State,
    engram_state: engram.State,
    logits: []f32,
    h: [20480]u16 = undefined,
    input: [5120]u16 = undefined,
    output: [5120]u16 = undefined,
    gate: [2304]u16 = undefined,
    up: [2304]u16 = undefined,
    activation: [2304]u16 = undefined,
    expert_output: [5120]u16 = undefined,
    moe_sum: [5120]f32 = undefined,
    route_logits: [384]f32 = undefined,
    route_scores: [384]f32 = undefined,
    last_routes: [6]u32 = undefined,
    last_routing: [6]f32 = undefined,
    observer: ?Observer = null,
    pub const Observer = struct {
        ctx: *anyopaque,
        read: *const fn (*anyopaque, usize, Stage, *const Reference) void,
    };

    /// Borrowed model/checkpoint/config/map must outlive this instance; capacity is the teacher sequence length.
    pub fn init(gpa: std.mem.Allocator, c: config.Text, w: *const weights.Model, checkpoint: *const cp.Checkpoint, token_map: []const u8, capacity: u32) !Reference {
        if (capacity == 0 or capacity > c.max_position_embeddings or capacity > attention.max_capacity)
            return error.InvalidReferenceCapacity;
        if (c.hidden_size != 5120 or c.num_hidden_layers != 40 or c.vocab_size != 129280 or
            c.num_experts_per_tok != 6 or c.n_routed_experts != 384 or c.hc_mult != 4)
            return error.UnsupportedReferenceArchitecture;
        const engram_state = try engram.State.init(token_map);
        var projection_scratch = try projection.Scratch.init(gpa, 8192, 129280);
        errdefer projection_scratch.deinit();
        var attention_state = try attention.State.init(gpa, capacity);
        errdefer attention_state.deinit();
        const logits = try gpa.alloc(f32, 129280);
        return .{ .gpa = gpa, .c = c, .w = w, .checkpoint = checkpoint, .capacity = capacity, .projection_scratch = projection_scratch, .attention_state = attention_state, .engram_state = engram_state, .logits = logits };
    }
    pub fn deinit(self: *Reference) void {
        self.projection_scratch.deinit();
        self.attention_state.deinit();
        self.gpa.free(self.logits);
        self.* = undefined;
    }
    /// Consume exactly one teacher token and return all actual checkpoint logits, borrowed until the next step.
    pub fn step(self: *Reference, id: u32) ![]const f32 {
        if (self.position >= self.capacity) return error.ContextFull;
        const compressed = try self.engram_state.compressed(id);
        for (0..5120) |d| {
            const v = math.b(math.value(self.w.embed, @as(usize, id) * 5120 + d));
            for (0..4) |copy| self.h[copy * 5120 + d] = v;
        }
        var pre = [4]f32{ 1, 0, 0, 0 };
        for (&self.w.layers, 0..) |*layer, layer_id| {
            if (layer.engram) |e| try self.engram_state.apply(e, if (layer_id == 1) 0 else 1, compressed, &self.h, self.c, &self.projection_scratch);
            const attn_mix = math.mixes(&self.h, layer.hc_attn, self.c);
            math.collapse(&self.h, pre, &self.input);
            math.rms(&self.input, layer.attn_norm, self.c.rms_norm_eps, &self.input);
            try self.attention_state.apply(layer.attn, layer_id, &self.input, self.position, self.c, &self.output, &self.projection_scratch);
            math.expand(&self.h, &self.output, attn_mix);
            if (self.observer) |observer| observer.read(observer.ctx, layer_id, .attention, self);
            const ffn_mix = math.mixes(&self.h, layer.hc_ffn, self.c);
            math.collapse(&self.h, attn_mix.pre, &self.input);
            math.rms(&self.input, layer.ffn_norm, self.c.rms_norm_eps, &self.input);
            try self.moe(layer.moe, layer_id);
            math.expand(&self.h, &self.output, ffn_mix);
            if (self.observer) |observer| observer.read(observer.ctx, layer_id, .moe, self);
            pre = ffn_mix.pre;
        }
        math.collapse(&self.h, pre, &self.input);
        math.rms(&self.input, self.w.norm, self.c.rms_norm_eps, &self.input);
        try projection.applyF32(.{ .trellis = self.w.head }, &self.input, self.logits, &self.projection_scratch);
        for (self.logits) |v| if (!std.math.isFinite(v)) return error.NonFiniteReferenceLogits;
        self.engram_state.advance(compressed);
        self.position += 1;
        return self.logits;
    }
    fn moe(self: *Reference, moe_weights: weights.Moe, layer: usize) !void {
        @setFloatMode(.strict);
        math.dense(moe_weights.gate, &self.input, &self.route_logits);
        moeSelect(moe_weights, self.c, &self.route_logits, &self.route_scores, &self.last_routes, &self.last_routing);
        @memset(&self.moe_sum, 0);
        // Official MoE visits expert ids in ascending order, accumulates FP32, then adds shared expert last.
        for (0..384) |expert| {
            var route: ?f32 = null;
            for (self.last_routes, self.last_routing) |selected, weight|
                if (selected == expert) {
                    route = weight;
                };
            const expert_route = route orelse continue;
            var matrices: [3]exl3.Matrix = undefined;
            var name: [128]u8 = undefined;
            inline for (.{ "w1", "w3", "w2" }, 0..) |part, i| {
                const prefix = try std.fmt.bufPrint(&name, "layers.{d}.ffn.experts.{d}.{s}", .{ layer, expert, part });
                matrices[i] = try weights.lookup(self.checkpoint, prefix, if (i == 2) 2304 else 5120, if (i == 2) 5120 else 2304);
            }
            try self.applyExpert(matrices, expert_route);
            for (&self.moe_sum, self.expert_output) |*sum, v| sum.* += math.f(v);
        }
        try self.applyExpert(moe_weights.shared, 1);
        for (&self.output, self.moe_sum, self.expert_output) |*v, routed, shared|
            v.* = math.b(routed + math.f(shared));
    }
    fn applyExpert(self: *Reference, matrices: [3]exl3.Matrix, route: f32) !void {
        @setFloatMode(.strict);
        try projection.apply(.{ .trellis = matrices[0] }, &self.input, &self.gate, &self.projection_scratch);
        try projection.apply(.{ .trellis = matrices[1] }, &self.input, &self.up, &self.projection_scratch);
        moeActivation(&self.gate, &self.up, route, self.c, &self.activation);
        try projection.apply(.{ .trellis = matrices[2] }, &self.activation, &self.expert_output, &self.projection_scratch);
    }
};
