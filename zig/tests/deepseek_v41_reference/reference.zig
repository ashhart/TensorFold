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

/// The reduction order a caller uses to normalise the six selected scores. `f64_sequential` is this file's own
/// arithmetic; the two f32 variants are legal orders a GPU implementation may take, and exist so a gate can compare
/// an engine against the *envelope* of legal orders rather than against one member of it.
pub const WeightOrder = enum { f64_sequential, f32_pairwise, f32_ascending };

/// The observer's read callback carries the chain as an untyped pointer, because the chain is now generic over its
/// precision and cannot name itself inside its own declaration. The reader casts it back.
pub const Observer = struct {
    ctx: *anyopaque,
    read: *const fn (*anyopaque, usize, Stage, *anyopaque) void,
};

pub fn moeWeights(c: config.Config, scores: *const [384]f32, ids: *const [6]u32, order: WeightOrder, routing_weights: *[6]f32) void {
    moeWeightsOf(f32, c, scores, ids, order, routing_weights);
}

/// The six selected scores' normalisation at the chain's precision T. The FP32 contract is unchanged (bit-for-bit);
/// the FP64 chain keeps the six in FP64 and widens once.
pub fn moeWeightsOf(comptime T: type, c: config.Config, scores: []const T, ids: *const [6]u32, order: WeightOrder, routing_weights: *[6]T) void {
    @setFloatMode(.strict);
    for (ids, 0..) |id, i| routing_weights[i] = scores[id];
    switch (order) {
        // This file's arithmetic: sum the six in selection order, widen once.
        .f64_sequential => {
            var normalization: f64 = 0;
            for (routing_weights) |v| normalization += v;
            if (T == f32) {
                const denominator = @as(f32, @floatCast(normalization)) + 1e-20;
                for (routing_weights) |*v| v.* = (v.* / denominator) * c.routed_scale;
            } else {
                const denominator: f64 = normalization + 1e-20;
                for (routing_weights) |*v| v.* = (v.* / denominator) * @as(f64, c.routed_scale);
            }
        },
        // route.metal, verbatim: left=(p0+p4)+p2; right=(p1+p5)+p3; s=left+right.
        .f32_pairwise => {
            const w = routing_weights.*;
            const left = (w[0] + w[4]) + w[2];
            const right = (w[1] + w[5]) + w[3];
            const denominator: T = (left + right) + 1e-20;
            for (routing_weights) |*v| v.* = (v.* / denominator) * @as(T, c.routed_scale);
        },
        // Six f32 adds in selection order: the naive GPU reduction, a legal order of its own.
        .f32_ascending => {
            var denominator: T = 0;
            for (routing_weights) |v| denominator += v;
            denominator += 1e-20;
            for (routing_weights) |*v| v.* = (v.* / denominator) * @as(T, c.routed_scale);
        },
    }
}

/// Distance from the selection boundary: the lowest selected biased score minus the highest unselected one. Always
/// computed from this file's own scores, never from a caller-supplied or pinned weight vector, or the number says
/// nothing about the reference's own arithmetic.
pub fn selectionMargin(moe_weights: weights.Moe, scores: *const [384]f32, ids: *const [6]u32) f32 {
    return selectionMarginOf(f32, moe_weights, scores, ids);
}

/// The selection boundary at the chain's precision T, reported in FP32 (a diagnostic).
pub fn selectionMarginOf(comptime T: type, moe_weights: weights.Moe, scores: []const T, ids: *const [6]u32) f32 {
    @setFloatMode(.strict);
    var lowest_selected = std.math.inf(f32);
    for (ids) |id| lowest_selected = @min(lowest_selected, @as(f32, @floatCast(scores[id] + @as(T, math.value(moe_weights.bias, id)))));
    var highest_unselected = -std.math.inf(f32);
    for (0..384) |expert| {
        var selected = false;
        for (ids) |id| if (id == expert) {
            selected = true;
        };
        if (!selected) highest_unselected = @max(highest_unselected, @as(f32, @floatCast(scores[expert] + @as(T, math.value(moe_weights.bias, expert)))));
    }
    return lowest_selected - highest_unselected;
}

/// Which numerical form of the router's score transform this file uses. `libm` is Zig's `log1p(exp(z))`;
/// `kernel_lp` is `route.metal`'s form verbatim (`u=1+e; u==1 ? e: e*(log(u)/(u-1))`).
pub const ScoreForm = enum { libm, kernel_lp, stable };

fn scoreOf(z: f32, form: ScoreForm) f32 {
    return scoreOfOf(f32, z, form);
}

/// The router's score transform at the chain's precision T: FP32 is the source contract, bit-identical to the
/// shipped path; FP64 is the ideal chain's form.
fn scoreOfOf(comptime T: type, z: T, form: ScoreForm) T {
    @setFloatMode(.strict);
    // `stable` is the formulation most numerical libraries ship: softplus(z) = max(z,0) + log1p(exp(-|z|)). It is a
    // genuinely different algebraic route from `libm` and equally accurate, which is what makes it a fair test of
    // "does the choice of implementation move the answer" rather than "does one formula lose precision".
    // `kernel_lp` is route.metal's own formula: kept because it is what the engine executes, not because it is a
    // general-purpose implementation of softplus.
    const softplus: T = switch (form) {
        .kernel_lp => if (z > 20) z else blk: {
            const e = @exp(z);
            const u = 1.0 + e;
            break :blk if (u == 1.0) e else e * (@log(u) / (u - 1.0));
        },
        .stable => if (z > 20) z else @max(z, 0) + std.math.log1p(@exp(-@abs(z))),
        .libm => if (z > 20) z else std.math.log1p(@exp(z)),
    };
    return @sqrt(softplus);
}

pub fn moeSelectForm(moe_weights: weights.Moe, c: config.Config, logits: *const [384]f32, scores: *[384]f32, ids: *[6]u32, routing_weights: *[6]f32, order: WeightOrder, form: ScoreForm) void {
    moeSelectFormOf(f32, moe_weights, c, logits, scores, ids, routing_weights, order, form);
}

/// The router's own selection at the chain's precision T. The FP32 contract is unchanged.
pub fn moeSelectFormOf(comptime T: type, moe_weights: weights.Moe, c: config.Config, logits: []const T, scores: *[384]T, ids: *[6]u32, routing_weights: *[6]T, order: WeightOrder, form: ScoreForm) void {
    @setFloatMode(.strict);
    for (logits, scores) |z, *score| score.* = scoreOfOf(T, z, form);
    var used: [384]bool = @splat(false);
    for (0..6) |pick| {
        var best: usize = 0;
        var high: T = -std.math.inf(T);
        for (0..384) |expert| {
            const biased = scores[expert] + @as(T, math.value(moe_weights.bias, expert));
            if (!used[expert] and biased > high) {
                best = expert;
                high = biased;
            }
        }
        ids[pick] = @intCast(best);
        used[best] = true;
    }
    moeWeightsOf(T, c, scores, ids, order, routing_weights);
}

pub fn moeSelect(moe_weights: weights.Moe, c: config.Config, logits: *const [384]f32, scores: *[384]f32, ids: *[6]u32, routing_weights: *[6]f32) void {
    moeSelectForm(moe_weights, c, logits, scores, ids, routing_weights, .f64_sequential, .libm);
}

pub fn moeActivation(gate: []const u16, up: []const u16, route: f32, c: config.Config, activation: []u16) void {
    moeActivationOf(f32, gate, up, route, c, activation);
}

/// The SwiGLU at the chain's precision T. The FP32 contract is unchanged (bit-for-bit).
pub fn moeActivationOf(comptime T: type, gate: []const u16, up: []const u16, route: T, c: config.Config, activation: []u16) void {
    @setFloatMode(.strict);
    for (activation, gate, up) |*v, g_value, u_value| {
        const g: T = @min(@as(T, math.f(g_value)), @as(T, c.swiglu_limit));
        const u: T = @max(-@as(T, c.swiglu_limit), @min(@as(T, math.f(u_value)), @as(T, c.swiglu_limit)));
        if (T == f32) {
            v.* = math.b(route * ((g / (1 + @exp(-g))) * u));
        } else {
            v.* = math.b64(route * ((g / (1 + @exp(-g))) * u));
        }
    }
}

/// The reference chain at a chosen precision T. T == f32 is the source contract, bit-identical to the shipped
/// chain; T == f64 is the ideal chain (every reduction class widened: the RMS norms, the per-head query RMS, the
/// hyperconnection's affine/softmax/Sinkhorn stage, the router's dense projection, the attention's online softmax,
/// the compressor's pooling, the indexer's scores and the MoE's own arithmetic).
pub fn ReferenceOf(comptime T: type) type {
    return struct {
        pub const Pin = struct { ids: *const [40][6]u32 };

        gpa: std.mem.Allocator,
        c: config.Config,
        w: *const weights.Model,
        checkpoint: *const cp.Checkpoint,
        capacity: u32,
        position: u32 = 0,
        projection_scratch: projection.Scratch,
        rows: ?engram.RowSource = null,
        /// When set, expert ids come from the caller (an engine capture). Selection still runs on this file's own logits,
        /// and the weights are still this file's own arithmetic evaluated at the pinned ids — ids-only is the honest pin.
        /// Pinning the weights as well would make the reference share the engine's arithmetic, which is a diagnostic,
        /// never a qualification.
        pin: ?Pin = null,
        weight_order: WeightOrder = .f64_sequential,
        score_form: ScoreForm = .libm,
        /// This file's own distance from each layer's selection boundary, recorded whatever the pin says.
        route_margins: [40]f32 = @splat(0),
        /// The selection this file made before any pin was applied, so a gate can report where its own routing departs
        /// from the engine's instead of only reporting that the pin held.
        natural_ids: [40][6]u32 = std.mem.zeroes([40][6]u32),
        attention_state: attention.StateOf(T),
        engram_state: engram.State,
        logits: []T,
        h: [20480]u16 = undefined,
        input: [5120]u16 = undefined,
        output: [5120]u16 = undefined,
        gate: [2304]u16 = undefined,
        up: [2304]u16 = undefined,
        activation: [2304]u16 = undefined,
        expert_output: [5120]u16 = undefined,
        moe_sum: [5120]T = undefined,
        route_logits: [384]T = undefined,
        route_scores: [384]T = undefined,
        last_routes: [6]u32 = undefined,
        last_routing: [6]T = undefined,
        observer: ?Observer = null,

        /// Borrowed model/checkpoint/config/map must outlive this instance; capacity is the teacher sequence length.
        pub fn init(gpa: std.mem.Allocator, c: config.Config, w: *const weights.Model, checkpoint: *const cp.Checkpoint, token_map: []const u8, capacity: u32) !@This() {
            if (capacity == 0 or capacity > c.rope.max_positions or capacity > attention.max_capacity)
                return error.InvalidReferenceCapacity;
            if (c.hidden != 5120 or c.layers != 40 or c.vocab != 129280 or
                c.topk != 6 or c.experts != 384 or c.hc_mult != 4)
                return error.UnsupportedReferenceArchitecture;
            const engram_state = try engram.State.init(token_map);
            var projection_scratch = try projection.Scratch.init(gpa, 8192, 129280);
            errdefer projection_scratch.deinit();
            var attention_state = try attention.StateOf(T).init(gpa, capacity);
            errdefer attention_state.deinit();
            const logits = try gpa.alloc(T, 129280);
            return .{ .gpa = gpa, .c = c, .w = w, .checkpoint = checkpoint, .capacity = capacity, .projection_scratch = projection_scratch, .attention_state = attention_state, .engram_state = engram_state, .logits = logits };
        }
        /// Rows for a streamed model, when the caller has one (see engram_math.RowSource).
        pub fn attachRows(self: *@This(), source: engram.RowSource) void {
            self.rows = source;
        }
        pub fn deinit(self: *@This()) void {
            self.projection_scratch.deinit();
            self.attention_state.deinit();
            self.gpa.free(self.logits);
            self.* = undefined;
        }
        /// Consume exactly one teacher token and return all actual checkpoint logits, borrowed until the next step.
        pub fn step(self: *@This(), id: u32) ![]const T {
            if (self.position >= self.capacity) return error.ContextFull;
            const compressed = try self.engram_state.compressed(id);
            for (0..5120) |d| {
                const v = math.b(math.value(self.w.embed, @as(usize, id) * 5120 + d));
                for (0..4) |copy| self.h[copy * 5120 + d] = v;
            }
            var pre = [4]T{ 1, 0, 0, 0 };
            for (&self.w.layers, 0..) |*layer, layer_id| {
                if (layer.engram) |e| {
                    // A streamed model has no resident Engram rows: the caller may supply a row source, and if it does not,
                    // say so by name rather than reading an unbound view.
                    if (e.streamed and self.rows == null) return error.ReferenceNeedsResidentEngramTables;
                    try self.engram_state.apply(T, e, if (layer_id == 1) 0 else 1, compressed, &self.h, self.c, &self.projection_scratch, self.rows);
                }
                const attn_mix = math.mixesOf(T, &self.h, layer.hc_attn, self.c);
                math.collapseOf(T, &self.h, pre, &self.input);
                math.rmsOf(T, &self.input, layer.attn_norm, self.c.eps, &self.input);
                try self.attention_state.apply(layer.attn, layer_id, &self.input, self.position, self.c, &self.output, &self.projection_scratch);
                math.expandOf(T, &self.h, &self.output, attn_mix);
                if (self.observer) |observer| observer.read(observer.ctx, layer_id, .attention, self);
                const ffn_mix = math.mixesOf(T, &self.h, layer.hc_ffn, self.c);
                math.collapseOf(T, &self.h, attn_mix.pre, &self.input);
                math.rmsOf(T, &self.input, layer.ffn_norm, self.c.eps, &self.input);
                try self.moe(layer.moe, layer_id);
                math.expandOf(T, &self.h, &self.output, ffn_mix);
                if (self.observer) |observer| observer.read(observer.ctx, layer_id, .moe, self);
                pre = ffn_mix.pre;
            }
            math.collapseOf(T, &self.h, pre, &self.input);
            math.rmsOf(T, &self.input, self.w.norm, self.c.eps, &self.input);
            try projection.applyOf(T, T, .{ .trellis = self.w.head }, &self.input, self.logits, &self.projection_scratch);
            for (self.logits) |v| if (!std.math.isFinite(v)) return error.NonFiniteReferenceLogits;
            self.engram_state.advance(compressed);
            self.position += 1;
            return self.logits;
        }
        fn moe(self: *@This(), moe_weights: weights.Moe, layer: usize) !void {
            @setFloatMode(.strict);
            math.denseOf(T, moe_weights.gate, &self.input, &self.route_logits);
            moeSelectFormOf(T, moe_weights, self.c, &self.route_logits, &self.route_scores, &self.last_routes, &self.last_routing, self.weight_order, self.score_form);
            // Recorded from this file's own scores and its own selection, before any pin can touch either.
            self.route_margins[layer] = selectionMarginOf(T, moe_weights, &self.route_scores, &self.last_routes);
            @memcpy(self.natural_ids[layer][0..], self.last_routes[0..]);
            if (self.pin) |pin| @memcpy(self.last_routes[0..], pin.ids[layer][0..]);
            if (self.pin != null)
                moeWeightsOf(T, self.c, &self.route_scores, &self.last_routes, self.weight_order, &self.last_routing);
            @memset(&self.moe_sum, 0);
            // Official MoE visits expert ids in ascending order, accumulates FP32, then adds shared expert last.
            for (0..384) |expert| {
                var route: ?T = null;
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
                for (&self.moe_sum, self.expert_output) |*sum, v| sum.* += if (T == f32) @as(T, math.f(v)) else @as(T, @floatCast(math.f(v)));
            }
            try self.applyExpert(moe_weights.shared, 1);
            for (&self.output, self.moe_sum, self.expert_output) |*v, routed, shared| {
                if (T == f32) v.* = math.b(routed + math.f(shared)) else v.* = math.b64(routed + @as(f64, @floatCast(math.f(shared))));
            }
        }
        fn applyExpert(self: *@This(), matrices: [3]exl3.Matrix, route: T) !void {
            @setFloatMode(.strict);
            try projection.applyOf(u16, T, .{ .trellis = matrices[0] }, &self.input, &self.gate, &self.projection_scratch);
            try projection.applyOf(u16, T, .{ .trellis = matrices[1] }, &self.input, &self.up, &self.projection_scratch);
            moeActivationOf(T, &self.gate, &self.up, route, self.c, &self.activation);
            try projection.applyOf(u16, T, .{ .trellis = matrices[2] }, &self.activation, &self.expert_output, &self.projection_scratch);
        }
    };
}

/// The source-contract chain (FP32), the shipped reference chain's own.
pub const Reference = ReferenceOf(f32);
