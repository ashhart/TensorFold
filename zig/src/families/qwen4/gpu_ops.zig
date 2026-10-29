//! One dense forward adapter owns tensor layout on both native backends; no host arithmetic substitutes a layer.
const std = @import("std");
const graph = @import("hybrid_graph");
const cfg = @import("config.zig");
const arena_ = @import("arena.zig");
pub fn Ops(comptime G: type) type {
    const B = G.Storage;
    const Model = @import("model.zig").Model(B);
    const Scratch = arena_.Arena(B);
    return struct {
        const Self = @This();
        pub const Tensor = Scratch.Tensor;
        gpu: *G,
        model: *Model,
        scratch: *Scratch,
        host: std.mem.Allocator,
        problem: *cfg.Problem,
        hidden: ?Tensor = null,
        layer_mark: usize = 0,
        trace: ?@import("node_trace.zig").Sink(B) = null,
        trace_layer: ?usize = null,
        trace_scope: []const u8 = "",
        fn note(o: *Self, layer: ?usize, scope: []const u8, node: []const u8, x: Tensor) !void {
            const sink = o.trace orelse return;
            var buffer: [192]u8 = undefined;
            try sink.tensor(try @import("node_trace.zig").label(&buffer, layer, scope, node), x);
        }
        fn hcScope(b: graph.Boundary) []const u8 {
            return switch (b) {
                .readout => "model.hyper_connection_mixer",
                .attention => "attn_hyper_connection",
                .mlp => "mlp_hyper_connection",
            };
        }
        fn take(o: *Self, rows: usize, width: usize) !Tensor {
            return o.scratch.take(rows, width, .bf16);
        }
        pub fn project(o: *Self, matrix: Model.Matrix, x: Tensor) !Tensor {
            if (x.width != matrix.params.k or x.kind != .bf16) return error.Qwen4ProjectionShape;
            const y = try o.take(x.rows, @intCast(matrix.params.n));
            try o.gpu.project(matrix, x, y);
            return y;
        }
        pub fn embed(o: *Self, tokens: []const u32, hidden: usize, streams: usize) !Tensor {
            const h = try o.take(tokens.len, hidden * streams);
            const row = try o.host.alloc(u16, hidden);
            defer o.host.free(row);
            for (tokens, 0..) |token, r| {
                try o.model.weights.lookup(o.model.embedding, token, row, o.problem);
                for (0..streams) |s| try o.scratch.backend.write(h.buffer, h.offset + (r * streams + s) * hidden * 2, std.mem.sliceAsBytes(row));
            }
            try o.note(null, "", "embed", h);
            o.hidden = h;
            o.layer_mark = o.scratch.used;
            return h;
        }
        fn hc(o: *Self, layer: usize, b: graph.Boundary) Model.Hc {
            return switch (b) {
                .readout => o.model.readout,
                .attention => o.model.layers[layer].attention,
                .mlp => o.model.layers[layer].mlp,
            };
        }
        pub fn hcNorm(o: *Self, layer: usize, b: graph.Boundary, h: Tensor, hidden: usize, eps: f32) !Tensor {
            if (hidden != o.model.c.hidden or h.width != o.model.c.wide()) return error.Qwen4HcGeometry;
            const sums = try o.scratch.take(h.rows, h.width / 256, .f32);
            const out = try o.take(h.rows, h.width);
            try o.gpu.hcNorm(o.hc(layer, b).norm, h, sums, out, eps);
            o.trace_layer = if (b == .readout) null else layer;
            o.trace_scope = hcScope(b);
            try o.note(o.trace_layer, o.trace_scope, "sums", sums);
            try o.note(o.trace_layer, o.trace_scope, "norm", out);
            return out;
        }
        pub fn hcProject(o: *Self, layer: usize, b: graph.Boundary, p: graph.Projection, x: Tensor) !Tensor {
            const h = o.hc(layer, b);
            const out = try o.project(switch (p) {
                .down => h.down,
                .up => h.up,
                .inject => h.inject orelse return error.Qwen4InjectionMissing,
                else => return error.Qwen4ProjectionRole,
            }, x);
            try o.note(if (b == .readout) null else layer, hcScope(b), if (p == .inject) "raw_inject" else @tagName(p), out);
            return out;
        }
        pub fn hcActivate(o: *Self, x: Tensor, streams: usize) !Tensor {
            return o.activate(x, streams, false);
        }
        pub fn hcInjection(o: *Self, x: Tensor, streams: usize) !Tensor {
            return o.activate(x, streams, true);
        }
        fn activate(o: *Self, x: Tensor, streams: usize, injection: bool) !Tensor {
            if (streams != o.model.c.streams) return error.Qwen4HcGeometry;
            const out = try o.take(x.rows, x.width);
            try o.gpu.hcActivation(x, out, streams, injection);
            try o.note(o.trace_layer, o.trace_scope, if (injection) "injection" else "low", out);
            return out;
        }
        pub fn hcMix(o: *Self, up: Tensor, normed: Tensor, streams: usize) !Tensor {
            if (streams != o.model.c.streams or up.rows != normed.rows or up.width != normed.width) return error.Qwen4HcGeometry;
            const out = try o.take(up.rows, o.model.c.hidden);
            const sums = try o.scratch.take(up.rows, o.model.c.hidden / 32, .f32);
            try o.gpu.hcMix(up, normed, out, sums);
            try o.note(o.trace_layer, o.trace_scope, "mix", out);
            return out;
        }
        pub fn writeback(o: *Self, h: Tensor, branch: Tensor, injection: Tensor) !Tensor {
            // The residual is the only live input after its boundary, so this is the declared exact in-place view.
            try o.gpu.writeback(h, branch, injection, h);
            try o.note(o.trace_layer, o.trace_scope, "residual", h);
            return h;
        }
        pub fn denseProject(o: *Self, layer: usize, p: graph.Projection, x: Tensor) !Tensor {
            const d = o.model.layers[layer].dense;
            const out = try o.project(switch (p) {
                .gate => d.gate,
                .value => d.up,
                .output => d.down,
                else => return error.Qwen4ProjectionRole,
            }, x);
            try o.note(layer, "mlp", if (p == .value) "up" else @tagName(p), out);
            return out;
        }
        pub fn swiglu(o: *Self, gate: Tensor, up: Tensor) !Tensor {
            if (gate.rows != up.rows or gate.width != up.width) return error.Qwen4DenseShape;
            const out = try o.take(gate.rows, gate.width);
            try o.gpu.swiglu(gate, up, out);
            try o.note(o.trace_layer, "mlp", "act", out);
            return out;
        }
        /// Different quantization recipes project separately; queue-ordered row copies form the kernel's input pack.
        pub fn pack(o: *Self, x: Tensor, matrices: []const Model.Matrix) !Tensor {
            var width: usize = 0;
            for (matrices) |m| width = try std.math.add(usize, width, @intCast(m.params.n));
            const out = try o.take(x.rows, width);
            const mark = o.scratch.used;
            var col: usize = 0;
            for (matrices) |m| {
                const y = try o.project(m, x);
                for (0..x.rows) |r| try o.scratch.backend.copy(out.buffer, out.offset + (r * width + col) * 2, y.buffer, y.offset + r * y.width * 2, y.width * 2);
                col += y.width;
                try o.scratch.rewind(mark);
            }
            return out;
        }
        pub fn delta(o: *Self, layer: usize, state: anytype, x: Tensor) !Tensor {
            const w = switch (o.model.layers[layer].mixer) {
                .linear => |w| w,
                else => return error.Qwen4LayerKind,
            };
            const input = try o.pack(x, &.{ w.qkv, w.z, w.b, w.a });
            try o.note(layer, "linear_attn", "projections", input);
            const out = try o.take(x.rows, o.model.c.valueWidth());
            const sums = try o.scratch.take(x.rows, o.model.c.valueWidth() / 32, .f32);
            if (@hasDecl(@TypeOf(state.*), "segments")) {
                var off: usize = 0;
                for (state.segments()) |seg| {
                    const n = seg.tokens.len;
                    try o.gpu.delta(w, seg.state, layer, try input.subRows(off, n), try out.subRows(off, n), try sums.subRows(off, n));
                    off += n;
                }
            } else try o.gpu.delta(w, state, layer, input, out, sums);
            try state.captured(layer);
            try o.note(layer, "linear_attn", "gated", out);
            const y = try o.project(w.out, out);
            try o.note(layer, "linear_attn", "output", y);
            return y;
        }
        pub fn indexed(o: *Self, layer: usize, state: anytype, x: Tensor) !Tensor {
            const w = switch (o.model.layers[layer].mixer) {
                .indexed => |w| w,
                else => return error.Qwen4LayerKind,
            };
            const input = try o.pack(x, &.{ w.q, w.k, w.v, w.index });
            try o.note(layer, "self_attn", "projections", input);
            const out = try o.take(x.rows, o.model.c.heads * o.model.c.head_dim);
            if (@hasDecl(@TypeOf(state.*), "segments")) {
                var off: usize = 0;
                for (state.segments()) |seg| {
                    const n = seg.tokens.len;
                    try o.gpu.indexed(o.scratch, w, seg.state, layer, try input.subRows(off, n), try out.subRows(off, n));
                    off += n;
                }
            } else try o.gpu.indexed(o.scratch, w, state, layer, input, out);
            try state.captured(layer);
            try o.note(layer, "self_attn", "gated", out);
            const y = try o.project(w.out, out);
            try o.note(layer, "self_attn", "output", y);
            return y;
        }
        pub fn ple(o: *Self, layer: usize, state: anytype, tokens: []const u32, h: Tensor) !Tensor {
            const w = o.model.layers[layer].ple orelse return error.Qwen4PleMissing;
            const c = o.model.c;
            const embedding = try o.take(tokens.len, c.ple_dim);
            if (@hasDecl(@TypeOf(state.*), "segments")) {
                var off: usize = 0;
                for (state.segments()) |seg| {
                    try o.lexicalRows(w, seg.state.history(), seg.tokens, embedding, off);
                    off += seg.tokens.len;
                }
            } else try o.lexicalRows(w, state.history(), tokens, embedding, 0);
            try o.note(layer, "ple", "embedding", embedding);
            const projections = try o.pack(embedding, &.{ w.key, w.value });
            try o.note(layer, "ple", "projections", projections);
            const gated = try o.take(h.rows, h.width);
            const normed = try o.take(h.rows, h.width);
            const tail_rows = (c.ple_conv - 1) * c.ngram;
            if (@hasDecl(@TypeOf(state.*), "segments")) {
                var off: usize = 0;
                for (state.segments()) |seg| {
                    const n = seg.tokens.len;
                    const cin = try o.take(tail_rows + n, h.width);
                    try o.gpu.ple(o.scratch, w, seg.state, layer, try projections.subRows(off, n), try h.subRows(off, n), try gated.subRows(off, n), try normed.subRows(off, n), cin);
                    off += n;
                }
            } else {
                const cin = try o.take(tail_rows + h.rows, h.width);
                try o.gpu.ple(o.scratch, w, state, layer, projections, h, gated, normed, cin);
            }
            try o.note(layer, "ple", "gated", gated);
            try o.note(layer, "ple", "normed", normed);
            try o.note(layer, "ple", "residual", h);
            return h;
        }
        fn lexicalRows(o: *Self, w: Model.Ple, history: []const u32, tokens: []const u32, embedding: Tensor, offset: usize) !void {
            const c = o.model.c;
            const heads = (c.ngram - 1) * c.ngram_heads;
            const width = c.ple_dim / heads;
            const ids = try o.host.alloc(usize, heads);
            defer o.host.free(ids);
            const tail = try o.host.alloc(u32, c.ngram - 1);
            defer o.host.free(tail);
            @memset(tail, @intCast(c.ple_eos));
            const kept = @min(tail.len, history.len);
            @memcpy(tail[tail.len - kept ..], history[history.len - kept ..]);
            const row = try o.host.alloc(u16, width);
            defer o.host.free(row);
            for (tokens, 0..) |token, r| {
                try w.policy.step(tail, token, ids);
                for (ids, 0..) |id, j| {
                    try o.model.weights.lookup(w.table, id, row, o.problem);
                    try o.scratch.backend.write(embedding.buffer, embedding.offset + ((r + offset) * c.ple_dim + j * width) * 2, std.mem.sliceAsBytes(row));
                }
            }
        }
        pub fn head(o: *Self, x: Tensor) !Tensor {
            const y = try o.project(o.model.head, x);
            try o.note(null, "", "head", y);
            return y;
        }
        pub fn routed(_: *Self, _: usize, _: Tensor) !Tensor {
            return error.Qwen4RoutedLayer;
        }
        /// Every boundary writes the persistent residual in place before temporary ranges can be reused.
        pub fn finishLayer(o: *Self, h: Tensor) !void {
            const live = o.hidden orelse return error.Qwen4ResidualMissing;
            if (h.offset != live.offset or h.width != live.width or h.rows != live.rows) return error.Qwen4ResidualMoved;
            try o.scratch.rewind(o.layer_mark);
        }
    };
}
