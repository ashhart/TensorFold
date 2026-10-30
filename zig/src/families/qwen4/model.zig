//! Typed, borrowed device weights for the common forward; mixed projections retain separate codes and metadata.
const std = @import("std");
const cfg = @import("config.zig");
const ngram = @import("hybrid_ngram");
pub fn Model(comptime B: type) type {
    const Weights = @import("device_weights.zig").Placed(B);
    return struct {
        const Self = @This();
        pub const Matrix = Weights.Matrix;
        pub const Norm = struct { buffer: B.Buffer, centered: bool, prepared: bool = true, mid_bf16: bool = true };
        pub const Hc = struct { norm: Norm, down: Matrix, up: Matrix, inject: ?Matrix };
        pub const Dense = struct { gate: Matrix, up: Matrix, down: Matrix };
        pub const Linear = struct { qkv: Matrix, z: Matrix, a: Matrix, b: Matrix, out: Matrix, conv: B.Buffer, conv_bf16: bool, alog: B.Buffer, dt: B.Buffer, norm: Norm };
        pub const Indexed = struct { q: Matrix, k: Matrix, v: Matrix, out: Matrix, index: Matrix, qn: Norm, kn: Norm, iqn: Norm, ikn: Norm };
        pub const Ple = struct { key: Matrix, value: Matrix, key_norm: Norm, query_norm: Norm, conv_norm: Norm, conv: B.Buffer, table: []const u8, policy: ngram.View };
        pub const Layer = struct { attention: Hc, mlp: Hc, dense: Dense, mixer: union(enum) { linear: Linear, indexed: Indexed }, ple: ?Ple };
        storage: std.heap.ArenaAllocator,
        weights: *Weights,
        c: *const cfg.Config,
        layers: []Layer,
        readout: Hc,
        head: Matrix,
        embedding: []const u8,
        /// Readers and placed allocations outlive this descriptor; closing it frees no borrowed device buffer.
        pub fn init(a: std.mem.Allocator, w: *Weights, p: *cfg.Problem) !Self {
            var arena: std.heap.ArenaAllocator = .init(a);
            errdefer arena.deinit();
            var loader: Loader = .{ .a = arena.allocator(), .w = w, .p = p };
            const c = w.c;
            const pre = w.reader.namespace.prefix();
            const layers = try loader.a.alloc(Layer, c.layers);
            var ordinal: usize = 0;
            for (layers, 0..) |*layer, i| {
                const stem = try loader.name("{s}layers.{d}", .{ pre, i });
                layer.* = .{
                    .attention = try loader.hc(try loader.name("{s}.attn_hyper_connection", .{stem}), true),
                    .mlp = try loader.hc(try loader.name("{s}.mlp_hyper_connection", .{stem}), true),
                    .dense = .{ .gate = try loader.matrix(stem, "mlp.gate_proj", c.intermediate, c.hidden), .up = try loader.matrix(stem, "mlp.up_proj", c.intermediate, c.hidden), .down = try loader.matrix(stem, "mlp.down_proj", c.hidden, c.intermediate) },
                    .mixer = switch (c.kinds[i]) {
                        .linear_attention => .{ .linear = try loader.linear(stem) },
                        .indexed_attention => .{ .indexed = try loader.indexed(stem) },
                    },
                    .ple = null,
                };
                if (c.ple[i]) {
                    layer.ple = try loader.ple(stem, ordinal);
                    ordinal += 1;
                }
            }
            const embedding = try loader.name("{s}embed_tokens", .{pre});
            const head = if (c.tie_embeddings) embedding else if (w.reader.namespace == .mlx) "language_model.lm_head" else "lm_head";
            const readout = try loader.hc(try loader.name("{s}hyper_connection_mixer", .{pre}), false);
            const output_head = try loader.matrix(head, "", c.vocab, c.hidden);
            // Finish every arena allocation before copying its final ownership state into the result.
            return .{ .storage = arena, .weights = w, .c = c, .layers = layers, .readout = readout, .head = output_head, .embedding = embedding };
        }
        pub fn deinit(s: *Self) void {
            s.storage.deinit();
            s.* = undefined;
        }
        const Loader = struct {
            a: std.mem.Allocator,
            w: *Weights,
            p: *cfg.Problem,
            fn name(l: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
                return std.fmt.allocPrint(l.a, fmt, args);
            }
            fn join(l: *Loader, stem: []const u8, tail: []const u8) ![]const u8 {
                return if (tail.len == 0) stem else l.name("{s}.{s}", .{ stem, tail });
            }
            fn matrix(l: *Loader, stem: []const u8, tail: []const u8, n: usize, k: usize) !Matrix {
                const key = try l.join(stem, tail);
                const m = l.w.matrices.get(key) orelse return l.p.reject(key);
                if (m.params.n != n or m.params.k != k) return l.p.reject(key);
                return m;
            }
            fn vector(l: *Loader, stem: []const u8, tail: []const u8) !B.Buffer {
                const key = try l.join(stem, tail);
                return l.w.vectors.get(key) orelse l.p.reject(key);
            }
            fn norm(l: *Loader, stem: []const u8, tail: []const u8, centered: bool) !Norm {
                const key = try l.join(stem, tail);
                return .{ .buffer = try l.vector(stem, tail), .centered = centered and l.w.c.norm_weights == .hf_gamma, .mid_bf16 = (try l.w.reader.get(key)).dtype == .bf16 };
            }
            fn hc(l: *Loader, stem: []const u8, inject: bool) !Hc {
                const c = l.w.c;
                return .{ .norm = try l.norm(stem, "hc_norm.weight", true), .down = try l.matrix(stem, "input_mix_weight_down", c.lowrank, c.wide()), .up = try l.matrix(stem, "input_mix_weight_up", c.wide(), c.lowrank), .inject = if (inject) try l.matrix(stem, "block_inject_weight", c.streams, c.wide()) else null };
            }
            fn linear(l: *Loader, layer: []const u8) !Linear {
                const c = l.w.c;
                const stem = try l.join(layer, "linear_attn");
                return .{
                    .qkv = try l.matrix(stem, "in_proj_qkv", c.convWidth(), c.hidden),
                    .z = try l.matrix(stem, "in_proj_z", c.valueWidth(), c.hidden),
                    .a = try l.matrix(stem, "in_proj_a", c.value_heads, c.hidden),
                    .b = try l.matrix(stem, "in_proj_b", c.value_heads, c.hidden),
                    .out = try l.matrix(stem, "out_proj", c.hidden, c.valueWidth()),
                    .conv = try l.vector(stem, "conv1d.weight"),
                    .conv_bf16 = (try l.w.reader.get(try l.join(stem, "conv1d.weight"))).dtype == .bf16,
                    .alog = try l.vector(stem, "A_log"),
                    .dt = try l.vector(stem, "dt_bias"),
                    .norm = try l.norm(stem, "norm.weight", false),
                };
            }
            fn indexed(l: *Loader, layer: []const u8) !Indexed {
                const c = l.w.c;
                const stem = try l.join(layer, "self_attn");
                return .{
                    .q = try l.matrix(stem, "q_proj", 2 * c.heads * c.head_dim, c.hidden),
                    .k = try l.matrix(stem, "k_proj", c.kv_heads * c.head_dim, c.hidden),
                    .v = try l.matrix(stem, "v_proj", c.kv_heads * c.head_dim, c.hidden),
                    .out = try l.matrix(stem, "o_proj", c.hidden, c.heads * c.head_dim),
                    .index = try l.matrix(stem, "indexer.index_qk_proj", (c.index_heads + 1) * c.index_dim, c.hidden),
                    .qn = try l.norm(stem, "q_norm.weight", true),
                    .kn = try l.norm(stem, "k_norm.weight", true),
                    .iqn = try l.norm(stem, "indexer.q_layernorm.weight", true),
                    .ikn = try l.norm(stem, "indexer.k_layernorm.weight", true),
                };
            }
            fn ple(l: *Loader, layer: []const u8, ordinal: usize) !Ple {
                const c = l.w.c;
                const stem = try l.join(layer, "ple");
                const policy = try ngram.Policy.init(l.a, c.ngram, c.ngram_heads, c.vocab, c.ple_eos, c.ngram_vocab, c.seed, ordinal);
                return .{
                    .key = try l.matrix(stem, "key_proj", c.wide(), c.ple_dim),
                    .value = try l.matrix(stem, "value_proj", c.hidden, c.ple_dim),
                    .key_norm = try l.norm(stem, "norm_key.weight", true),
                    .query_norm = try l.norm(stem, "norm_query.weight", true),
                    .conv_norm = try l.norm(stem, "norm_conv.weight", true),
                    .conv = try l.vector(stem, "conv1d.weight"),
                    .table = try l.join(stem, "ple_embedding.ngram_embedding"),
                    .policy = policy.view(),
                };
            }
        };
    };
}
