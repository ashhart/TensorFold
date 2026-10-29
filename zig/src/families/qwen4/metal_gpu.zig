//! Dense graph operations on one Metal inference session; all layer arithmetic stays on the device.
const std = @import("std");
const cfg = @import("config.zig");
const core = @import("core");
pub fn Gpu(comptime M: type) type {
    const B = @import("metal_storage.zig").Backend(M);
    const Model = @import("model.zig").Model(B);
    const Scratch = @import("arena.zig").Arena(B);
    const Tensor = Scratch.Tensor;
    const Projection = core.weight_only_metal_adapter(core.weight_only, M);
    const Activation = core.hc_activation_metal_adapter(core.hc_activation, M);
    const Pointwise = core.hc_pointwise_metal_adapter(core.hc_pointwise, M);
    const Hc = @import("metal_hc.zig").Kernels(M);
    const Gdn = @import("metal_gdn.zig").Kernels(M);
    const Attention = @import("metal_attention.zig").Kernels(M);
    const Ple = @import("metal_ple.zig").Kernels(M);
    return struct {
        const Self = @This();
        pub const Storage = B;
        pub const kind: @import("native_admission.zig").Backend = .metal;
        backend: *B,
        c: *const cfg.Config,
        rounding: core.hc_pointwise.Rounding,
        projection: ?Projection.Kernels = null,
        activation: ?Activation.Kernels = null,
        pointwise: ?Pointwise.Kernels = null,
        hc: ?Hc = null,
        gdn: ?Gdn = null,
        attention: ?Attention = null,
        lexical: ?Ple = null,
        rope: ?M.Buffer = null,
        trace: ?@import("node_trace.zig").Sink(B) = null,
        pub fn load(a: std.mem.Allocator, b: *B, c: *const cfg.Config, p: *cfg.Problem, rounding: core.hc_pointwise.Rounding) !Self {
            _ = b.commands orelse return error.Qwen4CommandSessionMissing;
            const g = try @import("native_admission.zig").check(c, .metal, p);
            var out: Self = .{ .backend = b, .c = c, .rounding = rounding };
            errdefer out.drop();
            const body = @embedFile("../../../kernels/metal/core/weight_only.h");
            const hc_body = @embedFile("../../../kernels/metal/core/hc_activation.h");
            const projection_text = try Projection.source(a, body, @embedFile("../../../kernels/metal/core/weight_only.metal"), false);
            defer a.free(projection_text);
            out.projection = try Projection.Kernels.load(b.device, projection_text, false);
            const activation_text = try Activation.source(a, body, hc_body, @embedFile("../../../kernels/metal/core/hc_activation.metal"), false);
            defer a.free(activation_text);
            out.activation = try Activation.Kernels.load(b.device, activation_text, false);
            const pointwise_text = try Pointwise.source(a, body, hc_body, @embedFile("../../../kernels/metal/core/hc_pointwise.metal"), false);
            defer a.free(pointwise_text);
            out.pointwise = try Pointwise.Kernels.load(b.device, pointwise_text, false);
            out.hc = try Hc.load(a, b.device, .{ .hidden = c.hidden, .streams = c.streams });
            var linear = false;
            var has_indexed = false;
            for (c.kinds[0..c.layers]) |layer_kind| {
                linear = linear or layer_kind == .linear_attention;
                has_indexed = has_indexed or layer_kind == .indexed_attention;
            }
            if (linear) {
                const text = try @import("../flashnext/hybrid_sources.zig").Gdn.metal(a, @embedFile("../../../kernels/metal/decode/fn_gdn.metal"), @embedFile("../../../kernels/metal/flashnext/q4_gdn_step_af87f500308a9afb.metal"), c.key_heads, c.value_heads, c.gate == .sigmoid);
                defer a.free(text);
                out.gdn = try Gdn.load(b.device, text, c.key_heads, c.value_heads, if (c.gate == .silu) .silu else .sigmoid);
            }
            if (has_indexed) {
                out.attention = try Attention.load(a, b.device, g);
                const table = try a.alloc(f32, 3 + c.rotary / 2);
                defer a.free(table);
                try @import("rotary_table.zig").fill(c.theta, c.rotary, table);
                out.rope = try b.upload(std.mem.sliceAsBytes(table));
            }
            if (c.ple_count > 0) out.lexical = try Ple.load(a, b.device, c);
            return out;
        }
        fn drop(g: *Self) void {
            inline for (.{ "lexical", "attention", "gdn", "hc", "pointwise", "activation", "projection" }) |field| if (@field(g, field)) |*k| k.deinit();
            if (g.rope) |b| g.backend.release(b);
            g.* = undefined;
        }
        pub fn close(g: *Self) !void {
            try g.backend.drain();
            g.drop();
        }
        pub fn deinitDrained(g: *Self) void {
            g.drop();
        }
        pub fn begin(g: *Self) !void {
            try g.backend.commands.?.begin();
        }
        pub fn submit(g: *Self) !void {
            try g.backend.commands.?.submit();
        }
        pub fn discard(g: *Self) void {
            g.backend.commands.?.discard();
        }
        fn encoder(g: *Self) !M.ComputeEncoder {
            return (g.backend.commands orelse return error.Qwen4CommandSessionMissing).encoder();
        }
        fn view(comptime V: type, t: Tensor) !V {
            return .{ .buffer = t.buffer, .offset = t.offset, .len = try t.bytes() };
        }
        fn raw(comptime V: type, b: M.Buffer) V {
            return .{ .buffer = b, .len = b.length() };
        }
        pub fn project(g: *Self, w: Model.Matrix, x: Tensor, y: Tensor) !void {
            var p = w.params;
            p.rows = @intCast(x.rows);
            if (y.rows != x.rows or y.width != w.params.n) return error.Qwen4ProjectionShape;
            try g.projection.?.encode(try g.encoder(), p, try view(Projection.View, x), raw(Projection.View, w.weight), raw(Projection.View, w.scales), raw(Projection.View, w.biases), try view(Projection.View, y));
        }
        pub fn hcNorm(g: *Self, n: Model.Norm, x: Tensor, sums: Tensor, y: Tensor, eps: f32) !void {
            if (!n.prepared) return error.Qwen4NormNotPrepared;
            const p: @import("hc_layout.zig").Call = .{ .rows = x.rows, .eps = eps };
            try g.hc.?.encode(try g.encoder(), .sum, p, &.{ try view(Hc.View, x), try view(Hc.View, sums) });
            try g.hc.?.encode(try g.encoder(), .norm, p, &.{ try view(Hc.View, x), try view(Hc.View, sums), raw(Hc.View, n.buffer), try view(Hc.View, y) });
        }
        pub fn hcActivation(g: *Self, x: Tensor, y: Tensor, streams: usize, injection: bool) !void {
            try g.activation.?.encode(try g.encoder(), .{ .rows = @intCast(x.rows), .cols = @intCast(x.width), .streams = @intCast(streams), .injection = @intFromBool(injection) }, try view(Activation.View, x), try view(Activation.View, y));
        }
        pub fn hcMix(g: *Self, up: Tensor, x: Tensor, y: Tensor, _: Tensor) !void {
            try g.hc.?.encode(try g.encoder(), .mix, .{ .rows = x.rows }, &.{ try view(Hc.View, up), try view(Hc.View, x), try view(Hc.View, y) });
        }
        pub fn writeback(g: *Self, x: Tensor, branch: Tensor, injection: Tensor, y: Tensor) !void {
            const p: core.hc_pointwise.Params = .{ .rows = @intCast(x.rows), .hidden = @intCast(g.c.hidden), .streams = @intCast(g.c.streams), .op = .branch, .round_mid = g.rounding };
            try g.pointwise.?.encode(try g.encoder(), p, .{ try view(Pointwise.View, x), try view(Pointwise.View, branch), try view(Pointwise.View, injection), try view(Pointwise.View, y) });
        }
        pub fn swiglu(g: *Self, gate: Tensor, up: Tensor, y: Tensor) !void {
            try g.hc.?.encode(try g.encoder(), .act, .{ .rows = gate.rows, .width = gate.width }, &.{ try view(Hc.View, gate), try view(Hc.View, up), try view(Hc.View, y) });
        }
        fn stateView(comptime V: type, v: anytype) V {
            return .{ .buffer = v.buffer, .offset = v.offset, .len = v.len };
        }
        fn note(g: *Self, layer: usize, scope: []const u8, node: []const u8, v: anytype, rows: usize, width: usize, element: @import("node_trace.zig").Kind, first: usize) !void {
            const sink = g.trace orelse return;
            var key: [192]u8 = undefined;
            try sink.view(try @import("node_trace.zig").label(&key, layer, scope, node), v, rows, width, element, first);
        }
        pub fn delta(g: *Self, w: Model.Linear, state: anytype, layer: usize, x: Tensor, y: Tensor, _: Tensor) !void {
            const l = state.layout.layers[layer];
            const p: @import("cuda_gdn.zig").Params = .{ .rows = x.rows, .keys = g.c.key_heads, .values = g.c.value_heads, .eps = g.c.eps, .gate = if (g.c.gate == .silu) .silu else .sigmoid, .row_stride = state.layout.state_bytes, .weight_flags = @import("cuda_gdn.zig").weightFlags(w.conv_bf16, w.norm.mid_bf16) };
            try g.gdn.?.encode(try g.encoder(), p, .{
                try view(Gdn.View, x),                               stateView(Gdn.View, try state.current(l.conv)), stateView(Gdn.View, try state.current(l.recurrent)),
                raw(Gdn.View, w.conv),                               raw(Gdn.View, w.alog),                          raw(Gdn.View, w.dt),
                raw(Gdn.View, w.norm.buffer),                        try view(Gdn.View, y),                          stateView(Gdn.View, try state.rowSpan(l.conv)),
                stateView(Gdn.View, try state.rowSpan(l.recurrent)),
            });
            if (g.trace != null) for (0..x.rows) |r| {
                const conv = try state.row(r, l.conv);
                const recurrent = try state.row(r, l.recurrent);
                try g.note(layer, "linear_attn", "conv_tail", conv, 1, conv.len / 2, .bf16, r);
                try g.note(layer, "linear_attn", "recurrent", recurrent, 1, recurrent.len / 4, .f32, r);
            };
        }
        pub fn indexed(g: *Self, scratch: *Scratch, w: Model.Indexed, state: anytype, layer: usize, x: Tensor, y: Tensor) !void {
            const k = &g.attention.?;
            const p: @import("cuda_attention.zig").Call = .{ .rows = x.rows, .start = state.base_pos, .capacity = state.layout.capacity, .eps = g.c.eps };
            const z = try @import("metal_attention.zig").sizes(k.geometry, p);
            const l = state.layout.layers[layer];
            var v: [18]Attention.View = undefined;
            v[0] = try view(Attention.View, x);
            v[1] = raw(Attention.View, w.qn.buffer);
            v[2] = raw(Attention.View, w.kn.buffer);
            v[3] = raw(Attention.View, w.iqn.buffer);
            v[4] = raw(Attention.View, w.ikn.buffer);
            v[5] = raw(Attention.View, g.rope.?);
            v[8] = stateView(Attention.View, try state.kv(l.key));
            v[9] = stateView(Attention.View, try state.kv(l.value));
            v[11] = stateView(Attention.View, try state.kv(l.index));
            v[12] = stateView(Attention.View, try state.kv(l.pooled));
            const mark = scratch.used;
            defer scratch.rewind(mark) catch unreachable;
            for ([_]usize{ 6, 7, 10, 13, 14, 15, 16, 17 }) |i| {
                const buffer = try scratch.take(1, (z.bytes[i] + 3) / 4, .f32);
                v[i] = try view(Attention.View, buffer);
            }
            try k.encode(g.backend.commands.?, p, v);
            if (g.trace != null) {
                try g.note(layer, "self_attn", "query", v[6], x.rows, g.c.heads * 256, .bf16, 0);
                try g.note(layer, "self_attn", "key", v[7], x.rows, g.c.kv_heads * 256, .bf16, 0);
                try g.note(layer, "self_attn", "index_query", v[10], x.rows, g.c.index_heads * g.c.index_dim, .bf16, 0);
                const plan = try @import("attention_plan.zig").Plan.init(k.geometry, p);
                for (0..plan.pool_count) |i| {
                    var pooled = v[12];
                    pooled.offset += (plan.first_pool + i) * g.c.index_dim * 2;
                    try g.note(layer, "self_attn", "pooled", pooled, 1, g.c.index_dim, .bf16, (plan.first_pool + i + 1) * g.c.index_ratio - 1);
                }
                for (0..x.rows) |r| if (plan.complete[r] > 0) {
                    var scores = v[13];
                    scores.offset += r * z.score_stride * 4;
                    try g.note(layer, "self_attn", "scores", scores, 1, @intCast(plan.complete[r]), .f32, r);
                };
                try g.note(layer, "self_attn", "parts_output", v[15], x.rows, g.c.heads * @import("metal_attention_source.zig").parts * 256, .f32, 0);
                try g.note(layer, "self_attn", "parts_stats", v[16], x.rows, g.c.heads * @import("metal_attention_source.zig").parts * 2, .f32, 0);
                try g.note(layer, "self_attn", "attention", v[17], x.rows, y.width, .bf16, 0);
                // No selection writes or IDS reads occur in this two-row dense range.
                var sparse: usize = 0;
                for (plan.sparse[0..p.rows]) |flag| sparse += @intFromBool(flag != 0);
                std.debug.print("QWEN4_SELECTION start={d} dense_rows={d} sparse_rows={d}\n", .{ p.start, p.rows - sparse, sparse });
            }
            const pt: core.hc_pointwise.Params = .{ .rows = @intCast(x.rows), .hidden = @intCast(y.width), .streams = 1, .op = .output_gate, .round_mid = g.rounding, .gate_stride = @intCast(x.width), .gate_dim = 256 };
            try g.pointwise.?.encode(try g.encoder(), pt, .{ .{ .buffer = v[17].buffer, .offset = v[17].offset, .len = v[17].len }, try view(Pointwise.View, x), raw(Pointwise.View, w.qn.buffer), try view(Pointwise.View, y) });
        }
        pub fn ple(g: *Self, _: *Scratch, w: Model.Ple, state: anytype, layer: usize, projections: Tensor, h: Tensor, gated: Tensor, normed: Tensor, cin: Tensor) !void {
            const l = state.layout.layers[layer];
            try g.lexical.?.encode(g.backend.commands.?, h.rows, .{
                try view(Ple.View, projections),               try view(Ple.View, h),      raw(Ple.View, w.key_norm.buffer), raw(Ple.View, w.query_norm.buffer), raw(Ple.View, w.conv_norm.buffer),
                try view(Ple.View, gated),                     try view(Ple.View, normed), try view(Ple.View, cin),          raw(Ple.View, w.conv),              stateView(Ple.View, try state.current(l.ple)),
                stateView(Ple.View, try state.rowSpan(l.ple)),
            }, state.layout.state_bytes);
        }
    };
}
