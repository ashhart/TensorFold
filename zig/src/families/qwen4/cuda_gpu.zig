//! CUDA implements the same dense graph and raw weight products on the engine's own driver and inference stream.
const std = @import("std");
const cfg = @import("config.zig");
const core = @import("core");
pub fn Gpu(comptime C: type) type {
    const B = @import("cuda_storage.zig").Backend(C);
    const Model = @import("model.zig").Model(B);
    const Scratch = @import("arena.zig").Arena(B);
    const Tensor = Scratch.Tensor;
    const Projection = core.weight_only_cuda_adapter(core.weight_only, C);
    const Activation = core.hc_activation_cuda_adapter(core.hc_activation, C);
    const Pointwise = core.hc_pointwise_cuda_adapter(core.hc_pointwise, C);
    const Hc = @import("cuda_hc.zig").Kernels(C);
    const Gdn = @import("cuda_gdn.zig").Kernels(C);
    const Attention = @import("cuda_attention.zig").Kernels(C);
    const Ple = @import("cuda_ple.zig").Kernels(C);
    return struct {
        const Self = @This();
        pub const Storage = B;
        pub const kind: @import("native_admission.zig").Backend = .cuda;
        pub const Images = struct { projection: []const u8, hc_scalar: []const u8, hc: []const u8, gdn: []const u8, attention: []const u8, batch: []const u8, ple: []const u8 };
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
        rope: ?C.DeviceBuffer = null,
        pub fn load(a: std.mem.Allocator, b: *B, c: *const cfg.Config, p: *cfg.Problem, images: Images, rounding: core.hc_pointwise.Rounding) !Self {
            try b.enter();
            if (b.stream.d != b.driver) return error.Qwen4DeviceContext;
            _ = try @import("native_admission.zig").check(c, .cuda, p);
            var out: Self = .{ .backend = b, .c = c, .rounding = rounding };
            errdefer out.drop();
            out.projection = try Projection.Kernels.load(b.driver, images.projection, false);
            out.activation = try Activation.Kernels.load(b.driver, images.hc_scalar, false);
            out.pointwise = try Pointwise.Kernels.load(b.driver, images.hc_scalar, false);
            out.hc = try Hc.load(b.driver, images.hc, .{ .hidden = c.hidden, .streams = c.streams });
            try out.hc.?.bindDense(try out.activation.?.module.function("qwen4_dense_act"));
            var linear = false;
            var sparse = false;
            for (c.kinds[0..c.layers]) |layer_kind| {
                linear = linear or layer_kind == .linear_attention;
                sparse = sparse or layer_kind == .indexed_attention;
            }
            if (linear) out.gdn = try Gdn.load(b.driver, images.gdn);
            if (sparse) {
                out.attention = try Attention.load(b.driver, images.attention, images.batch);
                const table = try a.alloc(f32, @max(@as(usize, 1), c.rotary / 2));
                defer a.free(table);
                try @import("rotary_table.zig").fill(c.theta, c.rotary, table);
                out.rope = try b.upload(std.mem.sliceAsBytes(table));
            }
            if (c.ple_count > 0) out.lexical = try Ple.load(b.driver, images.ple, c);
            return out;
        }
        fn drop(g: *Self) void {
            inline for (.{ "lexical", "gdn", "hc", "pointwise", "activation", "projection" }) |field| if (@field(g, field)) |*k| k.module.unload();
            if (g.attention) |*k| {
                k.batch.unload();
                k.attention.unload();
            }
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
            try g.backend.enter();
        }
        pub fn submit(_: *Self) !void {}
        pub fn discard(_: *Self) void {}
        fn view(comptime V: type, t: *const Tensor) !V {
            return .{ .buffer = &t.buffer, .offset = t.offset, .len = try t.bytes() };
        }
        fn raw(comptime V: type, b: *const C.DeviceBuffer) V {
            return .{ .buffer = b, .len = b.len };
        }
        pub fn project(g: *Self, w: Model.Matrix, x: Tensor, y: Tensor) !void {
            var p = w.params;
            p.rows = @intCast(x.rows);
            if (y.rows != x.rows or y.width != w.params.n) return error.Qwen4ProjectionShape;
            try g.projection.?.encode(g.backend.stream, p, try view(Projection.View, &x), raw(Projection.View, &w.weight), raw(Projection.View, &w.scales), raw(Projection.View, &w.biases), try view(Projection.View, &y));
        }
        pub fn hcNorm(g: *Self, n: Model.Norm, x: Tensor, sums: Tensor, y: Tensor, eps: f32) !void {
            if (!n.prepared) return error.Qwen4NormNotPrepared;
            const p: @import("hc_layout.zig").Call = .{ .rows = x.rows, .eps = eps };
            try g.hc.?.encode(g.backend.stream, .sum, p, &.{ try view(Hc.View, &x), try view(Hc.View, &sums) });
            try g.hc.?.encode(g.backend.stream, .norm, p, &.{ try view(Hc.View, &x), try view(Hc.View, &sums), raw(Hc.View, &n.buffer), try view(Hc.View, &y) });
        }
        pub fn hcActivation(g: *Self, x: Tensor, y: Tensor, streams: usize, injection: bool) !void {
            try g.activation.?.encode(g.backend.stream, .{ .rows = @intCast(x.rows), .cols = @intCast(x.width), .streams = @intCast(streams), .injection = @intFromBool(injection) }, try view(Activation.View, &x), try view(Activation.View, &y));
        }
        pub fn hcMix(g: *Self, up: Tensor, x: Tensor, y: Tensor, sums: Tensor) !void {
            try g.hc.?.encode(g.backend.stream, .mix, .{ .rows = x.rows }, &.{ try view(Hc.View, &up), try view(Hc.View, &x), try view(Hc.View, &y), try view(Hc.View, &sums) });
        }
        pub fn writeback(g: *Self, x: Tensor, branch: Tensor, injection: Tensor, y: Tensor) !void {
            try g.pointwise.?.encode(g.backend.stream, .{ .rows = @intCast(x.rows), .hidden = @intCast(g.c.hidden), .streams = @intCast(g.c.streams), .op = .branch, .round_mid = g.rounding }, .{ try view(Pointwise.View, &x), try view(Pointwise.View, &branch), try view(Pointwise.View, &injection), try view(Pointwise.View, &y) });
        }
        pub fn swiglu(g: *Self, gate: Tensor, up: Tensor, y: Tensor) !void {
            try g.hc.?.encode(g.backend.stream, .act, .{ .rows = gate.rows, .width = gate.width }, &.{ try view(Hc.View, &gate), try view(Hc.View, &up), try view(Hc.View, &y) });
        }
        fn range(g: *Self, b: C.DeviceBuffer, offset: usize, len: usize) !core.weight_only.Range {
            if (b.d != g.backend.driver) return error.Qwen4DeviceContext;
            if (offset > b.len or len > b.len - offset) return error.Qwen4StateRange;
            return .{ .address = try std.math.add(usize, @intCast(b.ptr), offset), .len = len };
        }
        pub fn delta(g: *Self, w: Model.Linear, state: anytype, layer: usize, x: Tensor, y: Tensor, sums: Tensor) !void {
            const l = state.layout.layers[layer];
            const cs = try state.current(l.conv);
            const input = try state.current(l.recurrent);
            const out = try state.rowSpan(l.recurrent);
            const p: @import("cuda_gdn.zig").Params = .{ .rows = x.rows, .keys = g.c.key_heads, .values = g.c.value_heads, .eps = g.c.eps, .gate = if (g.c.gate == .silu) .silu else .sigmoid, .row_stride = state.layout.state_bytes, .weight_flags = @import("cuda_gdn.zig").weightFlags(w.conv_bf16, w.norm.mid_bf16) };
            try g.gdn.?.encode(g.backend.stream, p, .{
                try g.range(x.buffer, x.offset, try x.bytes()),          try g.range(cs.buffer, cs.offset, cs.len),    try g.range(w.conv, 0, w.conv.len),               try g.range(input.buffer, input.offset, input.len),
                try g.range(w.alog, 0, w.alog.len),                      try g.range(w.dt, 0, w.dt.len),               try g.range(w.norm.buffer, 0, w.norm.buffer.len), try g.range(y.buffer, y.offset, try y.bytes()),
                try g.range(sums.buffer, sums.offset, try sums.bytes()), try g.range(out.buffer, out.offset, out.len),
            });
            try @import("cuda_gdn.zig").saveConv(g.backend, p, try state.rowSpan(l.conv), .{ .buffer = x.buffer, .offset = x.offset, .len = try x.bytes() }, cs);
        }
        fn stateView(v: anytype) Tensor {
            return .{ .buffer = v.buffer, .offset = v.offset, .rows = 1, .width = (v.len + 3) / 4, .kind = .f32 };
        }
        pub fn indexed(g: *Self, scratch: *Scratch, w: Model.Indexed, state: anytype, layer: usize, x: Tensor, y: Tensor) !void {
            var problem: cfg.Problem = .{};
            const geometry = try @import("cuda_attention.zig").Geometry.of(g.c, &problem);
            const p: @import("cuda_attention.zig").Call = .{ .rows = x.rows, .start = state.base_pos, .capacity = state.layout.capacity, .eps = g.c.eps, .gate = false };
            const z = try @import("cuda_attention.zig").sizes(geometry, p);
            const l = state.layout.layers[layer];
            const mark = scratch.used;
            defer scratch.rewind(mark) catch unreachable;
            var buffers: [23]C.DeviceBuffer = undefined;
            var views: [23]Attention.View = undefined;
            for ([_]usize{ 1, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 }) |i| {
                const span = try scratch.take(1, (z.bytes[i] + 3) / 4, .f32);
                buffers[i] = span.buffer;
                views[i] = .{ .buffer = &buffers[i], .offset = span.offset, .len = try span.bytes() };
            }
            buffers[0] = x.buffer;
            views[0] = try view(Attention.View, &x);
            buffers[2] = w.qn.buffer;
            buffers[3] = w.kn.buffer;
            buffers[4] = w.iqn.buffer;
            buffers[5] = w.ikn.buffer;
            buffers[6] = g.rope.?;
            for (2..7) |i| views[i] = raw(Attention.View, &buffers[i]);
            const kv = [_]@import("state.zig").State(B).View{ try state.kv(l.key), try state.kv(l.value), try state.kv(l.index), try state.kv(l.pooled) };
            for (kv, [_]usize{ 8, 9, 11, 12 }) |span, i| {
                buffers[i] = span.buffer;
                views[i] = .{ .buffer = &buffers[i], .offset = span.offset, .len = span.len };
            }
            const start: i32 = @intCast(state.base_pos);
            try g.backend.write(buffers[1], views[1].offset, std.mem.asBytes(&start));
            try g.attention.?.encode(g.backend.stream, geometry, p, views);
            try g.pointwise.?.encode(g.backend.stream, .{ .rows = @intCast(x.rows), .hidden = @intCast(y.width), .streams = 1, .op = .output_gate, .round_mid = g.rounding, .gate_stride = @intCast(x.width), .gate_dim = 256 }, .{ .{ .buffer = views[20].buffer, .offset = views[20].offset, .len = views[20].len }, try view(Pointwise.View, &x), raw(Pointwise.View, &w.qn.buffer), try view(Pointwise.View, &y) });
        }
        pub fn ple(g: *Self, scratch: *Scratch, w: Model.Ple, state: anytype, layer: usize, projections: Tensor, h: Tensor, gated: Tensor, normed: Tensor, cin: Tensor) !void {
            const l = state.layout.layers[layer];
            const tail = try state.current(l.ple);
            const kept = try state.rowSpan(l.ple);
            const sums = try scratch.take(h.rows, g.c.streams, .f32);
            const buffers = [_]C.DeviceBuffer{ projections.buffer, h.buffer, w.key_norm.buffer, w.query_norm.buffer, w.conv_norm.buffer, gated.buffer, normed.buffer, sums.buffer, w.conv, tail.buffer, cin.buffer, kept.buffer };
            var views: [12]Ple.View = undefined;
            for (&views, &buffers) |*v, *b| v.* = raw(Ple.View, b);
            views[0] = try view(Ple.View, &projections);
            views[1] = try view(Ple.View, &h);
            views[5] = try view(Ple.View, &gated);
            views[6] = try view(Ple.View, &normed);
            views[7] = try view(Ple.View, &sums);
            views[10] = try view(Ple.View, &cin);
            views[9] = .{ .buffer = &buffers[9], .offset = tail.offset, .len = tail.len };
            views[11] = .{ .buffer = &buffers[11], .offset = kept.offset, .len = kept.len };
            try g.lexical.?.encode(g.backend, h.rows, views, state.layout.state_bytes);
        }
    };
}
