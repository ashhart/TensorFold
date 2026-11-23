//! Training dispatches keep explicit byte views and use only native backend row operators and ordered copies.
const std = @import("std");
const core = @import("core");
pub fn Ops(comptime E: type, comptime K: type) type {
    const Ref = @import("train_tensor.zig").Ref(E.Storage);
    return struct {
        const Self = @This();
        pub const Binding = struct { usize, Ref };
        e: *E,
        k: *K,
        dummy: Ref,
        pub fn raw(_: Self, b: E.Storage.Buffer) Ref {
            return .{ .buffer = b, .len = K.length(b) };
        }
        fn views(o: Self, bindings: []const Binding) ![11]Ref {
            var out: [11]Ref = @splat(o.dummy);
            var seen: u16 = 0;
            for (bindings) |v| {
                if (v[0] >= out.len) return error.Qwen4TrainingBinding;
                const bit = @as(u16, 1) << @intCast(v[0]);
                if (seen & bit != 0) return error.Qwen4TrainingBinding;
                seen |= bit;
                out[v[0]] = v[1];
            }
            return out;
        }
        pub fn row(o: Self, op: core.living_rows.Op, p: core.living_rows.Params, bindings: []const Binding) !void {
            try o.k.row(&o.e.gpu, op, try core.living_rows.finish(op, p), try o.views(bindings));
        }
        pub fn gdn(o: Self, op: core.living_gdn.Op, p: core.living_gdn.Params, bindings: []const Binding) !void {
            try o.k.gdn(&o.e.gpu, op, try core.living_gdn.finish(op, p), try o.views(bindings));
        }
        pub fn qsa(o: Self, op: core.living_qsa.Op, p: core.living_qsa.Params, bindings: []const Binding) !void {
            try o.k.qsa(&o.e.gpu, op, try core.living_qsa.finish(op, p), try o.views(bindings));
        }
        pub fn zero(o: Self, ref: Ref) !void {
            try o.e.backend.zero(ref.buffer, ref.offset, ref.len);
        }
        pub fn copy(o: Self, from: Ref, to: Ref) !void {
            if (from.len != to.len) return error.Qwen4TrainingCopy;
            try o.e.backend.copy(to.buffer, to.offset, from.buffer, from.offset, from.len);
        }
        pub fn matrix(o: Self, matrix_: anytype, grad: Ref, out: Ref, count: usize, stride: usize, offset: usize, add: bool) !void {
            const m = matrix_.params;
            if (m.format != .bf16 and (m.format != .mlx4 or m.group != 64 or m.scale_kind != .bf16)) return error.Qwen4TrainingFormat;
            var p: core.living_rows.Params = .{};
            p.u[0] = @intCast(count);
            p.u[1] = @intCast(m.n);
            p.u[2] = @intCast(m.k);
            p.u[3] = @intFromBool(m.format == .mlx4);
            p.u[4] = @intCast(m.group);
            p.u[5] = @intFromBool(add);
            p.u[6] = @intCast(stride);
            p.u[7] = @intCast(offset);
            try o.row(.linear, p, &.{ .{ 0, grad }, .{ 1, o.raw(matrix_.weight) }, .{ 2, o.raw(matrix_.scales) }, .{ 3, o.raw(matrix_.biases) }, .{ 8, out } });
        }
    };
}
