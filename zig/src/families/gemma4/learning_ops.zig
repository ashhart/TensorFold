//! The trainer binds shared native primitives through explicit views, never host inference arithmetic.
const std = @import("std");
pub fn Ops(comptime R: type) type {
    const Core = R.Kernels.Core;
    const Ref = R.Ref;
    return struct {
        const Self = @This();
        pub const Binding = struct { usize, Ref };
        r: *R,
        pass: R.Backend.Pass,
        dummy: Ref,
        pub fn raw(_: Self, b: R.Backend.Buffer) Ref {
            return .{ .buffer = b, .len = R.Kernels.length(b) };
        }
        fn refs(o: Self, bindings: []const Binding) ![11]Ref {
            var out: [11]Ref = @splat(o.dummy);
            var seen: u16 = 0;
            for (bindings) |v| {
                if (v[0] >= out.len) return error.GemmaTrainingBinding;
                const bit = @as(u16, 1) << @intCast(v[0]);
                if (seen & bit != 0) return error.GemmaTrainingBinding;
                seen |= bit;
                out[v[0]] = v[1];
            }
            return out;
        }
        pub fn row(o: Self, op: Core.living_rows.Op, p: Core.living_rows.Params, bindings: []const Binding) !void {
            try o.r.kernels.row(o.r.backend, o.pass, op, try Core.living_rows.finish(op, p), try o.refs(bindings));
        }
        pub fn gemma(o: Self, op: Core.living_gemma.Op, p: Core.living_rows.Params, bindings: []const Binding) !void {
            try o.r.kernels.gemma(o.r.backend, o.pass, op, try Core.living_gemma.finish(op, p), try o.refs(bindings));
        }
        pub fn qsa(o: Self, op: Core.living_qsa.Op, p: Core.living_rows.Params, bindings: []const Binding) !void {
            try o.r.kernels.qsa(o.r.backend, o.pass, op, try Core.living_qsa.finish(op, p), try o.refs(bindings));
        }
        pub fn zero(o: Self, v: Ref) !void {
            var p: Core.living_rows.Params = .{};
            p.u[0] = 1;
            p.u[1] = @intCast(v.len / 4);
            try o.gemma(.zero, p, &.{.{ 8, v }});
        }
        pub fn move(o: Self, from: Ref, to: Ref, add: bool, factor: f32) !void {
            if (from.len != to.len) return error.GemmaTrainingRange;
            var p: Core.living_rows.Params = .{};
            p.u[0] = 1;
            p.u[1] = @intCast(from.len / 4);
            p.u[2] = @intFromBool(add);
            p.f[0] = factor;
            try o.gemma(.move, p, &.{ .{ 0, from }, .{ 8, to } });
        }
    };
}
