//! Metal training resolves pipelines before recording and uses the serving device and ordered command queue.
const std = @import("std");
pub fn Kernels(comptime M: type, comptime Core_: type, comptime Source: type) type {
    const B = Core_.dense_storage_metal.Backend(M);
    const Rows = Core_.living_rows_metal.Adapter(M, Core_.living_rows, Source);
    const Gemma = Core_.living_rows_metal.Adapter(M, Core_.living_gemma, Source);
    const Qsa = Core_.living_rows_metal.Adapter(M, Core_.living_qsa, Source);
    return struct {
        const Self = @This();
        pub const Core = Core_;
        pub const Backend = B;
        pub const Ref = @import("learning_ref.zig").Ref(B);
        a: std.mem.Allocator,
        rows: ?Rows = null,
        gemma_: ?Gemma = null,
        qsa_: ?Qsa = null,
        pub fn init(a: std.mem.Allocator) Self {
            return .{ .a = a };
        }
        pub fn length(b: B.Buffer) usize {
            return b.length();
        }
        pub fn ensure(k: *Self, b: *B) !void {
            if (k.rows == null) k.rows = try Rows.load(k.a, b.device);
            if (k.gemma_ == null) k.gemma_ = try Gemma.load(k.a, b.device);
            if (k.qsa_ == null) k.qsa_ = try Qsa.load(k.a, b.device);
        }
        pub fn deinitDrained(k: *Self) void {
            if (k.qsa_) |*v| v.deinitDrained();
            if (k.gemma_) |*v| v.deinitDrained();
            if (k.rows) |*v| v.deinitDrained();
        }
        fn views(comptime T: type, refs: [11]Ref) [11]T {
            var out: [11]T = undefined;
            for (&out, refs) |*v, r| v.* = .{ .buffer = r.buffer, .offset = r.offset, .len = r.len };
            return out;
        }
        pub fn row(k: *Self, b: *B, pass: B.Pass, op: Core.living_rows.Op, p: Core.living_rows.Params, refs: [11]Ref) !void {
            try k.ensure(b);
            try k.rows.?.encode(if (b.commands.?.active != null) try b.commands.?.encoder() else pass, op, p, views(Rows.View, refs));
        }
        pub fn gemma(k: *Self, b: *B, pass: B.Pass, op: Core.living_gemma.Op, p: Core.living_rows.Params, refs: [11]Ref) !void {
            try k.ensure(b);
            try k.gemma_.?.encode(if (b.commands.?.active != null) try b.commands.?.encoder() else pass, op, p, views(Gemma.View, refs));
        }
        pub fn qsa(k: *Self, b: *B, pass: B.Pass, op: Core.living_qsa.Op, p: Core.living_rows.Params, refs: [11]Ref) !void {
            try k.ensure(b);
            try k.qsa_.?.encode(if (b.commands.?.active != null) try b.commands.?.encoder() else pass, op, p, views(Qsa.View, refs));
        }
    };
}
