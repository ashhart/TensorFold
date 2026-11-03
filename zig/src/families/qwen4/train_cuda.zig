//! The native CUDA learner uses the engine's own context and stream, loading training images only on demand.
const std = @import("std");
const core = @import("core");
pub fn Kernels(comptime C: type) type {
    const B = @import("cuda_storage.zig").Backend(C);
    const Rows = @import("train_rows_cuda.zig").Adapter(C, core.living_rows);
    const Gdn = @import("train_rows_cuda.zig").Adapter(C, core.living_gdn);
    const Qsa = @import("train_rows_cuda.zig").Adapter(C, core.living_qsa);
    return struct {
        const Self = @This();
        pub const Storage = B;
        pub const Ref = @import("train_tensor.zig").Ref(B);
        a: std.mem.Allocator,
        rows_k: ?Rows = null,
        gdn_k: ?Gdn = null,
        qsa_k: ?Qsa = null,
        pub fn init(a: std.mem.Allocator) Self {
            return .{ .a = a };
        }
        pub fn length(b: B.Buffer) usize {
            return b.len;
        }
        pub fn probabilityCut(_: *Self, _: anytype) bool {
            return true;
        }
        pub fn deinitDrained(k: *Self) void {
            if (k.qsa_k) |*v| v.deinitDrained();
            if (k.gdn_k) |*v| v.deinitDrained();
            if (k.rows_k) |*v| v.deinitDrained();
        }
        fn views(comptime T: type, refs: [11]Ref) [11]T {
            var out: [11]T = undefined;
            for (&out, refs) |*v, r| v.* = .{ .buffer = r.buffer, .offset = r.offset, .len = r.len };
            return out;
        }
        pub fn row(k: *Self, gpu: anytype, op: core.living_rows.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
            const b = gpu.backend;
            try b.enter();
            if (k.rows_k == null) k.rows_k = try Rows.load(b.driver, C.kernels.living_rows);
            try k.rows_k.?.encode(b.stream, op, p, views(Rows.View, refs));
        }
        pub fn gdn(k: *Self, gpu: anytype, op: core.living_gdn.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
            const b = gpu.backend;
            try b.enter();
            if (k.gdn_k == null) k.gdn_k = try Gdn.load(b.driver, C.kernels.living_gdn);
            try k.gdn_k.?.encode(b.stream, op, p, views(Gdn.View, refs));
        }
        pub fn qsa(k: *Self, gpu: anytype, op: core.living_qsa.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
            const b = gpu.backend;
            try b.enter();
            if (k.qsa_k == null) k.qsa_k = try Qsa.load(b.driver, C.kernels.living_qsa);
            try k.qsa_k.?.encode(b.stream, op, p, views(Qsa.View, refs));
        }
        pub fn read(_: *Self, b: *B, r: Ref, out: []u8) !void {
            if (out.len > r.len) return error.Qwen4TrainingRange;
            try b.drain();
            try r.buffer.download(r.offset, out);
        }
        pub fn frequency(_: *Self, gpu: anytype, bytes: usize) !Ref {
            return .{ .buffer = gpu.rope orelse return error.Qwen4TrainingRope, .len = @max(4, bytes) };
        }
    };
}
