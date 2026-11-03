//! The native Metal learner encodes into the inference command session and retains all referenced pipelines.
const std = @import("std");
const core = @import("core");
pub fn Kernels(comptime M: type) type {
    const B = @import("metal_storage.zig").Backend(M);
    const Rows = @import("train_rows_metal.zig").Adapter(M, core.living_rows);
    const Gdn = @import("train_rows_metal.zig").Adapter(M, core.living_gdn);
    const Qsa = @import("train_rows_metal.zig").Adapter(M, core.living_qsa);
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
            return b.length();
        }
        pub fn probabilityCut(_: *Self, _: anytype) bool {
            return false;
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
            if (k.rows_k == null) k.rows_k = try Rows.load(k.a, b.device);
            try k.rows_k.?.encode(try b.commands.?.encoder(), op, p, views(Rows.View, refs));
        }
        pub fn gdn(k: *Self, gpu: anytype, op: core.living_gdn.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
            const b = gpu.backend;
            if (k.gdn_k == null) k.gdn_k = try Gdn.load(k.a, b.device);
            try k.gdn_k.?.encode(try b.commands.?.encoder(), op, p, views(Gdn.View, refs));
        }
        pub fn qsa(k: *Self, gpu: anytype, op: core.living_qsa.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
            const b = gpu.backend;
            if (k.qsa_k == null) k.qsa_k = try Qsa.load(k.a, b.device);
            try k.qsa_k.?.encode(try b.commands.?.encoder(), op, p, views(Qsa.View, refs));
        }
        pub fn read(_: *Self, b: *B, r: Ref, out: []u8) !void {
            if (r.offset > r.buffer.length() or out.len > r.len or out.len > r.buffer.length() - r.offset) return error.Qwen4TrainingRange;
            try b.drain();
            @memcpy(out, r.buffer.contents()[r.offset..][0..out.len]);
        }
        pub fn frequency(_: *Self, gpu: anytype, bytes: usize) !Ref {
            return .{ .buffer = gpu.rope orelse return error.Qwen4TrainingRope, .offset = if (bytes == 0) 0 else 12, .len = @max(4, bytes) };
        }
    };
}
