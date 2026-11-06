//! The native owner publishes GeGLU pockets and preserves every trainer allocation until device completion.
const std = @import("std");
const hook = @import("learning_hook.zig");
const profile = @import("learning_profile.zig");
pub fn Runtime(comptime B: type, comptime K: type) type {
    return struct {
        const Self = @This();
        pub const Backend = B;
        pub const Kernels = K;
        pub const Ref = @import("learning_ref.zig").Ref(B);
        pub const Hook = hook.Hook(B.Buffer, B.Pass);
        pub const View = hook.View(B.Buffer);
        pub const Retained = struct { next: ?*@This() = null, ctx: *anyopaque, drop: *const fn (*anyopaque) void };
        pub const Observer = struct { ctx: *anyopaque, begin: *const fn (*anyopaque, []const hook.Run, usize) anyerror!void, note: *const fn (*anyopaque, B.Pass, ?usize, hook.Node, View) anyerror!void, dense: *const fn (*anyopaque, B.Pass, usize, View, View) anyerror!void };
        a: std.mem.Allocator,
        backend: *B,
        kernels: K,
        shape: profile.Shape,
        pockets: @import("pocket.zig").Pockets(B),
        act: ?Ref = null,
        selected: hook.Selection = .{},
        observer: ?Observer = null,
        retained: ?*Retained = null,
        training_reserved: u64 = 0,
        failed: bool = false,
        closing_completed: bool = false,
        pub fn init(a: std.mem.Allocator, b: *B, shape: profile.Shape, budget: u64) !*Self {
            const r = try a.create(Self);
            errdefer a.destroy(r);
            r.* = .{ .a = a, .backend = b, .shape = shape, .kernels = K.init(a), .pockets = try @import("pocket.zig").Pockets(B).init(a, b, shape.hidden, shape.layers) };
            r.pockets.budget = budget;
            return r;
        }
        pub fn listener(r: *Self) Hook {
            return .{ .ctx = r, .active = active, .begin = begin, .dense = dense, .observe = if (r.observer != null) note else null };
        }
        fn active(ctx: *anyopaque) bool {
            const r: *Self = @ptrCast(@alignCast(ctx));
            return r.observer != null or (r.selected.count != 0 and r.pockets.mask() != 0);
        }
        fn begin(ctx: *anyopaque, runs: []const hook.Run, rows: usize) !void {
            const r: *Self = @ptrCast(@alignCast(ctx));
            if (r.failed) return error.GemmaLearningUnhealthy;
            r.selected = try hook.Selection.of(runs, rows);
            if (r.observer) |o| try o.begin(o.ctx, runs, rows);
        }
        fn note(ctx: *anyopaque, pass: B.Pass, l: ?usize, node: hook.Node, v: View) !void {
            const r: *Self = @ptrCast(@alignCast(ctx));
            if (r.observer) |o| try o.note(o.ctx, pass, l, node, v);
        }
        fn dense(ctx: *anyopaque, pass: B.Pass, l: usize, input: View, output: View) !void {
            const r: *Self = @ptrCast(@alignCast(ctx));
            if (r.observer) |o| return o.dense(o.ctx, pass, l, input, output);
            try r.apply(pass, l, input, output, 1, std.math.maxInt(u32));
        }
        pub fn apply(r: *Self, pass: B.Pass, l: usize, input: View, output: View, scale: f32, from: u32) !void {
            if (l >= r.pockets.layers.len or input.width != r.shape.hidden or output.width != input.width or input.rows != output.rows) return error.DensePocketShape;
            const pock = r.pockets.layers[l] orelse return;
            if (pock.used == 0 or r.selected.count == 0) return;
            if (input.rows > 512) return error.DensePocketShape;
            if (r.act == null) {
                const bytes = 512 * 64 * 4;
                if (bytes > r.pockets.budget -| r.pockets.bytes) return error.DensePocketBudget;
                r.act = .{ .buffer = try r.backend.allocate(bytes), .len = bytes };
                r.pockets.budget -= bytes;
            }
            const stored: Ref = .{ .buffer = pock.buffer, .len = pock.values.len * 4 };
            for (r.selected.ranges[0..r.selected.count]) |range| {
                const x: Ref = .{ .buffer = input.buffer, .offset = input.offset + range.first * input.width * 2, .len = range.rows * input.width * 2 };
                const y: Ref = .{ .buffer = output.buffer, .offset = output.offset + range.first * output.width * 2, .len = range.rows * output.width * 2 };
                const act_ = try r.act.?.plus(0, range.rows * pock.used * 4);
                const o: @import("learning_ops.zig").Ops(Self) = .{ .r = r, .pass = pass, .dummy = stored };
                var p: K.Core.living_rows.Params = .{};
                p.u[0] = @intCast(range.rows);
                p.u[1] = @intCast(input.width);
                p.u[2] = @intCast(pock.used);
                p.u[3] = from;
                p.u[4] = 64;
                p.u[12] = 1;
                p.f[0] = scale;
                try o.row(.pocket_act, p, &.{ .{ 0, x }, .{ 1, stored }, .{ 8, act_ } });
                try o.row(.pocket_add, p, &.{ .{ 0, act_ }, .{ 1, stored }, .{ 2, y }, .{ 8, y } });
            }
        }
        pub fn reserve(r: *Self, bytes: u64) !void {
            if (r.training_reserved != 0 or bytes > r.pockets.budget -| r.pockets.bytes) return error.GemmaLearningBudget;
            r.pockets.budget -= bytes;
            r.training_reserved = bytes;
        }
        pub fn unreserve(r: *Self) void {
            r.pockets.budget += r.training_reserved;
            r.training_reserved = 0;
        }
        pub fn retain(r: *Self, item: *Retained) void {
            item.next = r.retained;
            r.retained = item;
            r.failed = true;
        }
        pub fn prepareClose(r: *Self) !void {
            try r.backend.drain();
            r.closing_completed = true;
        }
        pub fn close(r: *Self) !void {
            if (!r.closing_completed) try r.prepareClose();
            while (r.retained) |item| {
                r.retained = item.next;
                item.drop(item.ctx);
            }
            r.pockets.deinitDrained();
            if (r.act) |v| r.backend.release(v.buffer);
            r.kernels.deinitDrained();
            r.a.destroy(r);
        }
    };
}
