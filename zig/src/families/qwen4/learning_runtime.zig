//! A native pocket owner borrows the engine; empty or untaught row ranges encode no training or pocket operation.
const std = @import("std");
const rows = @import("core").living_rows;
const hook = @import("learning_hook.zig");

pub fn Runtime(comptime E: type, comptime K: type) type {
    const B = E.Storage;
    const Ref = @import("train_tensor.zig").Ref(B);
    const Pockets = @import("pocket.zig").Pockets(B);
    return struct {
        const Self = @This();
        pub const Retained = struct { next: ?*@This() = null, ctx: *anyopaque, drop: *const fn (*anyopaque) void };
        a: std.mem.Allocator,
        e: *E,
        kernels: K,
        pockets: Pockets,
        selected: hook.Selection = .{},
        act: ?Ref = null,
        retained: ?*Retained = null,
        training_reserved: u64 = 0,
        pub fn init(a: std.mem.Allocator, e: *E) !*Self {
            const r = try a.create(Self);
            errdefer a.destroy(r);
            r.* = .{ .a = a, .e = e, .kernels = K.init(a), .pockets = try Pockets.init(a, e.backend, e.c.hidden, e.c.layers) };
            errdefer r.pockets.deinitDrained();
            const act_bytes = try std.math.mul(usize, e.plan.rows, 64 * 4);
            if (act_bytes > e.o.budget -| e.plan.device) return error.Qwen4LearningBudget;
            r.pockets.budget = e.o.budget - e.plan.device - act_bytes;
            try r.pockets.load(&e.reader, &e.c);
            return r;
        }
        pub fn close(r: *Self) !void {
            if (r.e.learning) |listener_| if (listener_.ctx != @as(*anyopaque, @ptrCast(r))) return error.Qwen4LearningActive;
            try r.e.backend.drain();
            r.e.learning = null;
            while (r.retained) |owned| {
                r.retained = owned.next;
                owned.drop(owned.ctx);
            }
            r.pockets.deinitDrained();
            if (r.act) |v| r.e.backend.release(v.buffer);
            r.kernels.deinitDrained();
            r.a.destroy(r);
        }
        pub fn retain(r: *Self, owned: *Retained) void {
            owned.next = r.retained;
            r.retained = owned;
            r.e.fail();
        }
        pub fn reserveTraining(r: *Self, bytes: u64) !void {
            if (r.training_reserved != 0 or bytes > r.pockets.budget -| r.pockets.bytes) return error.Qwen4LearningBudget;
            r.pockets.budget -= bytes;
            r.training_reserved = bytes;
        }
        pub fn releaseTraining(r: *Self) void {
            r.pockets.budget += r.training_reserved;
            r.training_reserved = 0;
        }
        pub fn listener(r: *Self) hook.Hook(E.Gpu) {
            return .{ .ctx = r, .begin = begin, .dense = dense };
        }
        fn begin(ctx: *anyopaque, runs: []const hook.Run, count: usize) !void {
            const r: *Self = @ptrCast(@alignCast(ctx));
            r.selected = try hook.Selection.of(runs, count);
        }
        fn dense(ctx: *anyopaque, _: *E.Gpu, layer: usize, input: E.Tensor, output: E.Tensor) !void {
            const r: *Self = @ptrCast(@alignCast(ctx));
            try r.apply(layer, input, output, 1, std.math.maxInt(u32));
        }
        pub fn prepare(r: *Self, runs: []const hook.Run, count: usize) !void {
            try begin(r, runs, count);
        }
        pub fn apply(r: *Self, layer: usize, input: E.Tensor, output: E.Tensor, scale: f32, from: u32) !void {
            if (layer >= r.pockets.layers.len or input.width != r.e.c.hidden or output.width != input.width or
                input.rows != output.rows or input.kind != .bf16 or output.kind != .bf16) return error.Qwen4PocketShape;
            const pocket = r.pockets.layers[layer] orelse return;
            if (pocket.used == 0 or r.selected.count == 0) return;
            if (r.act == null) {
                const len = try std.math.mul(usize, r.e.plan.rows, 64 * 4);
                r.act = .{ .buffer = try r.e.backend.allocate(len), .len = len };
            }
            const stored: Ref = .{ .buffer = pocket.buffer, .len = 3 * pocket.cap * input.width * 4 };
            for (r.selected.ranges[0..r.selected.count]) |range| {
                const xs = try input.subRows(range.first, range.rows);
                const ys = try output.subRows(range.first, range.rows);
                var p: rows.Params = .{};
                p.u[0] = @intCast(range.rows);
                p.u[1] = @intCast(input.width);
                p.u[2] = @intCast(pocket.used);
                p.u[3] = from;
                p.u[4] = @intCast(pocket.cap);
                p.f[0] = scale;
                const act = try r.act.?.plus(0, range.rows * pocket.used * 4);
                var refs: [11]Ref = @splat(stored);
                refs[0] = .{ .buffer = xs.buffer, .offset = xs.offset, .len = try xs.bytes() };
                refs[1] = stored;
                refs[8] = act;
                try r.kernels.row(&r.e.gpu, .pocket_act, try rows.finish(.pocket_act, p), refs);
                refs[0] = act;
                refs[2] = .{ .buffer = ys.buffer, .offset = ys.offset, .len = try ys.bytes() };
                refs[8] = refs[2];
                try r.kernels.row(&r.e.gpu, .pocket_add, try rows.finish(.pocket_add, p), refs);
            }
        }
    };
}
