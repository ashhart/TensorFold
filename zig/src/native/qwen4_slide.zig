//! The native lane host owns the pocket, learner adapter and trainer until all queued device users complete.
const std = @import("std");
const F = @import("qwen4").family;
const Api = @import("engine_api");
pub fn Owner(comptime E: type, comptime K: type) type {
    const R = F.learning_runtime.Runtime(E, K);
    const T = F.train.Trainer(E, K);
    const Slide = F.slide.Slide(E, T);
    const Adapter = @import("slide_adapter.zig").Adapter(Slide);
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        runtime: *R,
        adapter: ?Adapter = null,
        dir: []u8,
        pub fn bytes(a: std.mem.Allocator, c: *const F.Config, capacity: usize, enabled: bool) !u64 {
            const charge = try std.math.add(u64, try E.Storage.allocationBytes(3 * 64 * c.hidden * 4), 3 * 64 * c.hidden * 4);
            const pocket = try std.math.mul(u64, charge, c.layers + @as(usize, @intFromBool(enabled)));
            return std.math.add(u64, pocket + 128 * 64 * 4, if (enabled) try F.train_budget.bytes(E.Storage, a, c, capacity) else 0);
        }
        pub fn open(a: std.mem.Allocator, io: std.Io, e: *E, dir: []const u8, enabled: bool) !*Self {
            const owned = try a.dupe(u8, dir);
            errdefer a.free(owned);
            const self = try a.create(Self);
            errdefer a.destroy(self);
            const runtime = try R.init(a, e);
            self.* = .{ .a = a, .runtime = runtime, .dir = owned };
            if (enabled) self.adapter = Adapter.init(a, io, Slide.Model{ .e = e, .pockets = &runtime.pockets, .dir = owned, .trainer = runtime });
            e.learning = runtime.listener();
            return self;
        }
        pub fn hook(self: *Self) ?Api.Learner {
            return if (self.adapter) |*ad| ad.hook() else null;
        }
        pub fn close(self: *Self) !void {
            if (self.adapter) |*ad| {
                ad.deinit();
                self.adapter = null;
            }
            try self.runtime.close();
            self.a.free(self.dir);
            self.a.destroy(self);
        }
    };
}
