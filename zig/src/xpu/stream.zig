//! An in-order immediate command list: copies and launches run in the order queued; synchronize waits for all of them.

const ze = @import("ze");
const Context = @import("context.zig").Context;
const Error = @import("context.zig").Error;

pub const Stream = struct {
    ctx: *const Context,
    list: ze.H,

    pub fn init(ctx: *const Context) Error!Stream {
        const d = ctx.d;
        var list: ze.H = null;
        try d.check(d.api.zeCommandListCreateImmediate(ctx.gpu.context, ctx.gpu.device, &.{ .flags = ze.queue_in_order, .mode = 2 }, &list), "zeCommandListCreateImmediate");
        return .{ .ctx = ctx, .list = list };
    }

    /// Waits for queued work first, so nothing the list still reads is freed under it.
    pub fn deinit(self: *Stream) void {
        self.synchronize() catch {};
        _ = self.ctx.d.x.zeCommandListDestroy(self.list);
        self.* = undefined;
    }

    pub fn synchronize(self: Stream) Error!void {
        const d = self.ctx.d;
        try d.check(d.api.zeCommandListHostSynchronize(self.list, @import("std").math.maxInt(u64)), "zeCommandListHostSynchronize");
    }

    /// Queues a copy; host memory must stay valid until the next synchronize.
    pub fn copy(self: Stream, dst: ?*anyopaque, src: ?*const anyopaque, n: usize) Error!void {
        if (n == 0) return;
        const d = self.ctx.d;
        try d.check(d.api.zeCommandListAppendMemoryCopy(self.list, dst, src, n, null, 0, null), "zeCommandListAppendMemoryCopy");
    }

    /// Queues a fill of `n` bytes with one byte value.
    pub fn fill8(self: Stream, dst: ?*anyopaque, value: u8, n: usize) Error!void {
        if (n == 0) return;
        const d = self.ctx.d;
        try d.check(d.x.zeCommandListAppendMemoryFill(self.list, dst, &value, 1, n, null, 0, null), "zeCommandListAppendMemoryFill");
    }
};
