// CUDA storage adapter for the shared logical transaction policy.
const std = @import("std");
const cuda = @import("cuda");
const logical = @import("kv_state.zig");
const Forward = @import("forward.zig").Forward;
pub const Transaction = struct {
    state: logical.State = .{},
    regions: std.ArrayList(logical.Region) = .empty,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    runner: ?*@import("run.zig").Runner = null,
    // Refreshed for every selected slot. Backups remain owned until Forward.close drains the stream.
    pub fn begin(t: *Transaction, f: *Forward, n: u32) !void {
        if (t.state.pending != null) return error.CacheTransactionActive;
        t.runner = f.runner;
        t.regions.clearRetainingCapacity();
        var nb: [64]u8 = undefined;
        var la = std.heap.ArenaAllocator.init(f.gpa);
        defer la.deinit();
        // Carries only: rows past the kept ones sit in SWA ring slots (pos % 2 window) rewritten before any read.
        if (n > f.cfg.window) return error.BadWindow;
        for (try f.backbone(la.allocator())) |L| {
            const role = try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L});
            const ptr = f.runner.addressOf(role) orelse continue;
            const row = 8 * @as(usize, f.cfg.head_dim);
            const i = t.regions.items.len;
            if (i == t.owned.items.len) {
                try t.owned.ensureUnusedCapacity(f.gpa, 1);
                t.owned.appendAssumeCapacity(try cuda.DeviceBuffer.alloc(f.runner.d, row));
            }
            if (t.owned.items[i].len < row) return error.TransactionShapeChanged;
            try t.regions.append(f.gpa, .{ .live = ptr, .backup = t.owned.items[i].ptr, .row_bytes = row, .rows = 1, .kind = .carry, .encoding = .f32 });
        }
        t.state.position = f.slot.pos;
        try t.state.begin(t.regions.items, t.copier(), n);
    }
    fn copy(ctx: *anyopaque, dst: u64, src: u64, bytes: usize) !void {
        const t: *Transaction = @ptrCast(@alignCast(ctx));
        const r = t.runner.?;
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, r.stream.handle), "cuMemcpyDtoDAsync");
    }
    fn copier(t: *Transaction) logical.Copy {
        return .{ .ctx = t, .run = copy };
    }
    pub fn keep(t: *Transaction, n: u32) !void {
        try t.state.keep(t.regions.items, t.copier(), n);
    }
    pub fn rollback(t: *Transaction) !void {
        try t.state.rollback(t.regions.items, t.copier());
    }
    pub fn deinit(t: *Transaction, a: std.mem.Allocator) void {
        for (t.owned.items) |*b| b.free();
        t.owned.deinit(a);
        t.regions.deinit(a);
        t.* = .{};
    }
};
