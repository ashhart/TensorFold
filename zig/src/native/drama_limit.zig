//! A device plus pinned cap held for the opened speech engine's entire lifetime.
const std = @import("std");
const cuda = @import("cuda");
const budget = @import("cuda_memory.zig");
pub const Limit = struct {
    driver: cuda.Driver,
    ctx: cuda.Context,
    previous: ?u64,
    pub fn init(l: *Limit, gpa: std.mem.Allocator, io: std.Io, device: u32) !void {
        l.driver = try cuda.Driver.open();
        errdefer l.driver.close();
        l.ctx = try cuda.Context.init(&l.driver, std.math.cast(c_int, device) orelse return error.NoSuchCudaDevice);
        errdefer l.ctx.deinit();
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var problem: []const u8 = "";
        const plan = try budget.readPool(arena.allocator(), io, &l.ctx, &problem) orelse return error.DramaMemoryAdmission;
        const used = cuda.usage(false);
        const held = try std.math.add(u64, used.device, used.host);
        l.previous = try cuda.setAllocationLimit(try std.math.add(u64, held, plan.room(held)));
    }
    pub fn deinit(l: *Limit) void {
        _ = cuda.setAllocationLimit(l.previous) catch unreachable;
        l.ctx.deinit();
        l.driver.close();
    }
};
