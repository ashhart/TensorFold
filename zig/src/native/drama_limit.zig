//! A device plus pinned cap held for the opened speech engine's entire lifetime.
const std = @import("std");
const cuda = @import("cuda");
const budget = @import("cuda_memory.zig");
pub const Limit = struct {
    driver: cuda.Driver,
    ctx: cuda.Context,
    previous: ?u64,
    cap: u64,
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
        l.cap = try std.math.add(u64, held, plan.room(held));
        l.previous = try cuda.setAllocationLimit(l.cap);
    }
    /// Before the remaining lanes fork: each needs a lane's state and, with CFG, a guide's, under the cap.
    pub fn admit(l: *const Limit, more: usize, lane_bytes: u64, guide_bytes: u64) !void {
        const used = cuda.usage(false);
        if (!fits(l.cap -| (used.device +| used.host), more, lane_bytes, guide_bytes)) return error.DramaLanesOverMemory;
    }
    pub fn deinit(l: *Limit) void {
        _ = cuda.setAllocationLimit(l.previous) catch unreachable;
        l.ctx.deinit();
        l.driver.close();
    }
};

fn fits(room: u64, more: usize, lane_bytes: u64, guide_bytes: u64) bool {
    const need = std.math.mul(u64, more, std.math.add(u64, lane_bytes, guide_bytes) catch return false) catch return false;
    return need <= room;
}

test "drama admission: each further lane counts its guidance state with its own" {
    try std.testing.expect(fits(30, 3, 10, 0));
    try std.testing.expect(!fits(30, 3, 10, 1)); // the plain plan fits, the guided one does not
    try std.testing.expect(fits(33, 3, 10, 1));
    try std.testing.expect(fits(0, 0, 10, 1));
    try std.testing.expect(!fits(std.math.maxInt(u64), 2, std.math.maxInt(u64), 1));
}
