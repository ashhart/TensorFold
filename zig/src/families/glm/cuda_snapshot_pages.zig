//! Preallocated snapshot pages inside the prompt-cache budget; a keep or restore makes no device allocation.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const Plan = @import("cuda_snapshot_layout.zig").Plan;

pub const page: usize = 2 << 20;
pub const Lease = kv.Lease;

pub fn rounded(bytes: usize) !usize {
    return (try std.math.add(usize, bytes, page - 1)) & ~(page - 1);
}

pub const Pool = struct {
    pages: kv.Pool,
    arena: ?cuda.DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, budget: u64) !Pool {
        const count = std.math.cast(usize, budget / page) orelse return error.SnapshotBudget;
        if (count > std.math.maxInt(u32)) return error.SnapshotBudget;
        var pages = try kv.Pool.init(gpa, count, page);
        errdefer pages.deinit();
        const arena = if (count > 0) try cuda.DeviceBuffer.alloc(d, count * page) else null;
        return .{ .pages = pages, .arena = arena };
    }

    pub fn deinit(p: *Pool) void {
        std.debug.assert(p.pages.available() == p.pages.total());
        if (p.arena) |*a| a.free();
        p.pages.deinit();
    }

    pub fn spare(p: *const Pool) u64 {
        return p.pages.available() * page;
    }

    pub fn take(p: *Pool, bytes: usize) !Lease {
        const n = (try rounded(bytes)) / page;
        if (n == 0 or n > p.pages.available()) return error.NoSnapshotRoom;
        var lease: Lease = .{};
        errdefer p.pages.finish(&lease);
        try p.pages.promise(&lease, n);
        try p.pages.cover(&lease, n, null);
        return lease;
    }

    pub fn give(p: *Pool, lease: *Lease) void {
        p.pages.finish(lease);
    }

    pub fn move(p: *const Pool, stream: cuda.Stream, lease: *const Lease, plan: *const Plan, into: bool) !void {
        if (plan.len > lease.ids.items.len * page) return error.SnapshotSize;
        const arena = p.arena orelse return error.NoSnapshotRoom;
        var off: usize = 0;
        for (plan.parts[0..plan.count]) |part| {
            var done: usize = 0;
            while (done < part.len) {
                const at = off + done;
                const target = arena.ptr + @as(u64, lease.ids.items[at / page]) * page + at % page;
                const n = @min(part.len - done, page - at % page);
                const live = part.live + done;
                try arena.d.check(arena.d.api.cuMemcpyDtoDAsync_v2(if (into) target else live, if (into) live else target, n, stream.handle), "cuMemcpyDtoDAsync");
                done += n;
            }
            off += part.len;
        }
    }
};
