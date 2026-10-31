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

/// TF_GLM_HOST_CACHE=1: kept states in pinned host memory, not on the device; every rank must be given the same value.
pub fn onHost() bool {
    const v = std.c.getenv("TF_GLM_HOST_CACHE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The pages' memory: a device buffer, or mapped pinned host memory that the same device-to-device copies reach.
const Arena = union(enum) {
    device: cuda.DeviceBuffer,
    host: struct { buf: cuda.HostBuffer, ptr: u64 },

    fn ptr(a: Arena) u64 {
        return switch (a) {
            .device => |b| b.ptr,
            .host => |h| h.ptr,
        };
    }

    pub fn len(a: Arena) usize {
        return switch (a) {
            .device => |b| b.len,
            .host => |h| h.buf.bytes.len,
        };
    }

    fn driver(a: Arena) *const cuda.Driver {
        return switch (a) {
            .device => |b| b.d,
            .host => |h| h.buf.d,
        };
    }
};

pub const Pool = struct {
    pages: kv.Pool,
    arena: ?Arena,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, budget: u64, host: bool) !Pool {
        const count = std.math.cast(usize, budget / page) orelse return error.SnapshotBudget;
        if (count > std.math.maxInt(u32)) return error.SnapshotBudget;
        var pages = try kv.Pool.init(gpa, count, page);
        errdefer pages.deinit();
        const arena: ?Arena = if (count == 0) null else if (host) blk: {
            var buf = try cuda.HostBuffer.allocMapped(d, count * page);
            errdefer buf.free();
            break :blk .{ .host = .{ .buf = buf, .ptr = try buf.device() } };
        } else .{ .device = try cuda.DeviceBuffer.alloc(d, count * page) };
        return .{ .pages = pages, .arena = arena };
    }

    pub fn deinit(p: *Pool) void {
        std.debug.assert(p.pages.available() == p.pages.total());
        if (p.arena) |*a| switch (a.*) {
            .device => |*b| b.free(),
            .host => |*h| h.buf.free(),
        };
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
        const base = arena.ptr();
        const d = arena.driver();
        var off: usize = 0;
        for (plan.parts[0..plan.count]) |part| {
            var done: usize = 0;
            while (done < part.len) {
                const at = off + done;
                const target = base + @as(u64, lease.ids.items[at / page]) * page + at % page;
                const live = plan.piece(part, done);
                const n = @min(live.len, page - at % page);
                try d.check(d.api.cuMemcpyDtoDAsync_v2(if (into) target else live.live, if (into) live.live else target, n, stream.handle), "cuMemcpyDtoDAsync");
                done += n;
            }
            off += part.len;
        }
    }
};
