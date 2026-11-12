//! Preallocated snapshot pages inside the prompt-cache budget; a keep or restore makes no device allocation.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const walk = @import("core").state_walk;
const Plan = @import("cuda_snapshot_layout.zig").Plan;

pub const page = walk.page;
pub const Lease = kv.Lease;
pub const rounded = walk.rounded;

/// TF_GLM_HOST_CACHE=1: kept states in pinned host memory, not on the device; every rank must be given the same value.
pub fn onHost() !bool {
    return @import("sys").environment.value(bool, "TF_GLM_HOST_CACHE", @import("profile").lookup("TF_GLM_HOST_CACHE"), false);
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
        return walk.lease(&p.pages, bytes);
    }

    pub fn give(p: *Pool, lease: *Lease) void {
        p.pages.finish(lease);
    }

    /// The plan's parts to (`into`) or from the lease's pages, queued on `stream`: the core walk, a copy a piece.
    pub fn move(p: *const Pool, stream: cuda.Stream, lease: *const Lease, plan: *const Plan, into: bool) !void {
        if (plan.len > lease.ids.items.len * page) return error.SnapshotSize;
        const arena = p.arena orelse return error.NoSnapshotRoom;
        const kept: walk.Kept = .{ .ids = lease.ids.items, .base = arena.ptr() };
        const c: Copy = .{ .d = arena.driver(), .s = stream, .into = into };
        try walk.Device.each(plan.parts[0..plan.count], plan.ids, kept, c, Copy.run);
    }
};

const Copy = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    into: bool,
    fn run(c: Copy, live: u64, kept: u64, n: usize) anyerror!void {
        const dst, const src = if (c.into) .{ kept, live } else .{ live, kept };
        try c.d.check(c.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, c.s.handle), "cuMemcpyDtoDAsync");
    }
};
