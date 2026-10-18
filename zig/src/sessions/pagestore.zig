//! The bytes behind the pool's pages: a family's tensor of pages, read and written a page run at a time.

const std = @import("std");
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;
const Family = pool_mod.Family;

pub const PageStore = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// family fam's pages, back to back into dst (pages.len x a page's bytes)
        read: *const fn (ptr: *anyopaque, fam: u32, pages: []const u32, dst: []u8) anyerror!void,
        /// src (pages.len x a page's bytes) into family fam's pages
        write: *const fn (ptr: *anyopaque, fam: u32, pages: []const u32, src: []const u8) anyerror!void,
        /// one page's rows of family fam into another page (copy on write of a RAM resume's partial page)
        copyPage: *const fn (ptr: *anyopaque, fam: u32, src: u32, dst: u32) anyerror!void,
        /// round thread: a point after every row the compute stream has written so far
        fence: *const fn (ptr: *anyopaque) anyerror!u64,
        /// before reads: wait until the fence's rows are visible
        waitFence: *const fn (ptr: *anyopaque, token: u64) anyerror!void,
        /// after writes: make them visible to the compute stream
        publish: *const fn (ptr: *anyopaque) anyerror!void,
    };

    pub fn read(s: PageStore, fam: u32, pages: []const u32, dst: []u8) !void {
        return s.vtable.read(s.ptr, fam, pages, dst);
    }
    pub fn write(s: PageStore, fam: u32, pages: []const u32, src: []const u8) !void {
        return s.vtable.write(s.ptr, fam, pages, src);
    }
    pub fn copyPage(s: PageStore, fam: u32, src: u32, dst: u32) !void {
        return s.vtable.copyPage(s.ptr, fam, src, dst);
    }
    pub fn fence(s: PageStore) !u64 {
        return s.vtable.fence(s.ptr);
    }
    pub fn waitFence(s: PageStore, token: u64) !void {
        return s.vtable.waitFence(s.ptr, token);
    }
    pub fn publish(s: PageStore) !void {
        return s.vtable.publish(s.ptr);
    }
};

/// Every family's pages in host memory: [familyTensorPages x page bytes] a family, zeroed (the null page stays zero).
pub const HostPool = struct {
    gpa: std.mem.Allocator,
    page: u32,
    families: []const Family,
    tensors: [][]u8,

    pub fn init(gpa: std.mem.Allocator, p: *const Pool) !HostPool {
        const t = try gpa.alloc([]u8, p.layout.families.len);
        errdefer gpa.free(t);
        var made: usize = 0;
        errdefer for (t[0..made]) |x| gpa.free(x);
        for (p.layout.families, 0..) |f, i| {
            t[i] = try gpa.alloc(u8, @as(usize, p.familyTensorPages(f)) * f.pageBytes(p.page));
            @memset(t[i], 0);
            made += 1;
        }
        return .{ .gpa = gpa, .page = p.page, .families = p.layout.families, .tensors = t };
    }

    pub fn deinit(h: *HostPool) void {
        for (h.tensors) |x| h.gpa.free(x);
        h.gpa.free(h.tensors);
        h.* = undefined;
    }

    pub fn store(h: *HostPool) PageStore {
        return .{ .ptr = h, .vtable = &vtable };
    }

    /// One page of family fam (a view).
    pub fn pageOf(h: *HostPool, fam: u32, pg: u32) []u8 {
        const n: usize = @intCast(h.families[fam].pageBytes(h.page));
        return h.tensors[fam][pg * n ..][0..n];
    }

    const vtable: PageStore.VTable = .{ .read = read, .write = write, .copyPage = copyPage, .fence = fence, .waitFence = waitFence, .publish = publish };

    fn read(ptr: *anyopaque, fam: u32, pages: []const u32, dst: []u8) anyerror!void {
        const h: *HostPool = @ptrCast(@alignCast(ptr));
        const n: usize = @intCast(h.families[fam].pageBytes(h.page));
        if (dst.len != pages.len * n) return error.BadLength;
        for (pages, 0..) |pg, i| @memcpy(dst[i * n ..][0..n], h.pageOf(fam, pg));
    }

    fn write(ptr: *anyopaque, fam: u32, pages: []const u32, src: []const u8) anyerror!void {
        const h: *HostPool = @ptrCast(@alignCast(ptr));
        const n: usize = @intCast(h.families[fam].pageBytes(h.page));
        if (src.len != pages.len * n) return error.BadLength;
        for (pages, 0..) |pg, i| @memcpy(h.pageOf(fam, pg), src[i * n ..][0..n]);
    }

    fn copyPage(ptr: *anyopaque, fam: u32, src: u32, dst: u32) anyerror!void {
        const h: *HostPool = @ptrCast(@alignCast(ptr));
        @memcpy(h.pageOf(fam, dst), h.pageOf(fam, src));
    }

    fn fence(_: *anyopaque) anyerror!u64 {
        return 0;
    }
    fn waitFence(_: *anyopaque, _: u64) anyerror!void {}
    fn publish(_: *anyopaque) anyerror!void {}
};

test "host pool pages round-trip through the interface" {
    const gpa = std.testing.allocator;
    const fams = [_]Family{ .{ .name = "comp", .ratio = 1, .row_bytes = 8 }, .{ .name = "ik", .ratio = 2, .row_bytes = 4 } };
    var p = try Pool.init(gpa, .{ .families = &fams, .page = 16 }, 64, 1, 0);
    defer p.deinit();
    var h = try HostPool.init(gpa, &p);
    defer h.deinit();
    const s = h.store();
    var src: [2 * 128]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @truncate(i * 7 + 3);
    try s.write(0, &.{ 3, 1 }, &src);
    var dst: [2 * 128]u8 = undefined;
    try s.read(0, &.{ 3, 1 }, &dst);
    try std.testing.expectEqualSlices(u8, &src, &dst);
    try s.copyPage(0, 3, 0);
    try std.testing.expectEqualSlices(u8, src[0..128], h.pageOf(0, 0));
    try std.testing.expect(std.mem.allEqual(u8, h.pageOf(0, p.null_page), 0));
    try std.testing.expectError(error.BadLength, s.read(1, &.{0}, &dst));
}
