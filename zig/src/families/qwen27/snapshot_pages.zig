//! Snapshot-only immutable KV pages; independent of the attention kernels and their page tables.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
pub const kept = core.kept_pages;
pub const Layout = kept.Layout;
pub const Image = kept.Image;
pub const page_tokens = core.metal_kv_pages.page_tokens;

pub const Store = struct {
    gpa: std.mem.Allocator,
    pages: kept.Store,
    buffers: []?mtl.Buffer,
    follow: ?mtl.ResidencySet.Follow,

    pub fn init(a: std.mem.Allocator, budget: u64, layout: Layout, follow: ?mtl.ResidencySet.Follow) !*Store {
        const count: usize = @intCast(budget / layout.pageBytes());
        const s = try a.create(Store);
        errdefer a.destroy(s);
        var pages = try kept.Store.init(a, count, layout.pageBytes(), page_tokens);
        errdefer pages.deinit();
        const buffers = try a.alloc(?mtl.Buffer, count);
        @memset(buffers, null);
        s.* = .{ .gpa = a, .pages = pages, .buffers = buffers, .follow = follow };
        return s;
    }

    pub fn keep(s: *Store, device: mtl.Device, old: ?Image, at: u32) !Image {
        const image = try s.pages.keep(old, at);
        errdefer s.drop(image);
        for (image.ids[image.fresh..]) |id| {
            std.debug.assert(s.buffers[id] == null);
            const buf = try device.buffer(@intCast(s.pages.pool.page_bytes), mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
            s.buffers[id] = buf;
            if (s.follow) |f| f.join(f.ctx, buf);
        }
        return image;
    }

    pub fn drop(s: *Store, image: Image) void {
        for (image.ids) |id| if (s.pages.pool.holders(id) == 1) {
            if (s.buffers[id]) |buf| {
                if (s.follow) |f| f.leave(f.ctx, buf);
                buf.deinit();
                s.buffers[id] = null;
            }
        };
        s.pages.drop(image);
    }

    pub fn deinit(s: *Store) void {
        s.pages.deinit();
        for (s.buffers) |buf| std.debug.assert(buf == null);
        s.gpa.free(s.buffers);
        s.gpa.destroy(s);
    }
};
