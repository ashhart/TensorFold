//! Session memory, family-neutral: the KV pool book, page bytes, prefix entries, the RAM tier and the NVMe tier.

pub const freeset = @import("freeset.zig");
pub const pool = @import("pool.zig");
pub const Pool = pool.Pool;
pub const Slot = pool.Slot;
pub const Family = pool.Family;
pub const Layout = pool.Layout;
pub const pagestore = @import("pagestore.zig");
pub const PageStore = pagestore.PageStore;
pub const HostPool = pagestore.HostPool;
pub const prefix = @import("prefix.zig");
pub const lanes = @import("lanes.zig");
pub const format = @import("format.zig");
pub const disk = @import("disk.zig");
pub const Disk = disk.Disk;
pub const jobs = @import("jobs.zig");
pub const store = @import("store.zig");
pub const Store = store.Store;

test {
    _ = freeset;
    _ = pool;
    _ = pagestore;
    _ = prefix;
    _ = lanes;
    _ = format;
    _ = disk;
    _ = jobs;
    _ = store;
    _ = @import("disk_test.zig");
    _ = @import("store_test.zig");
}
