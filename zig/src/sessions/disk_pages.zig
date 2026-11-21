const pool_mod = @import("pool.zig");
const Family = pool_mod.Family;
const PageStore = @import("pagestore.zig").PageStore;

/// Where a restore puts rows: per family, the family-tensor page of each of the slot's logical pages.
pub const ReadSink = struct {
    store: PageStore,
    families: []const Family,
    page: u32,
    world: u32,
    rank: u32,
    /// per family: family-tensor pages for logical pages [0, npages)
    pages: []const []const u32,
};

/// Owned (this rank's) logical pages of family f in [first, k): the file's index of logical page k.
pub fn ownedIn(f: Family, sink: *const ReadSink, first: u32, k: u32) u32 {
    if (!(f.split and sink.world > 1)) return k - first;
    var by: [pool_mod.max_world]u32 = undefined;
    pool_mod.span(first, k, sink.world, &by);
    return by[sink.rank];
}

/// The logical page of a family's j-th stored page of a file that starts at first.
pub fn logicalOf(f: Family, sink: *const ReadSink, first: u32, j: u32) u32 {
    if (!(f.split and sink.world > 1)) return first + j;
    const w = sink.world;
    const r0 = first + (sink.rank + w - first % w) % w; // the first owned logical page at or after first
    return r0 + j * w;
}
