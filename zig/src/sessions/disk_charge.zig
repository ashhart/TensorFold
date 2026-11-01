//! Split-rank eviction uses logical bytes; physical file sizes remain available for disk headroom reporting.

const disk = @import("disk.zig");

pub fn charge(src: *const disk.WriteSrc, world: u32, physical: u64) u64 {
    if (world <= 1) return physical;
    var bytes: u64 = @as(u64, src.ids.len) * 4 + src.measure;
    for (src.families) |f| {
        const pages: u64 = src.npages - src.first_page;
        bytes += pages * f.pageBytes(src.page) / (if (f.split) world else 1);
    }
    return bytes;
}
