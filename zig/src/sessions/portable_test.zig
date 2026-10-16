//! Portable session bookkeeping tests; the Linux NVMe tier is tested separately.
test {
    _ = @import("freeset.zig");
    _ = @import("pool.zig");
    _ = @import("pagestore.zig");
    _ = @import("prefix.zig");
    _ = @import("format.zig");
}
