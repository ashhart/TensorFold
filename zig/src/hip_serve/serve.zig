//! What every HIP family's lane backend shares beside the kernels: the draw, pages, prompt cache, callbacks, tp steps.
pub const draw = @import("draw.zig");
pub const sample = @import("sample.zig");
pub const pages = @import("pages.zig");
pub const snapshots = @import("snapshots.zig");
pub const protocol = @import("protocol.zig");
pub const prefix = @import("prefix.zig");
pub const backend = @import("backend.zig");
pub const costs = @import("costs.zig");

test {
    _ = draw;
    _ = sample;
    _ = pages;
    _ = snapshots;
    _ = protocol;
    _ = prefix;
    _ = backend;
    _ = costs;
}
