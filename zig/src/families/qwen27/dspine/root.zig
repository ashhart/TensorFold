//! DSpine (Tencent's code-bridge block drafter) for the 27B on Metal: a second drafter kind beside DFlash2.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const host = @import("host.zig");
pub const kernels = @import("kernels.zig");
pub const runtime = @import("runtime.zig");
pub const forward = @import("forward.zig");
test {
    _ = config;
    _ = weights;
    _ = host;
    _ = kernels;
    _ = runtime;
    _ = forward;
    _ = @import("kernels_test.zig");
}
