//! Qwen3.8 Flash Next for the native HIP engine: the family's config and index checks, the plan of a rank's bytes.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const index = @import("index.zig");
pub const Spec = @import("spec.zig").Spec;
pub const plan = @import("hip/plan.zig");
pub const ngram = @import("hip/ngram.zig");
/// The registry entry of the native server.
pub const native = @import("hip/native.zig");

test {
    _ = config;
    _ = index;
    _ = plan;
    _ = ngram;
    _ = native;
}
