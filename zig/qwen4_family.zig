//! Dense Qwen4's common family module; device adapters plug into the existing native interfaces.
pub const family = @import("src/families/qwen4/qwen4.zig");
pub const config = family.config;
pub const Config = family.Config;
pub const Problem = family.Problem;
pub const graph = family.graph;
pub const weights = family.weights;
pub const reader = family.reader;
test {
    _ = family;
}
