//! Source assembly shares one module owner with the engine's family and core files.
pub const metal_runtime_sources = @import("runtime_sources/root.zig");
const engine = @import("src/tensorfold.zig");
pub const gpu_profile = engine.gpu_profile;
pub const sys = @import("sys");
pub const rope_freq = engine.rope_freq;
pub const weight_only = engine.weight_only;
